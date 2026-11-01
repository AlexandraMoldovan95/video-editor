"""Validated EDL -> ffmpeg commands -> MP4.

Strategy: each clip is rendered separately into a NORMALIZED segment
(same resolution, fps, codec, pix_fmt) with an EXACT frame count.
- no transitions and no text: the segments are joined with the concat demuxer, no re-encoding (fast)
- otherwise: a single final re-encoded pass (xfade for transitions, concat for cuts,
  drawtext for text). The transition starts EXACTLY on the boundary between clips: the previous
  clip is extended by the transition's duration -> the total duration and beat alignment stay unchanged.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from .edl import EDL, Clip, OverlayItem, TextItem, clip_duration, overlay_duration, total_duration, validate_edl
from .probe import IMAGE_EXT, MediaInfo, probe

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
ZOOM_AMOUNT = 0.12  # how much photos zoom (12%)


def fit_filter(fit: str, w: int, h: int, inp: str, out: str, pan_x: float = 0.0, pan_y: float = 0.0) -> str:
    """Brings any frame to exactly w x h.

    crop_center normally crops dead-center after the upscale; pan_x/pan_y (-1..1, see edl.py's
    Clip) slide that crop window within whatever slack the upscale leaves on each axis - 0 is the
    old centered behavior exactly. Only one axis typically HAS slack (scale locks one dimension to
    an exact match), so panning on the other axis is naturally a no-op - no extra clamping needed.
    """
    if fit == "crop_center":
        x = f"(in_w-out_w)/2*(1+({pan_x}))" if pan_x else "(in_w-out_w)/2"
        y = f"(in_h-out_h)/2*(1+({pan_y}))" if pan_y else "(in_h-out_h)/2"
        return (f"[{inp}]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:{x}:{y}[{out}]")
    # blur_pad: blurred background + the full frame centered (no crop, so no pan either)
    return (f"[{inp}]split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2[bgb];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2[{out}]")


def color_filter(c: Clip) -> str:
    return (f"eq=brightness={c.brightness}:contrast={c.contrast}:saturation={c.saturation}"
            f":gamma_r={1 + c.warmth:.3f}:gamma_b={1 - c.warmth:.3f}")


FONT_CANDIDATES = [
    Path(__file__).parent / "fonts" / "Poppins-Bold.ttf",       # bundled with the project (OFL)
    Path("C:/Windows/Fonts/segoeuib.ttf"), Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
]
TEXT_SIZE = {"title": 0.058, "caption": 0.036, "cta": 0.042, "subtitle": 0.04}   # fraction of height
TEXT_FADE = 0.25


def find_font() -> Path:
    env = os.getenv("VEDIT_FONT")
    for f in ([Path(env)] if env else []) + FONT_CANDIDATES:
        if f.exists():
            return f
    raise RuntimeError("No font found for text (set VEDIT_FONT=path/to/font.ttf)")


def build_segment_cmd(c: Clip, info: MediaInfo, w: int, h: int, fps: int,
                      crf: int, dst: Path, tail_s: float = 0.0) -> tuple[list[str], int]:
    """The command for one segment + the EXACT frame count (for offset calculations).

    tail_s = how much the segment must be extended at the end, for the transition into the next clip.
    If the source has no extra material, the last frame is frozen (tpad clone).
    """
    main_frames = round(clip_duration(c) * fps)
    tail_frames = round(tail_s * fps)
    frames = main_frames + tail_frames
    out = ["-map", "[v]", "-an", "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
           "-video_track_timescale", "15360", str(dst)]

    if info.kind == "video":
        # tail_s/clip_duration are OUTPUT (on-screen, post-speed) seconds; converting to RAW source
        # seconds to read is what speed actually means: speed=2x timelapse needs HALF the raw footage
        # for the same on-screen time, speed=0.5x slow-mo needs DOUBLE.
        raw_main = c.out - c.in_
        raw_tail = tail_s * c.speed
        raw_need = raw_main + raw_tail
        has_room = c.in_ + raw_need <= info.duration - 0.05
        pad = "" if has_room or not tail_frames else f",tpad=stop_mode=clone:stop_duration={tail_s + 0.2:.3f}"
        speed_f = "" if c.speed == 1.0 else f"setpts=PTS/{c.speed:.4f},"
        # setpts must run BEFORE the fps filter: fps resamples to a frame count based on the
        # CURRENT duration, so the speed change has to already be applied for it to land right.
        graph = (fit_filter(c.fit, w, h, "0:v", "f", c.pan_x, c.pan_y) +
                 f";[f]{speed_f}fps={fps},{color_filter(c)},setsar=1,format=yuv420p{pad}[v]")
        take = raw_need + 0.2 if has_room else raw_main + 0.1
        return ["ffmpeg", "-y", "-v", "error", "-ss", f"{c.in_:.3f}", "-t", f"{take:.3f}",
                "-i", str(info.path), "-filter_complex", graph] + out, frames

    # photo: fit at 2x resolution (zoom without pixelation), then zoompan over all frames (tail included)
    if c.motion == "zoom_in":
        z = f"1+{ZOOM_AMOUNT}*on/{frames}"
    elif c.motion == "zoom_out":
        z = f"{1 + ZOOM_AMOUNT}-{ZOOM_AMOUNT}*on/{frames}"
    else:
        z = "1"
    graph = (fit_filter(c.fit, w * 2, h * 2, "0:v", "f", c.pan_x, c.pan_y) +
             f";[f]zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
             f":d={frames}:s={w}x{h}:fps={fps},{color_filter(c)},setsar=1,format=yuv420p[v]")
    return ["ffmpeg", "-y", "-v", "error", "-i", str(info.path),
            "-filter_complex", graph] + out, frames


def clip_cache_key(c: Clip, info: MediaInfo, w: int, h: int, fps: int, crf: int, tail_s: float) -> str:
    """Identifies a rendered segment by everything that affects its own pixels: the clip's own
    fields, the SOURCE FILE actually read (path + mtime - a proxy that finishes transcoding mid-
    session, or gets rebuilt, must invalidate any segment already cached from the stale file) and
    the output format (resolution/fps/quality/transition tail). Preview-only (see _render_in)."""
    payload = {
        "clip": c.model_dump(by_alias=True, exclude_none=True),
        "path": str(info.path), "mtime": info.path.stat().st_mtime if info.path.exists() else 0,
        "w": w, "h": h, "fps": fps, "crf": crf, "tail": round(tail_s, 3),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def scan_media(input_dir: Path) -> dict[str, MediaInfo]:
    """Audit fix: a single corrupt/unreadable file used to take down every endpoint that calls this
    (ffprobe raising via probe()'s check=True propagated straight out of scan_media, and almost every
    route - get_project, upload, render, plan, ... - calls scan_media unconditionally). One bad file
    could permanently lock a user out of an otherwise-fine project. Now a file that fails to probe is
    just skipped (and logged) instead of taking the whole directory listing down with it; callers that
    reference it by name (EDL validation, etc.) already handle a missing media entry gracefully."""
    media = {}
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
            try:
                media[p.name] = probe(p)
            except Exception:  # noqa: BLE001 - a corrupt/unsupported file must not break the whole project
                traceback.print_exc()
    return media


def render(edl: EDL, media: dict[str, MediaInfo], dst: Path, draft: bool,
           music: dict | None = None, ambience: dict | None = None, sfx: list[dict] | None = None,
           subtitles: list[TextItem] | None = None) -> Path:
    """music / ambience: {"path", "start", "volume", "fade_in", "fade_out", "duration", "duck", ...} or
    None - two independent background layers (ambience has no beat-sync, just a second bed under
    the music). music can also carry "punch"/"punch_every"/"punch_amount" (B1 - beat-synced visual
    punches): a quick zoom pulse on the whole frame on every Nth beat, music_ctx() in main.py is what
    actually fills these in from the project's saved music settings. sfx: [{"path", "start", "volume"},
    ...] - one-shot cues from the SFX library, never ducked. The clips' original sound (Clip.volume)
    is added automatically if any source has audio."""
    w, h, fps = edl.output.width, edl.output.height, edl.output.fps
    crf = 28 if draft else 20
    if draft:
        w, h = w // 2, h // 2
    dst = Path(dst).resolve()
    with tempfile.TemporaryDirectory(prefix="vedit_") as tmp:
        work = Path(tmp)
        video = work / "video.mp4"
        has_overlays = bool(edl.overlays)
        main_frames = _render_in(edl, media, video, work, w, h, fps, crf, subtitles or [], skip_text=has_overlays)
        if has_overlays:
            composite_overlays(edl, media, video, work, w, h, fps)
            apply_text_pass(edl.texts, subtitles or [], work, video, w, h, fps, crf)
        apply_beat_punch_if_configured(music, video, work, w, h, fps, crf, main_frames)
        orig = build_original_audio(edl, media, work, main_frames, fps)
        length = sum(main_frames) / fps
        if music or ambience or sfx or orig:
            mux_audio(video, music, ambience, sfx, orig, length, dst)
        else:
            shutil.move(str(video), dst)
    return dst


def beat_punch_graph(beats, fps: int, total_frames: int, amount: float, width_s: float = 0.12) -> str | None:
    """A zoom-factor expression (as a function of `on`, zoompan's OUTPUT frame counter) that sits at
    1.0 except for a short triangular bump of height `amount` centered on each beat's frame - the
    whole visible frame (picture, overlays, text - everything already composited by this point) snaps
    out and back in on the beat. Frame-based rather than `t`-based: ffmpeg's crop filter only
    evaluates w/h once at init in this build (no per-frame `eval=frame` option on it), so zoompan -
    whose z/x/y genuinely re-evaluate every output frame - is what actually animates here. None when
    nothing lands inside the clip (e.g. the music starts after the video ends)."""
    width_f = max(1, round(width_s * fps))
    sel = [round(float(b) * fps) for b in beats if 0 <= b * fps < total_frames]
    if not sel:
        return None
    terms = "+".join(f"{amount:.4f}*max(0,1-abs(on-{f})/{width_f})" for f in sel)
    return f"1+({terms})"


def apply_beat_punch(video: Path, work: Path, w: int, h: int, fps: int, crf: int, zexpr: str) -> None:
    """zoompan with d=1 (no frame-hold/duplication - each input frame maps to exactly one output
    frame) re-centers and scales by the animated zoom factor every frame, for a whole-frame punch on
    an already-assembled, already-correct-size video."""
    tmp = work / "video_punch.mp4"
    vf = (f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
           "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed applying beat punches:\n{r.stderr[-2000:]}")
    tmp.replace(video)


def apply_beat_punch_if_configured(music: dict | None, video: Path, work: Path, w: int, h: int,
                                    fps: int, crf: int, main_frames: list[int]) -> None:
    if not music or not music.get("punch") or not music.get("info"):
        return
    from .music import beat_grid
    beats = beat_grid(music["info"], music["start"])
    every = max(1, int(music.get("punch_every", 2)))
    total_frames = sum(main_frames)
    zexpr = beat_punch_graph(beats[::every], fps, total_frames, float(music.get("punch_amount", 0.06)))
    if zexpr:
        apply_beat_punch(video, work, w, h, fps, crf, zexpr)


PREVIEW_SEGCACHE_DIR = "previews/segcache"


def render_preview(edl: EDL, media: dict[str, MediaInfo], dst: Path, project_dir: Path,
                    music: dict | None = None, ambience: dict | None = None, sfx: list[dict] | None = None,
                    subtitles: list[TextItem] | None = None) -> Path:
    """The fast path for 'see my edit' (task A1 - live/instant preview), NOT a substitute for
    render(). Two things make it fast instead of a full draft render: `media` is expected to already
    be proxy-swapped (app.proxy.preview_media - small, fast-seeking files) by the caller, and each
    clip's own encoded segment is cached on disk across calls (clip_cache_key) so editing ONE clip
    only re-encodes that one - every untouched clip reuses its already-rendered segment. Always
    draft-quality/half-resolution, regardless of edl.output, and always writes to the same `dst` path
    (overwritten every call) - there's exactly one live preview per project, not a history of them.

    Safety note: this is a COMPLETELY SEPARATE path from render() - render() never receives a
    cache_dir and never sees proxy media, so a low-quality/proxy-sourced segment can never end up in
    a real draft/final export by accident."""
    # quarter-res (half of draft's own half) - speed matters more than fidelity for a quick preview,
    # and the segment cache means paying this resolution's encode cost at most once per clip-state.
    w, h, fps = max(64, edl.output.width // 4), max(64, edl.output.height // 4), edl.output.fps
    crf = 30
    dst = Path(dst).resolve()
    cache_dir = project_dir / PREVIEW_SEGCACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vedit_preview_") as tmp:
        work = Path(tmp)
        video = work / "video.mp4"
        has_overlays = bool(edl.overlays)
        main_frames = _render_in(edl, media, video, work, w, h, fps, crf, subtitles or [],
                                  skip_text=has_overlays, cache_dir=cache_dir)
        if has_overlays:
            composite_overlays(edl, media, video, work, w, h, fps)
            apply_text_pass(edl.texts, subtitles or [], work, video, w, h, fps, crf)
        apply_beat_punch_if_configured(music, video, work, w, h, fps, crf, main_frames)
        orig = build_original_audio(edl, media, work, main_frames, fps)
        length = sum(main_frames) / fps
        if music or ambience or sfx or orig:
            mux_audio(video, music, ambience, sfx, orig, length, dst)
        else:
            shutil.move(str(video), dst)
    return dst


def prune_preview_cache(project_dir: Path, max_age_days: float = 2.0) -> None:
    """Segment cache files pile up (one per distinct clip-state ever previewed) and old entries are
    nearly always dead (the clip was edited again, or the project moved on) - called opportunistically
    on upload/delete rather than on any clock, no background scheduler needed."""
    cache_dir = project_dir / PREVIEW_SEGCACHE_DIR
    if not cache_dir.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    for f in cache_dir.glob("*.mp4"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def wrap(text: str, max_chars: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def text_filters(texts: list[TextItem], w: int, h: int, work: Path) -> list[str]:
    """One drawtext per line (individually centered). The text and font live in files in the work
    folder, referenced relatively -> no escaping issues (diacritics, ':' from Windows paths)."""
    if not texts:
        return []
    shutil.copy(find_font(), work / "font.ttf")
    out = []
    for k, t in enumerate(texts):
        fs = round(TEXT_SIZE[t.style] * h)
        lines = wrap(t.content, max(8, int(0.84 * w / (fs * 0.56))))
        line_h = round(fs * 1.3)
        block = line_h * len(lines)
        free = t.x is not None or t.y is not None
        if free:
            # (x, y) is the CENTER of the whole (possibly multi-line) block, as a fraction of the
            # frame - each line still gets its own x expression below so it stays centered on cx
            # even though lines can differ in width.
            cx, cy = (t.x if t.x is not None else 0.5), (t.y if t.y is not None else 0.5)
            x_expr = f"{cx:.4f}*w-text_w/2"
            y0 = round(cy * h - block / 2)
        else:
            x_expr = "(w-text_w)/2"
            y0 = {"top": round(0.13 * h), "center": round((h - block) / 2), "bottom": round(0.70 * h - block)}[t.position]
        st, en = t.start, t.end
        alpha = (f"if(lt(t,{st + TEXT_FADE:.3f}),(t-{st:.3f})/{TEXT_FADE},"
                 f"if(gt(t,{en - TEXT_FADE:.3f}),({en:.3f}-t)/{TEXT_FADE},1))")
        if t.style == "subtitle":
            look = f"fontcolor=white:borderw={max(2, round(fs * 0.09))}:bordercolor=black@0.85"
        elif t.style == "title":
            look = f"fontcolor=white:borderw={max(2, round(fs * 0.07))}:bordercolor=black@0.55:shadowx=0:shadowy={max(1, fs // 18)}:shadowcolor=black@0.4"
        elif t.style == "cta":
            look = f"fontcolor=white:box=1:boxcolor=0xFF7A45@0.92:boxborderw={round(fs * 0.4)}"
        else:
            look = f"fontcolor=white:box=1:boxcolor=black@0.5:boxborderw={round(fs * 0.3)}"
        for li, line in enumerate(lines):
            name = f"t{k}_{li}.txt"
            (work / name).write_text(line, encoding="utf-8")
            out.append(f"drawtext=fontfile=font.ttf:textfile={name}:fontsize={fs}:{look}"
                       f":x={x_expr}:y={y0 + li * line_h}"
                       f":alpha='{alpha}':enable='between(t,{st:.3f},{en:.3f})'")
    return out


def place_texts(texts: list[TextItem], subtitles: list[TextItem]) -> list[TextItem]:
    """Subtitles occupy the bottom area: bottom texts that overlap them in time move up."""
    out = []
    for t in texts:
        if t.position == "bottom" and any(s.start < t.end and t.start < s.end for s in subtitles):
            t = t.model_copy(update={"position": "top"})
        out.append(t)
    return out + subtitles


def _render_in(edl, media, dst, work, w, h, fps, crf, subtitles=(), skip_text=False, cache_dir=None):
    """cache_dir: ONLY ever set by the quick-preview pipeline (render_preview, below) - a persistent,
    cross-call segment cache keyed by everything that affects a clip's own rendered bytes, so an edit
    to ONE clip doesn't force every other clip to be re-encoded again on the next preview. The real
    export path (render()) never passes this - it always encodes fresh into the throwaway temp `work`
    dir, so a preview-quality/proxy-sourced segment can never leak into a real draft/final export."""
    clips = edl.clips
    tails = [(clips[i + 1].transition_in.duration if i + 1 < len(clips) and clips[i + 1].transition_in else 0.0)
             for i in range(len(clips))]
    seg_paths, main_frames = [], []
    for i, c in enumerate(clips):
        info = media[c.source]
        if cache_dir is not None:
            seg_path = cache_dir / f"{clip_cache_key(c, info, w, h, fps, crf, tails[i])}.mp4"
            cached = seg_path.exists()
        else:
            seg_path = work / f"seg_{i:03d}.mp4"
            cached = False
        if not cached:
            t0 = time.time()
            cmd, frames = build_segment_cmd(c, info, w, h, fps, crf, seg_path, tails[i])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed on clip {i} ({c.source}):\n{r.stderr}")
            print(f"  [{i + 1}/{len(clips)}] {c.source} -> {seg_path.name} ({time.time() - t0:.1f}s)")
        else:
            frames = round(clip_duration(c) * fps) + round(tails[i] * fps)
        seg_paths.append(seg_path)
        main_frames.append(frames - round(tails[i] * fps))

    has_tr = any(c.transition_in for c in clips[1:])
    # skip_text: overlays (B-roll/PIP) need to end up UNDER the text, but they're composited in
    # their own pass AFTER this one - so when there are overlays, text is deferred to a final pass
    # over the composited result instead of being baked in here (apply_text_pass, called from render()).
    texts = [] if skip_text else text_filters(place_texts(edl.texts, list(subtitles)), w, h, work)

    if not has_tr and not texts:
        # fastest path: join without re-encoding
        (work / "concat.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in seg_paths), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                        "-c", "copy", "-movflags", "+faststart", str(dst)], check=True, cwd=work)
        return main_frames

    # a single final re-encoded pass: transitions (xfade) / cuts (concat) + text (drawtext)
    # all inputs on the same timebase (concat strips AVTB, xfade needs identical timebases)
    parts = [f"[{i}:v]settb=AVTB[s{i}]" for i in range(len(clips))]
    cur = "[s0]"
    start_frame = main_frames[0]
    for i in range(1, len(clips)):
        nxt = f"[x{i}]"
        tr = clips[i].transition_in
        if tr:
            parts.append(f"{cur}[s{i}]xfade=transition={tr.type}:duration={tr.duration:.3f}"
                         f":offset={start_frame / fps:.4f}{nxt}")
        else:
            parts.append(f"{cur}[s{i}]concat=n=2:v=1:a=0{nxt}")
        cur = nxt
        start_frame += main_frames[i]
    chain = ",".join(texts + ["format=yuv420p"])
    parts.append(f"{cur}{chain}[v]")
    total = start_frame / fps
    inputs = [a for p in seg_paths for a in ("-i", str(p))]
    # the graph is merged inline (not -filter_complex_script: deprecated in ffmpeg 7+, possibly removed in 8/9)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(parts),
                        "-map", "[v]", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
                        "-movflags", "+faststart", str(dst)], capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on the final assembly:\n{r.stderr[-2000:]}")
    return main_frames


OVERLAY_MARGIN = 0.04   # fraction of frame width - PIP corner margin


def build_overlay_segment_cmd(o: OverlayItem, info: MediaInfo, fps: int, dst: Path) -> list[str]:
    """Normalizes one overlay's own footage/photo to an fps-locked clip of the right length, at its
    OWN native resolution/aspect (no scale/crop here - composite_overlays does that per-mode, against
    the main video's dimensions)."""
    dur = overlay_duration(o)
    frames = round(dur * fps)
    out = ["-map", "[v]", "-an", "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", "15360", str(dst)]
    graph = ["-filter_complex", f"[0:v]fps={fps},setsar=1,format=yuv420p[v]"]
    if info.kind == "video":
        return (["ffmpeg", "-y", "-v", "error", "-ss", f"{o.in_:.3f}", "-t", f"{dur + 0.2:.3f}",
                 "-i", str(info.path)] + graph + out)
    return ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", f"{dur + 0.2:.3f}", "-i", str(info.path)] + graph + out


def composite_overlays(edl: EDL, media: dict, video: Path, work: Path, w: int, h: int, fps: int) -> None:
    """Composites each OverlayItem (B-roll-in-a-corner / split-screen) on top of the already-
    finished main video, time-gated to its own [start, start+duration] window. Its own filter_complex
    pass (not folded into _render_in's already-busy xfade/concat/drawtext graph) - re-encodes
    video.mp4 in place. A no-op when there are no overlays, so every project without one pays nothing."""
    if not edl.overlays:
        return
    names = []
    for i, o in enumerate(edl.overlays):
        name = f"ov_{i:03d}.mp4"
        subprocess.run(build_overlay_segment_cmd(o, media[o.source], fps, work / name), check=True, cwd=work)
        names.append(name)

    margin = round(OVERLAY_MARGIN * w)
    # overlay's x/y expressions see the OVERLAY input's own size as overlay_w/overlay_h (NOT ow/oh -
    # that's a different filter's variable naming; getting this wrong fails at filter-graph eval time).
    pip_xy = {
        "pip_tl": (str(margin), str(margin)),
        "pip_tr": (f"({w}-overlay_w-{margin})", str(margin)),
        "pip_bl": (str(margin), f"({h}-overlay_h-{margin})"),
        "pip_br": (f"({w}-overlay_w-{margin})", f"({h}-overlay_h-{margin})"),
    }
    inputs = ["-i", str(video)]
    parts = []
    cur = "0:v"
    for i, (o, name) in enumerate(zip(edl.overlays, names)):
        inputs += ["-i", name]
        st, en = o.start, o.start + overlay_duration(o)
        scaled = f"[sov{i}]"
        if o.mode in ("split_left", "split_right"):
            bw, bh = w // 2, h
            x, y = ("0" if o.mode == "split_left" else str(w - bw)), "0"
            parts.append(f"[{i + 1}:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                         f"crop={bw}:{bh}{scaled}")
        else:
            bw = round(o.scale * w)
            x, y = pip_xy[o.mode]
            parts.append(f"[{i + 1}:v]scale={bw}:-2{scaled}")
        nxt_lbl = f"c{i}"
        parts.append(f"[{cur}]{scaled}overlay=x={x}:y={y}:enable='between(t,{st:.3f},{en:.3f})'[{nxt_lbl}]")
        cur = nxt_lbl
    tmp = work / "video_ov.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(parts),
           "-map", f"[{cur}]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed compositing overlays:\n{r.stderr[-2000:]}")
    tmp.replace(video)


def apply_text_pass(texts_src: list[TextItem], subtitles: list[TextItem], work: Path, video: Path,
                     w: int, h: int, fps: int, crf: int) -> None:
    """The deferred drawtext pass used when there are overlays (see _render_in's skip_text) - runs
    AFTER composite_overlays so captions/titles always end up on top of a PIP/split layer, never
    hidden behind one. A no-op when there's no text to draw."""
    texts = text_filters(place_texts(texts_src, list(subtitles)), w, h, work)
    if not texts:
        return
    chain = ",".join(texts + ["format=yuv420p"])
    tmp = work / "video_txt.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-filter_complex", f"[0:v]{chain}[v]",
           "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed applying text over the overlay layer:\n{r.stderr[-2000:]}")
    tmp.replace(video)


AR = 48000


def atempo_chain(speed: float) -> str:
    """ffmpeg's atempo filter only accepts 0.5-2.0 per instance; chain instances to cover the
    clip speed range [0.25, 4.0] (e.g. 4.0 -> atempo=2.0,atempo=2.0)."""
    factors, s = [], speed
    while s < 0.5:
        factors.append(0.5); s /= 0.5
    while s > 2.0:
        factors.append(2.0); s /= 2.0
    factors.append(round(s, 6))
    return ",".join(f"atempo={f}" for f in factors)


def build_original_audio(edl: EDL, media: dict, work: Path, main_frames: list[int], fps: int) -> Path | None:
    """The clips' ORIGINAL sound track, placed on the timeline.

    Normally each clip's audio chunk is exactly main_frames/fps seconds long and chunks sit
    end-to-end with no overlap (same as a plain cut). A clip with audio_lead/audio_lag (J/L cuts,
    edl.py) gets a LONGER chunk that starts earlier and/or ends later than its own on-screen video
    window, so it's placed with a negative offset into the PREVIOUS clip's time (J-cut) and/or
    spills past its own end into the NEXT clip's time (L-cut); overlapping chunks are mixed
    together rather than concatenated, with a fade across the borrowed window so the handoff is
    soft, not a hard double-audio clash. With no lead/lag anywhere this produces the exact same
    result as a plain concat (the general case subsumes the old one).

    Clips with no sound, photos, and clips with volume=0 become silence (a pure delay offset, no
    audio input). A sped-up/slowed-down clip's audio is time-stretched (atempo) to the same factor,
    so it stays in sync with the picture - no raw pitch-shifted sound. Returns None if there's no
    sound to use at all.
    """
    clips = edl.clips
    use = [media[c.source].kind == "video" and media[c.source].has_audio and c.volume > 0 for c in clips]
    if not any(use):
        return None
    starts, acc = [], 0.0
    for n in main_frames:
        starts.append(acc)
        acc += n / fps
    total_len = acc

    chunks = []   # (name, absolute start offset on the final timeline, in seconds)
    for i, (c, ok) in enumerate(zip(clips, use)):
        if not ok:
            continue
        dur = main_frames[i] / fps
        info = media[c.source]
        lead = min(c.audio_lead, main_frames[i - 1] / fps if i > 0 else 0.0)
        lag = min(c.audio_lag, main_frames[i + 1] / fps if i + 1 < len(clips) else 0.0)
        raw_in, raw_out = c.in_ - lead * c.speed, c.out + lag * c.speed
        if raw_in < 0:                                  # the source doesn't have that much room before `in`
            lead, raw_in = c.in_ / c.speed, 0.0
        if raw_out > info.duration:                      # ...or after `out`
            lag, raw_out = max(0.0, (info.duration - c.out) / c.speed), info.duration
        ext_dur = lead + dur + lag   # this chunk's own, possibly-extended on-screen length
        raw_dur = raw_out - raw_in
        name = f"a_{i:03d}.wav"
        tempo = f"{atempo_chain(c.speed)}," if c.speed != 1.0 else ""
        # fade across the borrowed window when there's a J/L cut; otherwise the usual tiny
        # anti-pop fade at the true (unextended) edges, same as a plain cut always had
        fade_in = f"afade=t=in:d={lead:.3f}" if lead > 0 else "afade=t=in:d=0.03"
        fade_out = (f"afade=t=out:st={max(0, ext_dur - lag):.3f}:d={lag:.3f}" if lag > 0
                    else f"afade=t=out:st={max(0, ext_dur - 0.03):.3f}:d=0.03")
        af = (f"aresample={AR},aformat=channel_layouts=stereo,{tempo}volume={c.volume:.2f},"
              f"{fade_in},{fade_out},apad,atrim=0:{ext_dur:.5f}")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{raw_in:.3f}", "-t", f"{raw_dur + 0.1:.3f}",
               "-i", str(info.path), "-vn", "-af", af, "-ar", str(AR), "-ac", "2", "-c:a", "pcm_s16le", name]
        subprocess.run(cmd, check=True, cwd=work)
        chunks.append((name, starts[i] - lead))

    inputs = [a for n, _ in chunks for a in ("-i", n)]
    graph: list[str] = []
    cur = None
    for i, (_, off) in enumerate(chunks):
        ms = max(0, round(off * 1000))
        lbl = f"[oa{i}]"
        graph.append(f"[{i}:a]adelay={ms}:all=1,apad{lbl}")
        if cur is None:
            cur = lbl
        else:
            out = f"[om{i}]"
            graph.append(f"{cur}{lbl}amix=inputs=2:duration=first:normalize=0{out}")
            cur = out
    graph.append(f"{cur}atrim=0:{total_len:.5f}[a]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "[a]", "-ar", str(AR), "-ac", "2", "-c:a", "pcm_s16le", "original.wav"],
                   check=True, cwd=work)
    return work / "original.wav"


def mux_audio(video: Path, music: dict | None, ambience: dict | None, sfx: list[dict] | None,
              orig: Path | None, length: float, dst: Path) -> None:
    """Puts the sound on top of the video. The video dictates the length.

    - music / ambience: two independent background beds (start in the track, volume, fade in/out,
      apad if shorter -> silence); if both are present they're summed into one background first.
      ambience has no beat-sync - it's just a second, usually-quieter layer under the music (room
      tone, crowd noise, rain...).
    - original sound: the synced track built by build_original_audio
    - background vs original: the background is automatically DUCKED while the original sound is
      heard (sidechaincompress), unless either layer turned that off
    - sfx: one-shot cues from the library (path, start, volume), each delayed to its start time and
      mixed in LAST, after ducking - a whoosh/ding should cut through, not get dampened like a bed
    """
    inputs = ["-i", str(video)]
    graph: list[str] = []

    def next_idx() -> int:
        return inputs.count("-i")   # index of the next input (careful: -ss also takes a slot in the list)

    def bed_filter(info: dict) -> str:
        avail = max(0.1, info["duration"] - info["start"])
        end = min(length, avail)
        mf = [f"volume={info['volume']:.2f}"]
        if info.get("fade_in", 0) > 0:
            mf.append(f"afade=t=in:st=0:d={info['fade_in']:.2f}")
        if info.get("fade_out", 0) > 0:
            mf.append(f"afade=t=out:st={max(0, end - info['fade_out']):.2f}:d={info['fade_out']:.2f}")
        mf += [f"aresample={AR}", "aformat=channel_layouts=stereo", "apad"]
        return ",".join(mf)

    bg = None   # current background label, or None if neither music nor ambience is present
    for info in (music, ambience):
        if not info:
            continue
        k = next_idx()
        inputs += ["-ss", f"{info['start']:.3f}", "-i", str(info["path"])]
        lbl = f"[bgin{k}]"
        graph.append(f"[{k}:a]{bed_filter(info)}{lbl}")
        if bg is None:
            bg = lbl
        else:
            graph.append(f"{bg}{lbl}amix=inputs=2:duration=first:normalize=0[bg]")
            bg = "[bg]"

    cur = None   # running label for everything mixed so far
    if orig:
        k = next_idx()
        inputs += ["-i", str(orig)]
        if bg:
            duck = (music or {}).get("duck", True) or (ambience or {}).get("duck", True)
            if duck:
                graph.append(f"[{k}:a]asplit=2[o1][o2]")
                graph.append(f"{bg}[o1]sidechaincompress=threshold=0.015:ratio=12:attack=20:release=450[bgd]")
                graph.append("[bgd][o2]amix=inputs=2:duration=first:normalize=0[base]")
            else:
                graph.append(f"{bg}[{k}:a]amix=inputs=2:duration=first:normalize=0[base]")
            cur = "[base]"
        else:
            cur = f"[{k}:a]"
    elif bg:
        cur = bg

    for i, s in enumerate(sfx or []):
        k = next_idx()
        inputs += ["-i", str(s["path"])]
        ms = max(0, round(s["start"] * 1000))
        lbl = f"[sfx{i}]"
        graph.append(f"[{k}:a]aresample={AR},aformat=channel_layouts=stereo,volume={s['volume']:.2f},"
                     f"adelay={ms}:all=1,apad{lbl}")
        if cur is None:
            cur = lbl
        else:
            out = f"[mix{i}]"
            graph.append(f"{cur}{lbl}amix=inputs=2:duration=first:normalize=0{out}")
            cur = out

    graph.append(f"{cur}anull[a]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{length:.3f}", "-movflags", "+faststart", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser(description="Renders an EDL JSON into an MP4.")
    ap.add_argument("edl", help="the EDL .json file")
    ap.add_argument("--input", default=".", help="the folder with the raw material")
    ap.add_argument("--out", default=None, help="the resulting MP4 file")
    ap.add_argument("--final", action="store_true", help="final render (default: draft at half resolution)")
    a = ap.parse_args()

    input_dir = Path(a.input).resolve()
    media = scan_media(input_dir)
    edl = validate_edl(json.loads(Path(a.edl).read_text(encoding="utf-8")), media)
    draft = not a.final
    out = Path(a.out) if a.out else Path("output") / ("drafts" if draft else "final") / (Path(a.edl).stem + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Valid EDL: {len(edl.clips)} clips, {total_duration(edl):.1f}s, "
          f"{'DRAFT' if draft else 'FINAL'}")
    t0 = time.time()
    render(edl, media, out, draft)
    info = probe(out)
    print(f"Done in {time.time() - t0:.1f}s -> {out} ({info.width}x{info.height}, "
          f"{info.fps:.0f}fps, {info.duration:.2f}s)")


if __name__ == "__main__":
    main()
