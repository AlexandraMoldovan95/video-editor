"""Validated EDL -> ffmpeg commands -> MP4.

Strategy: each clip is rendered separately into a NORMALIZED segment
(same resolution, fps, codec, pix_fmt) with an EXACT frame count.
- no transitions and no text: the segments are joined with the concat demuxer, no re-encoding (fast)
- otherwise: a single final re-encoded pass (xfade for transitions, concat for cuts,
  drawtext for text). The transition starts EXACTLY on the boundary between clips: the previous
  clip is extended by the transition's duration -> the total duration and beat alignment stay unchanged.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .edl import EDL, Clip, TextItem, clip_duration, total_duration, validate_edl
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


def scan_media(input_dir: Path) -> dict[str, MediaInfo]:
    media = {}
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
            media[p.name] = probe(p)
    return media


def render(edl: EDL, media: dict[str, MediaInfo], dst: Path, draft: bool,
           audio: dict | None = None, subtitles: list[TextItem] | None = None) -> Path:
    """audio = music: {"path", "start", "volume", "fade_in", "fade_out", "duration", "duck"} or None.
    The clips' original sound (Clip.volume) is added automatically if any source has audio."""
    w, h, fps = edl.output.width, edl.output.height, edl.output.fps
    crf = 28 if draft else 20
    if draft:
        w, h = w // 2, h // 2
    dst = Path(dst).resolve()
    with tempfile.TemporaryDirectory(prefix="vedit_") as tmp:
        work = Path(tmp)
        video = work / "video.mp4"
        main_frames = _render_in(edl, media, video, work, w, h, fps, crf, subtitles or [])
        orig = build_original_audio(edl, media, work, main_frames, fps)
        length = sum(main_frames) / fps
        if audio or orig:
            mux_audio(video, audio, orig, length, dst)
        else:
            shutil.move(str(video), dst)
    return dst


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


def _render_in(edl, media, dst, work, w, h, fps, crf, subtitles=()):
    clips = edl.clips
    tails = [(clips[i + 1].transition_in.duration if i + 1 < len(clips) and clips[i + 1].transition_in else 0.0)
             for i in range(len(clips))]
    seg_names, main_frames = [], []
    for i, c in enumerate(clips):
        name = f"seg_{i:03d}.mp4"
        t0 = time.time()
        cmd, frames = build_segment_cmd(c, media[c.source], w, h, fps, crf, work / name, tails[i])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on clip {i} ({c.source}):\n{r.stderr}")
        print(f"  [{i + 1}/{len(clips)}] {c.source} -> {name} ({time.time() - t0:.1f}s)")
        seg_names.append(name)
        main_frames.append(frames - round(tails[i] * fps))

    has_tr = any(c.transition_in for c in clips[1:])
    texts = text_filters(place_texts(edl.texts, list(subtitles)), w, h, work)

    if not has_tr and not texts:
        # fastest path: join without re-encoding
        (work / "concat.txt").write_text("".join(f"file '{n}'\n" for n in seg_names), encoding="utf-8")
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
    inputs = [a for n in seg_names for a in ("-i", n)]
    # the graph is merged inline (not -filter_complex_script: deprecated in ffmpeg 7+, possibly removed in 8/9)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(parts),
                        "-map", "[v]", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
                        "-movflags", "+faststart", str(dst)], capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on the final assembly:\n{r.stderr[-2000:]}")
    return main_frames


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
    """The clips' ORIGINAL sound track, sample-synced to the video.

    Each chunk is exactly main_frames/fps seconds long (48000/30 = 1600 samples per frame, exact).
    Clips with no sound, photos, and clips with volume=0 become silence. A 30ms fade at both ends
    removes the pops at cuts. A sped-up/slowed-down clip's audio is time-stretched (atempo) to the
    same factor, so it stays in sync with the picture - no raw pitch-shifted sound. Returns None if
    there's no sound to use at all.
    """
    use = [media[c.source].kind == "video" and media[c.source].has_audio and c.volume > 0 for c in edl.clips]
    if not any(use):
        return None
    names = []
    for i, (c, ok) in enumerate(zip(edl.clips, use)):
        dur = main_frames[i] / fps
        name = f"a_{i:03d}.wav"
        if ok:
            raw_dur = dur * c.speed   # RAW source seconds needed before the tempo change gets us back to `dur`
            tempo = f"{atempo_chain(c.speed)}," if c.speed != 1.0 else ""
            af = (f"aresample={AR},aformat=channel_layouts=stereo,{tempo}volume={c.volume:.2f},"
                  f"afade=t=in:d=0.03,afade=t=out:st={max(0, dur - 0.03):.3f}:d=0.03,apad,atrim=0:{dur:.5f}")
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c.in_:.3f}", "-t", f"{raw_dur + 0.1:.3f}",
                   "-i", str(media[c.source].path), "-vn", "-af", af]
        else:
            cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=stereo",
                   "-t", f"{dur:.5f}"]
        subprocess.run(cmd + ["-ar", str(AR), "-ac", "2", "-c:a", "pcm_s16le", name], check=True, cwd=work)
        names.append(name)
    (work / "aconcat.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "aconcat.txt",
                    "-c", "copy", "original.wav"], check=True, cwd=work)
    return work / "original.wav"


def mux_audio(video: Path, music: dict | None, orig: Path | None, length: float, dst: Path) -> None:
    """Puts the sound on top of the video. The video dictates the length.

    - music: start in the track, volume, fade in/out, apad (if shorter -> silence)
    - original sound: the synced track built by build_original_audio
    - both: the music is automatically DUCKED while the original sound is heard (sidechaincompress)
    """
    inputs = ["-i", str(video)]
    graph, label = [], None
    if music:
        avail = max(0.1, music["duration"] - music["start"])
        end = min(length, avail)
        mf = [f"volume={music['volume']:.2f}"]
        if music["fade_in"] > 0:
            mf.append(f"afade=t=in:st=0:d={music['fade_in']:.2f}")
        if music["fade_out"] > 0:
            mf.append(f"afade=t=out:st={max(0, end - music['fade_out']):.2f}:d={music['fade_out']:.2f}")
        mf += [f"aresample={AR}", "aformat=channel_layouts=stereo", "apad"]
        inputs += ["-ss", f"{music['start']:.3f}", "-i", str(music["path"])]
        graph.append(f"[1:a]{','.join(mf)}[m]")
        label = "[m]"
    if orig:
        k = inputs.count("-i")   # index of the new input (careful: -ss also takes a slot in the list)
        inputs += ["-i", str(orig)]
        if label:
            if music.get("duck", True):
                graph.append(f"[{k}:a]asplit=2[o1][o2]")
                graph.append("[m][o1]sidechaincompress=threshold=0.015:ratio=12:attack=20:release=450[md]")
                graph.append("[md][o2]amix=inputs=2:duration=first:normalize=0[a]")
            else:
                graph.append(f"[m][{k}:a]amix=inputs=2:duration=first:normalize=0[a]")
            label = "[a]"
        else:
            graph.append(f"[{k}:a]anull[a]")
            label = "[a]"
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "0:v", "-map", label, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
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
