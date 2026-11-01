"""Local FastAPI server: projects, upload, per-file settings, AI plan, EDL editing, rendering.

Start:  uvicorn app.main:app --port 8000   ->  http://localhost:8000
Long operations (AI plan, render) run in the background; the frontend polls /api/jobs/{id}.
"""
import json
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .demo_presets import DEMO_PRESETS
from .edl import EDL, clamp_transitions, overlap_errors, text_errors, total_duration, validate_edl
from .frames import extract_frames, thumbnail
from .hints import FileHint
from .lockrules import LockRule, apply_lock_rules
from .proxy import build_proxies_bg, preview_info, preview_media, proxy_path
from . import fcpxml
from . import subtitles as subs_mod
from . import transcribe as tr_mod
from .music import AUDIO_EXT, MusicInfo, analyze, beat_grid, snap_to_beats, waveform as compute_waveform
from .presets import (COLOR_BASE, CUSTOM_KEYS, PACE_RANGE, PRESETS, Preset, add_custom_preset,
                       list_custom_presets, remove_custom_preset)
from .probe import IMAGE_EXT
from .render import VIDEO_EXT, prune_preview_cache, render, render_preview, scan_media

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
PROJECTS.mkdir(exist_ok=True)
ALLOWED = VIDEO_EXT | IMAGE_EXT

app = FastAPI(title="Video Editor")
JOBS: dict[str, dict] = {}
PLANNING: set[str] = set()   # projects with an AI plan in progress -> EDL edits are refused (would be lost otherwise)


# ---------- utilities ----------
def pdir(pid: str) -> Path:
    if not re.fullmatch(r"[a-z0-9]{8}", pid):
        raise HTTPException(400, "invalid project id")
    d = PROJECTS / pid
    if not d.exists():
        raise HTTPException(404, "project does not exist")
    return d


def safe_name(name: str) -> str:
    name = Path(name).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"file type not allowed: {name}")
    return name


def load_state(d: Path) -> dict:
    f = d / "state.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {
        "preset": "neutral", "overrides": {}, "hints": {}, "edl": None, "reasoning": "", "usage": None,
        "music": None, "ambience": None, "is_demo": None, "lock_rules": []}


def save_state(d: Path, st: dict) -> None:
    """Audit fix: write to a temp file + atomic replace, same pattern already used for proxies
    (app/proxy.py's build_proxy). A plain write_text() truncates the file first - if the process
    is killed mid-write (server crash, force-quit, power loss) while state.json is being saved,
    the project's entire state (EDL, music, hints, everything) is lost/corrupted, and every future
    load_state() call for that project throws on the truncated JSON. The temp+replace swap is
    atomic at the OS level: a reader always sees either the old, complete file or the new, complete
    one - never a half-written one."""
    f = d / "state.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(f)


def example_path(d: Path) -> Path | None:
    files = [p for p in (d / "example").glob("*") if p.suffix.lower() in VIDEO_EXT]
    return files[0] if files else None


MUSIC_INFO_KEYS = ("duration", "bpm", "beats", "energy")


def music_ctx(d: Path, st: dict) -> dict | None:
    """The music state -> dict for the planner/render (None if there's no music)."""
    m = st.get("music")
    if not m or not (d / "music" / m["file"]).exists():
        return None
    return {"info": MusicInfo(**{k: m[k] for k in MUSIC_INFO_KEYS}), "start": m["start"],
            "beat_sync": m["beat_sync"], "path": d / "music" / m["file"], "volume": m["volume"],
            "fade_in": m["fade_in"], "fade_out": m["fade_out"], "duration": m["duration"],
            "duck": m.get("duck", True), "punch": m.get("punch", False),
            "punch_every": m.get("punch_every", 2), "punch_amount": m.get("punch_amount", 0.06)}


def audio_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ambience_ctx(d: Path, st: dict) -> dict | None:
    """A second, simpler background bed - same shape as music_ctx's return but no beat info: it
    never drives beat-sync, it's just another layer under (or instead of) the music."""
    a = st.get("ambience")
    if not a or not (d / "ambience" / a["file"]).exists():
        return None
    return {"path": d / "ambience" / a["file"], "start": a["start"], "volume": a["volume"],
            "fade_in": a["fade_in"], "fade_out": a["fade_out"], "duration": a["duration"],
            "duck": a.get("duck", True)}


# ---------- sfx library (bundled, procedurally-generated sounds - no licensing to worry about) ----------
SFX_DIR = ROOT / "app" / "assets" / "sfx"
SFX_LIBRARY = {
    "whoosh": {"label": "Whoosh", "file": "whoosh.mp3"},
    "ding": {"label": "Ding", "file": "ding.mp3"},
    "impact": {"label": "Impact", "file": "impact.mp3"},
    "pop": {"label": "Pop", "file": "pop.mp3"},
    "click": {"label": "Click", "file": "click.mp3"},
}


def sfx_ctx(edl: EDL) -> list[dict]:
    return [{"path": SFX_DIR / SFX_LIBRARY[c.name]["file"], "start": c.start, "volume": c.volume}
            for c in edl.sfx]


def load_transcripts(d: Path, media: dict | None = None) -> dict[str, dict]:
    out = {}
    for f in (d / "transcripts").glob("*.json"):
        name = f.stem
        if media is None or name in media:
            out[name] = json.loads(f.read_text("utf-8"))
    return out


def subtitles_for(d: Path, st: dict, edl: EDL, media: dict) -> list:
    if not st.get("subtitles", True):
        return []
    return subs_mod.build(edl, load_transcripts(d, media))


JOB_MAX_AGE_S = 3600   # audit fix: see prune_jobs()


def prune_jobs() -> None:
    """JOBS never had anything removing old entries - every plan/render/preview/deepscan/transcribe
    click adds one and it stayed forever, for the life of the server process. Harmless for a short
    session, but a long-running one (this is a local app meant to be left open) would slowly grow an
    unbounded dict. Cheap to prune opportunistically (no background thread/scheduler needed) whenever
    a new job is about to be created - finished jobs older than JOB_MAX_AGE_S are dropped; anything
    still "running" is kept no matter its age."""
    cutoff = time.time() - JOB_MAX_AGE_S
    for jid in [j for j, v in JOBS.items() if v["status"] != "running" and v["started"] < cutoff]:
        JOBS.pop(jid, None)


def run_job(fn, *args) -> str:
    prune_jobs()
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "started": time.time(), "result": None, "error": None}

    def worker():
        try:
            JOBS[jid]["result"] = fn(*args)
            JOBS[jid]["status"] = "done"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            JOBS[jid]["status"], JOBS[jid]["error"] = "error", str(e)
    threading.Thread(target=worker, daemon=True).start()
    return jid


# ---------- config ----------
@app.get("/api/config")
def config():
    return {"presets": {k: v.model_dump() for k, v in PRESETS.items()},
            "custom_keys": sorted(CUSTOM_KEYS), "custom_presets": list_custom_presets(),
            "pace_range": PACE_RANGE, "color_base": COLOR_BASE}


# ---------- custom presets (save the current preset+overrides under a name, reuse it later) ----------
class CustomPresetIn(BaseModel):
    label: str = Field(max_length=60)
    preset: Preset


@app.post("/api/presets/custom")
def create_custom_preset(body: CustomPresetIn):
    label = body.label.strip()
    if not label:
        raise HTTPException(400, "name required")
    key = add_custom_preset(label, body.preset, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return {"ok": True, "key": key, "label": label, **body.preset.model_dump()}


@app.delete("/api/presets/custom/{key}")
def delete_custom_preset(key: str):
    if not remove_custom_preset(key):
        raise HTTPException(404, "no such custom preset (built-ins can't be deleted)")
    return {"ok": True}


# ---------- demo (pre-built example plans, no AI call) ----------
@app.get("/api/demo")
def list_demo_presets():
    return [{"key": k, "label": v["label"], "description": v["description"], "sources": v["sources"]}
            for k, v in DEMO_PRESETS.items()]


@app.post("/api/projects/{pid}/demo/{key}")
def apply_demo_preset(pid: str, key: str):
    d = pdir(pid)
    if pid in PLANNING:
        raise HTTPException(409, "the AI is generating the plan - wait for it to finish")
    preset = DEMO_PRESETS.get(key)
    if not preset:
        raise HTTPException(404, "unknown demo preset")
    media = scan_media(d / "input")
    missing = [s for s in preset["sources"] if s not in media]
    if missing:
        raise HTTPException(400, f"this demo needs these sample files uploaded first: {', '.join(missing)}")
    try:
        edl = validate_edl(preset["edl"], media)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    st = load_state(d)
    st["edl"] = edl.model_dump(by_alias=True, exclude_none=True)
    st["reasoning"] = preset.get("reasoning", "")
    st["usage"] = None
    st["is_demo"] = key
    if preset.get("overrides"):
        st["overrides"] = {**st.get("overrides", {}), **preset["overrides"]}
    save_state(d, st)
    return {"ok": True, "total": total_duration(edl), "overrides": st["overrides"]}


# ---------- projects ----------
@app.post("/api/projects")
def create_project():
    pid = uuid.uuid4().hex[:8]
    d = PROJECTS / pid
    for sub in ("input", "example", "music", "ambience", "thumbs", "output/drafts", "output/final"):
        (d / sub).mkdir(parents=True)
    save_state(d, load_state(d))
    return {"id": pid}


@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(PROJECTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "state.json").exists():
            out.append({"id": d.name, "files": len(list((d / "input").iterdir()))})
    return out


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    d = pdir(pid)
    st = load_state(d)
    st.setdefault("music", None)
    st.setdefault("ambience", None)
    st.setdefault("lock_rules", [])
    media = scan_media(d / "input")
    ex = example_path(d)
    trs = load_transcripts(d, media)
    ok, msg = tr_mod.available()
    preview = []
    if st["edl"] and trs:
        try:
            preview = [t.model_dump() for t in subtitles_for(d, st, EDL.model_validate(st["edl"]), media)]
        except Exception:  # noqa: BLE001
            preview = []
    return {
        "id": pid, **st,
        "example": ex.name if ex else None,
        "whisper": {"available": ok, "message": msg, "model": tr_mod.MODEL_NAME},
        "transcripts": {n: {"language": t.get("language"), "text": " ".join(s["text"] for s in t["segments"])[:300]}
                        for n, t in trs.items()},
        "subtitles": st.get("subtitles", True),
        "subtitles_preview": preview,
        "total": total_duration(EDL.model_validate(st["edl"])) if st["edl"] else 0,
        "media": [{"name": n, "kind": m.kind, "width": m.width, "height": m.height,
                   "fps": round(m.fps, 2) if m.fps else None, "duration": m.duration,
                   "has_audio": m.has_audio} for n, m in media.items()],
    }


@app.post("/api/projects/{pid}/upload")
async def upload(pid: str, files: list[UploadFile] = File(...)):
    d = pdir(pid)
    saved = []
    for f in files:
        name = safe_name(f.filename)
        dst = d / "input" / name
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out, 1024 * 1024)
        saved.append(name)
    media = scan_media(d / "input")
    # Audit fix: `media` only contains files ffprobe could actually read (scan_media now skips
    # anything corrupt/unsupported instead of raising - see render.py). A file that just failed to
    # upload cleanly (truncated transfer, not really a video/photo despite its extension) used to
    # raise a bare KeyError here and 500 the WHOLE batch, including every other file that uploaded
    # fine. Now each file is handled independently: good ones get a thumbnail and are reported back
    # as "saved"; bad ones are removed (nothing usable to keep around) and reported as "rejected" so
    # the UI can tell the user exactly which file(s) didn't work, instead of a generic failure.
    ok, rejected = [], []
    for n in saved:
        if n not in media:
            (d / "input" / n).unlink(missing_ok=True)
            rejected.append(n)
            continue
        try:
            thumbnail(media[n], d / "thumbs" / (Path(n).stem + ".jpg"))
        except Exception:  # noqa: BLE001 - a missing thumbnail is cosmetic, not worth failing the upload over
            traceback.print_exc()
        ok.append(n)
    threading.Thread(target=build_proxies_bg, args=(d, media, ok), daemon=True).start()
    prune_preview_cache(d)
    return {"saved": ok, "rejected": rejected}


@app.post("/api/projects/{pid}/example")
async def upload_example(pid: str, file: UploadFile = File(...)):
    d = pdir(pid)
    name = safe_name(file.filename)
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "the example video must be a video")
    for old in (d / "example").glob("*"):
        old.unlink()
    with (d / "example" / name).open("wb") as out:
        shutil.copyfileobj(file.file, out, 1024 * 1024)
    return {"example": name}


@app.delete("/api/projects/{pid}/example")
def delete_example(pid: str):
    for old in (pdir(pid) / "example").glob("*"):
        old.unlink()
    return {"example": None}


# ---------- music ----------
class MusicSettings(BaseModel):
    start: float = 0.0
    volume: float = 1.0
    fade_in: float = 0.5
    fade_out: float = 1.5
    beat_sync: bool = True
    duck: bool = True   # lowers the music while the original sound is heard
    # B1 (hard): beat-synced visual punches - a quick whole-frame zoom pulse on every Nth beat.
    # Lives alongside beat_sync rather than as its own endpoint: it's the same beat grid, same panel.
    punch: bool = False
    punch_every: int = Field(2, ge=1, le=8)
    punch_amount: float = Field(0.06, ge=0.0, le=0.2)


@app.post("/api/projects/{pid}/music")
async def upload_music(pid: str, file: UploadFile = File(...)):
    d = pdir(pid)
    name = Path(file.filename).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in AUDIO_EXT:
        raise HTTPException(400, f"audio format not allowed (accepted: {', '.join(sorted(AUDIO_EXT))})")
    (d / "music").mkdir(exist_ok=True)
    for old in (d / "music").glob("*"):
        old.unlink()
    dst = d / "music" / name
    with dst.open("wb") as out:
        shutil.copyfileobj(file.file, out, 1024 * 1024)
    try:
        info = analyze(dst)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"can't analyze the track: {e}")
    st = load_state(d)
    st["music"] = {"file": name, **info.to_dict(), **MusicSettings().model_dump()}
    save_state(d, st)
    return {"music": st["music"]}


@app.put("/api/projects/{pid}/music")
def put_music(pid: str, body: MusicSettings):
    d = pdir(pid)
    if pid in PLANNING:
        raise HTTPException(409, "the AI is generating the plan - wait for it to finish")
    st = load_state(d)
    if not st.get("music"):
        raise HTTPException(404, "there is no music")
    m = st["music"]
    body.start = min(max(0.0, body.start), max(0.0, m["duration"] - 1))
    body.volume = min(max(0.0, body.volume), 2.0)
    body.fade_in, body.fade_out = min(max(0.0, body.fade_in), 5.0), min(max(0.0, body.fade_out), 5.0)
    m.update(body.model_dump())
    resnapped = False
    if body.beat_sync and st["edl"]:
        media = scan_media(d / "input")
        edl = clamp_transitions(snap_to_beats(EDL.model_validate(st["edl"]), media,
                                              beat_grid(music_ctx(d, st)["info"], body.start)))
        text_errors(edl)
        st["edl"] = edl.model_dump(by_alias=True, exclude_none=True)
        resnapped = True
    save_state(d, st)
    return {"music": m, "resnapped": resnapped}


@app.delete("/api/projects/{pid}/music")
def delete_music(pid: str):
    d = pdir(pid)
    for old in (d / "music").glob("*"):
        old.unlink()
    st = load_state(d)
    st["music"] = None
    save_state(d, st)
    return {"music": None}


@app.get("/api/projects/{pid}/music/file")
def get_music_file(pid: str):
    d = pdir(pid)
    st = load_state(d)
    if not st.get("music"):
        raise HTTPException(404)
    return FileResponse(d / "music" / st["music"]["file"])


# ---------- ambience (a second, simpler background bed - no beat analysis, just volume/start/fades) ----------
class AmbienceSettings(BaseModel):
    start: float = 0.0
    volume: float = 0.5
    fade_in: float = 1.0
    fade_out: float = 1.5
    duck: bool = True


@app.post("/api/projects/{pid}/ambience")
async def upload_ambience(pid: str, file: UploadFile = File(...)):
    d = pdir(pid)
    name = Path(file.filename).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in AUDIO_EXT:
        raise HTTPException(400, f"audio format not allowed (accepted: {', '.join(sorted(AUDIO_EXT))})")
    (d / "ambience").mkdir(exist_ok=True)
    for old in (d / "ambience").glob("*"):
        old.unlink()
    dst = d / "ambience" / name
    with dst.open("wb") as out:
        shutil.copyfileobj(file.file, out, 1024 * 1024)
    try:
        duration = audio_duration(dst)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"can't read this file: {e}")
    st = load_state(d)
    st["ambience"] = {"file": name, "duration": duration, **AmbienceSettings().model_dump()}
    save_state(d, st)
    return {"ambience": st["ambience"]}


@app.put("/api/projects/{pid}/ambience")
def put_ambience(pid: str, body: AmbienceSettings):
    d = pdir(pid)
    st = load_state(d)
    if not st.get("ambience"):
        raise HTTPException(404, "there is no ambience track")
    a = st["ambience"]
    body.start = min(max(0.0, body.start), max(0.0, a["duration"] - 1))
    body.volume = min(max(0.0, body.volume), 2.0)
    body.fade_in, body.fade_out = min(max(0.0, body.fade_in), 5.0), min(max(0.0, body.fade_out), 5.0)
    a.update(body.model_dump())
    save_state(d, st)
    return {"ambience": a}


@app.delete("/api/projects/{pid}/ambience")
def delete_ambience(pid: str):
    d = pdir(pid)
    for old in (d / "ambience").glob("*"):
        old.unlink()
    st = load_state(d)
    st["ambience"] = None
    save_state(d, st)
    return {"ambience": None}


@app.get("/api/projects/{pid}/ambience/file")
def get_ambience_file(pid: str):
    d = pdir(pid)
    st = load_state(d)
    if not st.get("ambience"):
        raise HTTPException(404)
    return FileResponse(d / "ambience" / st["ambience"]["file"])


# ---------- sfx library ----------
@app.get("/api/sfx")
def list_sfx():
    return [{"key": k, "label": v["label"]} for k, v in SFX_LIBRARY.items()]


@app.get("/api/sfx/{key}/file")
def get_sfx_file(key: str):
    v = SFX_LIBRARY.get(key)
    if not v:
        raise HTTPException(404)
    return FileResponse(SFX_DIR / v["file"], media_type="audio/mpeg")


@app.delete("/api/projects/{pid}/media/{name}")
def delete_media(pid: str, name: str):
    d = pdir(pid)
    name = safe_name(name)
    (d / "input" / name).unlink(missing_ok=True)
    (d / "thumbs" / (Path(name).stem + ".jpg")).unlink(missing_ok=True)
    (d / "waveforms" / (Path(name).stem + ".json")).unlink(missing_ok=True)
    shutil.rmtree(d / "filmstrips" / Path(name).stem, ignore_errors=True)
    proxy_path(d, name).unlink(missing_ok=True)
    proxy_path(d, name).with_suffix(".tmp.mp4").unlink(missing_ok=True)
    st = load_state(d)
    st["hints"].pop(name, None)
    save_state(d, st)
    return {"deleted": name}


@app.get("/api/projects/{pid}/thumb/{name}")
def get_thumb(pid: str, name: str):
    f = pdir(pid) / "thumbs" / (Path(safe_name(name)).stem + ".jpg")
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/projects/{pid}/waveform/{name}")
def get_waveform(pid: str, name: str):
    """Cached per-file RMS envelope for the timeline. Computed lazily on first request (not at
    upload) and cached to disk after - works retroactively for files uploaded before this existed."""
    d = pdir(pid)
    name = safe_name(name)
    f = d / "waveforms" / (Path(name).stem + ".json")
    if not f.exists():
        media = scan_media(d / "input")
        info = media.get(name)
        if not info or info.kind != "video" or not info.has_audio:
            raise HTTPException(404, "no audio to show a waveform for")
        f.parent.mkdir(exist_ok=True)
        try:
            wf = compute_waveform(preview_info(info, d).path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"could not analyze audio: {e}")
        f.write_text(json.dumps(wf), encoding="utf-8")
    return FileResponse(f, media_type="application/json")


# ---------- filmstrip thumbnails (several frames per source file, spanning its whole duration) ----------
FILMSTRIP_N = 10     # frames per video file - enough that even a short clip's in/out usually covers one
FILMSTRIP_WIDTH = 96  # small: these tile across a ~34px-tall timeline segment, not a full preview


@app.get("/api/projects/{pid}/filmstrip/{name}")
def get_filmstrip(pid: str, name: str):
    """Cached per-FILE (not per-clip-range) evenly-spaced frames for the timeline filmstrip -
    computed lazily on first request and cached to disk after, same pattern as the waveform
    endpoint above. One shared set of frames per source file lets every clip that uses that file
    (at whatever in/out) reuse it instead of paying for its own extraction. Video only - a photo's
    single timeline thumbnail already shows the whole picture, there's nothing to scrub through."""
    d = pdir(pid)
    name = safe_name(name)
    fs_dir = d / "filmstrips" / Path(name).stem
    manifest = fs_dir / "manifest.json"
    if not manifest.exists():
        media = scan_media(d / "input")
        info = media.get(name)
        if not info or info.kind != "video":
            raise HTTPException(404, "filmstrips are for video clips only")
        fs_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames = extract_frames(preview_info(info, d), fs_dir, n=FILMSTRIP_N, width=FILMSTRIP_WIDTH)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"could not extract frames: {e}")
        manifest.write_text(json.dumps({"n": len(frames), "times": [f.timestamp for f in frames]}), "utf-8")
    return json.loads(manifest.read_text("utf-8"))


@app.get("/api/projects/{pid}/filmstrip/{name}/{i}")
def get_filmstrip_frame(pid: str, name: str, i: int):
    d = pdir(pid)
    name = safe_name(name)
    f = d / "filmstrips" / Path(name).stem / f"{Path(name).stem}_{i}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/projects/{pid}/media/{name}")
def get_media(pid: str, name: str):
    f = pdir(pid) / "input" / safe_name(name)
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/projects/{pid}/output/{kind}")
def get_output(pid: str, kind: str):
    if kind == "preview":
        f = pdir(pid) / "output" / "preview.mp4"
    elif kind in ("draft", "final", "draft_no_music", "final_no_music"):
        f = pdir(pid) / "output" / ("final" if kind.startswith("final") else "drafts") / f"{kind}.mp4"
    else:
        raise HTTPException(400)
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="video/mp4")


# ---------- settings ----------
class Settings(BaseModel):
    preset: str = "neutral"
    overrides: dict = {}
    subtitles: bool | None = None


@app.put("/api/projects/{pid}/settings")
def put_settings(pid: str, body: Settings):
    if body.preset not in PRESETS:
        raise HTTPException(400, "unknown preset")
    if body.overrides:
        try:
            clean = {k: v for k, v in body.overrides.items() if v not in (None, "")}
            merged = PRESETS[body.preset].model_copy(update=clean)
            Preset.model_validate(merged.model_dump())  # model_copy skips validation - check it explicitly
        except Exception as e:  # noqa: BLE001 - surface the bad field now instead of failing later at plan time
            raise HTTPException(400, f"invalid override: {e}")
    d = pdir(pid)
    st = load_state(d)
    st["preset"], st["overrides"] = body.preset, body.overrides
    if body.subtitles is not None:
        st["subtitles"] = body.subtitles
    save_state(d, st)
    return {"ok": True}


@app.put("/api/projects/{pid}/hints/{name}")
def put_hint(pid: str, name: str, hint: FileHint):
    d = pdir(pid)
    name = safe_name(name)
    st = load_state(d)
    if hint.is_default():
        st["hints"].pop(name, None)
    else:
        st["hints"][name] = hint.model_dump()
    save_state(d, st)
    return {"ok": True}


class LockRulesIn(BaseModel):
    rules: list[LockRule] = Field(default_factory=list, max_length=20)


@app.put("/api/projects/{pid}/lockrules")
def put_lock_rules(pid: str, body: LockRulesIn):
    d = pdir(pid)
    if pid in PLANNING:
        raise HTTPException(409, "the AI is generating the plan - try again once it finishes")
    st = load_state(d)
    st["lock_rules"] = [r.model_dump() for r in body.rules]
    locked_clips = None
    if st["edl"]:
        media = scan_media(d / "input")
        edl = validate_edl(st["edl"], media)
        edl = apply_lock_rules(edl, body.rules)
        locked_clips = sum(1 for c in edl.clips if c.locked)
        st["edl"] = edl.model_dump(by_alias=True, exclude_none=True)
    save_state(d, st)
    return {"ok": True, "lock_rules": st["lock_rules"], "locked_clips": locked_clips}


@app.put("/api/projects/{pid}/edl")
def put_edl(pid: str, raw: dict):
    d = pdir(pid)
    if pid in PLANNING:
        raise HTTPException(409, "the AI is generating the plan - editing is locked until it finishes")
    try:
        edl = validate_edl(raw, scan_media(d / "input"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    st = load_state(d)
    st["edl"] = edl.model_dump(by_alias=True, exclude_none=True)
    st["is_demo"] = None
    save_state(d, st)
    return {"ok": True, "total": total_duration(edl)}


# ---------- jobs ----------
def _plan_job(d: Path) -> dict:
    PLANNING.add(d.name)
    try:
        return _plan(d)
    finally:
        PLANNING.discard(d.name)


def _plan(d: Path) -> dict:
    from .planner import plan
    st = load_state(d)
    media = scan_media(d / "input")
    if not media:
        raise RuntimeError("You haven't uploaded any files")
    hints = {k: FileHint.model_validate(v) for k, v in st["hints"].items() if k in media}
    current = EDL.model_validate(st["edl"]) if st["edl"] else None
    if current and st.get("lock_rules"):
        current = apply_lock_rules(current, [LockRule.model_validate(r) for r in st["lock_rules"]])
    if current and not any(c.locked for c in current.clips) and not any(t.locked for t in current.texts):
        current = None
    trs = load_transcripts(d, media)   # the AI sees the transcript even if subtitles are turned off
    r = plan(preview_media(media, d), st["preset"], st["overrides"], example_path(d), hints, current,
             music_ctx(d, st), trs)
    st["edl"] = r["edl"].model_dump(by_alias=True, exclude_none=True)
    st["reasoning"], st["usage"] = r["reasoning"], r["usage"]
    st["is_demo"] = None
    save_state(d, st)
    return {"reasoning": r["reasoning"], "usage": r["usage"], "total": total_duration(r["edl"])}


def _plan_variants_job(d: Path, n: int) -> dict:
    PLANNING.add(d.name)
    try:
        from .planner import plan_variants
        st = load_state(d)
        media = scan_media(d / "input")
        if not media:
            raise RuntimeError("You haven't uploaded any files")
        hints = {k: FileHint.model_validate(v) for k, v in st["hints"].items() if k in media}
        current = EDL.model_validate(st["edl"]) if st["edl"] else None
        if current and st.get("lock_rules"):
            current = apply_lock_rules(current, [LockRule.model_validate(r) for r in st["lock_rules"]])
        if current and not any(c.locked for c in current.clips) and not any(t.locked for t in current.texts):
            current = None
        trs = load_transcripts(d, media)
        results = plan_variants(preview_media(media, d), st["preset"], st["overrides"], example_path(d), hints,
                                current, music_ctx(d, st), trs, n=n)
        out = []
        for r in results:
            if "edl" not in r:
                out.append({"label": r["label"], "error": r["error"]})
                continue
            out.append({"label": r["label"], "reasoning": r["reasoning"], "usage": r["usage"],
                        "total": total_duration(r["edl"]), "clips": len(r["edl"].clips),
                        "edl": r["edl"].model_dump(by_alias=True, exclude_none=True)})
        return {"variants": out}
    finally:
        PLANNING.discard(d.name)


def _deepscan_job(d: Path, idx: int) -> dict:
    PLANNING.add(d.name)
    try:
        from .planner import deep_scan_clip
        import tempfile
        st = load_state(d)
        if not st["edl"]:
            raise RuntimeError("There is no plan yet")
        media = scan_media(d / "input")
        edl = validate_edl(st["edl"], media)
        if not (0 <= idx < len(edl.clips)):
            raise RuntimeError("clip index out of range")
        with tempfile.TemporaryDirectory(prefix="vedit_deepscan_") as tmp:
            updated, usage = deep_scan_clip(preview_media(media, d), edl.clips[idx], Path(tmp))
        new_clips = list(edl.clips)
        new_clips[idx] = updated
        new_edl = edl.model_copy(update={"clips": new_clips})
        errs = overlap_errors(new_edl)
        if errs:
            raise RuntimeError("The refined cut would overlap another clip using the same file - "
                                "adjust it manually instead:\n  " + "\n  ".join(errs))
        st["edl"] = new_edl.model_dump(by_alias=True, exclude_none=True)
        st["is_demo"] = None
        save_state(d, st)
        return {"clip": updated.model_dump(by_alias=True, exclude_none=True), "index": idx,
                "usage": usage, "total": total_duration(new_edl)}
    finally:
        PLANNING.discard(d.name)


def _render_job(d: Path, final: bool, mute: bool = False) -> dict:
    """mute=True: export without music, but with cuts aligned to its beat
    (for TikTok/Reels, where you add the licensed track directly in the app)."""
    st = load_state(d)
    if not st["edl"]:
        raise RuntimeError("There is no plan yet")
    media = scan_media(d / "input")
    edl = validate_edl(st["edl"], media)
    kind = ("final" if final else "draft") + ("_no_music" if mute else "")
    dst = d / "output" / ("final" if final else "drafts") / f"{kind}.mp4"
    t0 = time.time()
    render(edl, media, dst, draft=not final, music=None if mute else music_ctx(d, st),
           ambience=ambience_ctx(d, st), sfx=sfx_ctx(edl), subtitles=subtitles_for(d, st, edl, media))
    return {"url": f"/api/projects/{d.name}/output/{kind}?t={int(time.time())}",
            "file": f"{kind}.mp4", "seconds": round(time.time() - t0, 1)}


def _preview_job(d: Path) -> dict:
    """A2/A1's quick preview: same EDL, but proxy media + a cross-call segment cache (render_preview)
    so only clips that actually changed since the last preview get re-encoded. Half-resolution,
    draft quality, always - this is for iterating fast, not for judging final quality (use Render
    draft/final for that)."""
    st = load_state(d)
    if not st["edl"]:
        raise RuntimeError("There is no plan yet")
    media = scan_media(d / "input")
    edl = validate_edl(st["edl"], media)
    dst = d / "output" / "preview.mp4"
    t0 = time.time()
    render_preview(edl, preview_media(media, d), dst, d, music=music_ctx(d, st),
                    ambience=ambience_ctx(d, st), sfx=sfx_ctx(edl), subtitles=subtitles_for(d, st, edl, media))
    return {"url": f"/api/projects/{d.name}/output/preview?t={int(time.time())}",
            "seconds": round(time.time() - t0, 1)}


def _transcribe_job(d: Path) -> dict:
    ok, msg = tr_mod.available()
    if not ok:
        raise RuntimeError(msg)
    media = scan_media(d / "input")
    done = []
    for name, m in media.items():
        if m.kind == "video" and m.has_audio:
            tr_mod.transcribe(m.path, d / "transcripts" / f"{name}.json")
            done.append(name)
    if not done:
        raise RuntimeError("No clip has sound - nothing to transcribe")
    return {"files": done}


@app.post("/api/projects/{pid}/transcribe")
def start_transcribe(pid: str):
    return {"job": run_job(_transcribe_job, pdir(pid))}


@app.delete("/api/projects/{pid}/transcripts/{name}")
def delete_transcript(pid: str, name: str):
    (pdir(pid) / "transcripts" / f"{safe_name(name)}.json").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/projects/{pid}/export/fcpxml")
def export_fcpxml(pid: str):
    """Editable timeline for DaVinci Resolve (File > Import > Timeline) with the ORIGINAL files."""
    d = pdir(pid)
    st = load_state(d)
    if not st["edl"]:
        raise HTTPException(400, "There is no plan yet")
    media = scan_media(d / "input")
    edl = validate_edl(st["edl"], media)
    mc = music_ctx(d, st)
    xml = fcpxml.build(edl, media, f"Project {pid}", mc, subtitles_for(d, st, edl, media))
    dst = d / "output" / f"project_{pid}.fcpxml"
    dst.write_bytes(xml)
    return FileResponse(dst, media_type="application/xml", filename=dst.name)


@app.post("/api/projects/{pid}/plan")
def start_plan(pid: str):
    if pid in PLANNING:
        raise HTTPException(409, "A plan is already in progress")
    return {"job": run_job(_plan_job, pdir(pid))}


@app.post("/api/projects/{pid}/plan/variants")
def start_plan_variants(pid: str, n: int = 3):
    if pid in PLANNING:
        raise HTTPException(409, "A plan is already in progress")
    return {"job": run_job(_plan_variants_job, pdir(pid), max(1, min(n, 3)))}


class SelectVariantIn(BaseModel):
    edl: dict
    reasoning: str = ""
    usage: dict | None = None


@app.post("/api/projects/{pid}/plan/select")
def select_plan_variant(pid: str, body: SelectVariantIn):
    d = pdir(pid)
    if pid in PLANNING:
        raise HTTPException(409, "the AI is generating the plan - try again once it finishes")
    try:
        edl = validate_edl(body.edl, scan_media(d / "input"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, str(e))
    st = load_state(d)
    st["edl"] = edl.model_dump(by_alias=True, exclude_none=True)
    st["reasoning"], st["usage"] = body.reasoning, body.usage
    st["is_demo"] = None
    save_state(d, st)
    return {"ok": True, "total": total_duration(edl)}


@app.post("/api/projects/{pid}/deepscan/{idx}")
def start_deepscan(pid: str, idx: int):
    if pid in PLANNING:
        raise HTTPException(409, "A plan is already in progress")
    return {"job": run_job(_deepscan_job, pdir(pid), idx)}


@app.post("/api/projects/{pid}/render")
def start_render(pid: str, final: bool = False, mute: bool = False):
    return {"job": run_job(_render_job, pdir(pid), final, mute)}


@app.post("/api/projects/{pid}/preview")
def start_preview(pid: str):
    return {"job": run_job(_preview_job, pdir(pid))}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return {**j, "elapsed": round(time.time() - j["started"], 1)}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
