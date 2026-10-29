"""Local FastAPI server: projects, upload, per-file settings, AI plan, EDL editing, rendering.

Start:  uvicorn app.main:app --port 8000   ->  http://localhost:8000
Long operations (AI plan, render) run in the background; the frontend polls /api/jobs/{id}.
"""
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .edl import EDL, clamp_transitions, text_errors, total_duration, validate_edl
from .frames import thumbnail
from .hints import FileHint
from . import fcpxml
from . import subtitles as subs_mod
from . import transcribe as tr_mod
from .music import AUDIO_EXT, MusicInfo, analyze, beat_grid, snap_to_beats
from .presets import COLOR_BASE, PACE_RANGE, PRESETS
from .probe import IMAGE_EXT
from .render import VIDEO_EXT, render, scan_media

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
        "music": None}


def save_state(d: Path, st: dict) -> None:
    (d / "state.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), "utf-8")


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
            "duck": m.get("duck", True)}


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


def run_job(fn, *args) -> str:
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
            "pace_range": PACE_RANGE, "color_base": COLOR_BASE}


# ---------- projects ----------
@app.post("/api/projects")
def create_project():
    pid = uuid.uuid4().hex[:8]
    d = PROJECTS / pid
    for sub in ("input", "example", "music", "thumbs", "output/drafts", "output/final"):
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
    for n in saved:
        thumbnail(media[n], d / "thumbs" / (Path(n).stem + ".jpg"))
    return {"saved": saved}


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


@app.delete("/api/projects/{pid}/media/{name}")
def delete_media(pid: str, name: str):
    d = pdir(pid)
    name = safe_name(name)
    (d / "input" / name).unlink(missing_ok=True)
    (d / "thumbs" / (Path(name).stem + ".jpg")).unlink(missing_ok=True)
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


@app.get("/api/projects/{pid}/media/{name}")
def get_media(pid: str, name: str):
    f = pdir(pid) / "input" / safe_name(name)
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/projects/{pid}/output/{kind}")
def get_output(pid: str, kind: str):
    if kind not in ("draft", "final", "draft_no_music", "final_no_music"):
        raise HTTPException(400)
    f = pdir(pid) / "output" / ("final" if kind.startswith("final") else "drafts") / f"{kind}.mp4"
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
    if current and not any(c.locked for c in current.clips) and not any(t.locked for t in current.texts):
        current = None
    trs = load_transcripts(d, media)   # the AI sees the transcript even if subtitles are turned off
    r = plan(media, st["preset"], st["overrides"], example_path(d), hints, current, music_ctx(d, st), trs)
    st["edl"] = r["edl"].model_dump(by_alias=True, exclude_none=True)
    st["reasoning"], st["usage"] = r["reasoning"], r["usage"]
    save_state(d, st)
    return {"reasoning": r["reasoning"], "usage": r["usage"], "total": total_duration(r["edl"])}


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
    render(edl, media, dst, draft=not final, audio=None if mute else music_ctx(d, st),
           subtitles=subtitles_for(d, st, edl, media))
    return {"url": f"/api/projects/{d.name}/output/{kind}?t={int(time.time())}",
            "file": f"{kind}.mp4", "seconds": round(time.time() - t0, 1)}


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


@app.post("/api/projects/{pid}/render")
def start_render(pid: str, final: bool = False, mute: bool = False):
    return {"job": run_job(_render_job, pdir(pid), final, mute)}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return {**j, "elapsed": round(time.time() - j["started"], 1)}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
