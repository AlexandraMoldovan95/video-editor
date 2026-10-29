<h1 align="center">Video Editor</h1>

<p align="center">
  <b>A local web app that turns raw footage — clips, photos, a song — into a vertical 9:16 video for social media.</b><br>
  Claude (the AI) chooses the order, the cuts, the color, the transitions and the on-screen text. You can adjust anything and <b>lock</b> what you like.
</p>

<p align="center">
  <i>Python + FastAPI + ffmpeg + Claude API (<code>claude-sonnet-5-5</code>)</i>
</p>

<!-- ![Video Editor — upload, AI plan and timeline, preview & export](docs/media/hero.png) -->
> 📸 *Screenshot pending — add one of the three-column UI (Material / Plan / Preview) before publishing.*

| | |
|---|---|
| **What** | Local Python web app (FastAPI backend, single-file vanilla-JS frontend) · ffmpeg for rendering · Claude API for the editing decisions · Windows |
| **My role** | Product idea, UX, the "AI decides, code executes" architecture, and all manual testing. The code was written in pair-programming with an AI assistant (Claude): see [How it was built](#how-it-was-built). |
| **Quality** | Manually verified end to end: render pipeline (hand-written EDL → correct MP4), the AI planner with a real preset, and the FastAPI server + frontend (Playwright). The manual editing tools were verified against synthetic ffmpeg test media and, for the full combined feature set, a real end-to-end render through the live FastAPI app against the project's own real sample files. No automated test suite yet. |
| **Status** | Personal/portfolio project, all planned features implemented. Honest open items below — Whisper transcription and the DaVinci Resolve import still need to be tried on a normal Windows network/install. |

## Why it exists

Editing a 30-second reel by hand — picking the best seconds out of raw clips, cutting them to a beat, adding text — easily takes an hour. This app gets a solid first draft in about two minutes: you drop in your clips and photos, pick a style, and an AI plans the edit. You stay in control — adjust anything, **lock 🔒** the cuts and texts you like, and regenerate only the rest.

## Highlights

- **"AI decides, code executes."** Claude never touches ffmpeg. It returns a JSON edit plan that is validated against a strict schema (Pydantic) before any video command runs — no command injection, reproducible output, and errors that are easy to debug and retry.
- **Per-file control the AI can't override**: mark a clip required or excluded, lock its usable time range, fix its color or volume, add a note — the code enforces these as hard constraints, the AI only fills in the rest.
- **Lock and regenerate**: lock the cuts and texts that already work, hit Regenerate, and the AI only reworks what's left — your edits are never silently lost.
- **Music-aware editing**: custom beat detection written in plain numpy (no librosa) finds the BPM and beat grid; cuts can snap to the beat, with ducking so the original clip audio stays audible under the music.
- **Two export paths**: a finished MP4 (draft or 1080×1920 final), or an FCPXML timeline that reopens the *original* files in DaVinci Resolve/Final Cut for a full manual re-edit.
- **Optional subtitles**: Whisper transcribes speech and subtitles are recalculated live from the current timeline, so they never go out of sync with your edits.
- **Try a demo, no API key needed**: three pre-built example plans (`app/demo_presets.py`) load instantly over the same sample clips and show three very different edits — fast/contrasty, slow/warm, cinematic travel. Two are hand-built to match a preset's rules; the third replays a real saved Claude run. None of the three calls the API, so anyone can see the mechanism work without spending anything.
- **A full manual editing pass once the AI draft lands** — none of it calls the API, all of it is plain timeline editing: undo/redo and NLE-style keyboard shortcuts (Space/J/K/L/I/O/S/Del/Ctrl+Z); drag-trim a clip's in/out directly on the timeline bar; a split (razor) tool at the preview's playhead; per-clip speed (0.25×–4×, with matching audio time-stretch); manual reframe/pan on top of the auto-crop; a per-clip waveform drawn on the timeline from the source audio; shift/ctrl-click multi-select with bulk lock/delete/color-grade; and free-drag text placement instead of only top/center/bottom.

## Engineering in short

- **The AI never writes ffmpeg.** `planner.py` calls Claude with a `submit_edl` tool; the result is validated by `edl.py` (overlaps, transition rules, text placement) before `render.py` ever runs. An invalid plan is rejected and Claude gets one retry with the validation error.
- **Beat tracking from scratch**: `music.py` computes an onset envelope and runs dynamic-programming beat tracking (Ellis 2007-style) in numpy only — no external beat-detection library.
- **Subtitles are never stored, only recomputed**: `subtitles.py` derives subtitle timing from the transcript and the *current* cut list on every render, so moving a clip can never desync a caption.
- **Hard constraints vs. AI suggestions**: `hints.py` keeps the two cleanly separate — what you fix on a file is enforced by code regardless of what the AI proposes.
- **ffmpeg orchestration**: `render.py` normalizes every clip to a common format, applies `xfade`/concat for transitions and `drawtext` for on-screen text, then mixes original audio with music and ducking.
- **FCPXML export**: `fcpxml.py` builds a DaVinci/Final Cut timeline that references the *original* source files (not the rendered MP4), so color, cuts and framing stay editable — validated against OpenTimelineIO for frame-accurate timing.

| Module | Role |
|---|---|
| `app/probe.py` | ffprobe metadata |
| `app/edl.py` | the plan's schema + validation (overlaps, transitions, text) |
| `app/hints.py` | per-file settings: constraints (checked) and overrides (applied by code) |
| `app/presets.py` | presets + the rules handed to the AI |
| `app/frames.py` | frames for the AI, thumbnails, example-video pacing |
| `app/music.py` | beat tracking in plain numpy, energy curve, beat alignment |
| `app/planner.py` | the prompt + Claude call + validation retry + re-inserting locked items |
| `app/render.py` | segments → transitions/text → audio → MP4 |
| `app/subtitles.py` / `app/transcribe.py` | Whisper → timeline-synced subtitles |
| `app/fcpxml.py` | DaVinci Resolve / Final Cut export |
| `app/main.py` | FastAPI server + background jobs |
| `static/index.html` | the whole UI (one file) |

## Getting started

Needs Python **3.10+**, **ffmpeg** on PATH, and an Anthropic API key:

```
py --version
ffmpeg -version                         # if missing: winget install Gyan.FFmpeg
py -m pip install -r requirements.txt
```

1. Copy `.env.example` to **`.env`** and set `ANTHROPIC_API_KEY=sk-ant-...` (never put the real key in `.env.example` — that file is the one tracked in git).
2. Double-click `start.bat`. It opens `http://localhost:8000`. Keep the window open — closing it stops the app. Stop with `Ctrl+C`.

Optional, for automatic subtitles (Whisper):
```
py -m pip install -r requirements-whisper.txt
```
The model downloads on first use (~460 MB, `small`). A smaller model is available via `VEDIT_WHISPER_MODEL=base`, at some accuracy cost.

**To try the 3 demo plans without an API key**, upload a project with these exact Pexels sample filenames (any free Pexels clips/photos renamed to match work — see `app/demo_presets.py` for the full list): `16928481_1080_1920_50fps.mp4`, `16581829_1440_2560_60fps.mp4`, `15940078_2160_3840_30fps.mp4`, `16615054_2160_3840_25fps.mp4`, `16931601_1080_1920_120fps.mp4`, `17198558_3840_2160_30fps.mp4`, `12772832_1080_1920_30fps.mp4`, plus 3 photos (`pexels-ayse-ela-dikye-49914919-16268987.jpg`, `pexels-hayriyenur-351892318-39457179.jpg`, `pexels-santesson89-37765467.jpg`). The demo buttons on a project missing any of those stay disabled with a tooltip naming what's missing.

## Known limitations (honest)

- **The AI only sees 4 frames per clip.** It picks well on calm footage, but can miss the exact right moment in a clip with a lot of action. Workaround: set an allowed range and a note for the AI on that file.
- **Beat tracking is tested on synthetic tracks** (128 and 95 BPM, ~22ms error). On music without a clear drum hit it can get it wrong — turn off "On beat" for that track.
- **Whisper and the DaVinci import are still untested on a normal network/install.** Whisper's model download is blocked in the sandboxed environment this was developed in; the subtitle *logic* was verified with a simulated transcript, and the FCPXML timing was validated against OpenTimelineIO, but neither has been exercised for real yet.
- **No YouTube support, on purpose** — downloading from it breaks YouTube's terms, and licensed music gets muted or claimed on most platforms anyway. For TikTok/Reels, export without music and add the same track from inside the app.
- Uploaded files are copied into the project folder, so a project's disk footprint adds up.
- **The FCPXML export doesn't carry over three of the manual tools**: a sped-up/slowed-down clip, a manually panned crop, and a freely-dragged text all land in DaVinci/Final Cut at their old, un-adjusted spot — redo those three by hand there if you want to keep the look. Everything else (cuts, color look's base values, transitions-as-dissolve, music) does transfer.
- **A clip's waveform only shows up if the clip has its own audio.** Silent b-roll (most stock footage, including this project's own Pexels demo sample clips) has nothing to draw — that's expected, not a bug.

## API cost

One plan generation costs roughly 18–22k input tokens and 2–3k output tokens for 10 files (~31 frames) — a few cents per run. Excluded clips are never sent to the AI.

## How it was built

I own the product: I picked the idea (I kept hand-editing the same kind of 30-second clip and wanted the first draft automated), decided the "AI decides, code executes" boundary, and tested every stage by hand against real footage — a hand-written EDL through the render pipeline, the AI planner end to end with a real preset, and the FastAPI server with the frontend via Playwright.

The code was written in pair-programming with Claude (Anthropic's AI assistant), starting from a written requirements doc and working through the pipeline stage by stage: metadata and plain-ffmpeg rendering first, then frame sampling and the AI planner, then the FastAPI server and UI, then music/beat sync, subtitles and the DaVinci export.
