# CLAUDE.md — Agent notes for this project

> This file is the working context an AI coding agent (Claude) reads before touching this codebase: what the app does, the architecture rules that must not be broken, and the decisions already made so they don't get re-litigated every session. It doubles as the deepest technical write-up of the project, for anyone who wants more detail than the README.

## 1. What the app does

The user supplies raw material — video clips, photos, optionally a song and an example edited video — plus a style preset and notes. The app produces a short video for social media, vertical (9:16) by default, or square (1:1) or horizontal (16:9) per project: it orders the clips, cuts them, applies color correction, and adds on-screen text and visual elements where it helps. An AI (Claude) proposes the edit; the user can adjust or lock any part of it and regenerate only what's left.

**Who it's for:** anyone editing short clips for social media who wants a fast first draft instead of starting from a blank timeline.

## 2. Pipeline

1. **Input** — video, photos, audio.
2. **ffprobe** — reads metadata (duration, resolution, fps, has-audio).
3. **Frame extraction** — 4–5 frames per clip, with timestamps.
4. **AI** (Claude API, `claude-sonnet-5-5`) — receives the frames + metadata + rules → returns an edit plan.
5. **Validation** — the plan is checked before anything is rendered.
6. **Rendering** with ffmpeg.
7. **Output** — MP4 (H.264 + AAC), 30 fps by default, at the project's format: 1080×1920 (9:16, the default), 1080×1080 (1:1) or 1920×1080 (16:9). The project stores its `format`; `with_format` sets every new plan (AI, variant, demo) to it, and changing it re-sizes the current plan. Rendering and FCPXML read `edl.output`; the planner is told the format and which clip shape gets a plain centre crop.

## 3. Architecture rules (non-negotiable)

- **The AI never writes ffmpeg commands.** It returns a validated JSON Edit Decision List (EDL); the code translates JSON → ffmpeg. This buys safety (no command injection), reproducibility, and a debuggable, retryable failure mode.
- **Normalize before anything else**: every clip is brought to the same resolution/fps/format before concatenation.
- **The API key lives only on the server** (`.env`, never in the frontend or in git).
- **ffmpeg runs locally/on the server**, never in the browser (the files are too large for ffmpeg.wasm).
- **Cheap previews**: drafts render at half resolution with `-preset veryfast`; full-resolution rendering only happens once the user asks for it.
- **A corrupt or unreadable file must never take down the whole project.** Every route that lists media handles a bad file by skipping it, not by crashing (see §10).

## 4. EDL schema (core shape)

```json
{
  "output": { "width": 1080, "height": 1920, "fps": 30 },
  "clips": [
    { "source": "clip1.mp4", "in": 2.0, "out": 5.5,
      "fit": "crop_center", "brightness": 0.05, "contrast": 1.1, "saturation": 1.15 },
    { "source": "photo1.jpg", "duration": 2.5, "motion": "zoom_in",
      "fit": "blur_pad", "brightness": 0.0, "contrast": 1.0, "saturation": 1.0 }
  ],
  "texts": [],
  "sfx": [],
  "overlays": []
}
```

Every field is a Pydantic model with hard-coded safe ranges (`ge`/`le`), and `source` is validated to be a bare filename inside the project's own input folder — the AI (or a crafted request) can't reference an arbitrary path on disk.

## 5. Stack

- **Backend**: Python 3 + FastAPI (uvicorn), served locally at `http://localhost:8000`.
- **Processing**: `ffmpeg` / `ffprobe` called via `subprocess.run([...])` with a list of arguments — never `shell=True`.
- **AI**: the official `anthropic` SDK, model `claude-sonnet-5-5`, key read from `.env` (`ANTHROPIC_API_KEY`) via `python-dotenv`.
- **EDL validation**: Pydantic models (safe value ranges, files restricted to the current project's folder, `in < out`, durations within the clip's own length).
- **Frontend**: a single-page HTML/CSS/JS file served by FastAPI — upload, presets, per-file settings, timeline, preview `<video>`, render buttons.
- **Dependencies**: `requirements.txt` (fastapi, uvicorn, python-multipart, anthropic, python-dotenv, pydantic); `requirements-whisper.txt` is optional, for subtitles.

### Project layout

```
app/
  main.py          # FastAPI routes + background jobs
  probe.py         # ffprobe -> metadata
  frames.py        # frame extraction + scene detection on an example video
  planner.py       # prompt + Claude call -> EDL JSON; also deep-scan and multi-variant generation
  edl.py            # Pydantic models + validation
  render.py         # EDL -> ffmpeg commands (normalize, crop/pad, color, overlays, beat punches, audio, concat)
  lockrules.py       # standing lock rules re-applied before every regeneration
  proxy.py           # background low-res proxies for previews/AI frames (render.py never touches these)
  music.py           # beat tracking (plain numpy), energy curve, beat alignment
  subtitles.py / transcribe.py   # Whisper -> timeline-synced subtitles
  fcpxml.py          # DaVinci Resolve / Final Cut export
  presets.py         # style presets (fixed fields handed to the AI)
static/index.html
projects/<id>/       # input/, example/, thumbs/, proxies/, previews/segcache/, output/, state.json
```

### Running it

```
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # -> http://localhost:8000
python -m app.render edl_examples/test_manual.json            # draft, half resolution
python -m app.render edl_examples/test_manual.json --final    # full 1080x1920
python -m app.planner --preset travel_cinematic --dry-run     # see the prompt, no API call
python -m pytest -q                                           # plan validation (tests/test_edl.py) + AI request loop with a fake client (tests/test_planner_loop.py) + missing-key check on the AI routes (tests/test_api_key.py) + project names and output formats (tests/test_projects.py); CI runs them on every push
```

**The server runs without `--reload`.** Any change to backend code (`app/*.py`) needs a manual restart to take effect.

## 6. How the planner works

- 4 frames per clip (midpoint of each quarter), 512px wide, JPEG; photos send 1 frame. An example video also sends 4 frames, for visual style.
- Claude answers through a `submit_edl` tool with `tool_choice: "auto"` (the model used here doesn't accept a forced `tool_choice`). If it answers with plain text instead, the request is retried once.
- The response goes through `validate_edl`; on failure, the validation error is sent back once for a correction (`correction_turn`). Every `tool_use` in that reply gets its own `tool_result`, not only the one that was checked: the API rejects a turn that leaves a tool call unanswered.
- The client's timeout and retries are set explicitly in `make_client()`: up to 10 minutes for a reply (a plan with thinking arrives in one piece and can take minutes), 10 seconds to open the connection, 2 SDK retries on connection errors, 408/409/429 and 5xx.
- A reply cut off at the token limit, or refused, is reported as exactly that (`check_stop`), instead of being validated as a "wrong plan" and using up the one correction. The model thinks by default and thinking counts toward `max_tokens`, so the limits leave room for it: 16,000 for the plan, 4,000 for a deep scan.
- Color warmth (-0.2..0.2) is implemented as `eq` filter `gamma_r`/`gamma_b` adjustments.
- Locked clips are **not** sent to the AI for modification — only the free clips are generated, and the locked ones are reinserted at their original positions afterward, with an overlap check.
- **Deep scan**: for one ambiguous clip, a second call re-examines it with 20 frames (vs. 4 in the first pass) across a wider window and refines just that cut.
- **Multiple variants**: up to 3 structurally different takes are generated in parallel from the same material and frames (reusing the expensive frame-extraction step); one variant failing doesn't take the others down.

## 7. How rendering works

- Each clip becomes a normalized segment (same resolution, fps, codec, pixel format) with an exact frame count, in a temp folder.
- No transitions, no text, no overlays → segments are joined with the concat demuxer, no re-encoding (fast).
- Otherwise → a single final re-encoded pass: `xfade` for transitions, `concat` for plain cuts, `drawtext` for text. All inputs go through `settb=AVTB` first (concat + xfade otherwise disagree on timebase).
- A transition starts exactly on the clip boundary: the previous clip is extended by the transition's duration (real extra source footage if available, otherwise the last frame frozen with `tpad`) — so total duration and beat alignment never shift.
- Photos: fit at 2× resolution, then `zoompan` (12% zoom) — avoids visible pixelation from zooming a still image.
- `fit`: `crop_center` fills 9:16 by cropping the edges; `blur_pad` keeps the whole frame on a blurred background.
- Draft = half resolution + CRF 28; final = full resolution + CRF 20.

### Overlays (B-roll / PIP / split-screen)

A second video/photo layer over part of the timeline — a PIP corner box or a 50/50 split — always muted so the main audio keeps playing underneath. It's its own `filter_complex` pass, run **after** the main timeline is assembled and **before** text is drawn, so captions stay on top of the overlay instead of hidden behind it.

Gotcha: the `overlay` filter's position expressions need `overlay_w`/`overlay_h`, not `ow`/`oh` (that naming belongs to a different filter). Getting it wrong only fails at filter-graph evaluation time, not at validation.

### J/L cuts (asymmetric audio trim)

`Clip.audio_lead` / `audio_lag` let a clip's sound start before its picture (J-cut) or run past it (L-cut), each with a crossfade over the borrowed window.

Clamped twice, so it can never ask for more than exists: once structurally, against the neighboring clip's own on-screen duration (`clamp_audio_trim` in `edl.py`); again at render time, against how much raw source footage actually exists past the cut (`build_original_audio` in `render.py`).

### Beat-synced visual punches

A quick whole-frame zoom pulse on every Nth beat (`MusicSettings.punch`/`punch_every`/`punch_amount`), reading the same beat grid as beat-sync — same feature, not a separate system.

Why `zoompan` and not `crop`: this ffmpeg build evaluates `crop`'s width/height once at filter-graph init, so it can't pulse frame by frame. `zoompan` with `d=1` (no frame duplication), indexed on the output frame counter instead of time, does the same job per frame. Found by reading `ffmpeg -h filter=crop` after the obvious `crop` approach failed at graph-build time. Applied last, after overlays and text, so the punch hits the whole composited frame.

### Live/quick preview

`render_preview()` is a separate, faster path for "see my edit now" — not a substitute for a real draft. It runs on proxy media at quarter resolution and caches each clip's rendered segment on disk, keyed by the clip's parameters and the source file read (path + mtime). Editing one clip only re-encodes that one; every untouched clip reuses its cached segment.

Kept away from the real export on purpose: `render()` (draft/final) never receives a cache directory and never sees proxy media, so a preview-quality segment can't end up in a real export. Old cache entries older than 2 days are pruned on upload.

### Background proxies

`proxy.py` transcodes every uploaded video to a small, fast-seeking 640px-wide copy in the background right after upload. Anything preview-only — AI frame extraction, the timeline filmstrip, deep scan, the waveform — reads the proxy once it's ready, and falls back to the original while it's still transcoding or if it fails. `render.py` never reads a proxy: the final export always uses the original file.

## 8. Audio: original sound, ducking, music, ambience, SFX

- `Clip.volume` controls the clip's *original* sound; video segments are always rendered muted (`-an`), with audio built as a separate pass.
- `build_original_audio()` produces one audio chunk per cut, exactly `main_frames/fps` seconds long (sample-accurate at 48kHz/30fps), with a short anti-pop fade; silent clips/photos become silence.
- `mux_audio()` layers music + ambience (summed first if both are present) + the original track + one-shot SFX. The background bed is ducked under the original sound via `sidechaincompress`, unless turned off. SFX are mixed in last, after ducking, so a whoosh/ding cuts through instead of being dampened like a bed.
- Beat detection is hand-rolled in plain numpy, no external library: spectral flux over 40 log-spaced frequency bands → tempo from autocorrelation (weighted toward ~120 BPM) → beat positions via dynamic-programming beat tracking (Ellis 2007-style). The AI sees BPM, time remaining in the track, and a per-second energy curve, and is asked to place more dynamic clips on higher-energy sections. The actual snap-to-beat is done by code (`snap_to_beats`), never by the AI.

## 9. Subtitles and DaVinci export

- Whisper transcription (`faster-whisper`, optional) is cached per file; **subtitles are never stored** — `subtitles.py` recomputes their timing from the transcript and the *current* cut list on every render, so editing the timeline can never desync a caption.
- The AI receives the transcript with timestamps and is told not to cut mid-sentence.
- FCPXML export (`fcpxml.py`) builds a DaVinci Resolve / Final Cut timeline that references the *original* source files, validated against OpenTimelineIO for frame-accurate timing. It does not carry over per-clip speed changes, manual pan/reframe, freely-positioned text, overlays, or beat punches — those are specific to this project's own renderer.

## 10. Robustness audit (fixes applied)

A pass through the whole codebase, looking specifically for what could crash the server, lose or corrupt project data, grow unbounded on disk/memory, or confuse the user. Found and fixed four real issues:

1. **One corrupt file could take down the whole project.** `scan_media()` ran `ffprobe` with `check=True` on every file; one bad file (an interrupted upload, wrong format under the right extension) raised an exception, and nearly every route (`get_project`, `upload`, `render`, `plan`, `deepscan`, FCPXML export...) calls `scan_media`. The whole project became inaccessible from the UI, with no fix short of editing files on disk. **Fixed**: a file that fails to probe is now skipped and logged instead of taking the rest down with it.
2. **Upload failed the whole batch for one bad file.** `upload()` assumed every saved file would also show up in `scan_media()`'s result, so one bad file in a batch raised a bare `KeyError` and failed every file in that upload, including the good ones. **Fixed**: each file is now handled independently. Good ones get a thumbnail as before; bad ones are deleted and reported separately (`{"saved": [...], "rejected": [...]}`), so the user sees exactly which file failed instead of one opaque error.
3. **Project state wasn't written atomically.** `save_state()` wrote straight over `state.json`; a server interruption at exactly that moment (crash, forced quit, power loss) could leave the file truncated and the project unloadable. **Fixed**: write to a temp file and atomically replace it, the same pattern already used for proxies — a reader always sees either the complete old file or the complete new one, never a half-written one.
4. **The in-memory job registry never shrank.** Every plan/render/preview/deep-scan click added an entry that stayed for the life of the process. **Fixed**: finished jobs older than an hour are pruned whenever a new job is created; anything still running is left alone.

Verified with a real end-to-end test, not just "starts without error": uploading a valid file and a corrupt one in the same request now returns `200` with the corrupt one correctly reported as rejected, and loading the project afterward — which used to return `500` — works normally and shows only the valid file.

**Known, accepted trade-offs** (not bugs): the preview segment cache is pruned by age (2 days), not by size, so a very long single session can accumulate a fair number of small cached segments. ffmpeg calls have no timeout, since a long render is often legitimate and a blanket timeout would cut those off too.

## 11. Conventions

- Generated files go into `output/` (drafts into `output/drafts/`), never over the source material.
- The user's original files are never modified or deleted automatically.
- A new plan (a demo, a regenerate, a picked variant) never silently wipes hand edits: `confirmReplacePlan` asks first when the plan was edited by hand, and `keepUndoAfterReplace` keeps the previous plan one Undo away. Error messages meant for the person editing are in plain words (e.g. `friendlyAiError` for a missing API key); technical detail stays in the server's logs.
- `validate_edl`'s messages stay as they are: the planner sends them back to Claude for a correction, and they count clips from 0 like the JSON. The page translates them for the person editing (`friendlyEdlError`, clip numbers from 1 as on screen); a new message in `edl.py` needs a matching rule there, or it shows untranslated.
- Projects can have a name (`state.json` "name", cleaned by `clean_project_name`; unnamed ones show their id), asked for on "+ New project" and changed with Rename. The cover is saved in the page from the player's current frame (no server route), at the rendered video's size.
- A first visit starts in simple mode (`SIMPLE_FEATURES`: example video, DaVinci, SFX/ambience, deep scan, lock rules, variants and B-roll off); the header's Simple / All features switch and Settings change it, saved in `localStorage`.
- Every major feature that adds UI surface gets a toggle in the Settings panel (`FEATURES`), so the interface stays as uncluttered as the user wants it; a right-click context menu on a clip offers quick access to actions that already exist as buttons elsewhere — it's a shortcut, never a new capability.
