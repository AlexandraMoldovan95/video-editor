"""Material + preset + user settings -> prompt + frames -> Claude -> validated EDL.

- Tool `submit_edl` with tool_choice "auto" (claude-sonnet-5-5 doesn't accept a forced tool).
- Per-file settings (hints): constraints are checked, overrides are applied by the code.
- Locked clips are NOT sent to the AI for modification: the AI only generates the free
  clips, and the locked ones are then reinserted at their positions.
"""
import argparse
import base64
import json
import os
import tempfile
from pathlib import Path

from . import hints as hints_mod
from .edl import (EDL, Clip, TextItem, clamp_transitions, clip_duration, TRANSITIONS_DYNAMIC, TRANSITIONS_SUBTLE,
                  overlap_errors, text_errors, total_duration, transition_errors, validate_edl)
from .frames import extract_frames, measure_rhythm
from .hints import FileHint
from . import subtitles as subs_mod
from .music import MusicInfo, beat_grid, prompt_block, snap_to_beats
from .presets import COLOR_BASE, PRESETS, TEXT_RULES, TRANSITION_RULES, resolve
from .probe import MediaInfo, probe
from .render import scan_media

MODEL = "claude-sonnet-5-5"
MAX_RETRIES = 1
# This model thinks by default (adaptive thinking), and thinking counts toward max_tokens,
# so the limits leave room for it on top of the tool call itself.
MAX_TOKENS_PLAN = 16000
MAX_TOKENS_DEEPSCAN = 4000


def check_stop(resp, what: str) -> None:
    """A cut-off or refused answer is reported as such, instead of failing validation as a
    "wrong plan" and using up the one correction retry on a reply that was never complete."""
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"The AI's {what} was cut off at the token limit "
                           f"({resp.usage.output_tokens} output tokens). Try again with fewer clips.")
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"The AI declined to produce the {what}.")


def make_client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")
    return anthropic.Anthropic()

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "the exact file name"},
        "in": {"type": "number", "description": "video only: start second"},
        "out": {"type": "number", "description": "video only: end second"},
        "duration": {"type": "number", "minimum": 2, "maximum": 4, "description": "photos only"},
        "motion": {"enum": ["zoom_in", "zoom_out", "static"]},
        "fit": {"enum": ["crop_center", "blur_pad"]},
        "brightness": {"type": "number", "minimum": -0.3, "maximum": 0.3},
        "contrast": {"type": "number", "minimum": 0.5, "maximum": 2.0},
        "saturation": {"type": "number", "minimum": 0.0, "maximum": 3.0},
        "warmth": {"type": "number", "minimum": -0.2, "maximum": 0.2},
        "volume": {"type": "number", "minimum": 0, "maximum": 2,
                   "description": "the clip's ORIGINAL sound volume (1 = normal, 0 = muted); clips WITH SOUND only"},
        "transition_in": {
            "type": "object", "description": "the transition FROM the previous clip; omit for a direct cut",
            "properties": {"type": {"enum": TRANSITIONS_SUBTLE + TRANSITIONS_DYNAMIC},
                           "duration": {"type": "number", "minimum": 0.15, "maximum": 1.0}},
            "required": ["type", "duration"]},
        "reason": {"type": "string", "maxLength": 200,
                   "description": "ONE short sentence: why THIS clip, THIS cut, in THIS position - "
                                  "not a repeat of the global 'reasoning'."},
        "low_confidence": {"type": "boolean",
                           "description": "true if you're NOT confident about this exact cut - fast motion, "
                                          "action-heavy, motion blur, or an ambiguous moment where the frames "
                                          "don't give you enough to judge precisely. false when you're sure."},
    },
    "required": ["source", "fit", "brightness", "contrast", "saturation", "warmth", "reason", "low_confidence"],
}
TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "maxLength": 80, "description": "in English"},
        "start": {"type": "number"}, "end": {"type": "number"},
        "position": {"enum": ["top", "center", "bottom"]},
        "style": {"enum": ["title", "caption", "cta"]},
        "variants": {
            "type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 2,
            "description": "ONLY for the opening hook (the first title-style text): 1-2 alternate "
                           "wordings of the same hook, same meaning/length ballpark, different angle "
                           "(e.g. question vs statement vs curiosity gap), so the user can A/B test it. "
                           "Leave empty for every other text.",
        },
        "reason": {"type": "string", "maxLength": 200,
                   "description": "ONE short sentence: why this text, this wording, this placement/timing - "
                                  "not a repeat of the global 'reasoning'."},
    },
    "required": ["content", "start", "end", "position", "style", "reason"],
}
TOOL = {
    "name": "submit_edl",
    "description": "Submits the final edit plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "2-4 sentences: the GLOBAL overview - why this "
                                                            "order/arc overall. Per-clip/per-text detail goes "
                                                            "in each item's own 'reason' field instead."},
            "clips": {"type": "array", "items": CLIP_SCHEMA, "minItems": 1},
            "texts": {"type": "array", "items": TEXT_SCHEMA, "description": "on-screen texts, times on the final video"},
        },
        "required": ["reasoning", "clips"],
    },
}

SYSTEM = """You are a video editor for social media (vertical format 9:16, 1080x1920, 30fps).
You receive the raw material (metadata + frames extracted at timestamps), a preset, the user's settings and rules.
You build an edit plan and send it ONLY through the submit_edl tool, with no other text.

Strict rules:
- Use only the listed files, with their exact names.
- Video: "in" < "out" <= the clip's duration; every cut respects the requested length range.
- VARY the cut lengths within the range (don't make them all equal) - the rhythm needs to breathe.
- Photos: use "duration" (2-4s), not "in"/"out".
- Total duration as close as possible to the target duration (+/-10%). If the material falls short, don't invent more.
- The same clip can be used more than once only with intervals that do NOT overlap.
- Horizontal clips get the fit from the preset; vertical ones get "crop_center".
- Color: start from the look's base values and adjust per clip only as much as needed for everything to look consistent.
  If an example video is attached below, ITS color grade is the real target: match its warmth, contrast and
  saturation mood as closely as the brightness/contrast/saturation/warmth ranges allow, even where that means
  drifting from the preset's base numbers.
- The user's per-file settings are MANDATORY and take priority over the preset.
- Original sound: clips WITH SOUND keep their sound (volume 1). Only set volume 0 if the user asks for it,
  or if the sound would hurt the result (e.g. wind over a musical moment). If a transcript exists, do NOT cut
  in the middle of a sentence: align "in"/"out" to the start/end of sentences.
- Transitions: a transition lasts at most 40% of the clips it joins. It doesn't change the total duration.
- Texts: times are on the FINAL video (second 0 = the start). Short, natural, no marketing cliches;
  if the user gave no context (the place, the occasion), write generic but understated texts, don't invent names or facts.
  If an example video is attached below and it shows on-screen text/captions, match ITS general style - tone,
  capitalization, density (how often text appears), placement - never its exact wording, only the style.
- Opening hook: for the FIRST title-style text only, also fill "variants" with 1-2 alternate wordings
  (same idea, different angle) so the user can flip between options without regenerating. Leave
  "variants" empty for every other text.
- Order: build an arc (strong opening, rhythm, memorable ending). Choose the moments based on the frames.
- Per-item "reason": every clip and every text also gets its OWN one-sentence "reason" field -
  specific to that item (why this exact moment/cut, why this order, why this wording/timing), never
  a copy-paste of the global "reasoning" or of another item's reason.
- Per-clip "low_confidence": be HONEST about it. Fast motion, action-heavy footage, motion blur, or
  frames too sparse/ambiguous to judge the exact in/out precisely all deserve low_confidence=true -
  don't mark everything confident just to look decisive. Most calm/clear clips are low_confidence=false.
- Write "reasoning" (the short global overview) and every "reason" in English."""


def _img_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def _clip_dur(c: Clip) -> float:
    # on-screen duration, not raw source length - a locked slow-mo'd clip takes longer on screen
    # than its in/out span, and the AI's remaining time budget needs to account for that.
    return clip_duration(c)


def build_request(media: dict[str, MediaInfo], preset_name: str, overrides: dict,
                  example: Path | None, frames_dir: Path,
                  file_hints: dict[str, FileHint] | None = None,
                  locked: list[tuple[int, Clip]] | None = None,
                  music: dict | None = None,
                  locked_texts: list[TextItem] | None = None,
                  transcripts: dict | None = None) -> tuple[list[dict], dict]:
    file_hints = file_hints or {}
    locked = locked or []
    usable = {n: m for n, m in media.items() if file_hints.get(n, FileHint()).use != "exclude"}
    rhythm = measure_rhythm(probe(example)) if example else None
    preset, cut_range = resolve(preset_name, overrides, rhythm.pace_range if rhythm else None)
    locked_total = sum(_clip_dur(c) for _, c in locked)
    target = max(preset.target_duration - locked_total, 0)

    lines = ["## Material"]
    for name, m in usable.items():
        orient = "vertical" if m.height > m.width else "horizontal"
        if m.kind == "video":
            snd = "WITH SOUND" if m.has_audio else "no sound"
            lines.append(f"- {name}: video, {m.duration:.2f}s, {m.width}x{m.height} ({orient}), {snd}")
        else:
            lines.append(f"- {name}: photo, {m.width}x{m.height} ({orient})")
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]

    for name, m in usable.items():
        h = file_hints.get(name, FileHint())
        for f in extract_frames(m, frames_dir, t0=h.segment_in, t1=h.segment_out):
            label = f"{name} @ {f.timestamp}s" if f.timestamp is not None else f"{name} (photo)"
            content.append({"type": "text", "text": label})
            content.append(_img_block(f.path))

    spec = [
        "## Preset",
        f"- target duration: {target:.1f}s" + (f" (out of {preset.target_duration}s, {locked_total:.1f}s is already locked by the user)" if locked else ""),
        f"- cut length: between {cut_range[0]}s and {cut_range[1]}s"
        + (f" (measured from the example video: median {rhythm.median}s)" if rhythm and "pace" not in overrides else ""),
        f"- color look: {preset.color_look} -> base values {json.dumps(COLOR_BASE[preset.color_look])}"
        + (" (but see the example video below: match ITS color grade over these base numbers if they differ)"
           if example and "color_look" not in overrides else ""),
        f"- fit for horizontal clips: {preset.vertical_fit}",
        f"- photo motion: {preset.photo_motion}",
        f"- transitions: {TRANSITION_RULES[preset.transition_style]}",
        f"- on-screen text: {TEXT_RULES[preset.text_style]}"
        + (" (but see the example video below: if it has on-screen text, match its style/tone, not its wording)"
           if example and "text_style" not in overrides else ""),
    ]
    if preset.notes:
        spec.append(f"- the user's notes: {preset.notes}")
    content.append({"type": "text", "text": "\n".join(spec)})

    tb = subs_mod.prompt_block(transcripts or {}, list(usable))
    if tb:
        content.append({"type": "text", "text": tb})

    if music:
        content.append({"type": "text", "text": prompt_block(music["info"], music["start"], preset.target_duration)})

    hint_lines = [d for n, h in file_hints.items() if n in usable and (d := hints_mod.describe(n, h))]
    if hint_lines:
        content.append({"type": "text", "text": "## The user's per-file settings (MANDATORY)\n" + "\n".join(hint_lines)})

    if locked:
        ll = [f"- position {i}: {c.source} " + (f"{c.in_}-{c.out}s" if c.in_ is not None else f"photo {c.duration}s")
              + f", color b={c.brightness} c={c.contrast} s={c.saturation} w={c.warmth}" for i, c in locked]
        content.append({"type": "text", "text":
                        "## Clips LOCKED by the user (stay exactly as is, do NOT include them in your response)\n" + "\n".join(ll)
                        + "\nGenerate ONLY the rest of the clips. Don't use intervals that overlap the locked ones. "
                          "Harmonize the color with the locked clips. They will be reinserted at their indicated positions."})

    if locked_texts:
        tl = [f"- \"{t.content}\" {t.start}-{t.end}s ({t.style}, "
              f"{'a custom free position' if (t.x is not None or t.y is not None) else t.position})" for t in locked_texts]
        content.append({"type": "text", "text": "## Texts LOCKED by the user (they stay, do NOT include them in your response; "
                        "don't add other texts that overlap them in time)\n" + "\n".join(tl)})

    if example:
        ex_frames = extract_frames(probe(example), frames_dir / "_example")
        content.append({"type": "text", "text": "## Example video (the target style to match: color grade, "
                        "framing, AND on-screen text style/tone if it has captions - on top of the cut pace "
                        "already measured above)"})
        content += [_img_block(f.path) for f in ex_frames]

    meta = {"preset": preset.model_dump(), "cut_range": cut_range, "target_free": target,
            "rhythm_median": rhythm.median if rhythm else None,
            "images": sum(1 for b in content if b["type"] == "image")}
    return [{"role": "user", "content": content}], meta


def merge_locked(free: list[Clip], locked: list[tuple[int, Clip]]) -> list[Clip]:
    out = list(free)
    for i, c in sorted(locked, key=lambda x: x[0]):
        out.insert(min(i, len(out)), c)
    return out


def finalize(raw: dict, media: dict, file_hints: dict[str, FileHint],
             locked: list[tuple[int, Clip]], grid=None,
             locked_texts: list[TextItem] | None = None) -> EDL:
    """Validates the AI's response + reinserts the locked clips + checks/applies the user's settings."""
    free = validate_edl({"clips": raw["clips"]}, media).clips
    for c in free:
        c.locked = False
    texts = [TextItem.model_validate({**t, "locked": False}) for t in raw.get("texts") or []]
    texts = sorted((locked_texts or []) + texts, key=lambda t: t.start)
    edl = EDL(clips=merge_locked(free, locked), texts=texts)
    errs = (overlap_errors(edl) + transition_errors(edl) + text_errors(edl)
            + hints_mod.check(edl, file_hints))
    if errs:
        raise ValueError("Invalid plan:\n  " + "\n  ".join(errs))
    edl = hints_mod.apply(edl, file_hints)
    if grid is not None:
        edl = clamp_transitions(snap_to_beats(edl, media, grid))
    text_errors(edl)   # silently trims texts that now run past the new end
    return edl


def music_grid(music: dict | None):
    if music and music.get("beat_sync", True):
        return beat_grid(music["info"], music["start"])
    return None


def call_claude(messages: list[dict], media: dict, file_hints: dict[str, FileHint],
                locked: list[tuple[int, Clip]], grid=None,
                locked_texts: list[TextItem] | None = None, client=None) -> tuple[EDL, str, dict]:
    client = client or make_client()   # tests pass a fake client
    usage = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(MAX_RETRIES + 1):
        resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS_PLAN, system=SYSTEM, messages=messages,
                                      tools=[TOOL], tool_choice={"type": "auto"})
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        check_stop(resp, "edit plan")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            if attempt == MAX_RETRIES:
                raise RuntimeError("The AI did not call submit_edl")
            messages = messages + [{"role": "assistant", "content": resp.content},
                                   {"role": "user", "content": "Send the plan ONLY through the submit_edl tool."}]
            continue
        try:
            edl = finalize(block.input, media, file_hints, locked, grid, locked_texts)
            return edl, block.input.get("reasoning", ""), usage
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            print(f"  invalid plan, asking for a correction: {e}")
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                              "content": f"{e}\nCorrect it and resend."}]},
            ]
    raise RuntimeError("unreachable")


def plan(media: dict[str, MediaInfo], preset_name: str = "neutral", overrides: dict | None = None,
         example: Path | None = None, file_hints: dict[str, FileHint] | None = None,
         current: EDL | None = None, music: dict | None = None,
         transcripts: dict | None = None) -> dict:
    """The entry point used by the server and the CLI.

    music = {"info": MusicInfo, "start": float, "beat_sync": bool} or None
    """
    file_hints = file_hints or {}
    locked = [(i, c) for i, c in enumerate(current.clips) if c.locked] if current else []
    locked_texts = [t for t in current.texts if t.locked] if current else []
    with tempfile.TemporaryDirectory(prefix="vedit_frames_") as tmp:
        messages, meta = build_request(media, preset_name, overrides or {}, example, Path(tmp),
                                       file_hints, locked, music, locked_texts, transcripts)
        edl, reasoning, usage = call_claude(messages, media, file_hints, locked, music_grid(music),
                                            locked_texts)
    return {"edl": edl, "reasoning": reasoning, "usage": usage, "meta": meta}


# ---------------- multiple draft variants in one generation ----------------
# Reuses the SAME extracted frames (the expensive part) for every variant - only the final
# instruction block differs, nudging each call toward a structurally different take rather than
# just a reshuffle of the same cuts. Each variant still goes through the normal finalize()
# validation/retry loop independently, so one variant failing doesn't take the others down with it.
VARIANT_ANGLES = [
    "## This generation\nVariant A: your single best edit - whatever structure you think serves this material best.",
    "## This generation\nVariant B: a STRUCTURALLY different take from variant A - a different opening, a different "
    "pace, or a different subset/order of the moments. Don't just reshuffle the same cuts with the same shape.",
    "## This generation\nVariant C: a third, again structurally distinct take from both A and B - a different arc, "
    "a different emphasis, or a different selection of material.",
]


def plan_variants(media: dict[str, MediaInfo], preset_name: str = "neutral", overrides: dict | None = None,
                  example: Path | None = None, file_hints: dict[str, FileHint] | None = None,
                  current: EDL | None = None, music: dict | None = None,
                  transcripts: dict | None = None, n: int = 3) -> list[dict]:
    """Returns up to n variants: [{"label", "edl", "reasoning", "usage"} | {"label", "error"}, ...].
    Raises only if EVERY variant failed."""
    import concurrent.futures
    file_hints = file_hints or {}
    locked = [(i, c) for i, c in enumerate(current.clips) if c.locked] if current else []
    locked_texts = [t for t in current.texts if t.locked] if current else []
    n = max(1, min(n, len(VARIANT_ANGLES)))
    grid = music_grid(music)

    def run_one(label: str, angle: str, content: list[dict]) -> dict:
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": angle}]}]
        try:
            edl, reasoning, usage = call_claude(msgs, media, file_hints, locked, grid, locked_texts)
            return {"label": label, "edl": edl, "reasoning": reasoning, "usage": usage}
        except Exception as e:  # noqa: BLE001
            return {"label": label, "error": str(e)}

    with tempfile.TemporaryDirectory(prefix="vedit_frames_") as tmp:
        messages, _meta = build_request(media, preset_name, overrides or {}, example, Path(tmp),
                                        file_hints, locked, music, locked_texts, transcripts)
        base_content = messages[0]["content"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(run_one, chr(65 + i), VARIANT_ANGLES[i], base_content) for i in range(n)]
            results = [f.result() for f in futs]
    results.sort(key=lambda r: r["label"])
    if not any("edl" in r for r in results):
        raise RuntimeError("All variants failed: " + "; ".join(r.get("error", "?") for r in results))
    return results


# ---------------- deep scan: a close re-look at ONE clip with many more frames ----------------
# Opt-in, per-clip: the first pass only sends FRAMES_PER_CLIP (4) frames per file, which is plenty for
# most footage but can miss the right moment in fast/action-heavy material. This re-extracts DEEPSCAN_FRAMES
# frames across a window around the clip's current cut and asks the AI to refine just that one clip.
DEEPSCAN_FRAMES = 20

DEEPSCAN_CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "in": {"type": "number", "description": "refined start second, in the SOURCE file's own time"},
        "out": {"type": "number", "description": "refined end second, in the SOURCE file's own time"},
        "brightness": {"type": "number", "minimum": -0.3, "maximum": 0.3},
        "contrast": {"type": "number", "minimum": 0.5, "maximum": 2.0},
        "saturation": {"type": "number", "minimum": 0.0, "maximum": 3.0},
        "warmth": {"type": "number", "minimum": -0.2, "maximum": 0.2},
        "reason": {"type": "string", "maxLength": 200, "description": "one short sentence on this refined choice"},
        "low_confidence": {"type": "boolean",
                           "description": "after this much closer look, are you STILL not sure? usually false now - "
                                          "only keep it true if even 20 frames can't resolve the ambiguity."},
    },
    "required": ["in", "out", "brightness", "contrast", "saturation", "warmth", "reason", "low_confidence"],
}
DEEPSCAN_TOOL = {
    "name": "submit_clip",
    "description": "Submits the refined cut + color for this one clip, after a close look at many more frames.",
    "input_schema": {
        "type": "object",
        "properties": {"clip": DEEPSCAN_CLIP_SCHEMA},
        "required": ["clip"],
    },
}
DEEPSCAN_SYSTEM = """You are a video editor doing a CLOSE REVIEW of a single clip you cut earlier.
The user asked for a deeper look because this moment is fast/action-heavy or otherwise hard to judge from a
handful of frames. You now get many more frames, spread across a wider window around your original cut, from
the same source file.
Re-examine and refine ONLY this clip:
- Pick the best possible exact "in"/"out" in the source file's own time. It's fine to keep it the same as
  before if it's already right, or to shift it - don't change it just to look busy.
- Aim to keep roughly the same on-screen duration as the current cut unless the material clearly calls for
  a different length; a wildly different duration would throw off the rest of the timeline.
- Refine brightness/contrast/saturation/warmth only if the extra frames reveal something the first pass missed.
- Submit your answer ONLY through the submit_clip tool, with no other text.
- Write "reason" (one short sentence) and be honest in "low_confidence": most of the time this closer look
  resolves the uncertainty, so it should now be false; keep it true only if it genuinely still can't be judged."""


def build_deepscan_request(info: MediaInfo, clip: Clip, frames_dir: Path) -> tuple[list[dict], dict]:
    cur_len = clip.out - clip.in_
    pad = max(1.0, cur_len * 0.5)
    t0 = max(0.0, clip.in_ - pad)
    t1 = min(info.duration, clip.out + pad)
    frames = extract_frames(info, frames_dir, n=DEEPSCAN_FRAMES, t0=t0, t1=t1)
    lines = [
        "## Clip under review",
        f"- file: {clip.source}, {info.duration:.2f}s total, {info.width}x{info.height}",
        f"- your current cut: {clip.in_:.2f}-{clip.out:.2f}s ({cur_len:.2f}s on screen)",
        f"- current color: brightness={clip.brightness} contrast={clip.contrast} saturation={clip.saturation} warmth={clip.warmth}",
        f"- {len(frames)} frames below, evenly spread across {t0:.2f}-{t1:.2f}s (a window around your current cut)",
    ]
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for f in frames:
        content.append({"type": "text", "text": f"{f.source} @ {f.timestamp}s"})
        content.append(_img_block(f.path))
    meta = {"t0": t0, "t1": t1, "images": len(frames)}
    return [{"role": "user", "content": content}], meta


def deep_scan_clip(media: dict[str, MediaInfo], clip: Clip, frames_dir: Path,
                   client=None) -> tuple[Clip, dict]:
    """Re-examines ONE clip with many more frames and returns a refined, already-locked Clip."""
    if clip.in_ is None or clip.out is None:
        raise ValueError("deep scan only applies to a video clip, not a photo")
    info = media.get(clip.source)
    if info is None or info.kind != "video":
        raise ValueError(f"deep scan: source file not found or not a video: {clip.source!r}")

    messages, meta = build_deepscan_request(info, clip, frames_dir)
    client = client or make_client()   # tests pass a fake client
    usage = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(MAX_RETRIES + 1):
        resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS_DEEPSCAN, system=DEEPSCAN_SYSTEM,
                                      messages=messages, tools=[DEEPSCAN_TOOL], tool_choice={"type": "auto"})
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        check_stop(resp, "deep-scan answer")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            if attempt == MAX_RETRIES:
                raise RuntimeError("The AI did not call submit_clip")
            messages = messages + [{"role": "assistant", "content": resp.content},
                                   {"role": "user", "content": "Send your answer ONLY through the submit_clip tool."}]
            continue
        raw = block.input.get("clip") or {}
        try:
            if raw.get("in") is None or raw.get("out") is None:
                raise ValueError("missing in/out")
            if not (raw["in"] < raw["out"] <= info.duration + 0.05):
                raise ValueError(f"invalid interval {raw['in']}-{raw['out']} (duration {info.duration:.2f}s)")
            if raw["out"] - raw["in"] < 0.5:
                raise ValueError("cut too short (<0.5s)")
            data = clip.model_dump(by_alias=True, exclude_none=True)
            data.update({
                "in": round(raw["in"], 2), "out": round(raw["out"], 2),
                "brightness": raw["brightness"], "contrast": raw["contrast"],
                "saturation": raw["saturation"], "warmth": raw["warmth"],
                "reason": raw.get("reason"), "low_confidence": bool(raw.get("low_confidence", False)),
                "locked": True,
            })
            updated = Clip.model_validate(data)
            return updated, usage
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            print(f"  invalid deep-scan answer, asking for a correction: {e}")
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                              "content": f"{e}\nCorrect it and resend."}]},
            ]
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser(description="Generates an edit plan with Claude.")
    ap.add_argument("--input", default=".", help="the folder with the raw material")
    ap.add_argument("--preset", default="neutral", choices=list(PRESETS))
    ap.add_argument("--duration", type=int, help="target duration in seconds, 5-120")
    ap.add_argument("--pace", choices=["slow", "medium", "fast"])
    ap.add_argument("--look", dest="color_look", choices=list(COLOR_BASE))
    ap.add_argument("--notes", default="")
    ap.add_argument("--example", type=Path, help="an edited example video (for rhythm and style)")
    ap.add_argument("--hints", type=Path, help="JSON {file: settings} (see app/hints.py)")
    ap.add_argument("--current", type=Path, help="an existing EDL; clips with locked=true stay")
    ap.add_argument("--out", type=Path, default=Path("edl_examples/ai_plan.json"))
    ap.add_argument("--dry-run", action="store_true", help="prepare the prompt without calling the API")
    a = ap.parse_args()

    overrides = {"target_duration": a.duration, "pace": a.pace, "color_look": a.color_look, "notes": a.notes}
    input_dir = Path(a.input).resolve()
    media = scan_media(input_dir)
    if a.example and a.example.resolve().parent == input_dir:
        media.pop(a.example.name, None)
    fh = {k: FileHint.model_validate(v) for k, v in json.loads(a.hints.read_text("utf-8")).items()} if a.hints else {}
    cur = validate_edl(json.loads(a.current.read_text("utf-8")), media) if a.current else None

    if a.dry_run:
        locked = [(i, c) for i, c in enumerate(cur.clips) if c.locked] if cur else []
        with tempfile.TemporaryDirectory() as tmp:
            messages, meta = build_request(media, a.preset, overrides, a.example, Path(tmp), fh, locked)
        print(f"Prompt: {meta['images']} images | {meta}")
        print("\n".join(b["text"] for b in messages[0]["content"] if b["type"] == "text" and b["text"].startswith("##")))
        return

    r = plan(media, a.preset, overrides, a.example, fh, cur)
    edl = r["edl"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(edl.model_dump(by_alias=True, exclude_none=True), indent=2, ensure_ascii=False), "utf-8")
    print(f"Plan: {len(edl.clips)} clips, {total_duration(edl):.1f}s -> {a.out}")
    print(f"AI reasoning: {r['reasoning']}")
    print(f"Tokens: {r['usage']['input_tokens']} in / {r['usage']['output_tokens']} out")


if __name__ == "__main__":
    main()
