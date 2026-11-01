"""Edit plan (EDL) schema + validation. The AI produces JSON, this code checks it."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .probe import MediaInfo


class Output(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = Field(30, ge=24, le=60)


TransitionType = Literal["fade", "dissolve", "fadeblack", "fadewhite", "wipeleft", "slideleft",
                         "slideup", "smoothleft", "circleopen", "circleclose"]
TRANSITIONS_SUBTLE = ["fade", "dissolve", "fadeblack"]
TRANSITIONS_DYNAMIC = ["slideleft", "slideup", "wipeleft", "smoothleft", "circleopen", "circleclose", "fadewhite"]
# NOTE: "zoomin" is deliberately NOT offered here even though some newer ffmpeg builds support it as an
# xfade transition name - it isn't in the xfade transition list on ffmpeg 4.4 (e.g. Ubuntu 22.04's
# packaged build), so a plan that picks it fails at the final ffmpeg assembly step on an older install.
# All ten names above were checked against `ffmpeg -h filter=xfade` and work on 4.4+.


class Transition(BaseModel):
    type: TransitionType
    duration: float = Field(0.4, ge=0.15, le=1.0)


class TextItem(BaseModel):
    content: str = Field(min_length=1, max_length=80)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    position: Literal["top", "center", "bottom"] = "bottom"   # used while x/y are unset (AI-placed texts)
    x: float | None = Field(None, ge=0.0, le=1.0)   # manual free position: fraction of width, block CENTER
    y: float | None = Field(None, ge=0.0, le=1.0)   # same, fraction of height. Either set -> overrides `position`.
    style: Literal["title", "caption", "cta", "subtitle"] = "caption"   # subtitle = generated from Whisper
    locked: bool = False
    # alternate wordings for this same text (typically the opening hook): the UI lets the user flip
    # `content` between these without a second AI call. Not hook-specific by design - any text item
    # can carry variants, though the AI is only prompted to propose them for the opening title text.
    variants: list[str] = Field(default_factory=list, max_length=2)
    reason: str | None = Field(None, max_length=200)   # AI-only: one short sentence on why this text/wording/placement

    @model_validator(mode="after")
    def _times(self):
        if self.end - self.start < 0.5:
            raise ValueError(f"the text '{self.content}' must stay on screen for at least 0.5s")
        return self

    @field_validator("variants")
    @classmethod
    def _variant_length(cls, v: list[str]) -> list[str]:
        for s in v:
            if not (1 <= len(s) <= 80):
                raise ValueError("each text variant must be 1-80 characters")
        return v


class Clip(BaseModel):
    source: str
    # video
    in_: float | None = Field(None, alias="in", ge=0)
    out: float | None = Field(None, ge=0)
    # photo
    duration: float | None = Field(None, ge=2.0, le=4.0)
    motion: Literal["zoom_in", "zoom_out", "static"] = "zoom_in"
    # shared
    fit: Literal["crop_center", "blur_pad"] = "crop_center"
    brightness: float = Field(0.0, ge=-0.3, le=0.3)
    contrast: float = Field(1.0, ge=0.5, le=2.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0)
    warmth: float = Field(0.0, ge=-0.2, le=0.2)   # >0 warm (more red), <0 cool (more blue)
    volume: float = Field(1.0, ge=0.0, le=2.0)    # the clip's ORIGINAL sound (0 = mute); photos have no sound
    speed: float = Field(1.0, ge=0.25, le=4.0)    # playback rate: 1.0 normal, <1 slow-mo, >1 timelapse. Video only.
    pan_x: float = Field(0.0, ge=-1.0, le=1.0)    # manual reframe for fit=crop_center: -1 full left, 0 center, 1 full right
    pan_y: float = Field(0.0, ge=-1.0, le=1.0)    # same, vertically: -1 full top, 1 full bottom
    locked: bool = False   # locked by the user: the AI won't change it on regeneration
    transition_in: Transition | None = None   # the transition FROM the previous clip INTO this one
    reason: str | None = Field(None, max_length=200)   # AI-only: one short sentence on why THIS clip/cut/position
    low_confidence: bool = False   # AI-only: true if the AI wasn't sure about this cut (fast/action-heavy, hard to judge from frames)
    # J/L cuts: this clip's OWN sound (not music/sfx) can start before or keep playing after its own
    # video - "borrowed" from the neighboring clip's on-screen time. audio_lead = seconds this clip's
    # audio starts EARLY (plays under the END of the PREVIOUS clip's picture - a "J-cut"). audio_lag =
    # seconds this clip's audio keeps going AFTER its own picture ends (plays under the START of the
    # NEXT clip's picture - an "L-cut"). Both clamped at render/validation time to whatever room the
    # neighboring clip and the source file actually have; video clips only (video has_audio).
    audio_lead: float = Field(0.0, ge=0.0, le=5.0)
    audio_lag: float = Field(0.0, ge=0.0, le=5.0)

    model_config = {"populate_by_name": True}

    @field_validator("source")
    @classmethod
    def only_filename(cls, v: str) -> str:
        # no paths: the AI can't request files outside the input folder
        if Path(v).name != v or v.startswith("."):
            raise ValueError(f"source must be a filename only: {v!r}")
        return v

    @model_validator(mode="after")
    def _speed_needs_video(self):
        if self.in_ is None and self.speed != 1.0:
            raise ValueError("speed only applies to video clips, not photos")
        return self

    @model_validator(mode="after")
    def _pan_needs_crop_center(self):
        if self.fit != "crop_center" and (self.pan_x != 0.0 or self.pan_y != 0.0):
            raise ValueError("pan_x/pan_y only apply to fit='crop_center' (blur_pad shows the whole frame)")
        return self

    @model_validator(mode="after")
    def _audio_trim_needs_video(self):
        if self.in_ is None and (self.audio_lead != 0.0 or self.audio_lag != 0.0):
            raise ValueError("audio_lead/audio_lag only apply to video clips, not photos")
        return self


SfxName = Literal["whoosh", "ding", "impact", "pop", "click"]


class SfxCue(BaseModel):
    name: SfxName
    start: float = Field(ge=0)
    volume: float = Field(1.0, ge=0.0, le=3.0)


OverlayMode = Literal["pip_tl", "pip_tr", "pip_bl", "pip_br", "split_left", "split_right"]


class OverlayItem(BaseModel):
    """A second video/photo layer composited on TOP of the main timeline for a time window -
    B-roll-in-a-corner (PIP) or a side-by-side (split-screen). Manual-only for now (never produced
    by the AI planner): added, moved and removed by hand in the UI. Always muted - this is a
    picture-only layer, the main timeline's own audio (+ music/sfx) keeps playing underneath
    unchanged, so there's never a surprise second audio source to untangle."""
    source: str
    in_: float | None = Field(None, alias="in", ge=0)   # video
    out: float | None = Field(None, ge=0)
    duration: float | None = Field(None, ge=0.3, le=20.0)   # photo
    start: float = Field(ge=0)   # absolute position on the MAIN timeline (on-screen seconds)
    mode: OverlayMode = "pip_br"
    scale: float = Field(0.35, ge=0.15, le=0.6)   # PIP only (fraction of frame width); ignored for split_*
    locked: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("source")
    @classmethod
    def only_filename(cls, v: str) -> str:
        if Path(v).name != v or v.startswith("."):
            raise ValueError(f"source must be a filename only: {v!r}")
        return v


class EDL(BaseModel):
    output: Output = Output()
    clips: list[Clip] = Field(min_length=1, max_length=60)
    texts: list[TextItem] = Field(default_factory=list, max_length=12)
    sfx: list[SfxCue] = Field(default_factory=list, max_length=30)
    overlays: list[OverlayItem] = Field(default_factory=list, max_length=20)


def validate_edl(raw: dict, media: dict[str, MediaInfo]) -> EDL:
    """Validates the schema + checks the plan against the real files."""
    edl = EDL.model_validate(raw)
    errors = []
    for i, c in enumerate(edl.clips):
        info = media.get(c.source)
        if info is None:
            errors.append(f"clip {i}: file does not exist {c.source!r}")
            continue
        if info.kind == "video":
            if c.in_ is None or c.out is None:
                errors.append(f"clip {i}: video without 'in'/'out'")
            elif not (c.in_ < c.out <= info.duration + 0.05):
                errors.append(f"clip {i}: invalid interval {c.in_}-{c.out} (duration {info.duration:.2f}s)")
            elif c.out - c.in_ < 0.5:
                errors.append(f"clip {i}: cut too short (<0.5s)")
        else:
            if c.duration is None:
                c.duration = 2.5
    for i, o in enumerate(edl.overlays):
        info = media.get(o.source)
        if info is None:
            errors.append(f"overlay {i}: file does not exist {o.source!r}")
            continue
        if info.kind == "video":
            if o.in_ is None or o.out is None:
                errors.append(f"overlay {i}: video without 'in'/'out'")
            elif not (o.in_ < o.out <= info.duration + 0.05):
                errors.append(f"overlay {i}: invalid interval {o.in_}-{o.out} (duration {info.duration:.2f}s)")
            elif o.out - o.in_ < 0.3:
                errors.append(f"overlay {i}: too short (<0.3s)")
        else:
            if o.duration is None:
                o.duration = 2.5
    errors += overlap_errors(edl) + transition_errors(edl) + text_errors(edl) + sfx_errors(edl) + overlay_errors(edl)
    if errors:
        raise ValueError("Invalid EDL:\n  " + "\n  ".join(errors))
    clamp_audio_trim(edl)
    return edl


def total_duration(edl: EDL) -> float:
    return sum(clip_duration(c) for c in edl.clips)


def overlap_errors(edl: EDL) -> list[str]:
    """The same video clip used more than once can't have overlapping intervals."""
    by_src: dict[str, list[tuple[float, float, int]]] = {}
    for i, c in enumerate(edl.clips):
        if c.in_ is not None and c.out is not None:
            by_src.setdefault(c.source, []).append((c.in_, c.out, i))
    errs = []
    for src, spans in by_src.items():
        spans.sort()
        for (a0, a1, ia), (b0, b1, ib) in zip(spans, spans[1:]):
            if b0 < a1 - 0.05:
                errs.append(f"clips {ia} and {ib} ({src}) overlap: {a0}-{a1} vs {b0}-{b1}")
    return errs


def clip_duration(c: Clip) -> float:
    """The ON-SCREEN duration (what the timeline/total_duration/transition math uses) - for a
    video clip this already accounts for speed; `c.out - c.in_` alone is the RAW source range
    (used by overlap_errors, unaffected by speed)."""
    if c.in_ is not None:
        return (c.out - c.in_) / c.speed
    return c.duration or 2.5


def transition_errors(edl: EDL) -> list[str]:
    """A transition must be shorter than 40% of the clips it joins."""
    errs = []
    for i, c in enumerate(edl.clips):
        if c.transition_in is None:
            continue
        if i == 0:
            c.transition_in = None   # the first clip has nothing to transition from
            continue
        lim = 0.4 * min(clip_duration(edl.clips[i - 1]), clip_duration(c))
        if c.transition_in.duration > lim + 1e-6:
            errs.append(f"clip {i}: the {c.transition_in.duration}s transition is too long (max {lim:.2f}s)")
    return errs


def text_errors(edl: EDL) -> list[str]:
    total = total_duration(edl)
    errs = []
    for t in edl.texts:
        if t.start >= total - 0.3:
            errs.append(f"the text '{t.content}' starts after the end of the video ({total:.1f}s)")
        elif t.end > total:
            t.end = round(total, 2)   # silently trim whatever runs past the end
    # the same-position/overlapping-time check is a cheap stand-in for "would these two texts
    # visually collide" - it doesn't apply once a text has been manually, freely positioned: the
    # user placed it while watching a live preview, so trust their eyes over the heuristic.
    by_pos: dict[str, list[TextItem]] = {}
    for t in edl.texts:
        if t.x is not None or t.y is not None:
            continue
        by_pos.setdefault(t.position, []).append(t)
    for pos, items in by_pos.items():
        items.sort(key=lambda t: t.start)
        for a, b in zip(items, items[1:]):
            if b.start < a.end - 0.05:
                errs.append(f"the texts '{a.content}' and '{b.content}' overlap in time at the same position ({pos})")
    return errs


def sfx_errors(edl: EDL) -> list[str]:
    total = total_duration(edl)
    errs = []
    for s in edl.sfx:
        if s.start >= total - 0.05:
            errs.append(f"the {s.name!r} sound effect starts after the end of the video ({total:.1f}s)")
    return errs


def overlay_duration(o: OverlayItem) -> float:
    if o.in_ is not None:
        return o.out - o.in_
    return o.duration or 2.5


def overlay_errors(edl: EDL) -> list[str]:
    """Same spirit as text_errors: a window that starts past the end is a real error, one that only
    runs a bit past the end is silently trimmed to fit (no need to bother the user over a rounding
    edge from a clip they just shortened)."""
    total = total_duration(edl)
    errs = []
    for o in edl.overlays:
        if o.start >= total - 0.3:
            errs.append(f"the overlay {o.source!r} starts after the end of the video ({total:.1f}s)")
            continue
        over = o.start + overlay_duration(o) - total
        if over > 0:
            if o.in_ is not None:
                o.out = round(o.out - over, 2)
            else:
                o.duration = round((o.duration or 2.5) - over, 2)
    return errs


def clamp_transitions(edl: EDL) -> EDL:
    """After durations change (beat alignment), shorten transitions that no longer fit."""
    for i in range(1, len(edl.clips)):
        tr = edl.clips[i].transition_in
        if tr:
            lim = 0.4 * min(clip_duration(edl.clips[i - 1]), clip_duration(edl.clips[i]))
            if tr.duration > lim:
                tr.duration = round(max(0.15, lim), 2)
                if tr.duration > lim:          # the clip is too short even for 0.15s
                    edl.clips[i].transition_in = None
    return edl


def clamp_audio_trim(edl: EDL) -> EDL:
    """J/L cuts (Clip.audio_lead/audio_lag) can only borrow as much on-screen time as the
    neighboring clip actually has - and the first/last clip has no neighbor to borrow from at all.
    This is the structural cap (doesn't need the source file); render.py applies a second,
    tighter clamp against how much RAW source footage is actually available to extend into."""
    clips = edl.clips
    for i, c in enumerate(clips):
        if c.audio_lead:
            cap = clip_duration(clips[i - 1]) if i > 0 else 0.0
            c.audio_lead = round(min(c.audio_lead, cap), 2)
        if c.audio_lag:
            cap = clip_duration(clips[i + 1]) if i + 1 < len(clips) else 0.0
            c.audio_lag = round(min(c.audio_lag, cap), 2)
    return edl
