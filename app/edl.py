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
    position: Literal["top", "center", "bottom"] = "bottom"
    style: Literal["title", "caption", "cta", "subtitle"] = "caption"   # subtitle = generated from Whisper
    locked: bool = False

    @model_validator(mode="after")
    def _times(self):
        if self.end - self.start < 0.5:
            raise ValueError(f"the text '{self.content}' must stay on screen for at least 0.5s")
        return self


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
    locked: bool = False   # locked by the user: the AI won't change it on regeneration
    transition_in: Transition | None = None   # the transition FROM the previous clip INTO this one

    model_config = {"populate_by_name": True}

    @field_validator("source")
    @classmethod
    def only_filename(cls, v: str) -> str:
        # no paths: the AI can't request files outside the input folder
        if Path(v).name != v or v.startswith("."):
            raise ValueError(f"source must be a filename only: {v!r}")
        return v


class EDL(BaseModel):
    output: Output = Output()
    clips: list[Clip] = Field(min_length=1, max_length=60)
    texts: list[TextItem] = Field(default_factory=list, max_length=12)


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
    errors += overlap_errors(edl) + transition_errors(edl) + text_errors(edl)
    if errors:
        raise ValueError("Invalid EDL:\n  " + "\n  ".join(errors))
    return edl


def total_duration(edl: EDL) -> float:
    return sum((c.out - c.in_) if c.in_ is not None else c.duration for c in edl.clips)


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
    return (c.out - c.in_) if c.in_ is not None else (c.duration or 2.5)


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
    by_pos: dict[str, list[TextItem]] = {}
    for t in edl.texts:
        by_pos.setdefault(t.position, []).append(t)
    for pos, items in by_pos.items():
        items.sort(key=lambda t: t.start)
        for a, b in zip(items, items[1:]):
            if b.start < a.end - 0.05:
                errs.append(f"the texts '{a.content}' and '{b.content}' overlap in time at the same position ({pos})")
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
