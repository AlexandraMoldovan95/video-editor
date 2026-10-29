"""Per-file settings set by the user, BEFORE the AI.

Two kinds:
- CONSTRAINTS (use, segment): checked after the AI's response; if violated -> the AI corrects it.
- OVERRIDES (fit, fixed color, photo duration/motion): applied DETERMINISTICALLY by the code after
  the AI, so they are guaranteed. The AI still gets them in the prompt, to harmonize the rest.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .edl import EDL


class ColorLock(BaseModel):
    brightness: float = Field(0.0, ge=-0.3, le=0.3)
    contrast: float = Field(1.0, ge=0.5, le=2.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0)
    warmth: float = Field(0.0, ge=-0.2, le=0.2)


class FileHint(BaseModel):
    use: Literal["auto", "must", "exclude"] = "auto"
    segment_in: float | None = Field(None, ge=0)     # video only: the AI picks ONLY from this interval
    segment_out: float | None = Field(None, ge=0)
    fit: Literal["auto", "crop_center", "blur_pad"] = "auto"
    color: ColorLock | None = None                    # set = fixed
    photo_duration: float | None = Field(None, ge=2.0, le=4.0)
    photo_motion: Literal["auto", "zoom_in", "zoom_out", "static"] = "auto"
    volume: float | None = Field(None, ge=0.0, le=2.0)   # original sound volume (fixed)
    note: str = Field("", max_length=300)

    @model_validator(mode="after")
    def _segment_ok(self):
        if self.segment_in is not None and self.segment_out is not None and self.segment_in >= self.segment_out:
            raise ValueError("segment_in must be < segment_out")
        return self

    def is_default(self) -> bool:
        return self == FileHint()


def describe(name: str, h: FileHint) -> str | None:
    """The text that goes into the prompt for a file with settings."""
    if h.is_default():
        return None
    parts = []
    if h.use == "must":
        parts.append("MUST be used at least once")
    if h.segment_in is not None or h.segment_out is not None:
        parts.append(f"pick cuts ONLY from the interval {h.segment_in or 0}s-{h.segment_out or 'end'}s")
    if h.fit != "auto":
        parts.append(f"fit fixed: {h.fit}")
    if h.color:
        parts.append(f"color FIXED by the user: {h.color.model_dump()} - harmonize the other clips with it")
    if h.photo_duration:
        parts.append(f"fixed duration: {h.photo_duration}s")
    if h.photo_motion != "auto":
        parts.append(f"fixed motion: {h.photo_motion}")
    if h.volume is not None:
        parts.append(f"original sound volume fixed: {h.volume}")
    if h.note:
        parts.append(f"user's note: {h.note}")
    return f"- {name}: " + "; ".join(parts)


def check(edl: EDL, hints: dict[str, FileHint]) -> list[str]:
    errs = []
    used = {c.source for c in edl.clips}
    for name, h in hints.items():
        if h.use == "must" and name not in used:
            errs.append(f"{name} is marked as MUST but doesn't appear in the plan")
        if h.use == "exclude" and name in used:
            errs.append(f"{name} is EXCLUDED but appears in the plan")
    for i, c in enumerate(edl.clips):
        h = hints.get(c.source)
        if not h or c.in_ is None or c.locked:
            continue
        lo = h.segment_in or 0.0
        hi = h.segment_out if h.segment_out is not None else float("inf")
        if c.in_ < lo - 0.05 or c.out > hi + 0.05:
            errs.append(f"clip {i} ({c.source}): {c.in_}-{c.out} is outside the allowed interval {lo}-{hi}")
    return errs


def apply(edl: EDL, hints: dict[str, FileHint]) -> EDL:
    """Deterministically overwrites what the user fixed (doesn't touch locked clips)."""
    for c in edl.clips:
        h = hints.get(c.source)
        if not h or c.locked:
            continue
        if h.fit != "auto":
            c.fit = h.fit
        if h.volume is not None:
            c.volume = h.volume
        if h.color:
            c.brightness, c.contrast = h.color.brightness, h.color.contrast
            c.saturation, c.warmth = h.color.saturation, h.color.warmth
        if c.in_ is None:   # photo
            if h.photo_duration:
                c.duration = h.photo_duration
            if h.photo_motion != "auto":
                c.motion = h.photo_motion
    return edl
