"""Rule-based locks: instead of locking clips one by one, the user defines a small set of
persistent rules - "keep the opening Ns", "never touch clips from file X" - that get
re-applied to the clips on the timeline every time, including right before every AI
regeneration. A rule only ADDS locks (on top of whatever's already locked by hand); it never
unlocks anything, so manually unlocking a clip the rule would otherwise catch is still possible -
the rule just re-catches it again on the next apply/regenerate, which is the intended behavior
(it's a standing rule, not a one-shot action)."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .edl import EDL, clip_duration


class LockRule(BaseModel):
    type: Literal["opening_seconds", "source"]
    seconds: float | None = Field(None, ge=0.1, le=600)   # type == "opening_seconds"
    source: str | None = Field(None, min_length=1)         # type == "source": exact filename

    @model_validator(mode="after")
    def _fields_match_type(self):
        if self.type == "opening_seconds" and self.seconds is None:
            raise ValueError("opening_seconds rule needs 'seconds'")
        if self.type == "source" and not self.source:
            raise ValueError("source rule needs 'source'")
        return self

    def label(self) -> str:
        if self.type == "opening_seconds":
            return f"keep the opening {self.seconds:g}s"
        return f"never touch clips from {self.source}"


def apply_lock_rules(edl: EDL, rules: list[LockRule]) -> EDL:
    """Locks whichever CURRENT clips match a rule, in place. Returns the same edl for chaining."""
    if not rules:
        return edl
    acc = 0.0
    for c in edl.clips:
        start = acc
        acc += clip_duration(c)
        for r in rules:
            if r.type == "opening_seconds" and r.seconds is not None and start < r.seconds:
                c.locked = True
            elif r.type == "source" and r.source and c.source == r.source:
                c.locked = True
    return edl
