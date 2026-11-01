"""Presets with fixed fields + translating them into concrete values for the AI/render."""
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Pace = Literal["slow", "medium", "fast"]
ColorLook = Literal["neutral", "warm", "cool", "contrast", "pastel"]


class Preset(BaseModel):
    target_duration: int = Field(30, ge=5, le=120)
    pace: Pace = "medium"
    color_look: ColorLook = "neutral"
    vertical_fit: Literal["crop_center", "blur_pad"] = "crop_center"
    photo_motion: Literal["zoom_in", "zoom_out", "static"] = "zoom_in"
    transition_style: Literal["cuts", "subtle", "dynamic"] = "cuts"
    text_style: Literal["none", "title_only", "story"] = "none"
    notes: str = Field("", max_length=500)


PRESETS: dict[str, Preset] = {
    "neutral": Preset(),
    "reel_energetic": Preset(target_duration=15, pace="fast", color_look="contrast",
                             transition_style="dynamic", text_style="title_only"),
    "vlog_calm": Preset(target_duration=45, pace="slow", color_look="warm", vertical_fit="blur_pad",
                        transition_style="subtle", text_style="story"),
    "travel_cinematic": Preset(target_duration=30, pace="medium", color_look="cool", vertical_fit="blur_pad",
                               transition_style="subtle", text_style="title_only"),
}

# length of a cut, in seconds (min, max)
PACE_RANGE: dict[str, tuple[float, float]] = {
    "slow": (3.0, 5.0),
    "medium": (2.0, 3.0),
    "fast": (0.8, 1.5),
}

# starting values; the AI can adjust per clip around them
COLOR_BASE: dict[str, dict[str, float]] = {
    "neutral":  {"brightness": 0.0,  "contrast": 1.0,  "saturation": 1.0,  "warmth": 0.0},
    "warm":     {"brightness": 0.02, "contrast": 1.05, "saturation": 1.1,  "warmth": 0.08},
    "cool":     {"brightness": 0.0,  "contrast": 1.08, "saturation": 0.95, "warmth": -0.08},
    "contrast": {"brightness": 0.0,  "contrast": 1.2,  "saturation": 1.15, "warmth": 0.0},
    "pastel":   {"brightness": 0.05, "contrast": 0.9,  "saturation": 0.8,  "warmth": 0.03},
}


# ---------- custom presets (user-saved, persisted server-side, global to the app) ----------
BUILTIN_KEYS = frozenset(PRESETS)  # snapshot before any custom preset is loaded/added
CUSTOM_PRESETS_FILE = Path(__file__).resolve().parent.parent / "custom_presets.json"
CUSTOM_KEYS: set[str] = set()
CUSTOM_META: dict[str, dict] = {}  # key -> {"label": str, "created": str}


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "preset"


def _write_custom_presets() -> None:
    data = {k: {"label": CUSTOM_META[k]["label"], "created": CUSTOM_META[k].get("created", ""),
                "preset": PRESETS[k].model_dump()} for k in CUSTOM_KEYS}
    CUSTOM_PRESETS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


def _load_custom_presets() -> None:
    if not CUSTOM_PRESETS_FILE.exists():
        return
    try:
        data = json.loads(CUSTOM_PRESETS_FILE.read_text("utf-8"))
    except Exception:  # noqa: BLE001 - corrupt file shouldn't crash the app
        return
    for key, entry in data.items():
        if key in BUILTIN_KEYS:
            continue  # never let a stale file shadow a built-in
        try:
            PRESETS[key] = Preset.model_validate(entry["preset"])
        except Exception:  # noqa: BLE001 - skip anything that no longer validates
            continue
        CUSTOM_META[key] = {"label": entry.get("label", key), "created": entry.get("created", "")}
        CUSTOM_KEYS.add(key)


_load_custom_presets()


def list_custom_presets() -> dict[str, dict]:
    return {k: {"label": CUSTOM_META[k]["label"], "created": CUSTOM_META[k].get("created", ""),
                **PRESETS[k].model_dump()} for k in CUSTOM_KEYS}


def add_custom_preset(label: str, preset: Preset, created: str = "") -> str:
    base = _slugify(label)
    key = base
    i = 2
    while key in PRESETS:  # covers both built-ins and existing customs
        key = f"{base}_{i}"
        i += 1
    PRESETS[key] = preset
    CUSTOM_META[key] = {"label": label, "created": created}
    CUSTOM_KEYS.add(key)
    _write_custom_presets()
    return key


def remove_custom_preset(key: str) -> bool:
    if key not in CUSTOM_KEYS:
        return False  # either unknown, or a built-in - never deletable
    del PRESETS[key]
    del CUSTOM_META[key]
    CUSTOM_KEYS.discard(key)
    _write_custom_presets()
    return True


def resolve(preset_name: str, overrides: dict | None = None,
            measured_pace: tuple[float, float] | None = None) -> tuple[Preset, tuple[float, float]]:
    """Priority: field set by the user > preset > rhythm measured from the example (pace only).

    Returns the final preset + the cut-length range.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset: {preset_name}. Available: {', '.join(PRESETS)}")
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    final = PRESETS[preset_name].model_copy(update=overrides)
    final = Preset.model_validate(final.model_dump())  # re-validate the overrides
    if measured_pace and "pace" not in overrides:
        cut_range = measured_pace
    else:
        cut_range = PACE_RANGE[final.pace]
    return final, cut_range


TRANSITION_RULES = {
    "cuts": "ONLY direct cuts: do NOT set transition_in on any clip.",
    "subtle": ("Subtle transitions (fade, dissolve, fadeblack), 0.3-0.6s, on at most 1/3 of the boundaries - "
               "the rest direct cuts. Use them for mood changes, not between similar clips."),
    "dynamic": ("Dynamic transitions (slideleft, slideup, wipeleft, smoothleft, circleopen, zoomin, fadewhite), "
                "0.2-0.4s, on at most half of the boundaries; vary the types, don't repeat the same one twice in a row."),
}
TEXT_RULES = {
    "none": "Do NOT add any text (texts = []).",
    "title_only": ("One short TITLE (max 6 words) in the first 3 seconds, style=title, position=center. "
                   "If the user's notes ask for a call-to-action, add it at the end with style=cta."),
    "story": ("A title at the start (style=title) + 2-4 short texts (max 8 words, style=caption, "
              "position=bottom) that tell the story, synced with the clips they refer to; "
              "each text stays on screen 2-4s; they don't overlap in time."),
}
