"""Tests for app/edl.py: the code that checks the AI's edit plan before anything is rendered.

The planner asks Claude for a JSON edit plan (EDL) and never trusts it: validate_edl() must
reject a plan that points outside the project, at a missing file, at an impossible time range,
or with values outside the safe ranges, and it must quietly fix the small things it can.
Each test below feeds it one kind of bad plan a model could plausibly return.

Run from the repo root:  python -m pytest -q
"""
from pathlib import Path

import pytest

from app.edl import EDL, clamp_transitions, clip_duration, total_duration, validate_edl
from app.probe import MediaInfo


def video(name: str, duration: float) -> MediaInfo:
    return MediaInfo(path=Path(name), kind="video", width=1080, height=1920, fps=30.0,
                     duration=duration, has_audio=True)


def photo(name: str) -> MediaInfo:
    return MediaInfo(path=Path(name), kind="image", width=1080, height=1350, fps=None,
                     duration=None, has_audio=False)


@pytest.fixture
def media() -> dict[str, MediaInfo]:
    # what ffprobe would have reported for the project's input folder
    return {"a.mp4": video("a.mp4", 10.0), "b.mp4": video("b.mp4", 6.0), "p.jpg": photo("p.jpg")}


def plan(*clips: dict, **extra) -> dict:
    return {"clips": list(clips), **extra}


# ---------- a good plan passes, and small gaps are filled in ----------

def test_valid_plan_passes_and_photo_gets_default_duration(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 1.0, "out": 4.0},
                            {"source": "p.jpg"},
                            {"source": "b.mp4", "in": 0.0, "out": 2.5}), media)
    assert isinstance(edl, EDL)
    assert edl.clips[1].duration == 2.5            # photo without a duration: the safe default
    assert total_duration(edl) == pytest.approx(3.0 + 2.5 + 2.5)


# ---------- the AI can't reach files outside the project folder ----------

@pytest.mark.parametrize("source", ["../secret.mp4", "C:/Users/someone/a.mp4", "sub/a.mp4", ".env"])
def test_source_must_be_a_bare_filename(media, source):
    with pytest.raises(ValueError, match="filename only"):
        validate_edl(plan({"source": source, "in": 0, "out": 2}), media)


def test_unknown_file_is_rejected(media):
    with pytest.raises(ValueError, match="file does not exist 'missing.mp4'"):
        validate_edl(plan({"source": "missing.mp4", "in": 0, "out": 2}), media)


# ---------- impossible time ranges ----------

def test_cut_past_the_end_of_the_source_is_rejected(media):
    # b.mp4 is 6 s long; the model asked for up to 9 s
    with pytest.raises(ValueError, match="invalid interval 2.0-9.0"):
        validate_edl(plan({"source": "b.mp4", "in": 2.0, "out": 9.0}), media)


def test_in_after_out_is_rejected(media):
    with pytest.raises(ValueError, match="invalid interval"):
        validate_edl(plan({"source": "a.mp4", "in": 5.0, "out": 3.0}), media)


def test_cut_shorter_than_half_a_second_is_rejected(media):
    with pytest.raises(ValueError, match="cut too short"):
        validate_edl(plan({"source": "a.mp4", "in": 1.0, "out": 1.3}), media)


def test_video_clip_without_in_and_out_is_rejected(media):
    with pytest.raises(ValueError, match="video without 'in'/'out'"):
        validate_edl(plan({"source": "a.mp4"}), media)


def test_same_footage_used_twice_overlapping_is_rejected(media):
    with pytest.raises(ValueError, match="overlap"):
        validate_edl(plan({"source": "a.mp4", "in": 1.0, "out": 4.0},
                          {"source": "a.mp4", "in": 3.0, "out": 6.0}), media)


def test_same_footage_used_twice_without_overlap_is_fine(media):
    validate_edl(plan({"source": "a.mp4", "in": 1.0, "out": 3.0},
                      {"source": "a.mp4", "in": 3.0, "out": 6.0}), media)


# ---------- values outside the safe ranges ----------

@pytest.mark.parametrize("field, value", [("brightness", 0.9), ("contrast", 5.0), ("speed", 10.0), ("warmth", -1.0)])
def test_out_of_range_color_and_speed_values_are_rejected(media, field, value):
    with pytest.raises(ValueError):
        validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3, field: value}), media)


def test_speed_on_a_photo_is_rejected(media):
    with pytest.raises(ValueError, match="speed only applies to video"):
        validate_edl(plan({"source": "p.jpg", "speed": 2.0}), media)


def test_more_than_60_clips_is_rejected(media):
    clips = [{"source": "p.jpg"} for _ in range(61)]
    with pytest.raises(ValueError):
        validate_edl(plan(*clips), media)


# ---------- transitions ----------

def test_transition_longer_than_40_percent_of_the_shorter_clip_is_rejected(media):
    # clips of 2.0 s and 3.0 s: the limit is 0.8 s
    with pytest.raises(ValueError, match="transition is too long"):
        validate_edl(plan({"source": "a.mp4", "in": 0, "out": 2.0},
                          {"source": "b.mp4", "in": 0, "out": 3.0,
                           "transition_in": {"type": "fade", "duration": 1.0}}), media)


def test_transition_on_the_first_clip_is_dropped(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3.0,
                             "transition_in": {"type": "fade", "duration": 0.4}}), media)
    assert edl.clips[0].transition_in is None   # nothing to transition from


def test_unknown_transition_name_is_rejected(media):
    # e.g. "zoomin" isn't on the list because older ffmpeg builds don't have it
    with pytest.raises(ValueError):
        validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3},
                          {"source": "b.mp4", "in": 0, "out": 3,
                           "transition_in": {"type": "zoomin", "duration": 0.4}}), media)


def test_clamp_transitions_shortens_or_drops_after_durations_change(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3.0},
                            {"source": "b.mp4", "in": 0, "out": 3.0,
                             "transition_in": {"type": "fade", "duration": 1.0}}), media)
    edl.clips[1].out = 1.5                         # beat alignment shortened the clip
    clamp_transitions(edl)
    assert edl.clips[1].transition_in.duration == pytest.approx(0.6)   # 40% of 1.5 s


# ---------- on-screen text ----------

def test_text_starting_after_the_end_is_rejected(media):
    with pytest.raises(ValueError, match="starts after the end"):
        validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3.0},
                          texts=[{"content": "Late", "start": 5.0, "end": 6.0}]), media)


def test_text_running_past_the_end_is_trimmed(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 0, "out": 3.0},
                            texts=[{"content": "Hi", "start": 1.0, "end": 9.0}]), media)
    assert edl.texts[0].end == pytest.approx(3.0)


def test_two_texts_at_the_same_place_and_time_are_rejected(media):
    with pytest.raises(ValueError, match="overlap in time"):
        validate_edl(plan({"source": "a.mp4", "in": 0, "out": 6.0},
                          texts=[{"content": "One", "start": 0.5, "end": 3.0},
                                 {"content": "Two", "start": 2.0, "end": 4.0}]), media)


# ---------- J/L cuts can't borrow more than the neighbor has ----------

def test_audio_lead_and_lag_are_clamped_to_the_neighbors(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 0, "out": 2.0, "audio_lead": 1.0},
                            {"source": "b.mp4", "in": 0, "out": 3.0, "audio_lead": 4.0, "audio_lag": 4.0}), media)
    assert edl.clips[0].audio_lead == 0.0          # the first clip has nothing before it
    assert edl.clips[1].audio_lead == 2.0          # capped at the previous clip's 2.0 s
    assert edl.clips[1].audio_lag == 0.0           # the last clip has nothing after it


def test_on_screen_duration_accounts_for_speed(media):
    edl = validate_edl(plan({"source": "a.mp4", "in": 0, "out": 4.0, "speed": 2.0}), media)
    assert clip_duration(edl.clips[0]) == pytest.approx(2.0)
