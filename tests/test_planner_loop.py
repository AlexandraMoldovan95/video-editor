"""Tests for the request loop in app/planner.py: what happens with each kind of reply the AI
can send back, for the edit plan (call_claude) and for the deep scan of one clip
(deep_scan_clip).

No real API call is made. A fake client hands back prepared replies in order and records
every request, and make_client() is replaced so that a test trying to reach the real API
fails at once. No ffmpeg either: the deep scan's frame extraction is replaced by a stub.

Run from the repo root:  python -m pytest -q
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import planner
from app.edl import Clip
from app.probe import MediaInfo


# ---------- a fake Anthropic client and its replies ----------

class FakeClient:
    """Stands in for anthropic.Anthropic(): client.messages.create(...) returns the prepared
    replies one by one and keeps a copy of every request it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if not self.replies:
            raise AssertionError("the code asked the AI more times than the test prepared for")
        return self.replies.pop(0)


def reply(*blocks, stop="tool_use", input_tokens=100, output_tokens=50):
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def tool_call(data, id="toolu_1", name="submit_edl"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=data)


def text(t):
    return SimpleNamespace(type="text", text=t)


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch):
    def refuse():
        raise AssertionError("a test tried to create a real Anthropic client")
    monkeypatch.setattr(planner, "make_client", refuse)


# ---------- the project: what ffprobe would have reported ----------

def video(name, duration):
    return MediaInfo(path=Path(name), kind="video", width=1080, height=1920, fps=30.0,
                     duration=duration, has_audio=True)


@pytest.fixture
def media():
    photo = MediaInfo(path=Path("p.jpg"), kind="image", width=1080, height=1350, fps=None,
                      duration=None, has_audio=False)
    return {"a.mp4": video("a.mp4", 10.0), "b.mp4": video("b.mp4", 6.0), "p.jpg": photo}


ASK = [{"role": "user", "content": "the material, the preset and the rules"}]
GOOD_PLAN = {"reasoning": "Open on the wide shot, end on the close-up.",
             "clips": [{"source": "a.mp4", "in": 1.0, "out": 4.0}, {"source": "b.mp4", "in": 0.0, "out": 2.5}]}
BAD_PLAN = {"reasoning": "x", "clips": [{"source": "missing.mp4", "in": 0.0, "out": 2.0}]}


# ---------- the edit plan: call_claude() ----------

def test_a_valid_plan_on_the_first_reply_is_used_as_is(media):
    client = FakeClient(reply(tool_call(GOOD_PLAN)))
    edl, reasoning, usage = planner.call_claude(ASK, media, {}, [], client=client)
    assert [c.source for c in edl.clips] == ["a.mp4", "b.mp4"]
    assert reasoning == GOOD_PLAN["reasoning"]
    assert usage == {"input_tokens": 100, "output_tokens": 50}
    sent = client.requests[0]
    assert len(client.requests) == 1
    assert sent["model"] == planner.MODEL
    # The model thinks by default and thinking counts toward max_tokens: the old 4000 could
    # cut the plan off before the tool call.
    assert sent["max_tokens"] >= 16000
    assert sent["tool_choice"] == {"type": "auto"}   # this model rejects a forced tool_choice
    assert [t["name"] for t in sent["tools"]] == ["submit_edl"]


def test_a_text_reply_gets_one_reminder_then_the_plan_is_used(media):
    client = FakeClient(reply(text("Here is my plan: ..."), stop="end_turn"), reply(tool_call(GOOD_PLAN)))
    edl, _, usage = planner.call_claude(ASK, media, {}, [], client=client)
    assert len(edl.clips) == 2
    assert len(client.requests) == 2
    second = client.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"             # its text reply is kept in the conversation
    assert second[-1] == {"role": "user", "content": "Send the plan ONLY through the submit_edl tool."}
    assert usage == {"input_tokens": 200, "output_tokens": 100}   # both replies are counted


def test_two_text_replies_stop_with_a_clear_error(media):
    client = FakeClient(reply(text("plan..."), stop="end_turn"), reply(text("plan again..."), stop="end_turn"))
    with pytest.raises(RuntimeError, match="did not call submit_edl"):
        planner.call_claude(ASK, media, {}, [], client=client)
    assert len(client.requests) == 2


def test_an_invalid_plan_is_sent_back_once_with_the_reason(media):
    client = FakeClient(reply(tool_call(BAD_PLAN, id="toolu_bad")), reply(tool_call(GOOD_PLAN, id="toolu_good")))
    edl, _, _ = planner.call_claude(ASK, media, {}, [], client=client)
    assert [c.source for c in edl.clips] == ["a.mp4", "b.mp4"]
    correction = client.requests[1]["messages"][-1]["content"][0]
    assert correction["type"] == "tool_result"
    assert correction["tool_use_id"] == "toolu_bad"      # the answer is tied to the AI's own call
    assert correction["is_error"] is True
    assert "file does not exist 'missing.mp4'" in correction["content"]


def test_two_tool_calls_in_one_reply_each_get_an_answer_in_the_correction(media):
    # The API rejects a turn that leaves a tool_use without its tool_result (HTTP 400),
    # so the correction must answer both calls, not only the one that was checked.
    client = FakeClient(reply(tool_call(BAD_PLAN, id="toolu_a"), tool_call(GOOD_PLAN, id="toolu_b")),
                        reply(tool_call(GOOD_PLAN, id="toolu_c")))
    planner.call_claude(ASK, media, {}, [], client=client)
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert all(r["type"] == "tool_result" and r["is_error"] is True for r in results)
    assert "missing.mp4" in results[0]["content"]
    assert results[1]["content"] == "Ignored: send a single call."


def test_a_plan_still_invalid_after_the_correction_raises_the_validation_error(media):
    client = FakeClient(reply(tool_call(BAD_PLAN)), reply(tool_call(BAD_PLAN, id="toolu_2")))
    with pytest.raises(ValueError, match="missing.mp4"):
        planner.call_claude(ASK, media, {}, [], client=client)
    assert len(client.requests) == 2


def test_a_reply_cut_off_at_the_token_limit_is_reported_without_using_the_retry(media):
    half_plan = tool_call({"reasoning": "Open on", "clips": []})
    client = FakeClient(reply(half_plan, stop="max_tokens", output_tokens=16000))
    with pytest.raises(RuntimeError, match=r"cut off at the token limit \(16000 output tokens\)"):
        planner.call_claude(ASK, media, {}, [], client=client)
    assert len(client.requests) == 1


def test_a_refusal_is_reported_and_not_retried(media):
    client = FakeClient(reply(text(""), stop="refusal"))
    with pytest.raises(RuntimeError, match="declined to produce the edit plan"):
        planner.call_claude(ASK, media, {}, [], client=client)
    assert len(client.requests) == 1


def test_a_locked_clip_goes_back_to_its_place_in_the_timeline(media):
    # the user locked the 2nd clip; the AI only plans the free ones around it
    locked = [(1, Clip.model_validate({"source": "b.mp4", "in": 0.0, "out": 2.0, "locked": True}))]
    plan = {"reasoning": "x", "clips": [{"source": "a.mp4", "in": 1.0, "out": 4.0},
                                        {"source": "a.mp4", "in": 5.0, "out": 8.0}]}
    edl, _, _ = planner.call_claude(ASK, media, {}, locked, client=FakeClient(reply(tool_call(plan))))
    assert [(c.source, c.in_, c.locked) for c in edl.clips] == [
        ("a.mp4", 1.0, False), ("b.mp4", 0.0, True), ("a.mp4", 5.0, False)]


# ---------- the deep scan of one clip: deep_scan_clip() ----------

@pytest.fixture
def no_frames(monkeypatch):
    # the real request extracts 20 frames with ffmpeg; the loop doesn't need them
    def fake_request(info, clip, frames_dir):
        return [{"role": "user", "content": [{"type": "text", "text": "20 frames of the clip"}]}], {"images": 0}
    monkeypatch.setattr(planner, "build_deepscan_request", fake_request)


CLIP = {"source": "a.mp4", "in": 2.0, "out": 5.0, "reason": "first pass", "low_confidence": True}


def answer(start, end, id="toolu_1"):
    return tool_call({"clip": {"in": start, "out": end, "brightness": 0.05, "contrast": 1.1, "saturation": 1.2,
                               "warmth": 0.02, "reason": "Starts right on the jump.", "low_confidence": False}},
                     id=id, name="submit_clip")


def test_deep_scan_returns_the_refined_clip_locked(media, no_frames, tmp_path):
    client = FakeClient(reply(answer(2.346, 5.5)))
    updated, usage = planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    assert (updated.source, updated.in_, updated.out) == ("a.mp4", 2.35, 5.5)   # rounded to 1/100 s
    assert (updated.brightness, updated.contrast, updated.saturation, updated.warmth) == (0.05, 1.1, 1.2, 0.02)
    assert updated.locked is True                # a regeneration won't undo the closer look
    assert updated.low_confidence is False
    assert usage == {"input_tokens": 100, "output_tokens": 50}
    sent = client.requests[0]
    assert sent["max_tokens"] >= 4000            # was 1000: no room for thinking plus the answer
    assert [t["name"] for t in sent["tools"]] == ["submit_clip"]


def test_deep_scan_cut_past_the_end_of_the_file_is_sent_back_once(media, no_frames, tmp_path):
    # a.mp4 is 10 s long
    client = FakeClient(reply(answer(2.0, 12.0, id="toolu_bad")), reply(answer(2.0, 5.0, id="toolu_good")))
    updated, _ = planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    assert updated.out == 5.0
    correction = client.requests[1]["messages"][-1]["content"][0]
    assert correction["tool_use_id"] == "toolu_bad" and correction["is_error"] is True
    assert "invalid interval" in correction["content"]


def test_deep_scan_two_tool_calls_each_get_an_answer_in_the_correction(media, no_frames, tmp_path):
    client = FakeClient(reply(answer(2.0, 12.0, id="toolu_a"), answer(2.0, 5.0, id="toolu_b")),
                        reply(answer(2.0, 5.0, id="toolu_c")))
    planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert "invalid interval" in results[0]["content"]


def test_deep_scan_cut_too_short_twice_raises(media, no_frames, tmp_path):
    client = FakeClient(reply(answer(2.0, 2.3)), reply(answer(2.0, 2.2, id="toolu_2")))
    with pytest.raises(ValueError, match="cut too short"):
        planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    assert len(client.requests) == 2


def test_deep_scan_text_reply_gets_one_reminder(media, no_frames, tmp_path):
    client = FakeClient(reply(text("I'd move it a bit."), stop="end_turn"), reply(answer(2.5, 5.5)))
    updated, _ = planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    assert updated.in_ == 2.5
    assert client.requests[1]["messages"][-1] == {
        "role": "user", "content": "Send your answer ONLY through the submit_clip tool."}


def test_deep_scan_cut_off_reply_is_reported_without_using_the_retry(media, no_frames, tmp_path):
    client = FakeClient(reply(text(""), stop="max_tokens", output_tokens=4000))
    with pytest.raises(RuntimeError, match="deep-scan answer was cut off at the token limit"):
        planner.deep_scan_clip(media, Clip.model_validate(CLIP), tmp_path, client=client)
    assert len(client.requests) == 1


def test_deep_scan_refuses_a_photo_before_asking_the_ai(media, no_frames, tmp_path):
    client = FakeClient()
    photo_clip = Clip.model_validate({"source": "p.jpg", "duration": 3.0})
    with pytest.raises(ValueError, match="not a photo"):
        planner.deep_scan_clip(media, photo_clip, tmp_path, client=client)
    assert client.requests == []
