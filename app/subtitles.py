"""Subtitles from the transcript, computed on the CURRENT TIMELINE (not stored in the plan).

Each word in a cut's [in, out] interval is moved to its time in the final video:
    t_final = the cut's start in the video + (t_word - in)
Words are grouped into short chunks (max ~32 characters / 2.8s), breaking at punctuation.
"""
from .edl import EDL, TextItem, clip_duration

MAX_CHARS = 24   # fits on one line at 9:16
MAX_DUR = 2.8
END_PUNCT = (".", "!", "?", "…")


def build(edl: EDL, transcripts: dict[str, dict]) -> list[TextItem]:
    subs: list[TextItem] = []
    t0 = 0.0
    for c in edl.clips:
        dur = clip_duration(c)
        tr = transcripts.get(c.source)
        if tr and c.in_ is not None and c.volume > 0:
            words = [w for w in tr.get("words", []) if w["start"] >= c.in_ - 0.05 and w["end"] <= c.out + 0.1]
            chunk: list[dict] = []

            def flush():
                if not chunk:
                    return
                st = t0 + max(0.0, chunk[0]["start"] - c.in_)
                en = min(t0 + dur, max(t0 + chunk[-1]["end"] - c.in_ + 0.15, st + 0.6))
                st = min(st, en - 0.5)          # TextItem requires at least 0.5s on screen
                text = " ".join(w["word"] for w in chunk).strip()
                if text:
                    subs.append(TextItem(content=text[:80], start=round(max(0.0, st), 2), end=round(en, 2),
                                         position="bottom", style="subtitle"))
                chunk.clear()

            for w in words:
                cand = " ".join(x["word"] for x in chunk + [w])
                if chunk and (len(cand) > MAX_CHARS or w["end"] - chunk[0]["start"] > MAX_DUR):
                    flush()
                chunk.append(w)
                if w["word"].endswith(END_PUNCT):
                    flush()
            flush()
        t0 += dur
    # a subtitle never overlaps the next one
    for a, b in zip(subs, subs[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.3, b.start)
    return subs


def prompt_block(transcripts: dict[str, dict], names: list[str]) -> str | None:
    lines = []
    for n in names:
        tr = transcripts.get(n)
        if tr and tr.get("segments"):
            segs = " ".join(f"[{s['start']}-{s['end']}s] {s['text']}" for s in tr["segments"])
            lines.append(f"- {n} (language: {tr.get('language')}): {segs}")
    if not lines:
        return None
    return ("## Speech transcript (Whisper) - times in the SOURCE clip\n" + "\n".join(lines) +
            "\nDon't cut in the middle of a sentence. Subtitles are added automatically at the BOTTOM: "
            "only place your own texts at the top or center while someone is speaking.")
