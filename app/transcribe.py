"""Transcription with Whisper (faster-whisper) - OPTIONAL.

Install:  py -m pip install -r requirements-whisper.txt
On first use, the model downloads automatically (~460MB for "small"; "base" ~140MB, weaker on Romanian).
Other model: the VEDIT_WHISPER_MODEL env var (tiny / base / small / medium).
The result is cached (JSON), so each file is only transcribed once.
"""
import json
import os
from pathlib import Path

MODEL_NAME = os.getenv("VEDIT_WHISPER_MODEL", "small")
_model = None


def available() -> tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401
        return True, ""
    except ImportError:
        return False, "Whisper is not installed. Run: py -m pip install -r requirements-whisper.txt"


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model


def transcribe(path: Path, cache: Path, language: str | None = None) -> dict:
    """-> {"language", "segments": [{start, end, text}], "words": [{start, end, word}]}"""
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    ok, msg = available()
    if not ok:
        raise RuntimeError(msg)
    segments, info = _get_model().transcribe(str(path), language=language, vad_filter=True,
                                             word_timestamps=True, beam_size=5)
    segs, words = [], []
    for s in segments:
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()})
    out = {"language": info.language, "model": MODEL_NAME, "segments": segs, "words": words}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out
