"""Reads a media file's metadata with ffprobe."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class MediaInfo:
    path: Path
    kind: str            # "video" | "image"
    width: int
    height: int
    fps: float | None
    duration: float | None
    has_audio: bool


def _parse_fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def probe(path: str | Path) -> MediaInfo:
    path = Path(path).resolve()   # absolute: some ffmpeg commands run from the temp folder
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    vstreams = [s for s in data["streams"] if s["codec_type"] == "video"]
    if not vstreams:
        raise ValueError(f"{path.name}: has no video/image stream")
    v = vstreams[0]
    is_image = path.suffix.lower() in IMAGE_EXT
    return MediaInfo(
        path=path,
        kind="image" if is_image else "video",
        width=int(v["width"]),
        height=int(v["height"]),
        fps=None if is_image else _parse_fps(v.get("avg_frame_rate", "0/1")),
        duration=None if is_image else float(data["format"].get("duration", 0)),
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(probe(f))
