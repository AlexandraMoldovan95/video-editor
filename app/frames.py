"""Extracts frames for the AI + measures the cut rhythm from an example video."""
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .probe import MediaInfo

FRAMES_PER_CLIP = 4
FRAME_WIDTH = 512


@dataclass
class Frame:
    source: str
    timestamp: float | None   # None for photos
    path: Path


def thumbnail(info: MediaInfo, dst: Path, width: int = 240) -> Path:
    t = min(1.0, (info.duration or 0) / 2) if info.kind == "video" else 0
    args = ["-ss", f"{t}"] if info.kind == "video" else []
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, "-i", str(info.path),
                    "-vf", f"scale={width}:-2", "-q:v", "5", "-frames:v", "1", str(dst)], check=True)
    return dst


def extract_frames(info: MediaInfo, out_dir: Path, n: int = FRAMES_PER_CLIP,
                   width: int = FRAME_WIDTH, t0: float | None = None,
                   t1: float | None = None) -> list[Frame]:
    """n frames at the midpoint of each quarter of [t0, t1] (the whole clip by default)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    scale = f"scale={width}:-2"
    if info.kind == "image":
        dst = out_dir / f"{stem}.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(info.path), "-vf", scale,
                        "-q:v", "4", "-frames:v", "1", str(dst)], check=True)
        return [Frame(info.path.name, None, dst)]

    t0 = t0 or 0.0
    t1 = min(t1, info.duration) if t1 is not None else info.duration
    frames = []
    for i in range(n):
        t = round(t0 + (i + 0.5) * (t1 - t0) / n, 2)   # the midpoint of each quarter
        dst = out_dir / f"{stem}_{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t}", "-i", str(info.path),
                        "-vf", scale, "-q:v", "4", "-frames:v", "1", str(dst)], check=True)
        frames.append(Frame(info.path.name, t, dst))
    return frames


@dataclass
class Rhythm:
    cuts: list[float]
    shot_lengths: list[float]
    median: float
    pace_range: tuple[float, float]


def measure_rhythm(info: MediaInfo, threshold: float = 0.3) -> Rhythm:
    """Scene detection with ffmpeg -> shot lengths from the example."""
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(info.path), "-an",
         "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    cuts = [float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    bounds = [0.0] + cuts + [info.duration]
    shots = [round(b - a, 2) for a, b in zip(bounds, bounds[1:]) if b - a > 0.2]
    if len(shots) < 2:
        raise ValueError(f"{info.path.name}: no cuts detected - is this actually an edited video?")
    med = statistics.median(shots)
    return Rhythm(cuts, shots, round(med, 2), (round(med * 0.75, 2), round(med * 1.25, 2)))
