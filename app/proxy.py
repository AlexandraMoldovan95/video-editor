"""Low-res proxy transcodes, built in the background right after upload, so that anything
preview-only - AI-planning frames, the filmstrip, deep scan, the waveform - seeks/decodes a small
fast file instead of the original full-resolution upload. The proxy is swapped in purely by
`.path`; every other piece of metadata (duration, width/height, has_audio) still comes from the
ORIGINAL probe, so validation and timing stay exact regardless of which file is actually read.

IMPORTANT: render.py must NEVER use a proxy - the final export always reads the original file.
This module is only wired into preview/AI-frame code paths, never into the render pipeline.
"""
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path

from .probe import MediaInfo

PROXY_WIDTH = 640     # comfortably more than the 512px frames extract_frames() downscales to
PROXY_DIR_NAME = "proxies"


def proxy_dir(project_dir: Path) -> Path:
    return project_dir / PROXY_DIR_NAME


def proxy_path(project_dir: Path, name: str) -> Path:
    return proxy_dir(project_dir) / (Path(name).stem + ".mp4")


def build_proxy(src: Path, dst: Path) -> None:
    """Transcodes src down to a small, fast-seeking proxy (video + audio, so the waveform still
    works). Writes to a temp file first and renames, so a reader never sees a half-written proxy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"scale={PROXY_WIDTH}:-2:flags=fast_bilinear",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", "30",
        "-c:a", "aac", "-b:a", "96k",
        str(tmp),
    ], check=True)
    tmp.replace(dst)


def build_proxies_bg(project_dir: Path, media: dict[str, MediaInfo], names: list[str]) -> None:
    """Fire-and-forget: called in a background thread right after upload. Best-effort - any
    failure just means previews keep reading the original file (resolve_preview_path falls back
    automatically), never a hard error for the user."""
    for n in names:
        info = media.get(n)
        if not info or info.kind != "video":
            continue
        try:
            build_proxy(info.path, proxy_path(project_dir, n))
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def resolve_preview_path(info: MediaInfo, project_dir: Path) -> Path:
    """The file to read for anything preview-only. The proxy if it's ready, otherwise (still
    transcoding, transcode failed, or this is a photo - no proxy is built for those) the original."""
    if info.kind != "video":
        return info.path
    p = proxy_path(project_dir, info.path.name)
    return p if p.exists() else info.path


def preview_info(info: MediaInfo, project_dir: Path) -> MediaInfo:
    """A copy of `info` with only `.path` swapped to the proxy (when ready) - everything else
    (duration, width/height, has_audio) is kept from the original probe, so callers that check
    those fields (validation, timing) see the exact same values either way."""
    if info.kind != "video":
        return info
    return replace(info, path=resolve_preview_path(info, project_dir))


def preview_media(media: dict[str, MediaInfo], project_dir: Path) -> dict[str, MediaInfo]:
    """Same as preview_info(), applied to a whole media dict - for AI planning / deep scan, which
    take the dict as a whole. NEVER pass this to render.py."""
    return {n: preview_info(info, project_dir) for n, info in media.items()}
