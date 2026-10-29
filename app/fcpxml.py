"""Export FCPXML 1.10 -> DaVinci Resolve (File > Import > Timeline) / Final Cut Pro.

Exports an EDITABLE timeline that uses the ORIGINAL files (not the rendered MP4):
  OK ordering, cuts (in/out), photo durations, framing (crop -> "fill", blurred -> "fit")
  OK transitions -> Cross Dissolve (DaVinci has no faithful equivalent for ffmpeg's slide/wipe/zoom)
  OK the original sound's volume, music (on the audio track, with volume and start point in the track)
  OK texts (Basic Title, on video track 2)
  NOT TRANSFERRED: color grading (eq) - redo it in DaVinci's Color page
  NOT TRANSFERRED: the photos' slow zoom, the blurred background, text styling, ducking

FCPXML rules worth knowing:
  - times are rational fractions ("90/30s"); the sequence is 30 fps -> everything in 1/30s frames
  - a connected clip (title, music) has its offset in the PARENT's LOCAL time:
        local_offset = parent_start + (t_in_sequence - parent_offset)
    they all connect to the first clip in the spine.
"""
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from .edl import EDL, TextItem, clip_duration
from .probe import MediaInfo

FPS = 30
DISSOLVE_UID = "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"   # Cross Dissolve (standard FCP effect)
TITLE_UID = ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
TITLE_SIZE = {"title": 110, "caption": 64, "cta": 72, "subtitle": 70}
TITLE_POS_Y = {"top": 700, "center": 0, "bottom": -560}   # FCP points, from center, on a 1920 height

def frames(sec: float, fps: int = FPS) -> int:
    return round(sec * fps)


def t_seq(nframes: int) -> str:
    return f"{nframes}/{FPS}s" if nframes else "0s"


def t_src(sec: float, fps: float | None) -> str:
    """Time in the source's own frame base (25/50/60/29.97...)."""
    if not sec:
        return "0s"
    if not fps or fps <= 0:
        return f"{round(sec * 1000)}/1000s"
    if abs(fps - round(fps)) < 0.01:
        f = round(fps)
        return f"{round(sec * f)}/{f}s"
    num = round(fps * 1001)                 # NTSC: 29.97 -> 30000/1001
    return f"{round(sec * fps) * 1001}/{num}s"


def db(volume: float) -> str:
    return "-96dB" if volume <= 0.001 else f"{20 * math.log10(volume):.1f}dB"


def build(edl: EDL, media: dict[str, MediaInfo], name: str = "Video Editor",
          music: dict | None = None, subtitles: list[TextItem] | None = None) -> bytes:
    W, H = edl.output.width, edl.output.height
    root = ET.Element("fcpxml", version="1.10")
    res = ET.SubElement(root, "resources")
    ET.SubElement(res, "format", id="r0", name=f"FFVideoFormat{W}x{H}p{FPS}",
                  frameDuration=f"1/{FPS}s", width=str(W), height=str(H), colorSpace="1-1-1 (Rec. 709)")

    # ---- resources: one format + one asset per file used ----
    asset_id, fmt_id = {}, {}
    used = list(dict.fromkeys(c.source for c in edl.clips))
    for i, src in enumerate(used, 1):
        m = media[src]
        fid = f"r{i}"
        if m.kind == "video":
            fps = m.fps or FPS
            fd = f"1/{round(fps)}s" if abs(fps - round(fps)) < 0.01 else f"1001/{round(fps * 1001)}s"
            ET.SubElement(res, "format", id=fid, frameDuration=fd, width=str(m.width), height=str(m.height))
        else:
            # photo: use the sequence's rate (not "RateUndefined" - some importers reject it)
            ET.SubElement(res, "format", id=fid, frameDuration=f"1/{FPS}s",
                          width=str(m.width), height=str(m.height))
        aid = f"a{i}"
        attrs = dict(id=aid, name=src, start="0s", hasVideo="1", format=fid,
                     duration=t_src(m.duration, m.fps) if m.kind == "video" else "0s")
        if m.kind == "video" and m.has_audio:
            attrs.update(hasAudio="1", audioSources="1", audioChannels="2", audioRate="48000")
        a = ET.SubElement(res, "asset", **attrs)
        ET.SubElement(a, "media-rep", kind="original-media", src=m.path.resolve().as_uri())
        asset_id[src], fmt_id[src] = aid, fid

    if music:
        mp = Path(music["path"])
        a = ET.SubElement(res, "asset", id="amusic", name=mp.name, start="0s",
                          duration=f"{round(music['duration'] * 1000)}/1000s",
                          hasAudio="1", audioSources="1", audioChannels="2", audioRate="48000")
        ET.SubElement(a, "media-rep", kind="original-media", src=mp.resolve().as_uri())

    if any(c.transition_in for c in edl.clips[1:]):
        ET.SubElement(res, "effect", id="ediss", name="Cross Dissolve", uid=DISSOLVE_UID)
    texts = list(edl.texts) + list(subtitles or [])
    if texts:
        ET.SubElement(res, "effect", id="etitle", name="Basic Title", uid=TITLE_UID)

    # ---- the sequence ----
    total = sum(frames(clip_duration(c)) for c in edl.clips)
    lib = ET.SubElement(root, "library")
    ev = ET.SubElement(lib, "event", name=name)
    proj = ET.SubElement(ev, "project", name=name)
    seq = ET.SubElement(proj, "sequence", format="r0", duration=t_seq(total), tcStart="0s",
                        tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    spine = ET.SubElement(seq, "spine")

    first_el, first_start_sec = None, 0.0
    pos = 0
    for i, c in enumerate(edl.clips):
        m = media[c.source]
        nf = frames(clip_duration(c))
        if i > 0 and c.transition_in:
            d = frames(c.transition_in.duration)
            tr = ET.SubElement(spine, "transition", name="Cross Dissolve",
                               offset=t_seq(max(0, pos - d // 2)), duration=t_seq(d))
            ET.SubElement(tr, "filter-video", ref="ediss", name="Cross Dissolve")
        if m.kind == "video":
            el = ET.SubElement(spine, "asset-clip", ref=asset_id[c.source], name=c.source,
                               offset=t_seq(pos), start=t_src(c.in_, m.fps), duration=t_seq(nf),
                               format=fmt_id[c.source], tcFormat="NDF")
            start_sec = c.in_
        else:
            # photo: a <clip> containing a <video> - the most compatible shape (DaVinci, OTIO, Premiere)
            el = ET.SubElement(spine, "clip", name=c.source, offset=t_seq(pos), start="0s",
                               duration=t_seq(nf), format="r0", tcFormat="NDF")
            ET.SubElement(el, "video", ref=asset_id[c.source], offset="0s", start="0s", duration=t_seq(nf))
            start_sec = 0.0
        ET.SubElement(el, "adjust-conform", type="fill" if c.fit == "crop_center" else "fit")
        if m.kind == "video" and m.has_audio:
            ET.SubElement(el, "adjust-volume", amount=db(c.volume))
        if first_el is None:
            first_el, first_start_sec = el, start_sec
        pos += nf

    def local(t_sec: float) -> str:
        """Offset for a clip connected to the first clip in the spine (see the module's docstring)."""
        return t_seq(frames(first_start_sec + t_sec))

    if music:
        mc = ET.SubElement(first_el, "asset-clip", ref="amusic", lane="-1", name=Path(music["path"]).name,
                           offset=local(0), start=f"{round(music['start'] * 1000)}/1000s",
                           duration=t_seq(min(total, frames(music["duration"] - music["start"]))),
                           audioRole="music")
        ET.SubElement(mc, "adjust-volume", amount=db(music["volume"]))

    for k, t in enumerate(sorted(texts, key=lambda x: x.start)):
        lane = "2" if t.style == "subtitle" else "1"
        ti = ET.SubElement(first_el, "title", ref="etitle", lane=lane, name=t.content[:30],
                           offset=local(t.start), start="0s", duration=t_seq(frames(t.end - t.start)))
        ET.SubElement(ti, "param", name="Position", key="9999/999166631/999166633/1/100/101",
                      value=f"0 {TITLE_POS_Y[t.position]}")
        txt = ET.SubElement(ti, "text")
        ET.SubElement(txt, "text-style", ref=f"ts{k}").text = t.content
        tsd = ET.SubElement(ti, "text-style-def", id=f"ts{k}")
        ET.SubElement(tsd, "text-style", font="Poppins", fontSize=str(TITLE_SIZE[t.style]),
                      fontFace="Bold", fontColor="1 1 1 1", bold="1", alignment="center")

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + body + "\n").encode("utf-8")
