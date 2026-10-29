"""Music: decoding, BPM + beat detection (pure numpy, no librosa), energy curve,
aligning cuts to the beat.

Beat tracking algorithm (Ellis 2007, simplified):
  1. onset envelope = spectral flux on the log spectrogram
  2. tempo = peak of the onset's autocorrelation, weighted around 120 BPM
  3. beats = dynamic programming: maximizes the onset score on the beats, penalizing
     how far the interval between beats deviates from the tempo's period
"""
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .edl import EDL, Clip, clip_duration, overlap_errors

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
SR = 22050
N_FFT = 2048
HOP = 512


@dataclass
class MusicInfo:
    duration: float
    bpm: float
    beats: list[float]        # absolute times in the track (s)
    energy: list[float]       # 0..1, one value per second

    def to_dict(self) -> dict:
        return asdict(self)


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _band_matrix(n_bins: int, n_bands: int = 40, fmin: float = 30.0, fmax: float = 8000.0) -> np.ndarray:
    """Logarithmic frequency bands (like a mel scale): bass weighs as much as the highs."""
    freqs = np.linspace(0, SR / 2, n_bins)
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    M = np.zeros((n_bands, n_bins), dtype=np.float32)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        M[b] = np.maximum(0, np.minimum(up, down))
        M[b] /= M[b].sum() + 1e-9
    return M


def onset_envelope(y: np.ndarray) -> np.ndarray:
    y = np.pad(y, N_FFT // 2)   # centered windows: frame t <-> time t*HOP/SR (no offset)
    n_frames = 1 + (len(y) - N_FFT) // HOP
    if n_frames < 10:
        raise ValueError("the track is too short")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hanning(N_FFT)[None, :].astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    bands = mag @ _band_matrix(mag.shape[1]).T
    logb = np.log1p(1000 * bands)
    flux = np.maximum(0, np.diff(logb, axis=0)).mean(axis=1)
    flux = np.concatenate([[0], flux])
    # subtract the local average (~0.5s) -> only the peaks matter
    w = int(0.5 * SR / HOP)
    local = np.convolve(flux, np.ones(w) / w, mode="same")
    env = np.maximum(0, flux - local)
    return env / (env.max() + 1e-9)


def estimate_period(env: np.ndarray, bpm_min=60, bpm_max=200) -> float:
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    lo, hi = int(fps * 60 / bpm_max), int(fps * 60 / bpm_min)
    bpm = 60 * fps / np.maximum(lags[lo:hi], 1)
    weight = np.exp(-0.5 * (np.log2(bpm / 120) / 0.9) ** 2)   # preference toward ~120 BPM
    best = lo + int(np.argmax(ac[lo:hi] * weight))
    # sub-frame refinement (parabolic interpolation)
    if 0 < best < len(ac) - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        best = best + 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    return float(best)   # the period, in frames


def track_beats(env: np.ndarray, period: float, tightness: float = 100.0) -> np.ndarray:
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for t in range(hi, n):
        prev = np.arange(t - hi, t - lo + 1)
        pen = -tightness * np.log((t - prev) / period) ** 2
        cand = score[prev] + pen
        k = int(np.argmax(cand))
        score[t] = env[t] + cand[k]
        back[t] = prev[k]
    # start from the best beat in the last period
    t = int(n - hi + np.argmax(score[n - hi:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1])


def analyze(path: Path) -> MusicInfo:
    y = decode(path)
    env = onset_envelope(y)
    period = estimate_period(env)
    beats = track_beats(env, period) * HOP / SR
    # RMS energy per second, normalized
    sec = len(y) // SR
    rms = np.sqrt(np.mean(y[: sec * SR].reshape(sec, SR) ** 2, axis=1)) if sec else np.array([0.0])
    energy = rms / (rms.max() + 1e-9)
    return MusicInfo(duration=round(len(y) / SR, 2), bpm=round(60 * SR / HOP / period, 1),
                     beats=[round(float(b), 3) for b in beats], energy=[round(float(e), 2) for e in energy])


# ---------------- beat alignment ----------------
def beat_grid(info: MusicInfo, start: float) -> np.ndarray:
    """The beats relative to the track's start point in the video."""
    g = np.array([b - start for b in info.beats if b >= start - 1e-6])
    return g[g > 0.05]


def _candidates(t0: float, grid: np.ndarray, lo: float, hi: float, want: float) -> list[float]:
    ends = grid[(grid >= t0 + lo - 1e-6) & (grid <= t0 + hi + 1e-6)]
    return sorted((float(e - t0) for e in ends), key=lambda d: abs(d - want))


def snap_to_beats(edl: EDL, media: dict, grid: np.ndarray) -> EDL:
    """Moves the end of each cut to the nearest beat.

    - locked cuts keep their duration (the next one compensates)
    - video: 'out' moves first; if it goes past the clip, 'in' moves back
    - photos: duration stays within 2-4s
    - a candidate that would create an overlap with another interval on the same clip is rejected
    """
    t = 0.0
    for i, c in enumerate(edl.clips):
        # on-screen duration: a locked clip's own speed must not throw off where later clips land
        cur = clip_duration(c)
        if c.locked or len(grid) == 0:
            t += cur
            continue
        if c.in_ is None:
            cands = _candidates(t, grid, 2.0, 4.0, cur)
        else:
            src = media[c.source].duration
            cands = _candidates(t, grid, 0.5, min(src, cur * 2), cur)
        for d in cands:
            old = (c.in_, c.out, c.duration)
            if c.in_ is None:
                c.duration = round(d, 3)
            else:
                src = media[c.source].duration
                if c.in_ + d <= src:
                    c.out = round(c.in_ + d, 3)
                else:
                    c.in_ = round(max(0.0, src - d), 3)
                    c.out = round(c.in_ + d, 3)
            if not overlap_errors(edl):
                break
            c.in_, c.out, c.duration = old
        t += clip_duration(c)
    return edl


def prompt_block(info: MusicInfo, start: float, target: float) -> str:
    beat = 60 / info.bpm
    avail = info.duration - start
    e = info.energy[int(start): int(start + target) + 1]
    curve = "".join(str(min(9, int(v * 10))) for v in e)
    return (f"## Music\n- BPM {info.bpm} -> one beat = {beat:.3f}s. Make each cut a MULTIPLE of a beat "
            f"(1, 2, 4 or 8 beats), as close as possible to the requested length range.\n"
            f"- The track starts at second {start} and has {avail:.1f}s left"
            + (f" - WARNING: less than the target duration, don't exceed {avail:.1f}s" if avail < target else "") + ".\n"
            f"- Energy per second (0=quiet, 9=peak), starting at second 0 of the video: {curve}\n"
            f"  Put the most dynamic clips on the high-energy zones and longer cuts on the calm zones.")
