"""
WAV → variable-LPI preview.

Groove width is dominated by low-frequency lateral displacement. Adjacent grooves
share a wall, so pitch must open one revolution *before* a loud passage arrives.

At 33⅓ RPM one revolution is 1.800 s. That duration is the lookahead window, not
the analysis bin size: the envelope is computed at ~20 ms hops, then a sliding
max over the next revolution decides LPI. The resulting .idx is sparse
(time, lpi) change events, matching python-controller/lpi_manager.py.

G2 L values: higher L = slower carriage = tighter pitch.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

IdxEvent = Tuple[float, int]


@dataclass
class PreviewConfig:
    rpm: float = 33.333
    hop_ms: float = 20.0
    lpf_hz: float = 500.0
    lpi_tight: int = 6114
    lpi_open: int = 2048
    reference_peak: float = 0.40
    quantize: int = 64
    min_hold_ms: float = 100.0
    slew_ms: float = 250.0
    lookahead_revs: float = 1.0
    two_rev: bool = False


@dataclass
class WavData:
    path: str
    sample_rate: int
    channels: np.ndarray  # float32, shape (n_samples, n_channels)
    mid: np.ndarray  # float32 mono mid / (L+R)/2


@dataclass
class PreviewResult:
    config: PreviewConfig
    duration_s: float
    revolution_s: float
    hop_s: float
    envelope_times: np.ndarray
    envelope: np.ndarray
    lookahead: np.ndarray
    combined: np.ndarray
    lpi_dense: np.ndarray
    idx_events: List[IdxEvent] = field(default_factory=list)

    @property
    def n_hops(self) -> int:
        return int(self.envelope_times.shape[0])


def revolution_period_s(rpm: float) -> float:
    if rpm <= 0:
        raise ValueError(f'rpm must be positive, got {rpm}')
    return 60.0 / rpm


def load_wav(path: str | Path) -> WavData:
    path = str(path)
    sample_rate, data = wavfile.read(path)
    if data.size == 0:
        raise ValueError(f'WAV file is empty: {path}')

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data = _to_float32(data)
    if data.shape[1] == 1:
        mid = data[:, 0].copy()
    else:
        mid = data[:, :2].mean(axis=1)

    return WavData(path=path, sample_rate=int(sample_rate), channels=data, mid=mid)


def _to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    info = np.iinfo(data.dtype)
    peak = float(max(abs(info.min), info.max))
    return (data.astype(np.float32) / peak)


def lowpass_mid(mid: np.ndarray, sample_rate: int, lpf_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if lpf_hz <= 0 or lpf_hz >= nyquist:
        return mid
    sos = butter(4, lpf_hz / nyquist, btype='low', output='sos')
    return sosfiltfilt(sos, mid).astype(np.float32)


def peak_envelope(samples: np.ndarray, sample_rate: int, hop_ms: float) -> Tuple[np.ndarray, np.ndarray, float]:
    if hop_ms <= 0:
        raise ValueError(f'hop_ms must be positive, got {hop_ms}')
    hop_samples = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    hop_s = hop_samples / float(sample_rate)
    n = int(samples.shape[0])
    n_hops = int(np.ceil(n / hop_samples))
    padded_len = n_hops * hop_samples
    if padded_len != n:
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:n] = samples
    else:
        padded = samples
    frames = np.abs(padded).reshape(n_hops, hop_samples)
    envelope = frames.max(axis=1).astype(np.float32)
    times = np.arange(n_hops, dtype=np.float64) * hop_s
    return times, envelope, hop_s


def forward_max(values: np.ndarray, window: int, offset: int = 0) -> np.ndarray:
    """Max of values[i+offset : i+offset+window) for each i (lookahead)."""
    n = int(values.shape[0])
    window = max(1, int(window))
    offset = max(0, int(offset))
    padded = np.concatenate([
        values[offset:] if offset else values,
        np.zeros(window + offset, dtype=values.dtype),
    ])
    return sliding_window_view(padded, window)[:n].max(axis=1).astype(values.dtype, copy=False)


def map_combined_to_lpi(combined: np.ndarray, config: PreviewConfig) -> np.ndarray:
    tight = float(config.lpi_tight)
    open_l = float(config.lpi_open)
    lo = min(tight, open_l)
    hi = max(tight, open_l)
    # combined is envelope + lookahead, so ~0..2 * peak. reference_peak on
    # both sides saturates to fully open. If this track never reaches that,
    # stretch so the actual peak becomes Open LPI and Tight stays put —
    # a range fit, not a DC offset of the whole curve.
    ref_scale = 2.0 * max(config.reference_peak, 1e-6)
    peak = float(np.max(combined)) if combined.size else 0.0
    denom = min(max(peak, 1e-6), ref_scale)
    norm = np.clip(combined / denom, 0.0, 1.0)
    lpi = tight - norm * (tight - open_l)
    quantize = max(1, int(config.quantize))
    lpi = np.round(lpi / quantize) * quantize
    return np.clip(lpi, lo, hi).astype(np.int32)


def apply_slew(lpi: np.ndarray, hop_s: float, slew_ms: float, lpi_span: float) -> np.ndarray:
    if slew_ms <= 0 or lpi.shape[0] < 2:
        return lpi
    max_step = max(1.0, abs(lpi_span) * (hop_s * 1000.0) / slew_ms)
    out = lpi.astype(np.float64).copy()
    for i in range(1, out.shape[0]):
        delta = out[i] - out[i - 1]
        if abs(delta) > max_step:
            out[i] = out[i - 1] + np.sign(delta) * max_step
    return np.round(out).astype(np.int32)


def apply_min_hold(lpi: np.ndarray, min_hops: int) -> np.ndarray:
    if min_hops <= 1 or lpi.shape[0] < 2:
        return lpi
    out = np.empty_like(lpi)
    held = int(lpi[0])
    pending = None
    pending_count = 0
    for i, value in enumerate(lpi.tolist()):
        value = int(value)
        if value == held:
            pending = None
            pending_count = 0
        else:
            if pending == value:
                pending_count += 1
            else:
                pending = value
                pending_count = 1
            if pending_count >= min_hops:
                held = pending
                pending = None
                pending_count = 0
        out[i] = held
    return out


def collapse_events(times: np.ndarray, lpi: np.ndarray) -> List[IdxEvent]:
    events: List[IdxEvent] = []
    prev: int | None = None
    for t, value in zip(times.tolist(), lpi.tolist()):
        value = int(value)
        if prev is None or value != prev:
            events.append((float(t), value))
            prev = value
    if events and events[0][0] > 0.0:
        events.insert(0, (0.0, events[0][1]))
    elif not events:
        events.append((0.0, int(lpi[0]) if lpi.size else 0))
    return events


def analyse(wav: WavData, config: PreviewConfig | None = None) -> PreviewResult:
    config = config or PreviewConfig()
    revolution_s = revolution_period_s(config.rpm)
    filtered = lowpass_mid(wav.mid, wav.sample_rate, config.lpf_hz)
    times, envelope, hop_s = peak_envelope(filtered, wav.sample_rate, config.hop_ms)
    win = max(1, int(round(config.lookahead_revs * revolution_s / hop_s)))

    if config.two_rev:
        lookahead = forward_max(envelope, win, offset=0)
        following = forward_max(envelope, win, offset=win)
        combined = lookahead + following
    else:
        lookahead = forward_max(envelope, win, offset=0)
        combined = envelope + lookahead

    lpi = map_combined_to_lpi(combined, config)
    lpi = apply_slew(lpi, hop_s, config.slew_ms, float(config.lpi_tight - config.lpi_open))
    quantize = max(1, int(config.quantize))
    lo = min(config.lpi_tight, config.lpi_open)
    hi = max(config.lpi_tight, config.lpi_open)
    lpi = (np.round(lpi / quantize) * quantize).astype(np.int32)
    lpi = np.clip(lpi, lo, hi)
    min_hops = max(1, int(round(config.min_hold_ms / (hop_s * 1000.0))))
    lpi = apply_min_hold(lpi, min_hops)

    duration_s = wav.mid.shape[0] / float(wav.sample_rate)
    return PreviewResult(
        config=config,
        duration_s=duration_s,
        revolution_s=revolution_s,
        hop_s=hop_s,
        envelope_times=times,
        envelope=envelope,
        lookahead=lookahead,
        combined=combined,
        lpi_dense=lpi,
        idx_events=collapse_events(times, lpi),
    )


def write_idx(path: str | Path, events: Sequence[IdxEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        for time_s, lpi in events:
            writer.writerow([f'{time_s:.3f}', int(lpi)])


def write_env(path: str | Path, result: PreviewResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time', 'peak', 'lookahead', 'combined', 'lpi'])
        for row in zip(
            result.envelope_times,
            result.envelope,
            result.lookahead,
            result.combined,
            result.lpi_dense,
        ):
            writer.writerow([
                f'{row[0]:.3f}',
                f'{row[1]:.6f}',
                f'{row[2]:.6f}',
                f'{row[3]:.6f}',
                int(row[4]),
            ])


def default_idx_path(wav_path: str | Path) -> Path:
    wav_path = Path(wav_path)
    return wav_path.with_suffix('.idx')


def default_env_path(wav_path: str | Path) -> Path:
    wav_path = Path(wav_path)
    return wav_path.with_suffix('.env.csv')


def downsample_minmax(samples: np.ndarray, sample_rate: int, max_pairs: int = 6000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cool Edit-style peak reduction for drawing a long waveform."""
    n = int(samples.shape[0])
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    duration = n / float(sample_rate)
    if n <= max_pairs * 2:
        times = np.arange(n, dtype=np.float64) / float(sample_rate)
        return times, samples.astype(np.float64), samples.astype(np.float64)
    bin_size = int(np.ceil(n / max_pairs))
    n_bins = int(np.ceil(n / bin_size))
    padded = np.zeros(n_bins * bin_size, dtype=np.float32)
    padded[:n] = samples
    frames = padded.reshape(n_bins, bin_size)
    mins = frames.min(axis=1)
    maxs = frames.max(axis=1)
    times = (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_size / float(sample_rate)
    times = np.clip(times, 0.0, duration)
    return times, mins.astype(np.float64), maxs.astype(np.float64)
