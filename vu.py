"""Fast stereo peak meters with peak-hold, for the LPI preview GUI."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

VU_WINDOW_MS = 4.0
VU_FLOOR_DB = -70.0
VU_BAND_GREEN = float(10 ** (-12.0 / 20.0))
VU_BAND_YELLOW = float(10 ** (-6.0 / 20.0))
VU_BAND_ORANGE = float(10 ** (-3.0 / 20.0))
VU_COLORS = {
    'green': '#3d9a4a',
    'yellow': '#d4c04a',
    'orange': '#e08a3c',
    'red': '#d4453a',
}
VU_BAND_ORDER = ('green', 'yellow', 'orange', 'red')
VU_TICK_DB = (-70.0, -60.0, -48.0, -36.0, -24.0, -12.0, -6.0, -3.0, 0.0)


def instant_peaks(samples, frame: int, window: int) -> Tuple[float, float]:
    """Max abs L/R in a short window ending at frame (causal, fast peak)."""
    data = np.asarray(samples)
    if data.size == 0:
        return 0.0, 0.0
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n = int(data.shape[0])
    frame = int(frame)
    if frame < 0 or frame >= n:
        return 0.0, 0.0
    window = max(1, int(window))
    start = max(0, frame - window + 1)
    chunk = np.abs(data[start:frame + 1])
    peaks = np.max(chunk, axis=0)
    left = float(peaks[0])
    right = float(peaks[1]) if peaks.shape[0] > 1 else left
    return left, right


def db_from_level(level: float) -> float:
    value = float(level)
    if value <= 1e-9:
        return float('-inf')
    return float(20.0 * np.log10(value))


def meter_fraction_from_db(db: float) -> float:
    """Map dBFS to meter position: 0 at VU_FLOOR_DB, 1 at 0 dBFS."""
    if db == float('-inf'):
        return 0.0
    return float(min(max((float(db) - VU_FLOOR_DB) / (0.0 - VU_FLOOR_DB), 0.0), 1.0))


def meter_fraction(level: float) -> float:
    """Map a linear peak to meter position: 0 at VU_FLOOR_DB, 1 at 0 dBFS."""
    return meter_fraction_from_db(db_from_level(level))


def vu_tick_label(db: float) -> str:
    if abs(float(db)) < 1e-9:
        return '0 dB'
    return f'{int(db)} dB'


def vu_color(level: float) -> str:
    value = float(level)
    if value <= VU_BAND_GREEN:
        return 'green'
    if value <= VU_BAND_YELLOW:
        return 'yellow'
    if value <= VU_BAND_ORANGE:
        return 'orange'
    return 'red'


def lit_bands(level: float) -> List[Tuple[float, float, str]]:
    """Lit (lo, hi, zone) ranges on the 0 dBFS meter (0=floor, 1=0 dBFS)."""
    frac = meter_fraction(level)
    if frac <= 0.0:
        return []
    edges = (
        (VU_FLOOR_DB, -12.0, 'green'),
        (-12.0, -6.0, 'yellow'),
        (-6.0, -3.0, 'orange'),
        (-3.0, 0.0, 'red'),
    )
    bands = []
    for lo_db, hi_db, name in edges:
        lo = meter_fraction(10 ** (lo_db / 20.0)) if lo_db > VU_FLOOR_DB else 0.0
        hi = 1.0 if hi_db >= 0.0 else meter_fraction(10 ** (hi_db / 20.0))
        if frac <= lo:
            break
        bands.append((lo, min(frac, hi), name))
    return bands


class PeakHold:
    def __init__(self) -> None:
        self.left = 0.0
        self.right = 0.0

    def update(self, left: float, right: float) -> None:
        self.left = max(self.left, float(left))
        self.right = max(self.right, float(right))

    def reset(self) -> None:
        self.left = 0.0
        self.right = 0.0
