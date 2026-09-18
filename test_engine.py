#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (
    PreviewConfig,
    analyse,
    collapse_events,
    forward_max,
    load_wav,
    map_combined_to_lpi,
    write_idx,
)
from lpi_preview import main as lpi_main


def _write_tone_wav(path: Path, sample_rate: int, segments: list[tuple[float, float, float]]) -> None:
    """segments: list of (duration_s, freq_hz, amplitude)."""
    chunks = []
    for duration_s, freq_hz, amplitude in segments:
        n = int(round(duration_s * sample_rate))
        t = np.arange(n, dtype=np.float64) / sample_rate
        chunks.append((amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32))
    mono = np.concatenate(chunks)
    stereo = np.stack([mono, mono], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


class ForwardMaxTests(unittest.TestCase):
    def test_window_includes_future_peak_not_past(self):
        values = np.array([0.1, 0.2, 0.9, 0.1], dtype=np.float32)
        got = forward_max(values, window=2, offset=0)
        np.testing.assert_allclose(got, [0.2, 0.9, 0.9, 0.1], atol=1e-6)

    def test_offset_window_skips_current(self):
        values = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32)
        got = forward_max(values, window=2, offset=2)
        np.testing.assert_allclose(got, [0.9, 0.3, 0.0, 0.0], atol=1e-6)


class MapCombinedToLpiTests(unittest.TestCase):
    def _config(self, **overrides) -> PreviewConfig:
        values = dict(
            lpi_tight=24500,
            lpi_open=19500,
            reference_peak=0.40,
            quantize=1,
        )
        values.update(overrides)
        return PreviewConfig(**values)

    def test_track_peak_below_ref_stretches_floor_to_open_not_dc_shift(self):
        # combined.max() = 0.6 < 2*ref = 0.8, so the old linear map never
        # reached Open LPI (floor sat at 20750). Stretch the bottom to Open
        # while leaving Tight alone — not a DC offset of every sample.
        combined = np.array([0.0, 0.3, 0.6], dtype=np.float32)
        lpi = map_combined_to_lpi(combined, self._config())

        np.testing.assert_array_equal(lpi, [24500, 22000, 19500])

        old_denom = 2.0 * 0.40
        old_lpi = 24500 - (combined / old_denom) * (24500 - 19500)
        dc_shifted = old_lpi - (old_lpi.min() - 19500)
        self.assertGreater(int(lpi[1]), int(round(dc_shifted[1])))
        self.assertEqual(int(lpi[0]), 24500)

    def test_peak_above_ref_still_clips_to_open(self):
        combined = np.array([0.0, 0.4, 1.2], dtype=np.float32)
        lpi = map_combined_to_lpi(combined, self._config())
        self.assertEqual(int(lpi[0]), 24500)
        self.assertEqual(int(lpi[-1]), 19500)
        mid = 24500 - (0.4 / 0.8) * 5000
        self.assertEqual(int(lpi[1]), int(round(mid)))


class LookaheadTests(unittest.TestCase):
    def test_lpi_opens_one_revolution_before_loud_passage(self):
        sample_rate = 44100
        rpm = 33.333
        revolution_s = 60.0 / rpm
        loud_at = 6.0
        config = PreviewConfig(
            rpm=rpm,
            hop_ms=20.0,
            lpf_hz=500.0,
            lpi_tight=6114,
            lpi_open=2048,
            reference_peak=0.40,
            quantize=64,
            min_hold_ms=80.0,
            slew_ms=0.0,
            lookahead_revs=1.0,
            two_rev=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / 'fixture.wav'
            _write_tone_wav(
                wav_path,
                sample_rate,
                [
                    (loud_at, 100.0, 0.03),
                    (4.0, 100.0, 0.85),
                ],
            )
            result = analyse(load_wav(wav_path), config)

        def lpi_at(time_s: float) -> int:
            idx = int(np.searchsorted(result.envelope_times, time_s, side='right') - 1)
            idx = int(np.clip(idx, 0, result.n_hops - 1))
            return int(result.lpi_dense[idx])

        quiet = lpi_at(1.0)
        at_loud = lpi_at(loud_at)
        just_before_preview = lpi_at(loud_at - revolution_s - 0.4)
        into_preview = lpi_at(loud_at - revolution_s + 0.15)

        self.assertGreater(quiet, 5000)
        self.assertLess(at_loud, quiet - 500)
        self.assertGreater(just_before_preview, 5000)
        self.assertLess(into_preview, quiet - 500)
        self.assertLess(into_preview, just_before_preview)

        first_open_t = next(
            t for t, lpi in result.idx_events if lpi <= quiet - 500
        )
        self.assertLess(first_open_t, loud_at - 0.5)
        self.assertGreater(first_open_t, loud_at - revolution_s - 0.5)

    def test_idx_is_sparse_seconds_and_int_lpi(self):
        times = np.array([0.0, 0.02, 0.04, 0.06])
        lpi = np.array([6114, 6114, 2048, 2048], dtype=np.int32)
        events = collapse_events(times, lpi)
        self.assertEqual(events, [(0.0, 6114), (0.04, 2048)])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'track.idx'
            write_idx(path, events)
            with path.open() as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows, [['0.000', '6114'], ['0.040', '2048']])
        self.assertEqual(float(rows[0][0]), 0.0)
        self.assertEqual(int(rows[1][1]), 2048)


class CliTests(unittest.TestCase):
    def test_cli_writes_idx_next_to_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / '1.wav'
            _write_tone_wav(wav_path, 22050, [(2.0, 120.0, 0.2), (2.0, 120.0, 0.7)])
            idx_path = wav_path.with_suffix('.idx')
            rc = lpi_main([str(wav_path), '--quiet', '--slew-ms', '0'])
            self.assertEqual(rc, 0)
            self.assertTrue(idx_path.is_file())
            with idx_path.open() as handle:
                rows = list(csv.reader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(len(rows[0]), 2)
            float(rows[0][0])
            int(rows[0][1])


if __name__ == '__main__':
    unittest.main()
