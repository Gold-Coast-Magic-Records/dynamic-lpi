#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vu import (
    PeakHold,
    VU_TICK_DB,
    instant_peaks,
    lit_bands,
    meter_fraction,
    meter_fraction_from_db,
    vu_color,
    vu_tick_label,
)


class InstantPeaksTests(unittest.TestCase):
    def test_reads_left_and_right_max_in_window(self):
        samples = np.zeros((20, 2), dtype=np.float32)
        samples[10, 0] = 0.8
        samples[12, 1] = -0.4
        left, right = instant_peaks(samples, frame=12, window=5)
        self.assertAlmostEqual(left, 0.8, places=5)
        self.assertAlmostEqual(right, 0.4, places=5)

    def test_mono_copies_to_both_channels(self):
        samples = np.array([[0.1], [0.5], [0.2]], dtype=np.float32)
        left, right = instant_peaks(samples, frame=2, window=2)
        self.assertAlmostEqual(left, 0.5)
        self.assertEqual(left, right)

    def test_clamps_window_at_file_start(self):
        samples = np.array([[0.9, 0.1], [0.2, 0.3]], dtype=np.float32)
        left, right = instant_peaks(samples, frame=0, window=8)
        self.assertAlmostEqual(left, 0.9)
        self.assertAlmostEqual(right, 0.1)

    def test_empty_or_past_end_is_silence(self):
        self.assertEqual(instant_peaks(np.zeros((0, 2)), 0, 4), (0.0, 0.0))
        samples = np.ones((4, 2), dtype=np.float32)
        self.assertEqual(instant_peaks(samples, 99, 4), (0.0, 0.0))


class PeakHoldTests(unittest.TestCase):
    def test_hold_rises_with_peaks_and_does_not_fall(self):
        hold = PeakHold()
        hold.update(0.2, 0.1)
        hold.update(0.5, 0.4)
        hold.update(0.3, 0.05)
        self.assertAlmostEqual(hold.left, 0.5)
        self.assertAlmostEqual(hold.right, 0.4)

    def test_reset_clears_both_channels(self):
        hold = PeakHold()
        hold.update(0.9, 0.8)
        hold.reset()
        self.assertEqual((hold.left, hold.right), (0.0, 0.0))


class VuColorTests(unittest.TestCase):
    def test_bands_are_green_yellow_orange_red(self):
        self.assertEqual(vu_color(0.1), 'green')
        self.assertEqual(vu_color(0.3), 'yellow')
        self.assertEqual(vu_color(0.6), 'orange')
        self.assertEqual(vu_color(0.9), 'red')

    def test_zero_dbfs_is_full_scale(self):
        self.assertAlmostEqual(meter_fraction(1.0), 1.0)
        self.assertAlmostEqual(meter_fraction(0.0), 0.0)
        self.assertLess(meter_fraction(0.5), 1.0)
        self.assertGreater(meter_fraction(0.5), meter_fraction(0.25))
        self.assertEqual(meter_fraction(1.5), 1.0)

    def test_lit_bands_stop_at_current_level(self):
        bands = lit_bands(0.3)
        self.assertEqual(bands[0][2], 'green')
        self.assertEqual(bands[-1][2], 'yellow')
        self.assertAlmostEqual(bands[-1][1], meter_fraction(0.3), places=4)
        self.assertLess(bands[0][0], bands[-1][1])
        self.assertLessEqual(bands[-1][1], 1.0)

    def test_tick_labels_mark_dbfs_scale(self):
        self.assertEqual(vu_tick_label(0.0), '0 dB')
        self.assertEqual(vu_tick_label(-12.0), '-12 dB')
        self.assertEqual(vu_tick_label(-70.0), '-70 dB')
        self.assertAlmostEqual(meter_fraction_from_db(0.0), 1.0)
        self.assertAlmostEqual(meter_fraction_from_db(-70.0), 0.0)
        self.assertGreater(meter_fraction_from_db(-24.0), 0.0)
        self.assertIn(-70.0, VU_TICK_DB)
        self.assertIn(-12.0, VU_TICK_DB)
        self.assertIn(0.0, VU_TICK_DB)


if __name__ == '__main__':
    unittest.main()
