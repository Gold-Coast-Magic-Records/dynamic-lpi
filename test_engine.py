#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (
    PreviewConfig,
    analyse,
    apply_causal_tighten_lowpass,
    apply_idx_overlay,
    apply_lpi_hysteresis,
    apply_lpi_lowpass,
    apply_open_lookahead_hold,
    collapse_events,
    forward_max,
    idx_event_index_at,
    list_numbered_tracks,
    lpi_dense_from_events,
    load_wav,
    map_combined_to_lpi,
    read_idx,
    write_idx,
)
from lpi_preview import build_parser, config_from_args, main as lpi_main, resolve_gui_sources


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


class QuantizeTwoRevTests(unittest.TestCase):
    def test_larger_quantize_reduces_idx_events_in_two_rev_mode(self):
        sample_rate = 22050
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / 'kicks.wav'
            segments = []
            for amp in (0.08, 0.55, 0.12, 0.7, 0.09, 0.4, 0.15, 0.85, 0.1, 0.5):
                segments.append((0.35, 100.0, amp))
            _write_tone_wav(wav_path, sample_rate, segments)
            wav = load_wav(wav_path)

        def events_for(quantize: int) -> int:
            result = analyse(wav, PreviewConfig(
                rpm=33.333,
                hop_ms=20.0,
                lpf_hz=100.0,
                lpi_tight=24000,
                lpi_open=19250,
                reference_peak=0.40,
                quantize=quantize,
                min_hold_ms=100.0,
                slew_ms=250.0,
                lookahead_revs=1.0,
                two_rev=True,
                lpi_lpf_hz=0.0,
            ))
            return len(result.idx_events)

        fine = events_for(16)
        coarse = events_for(256)
        self.assertLess(coarse, fine)
        self.assertGreater(fine, 4)

    def test_hysteresis_ignores_mid_bin_chatter(self):
        # 23720 is the rounding midpoint between 23712 and 23728 at quantize 16.
        raw = np.array([23712.0, 23720.0, 23722.0, 23718.0, 23712.0])
        got = apply_lpi_hysteresis(raw, quantize=16, lo=19250, hi=24000)
        np.testing.assert_array_equal(got, [23712, 23712, 23712, 23712, 23712])

    def test_hysteresis_moves_on_a_full_quantize_step(self):
        raw = np.array([23712.0, 23728.0, 23728.0])
        got = apply_lpi_hysteresis(raw, quantize=16, lo=19250, hi=24000)
        np.testing.assert_array_equal(got, [23712, 23728, 23728])

    def test_hysteresis_holds_clipped_open_without_off_grid_chatter(self):
        raw = np.array([19250.0, 19258.0, 19264.0, 19252.0, 19250.0])
        got = apply_lpi_hysteresis(raw, quantize=16, lo=19250, hi=24000)
        np.testing.assert_array_equal(got, [19250, 19250, 19250, 19250, 19250])


class LpiLowpassTests(unittest.TestCase):
    def test_zero_cutoff_is_passthrough(self):
        lpi = np.array([24000.0, 23800.0, 20000.0, 19250.0])
        got = apply_lpi_lowpass(lpi, hop_s=0.02, cutoff_hz=0.0)
        np.testing.assert_allclose(got, lpi)

    def test_attenuates_hop_rate_chatter_more_than_phrase_moves(self):
        hop_s = 0.02
        t = np.arange(800) * hop_s
        slow = 22000.0 + 1500.0 * np.sin(2 * np.pi * 0.2 * t)
        chatter = 400.0 * np.sin(2 * np.pi * 8.0 * t)
        raw = slow + chatter
        filtered = apply_lpi_lowpass(raw, hop_s=hop_s, cutoff_hz=1.0)
        raw_fast = raw - slow
        filt_fast = filtered - slow
        self.assertLess(float(np.std(filt_fast)), 0.4 * float(np.std(raw_fast)))
        self.assertGreater(float(np.std(filtered)), 0.6 * float(np.std(slow)))

    def test_analyse_lpi_lpf_reduces_idx_events(self):
        sample_rate = 22050
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / 'kicks.wav'
            segments = []
            for amp in (0.08, 0.55, 0.12, 0.7, 0.09, 0.4, 0.15, 0.85, 0.1, 0.5):
                segments.append((0.35, 100.0, amp))
            _write_tone_wav(wav_path, sample_rate, segments)
            wav = load_wav(wav_path)

        def events_for(lpi_lpf_hz: float) -> int:
            result = analyse(wav, PreviewConfig(
                rpm=33.333,
                hop_ms=20.0,
                lpf_hz=100.0,
                lpi_tight=24000,
                lpi_open=19250,
                reference_peak=0.40,
                quantize=16,
                min_hold_ms=100.0,
                slew_ms=0.0,
                lookahead_revs=1.0,
                two_rev=True,
                lpi_lpf_hz=lpi_lpf_hz,
            ))
            return len(result.idx_events)

        self.assertLess(events_for(1.0), events_for(0.0))


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
            lpi_lpf_hz=0.0,
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


class AsymmetricLookaheadTests(unittest.TestCase):
    """Open early for upcoming loud audio; do not tighten until that audio has finished."""

    def test_hold_keeps_current_open_when_preview_starts_closing(self):
        preview = np.array([24000, 20000, 19250, 21000, 24000])
        current = np.array([24000, 24000, 19250, 19250, 24000])
        got = apply_open_lookahead_hold(preview, current)
        np.testing.assert_array_equal(got, [24000, 20000, 19250, 19250, 24000])

    def test_causal_tighten_lags_rise_but_opens_immediately(self):
        hop_s = 0.02
        up = np.array([19250.0] * 10 + [24000.0] * 50)
        got_up = apply_causal_tighten_lowpass(up, hop_s, 0.2)
        self.assertAlmostEqual(got_up[9], 19250.0)
        self.assertGreater(got_up[20], 19250.0)
        self.assertLess(got_up[20], 24000.0)
        down = np.array([24000.0] * 5 + [19250.0] * 5)
        got_down = apply_causal_tighten_lowpass(down, hop_s, 0.2)
        self.assertEqual(got_down[5], 19250.0)

    def _quiet_loud_quiet(self, **config_overrides):
        sample_rate = 22050
        rpm = 33.333
        revolution_s = 60.0 / rpm
        quiet_before = 5.0
        loud_s = 8.0
        values = dict(
            rpm=rpm,
            hop_ms=20.0,
            lpf_hz=500.0,
            lpi_tight=24000,
            lpi_open=19250,
            reference_peak=0.40,
            quantize=64,
            min_hold_ms=80.0,
            slew_ms=0.0,
            lookahead_revs=1.0,
            two_rev=True,
            lpi_lpf_hz=0.0,
        )
        values.update(config_overrides)
        config = PreviewConfig(**values)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / 'block.wav'
            _write_tone_wav(
                wav_path,
                sample_rate,
                [
                    (quiet_before, 100.0, 0.03),
                    (loud_s, 100.0, 0.85),
                    (5.0, 100.0, 0.03),
                ],
            )
            result = analyse(load_wav(wav_path), config)
        loud_at = quiet_before
        loud_end = quiet_before + loud_s
        return result, revolution_s, loud_at, loud_end

    def _lpi_at(self, result, time_s: float) -> int:
        idx = int(np.searchsorted(result.envelope_times, time_s, side='right') - 1)
        idx = int(np.clip(idx, 0, result.n_hops - 1))
        return int(result.lpi_dense[idx])

    def test_still_opens_one_revolution_before_loud_in_two_rev_mode(self):
        result, revolution_s, loud_at, _loud_end = self._quiet_loud_quiet()
        quiet = self._lpi_at(result, 1.0)
        into_preview = self._lpi_at(result, loud_at - revolution_s + 0.2)
        self.assertGreater(quiet, 22000)
        self.assertLess(into_preview, quiet - 1500)

    def test_does_not_tighten_while_loud_audio_still_playing(self):
        result, revolution_s, _loud_at, loud_end = self._quiet_loud_quiet()
        mid_loud = self._lpi_at(result, loud_end - 4.0)
        still_loud = self._lpi_at(result, loud_end - 0.4)
        # Following-rev lookahead is already quiet this close to the end;
        # current audio still needs the open groove.
        self.assertLess(mid_loud, 20500)
        self.assertLess(still_loud, mid_loud + 400)
        self.assertLess(still_loud, 20500)

    def test_tightens_only_after_loud_audio_finishes(self):
        result, revolution_s, _loud_at, loud_end = self._quiet_loud_quiet()
        still_loud = self._lpi_at(result, loud_end - 0.25)
        after = self._lpi_at(result, loud_end + revolution_s + 0.3)
        self.assertLess(still_loud, 20500)
        self.assertGreater(after, still_loud + 1500)

    def test_lpi_lpf_does_not_start_tightening_during_loud(self):
        result, _revolution_s, _loud_at, loud_end = self._quiet_loud_quiet(lpi_lpf_hz=0.2)
        mid_loud = self._lpi_at(result, loud_end - 4.0)
        still_loud = self._lpi_at(result, loud_end - 0.4)
        self.assertLess(mid_loud, 20500)
        self.assertLess(still_loud, 20500)
        self.assertLess(still_loud, mid_loud + 600)


class IdxEventIndexTests(unittest.TestCase):
    def test_idx_event_index_at_holds_until_next_event(self):
        events = [(0.0, 24000), (0.2, 19250), (0.8, 24000)]
        self.assertEqual(idx_event_index_at(events, 0.0), 0)
        self.assertEqual(idx_event_index_at(events, 0.15), 0)
        self.assertEqual(idx_event_index_at(events, 0.2), 1)
        self.assertEqual(idx_event_index_at(events, 0.5), 1)
        self.assertEqual(idx_event_index_at(events, 0.8), 2)

    def test_idx_event_index_at_empty_returns_minus_one(self):
        self.assertEqual(idx_event_index_at([], 1.0), -1)


class PreviewConfigDefaultTests(unittest.TestCase):
    def test_preview_config_default_lpi_tight(self):
        self.assertEqual(PreviewConfig().lpi_tight, 24500)

    def test_preview_config_default_lpf_hz(self):
        self.assertEqual(PreviewConfig().lpf_hz, 150.0)

    def test_cli_defaults_match_preview_config(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.lpi_tight, 24500)
        self.assertEqual(args.lpf_hz, 150.0)
        config = config_from_args(args)
        self.assertEqual(config.lpi_tight, 24500)
        self.assertEqual(config.lpf_hz, 150.0)


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


class NumberedTrackFolderTests(unittest.TestCase):
    def test_lists_digit_stems_numerically_and_detects_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / '2.wav').write_bytes(b'')
            (folder / '10.wav').write_bytes(b'')
            (folder / '1.wav').write_bytes(b'')
            (folder / '1.idx').write_text('0.000,24000\n')
            (folder / 'notes.wav').write_bytes(b'')
            (folder / '1.env.csv').write_text('x')
            tracks = list_numbered_tracks(folder)

            self.assertEqual([t.number for t in tracks], [1, 2, 10])
            self.assertTrue(tracks[0].has_idx)
            self.assertFalse(tracks[1].has_idx)
            self.assertEqual(tracks[0].wav_path.name, '1.wav')
            self.assertEqual(tracks[0].idx_path.name, '1.idx')

    def test_read_idx_roundtrip(self):
        events = [(0.0, 24000), (1.5, 19250)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '1.idx'
            write_idx(path, events)
            self.assertEqual(read_idx(path), events)

    def test_lpi_dense_follows_loaded_events(self):
        times = np.array([0.0, 0.02, 0.04, 0.06])
        events = [(0.0, 100), (0.04, 200)]
        got = lpi_dense_from_events(times, events)
        np.testing.assert_array_equal(got, [100, 100, 200, 200])

    def test_overlay_replaces_generated_idx(self):
        from engine import PreviewResult
        times = np.array([0.0, 0.02, 0.04])
        result = PreviewResult(
            config=PreviewConfig(),
            duration_s=1.0,
            revolution_s=1.8,
            hop_s=0.02,
            envelope_times=times,
            envelope=np.zeros(3),
            lookahead=np.zeros(3),
            combined=np.zeros(3),
            lpi_dense=np.array([1, 2, 3], dtype=np.int32),
            idx_events=[(0.0, 1)],
        )
        apply_idx_overlay(result, [(0.0, 24000), (0.04, 19250)])
        self.assertEqual(result.idx_events, [(0.0, 24000), (0.04, 19250)])
        np.testing.assert_array_equal(result.lpi_dense, [24000, 24000, 19250])

    def test_resolve_gui_sources_treats_directory_as_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            wav = folder / '1.wav'
            wav.write_bytes(b'')
            self.assertEqual(resolve_gui_sources(str(folder)), (None, str(folder)))
            self.assertEqual(resolve_gui_sources(str(wav)), (str(wav), None))
        self.assertEqual(resolve_gui_sources(None), (None, None))
        missing_dir = '/home/lathe/Desktop/records/PL-missing/Disc/A/Side/A'
        self.assertEqual(resolve_gui_sources(missing_dir), (None, missing_dir))

    def test_main_launches_gui_for_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('lpi_preview.launch_gui', return_value=0) as launch:
                rc = lpi_main([tmp])
            launch.assert_called_once()
            self.assertEqual(rc, 0)

    def test_main_gui_flag_with_folder_launches_gui(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('lpi_preview.launch_gui', return_value=0) as launch:
                rc = lpi_main(['--gui', tmp])
            launch.assert_called_once()
            self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
