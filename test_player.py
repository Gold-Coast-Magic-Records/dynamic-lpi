#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player import NullAudioOutput, Playhead, fill_output_block


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class RecordingOutput(NullAudioOutput):
    def __init__(self):
        self.calls = []

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        self.calls.append(('play', int(start_frame), int(sample_rate), int(np.asarray(samples).shape[0])))

    def stop(self) -> None:
        self.calls.append(('stop',))


class PlayheadTests(unittest.TestCase):
    def _player(self, seconds: float = 10.0, sr: int = 100):
        clock = FakeClock()
        output = RecordingOutput()
        player = Playhead(time_fn=clock, output=output)
        n = int(round(seconds * sr))
        player.load(np.zeros((n, 2), dtype=np.float32), sr)
        return player, clock, output

    def test_starts_stopped_at_zero(self):
        player, _clock, _output = self._player()
        self.assertFalse(player.is_playing)
        self.assertAlmostEqual(player.position_s, 0.0)
        self.assertAlmostEqual(player.duration_s, 10.0)

    def test_play_advances_with_clock(self):
        player, clock, output = self._player()
        player.play()
        self.assertTrue(player.is_playing)
        clock.advance(2.5)
        self.assertAlmostEqual(player.position_s, 2.5)
        self.assertIn(('play', 0, 100, 1000), output.calls)

    def test_pause_freezes_position_resume_continues(self):
        player, clock, output = self._player()
        player.play()
        clock.advance(2.0)
        player.pause()
        self.assertFalse(player.is_playing)
        self.assertAlmostEqual(player.position_s, 2.0)
        clock.advance(5.0)
        self.assertAlmostEqual(player.position_s, 2.0)
        player.play()
        clock.advance(1.0)
        self.assertAlmostEqual(player.position_s, 3.0)
        play_frames = [c[1] for c in output.calls if c[0] == 'play']
        self.assertEqual(play_frames[-1], 200)

    def test_stop_resets_to_zero(self):
        player, clock, _output = self._player()
        player.play()
        clock.advance(3.0)
        player.stop()
        self.assertFalse(player.is_playing)
        self.assertAlmostEqual(player.position_s, 0.0)

    def test_seek_clamps_and_updates_position(self):
        player, _clock, _output = self._player()
        player.seek(4.25)
        self.assertAlmostEqual(player.position_s, 4.25)
        player.seek(-1.0)
        self.assertAlmostEqual(player.position_s, 0.0)
        player.seek(99.0)
        self.assertAlmostEqual(player.position_s, 10.0)

    def test_seek_while_playing_keeps_playing_from_new_time(self):
        player, clock, output = self._player()
        player.play()
        clock.advance(1.0)
        player.seek(6.0)
        self.assertTrue(player.is_playing)
        self.assertAlmostEqual(player.position_s, 6.0)
        clock.advance(1.0)
        self.assertAlmostEqual(player.position_s, 7.0)
        self.assertEqual(output.calls[-1], ('play', 600, 100, 1000))

    def test_play_at_end_restarts_from_zero(self):
        player, clock, output = self._player()
        player.seek(10.0)
        player.play()
        self.assertTrue(player.is_playing)
        self.assertAlmostEqual(player.position_s, 0.0)
        self.assertEqual(output.calls[-1][1], 0)

    def test_reaching_end_stops_at_duration(self):
        player, clock, _output = self._player()
        player.play()
        clock.advance(10.5)
        self.assertAlmostEqual(player.position_s, 10.0)
        self.assertFalse(player.is_playing)


class FillOutputBlockTests(unittest.TestCase):
    def test_copies_a_full_block_and_advances_frame(self):
        source = np.arange(20, dtype=np.float32).reshape(10, 2)
        out = np.zeros((3, 2), dtype=np.float32)
        frame, finished = fill_output_block(source, 2, out)
        np.testing.assert_array_equal(out, source[2:5])
        self.assertEqual(frame, 5)
        self.assertFalse(finished)

    def test_pads_silence_and_finishes_at_end(self):
        source = np.ones((4, 2), dtype=np.float32)
        out = np.full((6, 2), 9.0, dtype=np.float32)
        frame, finished = fill_output_block(source, 3, out)
        np.testing.assert_array_equal(out[:1], source[3:4])
        np.testing.assert_array_equal(out[1:], 0)
        self.assertEqual(frame, 4)
        self.assertTrue(finished)


if __name__ == '__main__':
    unittest.main()
