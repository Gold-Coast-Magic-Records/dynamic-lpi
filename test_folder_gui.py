#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import write_idx
from gui import LpiPreviewApp
from player import NullAudioOutput, Playhead
from test_engine import _write_tone_wav
from test_player import FakeClock


def _pump(root: tk.Tk, rounds: int = 8) -> None:
    for _ in range(rounds):
        root.update_idletasks()
        root.update()


class FolderGuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk not available: {exc}')
        self.root.withdraw()
        self.app = LpiPreviewApp(self.root)

    def tearDown(self):
        try:
            self.app.stop_playback()
        except Exception:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_open_folder_loads_first_wav_and_existing_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '2.wav', 8000, [(0.4, 120.0, 0.2)])
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000), (0.2, 19250)])

            self.app.open_folder(str(folder))
            _pump(self.root)

            self.assertEqual(Path(self.app.wav.path).name, '1.wav')
            self.assertEqual([t.number for t in self.app.tracks], [1, 2])
            self.assertEqual(self.app.idx_origin, 'loaded')
            self.assertEqual(self.app.var_idx_origin.get(), 'Loaded IDX')
            self.assertEqual(self.app.result.idx_events, [(0.0, 24000), (0.2, 19250)])

    def test_generate_after_load_switches_origin_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000)])

            self.app.open_folder(str(folder))
            _pump(self.root)
            self.assertEqual(self.app.var_idx_origin.get(), 'Loaded IDX')

            self.app.generate()
            _pump(self.root, rounds=20)

            self.assertEqual(self.app.idx_origin, 'generated')
            self.assertEqual(self.app.var_idx_origin.get(), 'Generated IDX')

    def test_open_folder_generates_idx_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])

            self.app.open_folder(str(folder))
            _pump(self.root)

            self.assertEqual(Path(self.app.wav.path).name, '1.wav')
            self.assertEqual(self.app.idx_origin, 'generated')
            self.assertEqual(self.app.var_idx_origin.get(), 'Generated IDX')
            self.assertGreater(len(self.app.result.idx_events), 0)

    def test_preview_parameters_fit_in_two_rows(self):
        _pump(self.root)
        columns, rows = self.app.param_frame.grid_size()
        self.assertLessEqual(rows, 2)
        texts = []
        for child in self.app.param_frame.winfo_children():
            try:
                value = child.cget('text')
            except tk.TclError:
                continue
            if value:
                texts.append(str(value))
        self.assertIn('Two-revolution preview', texts)
        self.assertIn('LPI LPF Hz', texts)

    def test_transport_buttons_exist(self):
        self.assertEqual(str(self.app.play_button.cget('text')), 'Play')
        self.assertEqual(str(self.app.pause_button.cget('text')), 'Pause')
        self.assertEqual(str(self.app.stop_button.cget('text')), 'Stop')

    def test_plot_click_seeks_and_play_moves_cursor(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.5, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        class Click:
            inaxes = self.app.ax_wave
            xdata = 0.18

        self.app._on_plot_click(Click())
        self.assertAlmostEqual(self.app.cursor_s, 0.18, places=2)
        self.assertAlmostEqual(self.app.player.position_s, 0.18, places=2)

        self.app.play()
        self.assertTrue(self.app.player.is_playing)
        clock.advance(0.12)
        self.app._tick_playback()
        self.assertAlmostEqual(self.app.cursor_s, 0.30, places=2)

        self.app.pause()
        frozen = self.app.cursor_s
        clock.advance(1.0)
        self.assertAlmostEqual(self.app.cursor_s, frozen, places=2)

        self.app.stop_playback()
        self.assertFalse(self.app.player.is_playing)
        self.assertAlmostEqual(self.app.cursor_s, 0.0, places=2)

    def test_playhead_stats_strip_shows_current_lpi(self):
        self.assertIs(self.app.cursor_label.master, self.app.stats_frame)
        self.assertIs(self.app.lpi_label.master, self.app.stats_frame)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)
        self.app._update_cursor_label()
        self.assertRegex(self.app.var_now_lpi.get(), r'LPI \d+')
        self.assertIn('t =', self.app.var_cursor.get())

    def test_tracks_column_is_narrow(self):
        self.assertLessEqual(int(str(self.app.tracks_frame.cget('width'))), 120)


if __name__ == '__main__':
    unittest.main()
