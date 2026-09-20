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
from test_engine import _write_tone_wav


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
            _pump(self.root)

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


if __name__ == '__main__':
    unittest.main()
