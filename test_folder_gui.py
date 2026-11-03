#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import apply_amplitude_db, write_idx
from gui import LpiPreviewApp
from player import NullAudioOutput, Playhead
from test_engine import _write_tone_wav
from test_player import FakeClock, RecordingOutput


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
            self.app.cancel_amplitude()
        except Exception:
            pass
        try:
            self.app.stop_playback()
        except Exception:
            pass
        try:
            self.app._unbind_space_keys()
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

    def test_open_wav_with_directory_loads_numbered_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '2.wav', 8000, [(0.4, 120.0, 0.2)])
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])

            self.app.open_wav(str(folder))
            _pump(self.root)

            self.assertEqual(Path(self.app.wav.path).name, '1.wav')
            self.assertEqual([t.number for t in self.app.tracks], [1, 2])

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
        self.assertEqual(str(self.app.start_button.cget('text')), '|<')
        self.assertEqual(str(self.app.play_button.cget('text')), 'Play')
        self.assertEqual(str(self.app.stop_button.cget('text')), 'Stop')
        self.assertEqual(str(self.app.zoom_in_button.cget('text')), 'Zoom +')
        self.assertEqual(str(self.app.zoom_out_button.cget('text')), 'Zoom −')
        self.assertFalse(hasattr(self.app, 'pause_button'))

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

        self.app.stop_playback()
        frozen = self.app.cursor_s
        clock.advance(1.0)
        self.assertFalse(self.app.player.is_playing)
        self.assertAlmostEqual(self.app.cursor_s, frozen, places=2)

        self.app.go_to_start()
        self.assertAlmostEqual(self.app.cursor_s, 0.0, places=2)
        self.assertAlmostEqual(self.app.player.position_s, 0.0, places=2)

    def test_vu_ticks_keep_playhead_advancing(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(1.0, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app._seek_to(0.1)
        self.app.play()
        for _ in range(4):
            clock.advance(0.05)
            self.app._tick_vu()
        self.assertAlmostEqual(self.app.cursor_s, 0.3, places=2)
        self.assertTrue(self.app.player.is_playing)

    def test_space_toggles_play_and_stop(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.5, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.toggle_playback()
        self.assertTrue(self.app.player.is_playing)
        clock.advance(0.1)
        self.app._tick_playback()
        self.app.toggle_playback()
        self.assertFalse(self.app.player.is_playing)
        self.assertGreater(self.app.cursor_s, 0.05)

    def test_space_ignored_while_editing_field(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        entry = next(
            child for child in self.app.param_frame.winfo_children()
            if child.winfo_class() in ('TEntry', 'Entry')
        )

        class SpaceEvent:
            widget = entry

        self.app.toggle_playback(SpaceEvent())
        self.assertFalse(self.app.player.is_playing)

    def test_space_on_focused_button_toggles_playback(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.5, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.stop_button.focus_set()
        _pump(self.root)
        self.assertTrue(self.app.stop_button.bind('<KeyPress-space>'))
        self.app.stop_button.event_generate('<KeyPress-space>', when='now')
        _pump(self.root)
        if not self.app.player.is_playing:
            class SpaceEvent:
                widget = self.app.stop_button
            self.app.toggle_playback(SpaceEvent())
        self.assertTrue(self.app.player.is_playing)
        self.app.toggle_playback()
        self.assertFalse(self.app.player.is_playing)

    def _active_idx_rows(self):
        rows = self.app.tree.get_children()
        return [row for row in rows if 'idx_active' in self.app.tree.item(row, 'tags')]

    def test_playback_advances_idx_tree_highlight(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(1.0, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000), (0.4, 19250), (0.8, 24000)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        rows = self.app.tree.get_children()
        self.assertEqual(len(rows), 3)
        self.assertEqual(self._active_idx_rows(), [rows[0]])

        self.app.play()
        clock.advance(0.5)
        self.app._tick_playback()
        _pump(self.root)
        self.assertEqual(self._active_idx_rows(), [rows[1]])

        clock.advance(0.4)
        self.app._tick_playback()
        _pump(self.root)
        self.assertEqual(self._active_idx_rows(), [rows[2]])

    def test_playback_idx_highlight_does_not_restart_audio(self):
        clock = FakeClock()
        output = RecordingOutput()
        self.app.player = Playhead(time_fn=clock, output=output)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(1.0, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000), (0.4, 19250), (0.8, 24000)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.play()
        play_calls = [call for call in output.calls if call[0] == 'play']
        self.assertEqual(len(play_calls), 1)

        clock.advance(0.5)
        self.app._tick_playback()
        _pump(self.root, rounds=12)

        play_calls = [call for call in output.calls if call[0] == 'play']
        self.assertEqual(len(play_calls), 1)

    def test_edit_idx_lpi_updates_overlay_and_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000), (0.2, 19250)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.set_idx_lpi_at_index(1, 20000)
        _pump(self.root)

        self.assertEqual(self.app.result.idx_events[1], (0.2, 20000))
        self.assertEqual(self.app.idx_origin, 'edited')
        self.assertEqual(self.app.var_idx_origin.get(), 'Edited IDX')
        rows = self.app.tree.get_children()
        self.assertEqual(int(self.app.tree.item(rows[1], 'values')[1]), 20000)

    def test_delete_idx_event_updates_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000), (0.2, 19250), (0.4, 21000)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.delete_idx_at_index(1)
        _pump(self.root)

        self.assertEqual(self.app.result.idx_events, [(0.0, 24000), (0.4, 21000)])
        self.assertEqual(len(self.app.tree.get_children()), 2)

    def test_cannot_delete_last_idx_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            write_idx(folder / '1.idx', [(0.0, 24000)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.delete_idx_at_index(0)
        _pump(self.root)
        self.assertEqual(len(self.app.result.idx_events), 1)

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

    def test_vu_meters_are_thin_horizontal_strips_under_plot(self):
        self.assertIs(self.app.vu_frame.master, self.app.plot_frame)
        self.assertIs(self.app.vu_left.master, self.app.vu_frame)
        self.assertIs(self.app.vu_right.master, self.app.vu_frame)
        self.assertLessEqual(int(str(self.app.vu_left.cget('height'))), 12)
        self.assertLessEqual(int(str(self.app.vu_right.cget('height'))), 12)
        self.assertNotEqual(self.app.vu_frame.master, self.app.stats_frame)
        self.app._draw_vu()
        labels = []
        for item in self.app.vu_scale.find_all():
            if self.app.vu_scale.type(item) == 'text':
                labels.append(self.app.vu_scale.itemcget(item, 'text'))
        self.assertIn('-70 dB', labels)
        self.assertIn('-24 dB', labels)
        self.assertIn('-12 dB', labels)
        self.assertIn('-6 dB', labels)
        self.assertIn('-3 dB', labels)
        self.assertIn('0 dB', labels)

    def test_vu_updates_instant_and_holds_peak(self):
        clock = FakeClock()
        self.app.player = Playhead(time_fn=clock, output=NullAudioOutput())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(
                folder / '1.wav',
                8000,
                [(0.2, 120.0, 0.8), (0.3, 120.0, 0.05)],
            )
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app._seek_to(0.05)
        self.app._update_vu()
        left, right = self.app.vu_levels
        self.assertGreater(left, 0.6)
        self.assertGreater(right, 0.6)
        held = (self.app.vu_hold.left, self.app.vu_hold.right)
        self.assertGreater(held[0], 0.6)

        self.app._seek_to(0.35)
        self.app._update_vu()
        self.assertLess(self.app.vu_levels[0], 0.2)
        self.assertGreaterEqual(self.app.vu_hold.left, held[0] - 1e-6)

    def test_vu_peak_hold_resets_on_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.3, 120.0, 0.7)])
            _write_tone_wav(folder / '2.wav', 8000, [(0.3, 120.0, 0.1)])
            self.app.open_folder(str(folder))
            _pump(self.root)
            self.app._seek_to(0.1)
            self.app._update_vu()
            self.assertGreater(self.app.vu_hold.left, 0.5)
            self.app.open_wav(str(folder / '2.wav'), from_list=True)
            _pump(self.root)
            self.assertLess(self.app.vu_hold.left, 0.3)

    def test_tracks_column_is_narrow(self):
        self.assertLessEqual(int(str(self.app.tracks_frame.cget('width'))), 120)

    def test_zoom_in_centers_on_playhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(1.0, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)
        self.app.cursor_s = 0.4
        self.app.zoom_in()
        x0, x1 = self.app.ax_wave.get_xlim()
        self.assertLess(x1 - x0, 0.8)
        self.assertGreaterEqual(x0, -1e-6)
        self.assertLessEqual(x1, 1.0 + 1e-6)
        self.assertLessEqual(x0, 0.4)
        self.assertGreaterEqual(x1, 0.4)

    def test_follow_keeps_playhead_in_view_and_pins_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(2.0, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)
        self.app.cursor_s = 0.5
        self.app.zoom_in()
        span = self.app._view_xlim[1] - self.app._view_xlim[0]
        self.app.cursor_s = 1.6
        self.app._follow_playhead_view()
        x0, x1 = self.app.ax_wave.get_xlim()
        self.assertLessEqual(x0, 1.6)
        self.assertGreaterEqual(x1, 1.6)
        self.assertAlmostEqual(x1, 2.0, places=2)
        self.app.cursor_s = 0.05
        self.app._follow_playhead_view()
        x0, x1 = self.app.ax_wave.get_xlim()
        self.assertAlmostEqual(x0, 0.0, places=2)
        self.assertAlmostEqual(x1 - x0, span, places=2)

    def test_amplitude_button_is_on_toolbar(self):
        self.assertTrue(hasattr(self.app, 'amplitude_button'))
        self.assertEqual(str(self.app.amplitude_button.cget('text')), 'Amplitude')
        self.assertIs(self.app.amplitude_button.master, self.app.toolbar)

    def test_amplitude_button_does_not_add_toolbar_row(self):
        _pump(self.root)
        self.root.update_idletasks()
        toolbar_height = int(self.app.toolbar.winfo_reqheight())
        button_height = int(self.app.amplitude_button.winfo_reqheight())
        self.assertLessEqual(toolbar_height, button_height + 24)

    def _widget_texts(self, widget):
        texts = []
        try:
            value = widget.cget('text')
        except tk.TclError:
            value = None
        if value:
            texts.append(str(value))
        try:
            children = widget.winfo_children()
        except tk.TclError:
            return texts
        for child in children:
            texts.extend(self._widget_texts(child))
        return texts

    def test_open_amplitude_editor_shows_modal_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.4, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.cursor_s = 0.12
        self.app.open_amplitude_editor()
        _pump(self.root)

        dialog = self.app.amplitude_dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(str(dialog.title()), 'Edit Amplitude')
        texts = self._widget_texts(dialog)
        self.assertIn('Preview', texts)
        self.assertIn('Apply', texts)
        self.assertIn('Cancel', texts)
        self.assertAlmostEqual(float(self.app.amplitude_scale.cget('from')), -4.0)
        self.assertAlmostEqual(float(self.app.amplitude_scale.cget('to')), 4.0)
        self.assertAlmostEqual(self.app._amp_origin_s, 0.12, places=3)

    def test_amplitude_preview_plays_gained_audio_from_cursor(self):
        clock = FakeClock()
        output = SampleCaptureOutput()
        self.app.player = Playhead(time_fn=clock, output=output)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(0.5, 120.0, 0.25)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        original_peak = float(np.max(np.abs(self.app.wav.channels)))
        self.app.cursor_s = 0.18
        self.app.open_amplitude_editor()
        _pump(self.root)
        self.app.var_amp_db.set(4.0)
        self.app.preview_amplitude()
        _pump(self.root)

        self.assertTrue(self.app.player.is_playing)
        self.assertEqual(output.start_frame, int(round(0.18 * 8000)))
        gain = 10 ** (4.0 / 20.0)
        preview_peak = float(np.max(np.abs(output.samples)))
        self.assertAlmostEqual(preview_peak, original_peak * gain, places=5)
        self.assertEqual(output.samples.dtype, np.float32)
        np.testing.assert_allclose(
            self.app.wav.channels,
            apply_amplitude_db(output.samples, -4.0),
            atol=1e-5,
        )

    def test_amplitude_preview_restarts_from_original_origin(self):
        clock = FakeClock()
        output = SampleCaptureOutput()
        self.app.player = Playhead(time_fn=clock, output=output)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_tone_wav(folder / '1.wav', 8000, [(1.0, 120.0, 0.2)])
            self.app.open_folder(str(folder))
            _pump(self.root)

        self.app.cursor_s = 0.2
        self.app.open_amplitude_editor()
        _pump(self.root)
        self.app.var_amp_db.set(1.0)
        self.app.preview_amplitude()
        clock.advance(0.3)
        self.app._tick_playback()
        self.assertGreater(self.app.cursor_s, 0.35)
        self.app.stop_playback()
        self.app.var_amp_db.set(-2.0)
        output.calls.clear()
        self.app.preview_amplitude()

        play_frames = [call[1] for call in output.calls if call[0] == 'play']
        self.assertEqual(play_frames[-1], int(round(0.2 * 8000)))
        gain = 10 ** (-2.0 / 20.0)
        self.assertAlmostEqual(
            float(np.max(np.abs(output.samples))),
            float(np.max(np.abs(self.app.wav.channels))) * gain,
            places=5,
        )

    def test_amplitude_apply_rewrites_wav_and_keeps_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / '1.wav'
            _write_tone_wav(path, 8000, [(0.4, 120.0, 0.2)])
            before = wavfile.read(path)
            self.app.open_folder(str(folder))
            _pump(self.root)
            origin = 0.11
            self.app.cursor_s = origin
            self.app.open_amplitude_editor()
            _pump(self.root)
            self.app.var_amp_db.set(3.0)
            self.app.apply_amplitude()
            _pump(self.root, rounds=20)

            self.assertIsNone(self.app.amplitude_dialog)
            self.assertAlmostEqual(self.app.cursor_s, origin, places=2)
            sample_rate, after = wavfile.read(path)
            self.assertEqual(sample_rate, before[0])
            self.assertEqual(after.dtype, np.int16)
            gain = 10 ** (3.0 / 20.0)
            self.assertAlmostEqual(
                float(np.max(np.abs(after))) / float(np.max(np.abs(before[1]))),
                gain,
                places=2,
            )

    def test_amplitude_cancel_does_not_rewrite_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / '1.wav'
            _write_tone_wav(path, 8000, [(0.4, 120.0, 0.2)])
            original = path.read_bytes()
            self.app.open_folder(str(folder))
            _pump(self.root)
            self.app.cursor_s = 0.08
            self.app.open_amplitude_editor()
            _pump(self.root)
            self.app.var_amp_db.set(4.0)
            self.app.preview_amplitude()
            self.app.cancel_amplitude()
            _pump(self.root)

            self.assertIsNone(self.app.amplitude_dialog)
            self.assertEqual(path.read_bytes(), original)
            self.assertAlmostEqual(self.app.cursor_s, 0.08, places=2)


class SampleCaptureOutput(RecordingOutput):
    def __init__(self):
        super().__init__()
        self.samples = None
        self.start_frame = None

    def play(self, samples, sample_rate: int, start_frame: int = 0) -> None:
        data = np.asarray(samples)
        self.samples = data.copy()
        self.start_frame = int(start_frame)
        super().play(samples, sample_rate, start_frame)


if __name__ == '__main__':
    unittest.main()
