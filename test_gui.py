#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import discard_artists, focus_blocks_space, idx_origin_label, parse_tree_time, toolbar_blocks_cursor
from matplotlib.figure import Figure


class BoomArtist:
    def __init__(self, error):
        self.axes = object()
        self._error = error

    def remove(self):
        raise self._error


class GuiHelperTests(unittest.TestCase):
    def test_discard_artists_does_not_raise_after_axes_clear(self):
        fig = Figure()
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        artists = [ax.axvline(0.25) for ax in (ax1, ax2, ax3)]
        artists.append(ax1.axvspan(0.25, 1.0, alpha=0.1, linewidth=0))
        for ax in (ax1, ax2, ax3):
            ax.clear()

        leftover = discard_artists(artists)
        self.assertEqual(leftover, [])

        for ax in (ax1, ax2, ax3):
            line = ax.axvline(0.5)
            self.assertIs(line.axes, ax)

    def test_discard_artists_swallows_remove_errors(self):
        leftover = discard_artists([
            BoomArtist(NotImplementedError('cannot remove artist')),
            BoomArtist(ValueError('list.remove(x): x not in list')),
        ])
        self.assertEqual(leftover, [])

    def test_toolbar_only_blocks_pan_or_zoom(self):
        self.assertFalse(toolbar_blocks_cursor(''))
        self.assertFalse(toolbar_blocks_cursor(None))
        self.assertFalse(toolbar_blocks_cursor(' '))
        self.assertTrue(toolbar_blocks_cursor('pan/zoom'))
        self.assertTrue(toolbar_blocks_cursor('zoom rect'))
        self.assertTrue(toolbar_blocks_cursor('pan'))
        self.assertTrue(toolbar_blocks_cursor('zoom'))

    def test_parse_tree_time_ignores_stale_rows(self):
        self.assertEqual(parse_tree_time(('4.300', '2048')), 4.3)
        self.assertIsNone(parse_tree_time(('','2048')))
        self.assertIsNone(parse_tree_time(()))
        self.assertIsNone(parse_tree_time(None))

    def test_idx_origin_label(self):
        self.assertEqual(idx_origin_label('loaded'), 'Loaded IDX')
        self.assertEqual(idx_origin_label('generated'), 'Generated IDX')
        self.assertEqual(idx_origin_label(None), '')
        self.assertEqual(idx_origin_label(''), '')

    def test_space_is_blocked_only_in_value_fields(self):
        class Widget:
            def __init__(self, cls):
                self._cls = cls

            def winfo_class(self):
                return self._cls

        self.assertFalse(focus_blocks_space(None))
        self.assertFalse(focus_blocks_space(Widget('TButton')))
        self.assertFalse(focus_blocks_space(Widget('Listbox')))
        self.assertFalse(focus_blocks_space(Widget('Treeview')))
        self.assertTrue(focus_blocks_space(Widget('TEntry')))
        self.assertTrue(focus_blocks_space(Widget('Entry')))
        self.assertTrue(focus_blocks_space(Widget('Text')))
        self.assertTrue(focus_blocks_space(Widget('TCombobox')))


if __name__ == '__main__':
    unittest.main()
