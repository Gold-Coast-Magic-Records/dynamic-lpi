"""Tkinter + matplotlib debug UI: source WAV beside envelope and generated LPI."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (
    AMPLITUDE_DB_MAX,
    AMPLITUDE_DB_MIN,
    PreviewConfig,
    PreviewResult,
    WavData,
    analyse,
    apply_amplitude_db,
    apply_idx_overlay,
    default_idx_path,
    downsample_minmax,
    idx_event_index_at,
    list_numbered_tracks,
    load_wav,
    read_idx,
    remove_idx_event_at,
    set_idx_event_lpi,
    write_idx,
    write_wav,
)
from player import Playhead, create_audio_output, describe_audio_output
from vu import (
    PeakHold,
    VU_BAND_ORDER,
    VU_COLORS,
    VU_TICK_DB,
    VU_WINDOW_MS,
    instant_peaks,
    lit_bands,
    meter_fraction,
    meter_fraction_from_db,
    vu_tick_label,
)


BG = '#1b1d21'
PANEL = '#24262b'
FG = '#e6e6e6'
MUTED = '#9aa0a6'
ACCENT = '#c9a227'
TRACK_ACTIVE_BG = '#3a4556'

_PAN_ZOOM_MODES = {'pan/zoom', 'zoom rect', 'pan', 'zoom'}
_VALUE_FIELD_CLASSES = {'Entry', 'TEntry', 'Text', 'TCombobox', 'Spinbox', 'TSpinbox'}
_BUTTON_SPACE_CLASSES = ('TButton', 'Button', 'TCheckbutton', 'Checkbutton')


def discard_artists(artists) -> list:
    """Drop matplotlib artists even if the Axes already cleared them."""
    for artist in artists:
        try:
            if getattr(artist, 'axes', None) is None:
                continue
            artist.remove()
        except Exception:
            pass
    return []


def parse_tree_time(values) -> Optional[float]:
    if not values:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError, IndexError):
        return None


def zoom_xlim(
    center: float,
    x0: float,
    x1: float,
    duration: float,
    factor: float,
    min_span: float = 0.05,
) -> tuple[float, float]:
    """Return a new x window zoomed by factor around center.

    factor > 1 zooms in (narrower); 0 < factor < 1 zooms out. The window is
    clamped to [0, duration] and never narrower than min_span.
    """
    duration = max(float(duration), min_span)
    span = max(float(x1) - float(x0), min_span)
    if factor <= 0:
        new_span = duration
    else:
        new_span = span / float(factor)
    new_span = min(max(new_span, min_span), duration)
    half = new_span / 2.0
    center = min(max(float(center), 0.0), duration)
    start = center - half
    end = center + half
    if start < 0.0:
        end -= start
        start = 0.0
    if end > duration:
        start -= end - duration
        end = duration
    start = max(0.0, start)
    end = min(duration, max(end, start + min_span * 0.5))
    if end - start > duration:
        return 0.0, duration
    return start, end


def follow_xlim(center: float, span: float, duration: float) -> tuple[float, float]:
    """Keep the playhead in view: centered, except pinned at the file start/end."""
    duration = max(float(duration), 1e-6)
    span = min(max(float(span), 1e-6), duration)
    half = span / 2.0
    center = min(max(float(center), 0.0), duration)
    if center <= half:
        return 0.0, span
    if center >= duration - half:
        return duration - span, duration
    return center - half, center + half


def toolbar_blocks_cursor(mode) -> bool:
    return str(mode or '').strip().lower() in _PAN_ZOOM_MODES


def focus_blocks_space(widget) -> bool:
    """Space types into value fields; everywhere else it toggles playback."""
    if widget is None:
        return False
    try:
        cls = str(widget.winfo_class())
    except Exception:
        return False
    return cls in _VALUE_FIELD_CLASSES


def idx_origin_label(origin: Optional[str]) -> str:
    if origin == 'loaded':
        return 'Loaded IDX'
    if origin == 'generated':
        return 'Generated IDX'
    if origin == 'edited':
        return 'Edited IDX'
    return ''


def _float_var(value: float) -> tk.DoubleVar:
    return tk.DoubleVar(value=value)


def _int_var(value: int) -> tk.IntVar:
    return tk.IntVar(value=value)


class LpiPreviewApp:
    def __init__(
        self,
        root: tk.Tk,
        wav_path: Optional[str] = None,
        folder_path: Optional[str] = None,
        config: Optional[PreviewConfig] = None,
    ):
        self.root = root
        self.config = config or PreviewConfig()
        self.wav: Optional[WavData] = None
        self.result: Optional[PreviewResult] = None
        self.cursor_s = 0.0
        self._last_dir = str(Path.home())
        self._generating = False
        self._busy_depth = 0
        self._busy_plot_enter_bind = None
        self._busy_plot_motion_bind = None
        self._active_track_index: Optional[int] = None
        self.tracks = []
        self.idx_origin: Optional[str] = None
        self._syncing_list = False
        self.player = Playhead(output=create_audio_output())
        print(f'LPI preview audio backend: {describe_audio_output(self.player.output)}', flush=True)
        self._playback_after = None
        self._playback_n = 0
        self._cursor_lines = []
        self._plot_bg = None
        self._saved_button_space_binds = {}
        self._view_xlim = None
        self.amplitude_dialog = None
        self.amplitude_scale = None
        self._amp_origin_s = 0.0
        self.vu_hold = PeakHold()
        self.vu_levels = (0.0, 0.0)
        self._vu_after = None

        self._build_vars()
        self._build_ui()
        self._init_plot()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._bind_space_keys()

        if folder_path:
            self.root.after(80, lambda: self.open_folder(folder_path))
        elif wav_path:
            self.root.after(80, lambda: self.open_wav(wav_path))

    def _build_vars(self) -> None:
        c = self.config
        self.var_rpm = _float_var(c.rpm)
        self.var_hop = _float_var(c.hop_ms)
        self.var_lpf = _float_var(c.lpf_hz)
        self.var_tight = _int_var(c.lpi_tight)
        self.var_open = _int_var(c.lpi_open)
        self.var_ref = _float_var(c.reference_peak)
        self.var_quantize = _int_var(c.quantize)
        self.var_hold = _float_var(c.min_hold_ms)
        self.var_slew = _float_var(c.slew_ms)
        self.var_lookahead = _float_var(c.lookahead_revs)
        self.var_lpi_lpf = _float_var(c.lpi_lpf_hz)
        self.var_two_rev = tk.BooleanVar(value=c.two_rev)
        self.var_status = tk.StringVar(value='Open a WAV to generate an LPI preview.')
        self.var_busy = tk.StringVar(value='Working…')
        self.var_track = tk.StringVar(value='Track: —')
        self.var_cursor = tk.StringVar(value='t = —')
        self.var_now_lpi = tk.StringVar(value='LPI —')
        self.var_idx_events = tk.StringVar(value='.idx events')
        self.var_idx_origin = tk.StringVar(value='')
        self.var_amp_db = _float_var(0.0)
        self.var_amp_label = tk.StringVar(value='0.0 dB')

    def _build_ui(self) -> None:
        self.root.title('Dynamic LPI Preview')
        self.root.geometry('1280x820')
        self.root.minsize(960, 640)
        self.root.configure(bg=BG)

        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure('.', background=BG, foreground=FG, fieldbackground=PANEL)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TButton', background=PANEL, foreground=FG, padding=6)
        style.configure('TCheckbutton', background=BG, foreground=FG)
        style.configure('TEntry', fieldbackground=PANEL, foreground=FG)
        style.configure('TLabelframe', background=BG, foreground=MUTED)
        style.configure('TLabelframe.Label', background=BG, foreground=MUTED)
        style.configure('Treeview', background=PANEL, foreground=FG, fieldbackground=PANEL, rowheight=22)
        style.configure('Treeview.Heading', background=PANEL, foreground=MUTED)
        style.map('Treeview', background=[('selected', ACCENT)], foreground=[('selected', '#111')])

        toolbar = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        toolbar.pack(fill='x')
        self.toolbar = toolbar
        ttk.Button(toolbar, text='Open WAV…', command=self.browse_wav).pack(side='left')
        ttk.Button(toolbar, text='Open folder…', command=self.browse_folder).pack(side='left', padx=(8, 0))
        self.generate_button = ttk.Button(toolbar, text='Generate', command=self.generate)
        self.generate_button.pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text='Save .idx…', command=self.save_idx).pack(side='left', padx=(8, 0))
        self.start_button = ttk.Button(toolbar, text='|<', command=self.go_to_start, width=3)
        self.start_button.pack(side='left', padx=(16, 0))
        self.play_button = ttk.Button(toolbar, text='Play', command=self.play)
        self.play_button.pack(side='left', padx=(8, 0))
        self.stop_button = ttk.Button(toolbar, text='Stop', command=self.stop_playback)
        self.stop_button.pack(side='left', padx=(8, 0))
        self.zoom_out_button = ttk.Button(toolbar, text='Zoom −', command=self.zoom_out)
        self.zoom_out_button.pack(side='left', padx=(16, 0))
        self.zoom_in_button = ttk.Button(toolbar, text='Zoom +', command=self.zoom_in)
        self.zoom_in_button.pack(side='left', padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.var_idx_origin, foreground=ACCENT).pack(side='left', padx=(16, 0))
        self.amplitude_button = ttk.Button(toolbar, text='Amplitude', command=self.open_amplitude_editor)
        self.amplitude_button.pack(side='right')

        stats = ttk.Frame(self.root, padding=(10, 0, 10, 2))
        stats.pack(fill='x')
        self.stats_frame = stats
        self.lpi_label = ttk.Label(stats, textvariable=self.var_now_lpi, foreground=ACCENT)
        self.lpi_label.pack(side='right', anchor='e')
        self.track_label = ttk.Label(stats, textvariable=self.var_track, foreground=ACCENT)
        self.track_label.pack(side='left', anchor='w')
        self.cursor_label = ttk.Label(stats, textvariable=self.var_cursor, foreground=MUTED)
        self.cursor_label.pack(side='left', anchor='w', padx=(16, 0))

        controls = ttk.LabelFrame(self.root, text='Preview parameters', padding=(8, 2))
        controls.pack(fill='x', padx=10, pady=(0, 4))
        self.param_frame = controls
        self._add_control(controls, 'RPM', self.var_rpm, 0, 0, width=7)
        ttk.Button(controls, text='33⅓', command=lambda: self.var_rpm.set(33.333)).grid(
            row=0, column=2, padx=(0, 4)
        )
        ttk.Button(controls, text='45', command=lambda: self.var_rpm.set(45.0)).grid(
            row=0, column=3, padx=(0, 8)
        )
        self._add_control(controls, 'LPI tight', self.var_tight, 0, 4, width=7)
        self._add_control(controls, 'LPI open', self.var_open, 0, 6, width=7)
        self._add_control(controls, 'Ref peak', self.var_ref, 0, 8, width=6)
        self._add_control(controls, 'LPF Hz', self.var_lpf, 0, 10, width=6)
        self._add_control(controls, 'Hop ms', self.var_hop, 0, 12, width=6)
        self._add_control(controls, 'Lookahead revs', self.var_lookahead, 1, 0, width=6)
        self._add_control(controls, 'Quantize step', self.var_quantize, 1, 2, width=6)
        self._add_control(controls, 'Min hold ms', self.var_hold, 1, 4, width=6)
        self._add_control(controls, 'Slew ms', self.var_slew, 1, 6, width=6)
        self._add_control(controls, 'LPI LPF Hz', self.var_lpi_lpf, 1, 8, width=6)
        ttk.Checkbutton(controls, text='Two-revolution preview', variable=self.var_two_rev).grid(
            row=1, column=10, columnspan=4, sticky='w', padx=(8, 0)
        )

        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        tracks_frame = ttk.Frame(body, width=110)
        tracks_frame.pack(side='left', fill='y', padx=(0, 8))
        tracks_frame.pack_propagate(False)
        self.tracks_frame = tracks_frame
        ttk.Label(tracks_frame, text='Tracks', foreground=MUTED).pack(anchor='w', pady=(0, 4))
        self.track_list = tk.Listbox(
            tracks_frame,
            bg=PANEL,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground='#111',
            highlightthickness=0,
            borderwidth=0,
            font=('TkDefaultFont', 11),
            exportselection=False,
            activestyle='none',
        )
        self.track_list.pack(fill='both', expand=True)
        self.track_list.bind('<<ListboxSelect>>', self._on_track_select)
        self.track_list.bind('<FocusOut>', lambda _e: self._apply_track_list_highlight())

        plot_frame = ttk.Frame(body)
        plot_frame.pack(side='left', fill='both', expand=True)
        self.plot_frame = plot_frame

        import matplotlib
        matplotlib.use('TkAgg', force=True)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor=BG)
        self.ax_wave, self.ax_env, self.ax_lpi = self.figure.subplots(
            3, 1, sharex=True, gridspec_kw={'height_ratios': [2.2, 1.4, 1.6]}
        )
        self.figure.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.08, hspace=0.08)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.pack(side='bottom', fill='x')
        self._build_vu_meters(plot_frame)
        self.plot_tk = self.canvas.get_tk_widget()
        self.plot_tk.pack(side='top', fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
        self.canvas.mpl_connect('resize_event', lambda _e: setattr(self, '_plot_bg', None))
        self.cursor_artists = []
        self._cursor_lines = []
        self._plot_bg = None
        self._tree_click_bind = None
        self._lpi_edit_entry = None
        self._lpi_edit_item = None

        side = ttk.Frame(body, width=280)
        side.pack(side='right', fill='y', padx=(8, 0))
        side.pack_propagate(False)
        ttk.Label(side, textvariable=self.var_idx_events, foreground=MUTED).pack(anchor='w', pady=(0, 4))
        columns = ('time', 'lpi', 'del')
        self.tree = ttk.Treeview(side, columns=columns, show='headings', selectmode='browse')
        self.tree.heading('time', text='Time (s)')
        self.tree.heading('lpi', text='LPI')
        self.tree.heading('del', text='')
        self.tree.column('time', width=96, anchor='e')
        self.tree.column('lpi', width=88, anchor='e')
        self.tree.column('del', width=32, anchor='center', stretch=False)
        scroll = ttk.Scrollbar(side, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.tag_configure('idx_active', background='#3a4556', foreground=FG)
        self._tree_click_bind = self.tree.bind('<Button-1>', self._on_tree_click, add='+')

        status = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        status.pack(fill='x')
        self.status_frame = status
        self.busy_frame = ttk.Frame(status)
        self.busy_progress = ttk.Progressbar(
            self.busy_frame,
            mode='indeterminate',
            length=140,
        )
        self.busy_progress.pack(side='left', padx=(0, 8))
        ttk.Label(self.busy_frame, textvariable=self.var_busy, foreground=ACCENT).pack(side='left')
        ttk.Label(status, textvariable=self.var_status, foreground=MUTED).pack(anchor='w')

    def _build_vu_meters(self, parent) -> None:
        self.vu_frame = tk.Frame(parent, bg=BG)
        self.vu_frame.pack(side='bottom', fill='x', pady=(4, 2))
        self.vu_scale = tk.Canvas(
            self.vu_frame,
            height=14,
            bg=BG,
            highlightthickness=0,
            bd=0,
        )
        self.vu_scale.pack(fill='x')
        self.vu_left = self._add_vu_strip(self.vu_frame, pady=(0, 2))
        self.vu_right = self._add_vu_strip(self.vu_frame)

    def _add_vu_strip(self, parent, pady=(0, 0)):
        canvas = tk.Canvas(
            parent,
            height=8,
            bg='#14161a',
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill='x', pady=pady)
        canvas.bind('<Button-1>', self._reset_vu_hold)
        return canvas

    def _vu_samples(self):
        if self.player.samples is not None:
            return self.player.samples, self.player.sample_rate
        if self.wav is None:
            return None, 1
        return self.wav.channels, self.wav.sample_rate

    def _vu_time_s(self) -> float:
        if self.player.is_playing:
            return float(self.player.position_s)
        return float(self.cursor_s)

    def _update_vu(self) -> None:
        samples, sample_rate = self._vu_samples()
        if samples is None:
            self.vu_levels = (0.0, 0.0)
            self._draw_vu()
            return
        frame = int(round(self._vu_time_s() * float(sample_rate)))
        window = max(1, int(round(float(sample_rate) * VU_WINDOW_MS / 1000.0)))
        left, right = instant_peaks(samples, frame, window)
        self.vu_levels = (left, right)
        self.vu_hold.update(left, right)
        self._draw_vu()

    def _reset_vu_hold(self, _event=None):
        self.vu_hold.reset()
        self._draw_vu()

    def _draw_vu(self) -> None:
        self._draw_vu_scale()
        self._draw_vu_strip(self.vu_left, self.vu_levels[0], self.vu_hold.left)
        self._draw_vu_strip(self.vu_right, self.vu_levels[1], self.vu_hold.right)

    def _vu_scale_items(self):
        items = getattr(self.vu_scale, '_vu_scale_items', None)
        if items is not None:
            return items
        items = []
        for db in VU_TICK_DB:
            color = VU_COLORS['red'] if db >= 0.0 else MUTED
            item = self.vu_scale.create_text(
                0,
                7,
                text=vu_tick_label(db),
                fill=color,
                font=('TkDefaultFont', 8),
                anchor='center',
            )
            items.append((db, item))
        self.vu_scale._vu_scale_items = items
        return items

    def _draw_vu_scale(self) -> None:
        try:
            items = self._vu_scale_items()
            width = max(int(self.vu_scale.winfo_width()), 1)
            height = max(int(self.vu_scale.winfo_height()), 14)
        except tk.TclError:
            return
        for db, item in items:
            frac = meter_fraction_from_db(db)
            if frac <= 0.0:
                anchor = 'w'
                x = 0
            elif frac >= 1.0:
                anchor = 'e'
                x = width
            else:
                anchor = 'center'
                x = frac * width
            self.vu_scale.coords(item, x, height / 2.0)
            self.vu_scale.itemconfigure(item, anchor=anchor)

    def _vu_items(self, canvas):
        items = getattr(canvas, '_vu_items', None)
        if items is not None:
            return items
        items = {}
        for name in VU_BAND_ORDER:
            items[name] = canvas.create_rectangle(
                0, 0, 0, 0, fill=VU_COLORS[name], outline='',
            )
        items['hold'] = canvas.create_text(
            0,
            0,
            text='-',
            fill='#f4f1de',
            font=('TkDefaultFont', 9, 'bold'),
            anchor='center',
            state='hidden',
        )
        canvas._vu_items = items
        return items

    def _draw_vu_strip(self, canvas, level: float, hold: float) -> None:
        try:
            items = self._vu_items(canvas)
            width = max(int(canvas.winfo_width()), 1)
            height = max(int(canvas.winfo_height()), int(str(canvas.cget('height')) or 0), 1)
        except tk.TclError:
            return
        for name in VU_BAND_ORDER:
            canvas.coords(items[name], 0, 0, 0, 0)
        for lo, hi, name in lit_bands(level):
            canvas.coords(items[name], lo * width, 0, hi * width, height)
        if hold > 0:
            x = meter_fraction(hold) * width
            canvas.coords(items['hold'], x, height / 2.0)
            canvas.itemconfigure(items['hold'], state='normal')
        else:
            canvas.itemconfigure(items['hold'], state='hidden')

    def _add_control(self, parent, label, variable, row, column, width=8):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky='e', padx=(6, 3), pady=1)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, sticky='w', pady=1)
        entry.bind('<Return>', lambda _e: self.generate())

    def _busy_cursor_targets(self):
        widgets = [self.root, self.plot_frame, self.plot_tk]
        toolbar = self.plot_toolbar
        widgets.append(toolbar)
        if toolbar is not None:
            widgets.extend(toolbar.winfo_children())
        for widget in (
            getattr(self, 'vu_scale', None),
            getattr(self, 'vu_left', None),
            getattr(self, 'vu_right', None),
        ):
            if widget is not None:
                widgets.append(widget)
        return widgets

    def _apply_busy_cursor(self, cursor: str) -> None:
        for widget in self._busy_cursor_targets():
            try:
                widget.configure(cursor=cursor)
            except (tk.TclError, AttributeError):
                pass

    def _on_plot_busy_cursor(self, _event=None) -> None:
        if self._busy_depth <= 0:
            return
        try:
            self.plot_tk.configure(cursor='watch')
        except tk.TclError:
            pass

    def _enter_busy(self, message: str = 'Working…') -> None:
        self._busy_depth += 1
        self.var_busy.set(message)
        if self._busy_depth == 1:
            self.busy_frame.pack(anchor='w', fill='x', pady=(0, 4))
            try:
                self.busy_progress.start(12)
            except tk.TclError:
                pass
            self._apply_busy_cursor('watch')
            if self._busy_plot_enter_bind is None:
                self._busy_plot_enter_bind = self.plot_tk.bind(
                    '<Enter>',
                    self._on_plot_busy_cursor,
                    add='+',
                )
                self._busy_plot_motion_bind = self.plot_tk.bind(
                    '<Motion>',
                    self._on_plot_busy_cursor,
                    add='+',
                )
            try:
                self.generate_button.state(['disabled'])
            except tk.TclError:
                pass
        self.root.update_idletasks()

    def _leave_busy(self) -> None:
        if self._busy_depth <= 0:
            return
        self._busy_depth -= 1
        if self._busy_depth != 0:
            return
        try:
            self.busy_progress.stop()
        except tk.TclError:
            pass
        try:
            self.busy_frame.pack_forget()
        except tk.TclError:
            pass
        self._apply_busy_cursor('')
        if self._busy_plot_enter_bind is not None:
            try:
                self.plot_tk.unbind('<Enter>', self._busy_plot_enter_bind)
            except tk.TclError:
                pass
            self._busy_plot_enter_bind = None
        if self._busy_plot_motion_bind is not None:
            try:
                self.plot_tk.unbind('<Motion>', self._busy_plot_motion_bind)
            except tk.TclError:
                pass
            self._busy_plot_motion_bind = None
        try:
            self.generate_button.state(['!disabled'])
        except tk.TclError:
            pass

    def _update_track_label(self) -> None:
        if self.wav is None:
            self.var_track.set('Track: —')
            return
        self.var_track.set(f'Track: {Path(self.wav.path).name}')

    def _init_plot(self) -> None:
        self._style_axes()
        self._refresh_canvas()

    def _style_axes(self) -> None:
        for ax, title, ylabel in (
            (self.ax_wave, 'WAV', 'amp'),
            (self.ax_env, 'Peak envelope (lookahead in red)', 'peak'),
            (self.ax_lpi, 'LPI  (higher = tighter)', 'G2 L'),
        ):
            ax.set_facecolor('#14161a')
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#3a3d44')
            ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
            ax.set_title(title, color=FG, loc='left', fontsize=9, pad=2)
            ax.grid(True, color='#2c3036', linewidth=0.6)
        self.ax_lpi.set_xlabel('time (s)', color=MUTED, fontsize=8)
        self.ax_wave.set_ylim(-1.05, 1.05)

    def current_config(self) -> PreviewConfig:
        return PreviewConfig(
            rpm=float(self.var_rpm.get()),
            hop_ms=float(self.var_hop.get()),
            lpf_hz=float(self.var_lpf.get()),
            lpi_tight=int(self.var_tight.get()),
            lpi_open=int(self.var_open.get()),
            reference_peak=float(self.var_ref.get()),
            quantize=int(self.var_quantize.get()),
            min_hold_ms=float(self.var_hold.get()),
            slew_ms=float(self.var_slew.get()),
            lookahead_revs=float(self.var_lookahead.get()),
            two_rev=bool(self.var_two_rev.get()),
            lpi_lpf_hz=float(self.var_lpi_lpf.get()),
        )

    def browse_wav(self) -> None:
        path = filedialog.askopenfilename(
            title='Open WAV',
            initialdir=self._last_dir,
            filetypes=[('WAV files', '*.wav'), ('All files', '*.*')],
        )
        if path:
            self.open_wav(path)

    def browse_folder(self) -> None:
        path = filedialog.askdirectory(title='Open folder', initialdir=self._last_dir)
        if path:
            self.open_folder(path)

    def play(self) -> None:
        if self._amplitude_editor_open():
            self.preview_amplitude()
            return
        if self.wav is None:
            messagebox.showinfo('No WAV', 'Open a WAV file first.')
            return
        if not getattr(self.player.output, 'available', True):
            messagebox.showerror(
                'No audio output',
                'Install PortAudio and the sounddevice package:\n  pip install sounddevice',
            )
            return
        try:
            if self.player.samples is None:
                self.player.load(self.wav.channels, self.wav.sample_rate)
            if abs(self.player.position_s - self.cursor_s) > 0.02:
                self.player.seek(self.cursor_s)
            self.player.play()
        except Exception as exc:
            messagebox.showerror('Playback error', str(exc))
            return
        self.cursor_s = self.player.position_s
        self._sync_playhead_visual()
        self._schedule_playback_ticks()

    def stop_playback(self) -> None:
        if self.player.is_playing:
            self.player.pause()
        self._cancel_playback_ticks()
        if self._amplitude_editor_open():
            self.cursor_s = self._amp_origin_s
            if self.player.samples is not None:
                self.player.seek(self._amp_origin_s)
        else:
            self.cursor_s = self.player.position_s
        self._sync_playhead_visual()

    def go_to_start(self) -> None:
        self._seek_to(0.0)

    def _wav_duration(self) -> float:
        if self.wav is None:
            return 0.0
        return self.wav.mid.shape[0] / float(self.wav.sample_rate)

    def _apply_view_xlim(self) -> None:
        duration = max(self._wav_duration(), 0.001)
        if self._view_xlim is None:
            self.ax_wave.set_xlim(0.0, duration)
            return
        x0, x1 = self._view_xlim
        x0 = min(max(float(x0), 0.0), duration)
        x1 = min(max(float(x1), x0 + 0.001), duration)
        self.ax_wave.set_xlim(x0, x1)

    def zoom_in(self, factor: float = 2.0) -> None:
        self._zoom_at_cursor(factor)

    def zoom_out(self, factor: float = 0.5) -> None:
        self._zoom_at_cursor(factor)

    def _zoom_at_cursor(self, factor: float) -> None:
        duration = self._wav_duration()
        if duration <= 0:
            return
        if self._view_xlim is None:
            x0, x1 = 0.0, duration
        else:
            x0, x1 = self._view_xlim
        x0, x1 = zoom_xlim(self.cursor_s, x0, x1, duration, factor)
        if abs(x1 - x0 - duration) < 1e-6 and x0 <= 1e-6:
            self._view_xlim = None
        else:
            self._view_xlim = (x0, x1)
        self._apply_view_xlim()
        self._draw_cursor()
        self._refresh_canvas()

    def _follow_playhead_view(self) -> bool:
        """Pan a zoomed view so the playhead stays in the middle, pinned at the ends."""
        if self._view_xlim is None or self.wav is None:
            return False
        duration = self._wav_duration()
        if duration <= 0:
            return False
        span = self._view_xlim[1] - self._view_xlim[0]
        x0, x1 = follow_xlim(self.cursor_s, span, duration)
        old0, old1 = self._view_xlim
        if abs(x0 - old0) < 1e-4 and abs(x1 - old1) < 1e-4:
            return False
        self._view_xlim = (x0, x1)
        self._apply_view_xlim()
        self._draw_cursor()
        self._refresh_canvas()
        return True

    def _on_zoom_in_key(self, event=None):
        if focus_blocks_space(getattr(event, 'widget', None) if event is not None else self.root.focus_get()):
            return
        self.zoom_in()
        return 'break'

    def _on_zoom_out_key(self, event=None):
        if focus_blocks_space(getattr(event, 'widget', None) if event is not None else self.root.focus_get()):
            return
        self.zoom_out()
        return 'break'

    def _on_scroll_zoom(self, event) -> None:
        if event.inaxes is None:
            return
        step = getattr(event, 'step', 0) or 0
        if event.button == 'up' or step > 0:
            self.zoom_in(1.25)
        else:
            self.zoom_out(0.8)

    def toggle_playback(self, event=None):
        target = getattr(event, 'widget', None) if event is not None else None
        if target is None:
            target = self.root.focus_get()
        if focus_blocks_space(target):
            return
        if self._amplitude_editor_open():
            if self.player.is_playing:
                self.stop_playback()
            else:
                self.preview_amplitude()
            self._release_button_focus()
            return 'break'
        if self.player.is_playing:
            self.stop_playback()
        else:
            self.play()
        self._release_button_focus()
        return 'break'

    def _bind_space_keys(self) -> None:
        self.root.bind_all('<space>', self.toggle_playback)
        self.root.bind_all('<KeyPress-space>', self.toggle_playback)
        for cls in _BUTTON_SPACE_CLASSES:
            self._saved_button_space_binds[cls] = {
                'space': self.root.bind_class(cls, '<space>'),
                'keypress': self.root.bind_class(cls, '<KeyPress-space>'),
            }
            self.root.bind_class(cls, '<space>', self.toggle_playback)
            self.root.bind_class(cls, '<KeyPress-space>', self.toggle_playback)
            self.root.bind_class(cls, '<ButtonRelease-1>', self._release_button_focus, add='+')
        self._hook_button_space(self.root)
        for sequence, handler in (
            ('<plus>', self._on_zoom_in_key),
            ('<equal>', self._on_zoom_in_key),
            ('<KP_Add>', self._on_zoom_in_key),
            ('<minus>', self._on_zoom_out_key),
            ('<KP_Subtract>', self._on_zoom_out_key),
        ):
            self.root.bind_all(sequence, handler)

    def _hook_button_space(self, widget) -> None:
        try:
            cls = str(widget.winfo_class())
        except Exception:
            return
        if cls in _BUTTON_SPACE_CLASSES:
            widget.bind('<space>', self.toggle_playback)
            widget.bind('<KeyPress-space>', self.toggle_playback)
            widget.bind('<ButtonRelease-1>', self._release_button_focus, add='+')
            try:
                widget.configure(takefocus=False)
            except tk.TclError:
                pass
        try:
            children = widget.winfo_children()
        except Exception:
            return
        for child in children:
            self._hook_button_space(child)

    def _release_button_focus(self, _event=None):
        try:
            self.root.after_idle(self.root.focus_set)
        except Exception:
            pass

    def _unbind_space_keys(self) -> None:
        try:
            self.root.unbind_all('<space>')
            self.root.unbind_all('<KeyPress-space>')
            for sequence in ('<plus>', '<equal>', '<KP_Add>', '<minus>', '<KP_Subtract>'):
                self.root.unbind_all(sequence)
        except Exception:
            pass
        for cls, prev in self._saved_button_space_binds.items():
            try:
                self.root.bind_class(cls, '<space>', prev.get('space') or '')
                self.root.bind_class(cls, '<KeyPress-space>', prev.get('keypress') or '')
            except Exception:
                pass
        self._saved_button_space_binds = {}

    def _seek_to(self, time_s: float) -> None:
        if self.wav is None:
            return
        duration = self.wav.mid.shape[0] / self.wav.sample_rate
        t = min(max(float(time_s), 0.0), duration)
        self.cursor_s = t
        if self.player.samples is not None:
            self.player.seek(t)
        self._update_cursor_label()
        self._update_vu()
        self._sync_idx_tree_highlight(scroll=self.player.is_playing)
        if self._follow_playhead_view():
            return
        if self.player.is_playing and self._cursor_lines:
            self._blit_playhead()
        else:
            self._draw_cursor()
            self._refresh_canvas()

    def _sync_playhead_visual(self) -> None:
        self._update_cursor_label()
        self._update_vu()
        self._sync_idx_tree_highlight(scroll=False)
        if self._follow_playhead_view():
            return
        self._draw_cursor()
        self._refresh_canvas()

    def _schedule_playback_ticks(self) -> None:
        self._cancel_playback_ticks()
        self._playback_n = 0
        self._playback_after = self.root.after(16, self._tick_playback)

    def _cancel_playback_ticks(self) -> None:
        if self._playback_after is not None:
            try:
                self.root.after_cancel(self._playback_after)
            except Exception:
                pass
            self._playback_after = None
        self._cancel_vu_ticks()

    def _cancel_vu_ticks(self) -> None:
        if self._vu_after is not None:
            try:
                self.root.after_cancel(self._vu_after)
            except Exception:
                pass
            self._vu_after = None

    def _tick_vu(self) -> None:
        """Keep the playhead moving on the same cadence as the VU meters."""
        self._step_playback_visual()

    def _step_playback_visual(self) -> None:
        self.cursor_s = self.player.position_s
        self._update_cursor_label()
        try:
            self._update_vu()
        except Exception:
            pass
        self._sync_idx_tree_highlight(scroll=True)
        self._playback_n += 1
        if self._playback_n % 5 == 1 and self._follow_playhead_view():
            return
        self._blit_playhead()

    def _tick_playback(self) -> None:
        self._playback_after = None
        try:
            self._step_playback_visual()
        finally:
            if self.player.is_playing:
                self._playback_after = self.root.after(16, self._tick_playback)

    def _on_close(self) -> None:
        self._cancel_lpi_edit()
        self._close_amplitude_dialog()
        self._cancel_playback_ticks()
        self._unbind_space_keys()
        try:
            self.player.stop()
        except Exception:
            pass
        self.root.destroy()

    def open_folder(self, folder: str) -> None:
        tracks = list_numbered_tracks(folder)
        if not tracks:
            messagebox.showinfo('No tracks', f'No 1.wav, 2.wav, … files in:\n{folder}')
            return
        self._last_dir = str(Path(folder))
        self.tracks = tracks
        self._refresh_track_list()
        self.open_wav(str(tracks[0].wav_path), from_list=True)

    def open_wav(self, path: str, from_list: bool = False) -> None:
        if Path(path).is_dir():
            self.open_folder(path)
            return
        self._open_wav_args = (path, from_list)
        self._enter_busy('Loading audio…')
        self.root.after(1, self._open_wav_worker)

    def _open_wav_worker(self) -> None:
        path, from_list = self._open_wav_args
        try:
            try:
                self.wav = load_wav(path)
            except Exception as exc:
                messagebox.showerror('WAV error', str(exc))
                return
            wav_path = Path(path)
            self._cancel_playback_ticks()
            self.cursor_s = 0.0
            self._view_xlim = None
            self.vu_hold.reset()
            self.player.load(self.wav.channels, self.wav.sample_rate)
            self._last_dir = str(wav_path.parent)
            self.root.title(f'Dynamic LPI Preview — {wav_path.name}')
            self._update_track_label()
            if not from_list:
                sibling_tracks = list_numbered_tracks(wav_path.parent)
                if sibling_tracks:
                    self.tracks = sibling_tracks
                    self._refresh_track_list()
            self._select_track_in_list(wav_path)
            idx_path = default_idx_path(path)
            if idx_path.is_file():
                self.var_busy.set('Loading index…')
                self.root.update_idletasks()
                self._load_existing_idx(idx_path)
                return
            self.generate()
        finally:
            if not self._generating and self._busy_depth > 0:
                self._leave_busy()

    def _load_existing_idx(self, idx_path: Path) -> None:
        if self.wav is None:
            return
        try:
            events = read_idx(idx_path)
            if not events:
                raise ValueError(f'{idx_path.name} contained no valid events')
            self.result = analyse(self.wav, self.current_config())
            apply_idx_overlay(self.result, events)
        except Exception as exc:
            messagebox.showerror('IDX error', str(exc))
            self.generate()
            return
        self._set_idx_origin('loaded')
        self._redraw()
        self._fill_tree()
        self.var_status.set(
            f'Loaded {Path(self.wav.path).name}  ·  {idx_path.name}  ·  '
            f'{len(self.result.idx_events)} events'
        )
        self._update_cursor_label()
        self._update_vu()
        self._refresh_canvas()

    def _set_idx_origin(self, origin: Optional[str]) -> None:
        self.idx_origin = origin
        self.var_idx_origin.set(idx_origin_label(origin))

    def _refresh_track_list(self) -> None:
        self.track_list.delete(0, tk.END)
        for track in self.tracks:
            marker = '  idx' if track.has_idx else ''
            self.track_list.insert(tk.END, f'{track.wav_path.name}{marker}')
        self._apply_track_list_highlight()

    def _apply_track_list_highlight(self) -> None:
        count = self.track_list.size()
        if count <= 0:
            return
        for index in range(count):
            if index == self._active_track_index:
                self.track_list.itemconfig(index, bg=TRACK_ACTIVE_BG, fg=FG)
            else:
                self.track_list.itemconfig(index, bg=PANEL, fg=FG)

    def _select_track_in_list(self, wav_path: Path) -> None:
        target = wav_path.resolve()
        for index, track in enumerate(self.tracks):
            if track.wav_path.resolve() == target:
                self._active_track_index = index
                self._syncing_list = True
                try:
                    self.track_list.selection_clear(0, tk.END)
                    self.track_list.selection_set(index)
                    self.track_list.see(index)
                    self._apply_track_list_highlight()
                finally:
                    self._syncing_list = False
                return
        self._active_track_index = None
        self._apply_track_list_highlight()

    def _on_track_select(self, _event) -> None:
        if self._syncing_list:
            return
        selection = self.track_list.curselection()
        if not selection or not self.tracks:
            return
        index = int(selection[0])
        if index < 0 or index >= len(self.tracks):
            return
        path = self.tracks[index].wav_path
        if self.wav is not None and Path(self.wav.path).resolve() == path.resolve():
            return
        self.open_wav(str(path), from_list=True)

    def generate(self) -> None:
        if self.wav is None:
            messagebox.showinfo('No WAV', 'Open a WAV file first.')
            return
        if self._generating:
            return
        self._generating = True
        if self._busy_depth == 0:
            self._enter_busy('Generating LPI…')
        else:
            self.var_busy.set('Generating LPI…')
            self.root.update_idletasks()
        self.var_status.set('Generating…')
        # Return to the Tk event loop before the heavy analyse + draw so
        # Linux/TkAgg does not swallow the canvas refresh inside the button click.
        self.root.after(1, self._generate_impl)

    def _generate_impl(self) -> None:
        try:
            try:
                config = self.current_config()
                self.result = analyse(self.wav, config)
            except Exception as exc:
                messagebox.showerror('Analyse error', str(exc))
                return
            self._redraw()
            self._fill_tree()
            self._set_idx_origin('generated')
            self.var_status.set(
                f'{Path(self.wav.path).name}  ·  {self.result.duration_s:.3f} s  ·  '
                f'T={self.result.revolution_s:.3f} s  ·  '
                f'{self.result.n_hops} hops  ·  {len(self.result.idx_events)} LPI events'
            )
            self._update_cursor_label()
            self._update_vu()
            self._refresh_canvas()
            self.root.after_idle(lambda: self._refresh_canvas())
        finally:
            self._generating = False
            if self._busy_depth > 0:
                self._leave_busy()

    def save_idx(self) -> None:
        if self.result is None or self.wav is None:
            messagebox.showinfo('Nothing to save', 'Generate a preview first.')
            return
        initial = str(default_idx_path(self.wav.path))
        path = filedialog.asksaveasfilename(
            title='Save .idx',
            initialdir=self._last_dir,
            initialfile=Path(initial).name,
            defaultextension='.idx',
            filetypes=[('LPI index', '*.idx'), ('CSV', '*.csv'), ('All files', '*.*')],
        )
        if not path:
            return
        write_idx(path, self.result.idx_events)
        self.var_status.set(f'Wrote {path} ({len(self.result.idx_events)} events)')
        if self.tracks:
            self._refresh_track_list()

    def _amplitude_editor_open(self) -> bool:
        dialog = self.amplitude_dialog
        if dialog is None:
            return False
        try:
            return bool(dialog.winfo_exists())
        except tk.TclError:
            self.amplitude_dialog = None
            return False

    def _amplitude_db(self) -> float:
        db = round(float(self.var_amp_db.get()), 1)
        return min(max(db, AMPLITUDE_DB_MIN), AMPLITUDE_DB_MAX)

    def _sync_amp_label(self, _value=None) -> None:
        self.var_amp_label.set(f'{self._amplitude_db():+.1f} dB')

    def open_amplitude_editor(self) -> None:
        if self.wav is None:
            messagebox.showinfo('No WAV', 'Open a WAV file first.')
            return
        if self._amplitude_editor_open():
            try:
                self.amplitude_dialog.lift()
                self.amplitude_dialog.focus_set()
            except tk.TclError:
                pass
            return
        if self.player.is_playing:
            self.stop_playback()
        self._amp_origin_s = float(self.cursor_s)
        self.var_amp_db.set(0.0)
        self._sync_amp_label()

        dialog = tk.Toplevel(self.root)
        dialog.title('Edit Amplitude')
        dialog.configure(bg=BG)
        dialog.resizable(False, False)
        try:
            dialog.transient(self.root)
        except tk.TclError:
            pass
        self.amplitude_dialog = dialog

        body = ttk.Frame(dialog, padding=(16, 12, 16, 12))
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='Whole-file gain').pack(anchor='w')
        scale_row = ttk.Frame(body)
        scale_row.pack(fill='x', pady=(8, 4))
        ttk.Label(scale_row, text='−4', foreground=MUTED).pack(side='left')
        self.amplitude_scale = ttk.Scale(
            scale_row,
            from_=AMPLITUDE_DB_MIN,
            to=AMPLITUDE_DB_MAX,
            variable=self.var_amp_db,
            command=self._sync_amp_label,
            length=240,
        )
        self.amplitude_scale.pack(side='left', fill='x', expand=True, padx=8)
        ttk.Label(scale_row, text='+4', foreground=MUTED).pack(side='right')
        ttk.Label(body, textvariable=self.var_amp_label, foreground=ACCENT).pack(anchor='center')
        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(14, 0))
        ttk.Button(buttons, text='Preview', command=self.preview_amplitude).pack(side='left')
        ttk.Button(buttons, text='Apply', command=self.apply_amplitude).pack(side='left', padx=(8, 0))
        ttk.Button(buttons, text='Cancel', command=self.cancel_amplitude).pack(side='right')

        dialog.protocol('WM_DELETE_WINDOW', self.cancel_amplitude)
        dialog.bind('<Escape>', lambda _e: self.cancel_amplitude())
        self._hook_button_space(dialog)
        try:
            if bool(self.root.winfo_viewable()):
                dialog.grab_set()
            else:
                dialog.withdraw()
        except tk.TclError:
            pass
        dialog.focus_set()

    def preview_amplitude(self) -> None:
        if self.wav is None:
            messagebox.showinfo('No WAV', 'Open a WAV file first.')
            return
        if not getattr(self.player.output, 'available', True):
            messagebox.showerror(
                'No audio output',
                'Install PortAudio and the sounddevice package:\n  pip install sounddevice',
            )
            return
        if self.player.is_playing:
            self.player.pause()
            self._cancel_playback_ticks()
        scaled = apply_amplitude_db(self.wav.channels, self._amplitude_db())
        try:
            self.player.load(scaled, self.wav.sample_rate)
            self.player.seek(self._amp_origin_s)
            self.player.play()
        except Exception as exc:
            messagebox.showerror('Playback error', str(exc))
            return
        self.cursor_s = self._amp_origin_s
        self._sync_playhead_visual()
        self._schedule_playback_ticks()

    def apply_amplitude(self) -> None:
        if self.wav is None:
            self._close_amplitude_dialog()
            return
        db = self._amplitude_db()
        origin = self._amp_origin_s
        changed = abs(db) >= 0.05
        if self.player.is_playing:
            self.player.pause()
        self._cancel_playback_ticks()
        try:
            if changed:
                scaled = apply_amplitude_db(self.wav.channels, db)
                write_wav(self.wav.path, self.wav, scaled)
                self.wav = load_wav(self.wav.path)
                self.var_status.set(
                    f'{Path(self.wav.path).name}  ·  amplitude {db:+.1f} dB'
                )
            self.player.load(self.wav.channels, self.wav.sample_rate)
        except Exception as exc:
            messagebox.showerror('Amplitude error', str(exc))
            try:
                self.player.load(self.wav.channels, self.wav.sample_rate)
            except Exception:
                pass
            return
        self._close_amplitude_dialog()
        self.cursor_s = origin
        if self.player.samples is not None:
            self.player.seek(origin)
        if changed:
            self.generate()
        else:
            self._sync_playhead_visual()

    def cancel_amplitude(self) -> None:
        origin = self._amp_origin_s
        if self.player.is_playing:
            self.player.pause()
        self._cancel_playback_ticks()
        if self.wav is not None:
            try:
                self.player.load(self.wav.channels, self.wav.sample_rate)
            except Exception:
                pass
        self._close_amplitude_dialog()
        if self.wav is not None:
            self._seek_to(origin)

    def _close_amplitude_dialog(self) -> None:
        dialog = self.amplitude_dialog
        self.amplitude_dialog = None
        self.amplitude_scale = None
        if dialog is None:
            return
        try:
            dialog.grab_release()
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    def _fill_tree(self) -> None:
        self._cancel_lpi_edit()
        try:
            self.tree.delete(*self.tree.get_children())
            if self.result is None:
                self.var_idx_events.set('.idx events')
                return
            for time_s, lpi in self.result.idx_events:
                self.tree.insert('', 'end', values=(f'{time_s:.3f}', lpi, '×'))
            count = len(self.result.idx_events)
            self.var_idx_events.set(f'{count} .idx events')
        finally:
            self._sync_idx_tree_highlight()

    def _apply_idx_events(self, events) -> None:
        if self.result is None:
            return
        self._apply_idx_events_payload = list(events)
        self._enter_busy('Updating LPI…')
        self.root.after(1, self._apply_idx_events_worker)

    def _apply_idx_events_worker(self) -> None:
        events = self._apply_idx_events_payload
        try:
            apply_idx_overlay(self.result, events)
            self._set_idx_origin('edited')
            self._redraw()
            self._fill_tree()
            self._update_cursor_label()
            if self.wav is not None:
                self.var_status.set(
                    f'{Path(self.wav.path).name}  ·  edited  ·  {len(events)} LPI events'
                )
        finally:
            self._leave_busy()

    def _tree_row_index(self, item_id: str) -> int:
        return list(self.tree.get_children()).index(item_id)

    def set_idx_lpi_at_index(self, index: int, lpi: int) -> None:
        if self.result is None:
            return
        events = set_idx_event_lpi(self.result.idx_events, index, lpi)
        self._apply_idx_events(events)

    def delete_idx_at_index(self, index: int) -> None:
        if self.result is None:
            return
        try:
            events = remove_idx_event_at(self.result.idx_events, index)
        except ValueError as exc:
            messagebox.showinfo('Cannot delete', str(exc))
            return
        self._apply_idx_events(events)

    def _cancel_lpi_edit(self) -> None:
        entry = self._lpi_edit_entry
        self._lpi_edit_entry = None
        self._lpi_edit_item = None
        if entry is not None:
            try:
                entry.destroy()
            except Exception:
                pass

    def _on_lpi_edit_return(self, _event=None):
        self._commit_lpi_edit()
        return 'break'

    def _on_lpi_edit_escape(self, _event=None):
        self._cancel_lpi_edit()
        return 'break'

    def _commit_lpi_edit(self) -> None:
        entry = self._lpi_edit_entry
        item = self._lpi_edit_item
        if entry is None or item is None or self.result is None:
            self._cancel_lpi_edit()
            return
        raw = entry.get().strip()
        self._cancel_lpi_edit()
        try:
            lpi = int(raw)
        except ValueError:
            messagebox.showerror('Invalid LPI', f'Not an integer: {raw!r}')
            return
        if lpi <= 0:
            messagebox.showerror('Invalid LPI', 'LPI must be a positive integer.')
            return
        try:
            index = self._tree_row_index(item)
        except ValueError:
            return
        self.set_idx_lpi_at_index(index, lpi)

    def _begin_lpi_edit(self, item_id: str) -> None:
        if self.result is None:
            return
        self._cancel_lpi_edit()
        values = self.tree.item(item_id, 'values')
        if len(values) < 2:
            return
        bbox = self.tree.bbox(item_id, column='lpi')
        if not bbox:
            return
        x, y, width, height = bbox
        entry = ttk.Entry(self.tree)
        entry.insert(0, str(values[1]))
        entry.select_range(0, tk.END)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        self._lpi_edit_entry = entry
        self._lpi_edit_item = item_id
        entry.bind('<Return>', self._on_lpi_edit_return)
        entry.bind('<Escape>', self._on_lpi_edit_escape)
        entry.bind('<FocusOut>', lambda _e: self.root.after(1, self._commit_lpi_edit))

    def _sync_idx_tree_highlight(self, *, scroll: bool = False) -> None:
        """Mark the active .idx row without changing Treeview selection (avoids seek on play)."""
        if self.result is None or not self.result.idx_events:
            return
        index = idx_event_index_at(self.result.idx_events, self.cursor_s)
        if index < 0:
            return
        children = self.tree.get_children()
        if index >= len(children):
            return
        active = children[index]
        for item in children:
            tags = ('idx_active',) if item == active else ()
            self.tree.item(item, tags=tags)
        if scroll:
            self.tree.see(active)

    def _refresh_canvas(self, reset_toolbar: bool = False) -> None:
        # Hide the playhead so the cached background can be blitted while playing
        # without a full matplotlib redraw (which underruns the audio callback).
        try:
            for line in self._cursor_lines:
                line.set_visible(False)
            self.canvas.draw()
            try:
                self._plot_bg = self.canvas.copy_from_bbox(self.figure.bbox)
            except Exception:
                self._plot_bg = None
            for line in self._cursor_lines:
                line.set_visible(True)
            if self._plot_bg is not None and self._cursor_lines:
                try:
                    for line in self._cursor_lines:
                        if line.axes is not None:
                            line.axes.draw_artist(line)
                    self.canvas.blit(self.figure.bbox)
                except Exception:
                    self.canvas.draw()
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass
        if reset_toolbar:
            try:
                self.plot_toolbar.update()
            except Exception:
                pass

    def _redraw(self) -> None:
        self.cursor_artists = discard_artists(self.cursor_artists)
        self._cursor_lines = []
        self._plot_bg = None
        for ax in (self.ax_wave, self.ax_env, self.ax_lpi):
            ax.clear()
        self._style_axes()
        if self.wav is None:
            self._refresh_canvas(reset_toolbar=True)
            return

        times, mins, maxs = downsample_minmax(self.wav.mid, self.wav.sample_rate)
        self.ax_wave.fill_between(times, mins, maxs, color='#6ea8c9', linewidth=0, alpha=0.9)
        self._apply_view_xlim()

        if self.result is not None:
            t = self.result.envelope_times
            self.ax_env.fill_between(t, 0, self.result.envelope, color='#c9a227', alpha=0.85, linewidth=0, label='peak')
            self.ax_env.plot(t, self.result.lookahead, color='#e07a5f', linewidth=1.1, label='lookahead max')
            self.ax_env.legend(loc='upper right', fontsize=7, facecolor=PANEL, edgecolor='#3a3d44', labelcolor=FG)
            env_max = max(
                float(self.result.envelope.max(initial=0.0)),
                float(self.result.lookahead.max(initial=0.0)),
                0.05,
            )
            self.ax_env.set_ylim(0, env_max * 1.08)

            step_t, step_l = self._step_series(self.result.idx_events, self.result.duration_s)
            self.ax_lpi.plot(t, self.result.lpi_dense, color='#3d6b45', linewidth=1.0, alpha=0.7)
            self.ax_lpi.step(step_t, step_l, where='post', color='#81c784', linewidth=1.8)
            lo = min(self.result.config.lpi_open, self.result.config.lpi_tight)
            hi = max(self.result.config.lpi_open, self.result.config.lpi_tight)
            pad = max(32, (hi - lo) * 0.06)
            self.ax_lpi.set_ylim(lo - pad, hi + pad)

            rev = self.result.revolution_s
            if rev > 0 and self.result.duration_s / rev <= 400:
                for ax in (self.ax_wave, self.ax_env, self.ax_lpi):
                    ax.xaxis.set_minor_locator(MultipleLocator(rev))
                    ax.tick_params(axis='x', which='minor', color='#3a3d44')

        self._draw_cursor()
        self._refresh_canvas(reset_toolbar=True)

    @staticmethod
    def _step_series(events, duration_s):
        if not events:
            return [0.0, duration_s], [0, 0]
        times = [e[0] for e in events] + [duration_s]
        values = [e[1] for e in events] + [events[-1][1]]
        return times, values

    def _clear_cursor(self) -> None:
        self.cursor_artists = discard_artists(self.cursor_artists)
        self._cursor_lines = []

    def _draw_cursor(self) -> None:
        self._clear_cursor()
        if self.wav is None:
            return
        duration = self.wav.mid.shape[0] / self.wav.sample_rate
        t = min(max(self.cursor_s, 0.0), duration)
        lookahead_end = t
        if self.result is not None:
            lookahead_end = min(duration, t + self.result.revolution_s * self.result.config.lookahead_revs)
        for ax in (self.ax_wave, self.ax_env, self.ax_lpi):
            if lookahead_end > t:
                span = ax.axvspan(t, lookahead_end, color='#e07a5f', alpha=0.08, linewidth=0, zorder=20)
                self.cursor_artists.append(span)
            line = ax.axvline(t, color='#f4f1de', linewidth=0.9, alpha=0.9, zorder=21)
            self.cursor_artists.append(line)
            self._cursor_lines.append(line)

    def _blit_playhead(self) -> None:
        if self.wav is None:
            return
        duration = self.wav.mid.shape[0] / self.wav.sample_rate
        t = min(max(self.cursor_s, 0.0), duration)
        if not self._cursor_lines:
            self._draw_cursor()
            self._refresh_canvas()
            return
        for line in self._cursor_lines:
            line.set_xdata([t, t])
        if self._plot_bg is None:
            self._refresh_canvas()
            return
        try:
            self.canvas.restore_region(self._plot_bg)
            for line in self._cursor_lines:
                if line.axes is not None:
                    line.axes.draw_artist(line)
            self.canvas.blit(self.figure.bbox)
            self.canvas.flush_events()
        except Exception:
            self._plot_bg = None

    def _on_plot_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if self._amplitude_editor_open():
            return
        if toolbar_blocks_cursor(getattr(self.plot_toolbar, 'mode', None)):
            return
        self._seek_to(float(event.xdata))

    def _on_tree_click(self, event) -> None:
        if self.tree.identify_region(event.x, event.y) != 'cell':
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        column = self.tree.identify_column(event.x)
        if column == '#1':
            time_s = parse_tree_time(self.tree.item(item, 'values'))
            if time_s is not None:
                self._seek_to(time_s)
            return
        if column == '#2':
            self._begin_lpi_edit(item)
            return 'break'
        if column == '#3':
            try:
                index = self._tree_row_index(item)
            except ValueError:
                return
            self.delete_idx_at_index(index)
            return 'break'

    def _update_cursor_label(self) -> None:
        if self.wav is None:
            self.var_cursor.set('t = —')
            self.var_now_lpi.set('LPI —')
            return
        t = max(0.0, self.cursor_s)
        parts = [f't = {t:.3f}s']
        lpi_text = 'LPI —'
        if self.result is not None and self.result.n_hops:
            idx = int(np.clip(np.searchsorted(self.result.envelope_times, t), 0, self.result.n_hops - 1))
            parts.append(f'peak {self.result.envelope[idx]:.3f}')
            parts.append(f'look {self.result.lookahead[idx]:.3f}')
            lpi_text = f'LPI {int(self.result.lpi_dense[idx])}'
        self.var_cursor.set('   ·   '.join(parts))
        self.var_now_lpi.set(lpi_text)


def run_gui(
    wav_path: Optional[str] = None,
    folder_path: Optional[str] = None,
    config: Optional[PreviewConfig] = None,
) -> int:
    root = tk.Tk()
    LpiPreviewApp(root, wav_path=wav_path, folder_path=folder_path, config=config)
    root.mainloop()
    return 0
