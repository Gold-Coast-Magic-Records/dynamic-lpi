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
    PreviewConfig,
    PreviewResult,
    WavData,
    analyse,
    default_env_path,
    default_idx_path,
    downsample_minmax,
    load_wav,
    write_env,
    write_idx,
)


BG = '#1b1d21'
PANEL = '#24262b'
FG = '#e6e6e6'
MUTED = '#9aa0a6'
ACCENT = '#c9a227'

_PAN_ZOOM_MODES = {'pan/zoom', 'zoom rect', 'pan', 'zoom'}


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


def toolbar_blocks_cursor(mode) -> bool:
    return str(mode or '').strip().lower() in _PAN_ZOOM_MODES


def _float_var(value: float) -> tk.DoubleVar:
    return tk.DoubleVar(value=value)


def _int_var(value: int) -> tk.IntVar:
    return tk.IntVar(value=value)


class LpiPreviewApp:
    def __init__(self, root: tk.Tk, wav_path: Optional[str] = None, config: Optional[PreviewConfig] = None):
        self.root = root
        self.config = config or PreviewConfig()
        self.wav: Optional[WavData] = None
        self.result: Optional[PreviewResult] = None
        self.cursor_s = 0.0
        self._last_dir = str(Path.home())
        self._generating = False

        self._build_vars()
        self._build_ui()
        self._init_plot()

        if wav_path:
            self.root.after(80, lambda: self.open_wav(wav_path, generate=True))

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
        self.var_cursor = tk.StringVar(value='t = —')
        self.var_idx_events = tk.StringVar(value='.idx events')

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
        ttk.Button(toolbar, text='Open WAV…', command=self.browse_wav).pack(side='left')
        ttk.Button(toolbar, text='Generate', command=self.generate).pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text='Save .idx…', command=self.save_idx).pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text='Save .env.csv…', command=self.save_env).pack(side='left', padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.var_cursor, foreground=MUTED).pack(side='right')

        controls = ttk.LabelFrame(self.root, text='Preview parameters', padding=(10, 6))
        controls.pack(fill='x', padx=10, pady=(0, 6))
        self._add_control(controls, 'RPM', self.var_rpm, 0, 0, width=8)
        ttk.Button(controls, text='33⅓', command=lambda: self.var_rpm.set(33.333)).grid(row=0, column=2, padx=(0, 12))
        ttk.Button(controls, text='45', command=lambda: self.var_rpm.set(45.0)).grid(row=0, column=3, padx=(0, 16))
        self._add_control(controls, 'LPI tight', self.var_tight, 0, 4, width=8)
        self._add_control(controls, 'LPI open', self.var_open, 0, 6, width=8)
        self._add_control(controls, 'Ref peak', self.var_ref, 0, 8, width=8)
        self._add_control(controls, 'LPF Hz', self.var_lpf, 1, 0, width=8)
        self._add_control(controls, 'Hop ms', self.var_hop, 1, 4, width=8)
        self._add_control(controls, 'Lookahead revs', self.var_lookahead, 1, 6, width=8)
        self._add_control(controls, 'Quantize step', self.var_quantize, 1, 8, width=8)
        self._add_control(controls, 'Min hold ms', self.var_hold, 2, 0, width=8)
        self._add_control(controls, 'Slew ms', self.var_slew, 2, 4, width=8)
        self._add_control(controls, 'LPI LPF Hz', self.var_lpi_lpf, 2, 6, width=8)
        ttk.Checkbutton(controls, text='Two-revolution preview', variable=self.var_two_rev).grid(
            row=3, column=0, columnspan=4, sticky='w', padx=(8, 0), pady=4
        )
        for col in range(10):
            controls.grid_columnconfigure(col, weight=0)
        controls.grid_columnconfigure(9, weight=1)

        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        plot_frame = ttk.Frame(body)
        plot_frame.pack(side='left', fill='both', expand=True)

        import matplotlib
        matplotlib.use('TkAgg')
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
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.cursor_artists = []
        self._tree_bind = None

        side = ttk.Frame(body, width=240)
        side.pack(side='right', fill='y', padx=(8, 0))
        side.pack_propagate(False)
        ttk.Label(side, textvariable=self.var_idx_events, foreground=MUTED).pack(anchor='w', pady=(0, 4))
        columns = ('time', 'lpi')
        self.tree = ttk.Treeview(side, columns=columns, show='headings', selectmode='browse')
        self.tree.heading('time', text='Time (s)')
        self.tree.heading('lpi', text='LPI')
        self.tree.column('time', width=100, anchor='e')
        self.tree.column('lpi', width=90, anchor='e')
        scroll = ttk.Scrollbar(side, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self._tree_bind = self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        status = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        status.pack(fill='x')
        ttk.Label(status, textvariable=self.var_status, foreground=MUTED).pack(anchor='w')

    def _add_control(self, parent, label, variable, row, column, width=8):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky='e', padx=(8, 4), pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, sticky='w', pady=3)
        entry.bind('<Return>', lambda _e: self.generate())

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
            self.open_wav(path, generate=True)

    def open_wav(self, path: str, generate: bool = True) -> None:
        try:
            self.wav = load_wav(path)
        except Exception as exc:
            messagebox.showerror('WAV error', str(exc))
            return
        self._last_dir = str(Path(path).parent)
        self.root.title(f'Dynamic LPI Preview — {Path(path).name}')
        self.var_status.set(
            f'Loaded {Path(path).name}: {self.wav.sample_rate} Hz, '
            f'{self.wav.channels.shape[1]} ch, {self.wav.mid.shape[0] / self.wav.sample_rate:.3f} s'
        )
        if generate:
            self.generate()
        else:
            self.result = None
            self._redraw()

    def generate(self) -> None:
        if self.wav is None:
            messagebox.showinfo('No WAV', 'Open a WAV file first.')
            return
        if self._generating:
            return
        self._generating = True
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
            self.var_status.set(
                f'{Path(self.wav.path).name}  ·  {self.result.duration_s:.3f} s  ·  '
                f'T={self.result.revolution_s:.3f} s  ·  '
                f'{self.result.n_hops} hops  ·  {len(self.result.idx_events)} LPI events'
            )
            self._update_cursor_label()
            self._refresh_canvas()
            self.root.after_idle(lambda: self._refresh_canvas())
        finally:
            self._generating = False

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

    def save_env(self) -> None:
        if self.result is None or self.wav is None:
            messagebox.showinfo('Nothing to save', 'Generate a preview first.')
            return
        initial = str(default_env_path(self.wav.path))
        path = filedialog.asksaveasfilename(
            title='Save envelope CSV',
            initialdir=self._last_dir,
            initialfile=Path(initial).name,
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')],
        )
        if not path:
            return
        write_env(path, self.result)
        self.var_status.set(f'Wrote {path}')

    def _fill_tree(self) -> None:
        if self._tree_bind is not None:
            self.tree.unbind('<<TreeviewSelect>>', self._tree_bind)
            self._tree_bind = None
        try:
            self.tree.delete(*self.tree.get_children())
            if self.result is None:
                self.var_idx_events.set('.idx events')
                return
            for time_s, lpi in self.result.idx_events:
                self.tree.insert('', 'end', values=(f'{time_s:.3f}', lpi))
            count = len(self.result.idx_events)
            self.var_idx_events.set(f'{count} .idx events')
        finally:
            self._tree_bind = self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

    def _refresh_canvas(self, reset_toolbar: bool = False) -> None:
        # Direct draw: draw_idle is coalesced with <Configure> on TkAgg and
        # silently drops redraws after Generate on some Linux desktops.
        self.canvas.draw()
        if reset_toolbar:
            try:
                self.plot_toolbar.update()
            except Exception:
                pass

    def _redraw(self) -> None:
        self.cursor_artists = discard_artists(self.cursor_artists)
        for ax in (self.ax_wave, self.ax_env, self.ax_lpi):
            ax.clear()
        self._style_axes()
        if self.wav is None:
            self._refresh_canvas(reset_toolbar=True)
            return

        times, mins, maxs = downsample_minmax(self.wav.mid, self.wav.sample_rate)
        self.ax_wave.fill_between(times, mins, maxs, color='#6ea8c9', linewidth=0, alpha=0.9)
        self.ax_wave.set_xlim(0, max(self.wav.mid.shape[0] / self.wav.sample_rate, 0.001))

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

    def _on_plot_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if toolbar_blocks_cursor(getattr(self.plot_toolbar, 'mode', None)):
            return
        self.cursor_s = float(event.xdata)
        self._update_cursor_label()
        self._draw_cursor()
        self._refresh_canvas()

    def _on_tree_select(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        time_s = parse_tree_time(self.tree.item(selection[0], 'values'))
        if time_s is None:
            return
        self.cursor_s = time_s
        self._update_cursor_label()
        self._draw_cursor()
        self._refresh_canvas()

    def _update_cursor_label(self) -> None:
        if self.wav is None:
            self.var_cursor.set('t = —')
            return
        t = max(0.0, self.cursor_s)
        parts = [f't = {t:.3f}s']
        if self.result is not None and self.result.n_hops:
            idx = int(np.clip(np.searchsorted(self.result.envelope_times, t), 0, self.result.n_hops - 1))
            parts.append(f'peak {self.result.envelope[idx]:.3f}')
            parts.append(f'look {self.result.lookahead[idx]:.3f}')
            parts.append(f'LPI {int(self.result.lpi_dense[idx])}')
        self.var_cursor.set('   ·   '.join(parts))


def run_gui(wav_path: Optional[str] = None, config: Optional[PreviewConfig] = None) -> int:
    root = tk.Tk()
    LpiPreviewApp(root, wav_path=wav_path, config=config)
    root.mainloop()
    return 0
