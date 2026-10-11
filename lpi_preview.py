#!/usr/bin/env python3
"""
Generate a variable-LPI .idx file from a WAV.

CLI (batch / programmatic):
    python lpi_preview.py track.wav
    python lpi_preview.py track.wav -o track.idx --rpm 33.333 --lpi-tight 22000 --lpi-open 19250

GUI (debug the waveform against the peak / LPI curve):
    python lpi_preview.py --gui /path/to/Side/A
    python lpi_preview.py /path/to/Side/A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (
    PreviewConfig,
    analyse,
    default_env_path,
    default_idx_path,
    load_wav,
    write_env,
    write_idx,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='lpi_preview',
        description='Analyse a WAV and write a sparse TIME,LPI .idx file for dynamic cutting pitch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='With no WAV (or with --gui) the debug plotter opens. '
               'The .idx format is seconds,lpi with no header, matching LpiManager.',
    )
    parser.add_argument('path', nargs='?', help='Input WAV path, or a folder of 1.wav / 2.wav / … for the GUI')
    parser.add_argument('--gui', action='store_true', help='Open the waveform / LPI debug GUI')
    parser.add_argument('-o', '--output', help='Output .idx path (default: alongside the WAV)')
    parser.add_argument('--write-env', action='store_true', help='Also write a dense envelope CSV for inspection')
    parser.add_argument('--env-output', help='Envelope CSV path (implies --write-env)')
    parser.add_argument('--rpm', type=float, default=33.333, help='Platter speed; sets lookahead window 60/rpm')
    parser.add_argument('--hop-ms', type=float, default=20.0, help='Envelope hop size in milliseconds')
    parser.add_argument('--lpf', type=float, default=100.0, dest='lpf_hz', help='Low-pass Hz before peaking (0 disables)')
    parser.add_argument('--lpi-tight', type=int, default=22000, help='G2 L for quiet / narrow land (higher = tighter)')
    parser.add_argument('--lpi-open', type=int, default=19250, help='G2 L for loud / wide land (lower = faster feed)')
    parser.add_argument('--reference-peak', type=float, default=0.40, help='Envelope level that saturates to Open LPI; quieter tracks still stretch their peak to Open')
    parser.add_argument('--quantize', type=int, default=1024, help='Minimum LPI step before a new .idx event; larger = fewer updates')
    parser.add_argument('--min-hold-ms', type=float, default=100.0, help='Ignore LPI changes shorter than this')
    parser.add_argument('--slew-ms', type=float, default=250.0, help='Time to slew across the full LPI range; 0 disables')
    parser.add_argument('--lookahead-revs', type=float, default=1.0, help='Lookahead window in revolutions')
    parser.add_argument('--two-rev', action=argparse.BooleanOptionalAction, default=True, help='Use next-rev + following-rev (Neumann-style). Default: on.')
    parser.add_argument('--lpi-lpf', type=float, default=0.2, dest='lpi_lpf_hz', help='Low-pass Hz on the LPI curve after mapping; 0 disables')
    parser.add_argument('-q', '--quiet', action='store_true', help='Print only the written .idx path')
    return parser


def config_from_args(args: argparse.Namespace) -> PreviewConfig:
    return PreviewConfig(
        rpm=args.rpm,
        hop_ms=args.hop_ms,
        lpf_hz=args.lpf_hz,
        lpi_tight=args.lpi_tight,
        lpi_open=args.lpi_open,
        reference_peak=args.reference_peak,
        quantize=args.quantize,
        min_hold_ms=args.min_hold_ms,
        slew_ms=args.slew_ms,
        lookahead_revs=args.lookahead_revs,
        two_rev=args.two_rev,
        lpi_lpf_hz=args.lpi_lpf_hz,
    )


def convert_wav(wav_path: str, args: argparse.Namespace) -> Path:
    wav = load_wav(wav_path)
    result = analyse(wav, config_from_args(args))
    idx_path = Path(args.output) if args.output else default_idx_path(wav_path)
    write_idx(idx_path, result.idx_events)

    env_path = None
    if args.env_output or args.write_env:
        env_path = Path(args.env_output) if args.env_output else default_env_path(wav_path)
        write_env(env_path, result)

    if args.quiet:
        print(idx_path)
        return idx_path

    channels = wav.channels.shape[1]
    print(f'Loaded {wav.path}: {wav.sample_rate} Hz, {channels} ch, {result.duration_s:.3f} s')
    print(f'Revolution: {result.revolution_s:.3f} s ({args.rpm:g} RPM), hop {result.hop_s * 1000:.1f} ms')
    print(f'Envelope hops: {result.n_hops}')
    print(f'IDX events: {len(result.idx_events)}')
    print(f'Wrote {idx_path}')
    if env_path is not None:
        print(f'Wrote {env_path}')
    return idx_path


def resolve_gui_sources(path: str | None) -> tuple[str | None, str | None]:
    """Return (wav_path, folder_path) for GUI launch."""
    if not path:
        return None, None
    resolved = Path(path)
    if resolved.is_dir():
        return None, str(resolved)
    suffix = resolved.suffix.lower()
    if suffix in {'.wav', '.wave'}:
        return str(resolved), None
    # Side folders such as …/Disc/A/Side/A must not be opened as WAV files,
    # even when is_dir() is false (stale path, or CLI parsed before mount).
    return None, str(resolved)


def launch_gui(args: argparse.Namespace) -> int:
    from gui import run_gui
    wav_path, folder_path = resolve_gui_sources(args.path)
    return run_gui(wav_path=wav_path, folder_path=folder_path, config=config_from_args(args))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else None
    if path is not None and path.is_dir():
        return launch_gui(args)
    if args.gui or path is None:
        return launch_gui(args)

    if not path.is_file():
        parser.error(f'WAV not found: {path}')

    convert_wav(str(path), args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
