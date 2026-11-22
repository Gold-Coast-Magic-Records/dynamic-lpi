"""Load Dynamic LPI default parameters from config.json (override with DYNAMIC_LPI_CONFIG)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent

BUILTIN_DEFAULTS: dict[str, Any] = {
    'rpm': 33.333,
    'hop_ms': 20.0,
    'lpf_hz': 150.0,
    'lpi_tight': 24000,
    'lpi_open': 18900,
    'reference_peak': 0.40,
    'quantize': 512,
    'min_hold_ms': 400.0,
    'slew_ms': 250.0,
    'lookahead_revs': 1.0,
    'two_rev': True,
    'lpi_lpf_hz': 0.2,
}

_CONFIG_FIELD_NAMES = frozenset(BUILTIN_DEFAULTS)

_KEY_ALIASES: dict[str, str] = {
    'OPEN_LPI': 'lpi_open',
    'TIGHT_LPI': 'lpi_tight',
    'LPI_OPEN': 'lpi_open',
    'LPI_TIGHT': 'lpi_tight',
}


def default_config_path() -> Path:
    return ROOT / 'config.json'


def _config_path_from_env() -> Path:
    override = os.environ.get('DYNAMIC_LPI_CONFIG', '').strip()
    if override:
        return Path(override)
    return default_config_path()


def normalize_config_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map JSON keys (including OPEN_LPI / TIGHT_LPI) to PreviewConfig field names."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        field = _KEY_ALIASES.get(key, key)
        if field not in _CONFIG_FIELD_NAMES:
            continue
        out[field] = value
    return out


def _coerce_value(field: str, value: Any) -> Any:
    if field == 'two_rev':
        return bool(value)
    if field in {'lpi_tight', 'lpi_open', 'quantize'}:
        return int(value)
    if field in {
        'rpm', 'hop_ms', 'lpf_hz', 'reference_peak',
        'min_hold_ms', 'slew_ms', 'lookahead_revs', 'lpi_lpf_hz',
    }:
        return float(value)
    return value


def load_preview_config_defaults(config_path: Path | None = None) -> dict[str, Any]:
    """Merge config.json over built-in defaults. Missing file keeps built-ins only."""
    merged = BUILTIN_DEFAULTS.copy()
    path = config_path if config_path is not None else _config_path_from_env()
    if not path.is_file():
        return merged
    with path.open(encoding='utf-8') as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f'Dynamic LPI config must be a JSON object: {path}')
    for field, value in normalize_config_keys(raw).items():
        merged[field] = _coerce_value(field, value)
    return merged