#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preview_config import (
    BUILTIN_DEFAULTS,
    default_config_path,
    load_preview_config_defaults,
    normalize_config_keys,
)


class PreviewConfigFileTests(unittest.TestCase):
    def test_default_config_path_beside_package(self):
        self.assertEqual(default_config_path(), ROOT / 'config.json')

    def test_builtin_defaults_include_lpi_endpoints(self):
        self.assertIn('lpi_open', BUILTIN_DEFAULTS)
        self.assertIn('lpi_tight', BUILTIN_DEFAULTS)

    def test_open_and_tight_lpi_aliases(self):
        raw = {'OPEN_LPI': 18750, 'TIGHT_LPI': 24500}
        normalized = normalize_config_keys(raw)
        self.assertEqual(normalized['lpi_open'], 18750)
        self.assertEqual(normalized['lpi_tight'], 24500)

    def test_snake_case_keys_accepted(self):
        raw = {'lpi_open': 18000, 'lpi_tight': 25000, 'quantize': 256}
        normalized = normalize_config_keys(raw)
        self.assertEqual(normalized['lpi_open'], 18000)
        self.assertEqual(normalized['quantize'], 256)

    def test_unknown_keys_ignored(self):
        normalized = normalize_config_keys({'OPEN_LPI': 19000, 'extra': 1})
        self.assertEqual(normalized, {'lpi_open': 19000})

    def test_load_from_json_file_via_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lpi.json'
            path.write_text(
                json.dumps({'OPEN_LPI': 18800, 'TIGHT_LPI': 24100, 'slew_ms': 300.0}),
                encoding='utf-8',
            )
            with patch.dict(os.environ, {'DYNAMIC_LPI_CONFIG': str(path)}, clear=False):
                loaded = load_preview_config_defaults()
            self.assertEqual(loaded['lpi_open'], 18800)
            self.assertEqual(loaded['lpi_tight'], 24100)
            self.assertEqual(loaded['slew_ms'], 300.0)
            self.assertEqual(loaded['hop_ms'], BUILTIN_DEFAULTS['hop_ms'])

    def test_missing_file_uses_builtin(self):
        with patch.dict(os.environ, {'DYNAMIC_LPI_CONFIG': '/nonexistent/lpi.json'}, clear=False):
            loaded = load_preview_config_defaults()
        self.assertEqual(loaded, BUILTIN_DEFAULTS)

    def test_committed_config_json_loads(self):
        path = ROOT / 'config.json'
        self.assertTrue(path.is_file(), 'config.json should exist beside preview_config.py')
        loaded = load_preview_config_defaults(config_path=path)
        self.assertIsInstance(loaded['lpi_open'], int)
        self.assertIsInstance(loaded['lpi_tight'], int)
        self.assertGreater(loaded['lpi_tight'], loaded['lpi_open'])


if __name__ == '__main__':
    unittest.main()