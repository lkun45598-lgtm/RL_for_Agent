from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_loss import validate_smoke  # noqa: E402


class ValidateSmokeTests(unittest.TestCase):
    def test_validate_smoke_rejects_detached_zero_mask_loss(self) -> None:
        code = """
import torch

def sandbox_loss(pred, target, mask=None, **kwargs):
    if mask is not None and float(mask.float().sum().item()) == 0.0:
        return pred.new_zeros(())
    return (pred - target).pow(2).mean()
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'candidate_loss.py'
            path.write_text(code.strip() + '\n', encoding='utf-8')
            result = validate_smoke(str(path))

        self.assertFalse(result['passed'])
        self.assertEqual(result['error'], 'detached_zero_mask_loss')

    def test_validate_smoke_accepts_graph_connected_zero_mask_loss(self) -> None:
        code = """
import torch

def sandbox_loss(pred, target, mask=None, **kwargs):
    if mask is not None and float(mask.float().sum().item()) == 0.0:
        return pred.sum() * 0.0
    return (pred - target).pow(2).mean()
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'candidate_loss.py'
            path.write_text(code.strip() + '\n', encoding='utf-8')
            result = validate_smoke(str(path))

        self.assertTrue(result['passed'])
        self.assertTrue(result['smoke_detail']['zero_mask_test_passed'])


if __name__ == '__main__':
    unittest.main()
