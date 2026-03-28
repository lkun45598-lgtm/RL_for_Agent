from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.formula.sandbox_adapter_bridge import build_config_with_adapter  # noqa: E402


class SandboxAdapterBridgeTests(unittest.TestCase):
    def test_build_config_forces_single_worker_and_disables_amp(self) -> None:
        base_config = {
            'data': {
                'dataset_root': '/tmp/old',
                'num_workers': 2,
            },
            'train': {
                'use_amp': True,
            },
            'model': {
                'name': 'SwinIR',
                'in_channels': 2,
                'out_channels': 2,
            },
        }

        with patch(
            'loss_transfer.formula.sandbox_adapter_bridge._infer_dataset_metadata',
            return_value={
                'dyn_vars': ['uo', 'vo'],
                'shape': [400, 800],
                'sample_factor': 4,
                'num_channels': 2,
            },
        ):
            result = build_config_with_adapter(
                base_config,
                formula_spec=None,
                dataset_root='/tmp/new',
            )

        self.assertEqual(result['data']['dataset_root'], '/tmp/new')
        self.assertEqual(result['data']['num_workers'], 0)
        self.assertFalse(result['train']['use_amp'])


if __name__ == '__main__':
    unittest.main()
