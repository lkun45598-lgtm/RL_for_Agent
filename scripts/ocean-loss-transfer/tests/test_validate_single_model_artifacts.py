from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_loss import validate_single_model  # noqa: E402


class ValidateSingleModelArtifactsTests(unittest.TestCase):
    def test_timeout_returns_partial_metrics_and_writes_curve_artifact(self) -> None:
        combined_output = '\n'.join(
            [
                '__event__{"event":"training_start","total_epochs":15}__event__',
                '__event__{"event":"epoch_train","epoch":0,"metrics":{"train_loss":0.42}}__event__',
                '__event__{"event":"epoch_valid","epoch":4,"metrics":{"valid_loss":0.8,"psnr":12.3,"ssim":0.71,"rmse":0.31}}__event__',
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            loss_file = Path(temp_dir) / 'candidate_loss.py'
            loss_file.write_text(
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).pow(2).mean()\n',
                encoding='utf-8',
            )

            with patch('validate_loss.load_formula_spec', return_value=None), patch(
                'validate_loss._copy_loss_to_sandbox',
                return_value=None,
            ), patch(
                'validate_loss._resolve_sandbox_override_dir',
                return_value=None,
            ), patch(
                'validate_loss._prepare_config_path',
                return_value='/tmp/fake_swinir.yaml',
            ), patch(
                'validate_loss._run_subprocess_with_combined_log',
                return_value={
                    'timed_out': True,
                    'returncode': -9,
                    'combined_output': combined_output,
                },
            ):
                result = validate_single_model(str(loss_file), dataset_root='/tmp/data')

        self.assertFalse(result['passed'])
        self.assertEqual(result['error'], 'timeout')
        self.assertEqual(result['partial_metrics']['val_ssim'], 0.71)
        self.assertEqual(result['partial_metrics']['val_psnr'], 12.3)
        self.assertEqual(result['training_curve']['last_epoch'], 4)

        artifact_paths = result['artifact_paths']
        curve_path = Path(artifact_paths['training_curve_path'])
        self.assertTrue(curve_path.exists())
        curve = json.loads(curve_path.read_text(encoding='utf-8'))
        self.assertEqual(curve['last_epoch'], 4)
        self.assertEqual(curve['trend'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
