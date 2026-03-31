"""
@file test_attempt_state.py

@description Regression tests for attempt result and repair bookkeeping helpers.
@author kongzhiquan
@contributors kongzhiquan
@date 2026-03-30
@version 1.0.0

@changelog
  - 2026-03-30 kongzhiquan: v1.0.0 add coverage for persisted edit scope and evidence refs
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.attempts.attempt_state import (  # noqa: E402
    attach_repair_artifact,
    build_attempt_result,
    build_code_generation_failure_result,
    build_initial_repair_record,
    build_reward_summary,
    layer_rank,
    mark_repair_reverted,
    should_revert_repair,
    snapshot_path,
)


class AttemptStateTests(unittest.TestCase):
    def test_snapshot_path_and_layer_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_dir = Path(temp_dir)
            path = snapshot_path(attempt_dir, 'candidate_loss_before_repair', 2, '.py')

        self.assertTrue(str(path).endswith('candidate_loss_before_repair_round_2.py'))
        self.assertEqual(layer_rank('layer2', {'layer1': 1, 'layer2': 2}), 2)

    def test_build_code_generation_failure_result_shapes_default_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_dir = Path(temp_dir)
            code_path = attempt_dir / 'candidate_loss.py'
            result = build_code_generation_failure_result(
                attempt_id=3,
                attempt_spec={
                    'name': 'Attempt 3',
                    'notes': 'test',
                    'files_to_edit': ['candidate_loss.py'],
                    'required_edit_paths': ['sandbox_model_adapter.py'],
                    'evidence_refs': ['validator.layer3'],
                },
                attempt_dir=attempt_dir,
                code_path=code_path,
                baseline={'model': 'swinir'},
                max_agent_repair_rounds=3,
                error_text='generation failed',
            )

        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['stop_layer'], 'code_generation')
        self.assertEqual(result['reward_summary']['primary_metric'], None)
        self.assertEqual(result['metadata']['notes'], 'test')
        self.assertEqual(result['files_to_edit'], ['candidate_loss.py'])
        self.assertEqual(result['required_edit_paths'], ['sandbox_model_adapter.py'])
        self.assertEqual(result['evidence_refs'], ['validator.layer3'])

    def test_repair_record_and_reversion_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_dir = Path(temp_dir)
            pre_path = attempt_dir / 'before.py'
            restored_path = attempt_dir / 'restored.py'
            record = build_initial_repair_record(
                round_number=1,
                trigger_stop_layer='layer3',
                failure_feedback={'error': 'nan'},
                repair_info={'status': 'success'},
                pre_repair_code_path=pre_path,
            )
            attach_repair_artifact(record, key='post_repair_code_path', path=attempt_dir / 'after.py')
            record['post_stop_layer'] = 'layer2'
            self.assertTrue(
                should_revert_repair(
                    trigger_stop_layer='layer3',
                    post_stop_layer='layer2',
                    layer_order={'layer2': 2, 'layer3': 3},
                )
            )
            mark_repair_reverted(record, restored_code_path=restored_path)

        self.assertEqual(record['status'], 'reverted_regression')
        self.assertTrue(record['reverted'])
        self.assertIn('restored_code_path', record['artifacts'])

    def test_build_attempt_result_uses_metric_and_alignment_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_dir = Path(temp_dir)
            code_path = attempt_dir / 'candidate_loss.py'
            result = build_attempt_result(
                attempt_id=7,
                attempt_spec={
                    'name': 'Attempt 7',
                    'notes': 'ok',
                    'files_to_edit': ['candidate_loss.py'],
                    'required_edit_paths': ['sandbox_model_adapter.py'],
                    'evidence_refs': ['result.layer4'],
                },
                source_kind='agent_code',
                attempt_dir=attempt_dir,
                code_path=code_path,
                validation={
                    'layer1': {'passed': True},
                    'layer2': {'passed': True},
                    'formula_alignment': {'passed': True},
                },
                stop_layer=None,
                metrics={'swinir': 0.72},
                baseline={'ssim_mean': 0.66},
                repair_rounds=[{'round': 1}],
                run_training=True,
                formula_spec_path='formula.json',
                generation_info={'status': 'success'},
                repair_info={'status': 'success'},
                max_agent_repair_rounds=3,
                validation_error_text_fn=lambda stop_layer, validation: None,
                compute_baseline_delta_fn=lambda metrics, baseline: 0.06,
                extract_primary_metric_fn=lambda metrics: ('swinir', 0.72),
            )

        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['passed_formula_alignment'])
        self.assertEqual(result['files_to_edit'], ['candidate_loss.py'])
        self.assertEqual(result['required_edit_paths'], ['sandbox_model_adapter.py'])
        self.assertEqual(result['evidence_refs'], ['result.layer4'])
        self.assertEqual(
            result['reward_summary'],
            build_reward_summary(
                metric_name='swinir',
                metric_value=0.72,
                baseline_delta=0.06,
                passed=True,
                stop_layer=None,
                validation={
                    'layer1': {'passed': True},
                    'layer2': {'passed': True},
                    'formula_alignment': {'passed': True},
                },
                metrics={'swinir': 0.72},
                repair_rounds=[{'round': 1}],
                passed_static=True,
                passed_smoke=True,
                passed_formula_alignment=True,
            ),
        )


if __name__ == '__main__':
    unittest.main()
