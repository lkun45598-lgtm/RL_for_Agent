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

from loss_transfer.attempts.attempt_executor import _resolve_attempt_code, execute_attempt  # noqa: E402


class AttemptExecutorIntegrationTests(unittest.TestCase):
    def _baseline(self) -> dict[str, float | str]:
        return {
            'model': 'swinir',
            'ssim_mean': 0.66,
            'viable_threshold': 0.65,
            'improvement_threshold': 0.67,
        }

    def _read_event_types(self, output_dir: Path) -> list[str]:
        trajectory_path = output_dir / 'trajectory.jsonl'
        if not trajectory_path.exists():
            return []
        return [
            json.loads(line)['event_type']
            for line in trajectory_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def test_resolve_attempt_code_falls_back_to_objective_when_relative_code_path_is_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_code_path = Path(temp_dir) / 'attempt_1' / 'candidate_loss.py'
            generated_code = (
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).abs().mean()\n'
            )

            def fake_generate_candidate_loss(**kwargs):
                code_path = Path(kwargs['output_code_path'])
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(generated_code, encoding='utf-8')
                return {
                    'status': 'success',
                    'code_path': str(code_path),
                }

            with patch(
                'loss_transfer.attempts.attempt_executor.generate_candidate_loss',
                side_effect=fake_generate_candidate_loss,
            ) as mock_generate:
                code, source_kind, generation = _resolve_attempt_code(
                    {
                        'kind': 'agent_code',
                        'code_path': 'candidate_loss.py',
                        'objective': 'Write the candidate loss implementation',
                    },
                    None,
                    task_context_path='/tmp/task_context.json',
                    output_code_path=str(output_code_path),
                    agent_service_url='http://127.0.0.1:8888',
                    agent_api_key='secret_key',
                )

            self.assertEqual(code, generated_code)
            self.assertEqual(source_kind, 'agent_code')
            self.assertEqual(generation['status'], 'success')
            mock_generate.assert_called_once()

    def test_execute_attempt_records_successful_repair_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'experiment'
            output_dir.mkdir(parents=True, exist_ok=True)
            initial_code = (
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).pow(2).mean()\n'
            )
            repaired_code = (
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).abs().mean()\n'
            )
            validation_before = (
                {
                    'layer1': {'passed': True},
                    'layer2': {'passed': True},
                    'layer3': {'passed': False, 'detail': 'timeout'},
                },
                'layer3',
                {'val_ssim': 0.61},
            )
            validation_after = (
                {
                    'layer1': {'passed': True},
                    'layer2': {'passed': True},
                    'layer3': {'passed': True},
                    'layer4': {'passed': True},
                },
                None,
                {'swinir': 0.72},
            )
            eval_results = iter([validation_before, validation_after])

            def fake_evaluate(*args, **kwargs):
                return next(eval_results)

            def fake_repair(**kwargs):
                code_path = Path(kwargs['code_path'])
                code_path.write_text(repaired_code, encoding='utf-8')
                response_path = code_path.parent / 'repair_response.json'
                repair_plan_path = code_path.parent / 'repair_plan_round_1.json'
                response_path.write_text(json.dumps({'status': 'success'}), encoding='utf-8')
                repair_plan_path.write_text(
                    json.dumps(
                        {
                            'failure_hypothesis': 'layer3 timeout comes from unstable residual weighting',
                            'planned_changes': ['switch to L1 residual', 'preserve current adapter path'],
                            'target_metric': 'val_ssim',
                            'success_criteria': 'pass layer3 and improve validation SSIM',
                            'fallback_plan': 'add a reconstruction anchor term if metrics still collapse',
                            'evidence_refs': ['validator.layer3'],
                        }
                    ),
                    encoding='utf-8',
                )
                return {
                    'status': 'success',
                    'agent_response_path': str(response_path),
                    'repair_plan_path': str(repair_plan_path),
                    'repair_plan_summary': {
                        'failure_hypothesis': 'layer3 timeout comes from unstable residual weighting',
                        'target_metric': 'val_ssim',
                        'planned_change_count': 2,
                    },
                }

            with patch('loss_transfer.attempts.attempt_executor._load_baseline_thresholds', return_value=self._baseline()), patch(
                'loss_transfer.attempts.attempt_executor._resolve_attempt_code',
                return_value=(initial_code, 'agent_code', {'status': 'success', 'agent_id': 'gen-1'}),
            ), patch('loss_transfer.attempts.attempt_executor._evaluate_candidate', side_effect=fake_evaluate), patch(
                'loss_transfer.attempts.attempt_executor._maybe_repair_candidate_code',
                side_effect=fake_repair,
            ):
                result = execute_attempt(
                    'paper_success',
                    1,
                    {
                        'name': 'Attempt 1',
                        'kind': 'agent_code',
                        'run_training': True,
                        'strategy_delta': {
                            'previous_attempt_id': 0,
                            'why_previous_failed': 'bootstrap candidate was unavailable',
                            'what_changes_now': ['repair timeout with a smoother loss surface'],
                            'why_not_repeat_previous': 'the pre-repair candidate already timed out',
                            'expected_signal': 'layer3 should pass and validation SSIM should improve',
                        },
                    },
                    output_dir=str(output_dir),
                )

            attempt_dir = output_dir / 'attempt_1'
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['repair_rounds']), 1)
            self.assertEqual(result['repair_rounds'][0]['status'], 'success')
            self.assertEqual(result['reward_summary']['primary_metric_name'], 'swinir')
            self.assertEqual(result['reward_summary']['primary_metric'], 0.72)
            self.assertEqual(result['reward_summary']['stage_score'], 6)
            self.assertTrue(result['reward_summary']['passed'])
            self.assertEqual(result['reward_summary']['repair_rounds_used'], 1)
            self.assertEqual(result['strategy_delta']['previous_attempt_id'], 0)
            self.assertTrue((attempt_dir / 'candidate_loss_after_repair_round_1.py').exists())
            self.assertTrue((attempt_dir / 'agent_code_repair_response_round_1.json').exists())
            self.assertEqual(
                result['repair_rounds'][0]['artifacts']['repair_plan_path'],
                str(attempt_dir / 'repair_plan_round_1.json'),
            )
            self.assertEqual((attempt_dir / 'candidate_loss.py').read_text(encoding='utf-8'), repaired_code)
            self.assertEqual(
                self._read_event_types(output_dir),
                [
                    'attempt_started',
                    'attempt_repair_started',
                    'attempt_repair_finished',
                    'attempt_finished',
                ],
            )

    def test_execute_attempt_reverts_regressive_repair_and_keeps_failure_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'experiment'
            output_dir.mkdir(parents=True, exist_ok=True)
            initial_code = (
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return pred.mean() - target.mean()\n'
            )
            bad_repair_code = (
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return torch.tensor(float("nan"))\n'
            )
            validation_before = (
                {
                    'layer1': {'passed': True},
                    'layer2': {'passed': True},
                    'layer3': {'passed': False, 'detail': 'timeout'},
                },
                'layer3',
                {'val_ssim': 0.6},
            )
            validation_regressed = (
                {
                    'layer1': {'passed': True},
                    'layer2': {'passed': False, 'detail': 'nan in smoke'},
                },
                'layer2',
                {},
            )
            eval_results = iter([validation_before, validation_regressed])
            repair_calls = {'count': 0}

            def fake_evaluate(*args, **kwargs):
                return next(eval_results)

            def fake_repair(**kwargs):
                repair_calls['count'] += 1
                if repair_calls['count'] == 1:
                    code_path = Path(kwargs['code_path'])
                    code_path.write_text(bad_repair_code, encoding='utf-8')
                    response_path = code_path.parent / 'repair_response.json'
                    response_path.write_text(json.dumps({'status': 'success'}), encoding='utf-8')
                    return {
                        'status': 'success',
                        'agent_response_path': str(response_path),
                    }
                return None

            with patch('loss_transfer.attempts.attempt_executor._load_baseline_thresholds', return_value=self._baseline()), patch(
                'loss_transfer.attempts.attempt_executor._resolve_attempt_code',
                return_value=(initial_code, 'agent_code', {'status': 'success', 'agent_id': 'gen-2'}),
            ), patch('loss_transfer.attempts.attempt_executor._evaluate_candidate', side_effect=fake_evaluate), patch(
                'loss_transfer.attempts.attempt_executor._maybe_repair_candidate_code',
                side_effect=fake_repair,
            ):
                result = execute_attempt(
                    'paper_revert',
                    2,
                    {'name': 'Attempt 2', 'kind': 'agent_code', 'run_training': True},
                    output_dir=str(output_dir),
                )

            attempt_dir = output_dir / 'attempt_2'
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['passed'])
            self.assertEqual(result['stop_layer'], 'layer3')
            self.assertEqual(len(result['repair_rounds']), 1)
            self.assertEqual(result['repair_rounds'][0]['status'], 'reverted_regression')
            self.assertTrue(result['repair_rounds'][0]['reverted'])
            self.assertTrue((attempt_dir / 'candidate_loss_restored_round_1.py').exists())
            self.assertEqual((attempt_dir / 'candidate_loss.py').read_text(encoding='utf-8'), initial_code)
            self.assertEqual(
                self._read_event_types(output_dir),
                [
                    'attempt_started',
                    'attempt_repair_started',
                    'attempt_repair_finished',
                    'attempt_repair_started',
                    'attempt_repair_finished',
                    'attempt_finished',
                ],
            )


if __name__ == '__main__':
    unittest.main()
