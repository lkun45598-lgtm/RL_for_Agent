from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from decision_trace import (  # noqa: E402
    build_decision_trace_record,
    build_rl_decision_dataset_record,
    export_rl_dataset_from_decision_trace,
    write_decision_trace,
)


class DecisionTraceTests(unittest.TestCase):
    def test_build_decision_trace_record_carries_previous_attempt_context(self) -> None:
        task_context = {
            'paths': {
                'task_context_path': '/tmp/task_context.json',
            },
            'integration_assessment': {
                'recommended_path': 'adapter_wrapper',
                'requires_model_changes': False,
                'loss_only_pipeline_viable': False,
            },
            'formula_interface': {
                'requires_model_changes': True,
            },
        }
        previous_attempt = {
            'attempt_id': 1,
            'stop_layer': 'layer4',
            'error': 'ssim collapse',
            'reward_summary': {
                'primary_metric': 0.61,
                'stage_score': 5,
            },
        }
        attempt = {
            'attempt_id': 2,
            'name': 'Follow-up attempt',
            'kind': 'agent_code',
            'objective': 'Add a reconstruction anchor.',
            'files_to_edit': ['candidate_loss.py'],
            'required_edit_paths': ['sandbox_model_adapter.py'],
            'strategy_delta': {
                'previous_attempt_id': 1,
                'why_previous_failed': 'SSIM collapsed after the first transfer.',
                'what_changes_now': ['add a reconstruction anchor'],
                'why_not_repeat_previous': 'the previous strategy already failed',
                'expected_signal': 'validation SSIM should improve',
            },
            'reward_summary': {
                'primary_metric_name': 'swinir',
                'primary_metric': 0.7,
                'stage_score': 6,
            },
            'status': 'passed',
            'passed': True,
            'stop_layer': None,
            'error': None,
            'metrics': {'swinir': 0.7},
            'paths': {
                'attempt_dir': '/tmp/attempt_2',
                'result_path': '/tmp/attempt_2/result.json',
                'code_path': '/tmp/attempt_2/candidate_loss.py',
            },
        }

        record = build_decision_trace_record(
            paper_slug='demo-paper',
            task_context=task_context,
            attempt=attempt,
            analysis_plan_path='/tmp/analysis_plan.json',
            trajectory_path='/tmp/trajectory.jsonl',
            previous_attempt=previous_attempt,
        )

        self.assertEqual(record['state']['previous_attempt_id'], 1)
        self.assertEqual(record['state']['previous_stop_layer'], 'layer4')
        self.assertEqual(record['state']['previous_primary_metric'], 0.61)
        self.assertEqual(record['action']['strategy_delta']['expected_signal'], 'validation SSIM should improve')
        self.assertEqual(record['reward']['stage_score'], 6)

    def test_write_decision_trace_writes_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = {
                'paths': {'task_context_path': str(experiment_dir / 'task_context.json')},
                'integration_assessment': {'recommended_path': 'loss_only'},
            }
            attempts = [
                {
                    'attempt_id': 1,
                    'name': 'Attempt 1',
                    'kind': 'agent_code',
                    'reward_summary': {'primary_metric_name': 'val_ssim', 'primary_metric': 0.61, 'stage_score': 5},
                    'status': 'failed',
                    'passed': False,
                    'stop_layer': 'layer4',
                    'error': 'below threshold',
                    'metrics': {'val_ssim': 0.61},
                    'paths': {'attempt_dir': str(experiment_dir / 'attempt_1')},
                },
                {
                    'attempt_id': 2,
                    'name': 'Attempt 2',
                    'kind': 'agent_code',
                    'strategy_delta': {
                        'previous_attempt_id': 1,
                        'why_previous_failed': 'below threshold',
                        'what_changes_now': ['stabilize the objective'],
                        'why_not_repeat_previous': 'the previous strategy underperformed',
                        'expected_signal': 'validation SSIM should increase',
                    },
                    'reward_summary': {'primary_metric_name': 'val_ssim', 'primary_metric': 0.67, 'stage_score': 6},
                    'status': 'passed',
                    'passed': True,
                    'stop_layer': None,
                    'error': None,
                    'metrics': {'val_ssim': 0.67},
                    'paths': {'attempt_dir': str(experiment_dir / 'attempt_2')},
                },
            ]

            result = write_decision_trace(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
                analysis_plan_path=str(experiment_dir / 'analysis_plan.json'),
                trajectory_path=str(experiment_dir / 'trajectory.jsonl'),
                attempts=attempts,
            )

            trace_path = Path(result['decision_trace_path'])
            rl_dataset_path = Path(result['rl_dataset_path'])
            self.assertTrue(trace_path.exists())
            self.assertTrue(rl_dataset_path.exists())
            self.assertEqual(result['decision_trace_count'], 2)
            self.assertEqual(result['rl_dataset_count'], 2)
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            rl_records = [
                json.loads(line)
                for line in rl_dataset_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(len(rl_records), 2)
            self.assertEqual(records[1]['state']['previous_attempt_id'], 1)
            self.assertEqual(records[1]['state']['previous_primary_metric'], 0.61)
            self.assertFalse(rl_records[0]['terminal'])
            self.assertTrue(rl_records[1]['terminal'])
            self.assertEqual(rl_records[0]['next_attempt_id'], 2)
            self.assertEqual(rl_records[1]['reward']['stage_score'], 6)

    def test_build_rl_decision_dataset_record_keeps_controller_features(self) -> None:
        trace_record = {
            'paper_slug': 'demo-paper',
            'attempt_id': 2,
            'state': {
                'integration_path': 'adapter_wrapper',
                'requires_model_changes': False,
                'loss_only_pipeline_viable': False,
                'formula_requires_model_changes': True,
                'previous_attempt_id': 1,
                'previous_stop_layer': 'layer4',
                'previous_primary_metric': 0.61,
                'previous_stage_score': 5,
            },
            'action': {
                'name': 'Follow-up attempt',
                'kind': 'agent_code',
                'variant': None,
                'run_training': True,
                'files_to_edit': ['candidate_loss.py'],
                'required_edit_paths': ['sandbox_model_adapter.py'],
                'strategy_delta': {'previous_attempt_id': 1},
            },
            'reward': {
                'passed': True,
                'primary_metric_name': 'val_ssim',
                'primary_metric': 0.7,
                'stage_score': 6,
                'val_ssim': 0.7,
            },
            'outcome': {
                'status': 'passed',
                'stop_layer': None,
                'error': None,
            },
            'provenance': {
                'task_context_path': '/tmp/task_context.json',
            },
        }

        record = build_rl_decision_dataset_record(
            trace_record=trace_record,
            is_terminal=True,
            next_attempt_id=None,
        )

        self.assertEqual(record['schema_version'], 'rl_decision_dataset.v1')
        self.assertTrue(record['terminal'])
        self.assertEqual(record['state_features']['integration_path'], 'adapter_wrapper')
        self.assertEqual(record['action_features']['required_edit_paths_count'], 1)
        self.assertTrue(record['action_features']['strategy_has_delta'])
        self.assertEqual(record['reward']['primary_metric'], 0.7)

    def test_export_rl_dataset_from_decision_trace_rewrites_existing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / 'decision_trace.jsonl'
            trace_path.write_text(
                json.dumps(
                    {
                        'paper_slug': 'demo-paper',
                        'attempt_id': 1,
                        'state': {'integration_path': 'loss_only'},
                        'action': {'kind': 'agent_code', 'run_training': True},
                        'reward': {'primary_metric': 0.61, 'stage_score': 5, 'passed': False},
                        'outcome': {'passed': False, 'status': 'failed', 'stop_layer': 'layer4'},
                    },
                    ensure_ascii=False,
                )
                + '\n'
                + json.dumps(
                    {
                        'paper_slug': 'demo-paper',
                        'attempt_id': 2,
                        'state': {'integration_path': 'loss_only', 'previous_attempt_id': 1},
                        'action': {
                            'kind': 'agent_code',
                            'run_training': True,
                            'strategy_delta': {'previous_attempt_id': 1},
                        },
                        'reward': {'primary_metric': 0.7, 'stage_score': 6, 'passed': True},
                        'outcome': {'passed': True, 'status': 'passed', 'stop_layer': None},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )

            result = export_rl_dataset_from_decision_trace(trace_path)

            self.assertEqual(result['decision_trace_count'], 2)
            self.assertEqual(result['rl_dataset_count'], 2)
            rl_records = [
                json.loads(line)
                for line in Path(result['rl_dataset_path']).read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            self.assertFalse(rl_records[0]['terminal'])
            self.assertEqual(rl_records[0]['next_attempt_id'], 2)
            self.assertTrue(rl_records[1]['terminal'])


if __name__ == '__main__':
    unittest.main()
