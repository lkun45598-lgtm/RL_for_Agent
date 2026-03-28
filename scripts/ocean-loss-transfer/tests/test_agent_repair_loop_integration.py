"""
@file test_agent_repair_loop_integration.py
@description Integration tests for the agent repair loop, covering routing audits, run manifests, and case-memory exports.
@author kongzhiquan
@contributors OpenAI Codex
@date 2026-03-28
@version 1.1.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 add integration coverage for the agent repair loop
  - 2026-03-28 kongzhiquan: v1.1.0 merge routing-audit and case-memory conflict expectations
"""

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

from loss_transfer.agent.agent_repair_loop import run_agent_repair_loop  # noqa: E402


class AgentRepairLoopIntegrationTests(unittest.TestCase):
    def _write_task_context(self, experiment_dir: Path, paper_slug: str) -> dict[str, object]:
        task_context_path = experiment_dir / 'task_context.json'
        task_context = {
            'paper_slug': paper_slug,
            'inputs': {
                'dataset_root': '/data1/user/lz/RL_data_test',
                'code_repo_path': str(experiment_dir),
            },
            'paths': {
                'task_context_path': str(task_context_path),
                'contract_validation_path': str(experiment_dir / 'contract_validation.json'),
            },
            'integration_assessment': {
                'recommended_path': 'adapter_wrapper',
                'requires_model_changes': False,
            },
            'formula_spec': {
                'symbol_map': {},
            },
        }
        task_context_path.write_text(json.dumps(task_context, ensure_ascii=False), encoding='utf-8')
        return task_context

    def _write_analysis_plan(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    'summary': 'Stubbed analysis plan for integration test',
                    'stop_on_first_pass': True,
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'Loss needs adapter-exposed auxiliary tensors.',
                        'evidence_refs': ['paper.loss', 'code.model_forward'],
                    },
                    'attempts': [
                        {
                            'name': 'Attempt 1',
                            'kind': 'agent_code',
                            'objective': 'Implement faithful adapter-aware loss transfer.',
                            'evidence_refs': ['paper.loss'],
                        },
                        {
                            'name': 'Attempt 2',
                            'kind': 'agent_code',
                            'objective': 'Stabilize weighting while preserving adapter routing.',
                            'evidence_refs': ['paper.loss', 'code.model_forward'],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )

    def _write_single_attempt_plan(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    'summary': 'Single-attempt plan to trigger dynamic replanning',
                    'stop_on_first_pass': False,
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'Need adapter outputs before deciding on deeper edits.',
                        'evidence_refs': ['paper.loss', 'code.model_forward'],
                    },
                    'attempts': [
                        {
                            'name': 'Initial attempt',
                            'kind': 'agent_code',
                            'objective': 'Try the first adapter-aware loss transfer.',
                            'evidence_refs': ['paper.loss'],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )

    def _read_event_types(self, experiment_dir: Path) -> list[str]:
        trajectory_path = experiment_dir / 'trajectory.jsonl'
        if not trajectory_path.exists():
            return []
        return [
            json.loads(line)['event_type']
            for line in trajectory_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def test_run_agent_repair_loop_executes_normalized_attempts_and_stops_on_first_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            experiment_dir = temp_root / 'experiment'
            case_memory_path = temp_root / 'knowledge_base' / 'case_memories.jsonl'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            task_context = self._write_task_context(experiment_dir, paper_slug='paper_loop')
            input_plan_path = temp_root / 'input_analysis_plan.json'
            self._write_analysis_plan(input_plan_path)

            call_records: list[dict[str, object]] = []

            def fake_execute_attempt(**kwargs):
                call_records.append(kwargs)
                attempt_id = int(kwargs['attempt_id'])
                if attempt_id == 1:
                    return {
                        'attempt_id': 1,
                        'status': 'failed',
                        'passed': False,
                        'reward_summary': {
                            'primary_metric_name': 'swinir',
                            'primary_metric': 0.62,
                        },
                    }
                return {
                    'attempt_id': 2,
                    'status': 'passed',
                    'passed': True,
                    'reward_summary': {
                        'primary_metric_name': 'swinir',
                        'primary_metric': 0.73,
                    },
                }

            with patch(
                'loss_transfer.agent.agent_repair_loop.write_run_manifest',
                return_value={'run_manifest_path': str(experiment_dir / 'run_manifest.json')},
            ), patch('loss_transfer.attempts.attempt_executor.execute_attempt', side_effect=fake_execute_attempt), patch(
                'loss_transfer.common.decision_trace._DEFAULT_CASE_MEMORY_PATH',
                case_memory_path,
            ):
                result = run_agent_repair_loop(
                    task_context,
                    analysis_plan_path=str(input_plan_path),
                    max_attempts=4,
                    bootstrap_formula=False,
                    dataset_root='/data1/user/lz/RL_data_test',
                    output_dir=str(experiment_dir),
                )

            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['best_attempt_id'], 2)
            self.assertEqual(result['best_metric_name'], 'swinir')
            self.assertEqual(result['best_metric_value'], 0.73)
            self.assertEqual(result['attempt_count'], 2)
            self.assertEqual(result['best_reward_summary']['primary_metric'], 0.73)
            self.assertEqual(result['decision_trace_count'], 2)
            self.assertTrue(Path(result['decision_trace_path']).exists())
            self.assertTrue(Path(result['routing_audit_path']).exists())
            self.assertEqual(result['run_manifest_path'], str(experiment_dir / 'run_manifest.json'))
            self.assertEqual(result['case_memory_path'], str(case_memory_path.resolve()))
            self.assertTrue(case_memory_path.exists())
            self.assertEqual(len(result['attempts']), 2)
            self.assertEqual(len(call_records), 2)
            self.assertEqual(call_records[0]['dataset_root'], '/data1/user/lz/RL_data_test')
            self.assertEqual(
                call_records[0]['analysis_plan_path'],
                str(experiment_dir / 'analysis_plan.json'),
            )
            self.assertEqual(
                call_records[0]['attempt_spec']['required_edit_paths'],
                ['sandbox_model_adapter.py'],
            )
            self.assertIn(
                'sandbox trainer files',
                call_records[0]['attempt_spec']['files_to_edit'],
            )

            normalized_plan = json.loads(
                (experiment_dir / 'analysis_plan.json').read_text(encoding='utf-8')
            )
            self.assertEqual(normalized_plan['integration_decision']['path'], 'adapter_wrapper')
            self.assertEqual(len(normalized_plan['attempts']), 2)

            summary_on_disk = json.loads(
                (experiment_dir / 'agent_loop_summary.json').read_text(encoding='utf-8')
            )
            self.assertEqual(summary_on_disk['best_attempt_id'], 2)
            self.assertEqual(
                self._read_event_types(experiment_dir),
                ['task_context_ready', 'attempt_planned', 'attempt_planned'],
            )

    def test_run_agent_repair_loop_appends_followup_attempt_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            experiment_dir = temp_root / 'experiment'
            case_memory_path = temp_root / 'knowledge_base' / 'case_memories.jsonl'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            task_context = self._write_task_context(experiment_dir, paper_slug='paper_replan')
            input_plan_path = temp_root / 'single_attempt_plan.json'
            self._write_single_attempt_plan(input_plan_path)

            call_records: list[dict[str, object]] = []

            def fake_execute_attempt(**kwargs):
                call_records.append(kwargs)
                attempt_id = int(kwargs['attempt_id'])
                if attempt_id == 1:
                    return {
                        'attempt_id': 1,
                        'status': 'failed',
                        'passed': False,
                        'error': 'ssim collapse',
                        'reward_summary': {
                            'primary_metric_name': 'swinir',
                            'primary_metric': 0.61,
                        },
                        'paths': {
                            'result_path': str(experiment_dir / 'attempt_1' / 'result.json'),
                        },
                    }
                return {
                    'attempt_id': 2,
                    'status': 'passed',
                    'passed': True,
                    'strategy_delta': {
                        'previous_attempt_id': 1,
                        'why_previous_failed': 'SSIM collapsed after the first adapter-aware transfer.',
                        'what_changes_now': ['add a reconstruction anchor'],
                        'why_not_repeat_previous': 'the previous objective already failed and produced no viable recovery',
                        'expected_signal': 'validation SSIM should improve even if training loss decreases more slowly',
                    },
                    'reward_summary': {
                        'primary_metric_name': 'swinir',
                        'primary_metric': 0.7,
                    },
                    'paths': {
                        'result_path': str(experiment_dir / 'attempt_2' / 'result.json'),
                    },
                }

            with patch(
                'loss_transfer.agent.agent_repair_loop.write_run_manifest',
                return_value={'run_manifest_path': str(experiment_dir / 'run_manifest.json')},
            ), patch('loss_transfer.attempts.attempt_executor.execute_attempt', side_effect=fake_execute_attempt), patch(
                'loss_transfer.agent.agent_repair_loop.generate_followup_attempt',
                return_value={
                    'status': 'success',
                    'attempt_path': str(experiment_dir / 'followup_attempt_2.json'),
                    'attempt_spec': {
                        'name': 'Replanned attempt',
                        'kind': 'agent_code',
                        'objective': 'Add a reconstruction anchor after SSIM collapse.',
                        'files_to_edit': ['candidate_loss.py'],
                        'required_edit_paths': ['sandbox_model_adapter.py'],
                        'evidence_refs': ['result.layer4', 'repair_plan.fallback_plan'],
                        'strategy_delta': {
                            'previous_attempt_id': 1,
                            'why_previous_failed': 'SSIM collapsed after the first adapter-aware transfer.',
                            'what_changes_now': ['add a reconstruction anchor'],
                            'why_not_repeat_previous': 'the previous objective already failed and produced no viable recovery',
                            'expected_signal': 'validation SSIM should improve even if training loss decreases more slowly',
                        },
                        'run_training': True,
                        'notes': 'Use the failed attempt feedback to change strategy.',
                    },
                    'latest_repair_plan_path': str(experiment_dir / 'attempt_1' / 'repair_plan_round_1.json'),
                },
            ) as mock_replan, patch(
                'loss_transfer.common.decision_trace._DEFAULT_CASE_MEMORY_PATH',
                case_memory_path,
            ):
                result = run_agent_repair_loop(
                    task_context,
                    analysis_plan_path=str(input_plan_path),
                    max_attempts=3,
                    bootstrap_formula=False,
                    dataset_root='/data1/user/lz/RL_data_test',
                    output_dir=str(experiment_dir),
                    agent_service_url='http://agent.local',
                    agent_api_key='secret',
                )

            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['best_attempt_id'], 2)
            self.assertEqual(len(result['attempts']), 2)
            self.assertEqual(result['best_strategy_delta']['previous_attempt_id'], 1)
            self.assertEqual(result['decision_trace_count'], 2)
            self.assertTrue(Path(result['decision_trace_path']).exists())
            self.assertTrue(Path(result['routing_audit_path']).exists())
            self.assertEqual(result['run_manifest_path'], str(experiment_dir / 'run_manifest.json'))
            self.assertEqual(result['case_memory_path'], str(case_memory_path.resolve()))
            self.assertTrue(case_memory_path.exists())
            self.assertEqual(len(call_records), 2)
            self.assertEqual(call_records[1]['attempt_spec']['name'], 'Replanned attempt')
            self.assertEqual(call_records[1]['attempt_spec']['required_edit_paths'], ['sandbox_model_adapter.py'])
            self.assertEqual(call_records[1]['attempt_spec']['strategy_delta']['previous_attempt_id'], 1)
            mock_replan.assert_called_once()
            self.assertEqual(
                self._read_event_types(experiment_dir),
                ['task_context_ready', 'attempt_planned', 'attempt_replanned', 'attempt_planned'],
            )

    def test_run_agent_repair_loop_returns_contract_error_for_invalid_analysis_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            experiment_dir = temp_root / 'experiment'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            task_context = self._write_task_context(experiment_dir, paper_slug='paper_invalid_plan')
            bad_plan_path = temp_root / 'bad_analysis_plan.json'
            bad_plan_path.write_text(
                json.dumps(
                    {
                        'summary': 'Invalid plan',
                        'stop_on_first_pass': False,
                        'integration_decision': {
                            'path': 'agent_decides',
                            'rationale': 'Let the agent decide everything.',
                            'evidence_refs': ['paper.loss'],
                        },
                        'attempts': [
                            {
                                'name': 'Attempt 1',
                                'kind': 'agent_code',
                                'objective': 'Try something.',
                                'evidence_refs': ['paper.loss'],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with patch(
                'loss_transfer.agent.agent_repair_loop.write_run_manifest',
                return_value={'run_manifest_path': str(experiment_dir / 'run_manifest.json')},
            ), patch('loss_transfer.attempts.attempt_executor.execute_attempt') as mock_execute:
                result = run_agent_repair_loop(
                    task_context,
                    analysis_plan_path=str(bad_plan_path),
                    max_attempts=4,
                    bootstrap_formula=False,
                    dataset_root='/data1/user/lz/RL_data_test',
                    output_dir=str(experiment_dir),
                )

            self.assertEqual(result['status'], 'contract_error')
            self.assertTrue(Path(result['contract_validation_path']).exists())
            self.assertTrue(
                any('analysis_plan.json validation failed' in error for error in result['contract_validation_errors'])
            )
            mock_execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
