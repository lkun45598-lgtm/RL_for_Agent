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

from run_benchmark_batch import run_benchmark_batch, select_benchmark_entries  # noqa: E402


class RunBenchmarkBatchTests(unittest.TestCase):
    def test_select_benchmark_entries_filters_ready_entries(self) -> None:
        catalog = {
            'entries': [
                {'entry_id': 'a', 'paper_slug': 'paper-a', 'category': '通用Loss', 'relative_dir': 'x', 'status': 'ready'},
                {'entry_id': 'b', 'paper_slug': 'paper-b', 'category': '海洋Loss', 'relative_dir': 'y', 'status': 'incomplete'},
                {'entry_id': 'c', 'paper_slug': 'paper-c', 'category': '通用Loss', 'relative_dir': 'z', 'status': 'ready'},
            ]
        }

        selected = select_benchmark_entries(
            catalog,
            categories=['通用Loss'],
            limit=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['entry_id'], 'a')

    def test_run_benchmark_batch_context_only_writes_summary(self) -> None:
        catalog = {
            'benchmark_root': '/tmp/Benchmark',
            'entries': [
                {
                    'entry_id': 'demo-entry',
                    'paper_slug': 'demo-paper',
                    'title': 'Demo Paper',
                    'category': '通用Loss',
                    'relative_dir': '通用Loss/demo-entry',
                    'status': 'ready',
                }
            ],
        }

        task_context = {
            'status': 'context_ready',
            'formula_draft_status': {'status': 'success'},
            'paths': {
                'experiment_dir': '/tmp/run/entries/demo-entry',
                'task_context_path': '/tmp/run/entries/demo-entry/task_context.json',
                'loss_formula_path': '/tmp/run/entries/demo-entry/loss_formula.json',
                'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json',
                'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
                'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
                'trajectory_path': '/tmp/run/entries/demo-entry/trajectory.jsonl',
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            'run_benchmark_batch.build_benchmark_catalog',
            return_value=catalog,
        ), patch(
            'run_benchmark_batch.materialize_benchmark_entry',
            return_value={
                'status': 'ready',
                'paper_pdf_path': '/tmp/paper.pdf',
                'code_repo_path': '/tmp/code',
                'source_code_path': '/tmp/code',
                'materialized': False,
            },
        ), patch(
            'run_benchmark_batch.build_task_context',
            return_value=task_context,
        ) as mock_context, patch(
            'run_benchmark_batch.write_run_manifest',
            return_value={'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json'},
        ) as mock_manifest, patch(
            'run_benchmark_batch.write_contract_validation',
            return_value={
                'status': 'ok',
                'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
            },
        ), patch(
            'run_benchmark_batch.generate_analysis_plan',
        ) as mock_plan, patch(
            'run_benchmark_batch.run_agent_repair_loop',
        ) as mock_loop:
            result = run_benchmark_batch(
                benchmark_root='/tmp/Benchmark',
                output_root=temp_dir,
                run_id='demo-run',
                mode='context_only',
            )
            self.assertEqual(result['selected_count'], 1)
            self.assertEqual(result['overall_status_counts']['context_ready'], 1)
            self.assertTrue(Path(temp_dir, 'demo-run', 'benchmark_run_summary.json').exists())

            entry = result['results'][0]
            self.assertEqual(entry['context_status'], 'context_ready')
            self.assertEqual(entry['formula_status'], 'success')
            self.assertEqual(entry['run_manifest_path'], '/tmp/run/entries/demo-entry/run_manifest.json')
            self.assertEqual(entry['contract_validation_path'], '/tmp/run/entries/demo-entry/contract_validation.json')
            self.assertEqual(entry['plan_status'], 'skipped')
            self.assertEqual(entry['loop_status'], 'skipped')

            mock_context.assert_called_once()
            self.assertEqual(mock_manifest.call_count, 2)
            mock_plan.assert_not_called()
            mock_loop.assert_not_called()

    def test_run_benchmark_batch_reports_context_error_from_contract_validation(self) -> None:
        catalog = {
            'benchmark_root': '/tmp/Benchmark',
            'entries': [
                {
                    'entry_id': 'demo-entry',
                    'paper_slug': 'demo-paper',
                    'title': 'Demo Paper',
                    'category': '通用Loss',
                    'relative_dir': '通用Loss/demo-entry',
                    'status': 'ready',
                }
            ],
        }

        task_context = {
            'status': 'context_ready',
            'formula_draft_status': {'status': 'success'},
            'paths': {
                'experiment_dir': '/tmp/run/entries/demo-entry',
                'task_context_path': '/tmp/run/entries/demo-entry/task_context.json',
                'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json',
                'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
                'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            'run_benchmark_batch.build_benchmark_catalog',
            return_value=catalog,
        ), patch(
            'run_benchmark_batch.materialize_benchmark_entry',
            return_value={
                'status': 'ready',
                'paper_pdf_path': '/tmp/paper.pdf',
                'code_repo_path': '/tmp/code',
                'source_code_path': '/tmp/code',
                'materialized': False,
            },
        ), patch(
            'run_benchmark_batch.build_task_context',
            return_value=task_context,
        ), patch(
            'run_benchmark_batch.write_run_manifest',
            return_value={'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json'},
        ), patch(
            'run_benchmark_batch.write_contract_validation',
            return_value={
                'status': 'error',
                'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
                'errors': ['task_context.integration_assessment.recommended_path is invalid'],
            },
        ), patch('run_benchmark_batch.generate_analysis_plan') as mock_plan, patch(
            'run_benchmark_batch.run_agent_repair_loop',
        ) as mock_loop:
            result = run_benchmark_batch(
                benchmark_root='/tmp/Benchmark',
                output_root=temp_dir,
                run_id='demo-run',
                mode='context_only',
            )

        entry = result['results'][0]
        self.assertEqual(entry['context_status'], 'context_error')
        self.assertEqual(entry['overall_status'], 'context_error')
        self.assertIn('recommended_path', entry['error_summary'])
        self.assertEqual(result['overall_status_counts']['context_error'], 1)
        mock_plan.assert_not_called()
        mock_loop.assert_not_called()

    def test_run_benchmark_batch_agent_loop_with_generated_plan(self) -> None:
        catalog = {
            'benchmark_root': '/tmp/Benchmark',
            'entries': [
                {
                    'entry_id': 'demo-entry',
                    'paper_slug': 'demo-paper',
                    'title': 'Demo Paper',
                    'category': '通用Loss',
                    'relative_dir': '通用Loss/demo-entry',
                    'status': 'ready',
                }
            ],
        }

        task_context = {
            'status': 'context_ready',
            'formula_draft_status': {'status': 'success'},
            'paths': {
                'experiment_dir': '/tmp/run/entries/demo-entry',
                'task_context_path': '/tmp/run/entries/demo-entry/task_context.json',
                'loss_formula_path': '/tmp/run/entries/demo-entry/loss_formula.json',
                'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json',
                'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
                'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
                'trajectory_path': '/tmp/run/entries/demo-entry/trajectory.jsonl',
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            decision_trace_path = Path(temp_dir) / 'demo-run' / 'entries' / 'demo-entry' / 'decision_trace.jsonl'
            rl_dataset_path = Path(temp_dir) / 'demo-run' / 'entries' / 'demo-entry' / 'rl_decision_dataset.jsonl'
            decision_trace_path.parent.mkdir(parents=True, exist_ok=True)
            decision_trace_path.write_text(
                json.dumps({'attempt_id': 1, 'reward': {'primary_metric': 0.62}}, ensure_ascii=False) + '\n'
                + json.dumps({'attempt_id': 2, 'reward': {'primary_metric': 0.671}}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            rl_dataset_path.write_text(
                json.dumps({'attempt_id': 1, 'terminal': False, 'reward': {'stage_score': 5}}, ensure_ascii=False) + '\n'
                + json.dumps({'attempt_id': 2, 'terminal': True, 'reward': {'stage_score': 6}}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            loop_result = {
                'status': 'completed',
                'attempt_count': 2,
                'best_attempt_id': 2,
                'best_metric_name': 'val_ssim',
                'best_metric_value': 0.671,
                'best_reward_summary': {
                    'primary_metric_name': 'val_ssim',
                    'primary_metric': 0.671,
                    'stage_score': 6,
                },
                'best_strategy_delta': {
                    'previous_attempt_id': 1,
                    'why_previous_failed': 'the first attempt stayed below threshold',
                    'what_changes_now': ['stabilize the objective'],
                    'why_not_repeat_previous': 'the old strategy already underperformed',
                    'expected_signal': 'validation SSIM should increase',
                },
                'decision_trace_path': str(decision_trace_path),
                'decision_trace_count': 2,
                'rl_dataset_path': str(rl_dataset_path),
                'rl_dataset_count': 2,
                'trajectory_path': '/tmp/run/entries/demo-entry/trajectory.jsonl',
                'attempts': [
                    {
                        'attempt_id': 1,
                        'passed': False,
                        'stop_layer': 'layer4',
                        'error': 'below threshold',
                        'reward_summary': {'primary_metric_name': 'val_ssim', 'primary_metric': 0.62, 'stage_score': 5},
                    },
                    {
                        'attempt_id': 2,
                        'passed': True,
                        'stop_layer': None,
                        'error': None,
                        'reward_summary': {'primary_metric_name': 'val_ssim', 'primary_metric': 0.671, 'stage_score': 6},
                        'strategy_delta': {'previous_attempt_id': 1},
                    },
                ],
            }

            with patch(
                'run_benchmark_batch.build_benchmark_catalog',
                return_value=catalog,
            ), patch(
                'run_benchmark_batch.materialize_benchmark_entry',
                return_value={
                    'status': 'ready',
                    'paper_pdf_path': '/tmp/paper.pdf',
                    'code_repo_path': '/tmp/code',
                    'source_code_path': '/tmp/code.zip',
                    'materialized': True,
                    'cache_dir': '/tmp/cache/demo-entry',
                },
            ), patch(
                'run_benchmark_batch.build_task_context',
                return_value=task_context,
            ) as mock_context, patch(
                'run_benchmark_batch.write_run_manifest',
                return_value={'run_manifest_path': '/tmp/run/entries/demo-entry/run_manifest.json'},
            ), patch(
                'run_benchmark_batch.write_contract_validation',
                return_value={
                    'status': 'ok',
                    'contract_validation_path': '/tmp/run/entries/demo-entry/contract_validation.json',
                },
            ), patch(
                'run_benchmark_batch.generate_analysis_plan',
                return_value={
                    'status': 'success',
                    'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
                },
            ) as mock_plan, patch(
                'run_benchmark_batch.run_agent_repair_loop',
                return_value=loop_result,
            ) as mock_loop:
                result = run_benchmark_batch(
                    benchmark_root='/tmp/Benchmark',
                    output_root=temp_dir,
                    run_id='demo-run',
                    mode='agent_loop',
                    auto_generate_plan=True,
                    service_url='http://agent.local',
                    service_api_key='secret',
                )

            entry = result['results'][0]
            self.assertEqual(entry['plan_status'], 'success')
            self.assertEqual(entry['loop_status'], 'completed')
            self.assertEqual(entry['best_attempt_id'], 2)
            self.assertEqual(entry['best_metric_name'], 'val_ssim')
            self.assertEqual(entry['best_metric_value'], 0.671)
            self.assertEqual(entry['attempt_count'], 2)
            self.assertEqual(entry['best_reward_summary']['stage_score'], 6)
            self.assertEqual(entry['run_manifest_path'], '/tmp/run/entries/demo-entry/run_manifest.json')
            self.assertEqual(entry['contract_validation_path'], '/tmp/run/entries/demo-entry/contract_validation.json')
            self.assertEqual(entry['representative_reward_summary']['primary_metric'], 0.671)
            self.assertEqual(entry['representative_strategy_delta']['previous_attempt_id'], 1)
            self.assertEqual(entry['decision_trace_count'], 2)
            self.assertEqual(entry['rl_dataset_count'], 2)
            self.assertTrue(Path(result['decision_trace_path']).exists())
            self.assertEqual(result['decision_trace_count'], 2)
            self.assertTrue(Path(result['rl_dataset_path']).exists())
            self.assertEqual(result['rl_dataset_count'], 2)
            self.assertIsNone(entry['stop_layer'])

            mock_context.assert_called_once()
            mock_plan.assert_called_once()
            mock_loop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
