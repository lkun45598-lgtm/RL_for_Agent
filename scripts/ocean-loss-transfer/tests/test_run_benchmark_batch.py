from __future__ import annotations

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
                'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
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
            self.assertEqual(entry['plan_status'], 'skipped')
            self.assertEqual(entry['loop_status'], 'skipped')

            mock_context.assert_called_once()
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
                'analysis_plan_path': '/tmp/run/entries/demo-entry/analysis_plan.json',
                'trajectory_path': '/tmp/run/entries/demo-entry/trajectory.jsonl',
            },
        }

        loop_result = {
            'status': 'completed',
            'best_attempt_id': 2,
            'best_metric_name': 'val_ssim',
            'best_metric_value': 0.671,
            'trajectory_path': '/tmp/run/entries/demo-entry/trajectory.jsonl',
            'attempts': [
                {'attempt_id': 1, 'passed': False, 'stop_layer': 'layer4', 'error': 'below threshold'},
                {'attempt_id': 2, 'passed': True, 'stop_layer': None, 'error': None},
            ],
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
                'source_code_path': '/tmp/code.zip',
                'materialized': True,
                'cache_dir': '/tmp/cache/demo-entry',
            },
        ), patch(
            'run_benchmark_batch.build_task_context',
            return_value=task_context,
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
        self.assertIsNone(entry['stop_layer'])

        mock_plan.assert_called_once()
        mock_loop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
