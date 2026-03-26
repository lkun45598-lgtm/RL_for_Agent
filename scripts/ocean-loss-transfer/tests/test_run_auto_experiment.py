from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.orchestration.run_auto_experiment import run_auto_experiment  # noqa: E402


class RunAutoExperimentTests(unittest.TestCase):
    def _task_context(self) -> dict[str, object]:
        return {
            'paper_slug': 'sea_raft',
            'paths': {
                'task_context_path': '/tmp/task_context.json',
                'loss_formula_path': '/tmp/loss_formula.json',
                'loss_ir_path': '/tmp/loss_ir.yaml',
                'decision_trace_path': '/tmp/decision_trace.jsonl',
                'rl_dataset_path': '/tmp/rl_decision_dataset.jsonl',
            },
        }

    def test_context_only_returns_paths_without_running_agent_loop(self) -> None:
        task_context = self._task_context()

        with patch('loss_transfer.orchestration.run_auto_experiment.build_task_context', return_value=task_context), patch(
            'loss_transfer.orchestration.run_auto_experiment.generate_analysis_plan'
        ) as mock_generate:
            result = run_auto_experiment(
                paper_slug='sea_raft',
                code_repo_path='/repo',
                output_dir='/tmp/exp',
                mode='context_only',
            )

        self.assertEqual(result['status'], 'context_ready')
        self.assertEqual(result['task_context_path'], '/tmp/task_context.json')
        self.assertEqual(result['loss_formula_path'], '/tmp/loss_formula.json')
        self.assertEqual(result['loss_ir_path'], '/tmp/loss_ir.yaml')
        self.assertEqual(result['decision_trace_path'], '/tmp/decision_trace.jsonl')
        self.assertEqual(result['rl_dataset_path'], '/tmp/rl_decision_dataset.jsonl')
        mock_generate.assert_not_called()

    def test_auto_generate_plan_passes_generated_plan_into_agent_loop(self) -> None:
        task_context = self._task_context()
        plan_generation = {
            'status': 'success',
            'analysis_plan_path': '/tmp/generated_analysis_plan.json',
            'agent_response_path': '/tmp/analysis_plan_agent_response.json',
        }
        loop_result = {
            'status': 'completed',
            'paper_slug': 'sea_raft',
            'best_attempt_id': 1,
        }

        with patch('loss_transfer.orchestration.run_auto_experiment.build_task_context', return_value=task_context), patch(
            'loss_transfer.orchestration.run_auto_experiment.generate_analysis_plan',
            return_value=plan_generation,
        ) as mock_generate, patch(
            'loss_transfer.agent.agent_repair_loop.run_agent_repair_loop',
            return_value=dict(loop_result),
        ) as mock_loop:
            result = run_auto_experiment(
                paper_slug='sea_raft',
                code_repo_path='/repo',
                dataset_root='/data1/user/lz/RL_data_test',
                output_dir='/tmp/exp',
                auto_generate_plan=True,
                max_attempts=3,
                service_url='http://agent.local',
                service_api_key='secret',
            )

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['plan_generation'], plan_generation)
        mock_generate.assert_called_once()
        mock_loop.assert_called_once()
        loop_kwargs = mock_loop.call_args.kwargs
        self.assertEqual(loop_kwargs['analysis_plan_path'], '/tmp/generated_analysis_plan.json')
        self.assertEqual(loop_kwargs['dataset_root'], '/data1/user/lz/RL_data_test')
        self.assertEqual(loop_kwargs['output_dir'], '/tmp/exp')
        self.assertEqual(loop_kwargs['agent_service_url'], 'http://agent.local')
        self.assertEqual(loop_kwargs['agent_api_key'], 'secret')

    def test_plan_generation_failure_short_circuits_before_agent_loop(self) -> None:
        task_context = self._task_context()
        plan_generation = {
            'status': 'error',
            'error': 'service unavailable',
        }

        with patch('loss_transfer.orchestration.run_auto_experiment.build_task_context', return_value=task_context), patch(
            'loss_transfer.orchestration.run_auto_experiment.generate_analysis_plan',
            return_value=plan_generation,
        ), patch('loss_transfer.agent.agent_repair_loop.run_agent_repair_loop') as mock_loop:
            result = run_auto_experiment(
                paper_slug='sea_raft',
                code_repo_path='/repo',
                auto_generate_plan=True,
            )

        self.assertEqual(result['status'], 'plan_generation_failed')
        self.assertEqual(result['paper_slug'], 'sea_raft')
        self.assertEqual(result['plan_generation'], plan_generation)
        mock_loop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
