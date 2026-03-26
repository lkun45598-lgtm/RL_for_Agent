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

from agent_repair_loop import run_agent_repair_loop  # noqa: E402


class AgentRepairLoopIntegrationTests(unittest.TestCase):
    def _write_task_context(self, experiment_dir: Path, paper_slug: str) -> dict[str, object]:
        task_context_path = experiment_dir / 'task_context.json'
        task_context = {
            'paper_slug': paper_slug,
            'inputs': {
                'dataset_root': '/data1/user/lz/RL_data_test',
            },
            'paths': {
                'task_context_path': str(task_context_path),
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

            with patch('attempt_executor.execute_attempt', side_effect=fake_execute_attempt):
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


if __name__ == '__main__':
    unittest.main()
