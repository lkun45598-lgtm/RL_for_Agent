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

from loss_transfer.agent.agent_artifact_generator import generate_candidate_loss, repair_candidate_loss  # noqa: E402


class AgentArtifactGeneratorIntegrationTests(unittest.TestCase):
    def _write_task_context_bundle(self, root: Path) -> dict[str, Path]:
        experiment_dir = root / 'experiment'
        experiment_dir.mkdir(parents=True, exist_ok=True)
        task_context_path = experiment_dir / 'task_context.json'
        analysis_plan_path = experiment_dir / 'analysis_plan.json'
        loss_formula_path = experiment_dir / 'loss_formula.json'

        analysis_plan_path.write_text(
            json.dumps(
                {
                    'summary': 'stub plan',
                    'stop_on_first_pass': False,
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'need adapter outputs',
                        'evidence_refs': ['task_context.integration_assessment'],
                    },
                    'attempts': [],
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        loss_formula_path.write_text(json.dumps({'symbol_map': {}}, ensure_ascii=False), encoding='utf-8')

        task_context = {
            'inputs': {
                'code_repo_path': str(Path('/data1/user/lz/RL_for_Agent')),
            },
            'paths': {
                'experiment_dir': str(experiment_dir),
                'analysis_plan_path': str(analysis_plan_path),
                'loss_formula_path': str(loss_formula_path),
            },
            'integration_assessment': {
                'recommended_path': 'adapter_wrapper',
                'requires_model_changes': False,
            },
        }
        task_context_path.write_text(json.dumps(task_context, ensure_ascii=False), encoding='utf-8')
        return {
            'task_context_path': task_context_path,
            'analysis_plan_path': analysis_plan_path,
            'loss_formula_path': loss_formula_path,
            'experiment_dir': experiment_dir,
        }

    def test_generate_then_repair_candidate_loss_with_stubbed_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            output_code_path = bundle['experiment_dir'] / 'attempt_1' / 'candidate_loss.py'
            attempt_spec = {
                'name': 'Adapter Attempt',
                'kind': 'agent_code',
                'files_to_edit': ['candidate_loss.py'],
                'required_edit_paths': ['sandbox_model_adapter.py'],
            }
            call_records: list[dict[str, object]] = []

            def fake_run_agent_chat(**kwargs):
                files = [str(item) for item in kwargs['files']]
                call_records.append(
                    {
                        'mode': kwargs['mode'],
                        'files': files,
                        'message': kwargs['message'],
                    }
                )
                candidate_paths = [Path(path) for path in files if path.endswith('candidate_loss.py')]
                adapter_paths = [Path(path) for path in files if path.endswith('sandbox_model_adapter.py')]
                if len(call_records) == 1:
                    for path in candidate_paths:
                        path.write_text(
                            'import torch\n'
                            'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                            '    return (pred - target).pow(2).mean()\n',
                            encoding='utf-8',
                        )
                    for path in adapter_paths:
                        path.write_text('# adapter touched during generation\n', encoding='utf-8')
                    return {'status': 'success', 'agent_id': 'stub-generate', 'text': 'generated'}

                for path in candidate_paths:
                    path.write_text(
                        'import torch\n'
                        'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                        '    return (pred - target).abs().mean()\n',
                        encoding='utf-8',
                    )
                return {'status': 'success', 'agent_id': 'stub-repair', 'text': 'repaired'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                generation = generate_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )
                repair = repair_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    failure_feedback={'stop_layer': 'layer3', 'error': 'timeout'},
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )

            self.assertEqual(generation['status'], 'success')
            self.assertEqual(repair['status'], 'success')
            self.assertCountEqual(
                generation['touched_paths'],
                [
                    str(output_code_path),
                    str(output_code_path.parent / 'sandbox_overrides' / 'sandbox_model_adapter.py'),
                ],
            )
            self.assertIn(
                str(output_code_path.parent / 'sandbox_overrides' / 'sandbox_model_adapter.py'),
                repair['historical_touched_paths'],
            )
            self.assertIn(str(output_code_path), call_records[1]['files'])
            self.assertIn('current candidate_loss.py', str(call_records[1]['message']))

    def test_generate_candidate_loss_reports_required_edit_path_error_when_stub_skips_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            output_code_path = bundle['experiment_dir'] / 'attempt_2' / 'candidate_loss.py'
            attempt_spec = {
                'name': 'Broken Adapter Attempt',
                'kind': 'agent_code',
                'files_to_edit': ['candidate_loss.py'],
                'required_edit_paths': ['sandbox_model_adapter.py'],
            }

            def fake_run_agent_chat(**kwargs):
                files = [Path(str(item)) for item in kwargs['files']]
                for path in files:
                    if path.name == 'candidate_loss.py':
                        path.write_text(
                            'import torch\n'
                            'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                            '    return pred.mean() - target.mean()\n',
                            encoding='utf-8',
                        )
                return {'status': 'success', 'agent_id': 'stub-missing', 'text': 'generated without override'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                generation = generate_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )

            self.assertEqual(generation['status'], 'error')
            self.assertIn('required attempt-scoped paths', generation['error'])
            self.assertEqual(generation['required_edit_paths'], ['sandbox_model_adapter.py'])


if __name__ == '__main__':
    unittest.main()
