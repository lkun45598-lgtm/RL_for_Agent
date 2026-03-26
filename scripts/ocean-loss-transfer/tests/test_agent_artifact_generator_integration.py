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

from loss_transfer.agent.agent_artifact_generator import (  # noqa: E402
    generate_candidate_loss,
    generate_followup_attempt,
    repair_candidate_loss,
)


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
                repair_plan_paths = [Path(path) for path in files if 'repair_plan' in Path(path).name]
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
                for path in repair_plan_paths:
                    path.write_text(
                        json.dumps(
                            {
                                'failure_hypothesis': 'layer3 timeout comes from unstable weighting and missing safeguards',
                                'planned_changes': [
                                    'replace squared residual with absolute residual',
                                    'keep adapter routing untouched and simplify the reduction path',
                                ],
                                'target_metric': 'val_ssim',
                                'success_criteria': 'pass layer3 and improve validation SSIM',
                                'fallback_plan': 'if SSIM still collapses, add a reconstruction anchor term next round',
                                'evidence_refs': ['validator.layer3', 'paper.loss'],
                            },
                            ensure_ascii=False,
                        ),
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
            self.assertTrue(repair['repair_plan_path'].endswith('repair_plan.json'))
            self.assertEqual(repair['repair_plan']['target_metric'], 'val_ssim')
            self.assertIn(repair['repair_plan_path'], repair['touched_paths'])
            self.assertIn(str(output_code_path), call_records[1]['files'])
            self.assertIn('current candidate_loss.py', str(call_records[1]['message']))

    def test_repair_candidate_loss_reports_error_when_repair_plan_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            output_code_path = bundle['experiment_dir'] / 'attempt_3' / 'candidate_loss.py'
            output_code_path.parent.mkdir(parents=True, exist_ok=True)
            output_code_path.write_text(
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).pow(2).mean()\n',
                encoding='utf-8',
            )
            attempt_spec = {
                'name': 'Repair Without Plan',
                'kind': 'agent_code',
                'files_to_edit': ['candidate_loss.py'],
            }

            def fake_run_agent_chat(**kwargs):
                for item in kwargs['files']:
                    path = Path(str(item))
                    if path.name == 'candidate_loss.py':
                        path.write_text(
                            'import torch\n'
                            'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                            '    return (pred - target).abs().mean()\n',
                            encoding='utf-8',
                        )
                return {'status': 'success', 'agent_id': 'stub-repair-missing-plan', 'text': 'repaired'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                repair = repair_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    failure_feedback={'stop_layer': 'layer3', 'error': 'timeout'},
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )

            self.assertEqual(repair['status'], 'error')
            self.assertIn('repair_plan.json', repair['error'])

    def test_generate_followup_attempt_uses_latest_result_and_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            trajectory_path = bundle['experiment_dir'] / 'trajectory.jsonl'
            trajectory_path.write_text(
                json.dumps({'event_type': 'attempt_finished', 'payload': {'attempt_id': 1}}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            latest_attempt_dir = bundle['experiment_dir'] / 'attempt_1'
            latest_attempt_dir.mkdir(parents=True, exist_ok=True)
            repair_plan_path = latest_attempt_dir / 'repair_plan_round_1.json'
            repair_plan_path.write_text(
                json.dumps(
                    {
                        'failure_hypothesis': 'current loss underweights reconstruction fidelity',
                        'planned_changes': ['add reconstruction anchor'],
                        'target_metric': 'val_ssim',
                        'success_criteria': 'ssim stops collapsing',
                        'fallback_plan': 'if still unstable, simplify the weighting branch',
                        'evidence_refs': ['validator.layer4'],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            result_path = latest_attempt_dir / 'result.json'
            result_path.write_text(
                json.dumps(
                    {
                        'attempt_id': 1,
                        'stop_layer': 'layer4',
                        'error': 'SSIM too low',
                        'repair_rounds': [
                            {
                                'round': 1,
                                'artifacts': {'repair_plan_path': str(repair_plan_path)},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            output_attempt_path = bundle['experiment_dir'] / 'followup_attempt_2.json'

            call_records: list[dict[str, object]] = []

            def fake_run_agent_chat(**kwargs):
                files = [str(item) for item in kwargs['files']]
                call_records.append({'files': files, 'message': kwargs['message']})
                output_attempt_path.write_text(
                    json.dumps(
                        {
                            'name': 'Follow-up attempt with reconstruction anchor',
                            'kind': 'agent_code',
                            'objective': 'Add a light reconstruction anchor term while preserving the adapter path.',
                            'files_to_edit': ['candidate_loss.py'],
                            'required_edit_paths': ['sandbox_model_adapter.py'],
                            'evidence_refs': ['result.layer4', 'repair_plan.fallback_plan'],
                            'strategy_delta': {
                                'previous_attempt_id': 1,
                                'why_previous_failed': 'SSIM collapsed under the previous weighting-only objective.',
                                'what_changes_now': ['add a reconstruction anchor', 'keep adapter routing but change the optimization target'],
                                'why_not_repeat_previous': 'the previous strategy already failed after repair and gave no viable SSIM recovery',
                                'expected_signal': 'validation SSIM should stop collapsing even if train loss drops more slowly',
                            },
                            'run_training': True,
                            'notes': 'Use the previous repair fallback to change the objective rather than repeating the same weighted loss.',
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                return {'status': 'success', 'agent_id': 'stub-followup', 'text': 'attempt written'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                result = generate_followup_attempt(
                    str(result_path),
                    task_context_path=str(bundle['task_context_path']),
                    output_attempt_path=str(output_attempt_path),
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                    max_attempts=4,
                    next_attempt_id=2,
                )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['attempt_spec']['kind'], 'agent_code')
            self.assertEqual(result['attempt_spec']['required_edit_paths'], ['sandbox_model_adapter.py'])
            self.assertEqual(result['latest_repair_plan_path'], str(repair_plan_path))
            self.assertEqual(result['attempt_spec']['strategy_delta']['previous_attempt_id'], 1)
            self.assertIn(str(result_path), call_records[0]['files'])
            self.assertIn(str(repair_plan_path), call_records[0]['files'])
            self.assertIn('逐轮重规划阶段', str(call_records[0]['message']))

    def test_generate_followup_attempt_requires_strategy_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            latest_attempt_dir = bundle['experiment_dir'] / 'attempt_1'
            latest_attempt_dir.mkdir(parents=True, exist_ok=True)
            result_path = latest_attempt_dir / 'result.json'
            result_path.write_text(
                json.dumps(
                    {
                        'attempt_id': 1,
                        'stop_layer': 'layer4',
                        'error': 'SSIM too low',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            output_attempt_path = bundle['experiment_dir'] / 'followup_attempt_2.json'

            def fake_run_agent_chat(**kwargs):
                output_attempt_path.write_text(
                    json.dumps(
                        {
                            'name': 'Incomplete follow-up attempt',
                            'kind': 'agent_code',
                            'objective': 'Try a slightly different loss.',
                            'files_to_edit': ['candidate_loss.py'],
                            'evidence_refs': ['result.layer4'],
                            'run_training': True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                return {'status': 'success', 'agent_id': 'stub-followup-missing-delta', 'text': 'attempt written'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                result = generate_followup_attempt(
                    str(result_path),
                    task_context_path=str(bundle['task_context_path']),
                    output_attempt_path=str(output_attempt_path),
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                    max_attempts=4,
                    next_attempt_id=2,
                )

            self.assertEqual(result['status'], 'error')
            self.assertIn('strategy_delta', result['error'])

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
