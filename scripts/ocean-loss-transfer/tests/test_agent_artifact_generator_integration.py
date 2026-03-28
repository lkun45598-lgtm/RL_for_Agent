"""
@file test_agent_artifact_generator_integration.py
@description Integration tests for agent artifact generation, evidence probes, and case-memory prompt reuse.
@author kongzhiquan
@contributors OpenAI Codex
@date 2026-03-28
@version 1.1.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 add integration coverage for agent artifact generation
  - 2026-03-28 kongzhiquan: v1.1.0 merge evidence-probe and case-memory conflict expectations
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

from loss_transfer.agent.agent_artifact_generator import (  # noqa: E402
    generate_analysis_plan,
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
                'analysis_evidence_probe_request_path': str(experiment_dir / 'analysis_evidence_probe_request.json'),
                'analysis_evidence_probe_script_path': str(experiment_dir / 'analysis_evidence_probe.py'),
                'analysis_evidence_probe_result_path': str(experiment_dir / 'analysis_evidence_probe_result.json'),
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

    def test_generate_analysis_plan_runs_probe_before_writing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            task_context_path = bundle['task_context_path']
            experiment_dir = bundle['experiment_dir']
            analysis_plan_path = bundle['analysis_plan_path']

            call_records: list[dict[str, object]] = []

            def fake_run_agent_chat(**kwargs):
                call_records.append({'message': kwargs['message'], 'files': list(kwargs['files'])})
                message = str(kwargs['message'])
                if '分析补证据阶段' in message:
                    request_path = experiment_dir / 'analysis_evidence_probe_request.json'
                    script_path = experiment_dir / 'analysis_evidence_probe.py'
                    request_path.write_text(
                        json.dumps(
                            {
                                'status': 'probe_needed',
                                'reason': 'Need to verify which model files expose loss_inputs.',
                                'evidence_refs': ['code_analysis.focus_files[0]'],
                                'probe_goal': 'Scan repo for loss_inputs mentions.',
                                'expected_output_keys': ['files_with_loss_inputs'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    script_path.write_text(
                        'import argparse\n'
                        'import json\n'
                        'from pathlib import Path\n'
                        'parser = argparse.ArgumentParser()\n'
                        'parser.add_argument("--code_repo", required=True)\n'
                        'parser.add_argument("--task_context", required=True)\n'
                        'parser.add_argument("--output", required=True)\n'
                        'args = parser.parse_args()\n'
                        'repo = Path(args.code_repo)\n'
                        'hits = []\n'
                        'for path in repo.rglob("*.py"):\n'
                        '    text = path.read_text(encoding="utf-8", errors="ignore")\n'
                        '    if "loss_inputs" in text:\n'
                        '        hits.append(str(path.relative_to(repo)))\n'
                        'Path(args.output).write_text(json.dumps({"files_with_loss_inputs": hits}), encoding="utf-8")\n',
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-probe', 'text': 'probe ready'}

                analysis_plan_path.write_text(
                    json.dumps(
                        {
                            'summary': 'Use adapter path after checking evidence probe results.',
                            'stop_on_first_pass': False,
                            'integration_decision': {
                                'path': 'adapter_wrapper',
                                'rationale': 'Evidence probe confirms model-side loss_inputs style wiring.',
                                'evidence_refs': [
                                    'code_analysis.focus_files[0]',
                                    'analysis_evidence_probe_result.files_with_loss_inputs',
                                ],
                            },
                            'attempts': [
                                {
                                    'name': 'Attempt 1',
                                    'kind': 'agent_code',
                                    'objective': 'Implement adapter-aware loss transfer.',
                                    'evidence_refs': ['analysis_evidence_probe_result.files_with_loss_inputs'],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                return {'status': 'success', 'agent_id': 'stub-plan', 'text': 'plan ready'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                result = generate_analysis_plan(
                    str(task_context_path),
                    max_attempts=3,
                )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['analysis_evidence_probe_status'], 'success')
            self.assertTrue(Path(result['analysis_evidence_probe_request_path']).exists())
            self.assertTrue(Path(result['analysis_evidence_probe_result_path']).exists())
            probe_result = json.loads(
                Path(result['analysis_evidence_probe_result_path']).read_text(encoding='utf-8')
            )
            self.assertIn('files_with_loss_inputs', probe_result)
            self.assertEqual(len(call_records), 2)
            self.assertIn('analysis_evidence_probe_result.json', str(call_records[1]['message']))
            manifest = json.loads((experiment_dir / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['agent_calls']), 2)
            self.assertEqual(manifest['agent_calls'][0]['stage'], 'analysis_evidence_probe_generation')
            self.assertEqual(manifest['agent_calls'][1]['stage'], 'analysis_plan_generation')

    def test_generate_then_repair_candidate_loss_with_stubbed_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            case_memory_path = root / 'knowledge_base' / 'case_memories.jsonl'
            case_memory_path.parent.mkdir(parents=True, exist_ok=True)
            case_memory_path.write_text(
                json.dumps(
                    {
                        'schema_version': 'case_memory.v1',
                        'paper_slug': 'historical-paper',
                        'attempt_id': 4,
                        'integration_path': 'adapter_wrapper',
                        'kind': 'agent_code',
                        'objective': 'Historical adapter rescue with a reconstruction anchor.',
                        'strategy_delta': {
                            'what_changes_now': ['add a reconstruction anchor'],
                        },
                        'stop_layer': 'layer3',
                        'error': 'timeout',
                        'passed': False,
                        'primary_metric_name': 'val_ssim',
                        'primary_metric': 0.63,
                        'stage_score': 4,
                        'repair_rounds_used': 1,
                        'repair_hypothesis': 'weighting-only loss starved reconstruction fidelity',
                        'post_stop_layer': 'layer4',
                        'post_error': 'below threshold',
                        'provenance': {'result_path': str(root / 'historical_result.json')},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )
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
                message = str(kwargs['message'])
                candidate_paths = [Path(path) for path in files if path.endswith('candidate_loss.py')]
                adapter_paths = [Path(path) for path in files if path.endswith('sandbox_model_adapter.py')]
                repair_plan_paths = [Path(path) for path in files if 'repair_plan' in Path(path).name]
                if '代码修复补证据阶段' in message:
                    (bundle['experiment_dir'] / 'attempt_1' / 'repair_evidence_probe_request.json').write_text(
                        json.dumps(
                            {
                                'status': 'not_needed',
                                'reason': 'Current failure feedback and code already explain the repair.',
                                'evidence_refs': ['task_context.integration_assessment', 'validator.layer3'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-repair-probe', 'text': 'probe decision written'}

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

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat), patch(
                'loss_transfer.agent.agent_artifact_generator._CASE_MEMORY_PATH',
                case_memory_path,
            ):
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
            self.assertEqual(repair['repair_evidence_probe_status'], 'not_needed')
            self.assertEqual(len(call_records), 3)
            self.assertIn('代码修复补证据阶段', str(call_records[1]['message']))
            self.assertIn(str(output_code_path), call_records[2]['files'])
            self.assertIn('current candidate_loss.py', str(call_records[2]['message']))
            self.assertIn('Historical adapter rescue', str(call_records[2]['message']))
            self.assertIn('weighting-only loss starved reconstruction fidelity', str(call_records[2]['message']))
            manifest = json.loads((bundle['experiment_dir'] / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['agent_calls']), 3)
            self.assertEqual(manifest['agent_calls'][0]['stage'], 'candidate_loss_generation')
            self.assertEqual(manifest['agent_calls'][1]['stage'], 'candidate_loss_repair_evidence_probe_generation')
            self.assertEqual(manifest['agent_calls'][2]['stage'], 'candidate_loss_repair')
            self.assertEqual(manifest['agent_calls'][0]['session_scope'], 'new_request_session')

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
                message = str(kwargs['message'])
                if '代码修复补证据阶段' in message:
                    (bundle['experiment_dir'] / 'attempt_3' / 'repair_evidence_probe_request.json').write_text(
                        json.dumps(
                            {
                                'status': 'not_needed',
                                'reason': 'Failure feedback is already enough.',
                                'evidence_refs': ['validator.layer3'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-repair-probe', 'text': 'probe decision written'}

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

    def test_repair_candidate_loss_runs_probe_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            output_code_path = bundle['experiment_dir'] / 'attempt_4' / 'candidate_loss.py'
            output_code_path.parent.mkdir(parents=True, exist_ok=True)
            output_code_path.write_text(
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).pow(2).mean()\n',
                encoding='utf-8',
            )
            attempt_spec = {
                'name': 'Probe Repair Attempt',
                'kind': 'agent_code',
                'files_to_edit': ['candidate_loss.py'],
                'required_edit_paths': ['sandbox_model_adapter.py'],
            }
            call_records: list[dict[str, object]] = []

            def fake_run_agent_chat(**kwargs):
                files = [str(item) for item in kwargs['files']]
                call_records.append({'files': files, 'message': kwargs['message']})
                message = str(kwargs['message'])
                if '代码修复补证据阶段' in message:
                    request_path = bundle['experiment_dir'] / 'attempt_4' / 'repair_evidence_probe_request.json'
                    script_path = bundle['experiment_dir'] / 'attempt_4' / 'repair_evidence_probe.py'
                    request_path.write_text(
                        json.dumps(
                            {
                                'status': 'probe_needed',
                                'reason': 'Need to confirm which copied path exposes loss_inputs.',
                                'evidence_refs': ['validator.layer3'],
                                'probe_goal': 'Inspect the repository for loss_inputs mentions.',
                                'expected_output_keys': ['files_with_loss_inputs'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    script_path.write_text(
                        'import argparse\n'
                        'import json\n'
                        'from pathlib import Path\n'
                        'parser = argparse.ArgumentParser()\n'
                        'parser.add_argument("--code_repo", required=True)\n'
                        'parser.add_argument("--task_context", required=True)\n'
                        'parser.add_argument("--current_code")\n'
                        'parser.add_argument("--analysis_plan")\n'
                        'parser.add_argument("--failure_feedback")\n'
                        'parser.add_argument("--repair_plan")\n'
                        'parser.add_argument("--output", required=True)\n'
                        'args = parser.parse_args()\n'
                        'repo = Path(args.code_repo)\n'
                        'hits = []\n'
                        'for path in repo.rglob("*.py"):\n'
                        '    text = path.read_text(encoding="utf-8", errors="ignore")\n'
                        '    if "loss_inputs" in text:\n'
                        '        hits.append(str(path.relative_to(repo)))\n'
                        'Path(args.output).write_text(json.dumps({"files_with_loss_inputs": hits[:5]}), encoding="utf-8")\n',
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-repair-probe', 'text': 'probe ready'}

                candidate_paths = [Path(path) for path in files if path.endswith('candidate_loss.py')]
                adapter_paths = [Path(path) for path in files if path.endswith('sandbox_model_adapter.py')]
                repair_plan_paths = [Path(path) for path in files if Path(path).name == 'repair_plan.json']
                for path in candidate_paths:
                    path.write_text(
                        'import torch\n'
                        'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                        '    return (pred - target).abs().mean()\n',
                        encoding='utf-8',
                    )
                for path in adapter_paths:
                    path.write_text('# adapter touched during repair\n', encoding='utf-8')
                for path in repair_plan_paths:
                    path.write_text(
                        json.dumps(
                            {
                                'failure_hypothesis': 'layer3 failed because the repair targeted the wrong copied path',
                                'planned_changes': [
                                    'use the probe result to target the copied adapter path',
                                    'replace the unstable residual branch with a simpler absolute error',
                                ],
                                'target_metric': 'val_ssim',
                                'success_criteria': 'pass layer3 and restore SSIM growth',
                                'fallback_plan': 'if repair still fails, add a reconstruction anchor term',
                                'evidence_refs': [
                                    'validator.layer3',
                                    'repair_evidence_probe_result.files_with_loss_inputs',
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                return {'status': 'success', 'agent_id': 'stub-repair', 'text': 'repaired'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                repair = repair_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    failure_feedback={'stop_layer': 'layer3', 'error': 'loss_inputs missing'},
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )

            self.assertEqual(repair['status'], 'success')
            self.assertEqual(repair['repair_evidence_probe_status'], 'success')
            self.assertTrue(Path(repair['repair_evidence_probe_request_path']).exists())
            self.assertTrue(Path(repair['repair_evidence_probe_result_path']).exists())
            self.assertEqual(len(call_records), 2)
            self.assertIn('代码修复补证据阶段', str(call_records[0]['message']))
            self.assertIn('repair_evidence_probe_result.json', str(call_records[1]['message']))

    def test_repair_candidate_loss_requires_probe_result_reference_when_probe_was_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            output_code_path = bundle['experiment_dir'] / 'attempt_5' / 'candidate_loss.py'
            output_code_path.parent.mkdir(parents=True, exist_ok=True)
            output_code_path.write_text(
                'import torch\n'
                'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                '    return (pred - target).pow(2).mean()\n',
                encoding='utf-8',
            )
            attempt_spec = {
                'name': 'Probe Repair Missing Ref',
                'kind': 'agent_code',
                'files_to_edit': ['candidate_loss.py'],
            }

            def fake_run_agent_chat(**kwargs):
                message = str(kwargs['message'])
                if '代码修复补证据阶段' in message:
                    request_path = bundle['experiment_dir'] / 'attempt_5' / 'repair_evidence_probe_request.json'
                    script_path = bundle['experiment_dir'] / 'attempt_5' / 'repair_evidence_probe.py'
                    request_path.write_text(
                        json.dumps(
                            {
                                'status': 'probe_needed',
                                'reason': 'Need to identify the correct copied path.',
                                'evidence_refs': ['validator.layer3'],
                                'probe_goal': 'Inspect repository for loss_inputs mentions.',
                                'expected_output_keys': ['files_with_loss_inputs'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    script_path.write_text(
                        'import argparse\n'
                        'import json\n'
                        'from pathlib import Path\n'
                        'parser = argparse.ArgumentParser()\n'
                        'parser.add_argument("--code_repo", required=True)\n'
                        'parser.add_argument("--task_context", required=True)\n'
                        'parser.add_argument("--current_code")\n'
                        'parser.add_argument("--analysis_plan")\n'
                        'parser.add_argument("--failure_feedback")\n'
                        'parser.add_argument("--repair_plan")\n'
                        'parser.add_argument("--output", required=True)\n'
                        'args = parser.parse_args()\n'
                        'Path(args.output).write_text(json.dumps({"files_with_loss_inputs": ["models/demo.py"]}), encoding="utf-8")\n',
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-repair-probe', 'text': 'probe ready'}

                for item in kwargs['files']:
                    path = Path(str(item))
                    if path.name == 'candidate_loss.py':
                        path.write_text(
                            'import torch\n'
                            'def sandbox_loss(pred, target, mask=None, **kwargs):\n'
                            '    return (pred - target).abs().mean()\n',
                            encoding='utf-8',
                        )
                    if path.name == 'repair_plan.json':
                        path.write_text(
                            json.dumps(
                                {
                                    'failure_hypothesis': 'repair targeted the wrong path',
                                    'planned_changes': ['switch to absolute residual'],
                                    'target_metric': 'val_ssim',
                                    'success_criteria': 'pass layer3',
                                    'fallback_plan': 'add a reconstruction term next round',
                                    'evidence_refs': ['validator.layer3'],
                                },
                                ensure_ascii=False,
                            ),
                            encoding='utf-8',
                        )
                return {'status': 'success', 'agent_id': 'stub-repair', 'text': 'repaired'}

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat):
                repair = repair_candidate_loss(
                    task_context_path=str(bundle['task_context_path']),
                    attempt_spec=attempt_spec,
                    output_code_path=str(output_code_path),
                    failure_feedback={'stop_layer': 'layer3', 'error': 'loss_inputs missing'},
                    analysis_plan_path=str(bundle['analysis_plan_path']),
                )

            self.assertEqual(repair['status'], 'error')
            self.assertIn('repair_evidence_probe_result', repair['error'])

    def test_generate_followup_attempt_uses_latest_result_and_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_task_context_bundle(root)
            case_memory_path = root / 'knowledge_base' / 'case_memories.jsonl'
            case_memory_path.parent.mkdir(parents=True, exist_ok=True)
            case_memory_path.write_text(
                json.dumps(
                    {
                        'schema_version': 'case_memory.v1',
                        'paper_slug': 'historical-paper',
                        'attempt_id': 7,
                        'integration_path': 'adapter_wrapper',
                        'kind': 'agent_code',
                        'objective': 'Historical follow-up that added a reconstruction anchor after SSIM collapse.',
                        'strategy_delta': {
                            'what_changes_now': ['add a reconstruction anchor', 'stop repeating weighting-only plans'],
                        },
                        'stop_layer': 'layer4',
                        'error': 'SSIM too low',
                        'passed': True,
                        'primary_metric_name': 'swinir',
                        'primary_metric': 0.71,
                        'stage_score': 6,
                        'repair_rounds_used': 1,
                        'repair_hypothesis': 'loss-only weighting branch could not recover SSIM',
                        'post_stop_layer': None,
                        'post_error': None,
                        'provenance': {'result_path': str(root / 'historical_followup_result.json')},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )
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
                message = str(kwargs['message'])
                if '逐轮重规划补证据阶段' in message:
                    (bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe_request.json').write_text(
                        json.dumps(
                            {
                                'status': 'not_needed',
                                'reason': 'Existing result and repair plan already explain the next change.',
                                'evidence_refs': ['result.layer4', 'repair_plan.fallback_plan'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-followup-probe', 'text': 'probe decision written'}

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

            with patch('loss_transfer.agent.agent_artifact_generator.run_agent_chat', side_effect=fake_run_agent_chat), patch(
                'loss_transfer.agent.agent_artifact_generator._CASE_MEMORY_PATH',
                case_memory_path,
            ):
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
            self.assertEqual(result['followup_evidence_probe_status'], 'not_needed')
            self.assertEqual(result['attempt_spec']['strategy_delta']['previous_attempt_id'], 1)
            self.assertEqual(len(call_records), 2)
            self.assertIn(str(result_path), call_records[1]['files'])
            self.assertIn(str(repair_plan_path), call_records[1]['files'])
            self.assertIn('逐轮重规划补证据阶段', str(call_records[0]['message']))
            self.assertIn('逐轮重规划阶段', str(call_records[1]['message']))
            self.assertIn('Historical follow-up that added a reconstruction anchor', str(call_records[1]['message']))
            self.assertIn('loss-only weighting branch could not recover SSIM', str(call_records[1]['message']))

    def test_generate_followup_attempt_runs_probe_before_replan(self) -> None:
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
                        'stop_layer': 'layer3',
                        'error': 'loss_inputs missing',
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
                message = str(kwargs['message'])
                if '逐轮重规划补证据阶段' in message:
                    request_path = bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe_request.json'
                    script_path = bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe.py'
                    request_path.write_text(
                        json.dumps(
                            {
                                'status': 'probe_needed',
                                'reason': 'Need to confirm which copied model path exposes loss_inputs.',
                                'evidence_refs': ['result.layer3'],
                                'probe_goal': 'Scan repo for loss_inputs symbols.',
                                'expected_output_keys': ['files_with_loss_inputs'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    script_path.write_text(
                        'import argparse\n'
                        'import json\n'
                        'from pathlib import Path\n'
                        'parser = argparse.ArgumentParser()\n'
                        'parser.add_argument("--code_repo", required=True)\n'
                        'parser.add_argument("--task_context", required=True)\n'
                        'parser.add_argument("--latest_result")\n'
                        'parser.add_argument("--latest_repair_plan")\n'
                        'parser.add_argument("--analysis_plan")\n'
                        'parser.add_argument("--trajectory")\n'
                        'parser.add_argument("--output", required=True)\n'
                        'args = parser.parse_args()\n'
                        'repo = Path(args.code_repo)\n'
                        'hits = []\n'
                        'for path in repo.rglob("*.py"):\n'
                        '    text = path.read_text(encoding="utf-8", errors="ignore")\n'
                        '    if "loss_inputs" in text:\n'
                        '        hits.append(str(path.relative_to(repo)))\n'
                        'Path(args.output).write_text(json.dumps({"files_with_loss_inputs": hits[:5]}), encoding="utf-8")\n',
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-followup-probe', 'text': 'probe ready'}

                output_attempt_path.write_text(
                    json.dumps(
                        {
                            'name': 'Probe-informed follow-up attempt',
                            'kind': 'agent_code',
                            'objective': 'Patch the copied adapter/model path that already exposes loss_inputs.',
                            'files_to_edit': ['candidate_loss.py'],
                            'required_edit_paths': ['sandbox_model_adapter.py'],
                            'evidence_refs': [
                                'result.layer3',
                                'followup_evidence_probe_result.files_with_loss_inputs',
                            ],
                            'strategy_delta': {
                                'previous_attempt_id': 1,
                                'why_previous_failed': 'The previous attempt failed before training because loss_inputs wiring was missing.',
                                'what_changes_now': ['use probe result to target the copied model/adapter path'],
                                'why_not_repeat_previous': 'the previous attempt lacked evidence about where loss_inputs were exposed',
                                'expected_signal': 'layer3 should pass once the correct path is edited',
                            },
                            'run_training': True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                return {'status': 'success', 'agent_id': 'stub-followup-plan', 'text': 'attempt written'}

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
            self.assertEqual(result['followup_evidence_probe_status'], 'success')
            self.assertTrue(Path(result['followup_evidence_probe_request_path']).exists())
            self.assertTrue(Path(result['followup_evidence_probe_result_path']).exists())
            probe_result = json.loads(
                Path(result['followup_evidence_probe_result_path']).read_text(encoding='utf-8')
            )
            self.assertIn('files_with_loss_inputs', probe_result)
            self.assertEqual(len(call_records), 2)
            self.assertIn('followup_attempt_2_evidence_probe_result.json', str(call_records[1]['message']))

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
                message = str(kwargs['message'])
                if '逐轮重规划补证据阶段' in message:
                    (bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe_request.json').write_text(
                        json.dumps(
                            {
                                'status': 'not_needed',
                                'reason': 'The existing failure artifact is enough.',
                                'evidence_refs': ['result.layer4'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-followup-probe', 'text': 'probe decision written'}

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

    def test_generate_followup_attempt_requires_probe_result_reference_when_probe_was_used(self) -> None:
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
                        'stop_layer': 'layer3',
                        'error': 'loss_inputs missing',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            output_attempt_path = bundle['experiment_dir'] / 'followup_attempt_2.json'

            def fake_run_agent_chat(**kwargs):
                message = str(kwargs['message'])
                if '逐轮重规划补证据阶段' in message:
                    request_path = bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe_request.json'
                    script_path = bundle['experiment_dir'] / 'followup_attempt_2_evidence_probe.py'
                    request_path.write_text(
                        json.dumps(
                            {
                                'status': 'probe_needed',
                                'reason': 'Need to confirm copied model path.',
                                'evidence_refs': ['result.layer3'],
                                'probe_goal': 'Scan repo for loss_inputs symbols.',
                                'expected_output_keys': ['files_with_loss_inputs'],
                            },
                            ensure_ascii=False,
                        ),
                        encoding='utf-8',
                    )
                    script_path.write_text(
                        'import argparse\n'
                        'import json\n'
                        'from pathlib import Path\n'
                        'parser = argparse.ArgumentParser()\n'
                        'parser.add_argument("--code_repo", required=True)\n'
                        'parser.add_argument("--task_context", required=True)\n'
                        'parser.add_argument("--latest_result")\n'
                        'parser.add_argument("--latest_repair_plan")\n'
                        'parser.add_argument("--analysis_plan")\n'
                        'parser.add_argument("--trajectory")\n'
                        'parser.add_argument("--output", required=True)\n'
                        'args = parser.parse_args()\n'
                        'Path(args.output).write_text(json.dumps({"files_with_loss_inputs": ["models/demo.py"]}), encoding="utf-8")\n',
                        encoding='utf-8',
                    )
                    return {'status': 'success', 'agent_id': 'stub-followup-probe', 'text': 'probe ready'}

                output_attempt_path.write_text(
                    json.dumps(
                        {
                            'name': 'Probe ignored follow-up attempt',
                            'kind': 'agent_code',
                            'objective': 'Try a different path without citing the probe.',
                            'files_to_edit': ['candidate_loss.py'],
                            'evidence_refs': ['result.layer3'],
                            'strategy_delta': {
                                'previous_attempt_id': 1,
                                'why_previous_failed': 'Missing loss_inputs',
                                'what_changes_now': ['guess a different adapter tweak'],
                                'why_not_repeat_previous': 'need a different change',
                                'expected_signal': 'layer3 should pass',
                            },
                            'run_training': True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                return {'status': 'success', 'agent_id': 'stub-followup-plan', 'text': 'attempt written'}

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
            self.assertIn('followup_evidence_probe_result', result['error'])

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

