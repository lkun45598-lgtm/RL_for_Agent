from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.common.contract_validation import write_contract_validation  # noqa: E402


class ContractValidationTests(unittest.TestCase):
    def _task_context(
        self,
        experiment_dir: Path,
        *,
        recommended_path: str = 'loss_only',
        code_repo_path: str | None = None,
        include_code_context: bool = False,
        probe_status: str | None = None,
    ) -> dict[str, object]:
        task_context_path = experiment_dir / 'task_context.json'
        task_context = {
            'paper_slug': 'demo-paper',
            'inputs': {
                'code_repo_path': code_repo_path,
            },
            'paths': {
                'experiment_dir': str(experiment_dir),
                'task_context_path': str(task_context_path),
                'analysis_plan_path': str(experiment_dir / 'analysis_plan.json'),
                'routing_audit_path': str(experiment_dir / 'routing_audit.json'),
                'contract_validation_path': str(experiment_dir / 'contract_validation.json'),
                'analysis_evidence_probe_request_path': str(experiment_dir / 'analysis_evidence_probe_request.json'),
                'analysis_evidence_probe_result_path': str(experiment_dir / 'analysis_evidence_probe_result.json'),
            },
            'integration_assessment': {
                'recommended_path': recommended_path,
                'recommended_path_raw': recommended_path,
                'recommended_path_status': 'exact',
                'recommended_path_source': 'task_context',
            },
        }
        if include_code_context:
            task_context['prepared_context'] = {
                'primary_files': [
                    {
                        'path': 'losses/demo_loss.py',
                        'content': 'def demo_loss(pred, target):\n    return pred.mean()\n',
                    }
                ]
            }
            task_context['code_analysis'] = {
                'available': True,
                'focus_files': [
                    {
                        'path': 'losses/demo_loss.py',
                    }
                ],
            }
        if probe_status is not None:
            request_path = experiment_dir / 'analysis_evidence_probe_request.json'
            request_path.write_text(
                json.dumps(
                    {
                        'status': probe_status,
                        'reason': 'Need more evidence for route selection.',
                        'evidence_refs': ['code_analysis.focus_files[0]'],
                        'probe_goal': 'Check model-side aux outputs.',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            if probe_status == 'probe_needed':
                (experiment_dir / 'analysis_evidence_probe_result.json').write_text(
                    json.dumps(
                        {
                            'files_with_loss_inputs': ['models/demo.py'],
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
        task_context_path.write_text(json.dumps(task_context, ensure_ascii=False), encoding='utf-8')
        return task_context

    def test_write_contract_validation_accepts_consistent_override_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            repo_dir = experiment_dir / 'repo'
            repo_dir.mkdir(parents=True, exist_ok=True)
            task_context = self._task_context(
                experiment_dir,
                recommended_path='loss_only',
                code_repo_path=str(repo_dir),
            )

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
                analysis_plan={
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'Need adapter-routed aux tensors.',
                        'evidence_refs': ['paper.loss', 'code.forward'],
                    }
                },
            )

            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['effective_integration_path'], 'adapter_wrapper')
            self.assertEqual(result['effective_integration_path_source'], 'analysis_plan')
            self.assertTrue(Path(result['contract_validation_path']).exists())

    def test_write_contract_validation_fails_on_invalid_task_context_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = self._task_context(
                experiment_dir,
                recommended_path='agent_decides',
            )

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
            )

            self.assertEqual(result['status'], 'error')
            self.assertTrue(
                any('task_context.integration_assessment.recommended_path' in error for error in result['errors'])
            )
            self.assertTrue(Path(result['contract_validation_path']).exists())

    def test_write_contract_validation_fails_on_invalid_analysis_plan_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = self._task_context(experiment_dir)
            analysis_plan_path = experiment_dir / 'analysis_plan.json'
            analysis_plan_path.write_text(
                json.dumps(
                    {
                        'summary': 'Bad plan',
                        'stop_on_first_pass': False,
                        'integration_decision': {
                            'path': 'agent_decides',
                            'rationale': 'Let the agent improvise.',
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

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
                analysis_plan_path=str(analysis_plan_path),
            )

            self.assertEqual(result['status'], 'error')
            self.assertTrue(any('analysis_plan.json validation failed' in error for error in result['errors']))
            self.assertTrue(Path(result['contract_validation_path']).exists())

    def test_write_contract_validation_requires_code_repo_for_adapter_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = self._task_context(
                experiment_dir,
                recommended_path='adapter_wrapper',
            )

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
            )

            self.assertEqual(result['status'], 'error')
            self.assertTrue(any('inputs.code_repo_path' in error for error in result['errors']))

    def test_write_contract_validation_allows_bootstrap_plan_without_integration_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = self._task_context(experiment_dir, recommended_path='loss_only')

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
                analysis_plan={
                    'summary': 'Bootstrap plan',
                    'stop_on_first_pass': False,
                    'attempts': [
                        {
                            'name': 'Attempt 1',
                            'kind': 'formula_variant',
                            'variant': 'faithful',
                            'run_training': True,
                        }
                    ],
                },
            )

            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['effective_integration_path'], 'loss_only')
            self.assertEqual(result['effective_integration_path_source'], 'task_context')

    def test_write_contract_validation_fails_when_probe_result_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context = self._task_context(
                experiment_dir,
                recommended_path='adapter_wrapper',
                code_repo_path=str(experiment_dir),
                include_code_context=True,
                probe_status='probe_needed',
            )
            analysis_plan_path = experiment_dir / 'analysis_plan.json'
            analysis_plan_path.write_text(
                json.dumps(
                    {
                        'summary': 'Probe result was ignored',
                        'stop_on_first_pass': False,
                        'integration_decision': {
                            'path': 'adapter_wrapper',
                            'rationale': 'Need model-side aux tensors.',
                            'evidence_refs': ['paper.loss', 'code.model_forward'],
                        },
                        'attempts': [
                            {
                                'name': 'Attempt 1',
                                'kind': 'agent_code',
                                'objective': 'Implement adapter-aware routing.',
                                'evidence_refs': ['paper.loss'],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            result = write_contract_validation(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context=task_context,
                analysis_plan_path=str(analysis_plan_path),
            )

            self.assertEqual(result['status'], 'error')
            self.assertTrue(
                any('analysis_evidence_probe_result' in error for error in result['errors'])
            )


if __name__ == '__main__':
    unittest.main()
