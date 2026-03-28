from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.agent.validate_analysis_plan import validate_analysis_plan  # noqa: E402
from loss_transfer.common.integration_path import describe_integration_path  # noqa: E402
from loss_transfer.formula.formula_interface_analysis import analyze_formula_interface  # noqa: E402


class IntegrationPathContractTests(unittest.TestCase):
    def test_describe_integration_path_maps_alias_to_canonical(self) -> None:
        description = describe_integration_path('add_loss_inputs_adapter')

        self.assertEqual(description['status'], 'alias_mapped')
        self.assertEqual(description['canonical_path'], 'adapter_wrapper')

    def test_formula_interface_emits_canonical_path_and_preserves_raw_alias(self) -> None:
        analysis = analyze_formula_interface(
            {
                'latex': ['L = |pred - target|'],
                'symbol_map': {
                    'x': 'pred',
                    'y': 'target',
                },
                'params': {},
            }
        )

        self.assertEqual(analysis['recommended_integration_path'], 'loss_only')
        self.assertEqual(analysis['recommended_integration_path_raw'], 'reuse_existing_loss_config')
        self.assertEqual(analysis['recommended_integration_path_status'], 'alias_mapped')

    def test_validate_analysis_plan_normalizes_alias_path(self) -> None:
        result = validate_analysis_plan(
            {
                'summary': 'Alias path normalization test',
                'stop_on_first_pass': False,
                'integration_decision': {
                    'path': 'model_output_extension',
                    'rationale': 'Need copied model outputs for auxiliary tensors.',
                    'evidence_refs': ['paper.loss', 'code.forward'],
                },
                'attempts': [
                    {
                        'name': 'Attempt 1',
                        'kind': 'agent_code',
                        'objective': 'Implement deeper routing.',
                        'evidence_refs': ['paper.loss'],
                    }
                ],
            }
        )

        self.assertEqual(result['status'], 'warning')
        self.assertIn('normalized from', result['warnings'][0])
        self.assertEqual(
            result['normalized_plan']['integration_decision']['path'],
            'extend_model_outputs',
        )
        self.assertEqual(
            result['normalized_plan']['integration_decision']['path_raw'],
            'model_output_extension',
        )
        self.assertEqual(
            result['normalized_plan']['integration_decision']['path_status'],
            'alias_mapped',
        )

    def test_validate_analysis_plan_requires_probe_result_reference_when_probe_was_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            probe_request_path = experiment_dir / 'analysis_evidence_probe_request.json'
            probe_result_path = experiment_dir / 'analysis_evidence_probe_result.json'
            probe_request_path.write_text(
                json.dumps(
                    {
                        'status': 'probe_needed',
                        'reason': 'Need to inspect model-side aux outputs.',
                        'evidence_refs': ['code_analysis.focus_files[0]'],
                        'probe_goal': 'Find loss_inputs wiring.',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            probe_result_path.write_text(
                json.dumps(
                    {
                        'files_with_loss_inputs': ['models/demo.py'],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            task_context = {
                'paper_analysis': {'available': True},
                'code_analysis': {'available': True},
                'paths': {
                    'analysis_evidence_probe_request_path': str(probe_request_path),
                    'analysis_evidence_probe_result_path': str(probe_result_path),
                },
            }

            missing_probe_ref = validate_analysis_plan(
                {
                    'summary': 'Probe result ignored',
                    'stop_on_first_pass': False,
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'Need adapter outputs.',
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
                task_context=task_context,
            )

            self.assertEqual(missing_probe_ref['status'], 'error')
            self.assertTrue(
                any('analysis_evidence_probe_result' in error for error in missing_probe_ref['errors'])
            )

            resolved_probe_ref = validate_analysis_plan(
                {
                    'summary': 'Probe result consumed',
                    'stop_on_first_pass': False,
                    'integration_decision': {
                        'path': 'adapter_wrapper',
                        'rationale': 'Need adapter outputs.',
                        'evidence_refs': [
                            'paper.loss',
                            'analysis_evidence_probe_result.files_with_loss_inputs',
                        ],
                    },
                    'attempts': [
                        {
                            'name': 'Attempt 1',
                            'kind': 'agent_code',
                            'objective': 'Implement adapter-aware routing.',
                            'evidence_refs': ['analysis_evidence_probe_result.files_with_loss_inputs'],
                        }
                    ],
                },
                task_context=task_context,
            )

            self.assertIn(resolved_probe_ref['status'], {'ok', 'warning'})
            self.assertEqual(
                resolved_probe_ref['normalized_plan']['evidence_validation']['probe_result_refs'],
                [
                    'analysis_evidence_probe_result.files_with_loss_inputs',
                    'analysis_evidence_probe_result.files_with_loss_inputs',
                ],
            )


if __name__ == '__main__':
    unittest.main()
