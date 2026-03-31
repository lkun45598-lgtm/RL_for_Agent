"""
@file test_case_memory_retriever.py

@description Regression tests for the shared case-memory retriever.
@author kongzhiquan
@contributors kongzhiquan
@date 2026-03-28
@version 1.1.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 add regression tests for shared case-memory retrieval
  - 2026-03-30 kongzhiquan: v1.1.0 add success-aware retrieval coverage for case_memory.v2
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.memory.case_memory_retriever import (  # noqa: E402
    format_case_memory_block,
    load_similar_case_memories,
)


class CaseMemoryRetrieverTests(unittest.TestCase):
    def test_load_similar_case_memories_prefers_matching_stop_layer_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_memory_path = root / 'knowledge_base' / 'case_memories.jsonl'
            case_memory_path.parent.mkdir(parents=True, exist_ok=True)
            case_memory_path.write_text(
                '\n'.join(
                    [
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
                        ),
                        json.dumps(
                            {
                                'schema_version': 'case_memory.v1',
                                'paper_slug': 'timeout-paper',
                                'attempt_id': 2,
                                'integration_path': 'adapter_wrapper',
                                'kind': 'agent_code',
                                'objective': 'Stabilize weighting after timeout.',
                                'strategy_delta': {
                                    'what_changes_now': ['clip gradients'],
                                },
                                'stop_layer': 'layer3',
                                'error': 'timeout',
                                'passed': False,
                                'primary_metric_name': 'swinir',
                                'primary_metric': 0.63,
                                'stage_score': 4,
                                'repair_rounds_used': 1,
                                'repair_hypothesis': 'the branch was numerically unstable',
                                'post_stop_layer': 'layer4',
                                'post_error': 'below threshold',
                                'provenance': {'result_path': str(root / 'timeout_result.json')},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                'schema_version': 'case_memory.v1',
                                'paper_slug': 'loss-only-paper',
                                'attempt_id': 3,
                                'integration_path': 'loss_only',
                                'kind': 'formula_variant',
                                'objective': 'Faithful loss-only formula replay.',
                                'stop_layer': None,
                                'error': None,
                                'passed': True,
                                'primary_metric_name': 'val_ssim',
                                'primary_metric': 0.68,
                                'stage_score': 5,
                                'repair_rounds_used': 0,
                                'provenance': {'result_path': str(root / 'loss_only_result.json')},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + '\n',
                encoding='utf-8',
            )

            task_context = {
                'paper_slug': 'current-paper',
                'integration_assessment': {
                    'recommended_path': 'adapter_wrapper',
                },
                'paths': {
                    'experiment_dir': str(root / 'experiments' / 'current-paper'),
                },
            }

            cases = load_similar_case_memories(
                task_context=task_context,
                latest_attempt_result={
                    'attempt_id': 1,
                    'kind': 'agent_code',
                    'stop_layer': 'layer4',
                    'error': 'SSIM collapsed again',
                },
                case_memory_path=case_memory_path,
                top_k=2,
            )

            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]['paper_slug'], 'historical-paper')
            self.assertEqual(cases[0]['attempt_id'], 7)

            memory_block = format_case_memory_block(cases[:1])
            self.assertIn('Historical follow-up that added a reconstruction anchor', memory_block)
            self.assertIn('loss-only weighting branch could not recover SSIM', memory_block)

    def test_load_similar_case_memories_prefers_effective_repairs_in_case_memory_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_memory_path = root / 'knowledge_base' / 'case_memories.jsonl'
            case_memory_path.parent.mkdir(parents=True, exist_ok=True)
            case_memory_path.write_text(
                '\n'.join(
                    [
                        json.dumps(
                            {
                                'schema_version': 'case_memory.v2',
                                'paper_slug': 'effective-paper',
                                'attempt_id': 8,
                                'integration_path': 'adapter_wrapper',
                                'kind': 'agent_code',
                                'objective': 'Recover adapter training after timeout with a reconstruction anchor.',
                                'files_to_edit': ['candidate_loss.py'],
                                'required_edit_paths': ['sandbox_model_adapter.py'],
                                'stop_layer': None,
                                'error': None,
                                'trigger_stop_layer': 'layer3',
                                'trigger_error': 'timeout while validating adapter branch',
                                'passed': True,
                                'primary_metric_name': 'val_ssim',
                                'primary_metric': 0.69,
                                'stage_score': 6,
                                'repair_rounds_used': 1,
                                'baseline_delta': 0.03,
                                'repair_hypothesis': 'the adapter path needed a lighter reconstruction anchor',
                                'post_stop_layer': None,
                                'post_error': None,
                                'failure_signature': {
                                    'stop_layer': 'layer3',
                                    'error_family': 'timeout',
                                },
                                'repair_outcome': {
                                    'effective': True,
                                    'resolved_failure': True,
                                    'improved': True,
                                    'reverted': False,
                                    'post_stop_layer': None,
                                },
                                'provenance': {'result_path': str(root / 'effective_result.json')},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                'schema_version': 'case_memory.v2',
                                'paper_slug': 'reverted-paper',
                                'attempt_id': 9,
                                'integration_path': 'adapter_wrapper',
                                'kind': 'agent_code',
                                'objective': 'Retry timeout fix with the same adapter edit scope.',
                                'files_to_edit': ['candidate_loss.py'],
                                'required_edit_paths': ['sandbox_model_adapter.py'],
                                'stop_layer': 'layer3',
                                'error': 'timeout',
                                'passed': False,
                                'primary_metric_name': 'val_ssim',
                                'primary_metric': 0.61,
                                'stage_score': 4,
                                'repair_rounds_used': 1,
                                'baseline_delta': -0.02,
                                'repair_hypothesis': 'the same branch remained unstable',
                                'post_stop_layer': 'layer2',
                                'post_error': 'shape mismatch',
                                'failure_signature': {
                                    'stop_layer': 'layer3',
                                    'error_family': 'timeout',
                                },
                                'repair_outcome': {
                                    'effective': False,
                                    'resolved_failure': False,
                                    'improved': False,
                                    'reverted': True,
                                    'post_stop_layer': 'layer2',
                                },
                                'provenance': {'result_path': str(root / 'reverted_result.json')},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + '\n',
                encoding='utf-8',
            )

            task_context = {
                'paper_slug': 'current-paper',
                'integration_assessment': {
                    'recommended_path': 'adapter_wrapper',
                },
                'paths': {
                    'experiment_dir': str(root / 'experiments' / 'current-paper'),
                },
            }

            cases = load_similar_case_memories(
                task_context=task_context,
                attempt_spec={
                    'kind': 'agent_code',
                    'objective': 'Repair adapter timeout without repeating the same unstable plan.',
                    'files_to_edit': ['candidate_loss.py'],
                    'required_edit_paths': ['sandbox_model_adapter.py'],
                },
                failure_feedback={
                    'stop_layer': 'layer3',
                    'error': 'timeout while validating adapter branch',
                },
                case_memory_path=case_memory_path,
                top_k=2,
            )

            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]['paper_slug'], 'effective-paper')
            self.assertTrue(cases[0]['repair_outcome']['effective'])
            self.assertTrue(cases[0]['repair_outcome']['resolved_failure'])
            self.assertTrue(cases[1]['repair_outcome']['reverted'])


if __name__ == '__main__':
    unittest.main()
