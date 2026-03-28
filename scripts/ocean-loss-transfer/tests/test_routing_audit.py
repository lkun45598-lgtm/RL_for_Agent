from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.common.routing_audit import build_routing_audit, write_routing_audit  # noqa: E402


class RoutingAuditTests(unittest.TestCase):
    def test_build_routing_audit_tracks_override_and_normalization(self) -> None:
        audit = build_routing_audit(
            paper_slug='demo-paper',
            task_context={
                'paths': {
                    'task_context_path': '/tmp/task_context.json',
                    'analysis_plan_path': '/tmp/analysis_plan.json',
                    'routing_audit_path': '/tmp/routing_audit.json',
                },
                'integration_assessment': {
                    'recommended_path': 'loss_only',
                    'recommended_path_raw': 'reuse_existing_loss_config',
                    'recommended_path_status': 'alias_mapped',
                    'recommended_path_source': 'formula_interface',
                    'recommended_path_reason': 'Simple pointwise formula.',
                    'recommended_path_evidence_refs': ['formula_interface.change_level_reasons'],
                },
            },
            analysis_plan={
                'integration_decision': {
                    'path': 'extend_model_outputs',
                    'path_raw': 'model_output_extension',
                    'path_status': 'alias_mapped',
                    'path_source': 'analysis_plan.integration_decision',
                    'rationale': 'Need copied model outputs.',
                    'evidence_refs': ['paper.loss', 'code.forward'],
                }
            },
            routing_audit_path='/tmp/routing_audit.json',
            analysis_plan_path='/tmp/analysis_plan.json',
        )

        self.assertEqual(audit['routes']['task_context']['canonical_path'], 'loss_only')
        self.assertEqual(audit['routes']['analysis_plan']['canonical_path'], 'extend_model_outputs')
        self.assertEqual(audit['routes']['effective']['canonical_path'], 'extend_model_outputs')
        self.assertEqual(audit['routes']['effective']['selected_from'], 'analysis_plan')
        self.assertEqual(audit['routes']['effective']['selection_reason'], 'analysis_plan_overrides_task_context')

    def test_write_routing_audit_persists_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            result = write_routing_audit(
                experiment_dir=experiment_dir,
                paper_slug='demo-paper',
                task_context={
                    'paths': {
                        'task_context_path': str(experiment_dir / 'task_context.json'),
                        'analysis_plan_path': str(experiment_dir / 'analysis_plan.json'),
                    },
                    'integration_assessment': {
                        'recommended_path': 'adapter_wrapper',
                        'recommended_path_raw': 'add_loss_inputs_adapter',
                        'recommended_path_status': 'alias_mapped',
                    },
                },
            )

            audit_path = Path(result['routing_audit_path'])
            payload = json.loads(audit_path.read_text(encoding='utf-8'))
            self.assertTrue(audit_path.exists())
            self.assertEqual(payload['schema_version'], 'routing_audit.v1')
            self.assertEqual(payload['routes']['effective']['canonical_path'], 'adapter_wrapper')
            self.assertEqual(result['effective_integration_path'], 'adapter_wrapper')
            self.assertEqual(result['effective_integration_path_source'], 'task_context')


if __name__ == '__main__':
    unittest.main()
