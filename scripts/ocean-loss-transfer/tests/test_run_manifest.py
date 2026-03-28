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

from loss_transfer.common.run_manifest import append_run_manifest_agent_call, write_run_manifest  # noqa: E402


class RunManifestTests(unittest.TestCase):
    def test_write_run_manifest_records_paths_hashes_and_service_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            task_context_path = experiment_dir / 'task_context.json'
            analysis_plan_path = experiment_dir / 'analysis_plan.json'
            routing_audit_path = experiment_dir / 'routing_audit.json'
            contract_validation_path = experiment_dir / 'contract_validation.json'
            loss_formula_path = experiment_dir / 'loss_formula.json'

            task_context_path.write_text(json.dumps({'paper_slug': 'demo-paper'}), encoding='utf-8')
            analysis_plan_path.write_text(json.dumps({'summary': 'demo plan'}), encoding='utf-8')
            routing_audit_path.write_text(json.dumps({'routes': {}}), encoding='utf-8')
            contract_validation_path.write_text(json.dumps({'status': 'ok'}), encoding='utf-8')
            loss_formula_path.write_text(json.dumps({'symbol_map': {}}), encoding='utf-8')

            task_context = {
                'paths': {
                    'experiment_dir': str(experiment_dir),
                    'task_context_path': str(task_context_path),
                    'analysis_plan_path': str(analysis_plan_path),
                    'routing_audit_path': str(routing_audit_path),
                    'contract_validation_path': str(contract_validation_path),
                    'loss_formula_path': str(loss_formula_path),
                    'run_manifest_path': str(experiment_dir / 'run_manifest.json'),
                }
            }

            with patch(
                'loss_transfer.common.run_manifest.fetch_service_health',
                return_value={'status': 'ok', 'provider': 'openai', 'model': 'gpt-5.4', 'apiMode': 'responses'},
            ), patch(
                'loss_transfer.common.run_manifest._resolve_git_sha',
                return_value='abc123',
            ):
                result = write_run_manifest(
                    experiment_dir=experiment_dir,
                    paper_slug='demo-paper',
                    task_context=task_context,
                    mode='agent_loop',
                    bootstrap_formula=True,
                    max_attempts=4,
                    auto_generate_plan=False,
                    service_url='http://localhost:8787',
                    session_policy='new_request_session_per_call',
                )

            manifest = json.loads(Path(result['run_manifest_path']).read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema_version'], 'run_manifest.v1')
            self.assertEqual(manifest['paper_slug'], 'demo-paper')
            self.assertEqual(manifest['code_version']['git_commit_sha'], 'abc123')
            self.assertEqual(manifest['execution']['mode'], 'agent_loop')
            self.assertEqual(manifest['execution']['session_policy'], 'new_request_session_per_call')
            self.assertEqual(manifest['service']['health']['provider'], 'openai')
            self.assertIsNotNone(manifest['hashes']['task_context_sha256'])
            self.assertIsNotNone(manifest['hashes']['analysis_plan_sha256'])
            self.assertIsNotNone(manifest['hashes']['contract_validation_sha256'])
            self.assertEqual(manifest['paths']['contract_validation_path'], str(contract_validation_path))
            self.assertEqual(manifest['paths']['run_manifest_path'], str(experiment_dir / 'run_manifest.json'))

    def test_append_run_manifest_agent_call_persists_session_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            manifest_path = experiment_dir / 'run_manifest.json'
            manifest_path.write_text(
                json.dumps(
                    {
                        'schema_version': 'run_manifest.v1',
                        'paper_slug': 'demo-paper',
                        'agent_calls': [],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            append_run_manifest_agent_call(
                manifest_path,
                {
                    'stage': 'analysis_plan_generation',
                    'session_scope': 'new_request_session',
                    'requested_agent_id': None,
                    'resolved_agent_id': 'agt-demo',
                    'agent_response_path': str(experiment_dir / 'analysis_plan_agent_response.json'),
                },
            )

            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['agent_calls']), 1)
            self.assertEqual(manifest['agent_calls'][0]['stage'], 'analysis_plan_generation')
            self.assertEqual(manifest['agent_calls'][0]['session_scope'], 'new_request_session')
            self.assertEqual(manifest['agent_calls'][0]['resolved_agent_id'], 'agt-demo')


if __name__ == '__main__':
    unittest.main()
