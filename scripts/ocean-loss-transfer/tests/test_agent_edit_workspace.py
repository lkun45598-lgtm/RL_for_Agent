from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.agent.agent_edit_workspace import (  # noqa: E402
    check_required_edit_paths,
    load_existing_touched_paths,
    prepare_attempt_edit_workspace,
    resolve_requested_override_files,
)


class AgentEditWorkspaceTests(unittest.TestCase):
    def test_resolve_requested_override_files_uses_recommended_path_defaults(self) -> None:
        resolved = resolve_requested_override_files(
            {
                'integration_assessment': {
                    'recommended_path': 'extend_model_outputs',
                }
            },
            {'files_to_edit': ['candidate_loss.py']},
        )

        self.assertEqual(
            resolved,
            {
                'files': ['sandbox_model_adapter.py', 'sandbox_trainer.py'],
                'trees': ['models'],
            },
        )

    def test_prepare_attempt_edit_workspace_writes_manifest_and_copies_attempt_scoped_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_code_path = root / 'attempt_1' / 'candidate_loss.py'
            fake_adapter = root / 'sources' / 'sandbox_model_adapter.py'
            fake_trainer = root / 'sources' / 'sandbox_trainer.py'
            fake_models = root / 'sources' / 'models'
            fake_models.mkdir(parents=True)
            (fake_models / '__init__.py').write_text('# fake models\n', encoding='utf-8')
            fake_adapter.parent.mkdir(parents=True, exist_ok=True)
            fake_adapter.write_text('# fake adapter\n', encoding='utf-8')
            fake_trainer.write_text('# fake trainer\n', encoding='utf-8')

            workspace = prepare_attempt_edit_workspace(
                task_context={
                    'integration_assessment': {
                        'recommended_path': 'extend_model_outputs',
                        'requires_model_changes': True,
                    }
                },
                attempt_spec={'files_to_edit': ['candidate_loss.py']},
                output_code_path=output_code_path,
                override_file_sources={
                    'sandbox_model_adapter.py': fake_adapter,
                    'sandbox_trainer.py': fake_trainer,
                },
                override_tree_sources={'models': fake_models},
            )

            manifest_path = Path(workspace['manifest_path'])
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

            self.assertTrue(output_code_path.exists())
            self.assertEqual(manifest['routing_policy']['recommended_path'], 'extend_model_outputs')
            self.assertTrue(manifest['routing_policy']['requires_model_changes'])
            self.assertEqual(manifest['sandbox_override_dir'], str(output_code_path.parent / 'sandbox_overrides'))
            self.assertTrue((output_code_path.parent / 'sandbox_overrides' / 'sandbox_model_adapter.py').exists())
            self.assertTrue((output_code_path.parent / 'sandbox_overrides' / 'sandbox_trainer.py').exists())
            self.assertTrue((output_code_path.parent / 'sandbox_overrides' / 'models' / '__init__.py').exists())

    def test_load_existing_touched_paths_and_required_path_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt_dir = Path(temp_dir)
            (attempt_dir / 'agent_code_generation_response.json').write_text(
                json.dumps({'touched_paths': ['/tmp/a/models/x.py', '/tmp/a/candidate_loss.py']}),
                encoding='utf-8',
            )
            (attempt_dir / 'agent_code_repair_response.json').write_text(
                json.dumps({'touched_paths': ['/tmp/a/sandbox_trainer.py']}),
                encoding='utf-8',
            )

            touched = load_existing_touched_paths(attempt_dir)
            error = check_required_edit_paths(
                required_edit_paths=['models', 'sandbox_trainer.py'],
                touched_paths=touched,
            )

            self.assertEqual(
                touched,
                ['/tmp/a/models/x.py', '/tmp/a/candidate_loss.py', '/tmp/a/sandbox_trainer.py'],
            )
            self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()
