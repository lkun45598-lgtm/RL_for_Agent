from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agent_artifact_generator import (  # noqa: E402
    _build_agent_edit_input_files,
    _finalize_candidate_edit_result,
)
from agent_edit_workspace import snapshot_editable_targets  # noqa: E402


class AgentArtifactGeneratorFlowTests(unittest.TestCase):
    def test_build_agent_edit_input_files_preserves_order_and_repair_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_context_file = root / 'task_context.json'
            manifest_path = root / 'editable_files.json'
            candidate_path = root / 'candidate_loss.py'
            override_path = root / 'sandbox_model_adapter.py'
            formula_path = root / 'loss_formula.json'
            analysis_path = root / 'analysis_plan.json'
            for path in [task_context_file, manifest_path, candidate_path, override_path, formula_path, analysis_path]:
                path.write_text('{}\n', encoding='utf-8')

            files = _build_agent_edit_input_files(
                task_context_file=task_context_file,
                editable_manifest_path=manifest_path,
                editable_targets=[
                    {'path': str(candidate_path)},
                    {'path': str(override_path)},
                    {'path': str(root / 'models')},
                ],
                resolved_loss_formula=formula_path,
                resolved_analysis_plan=analysis_path,
                current_code_path=candidate_path,
            )

            self.assertEqual(
                files,
                [
                    str(task_context_file),
                    str(manifest_path),
                    str(candidate_path),
                    str(formula_path),
                    str(analysis_path),
                    str(candidate_path),
                    str(override_path),
                ],
            )

    def test_finalize_candidate_edit_result_records_success_and_touched_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / 'candidate_loss.py'
            manifest_path = root / 'editable_files.json'
            log_path = root / 'agent_code_generation_response.json'
            output_path.write_text('# before\n', encoding='utf-8')
            manifest_path.write_text('{}\n', encoding='utf-8')
            editable_targets = [{'path': str(output_path), 'kind': 'candidate_loss'}]
            before_snapshot = snapshot_editable_targets(editable_targets)
            output_path.write_text('# after\n', encoding='utf-8')

            result = _finalize_candidate_edit_result(
                output_path=output_path,
                response={'status': 'success', 'agent_id': 'agent-1', 'text': 'done'},
                log_path=log_path,
                edit_workspace={
                    'manifest_path': manifest_path,
                    'editable_targets': editable_targets,
                    'sandbox_override_dir': None,
                },
                before_snapshot=before_snapshot,
                attempt_spec={},
                missing_output_error='missing output',
            )

            log_payload = json.loads(log_path.read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['editable_manifest_path'], str(manifest_path))
            self.assertEqual(result['touched_paths'], [str(output_path)])
            self.assertEqual(log_payload['touched_paths'], [str(output_path)])
            self.assertEqual(log_payload['code_chars'], len('# after\n'))

    def test_finalize_candidate_edit_result_repair_accepts_historical_required_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / 'candidate_loss.py'
            manifest_path = root / 'editable_files.json'
            generation_log = root / 'agent_code_generation_response.json'
            repair_log = root / 'agent_code_repair_response.json'
            output_path.write_text('# candidate\n', encoding='utf-8')
            manifest_path.write_text('{}\n', encoding='utf-8')
            generation_log.write_text(
                json.dumps({'touched_paths': [str(root / 'sandbox_overrides' / 'models' / 'SwinIR.py')]}),
                encoding='utf-8',
            )
            editable_targets = [{'path': str(output_path), 'kind': 'candidate_loss'}]
            before_snapshot = snapshot_editable_targets(editable_targets)
            output_path.write_text('# repaired\n', encoding='utf-8')

            result = _finalize_candidate_edit_result(
                output_path=output_path,
                response={'status': 'success', 'agent_id': 'agent-2', 'text': 'repaired'},
                log_path=repair_log,
                edit_workspace={
                    'manifest_path': manifest_path,
                    'editable_targets': editable_targets,
                    'sandbox_override_dir': root / 'sandbox_overrides',
                },
                before_snapshot=before_snapshot,
                attempt_spec={'required_edit_paths': ['models']},
                missing_output_error='missing repaired output',
                failure_feedback={'stop_layer': 'layer3'},
                include_history=True,
            )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(
                result['historical_touched_paths'],
                [str(root / 'sandbox_overrides' / 'models' / 'SwinIR.py')],
            )
            self.assertEqual(result['touched_paths'], [str(output_path)])


if __name__ == '__main__':
    unittest.main()
