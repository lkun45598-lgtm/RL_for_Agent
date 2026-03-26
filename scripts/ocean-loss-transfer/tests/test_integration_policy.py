from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from integration_policy import (  # noqa: E402
    build_attempt_edit_policy,
    integration_path_needs_adapter_overrides,
    integration_path_needs_model_tree,
    merge_attempt_with_edit_policy,
    resolve_recommended_integration_path,
)
from agent_edit_workspace import resolve_requested_override_files  # noqa: E402
from agent_repair_loop import _normalize_attempts  # noqa: E402


class IntegrationPolicyTests(unittest.TestCase):
    def test_analysis_plan_path_overrides_task_context(self) -> None:
        task_context = {
            'integration_assessment': {
                'recommended_path': 'loss_only',
            }
        }
        analysis_plan = {
            'integration_decision': {
                'path': 'extend_model_outputs',
            }
        }

        resolved = resolve_recommended_integration_path(task_context, analysis_plan)

        self.assertEqual(resolved, 'extend_model_outputs')

    def test_task_context_path_used_when_analysis_plan_missing(self) -> None:
        task_context = {
            'integration_assessment': {
                'recommended_path': 'Adapter_Wrapper',
            }
        }

        resolved = resolve_recommended_integration_path(task_context)

        self.assertEqual(resolved, 'adapter_wrapper')

    def test_default_policy_for_extend_model_outputs(self) -> None:
        policy = build_attempt_edit_policy('extend_model_outputs')

        self.assertEqual(
            policy['files_to_edit'],
            [
                'candidate_loss.py',
                'sandbox model adapter files exposing extra loss inputs',
                'sandbox trainer files',
                'models',
            ],
        )
        self.assertEqual(policy['required_edit_paths'], ['models'])

    def test_merge_attempt_keeps_existing_required_paths(self) -> None:
        merged = merge_attempt_with_edit_policy(
            {
                'name': 'custom',
                'files_to_edit': ['candidate_loss.py', 'custom_override.py'],
                'required_edit_paths': ['sandbox_trainer.py'],
            },
            integration_path='extend_model_outputs',
        )

        self.assertEqual(
            merged['files_to_edit'],
            [
                'candidate_loss.py',
                'custom_override.py',
                'sandbox model adapter files exposing extra loss inputs',
                'sandbox trainer files',
                'models',
            ],
        )
        self.assertEqual(merged['required_edit_paths'], ['sandbox_trainer.py'])

    def test_adapter_and_model_tree_helpers_match_routing_expectations(self) -> None:
        self.assertTrue(integration_path_needs_adapter_overrides('adapter_wrapper'))
        self.assertTrue(integration_path_needs_adapter_overrides('extend_model_outputs'))
        self.assertFalse(integration_path_needs_adapter_overrides('model_surgery'))

        self.assertFalse(integration_path_needs_model_tree('adapter_wrapper'))
        self.assertTrue(integration_path_needs_model_tree('extend_model_outputs'))
        self.assertTrue(integration_path_needs_model_tree('model_surgery'))

    def test_normalize_attempts_uses_analysis_plan_routing_for_defaults(self) -> None:
        task_context = {
            'integration_assessment': {
                'recommended_path': 'loss_only',
            }
        }
        analysis_plan = {
            'integration_decision': {
                'path': 'extend_model_outputs',
            }
        }

        attempts = _normalize_attempts(
            [{'name': 'Attempt 1', 'kind': 'agent_code'}],
            task_context=task_context,
            analysis_plan=analysis_plan,
        )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['required_edit_paths'], ['models'])
        self.assertIn('models', attempts[0]['files_to_edit'])

    def test_override_resolution_follows_recommended_path_defaults(self) -> None:
        task_context = {
            'integration_assessment': {
                'recommended_path': 'extend_model_outputs',
            }
        }

        resolved = resolve_requested_override_files(
            task_context,
            {'files_to_edit': ['candidate_loss.py']},
        )

        self.assertEqual(
            resolved,
            {
                'files': ['sandbox_model_adapter.py', 'sandbox_trainer.py'],
                'trees': ['models'],
            },
        )


if __name__ == '__main__':
    unittest.main()
