from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from runtime_routing import (  # noqa: E402
    FULL_RUN_MODEL_CONFIGS,
    MODEL_OUTPUT_EXTENSION_POLICY,
    build_runtime_routing_feedback,
    collect_model_output_extension_support,
    formula_requires_model_output_extension,
    formula_requires_sandbox_adapter,
    needs_temporary_runtime_config,
)


class RuntimeRoutingTests(unittest.TestCase):
    def test_formula_requires_sandbox_adapter_when_extra_variables_exist(self) -> None:
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'extra_required_variables': ['weight'], 'requires_model_output_extension': False},
        ):
            self.assertTrue(formula_requires_sandbox_adapter({'symbol_map': {}}))
            self.assertFalse(formula_requires_model_output_extension({'symbol_map': {}}))

    def test_formula_requires_model_output_extension_blocks_adapter_path(self) -> None:
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'extra_required_variables': ['feat'], 'requires_model_output_extension': True},
        ):
            self.assertFalse(formula_requires_sandbox_adapter({'symbol_map': {}}))
            self.assertTrue(formula_requires_model_output_extension({'symbol_map': {}}))

    def test_needs_temporary_runtime_config_covers_output_extension_and_dataset_root(self) -> None:
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'extra_required_variables': [], 'requires_model_output_extension': True},
        ):
            self.assertTrue(needs_temporary_runtime_config({'symbol_map': {}}))
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={},
        ):
            self.assertTrue(needs_temporary_runtime_config(None, dataset_root='/tmp/data'))

    def test_collect_model_output_extension_support_uses_lowercase_model_keys(self) -> None:
        touched = []
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'requires_model_output_extension': True},
        ):
            support = collect_model_output_extension_support(
                config_dir=Path('/configs'),
                sandbox_override_dir='/override',
                formula_spec={'symbol_map': {}},
                model_configs={'SwinIR': 'swinir.yaml', 'EDSR': 'edsr.yaml'},
                support_probe=lambda path: touched.append(path.name) or path.name == 'swinir.yaml',
            )

        self.assertEqual(touched, ['swinir.yaml', 'edsr.yaml'])
        self.assertEqual(
            support,
            {
                'swinir': True,
                'edsr': False,
            },
        )

    def test_build_runtime_routing_feedback_returns_policy_and_support_map(self) -> None:
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'requires_model_output_extension': True},
        ):
            feedback = build_runtime_routing_feedback(
                config_dir=Path('/configs'),
                sandbox_override_dir='/override',
                formula_spec={'symbol_map': {}},
                model_configs={'SwinIR': 'swinir.yaml'},
                support_probe=lambda path: path.name == 'swinir.yaml',
            )

        self.assertEqual(
            feedback,
            {
                'requires_model_output_extension': True,
                'current_model_output_extension_support': {'swinir': True},
                'policy': MODEL_OUTPUT_EXTENSION_POLICY,
                'sandbox_override_dir': '/override',
            },
        )

    def test_build_runtime_routing_feedback_returns_none_when_not_needed(self) -> None:
        with patch(
            'runtime_routing.analyze_formula_interface',
            return_value={'requires_model_output_extension': False},
        ):
            feedback = build_runtime_routing_feedback(
                config_dir=Path('/configs'),
                sandbox_override_dir='/override',
                formula_spec={'symbol_map': {}},
                model_configs=FULL_RUN_MODEL_CONFIGS,
                support_probe=lambda _: True,
            )

        self.assertIsNone(feedback)


if __name__ == '__main__':
    unittest.main()
