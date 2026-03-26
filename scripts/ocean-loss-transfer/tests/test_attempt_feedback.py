from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from attempt_feedback import (  # noqa: E402
    build_failure_feedback,
    compute_baseline_delta,
    extract_primary_metric,
    summarize_repair_rounds,
    validation_error_text,
)


class AttemptFeedbackTests(unittest.TestCase):
    def test_extract_primary_metric_prefers_swinir(self) -> None:
        metric_name, metric_value = extract_primary_metric({'val_ssim': 0.7, 'swinir': 0.8})

        self.assertEqual(metric_name, 'swinir')
        self.assertEqual(metric_value, 0.8)

    def test_validation_error_text_handles_formula_alignment(self) -> None:
        error_text = validation_error_text(
            'formula_alignment',
            {'formula_alignment': {'errors': ['missing weight', 'missing log_b']}},
        )

        self.assertEqual(error_text, 'missing weight; missing log_b')

    def test_summarize_repair_rounds_keeps_recent_status_fields(self) -> None:
        summary = summarize_repair_rounds(
            [
                {
                    'round': 1,
                    'status': 'success',
                    'trigger_stop_layer': 'layer2',
                    'post_stop_layer': 'layer3',
                    'post_error': 'timeout',
                    'post_baseline_delta': 0.01,
                    'repair': {'agent_response_path': '/tmp/repair.json'},
                }
            ]
        )

        self.assertEqual(
            summary,
            [
                {
                    'round': 1,
                    'status': 'success',
                    'trigger_stop_layer': 'layer2',
                    'post_stop_layer': 'layer3',
                    'post_error': 'timeout',
                    'post_baseline_delta': 0.01,
                    'reverted': False,
                    'agent_response_path': '/tmp/repair.json',
                }
            ],
        )

    def test_compute_baseline_delta_returns_none_without_metric(self) -> None:
        self.assertIsNone(compute_baseline_delta({}, {'ssim_mean': 0.6}))

    def test_build_failure_feedback_includes_runtime_routing_and_performance_target(self) -> None:
        feedback = build_failure_feedback(
            stop_layer='layer4',
            validation={'layer4': {'detail': 'below threshold'}},
            metrics={'swinir': 0.71},
            baseline={'model': 'swinir', 'ssim_mean': 0.66, 'viable_threshold': 0.68, 'improvement_threshold': 0.69},
            repair_rounds=[{'round': 1, 'status': 'success', 'repair': {'agent_response_path': '/tmp/r.json'}}],
            runtime_routing={'requires_model_output_extension': True},
        )

        self.assertEqual(feedback['error'], 'below threshold')
        self.assertEqual(feedback['runtime_routing'], {'requires_model_output_extension': True})
        self.assertEqual(feedback['performance_target']['primary_metric_name'], 'swinir')
        self.assertEqual(feedback['performance_target']['baseline_delta'], 0.05)
        self.assertEqual(feedback['previous_repair_rounds'][0]['agent_response_path'], '/tmp/r.json')


if __name__ == '__main__':
    unittest.main()
