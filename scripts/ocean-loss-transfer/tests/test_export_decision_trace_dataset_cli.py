from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import export_decision_trace_dataset as export_cli  # noqa: E402


class ExportDecisionTraceDatasetCliTests(unittest.TestCase):
    def test_main_passes_explicit_args_to_exporter(self) -> None:
        fake_result = {
            'decision_trace_path': '/tmp/decision_trace.jsonl',
            'decision_trace_count': 2,
            'rl_dataset_path': '/tmp/custom_rl.jsonl',
            'rl_dataset_count': 2,
        }

        with patch.object(
            sys,
            'argv',
            [
                'export_decision_trace_dataset.py',
                '--decision_trace_path',
                '/tmp/decision_trace.jsonl',
                '--output_path',
                '/tmp/custom_rl.jsonl',
            ],
        ), patch(
            'export_decision_trace_dataset.export_rl_dataset_from_decision_trace',
            return_value=fake_result,
        ) as mock_export, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            export_cli.main()

        mock_export.assert_called_once_with(
            Path('/tmp/decision_trace.jsonl'),
            output_path=Path('/tmp/custom_rl.jsonl'),
        )
        self.assertEqual(json.loads(stdout.getvalue()), fake_result)

    def test_main_uses_default_output_when_optional_arg_omitted(self) -> None:
        fake_result = {
            'decision_trace_path': '/tmp/decision_trace.jsonl',
            'decision_trace_count': 2,
            'rl_dataset_path': '/tmp/rl_decision_dataset.jsonl',
            'rl_dataset_count': 2,
        }

        with patch.object(
            sys,
            'argv',
            [
                'export_decision_trace_dataset.py',
                '--decision_trace_path',
                '/tmp/decision_trace.jsonl',
            ],
        ), patch(
            'export_decision_trace_dataset.export_rl_dataset_from_decision_trace',
            return_value=fake_result,
        ) as mock_export, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            export_cli.main()

        mock_export.assert_called_once_with(
            Path('/tmp/decision_trace.jsonl'),
            output_path=None,
        )
        self.assertEqual(json.loads(stdout.getvalue()), fake_result)


if __name__ == '__main__':
    unittest.main()
