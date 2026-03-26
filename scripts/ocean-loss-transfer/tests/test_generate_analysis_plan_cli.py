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

import generate_analysis_plan as generate_analysis_plan_cli  # noqa: E402


class GenerateAnalysisPlanCliTests(unittest.TestCase):
    def test_main_passes_explicit_args_to_generator(self) -> None:
        fake_result = {
            'status': 'success',
            'analysis_plan_path': '/tmp/analysis_plan.json',
        }

        with patch.object(
            sys,
            'argv',
            [
                'generate_analysis_plan.py',
                '--task_context_path',
                '/tmp/task_context.json',
                '--max_attempts',
                '3',
                '--service_url',
                'http://agent.local',
                '--service_api_key',
                'secret',
                '--timeout_sec',
                '120',
            ],
        ), patch(
            'generate_analysis_plan._generate_analysis_plan',
            return_value=fake_result,
        ) as mock_generate, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            generate_analysis_plan_cli.main()

        mock_generate.assert_called_once_with(
            '/tmp/task_context.json',
            max_attempts=3,
            service_url='http://agent.local',
            api_key='secret',
            timeout_sec=120,
        )
        self.assertEqual(json.loads(stdout.getvalue()), fake_result)

    def test_main_uses_defaults_when_optional_args_omitted(self) -> None:
        fake_result = {'status': 'error', 'error': 'missing service'}

        with patch.object(
            sys,
            'argv',
            [
                'generate_analysis_plan.py',
                '--task_context_path',
                '/tmp/task_context.json',
            ],
        ), patch(
            'generate_analysis_plan._generate_analysis_plan',
            return_value=fake_result,
        ) as mock_generate, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            generate_analysis_plan_cli.main()

        mock_generate.assert_called_once_with(
            '/tmp/task_context.json',
            max_attempts=4,
            service_url=None,
            api_key=None,
            timeout_sec=900,
        )
        self.assertEqual(json.loads(stdout.getvalue()), fake_result)


if __name__ == '__main__':
    unittest.main()
