"""
@file generate_analysis_plan.py
@description CLI wrapper for generating analysis_plan.json via the local agent service
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import argparse
import json

from agent_artifact_generator import generate_analysis_plan as _generate_analysis_plan


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate analysis_plan.json from task_context.json')
    parser.add_argument('--task_context_path', required=True)
    parser.add_argument('--max_attempts', type=int, default=4)
    parser.add_argument('--service_url', default=None)
    parser.add_argument('--service_api_key', default=None)
    parser.add_argument('--timeout_sec', type=int, default=900)
    args = parser.parse_args()

    result = _generate_analysis_plan(
        args.task_context_path,
        max_attempts=args.max_attempts,
        service_url=args.service_url,
        api_key=args.service_api_key,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
