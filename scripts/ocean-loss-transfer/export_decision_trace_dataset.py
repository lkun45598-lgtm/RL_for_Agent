"""
@file export_decision_trace_dataset.py
@description Convert decision_trace.jsonl into rl_decision_dataset.jsonl
@author OpenAI Codex
@date 2026-03-26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loss_transfer.common.decision_trace import export_rl_dataset_from_decision_trace


def main() -> None:
    parser = argparse.ArgumentParser(description='Export RL dataset records from decision_trace.jsonl')
    parser.add_argument('--decision_trace_path', required=True)
    parser.add_argument('--output_path', default=None)
    args = parser.parse_args()

    result = export_rl_dataset_from_decision_trace(
        Path(args.decision_trace_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
