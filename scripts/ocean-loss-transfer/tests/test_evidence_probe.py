from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.agent.evidence_probe import (  # noqa: E402
    execute_evidence_probe,
    validate_evidence_probe_request,
)


class EvidenceProbeTests(unittest.TestCase):
    def test_validate_evidence_probe_request_accepts_probe_needed(self) -> None:
        validation = validate_evidence_probe_request(
            {
                'status': 'probe_needed',
                'reason': 'Need to confirm whether model.forward already emits loss_inputs.',
                'evidence_refs': ['code_analysis.focus_files[0]'],
                'probe_goal': 'Inspect forward returns and loss_inputs usage.',
                'expected_output_keys': ['model_files_with_loss_inputs'],
            }
        )

        self.assertEqual(validation['status'], 'ok')
        self.assertEqual(validation['normalized_request']['status'], 'probe_needed')

    def test_execute_evidence_probe_runs_script_and_reads_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / 'repo'
            repo.mkdir(parents=True, exist_ok=True)
            (repo / 'demo.py').write_text('def demo():\n    return 1\n', encoding='utf-8')

            task_context_path = root / 'task_context.json'
            task_context_path.write_text(json.dumps({'paper_slug': 'demo-paper'}), encoding='utf-8')
            script_path = root / 'analysis_evidence_probe.py'
            output_path = root / 'analysis_evidence_probe_result.json'
            script_path.write_text(
                'import argparse\n'
                'import json\n'
                'from pathlib import Path\n'
                'parser = argparse.ArgumentParser()\n'
                'parser.add_argument("--code_repo", required=True)\n'
                'parser.add_argument("--task_context", required=True)\n'
                'parser.add_argument("--output", required=True)\n'
                'args = parser.parse_args()\n'
                'repo = Path(args.code_repo)\n'
                'payload = {"python_files": sorted(str(p.relative_to(repo)) for p in repo.rglob("*.py"))}\n'
                'Path(args.output).write_text(json.dumps(payload), encoding="utf-8")\n',
                encoding='utf-8',
            )

            result = execute_evidence_probe(
                script_path=script_path,
                code_repo_path=str(repo),
                task_context_path=task_context_path,
                output_path=output_path,
            )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['result']['python_files'], ['demo.py'])
            self.assertTrue(output_path.exists())

    def test_execute_evidence_probe_passes_extra_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / 'repo'
            repo.mkdir(parents=True, exist_ok=True)

            task_context_path = root / 'task_context.json'
            task_context_path.write_text(json.dumps({'paper_slug': 'demo-paper'}), encoding='utf-8')
            latest_result_path = root / 'result.json'
            latest_result_path.write_text(json.dumps({'stop_layer': 'layer4'}), encoding='utf-8')
            script_path = root / 'analysis_evidence_probe.py'
            output_path = root / 'analysis_evidence_probe_result.json'
            script_path.write_text(
                'import argparse\n'
                'import json\n'
                'from pathlib import Path\n'
                'parser = argparse.ArgumentParser()\n'
                'parser.add_argument("--code_repo", required=True)\n'
                'parser.add_argument("--task_context", required=True)\n'
                'parser.add_argument("--latest_result")\n'
                'parser.add_argument("--output", required=True)\n'
                'args = parser.parse_args()\n'
                'payload = {"latest_result": None}\n'
                'if args.latest_result:\n'
                '    payload["latest_result"] = json.loads(Path(args.latest_result).read_text(encoding="utf-8"))\n'
                'Path(args.output).write_text(json.dumps(payload), encoding="utf-8")\n',
                encoding='utf-8',
            )

            result = execute_evidence_probe(
                script_path=script_path,
                code_repo_path=str(repo),
                task_context_path=task_context_path,
                output_path=output_path,
                extra_args=['--latest_result', str(latest_result_path)],
            )

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['result']['latest_result']['stop_layer'], 'layer4')


if __name__ == '__main__':
    unittest.main()
