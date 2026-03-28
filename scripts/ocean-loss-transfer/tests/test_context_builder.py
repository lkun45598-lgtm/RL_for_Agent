from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.context.context_builder import build_task_context  # noqa: E402


class ContextBuilderTests(unittest.TestCase):
    def test_build_task_context_surfaces_inventory_summary_and_evidence_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / 'repo'
            (repo / 'losses').mkdir(parents=True, exist_ok=True)
            (repo / 'models').mkdir(parents=True, exist_ok=True)
            (repo / 'configs').mkdir(parents=True, exist_ok=True)

            (repo / 'losses' / 'custom_loss.py').write_text(
                'def custom_loss(pred, target):\n'
                '    return (pred - target).abs().mean()\n',
                encoding='utf-8',
            )
            (repo / 'train_loop.py').write_text(
                'def train(model, optimizer, batch):\n'
                '    loss = model(batch).mean()\n'
                '    loss.backward()\n',
                encoding='utf-8',
            )
            (repo / 'models' / 'demo_model.py').write_text(
                'import torch.nn as nn\n'
                'class DemoModel(nn.Module):\n'
                '    def forward(self, x):\n'
                '        return {"pred": x, "loss_inputs": {"weight": x}}\n',
                encoding='utf-8',
            )
            (repo / 'configs' / 'train.yaml').write_text(
                'model: demo\nloss:\n  name: l1\noptimizer: adam\n',
                encoding='utf-8',
            )

            with patch(
                'loss_transfer.context.context_builder.extract_loss_formula_draft',
                return_value={'status': 'skipped'},
            ):
                task_context = build_task_context(
                    paper_slug='demo-paper',
                    code_repo_path=str(repo),
                    output_dir=str(root / 'experiment'),
                )

            code_analysis = task_context['code_analysis']
            self.assertTrue(code_analysis['available'])
            self.assertGreaterEqual(code_analysis['inventory_summary']['loss_files_count'], 1)
            self.assertGreaterEqual(code_analysis['inventory_summary']['trainer_files_count'], 1)
            self.assertGreaterEqual(code_analysis['inventory_summary']['model_files_count'], 1)
            self.assertGreaterEqual(code_analysis['inventory_summary']['config_files_count'], 1)
            self.assertTrue(task_context['paths']['evidence_graph_path'])
            self.assertTrue(Path(task_context['paths']['evidence_graph_path']).exists())
            self.assertTrue(code_analysis['evidence_graph']['claims'])
            self.assertTrue(code_analysis['focus_files'])


if __name__ == '__main__':
    unittest.main()
