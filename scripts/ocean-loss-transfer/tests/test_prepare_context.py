from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from loss_transfer.context.prepare_context import prepare_context  # noqa: E402


class PrepareContextTests(unittest.TestCase):
    def test_prepare_context_builds_inventory_and_evidence_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / 'repo'
            (repo / 'losses').mkdir(parents=True, exist_ok=True)
            (repo / 'models').mkdir(parents=True, exist_ok=True)
            (repo / 'configs').mkdir(parents=True, exist_ok=True)

            (repo / 'losses' / 'custom_loss.py').write_text(
                'import torch\n'
                'def custom_loss(pred, target):\n'
                '    return (pred - target).abs().mean()\n',
                encoding='utf-8',
            )
            (repo / 'train_engine.py').write_text(
                'def train_epoch(model, batch, optimizer):\n'
                '    loss = model(batch).mean()\n'
                '    loss.backward()\n'
                '    optimizer.step()\n',
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

            output_dir = root / 'experiment'
            result = prepare_context(
                code_repo_path=str(repo),
                paper_slug='demo-paper',
                output_dir=str(output_dir),
            )

            self.assertGreaterEqual(len(result['primary_files']), 1)
            self.assertGreaterEqual(len(result['code_inventory']['loss_files']), 1)
            self.assertGreaterEqual(len(result['code_inventory']['trainer_files']), 1)
            self.assertGreaterEqual(len(result['code_inventory']['model_files']), 1)
            self.assertGreaterEqual(len(result['code_inventory']['config_files']), 1)
            self.assertTrue(Path(result['evidence_graph_path']).exists())

            evidence_graph = json.loads(Path(result['evidence_graph_path']).read_text(encoding='utf-8'))
            claim_ids = {claim['claim_id'] for claim in evidence_graph['claims']}
            self.assertIn('loss_files_present', claim_ids)
            self.assertIn('trainer_loss_flow_present', claim_ids)
            self.assertIn('model_aux_loss_inputs_present', claim_ids)
            self.assertIn('config_training_controls_present', claim_ids)


if __name__ == '__main__':
    unittest.main()
