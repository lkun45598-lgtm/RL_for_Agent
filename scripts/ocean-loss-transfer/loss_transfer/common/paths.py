from __future__ import annotations

from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = _THIS_FILE.parents[1]
SCRIPT_ROOT = _THIS_FILE.parents[2]
SCRIPTS_ROOT = _THIS_FILE.parents[3]
PROJECT_ROOT = _THIS_FILE.parents[4]
PROJECT_ENV_FILE = PROJECT_ROOT / '.env'
SANDBOX_DIR = PROJECT_ROOT / 'sandbox'
SANDBOX_CONFIG_DIR = SANDBOX_DIR / 'configs'
LOSS_TRANSFER_EXPERIMENTS_DIR = SANDBOX_DIR / 'loss_transfer_experiments'
TRAINING_PIPELINE_DIR = SCRIPTS_ROOT / 'ocean-SR-training-masked'
WORKFLOW_LOSS_TRANSFER_DIR = PROJECT_ROOT / 'workflow' / 'loss_transfer'
KNOWLEDGE_BASE_DIR = WORKFLOW_LOSS_TRANSFER_DIR / 'knowledge_base'
KNOWLEDGE_BASE_MODULES_DIR = KNOWLEDGE_BASE_DIR / 'modules'
