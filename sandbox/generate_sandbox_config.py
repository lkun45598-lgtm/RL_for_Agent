"""
@file generate_sandbox_config.py

@description 根据模型名生成 sandbox_config.yaml，固定训练参数，只切换模型。
@author Leizheng
@date 2026-03-20
@version 1.0.0

@changelog
  - 2026-03-20 Leizheng: v1.0.0 初始版本
"""

import os
import sys
import yaml
import argparse
import copy

# 各模型的模型参数模板（从 template_configs 提取核心参数）
MODEL_TEMPLATES = {
    'SwinIR': {
        'name': 'SwinIR',
        'in_channels': 2,
        'out_channels': 2,
        'img_size': 32,
        'patch_size': 1,
        'embed_dim': 120,
        'depths': [6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6],
        'window_size': 8,
        'mlp_ratio': 2,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'upscale_factor': 4,
        'img_range': 1.0,
        'upsampler': 'pixelshuffle',
        'resi_connection': '1conv',
        'mean': 0,
        'std': 1,
    },
    'FNO2d': {
        'name': 'FNO2d',
        'modes1': [15, 12, 9, 9, 9],
        'modes2': [15, 12, 9, 9, 9],
        'width': 64,
        'fc_dim': 128,
        'layers': [16, 24, 24, 32, 32],
        'in_dim': 2,
        'out_dim': 2,
        'act': 'gelu',
        'pos_dim': 2,
        'upsample_factor': [4, 4],
    },
    'EDSR': {
        'name': 'EDSR',
        'in_channels': 2,
        'out_channels': 2,
        'hidden_channels': 128,
        'n_res_blocks': 16,
        'upscale_factor': 4,
    },
    'UNet2d': {
        'name': 'UNet2d',
        'in_channels': 2,
        'out_channels': 2,
        'init_features': 64,
        'scale_factor': 4,  # must match data.sample_factor
    },
}

# 固定的数据参数
DATA_TEMPLATE = {
    'name': 'OceanNPY',
    'dataset_root': '/data1/user/lz/SR_data_process/demo8',
    'dyn_vars': ['uo', 'vo'],
    'sample_factor': 4,
    'shape': [1440, 2160],
    'train_batchsize': 2,
    'eval_batchsize': 2,
    'normalize': True,
    'normalizer_type': 'PGN',
    'num_workers': 2,
    'patch_size': 128,
    'model_divisor': 1,
}

# 模型特定的 model_divisor 和 patch_size 调整
MODEL_DATA_OVERRIDES = {
    'SwinIR': {'model_divisor': 8, 'patch_size': 128},
    'FNO2d': {'model_divisor': 1, 'patch_size': 128},
    'EDSR': {'model_divisor': 1, 'patch_size': 128},
    'UNet2d': {'model_divisor': 16, 'patch_size': 128},
}

# 固定的训练参数
TRAIN_TEMPLATE = {
    'random_seed': 42,
    'cuda': True,
    'device': 0,
    'epochs': 15,
    'patience': -1,
    'eval_freq': 5,
    'saving_best': True,
    'load_ckpt': False,
    'saving_ckpt': False,
    'ckpt_freq': 100,
    'ckpt_max': 1,
    'distribute': False,
    'distribute_mode': 'DDP',
    'device_ids': [0],
    'use_amp': False,
}

# 模型特定的训练参数覆盖
MODEL_TRAIN_OVERRIDES = {
    'FNO2d': {'use_amp': False},  # FNO2d 的复数运算不兼容 AMP half precision
}

OPTIMIZE_TEMPLATE = {
    'optimizer': 'AdamW',
    'lr': 0.001,
    'weight_decay': 0.001,
}

SCHEDULE_TEMPLATE = {
    'scheduler': 'StepLR',
    'step_size': 300,
    'gamma': 0.5,
}


def generate_config(model_name: str, dataset_root: str = None, output_path: str = None):
    if model_name not in MODEL_TEMPLATES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_TEMPLATES.keys())}")

    config = {}
    config['model'] = copy.deepcopy(MODEL_TEMPLATES[model_name])

    config['data'] = copy.deepcopy(DATA_TEMPLATE)
    if dataset_root:
        config['data']['dataset_root'] = dataset_root
    # 应用模型特定的 data 覆盖
    if model_name in MODEL_DATA_OVERRIDES:
        config['data'].update(MODEL_DATA_OVERRIDES[model_name])

    config['train'] = copy.deepcopy(TRAIN_TEMPLATE)
    # 应用模型特定的训练参数覆盖
    if model_name in MODEL_TRAIN_OVERRIDES:
        config['train'].update(MODEL_TRAIN_OVERRIDES[model_name])
    config['optimize'] = copy.deepcopy(OPTIMIZE_TEMPLATE)
    config['schedule'] = copy.deepcopy(SCHEDULE_TEMPLATE)

    # log 目录：sandbox/runs/{model_name}_sandbox
    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(sandbox_dir, 'runs', f'{model_name}_sandbox')
    config['log'] = {
        'verbose': True,
        'log': True,
        'log_dir': log_dir,
        'wandb': False,
        'wandb_project': '',
    }

    output_path = output_path or os.path.join(sandbox_dir, 'sandbox_config.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"[generate_sandbox_config] Generated config for {model_name}")
    print(f"  → {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sandbox config')
    parser.add_argument('--model_name', type=str, default='SwinIR',
                        choices=list(MODEL_TEMPLATES.keys()),
                        help='Model name')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Override dataset root path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output config path')
    args = parser.parse_args()
    generate_config(args.model_name, args.dataset_root, args.output)
