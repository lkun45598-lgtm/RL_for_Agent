# Loss Transfer System 使用指南

## 快速开始

### 方式 1: 一键自动实验 (推荐)

```bash
cd /data1/user/lz/RL_for_Agent

# 如果你有论文代码仓库
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --code_repo /path/to/paper/code

# 如果你只想手动填写 Loss IR
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper
# 然后编辑生成的 YAML,再用 --loss_ir_yaml 重新运行
```

### 方式 2: 分步执行

**Step 1: 生成 Loss IR 模板**
```bash
python -c "
import sys
sys.path.insert(0, 'scripts/ocean-loss-transfer')
from extract_loss_ir import extract_loss_ir
extract_loss_ir(manual_mode=True, output_yaml_path='my_loss.yaml')
"
```

**Step 2: 编辑 my_loss.yaml**
填写你的 loss 信息

**Step 3: 运行实验**
```bash
python scripts/ocean-loss-transfer/run_auto_experiment.py \
  --paper_slug my_paper \
  --loss_ir_yaml my_loss.yaml
```

## 输出结果

结果保存在: `sandbox/loss_transfer_experiments/{paper_slug}/`
- `summary.yaml` - 5个trial的汇总
- `trial_1/` - 每个trial的详细结果
