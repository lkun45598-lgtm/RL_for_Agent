/**
 * @file prepare-context.ts
 * @description 准备 Loss IR 提取的上下文材料
 * @author Leizheng
 * @date 2026-03-23
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-23 Leizheng: v1.0.0 初始版本
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferPrepareContext = defineTool({
  name: 'ocean_loss_transfer_prepare_context',
  description: '扫描代码仓库，准备 Loss IR 提取的上下文材料（返回文件列表和内容供 Agent 分析）',

  params: {
    code_repo_path: { type: 'string', description: '代码仓库路径', required: true },
    paper_slug: { type: 'string', description: '论文标识符', required: true },
    output_dir: { type: 'string', description: '输出目录（可选）', required: false }
  },

  exec: async (args, ctx) => {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/prepare_context.py');

    const cmd = [
      `"${shellEscapeDouble(pythonPath)}"`,
      scriptPath,
      `--code_repo "${shellEscapeDouble(args.code_repo_path)}"`,
      `--paper_slug "${shellEscapeDouble(args.paper_slug)}"`,
      args.output_dir ? `--output_dir "${shellEscapeDouble(args.output_dir)}"` : ''
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd);

    if (result.code !== 0) {
      throw new Error(`prepare_context 失败: ${result.stderr}`);
    }

    const context = JSON.parse(result.stdout);

    return {
      ...context,
      analysis_guide: `
请按以下步骤分析代码并生成 Loss IR YAML：

1. **识别主 loss 函数**
   - 查看 primary_files 中优先级最高的文件
   - 找到被训练循环调用的 loss 函数

2. **分解 loss 组件**
   - 像素级 loss: L1/L2/Charbonnier (torch.abs, **2, sqrt)
   - 梯度 loss: Sobel/Laplacian (conv2d with kernel)
   - 频域 loss: FFT/DCT (torch.fft.rfft2, torch.fft.fft2)
   - 感知 loss: VGG features (需要预训练网络)

3. **提取关键参数**
   - reduction: .mean() / .sum()
   - mask_handling: * mask / mask.bool()
   - normalization: / target.norm() (relative)
   - epsilon/clamp: + eps / .clamp(min=eps)

4. **检查不兼容特征**
   - 需要模型中间层？→ requires_model_internals=true
   - 需要预训练网络？→ requires_pretrained_network=true
   - 需要对抗训练？→ requires_adversarial=true

5. **填充 Loss IR schema**
   - 参考 schema 定义填写所有字段
   - 如有不确定的地方，在 YAML 中加注释

完成后调用 ocean_loss_transfer_write_ir 写入文件。
`
    };
  }
});
