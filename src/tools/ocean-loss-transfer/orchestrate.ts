/**
 * @file orchestrate.ts
 * @description 编排 5-trial 搜索
 * @author kongzhiquan
 * @date 2026-03-22
 *
 * @changelog
 *   - 2026-03-23 kongzhiquan: 修复 defineTool 参数格式：parameters+Zod → params 简洁对象，execute → exec；
 *                           修复 ctx.sandbox.exec 调用为字符串形式
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';

export const oceanLossTransferOrchestrate = defineTool({
  name: 'ocean_loss_transfer_orchestrate',
  description: '编排 5-trial 搜索实验',

  params: {
    loss_ir_yaml: { type: 'string', description: 'Loss IR YAML 路径' },
    paper_slug: { type: 'string', description: '论文标识符' }
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const yamlArg = shellEscapeDouble(args.loss_ir_yaml);
    const slugArg = shellEscapeDouble(args.paper_slug);

    const pyCode = [
      'import sys, yaml; sys.path.insert(0, ".")',
      'from scripts.ocean_loss_transfer.orchestrate_trials import orchestrate_trials',
      'from scripts.ocean_loss_transfer.loss_ir_schema import LossIR',
      `data = yaml.safe_load(open("${yamlArg}"))`,
      'loss_ir = LossIR(**data)',
      `summary = orchestrate_trials(loss_ir, "${slugArg}")`,
      'print(yaml.dump(summary))'
    ].join('; ');

    const result = await ctx.sandbox.exec(`${python} -c "${shellEscapeDouble(pyCode)}"`);

    if (result.code !== 0) throw new Error(`orchestrate_trials 失败: ${result.stderr}`);
    return { summary: result.stdout };
  }
});
