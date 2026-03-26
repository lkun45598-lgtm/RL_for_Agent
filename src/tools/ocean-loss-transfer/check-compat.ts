/**
 * @file check-compat.ts
 * @description 检查 Loss IR 兼容性
 * @author kongzhiquan
 * @date 2026-03-22
 *
 * @changelog
 *   - 2026-03-23 kongzhiquan: 修复 defineTool 参数格式：parameters+Zod → params 简洁对象，execute → exec；
 *                           修复 ctx.sandbox.exec 调用为字符串形式
 *   - 2026-03-23 kongzhiquan: 修复 sys.path 使用相对路径导致 ModuleNotFoundError 的问题，改为绝对路径
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferCheckCompat = defineTool({
  name: 'ocean_loss_transfer_check_compat',
  description: '可选参考：检查 Loss IR 与目标接口的兼容性，不替代 analysis_plan 决策',

  params: {
    loss_ir_yaml: { type: 'string', description: 'Loss IR YAML 文件路径' }
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const yamlArg = shellEscapeDouble(args.loss_ir_yaml);
    const scriptsDir = shellEscapeDouble(path.resolve(process.cwd(), 'scripts/ocean-loss-transfer'));

    const pyCode = [
      `import sys, yaml, json; sys.path.insert(0, "${scriptsDir}")`,
      'from loss_transfer.ir.check_compatibility import check_compatibility',
      'from loss_transfer.ir.loss_ir_schema import LossIR',
      `data = yaml.safe_load(open("${yamlArg}"))`,
      'loss_ir = LossIR(**data)',
      'result = check_compatibility(loss_ir)',
      'print(json.dumps(result))'
    ].join('; ');

    const result = await ctx.sandbox.exec(`${python} -c "${shellEscapeDouble(pyCode)}"`);

    if (result.code !== 0) throw new Error(`check_compatibility 失败: ${result.stderr}`);
    return JSON.parse(result.stdout);
  }
});
