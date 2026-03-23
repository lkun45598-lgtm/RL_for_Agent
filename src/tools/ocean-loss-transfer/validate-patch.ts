/**
 * @file validate-patch.ts
 * @description 验证 loss 文件的 4 层渐进式检查
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
import path from 'node:path';

export const oceanLossTransferValidate = defineTool({
  name: 'ocean_loss_transfer_validate',
  description: '验证 loss 文件 (4层: static/smoke/single/full)',

  params: {
    loss_file_path: { type: 'string', description: 'Loss 文件路径' },
    mode: {
      type: 'string',
      description: '验证层级',
      enum: ['static', 'smoke', 'single', 'full']
    }
  },

  async exec (args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/validate_loss.py');
    const lossFile = shellEscapeDouble(args.loss_file_path);
    const mode = shellEscapeDouble(args.mode);

    const result = await ctx.sandbox.exec(
      `${python} "${shellEscapeDouble(scriptPath)}" --loss_file "${lossFile}" --mode "${mode}"`
    );

    if (result.code !== 0) throw new Error(`validate_loss 失败: ${result.stderr}`);
    return JSON.parse(result.stdout);
  }
});
