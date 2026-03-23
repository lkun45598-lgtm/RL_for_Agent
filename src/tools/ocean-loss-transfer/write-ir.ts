/**
 * @file write-ir.ts
 * @description 验证并写入 Loss IR YAML
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
import * as fs from 'fs';
import * as path from 'path';

export const oceanLossTransferWriteIr = defineTool({
  name: 'ocean_loss_transfer_write_ir',
  description: '验证并写入 Loss IR YAML（包含语法、schema、语义、已知失败模式检查）',

  params: {
    yaml_content: { type: 'string', description: 'Loss IR YAML 内容', required: true },
    output_path: { type: 'string', description: '输出文件路径', required: true },
    validate: { type: 'boolean', description: '是否验证（默认 true）', required: false }
  },

  exec: async (args, ctx) => {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');

    // 将 YAML 内容写入临时文件
    const tmpDir = path.join(process.cwd(), '.tmp');
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true });
    }
    const tmpFile = path.join(tmpDir, `loss_ir_${Date.now()}.yaml`);
    fs.writeFileSync(tmpFile, args.yaml_content, 'utf-8');

    const scriptPath = 'scripts/ocean-loss-transfer/write_loss_ir.py';

    const cmd = [
      `"${shellEscapeDouble(pythonPath)}"`,
      scriptPath,
      `--yaml_content "${shellEscapeDouble(tmpFile)}"`,
      `--output_path "${shellEscapeDouble(args.output_path)}"`,
      args.validate === false ? '--no-validate' : ''
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd);

    // 清理临时文件
    try {
      fs.unlinkSync(tmpFile);
    } catch (e) {
      // 忽略清理错误
    }

    if (result.code !== 0) {
      throw new Error(`write_loss_ir 失败: ${result.stderr}`);
    }

    const validationResult = JSON.parse(result.stdout);

    return validationResult;
  }
});
