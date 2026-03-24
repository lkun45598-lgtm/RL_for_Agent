/**
 * @file write-formula.ts
 * @description 验证并写入 Loss Formula Spec（LaTeX + params JSON + symbol↔variable 双射映射）
 * @author Leizheng
 * @date 2026-03-24
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-24 Leizheng: v1.0.0 初始版本
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import fs from 'node:fs';
import path from 'node:path';

export const oceanLossTransferWriteFormula = defineTool({
  name: 'ocean_loss_transfer_write_formula',
  description: '验证并写入 Loss Formula Spec（LaTeX + params JSON + symbol↔variable 一一映射）',

  params: {
    formula_json: { type: 'string', description: 'Formula Spec JSON 内容（字符串）', required: true },
    output_path: { type: 'string', description: '输出 JSON 文件路径', required: true },
    validate: { type: 'boolean', description: '是否验证（默认 true）', required: false },
  },

  exec: async (args, ctx) => {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');

    // 将 JSON 内容写入临时文件，避免命令行转义问题
    const tmpDir = path.join(process.cwd(), '.tmp');
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
    const tmpFile = path.join(tmpDir, `loss_formula_${Date.now()}.json`);
    fs.writeFileSync(tmpFile, args.formula_json, 'utf-8');

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/write_loss_formula.py');
    const cmd = [
      `"${shellEscapeDouble(pythonPath)}"`,
      scriptPath,
      `--formula_json "${shellEscapeDouble(tmpFile)}"`,
      `--output_path "${shellEscapeDouble(args.output_path)}"`,
      args.validate === false ? '--no-validate' : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd);

    try { fs.unlinkSync(tmpFile); } catch {}

    if (result.code !== 0) {
      throw new Error(`write_loss_formula 失败: ${result.stderr}`);
    }

    try {
      return JSON.parse(result.stdout);
    } catch {
      return { status: 'error', message: `无法解析输出: ${result.stdout?.slice(0, 500)}` };
    }
  },
});

