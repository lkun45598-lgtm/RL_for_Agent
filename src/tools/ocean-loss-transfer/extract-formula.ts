/**
 * @file extract-formula.ts
 * @description 从论文 PDF + 代码仓库自动起草 Loss Formula Spec（draft）
 * @author Leizheng
 * @date 2026-03-24
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-24 Leizheng: v1.0.0 initial version
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath, findPythonWithModule } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferExtractFormula = defineTool({
  name: 'ocean_loss_transfer_extract_formula',
  description: '从论文 PDF + 代码仓库自动起草 Loss Formula Spec（LaTeX/params/symbol_map draft）',

  params: {
    code_repo_path: { type: 'string', description: '代码仓库路径', required: true },
    paper_slug: { type: 'string', description: '论文标识符', required: true },
    paper_pdf_path: { type: 'string', description: '论文 PDF 路径（可选）', required: false },
    output_path: { type: 'string', description: '输出 JSON 路径（可选）', required: false },
  },

  exec: async (args, ctx) => {
    const pythonPath = args.paper_pdf_path
      ? (await findPythonWithModule('fitz')) ?? (await findFirstPythonPath())
      : await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/extract_loss_formula.py');
    const cmd = [
      `"${shellEscapeDouble(pythonPath)}"`,
      scriptPath,
      `--code_repo "${shellEscapeDouble(args.code_repo_path)}"`,
      `--paper_slug "${shellEscapeDouble(args.paper_slug)}"`,
      args.paper_pdf_path ? `--paper_pdf "${shellEscapeDouble(args.paper_pdf_path)}"` : '',
      args.output_path ? `--output_path "${shellEscapeDouble(args.output_path)}"` : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd);
    if (result.code !== 0) {
      throw new Error(`extract_loss_formula 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  },
});

