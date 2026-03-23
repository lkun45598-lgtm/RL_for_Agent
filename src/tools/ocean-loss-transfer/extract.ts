/**
 * @file extract.ts
 * @description 从论文+代码提取 Loss IR
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

export const oceanLossTransferExtract = defineTool({
  name: 'ocean_loss_transfer_extract',
  description: '从论文 PDF 和代码仓库提取 Loss IR',

  params: {
    paper_pdf_path: { type: 'string', description: '论文 PDF 路径', required: false },
    code_repo_path: { type: 'string', description: '代码仓库路径', required: false },
    output_yaml: { type: 'string', description: '输出 YAML 路径' },
    manual_mode: { type: 'boolean', description: '手动模式(生成模板)', required: false }
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptsDir = shellEscapeDouble(path.resolve(process.cwd(), 'scripts/ocean-loss-transfer'));

    const paperArg = args.paper_pdf_path ? `'${shellEscapeDouble(args.paper_pdf_path)}'` : 'None';
    const repoArg = args.code_repo_path ? `'${shellEscapeDouble(args.code_repo_path)}'` : 'None';
    const outputArg = shellEscapeDouble(args.output_yaml);
    const manualArg = args.manual_mode ? 'True' : 'False';

    const pyCode = [
      `import sys; sys.path.insert(0, "${scriptsDir}")`,
      'from extract_loss_ir import extract_loss_ir',
      `output = extract_loss_ir(paper_pdf_path=${paperArg}, code_repo_path=${repoArg}, output_yaml_path="${outputArg}", manual_mode=${manualArg})`,
      'print(output)'
    ].join('; ');

    const result = await ctx.sandbox.exec(`${python} -c "${shellEscapeDouble(pyCode)}"`);

    if (result.code !== 0) throw new Error(`extract_loss_ir 失败: ${result.stderr}`);
    return { output_file: result.stdout.trim() };
  }
});
