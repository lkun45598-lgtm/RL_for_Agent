/**
 * @file orchestrate.ts
 * @description 编排 5-trial 搜索
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-03-22
 *
 * @changelog
 *   - 2026-03-23 Leizheng: 添加 Loss IR 缺失检查和引导
 *   - 2026-03-23 kongzhiquan: 修复 defineTool 参数格式：parameters+Zod → params 简洁对象，execute → exec；
 *                           修复 ctx.sandbox.exec 调用为字符串形式
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import * as fs from 'fs';

export const oceanLossTransferOrchestrate = defineTool({
  name: 'ocean_loss_transfer_orchestrate',
  description: '编排 5-trial 搜索实验',

  params: {
    loss_ir_yaml: { type: 'string', description: 'Loss IR YAML 路径' },
    paper_slug: { type: 'string', description: '论文标识符' },
    code_repo_path: { type: 'string', description: '代码仓库路径（Loss IR 不存在时需要）', required: false }
  },

  async exec(args, ctx) {
    // 检查 Loss IR 是否存在
    if (!fs.existsSync(args.loss_ir_yaml)) {
      return {
        status: 'missing_loss_ir',
        message: `Loss IR 文件不存在: ${args.loss_ir_yaml}`,
        suggested_workflow: `请先提取 Loss IR：

1. 调用 ocean_loss_transfer_prepare_context 准备代码上下文
2. 分析代码并生成 Loss IR YAML
3. 调用 ocean_loss_transfer_write_ir 写入并验证
4. 再次调用本工具开始实验`,
        quick_start: args.code_repo_path ? {
          tool: 'ocean_loss_transfer_prepare_context',
          params: {
            code_repo_path: args.code_repo_path,
            paper_slug: args.paper_slug
          }
        } : null
      };
    }

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
