/**
 * @file orchestrate.ts
 * @description Build task context and optionally execute the new agentic loss loop
 * @author Leizheng
 * @date 2026-03-25
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath, findPythonWithModule } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferOrchestrate = defineTool({
  name: 'ocean_loss_transfer_orchestrate',
  description: '主入口：构建 task_context.json，并执行 analysis_plan.json 驱动的 agentic loss 迁移闭环',

  params: {
    paper_slug: { type: 'string', description: '论文标识符' },
    code_repo_path: { type: 'string', description: '论文代码仓库路径', required: false },
    paper_pdf_path: { type: 'string', description: '论文 PDF 路径', required: false },
    loss_ir_yaml: { type: 'string', description: '已有 Loss IR YAML 路径', required: false },
    analysis_plan_path: { type: 'string', description: 'Agent 生成的 analysis_plan.json 路径', required: false },
    dataset_root: { type: 'string', description: '训练/验证数据根目录', required: false },
    mode: { type: 'string', description: 'context_only 或 agent_loop，默认 agent_loop', required: false },
    bootstrap_formula: { type: 'boolean', description: '无 analysis_plan 时是否尝试公式原生 bootstrap', required: false },
    max_attempts: { type: 'number', description: '最多执行的 attempt 数', required: false },
    auto_generate_plan: { type: 'boolean', description: '是否通过本地 Agent 服务自动生成 analysis_plan.json', required: false },
    service_url: { type: 'string', description: '本地 Agent 服务 URL（可选）', required: false },
    service_api_key: { type: 'string', description: '本地 Agent 服务 API Key（可选）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = (await findPythonWithModule('torch')) ?? await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/run_auto_experiment.py');

    const cmd = [
      python,
      scriptPath,
      `--paper_slug "${shellEscapeDouble(args.paper_slug)}"`,
      args.code_repo_path ? `--code_repo "${shellEscapeDouble(args.code_repo_path)}"` : '',
      args.paper_pdf_path ? `--paper_pdf "${shellEscapeDouble(args.paper_pdf_path)}"` : '',
      args.loss_ir_yaml ? `--loss_ir_yaml "${shellEscapeDouble(args.loss_ir_yaml)}"` : '',
      args.analysis_plan_path ? `--analysis_plan_json "${shellEscapeDouble(args.analysis_plan_path)}"` : '',
      args.dataset_root ? `--dataset_root "${shellEscapeDouble(args.dataset_root)}"` : '',
      args.mode ? `--mode "${shellEscapeDouble(args.mode)}"` : '',
      typeof args.max_attempts === 'number' ? `--max_attempts ${args.max_attempts}` : '',
      args.auto_generate_plan ? '--auto_generate_plan' : '',
      args.service_url ? `--service_url "${shellEscapeDouble(args.service_url)}"` : '',
      args.service_api_key ? `--service_api_key "${shellEscapeDouble(args.service_api_key)}"` : '',
      args.bootstrap_formula === false ? '--no_bootstrap_formula' : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd, {
      timeoutMs: 900_000,
    });

    if (result.code !== 0) {
      throw new Error(`run_auto_experiment 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  }
});
