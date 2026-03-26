/**
 * @file generate-plan.ts
 * @description Generate analysis_plan.json from task_context.json via the local agent service
 * @author OpenAI Codex
 * @date 2026-03-26
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferGeneratePlan = defineTool({
  name: 'ocean_loss_transfer_generate_plan',
  description: '根据 task_context.json 自动生成 analysis_plan.json，供后续 orchestrate 使用',

  params: {
    task_context_path: { type: 'string', description: 'task_context.json 路径', required: true },
    max_attempts: { type: 'number', description: '最多生成的 attempts 数量', required: false },
    service_url: { type: 'string', description: '本地 Agent 服务 URL（可选）', required: false },
    service_api_key: { type: 'string', description: '本地 Agent 服务 API Key（可选）', required: false },
    timeout_sec: { type: 'number', description: 'Agent 调用超时时间（秒）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/generate_analysis_plan.py');

    const cmd = [
      python,
      scriptPath,
      `--task_context_path "${shellEscapeDouble(args.task_context_path)}"`,
      typeof args.max_attempts === 'number' ? `--max_attempts ${args.max_attempts}` : '',
      args.service_url ? `--service_url "${shellEscapeDouble(args.service_url)}"` : '',
      args.service_api_key ? `--service_api_key "${shellEscapeDouble(args.service_api_key)}"` : '',
      typeof args.timeout_sec === 'number' ? `--timeout_sec ${args.timeout_sec}` : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd, {
      timeoutMs: Math.max(60_000, Number(args.timeout_sec ?? 900) * 1000),
    });

    if (result.code !== 0) {
      throw new Error(`generate_analysis_plan 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  }
});
