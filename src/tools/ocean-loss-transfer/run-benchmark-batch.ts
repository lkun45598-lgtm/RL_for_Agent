/**
 * @file run-benchmark-batch.ts
 * @description Run a batch of benchmark-driven loss-transfer experiments
 * @author OpenAI Codex
 * @date 2026-03-26
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath, findPythonWithModule } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferRunBenchmarkBatch = defineTool({
  name: 'ocean_loss_transfer_run_benchmark_batch',
  description: '批量执行 benchmark loss-transfer 闭环，复用 catalog/materialize/context/plan/agent loop，输出统一 summary',

  params: {
    benchmark_root: { type: 'string', description: 'Benchmark 根目录（可选）', required: false },
    catalog_path: { type: 'string', description: '已有 benchmark catalog JSON 路径（可选）', required: false },
    entry_ids: {
      type: 'array',
      items: { type: 'string' },
      description: '只运行这些 entry_id（可选）',
      required: false,
    },
    paper_slugs: {
      type: 'array',
      items: { type: 'string' },
      description: '只运行这些 paper_slug（可选）',
      required: false,
    },
    categories: {
      type: 'array',
      items: { type: 'string' },
      description: '只运行这些 category（可选）',
      required: false,
    },
    limit: { type: 'number', description: '最多运行多少个 ready 条目', required: false },
    output_root: { type: 'string', description: '批量输出目录根路径（可选）', required: false },
    run_id: { type: 'string', description: '本次 batch run 的标识符（可选）', required: false },
    cache_root: { type: 'string', description: 'archive 解压缓存目录（可选）', required: false },
    dataset_root: { type: 'string', description: '训练/验证数据根目录（可选）', required: false },
    mode: {
      type: 'string',
      description: 'context_only / plan_only / agent_loop，默认 context_only',
      required: false,
    },
    max_attempts: { type: 'number', description: '每个条目最多执行的 attempts 数', required: false },
    auto_generate_plan: { type: 'boolean', description: '是否自动调用 Agent 生成 analysis plan', required: false },
    bootstrap_formula: { type: 'boolean', description: '无 analysis plan 时是否启用 bootstrap formula', required: false },
    service_url: { type: 'string', description: 'Agent 服务 URL（可选）', required: false },
    service_api_key: { type: 'string', description: 'Agent 服务 API Key（可选）', required: false },
    timeout_sec: { type: 'number', description: 'analysis plan 生成超时时间（秒）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = (await findPythonWithModule('torch')) ?? await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/run_benchmark_batch.py');

    const repeatedFlags = (flag: string, values?: unknown) => {
      if (!Array.isArray(values)) return [];
      return values
        .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
        .map((value) => `${flag} "${shellEscapeDouble(value)}"`);
    };

    const cmd = [
      python,
      scriptPath,
      args.benchmark_root ? `--benchmark_root "${shellEscapeDouble(args.benchmark_root)}"` : '',
      args.catalog_path ? `--catalog_path "${shellEscapeDouble(args.catalog_path)}"` : '',
      ...repeatedFlags('--entry_id', args.entry_ids),
      ...repeatedFlags('--paper_slug', args.paper_slugs),
      ...repeatedFlags('--category', args.categories),
      typeof args.limit === 'number' ? `--limit ${args.limit}` : '',
      args.output_root ? `--output_root "${shellEscapeDouble(args.output_root)}"` : '',
      args.run_id ? `--run_id "${shellEscapeDouble(args.run_id)}"` : '',
      args.cache_root ? `--cache_root "${shellEscapeDouble(args.cache_root)}"` : '',
      args.dataset_root ? `--dataset_root "${shellEscapeDouble(args.dataset_root)}"` : '',
      args.mode ? `--mode "${shellEscapeDouble(args.mode)}"` : '',
      typeof args.max_attempts === 'number' ? `--max_attempts ${args.max_attempts}` : '',
      args.auto_generate_plan ? '--auto_generate_plan' : '',
      args.bootstrap_formula === false ? '--no_bootstrap_formula' : '',
      args.service_url ? `--service_url "${shellEscapeDouble(args.service_url)}"` : '',
      args.service_api_key ? `--service_api_key "${shellEscapeDouble(args.service_api_key)}"` : '',
      typeof args.timeout_sec === 'number' ? `--timeout_sec ${args.timeout_sec}` : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd, {
      timeoutMs: 7_200_000,
    });

    if (result.code !== 0) {
      throw new Error(`run_benchmark_batch 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  }
});
