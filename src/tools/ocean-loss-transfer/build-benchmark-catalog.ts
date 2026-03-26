/**
 * @file build-benchmark-catalog.ts
 * @description Scan a benchmark root and build a normalized benchmark catalog
 * @author OpenAI Codex
 * @date 2026-03-26
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferBuildBenchmarkCatalog = defineTool({
  name: 'ocean_loss_transfer_build_benchmark_catalog',
  description: '扫描 Benchmark 目录并生成规范化 catalog，整理论文 PDF、代码候选和建议 slug',

  params: {
    benchmark_root: { type: 'string', description: 'Benchmark 根目录', required: false },
    output_path: { type: 'string', description: '输出 catalog JSON 路径（可选）', required: false },
    max_depth: { type: 'number', description: '扫描目录深度（默认 2）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/build_benchmark_catalog.py');

    const cmd = [
      python,
      scriptPath,
      args.benchmark_root ? `--benchmark_root "${shellEscapeDouble(args.benchmark_root)}"` : '',
      args.output_path ? `--output_path "${shellEscapeDouble(args.output_path)}"` : '',
      typeof args.max_depth === 'number' ? `--max_depth ${args.max_depth}` : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd, {
      timeoutMs: 120_000,
    });

    if (result.code !== 0) {
      throw new Error(`build_benchmark_catalog 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  }
});
