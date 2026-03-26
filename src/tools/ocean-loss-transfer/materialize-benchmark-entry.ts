/**
 * @file materialize-benchmark-entry.ts
 * @description Materialize one benchmark catalog entry into a ready-to-run paper/code bundle
 * @author OpenAI Codex
 * @date 2026-03-26
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferMaterializeBenchmarkEntry = defineTool({
  name: 'ocean_loss_transfer_materialize_benchmark_entry',
  description: '将一个 benchmark catalog 条目解析成可直接运行的论文 PDF 路径和代码仓库路径，必要时自动解压 archive',

  params: {
    benchmark_root: { type: 'string', description: 'Benchmark 根目录（可选）', required: false },
    catalog_path: { type: 'string', description: 'catalog JSON 路径（可选）', required: false },
    entry_id: { type: 'string', description: 'benchmark entry_id（推荐）', required: false },
    paper_slug: { type: 'string', description: 'benchmark paper_slug（可选）', required: false },
    cache_root: { type: 'string', description: '解压缓存目录（可选）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/materialize_benchmark_entry.py');

    const cmd = [
      python,
      scriptPath,
      args.benchmark_root ? `--benchmark_root "${shellEscapeDouble(args.benchmark_root)}"` : '',
      args.catalog_path ? `--catalog_path "${shellEscapeDouble(args.catalog_path)}"` : '',
      args.entry_id ? `--entry_id "${shellEscapeDouble(args.entry_id)}"` : '',
      args.paper_slug ? `--paper_slug "${shellEscapeDouble(args.paper_slug)}"` : '',
      args.cache_root ? `--cache_root "${shellEscapeDouble(args.cache_root)}"` : '',
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd, {
      timeoutMs: 180_000,
    });

    if (result.code !== 0) {
      throw new Error(`materialize_benchmark_entry 失败: ${result.stderr}`);
    }

    return JSON.parse(result.stdout);
  }
});
