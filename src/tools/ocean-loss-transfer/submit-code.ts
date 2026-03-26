/**
 * @file submit-code.ts
 * @description Agent 直接提交代码进行验证和训练测试
 * @author Leizheng
 * @date 2026-03-24
 * @version 1.1.0
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath, findPythonWithModule } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

export const oceanLossTransferSubmitCode = defineTool({
  name: 'ocean_loss_transfer_submit_code',
  description: '辅助调试：手工提交一版候选 loss 代码，交给 attempt_executor 做验证和训练测试',

  params: {
    code: { type: 'string', description: 'Agent 编写的完整 sandbox_loss.py 代码', required: true },
    paper_slug: { type: 'string', description: '论文标识符', required: true },
    trial_id: { type: 'number', description: 'Trial 编号（默认 4）', required: false },
    run_training: { type: 'boolean', description: '是否跑 Layer 3/4 训练测试（默认 false，只做 static+smoke）', required: false },
    strategy: { type: 'string', description: '生成策略 faithful/creative', required: false },
    dataset_root: { type: 'string', description: '训练/验证数据根目录（可选）', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = (await findPythonWithModule('torch')) ?? await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;

    const trialId = args.trial_id ?? 4;
    const strategy = args.strategy ?? 'faithful';
    const runTraining = args.run_training ?? false;
    const datasetRoot = args.dataset_root ?? null;

    // 写代码到临时文件
    const tmpFile = path.join(os.tmpdir(), `agent_trial_${trialId}_loss.py`);
    fs.writeFileSync(tmpFile, args.code, 'utf-8');

    const scriptDir = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer');
    const pyCode = `
import sys, json
sys.path.insert(0, '${shellEscapeDouble(scriptDir)}')
from attempt_executor import execute_attempt

attempt_spec = {
    'name': 'Agent-Generated (${strategy})',
    'kind': 'agent_code',
    'code': open('${shellEscapeDouble(tmpFile)}').read(),
    'run_training': ${runTraining ? 'True' : 'False'},
    'notes': 'Submitted through ocean_loss_transfer_submit_code',
}

result = execute_attempt(
    paper_slug='${shellEscapeDouble(args.paper_slug)}',
    attempt_id=${trialId},
    attempt_spec=attempt_spec,
    dataset_root=${datasetRoot ? `'${shellEscapeDouble(datasetRoot)}'` : 'None'},
)
print(json.dumps(result, ensure_ascii=False, default=str))
`;

    const result = await ctx.sandbox.exec(`${python} -c ${JSON.stringify(pyCode)}`, {
      timeoutMs: runTraining ? 900_000 : 120_000,
    });

    try { fs.unlinkSync(tmpFile); } catch {}

    if (result.code !== 0) {
      return {
        status: 'error',
        message: `attempt_executor 执行失败: ${result.stderr?.slice(-500)}`,
      };
    }

    try {
      const parsed = JSON.parse(result.stdout);
      if (runTraining) {
        return parsed;
      }

      const nextStep = parsed.passed
        ? '代码通过 static/smoke/公式对齐检查。如需训练验证，请再次调用并设置 run_training=true。'
        : '代码验证失败，请根据返回的 stop_layer 和 error 修正后重试。';

      return {
        ...parsed,
        next_step: nextStep,
      };
    } catch {
      return {
        status: 'error',
        message: `无法解析结果: ${result.stdout?.slice(0, 500)}`,
      };
    }
  },
});
