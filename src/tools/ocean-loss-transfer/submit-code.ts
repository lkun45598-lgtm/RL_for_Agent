/**
 * @file submit-code.ts
 * @description Agent 直接提交代码进行验证和训练测试
 * @author Leizheng
 * @date 2026-03-24
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-24 Leizheng: v1.0.0 initial version - Agent-Native code submission
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

export const oceanLossTransferSubmitCode = defineTool({
  name: 'ocean_loss_transfer_submit_code',
  description: 'Agent 直接提交 sandbox_loss.py 代码进行验证（static + smoke + 可选训练测试）',

  params: {
    code: { type: 'string', description: 'Agent 编写的完整 sandbox_loss.py 代码', required: true },
    paper_slug: { type: 'string', description: '论文标识符', required: true },
    trial_id: { type: 'number', description: 'Trial 编号（默认 4）', required: false },
    run_training: { type: 'boolean', description: '是否跑 Layer 3/4 训练测试（默认 false，只做 static+smoke）', required: false },
    strategy: { type: 'string', description: '生成策略 faithful/creative', required: false },
  },

  async exec(args, ctx) {
    const pythonPath = await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');
    const python = `"${shellEscapeDouble(pythonPath)}"`;

    const trialId = args.trial_id ?? 4;
    const strategy = args.strategy ?? 'faithful';
    const runTraining = args.run_training ?? false;

    // 写代码到临时文件
    const tmpFile = path.join(os.tmpdir(), `agent_trial_${trialId}_loss.py`);
    fs.writeFileSync(tmpFile, args.code, 'utf-8');

    const formulaSpecPath = path.resolve(
      process.cwd(),
      `sandbox/loss_transfer_experiments/${args.paper_slug}/loss_formula.json`,
    );

    if (!runTraining) {
      // 快速模式：只做 static + smoke 验证
      const pyCode = `
import sys, json, os
sys.path.insert(0, '${shellEscapeDouble(path.resolve(process.cwd(), 'scripts/ocean-loss-transfer'))}')
from llm_code_generator import generate_loss_code
formula_path = '${shellEscapeDouble(formulaSpecPath)}'
formula_spec = json.load(open(formula_path)) if os.path.exists(formula_path) else None
result = generate_loss_code(
    loss_ir={},
    code_snippets=[],
    strategy='${strategy}',
    code=open('${shellEscapeDouble(tmpFile)}').read(),
    formula_spec=formula_spec,
)
print(json.dumps(result, ensure_ascii=False))
`;
      const result = await ctx.sandbox.exec(`${python} -c ${JSON.stringify(pyCode)}`);

      // 清理临时文件
      try { fs.unlinkSync(tmpFile); } catch {}

      if (result.code !== 0) {
        return {
          status: 'error',
          message: `验证脚本执行失败: ${result.stderr?.slice(-500)}`,
        };
      }

      try {
        const parsed = JSON.parse(result.stdout);
        let formulaInterfaceIssue: string | null = null;
        if (fs.existsSync(formulaSpecPath)) {
          try {
            const formulaSpec = JSON.parse(fs.readFileSync(formulaSpecPath, 'utf-8'));
            const interfaceAnalysis = formulaSpec?.interface_analysis;
            const extraVars = Array.isArray(interfaceAnalysis?.extra_required_variables)
              ? interfaceAnalysis.extra_required_variables
              : Object.entries(formulaSpec?.symbol_map ?? {})
                .map(([, value]) => value)
                .filter((value): value is string =>
                  typeof value === 'string'
                  && !['pred', 'target', 'mask'].includes(value)
                  && !(value in (formulaSpec?.params ?? {})))
                );
            const interfaceStatus = interfaceAnalysis?.status ?? (extraVars.length > 0 ? 'incompatible' : 'fully_compatible');
            if (interfaceStatus === 'incompatible' && extraVars.length > 0) {
              formulaInterfaceIssue = `当前 loss-only 流程不会自动生成这些论文 loss 变量对应的模型输出: ${extraVars.join(', ')}`;
            }
          } catch {}
        }
        const formulaFailed = parsed.passed_formula_alignment === false;
        const status = (parsed.passed_smoke && !formulaFailed && !formulaInterfaceIssue) ? 'passed' : 'failed';
        const nextStep = parsed.passed_smoke
          ? (formulaFailed
            ? `代码通过 static+smoke，但未通过公式对齐校验（symbol_map/params 与代码不一致）。请修正 loss_formula.json 或修正代码后重试。`
            : (formulaInterfaceIssue
              ? `代码通过 static+smoke，但公式依赖额外模型输出。需要先让模型返回 {"pred": ..., "loss_inputs": {...}}，再跑训练测试。`
              : `代码通过 static + smoke 验证。如需跑训练测试，请重新调用并设置 run_training=true`))
          : `代码验证失败，请根据错误信息修改后重新提交`;

        return {
          status,
          passed_static: parsed.passed_static,
          passed_smoke: parsed.passed_smoke,
          error: parsed.error,
          formula_interface_issue: formulaInterfaceIssue,
          formula_alignment: formulaFailed ? {
            passed: false,
            errors: parsed.formula_alignment_error ? [parsed.formula_alignment_error] : [],
            warnings: parsed.formula_alignment_warnings ?? [],
          } : null,
          next_step: nextStep,
        };
      } catch {
        return {
          status: 'error',
          message: `无法解析验证结果: ${result.stdout?.slice(0, 200)}`,
        };
      }
    } else {
      // 完整模式：通过 run_trial 走 4 层验证 + 训练；run_trial 会自动加载可选 loss_formula.json
      const scriptDir = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer');
      const lossIrPath = path.resolve(
        process.cwd(),
        `sandbox/loss_transfer_experiments/${args.paper_slug}/loss_ir.yaml`
      );

      const pyCode = `
import sys, json
sys.path.insert(0, '${shellEscapeDouble(scriptDir)}')
sys.path.insert(0, '${shellEscapeDouble(path.resolve(process.cwd(), 'scripts'))}')
from run_trial import run_single_trial
from loss_ir_schema import LossIR
import os

loss_ir = LossIR.from_yaml('${shellEscapeDouble(lossIrPath)}') if os.path.exists('${shellEscapeDouble(lossIrPath)}') else {}

patch_spec = {
    'name': 'Agent-Generated (${strategy})',
    'mode': 'agent_generate',
    'strategy': '${strategy}',
    'code': open('${shellEscapeDouble(tmpFile)}').read(),
}

result = run_single_trial(loss_ir, patch_spec, ${trialId}, '${shellEscapeDouble(args.paper_slug)}')
print(json.dumps(result, ensure_ascii=False, default=str))
`;
      const result = await ctx.sandbox.exec(`${python} -c ${JSON.stringify(pyCode)}`, {
        timeout: 900_000, // 15 分钟
      });

      try { fs.unlinkSync(tmpFile); } catch {}

      if (result.code !== 0) {
        return {
          status: 'error',
          message: `Trial 执行失败: ${result.stderr?.slice(-500)}`,
        };
      }

      try {
        return JSON.parse(result.stdout);
      } catch {
        return {
          status: 'error',
          message: `无法解析结果: ${result.stdout?.slice(0, 500)}`,
        };
      }
    }
  },
});
