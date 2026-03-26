/**
 * @file prepare-context.ts
 * @description 准备论文/代码联合分析所需的上下文材料
 * @author Leizheng
 * @date 2026-03-23
 * @version 1.2.0
 *
 * @changelog
 *   - 2026-03-23 Leizheng: v1.0.0 初始版本
 *   - 2026-03-24 Leizheng: v1.1.0 支持可选 paper_pdf_path，将论文文本上下文一并返回（abstract/sections/loss_snippets）
 *   - 2026-03-26 OpenAI Codex: v1.2.0 调整 analysis_guide，使 Agent 优先产出 loss_formula 和 analysis_plan
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { findFirstPythonPath, findPythonWithModule } from '@/utils/python-manager';
import { shellEscapeDouble } from '@/utils/shell';
import path from 'node:path';

export const oceanLossTransferPrepareContext = defineTool({
  name: 'ocean_loss_transfer_prepare_context',
  description: '扫描代码仓库并整理论文/代码上下文，供 Agent 提取公式并编写 analysis_plan',

  params: {
    code_repo_path: { type: 'string', description: '代码仓库路径', required: true },
    paper_slug: { type: 'string', description: '论文标识符', required: true },
    paper_pdf_path: { type: 'string', description: '论文 PDF 路径（可选）', required: false },
    output_dir: { type: 'string', description: '输出目录（可选）', required: false }
  },

  exec: async (args, ctx) => {
    // 若用户提供了 PDF，则优先选择含 fitz(PyMuPDF) 的 Python，避免“系统 python 可用但 conda env 缺 fitz”导致 paper 上下文缺失。
    const pythonPath = args.paper_pdf_path
      ? (await findPythonWithModule('fitz')) ?? (await findFirstPythonPath())
      : await findFirstPythonPath();
    if (!pythonPath) throw new Error('未找到可用的Python解释器');

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-loss-transfer/prepare_context.py');

    const cmd = [
      `"${shellEscapeDouble(pythonPath)}"`,
      scriptPath,
      `--code_repo "${shellEscapeDouble(args.code_repo_path)}"`,
      `--paper_slug "${shellEscapeDouble(args.paper_slug)}"`,
      args.paper_pdf_path ? `--paper_pdf "${shellEscapeDouble(args.paper_pdf_path)}"` : '',
      args.output_dir ? `--output_dir "${shellEscapeDouble(args.output_dir)}"` : ''
    ].filter(Boolean).join(' ');

    const result = await ctx.sandbox.exec(cmd);

    if (result.code !== 0) {
      throw new Error(`prepare_context 失败: ${result.stderr}`);
    }

    const context = JSON.parse(result.stdout);

    return {
      ...context,
      analysis_guide: `
请按以下顺序分析并驱动 loss transfer 闭环：

0. **先读论文，再读代码**
   - 查看返回的 paper.abstract / paper.loss_snippets / paper.sections
   - 查看 primary_files 中的高优先级文件，确认 loss 是在独立 loss 文件、trainer，还是 model.forward 中实现

1. **先产出公式，不要先写 Loss IR**
   - 从论文和代码联合提取 loss 公式、关键参数、以及论文符号到代码变量名的双射映射
   - 将公式写入 formula_output_path 指向的 \`loss_formula.json\`
   - 推荐结构：
     {
       "latex": ["..."],
       "params": {"gamma": 0.85},
       "symbol_map": {"\\\\hat{y}": "pred", "y": "target", "\\\\gamma": "gamma"}
     }

2. **判断正确的 integration path**
   - 可选路径：\`loss_only\` / \`adapter_wrapper\` / \`extend_model_outputs\` / \`model_surgery\`
   - 如果论文 loss 需要中间特征、uncertainty、aux tensors 或 model.forward 的额外输出，不要强行走 \`loss_only\`
   - 模型级修改只能发生在 attempt-scoped sandbox 副本里，不能直接修改 repo-root 源码

3. **编写 analysis_plan.json**
   - 将计划写到 \`analysis_plan_output_path\`
   - 推荐结构：
     {
       "summary": "...",
       "stop_on_first_pass": false,
       "integration_decision": {
         "path": "adapter_wrapper",
         "rationale": "...",
         "evidence_refs": ["paper.loss", "code.model_forward"]
       },
       "attempts": [
         {
           "name": "Faithful attempt",
           "kind": "agent_code",
           "objective": "...",
           "files_to_edit": ["candidate_loss.py"],
           "required_edit_paths": ["sandbox_model_adapter.py"],
           "evidence_refs": ["paper.loss", "code.loss_callsite"]
         }
       ]
     }

4. **Loss IR 是可选参考，不是主产物**
   - 只有在你认为有助于总结 loss 结构时，再将 Loss IR YAML 写入 \`output_path\`
   - 不要让 Loss IR 取代 \`loss_formula.json\` 和 \`analysis_plan.json\`

完成后：
1. 调用 ocean_loss_transfer_write_formula 写入 \`loss_formula.json\`
2. 调用 ocean_loss_transfer_orchestrate，并传入 \`analysis_plan_path\`
`
    };
  }
});
