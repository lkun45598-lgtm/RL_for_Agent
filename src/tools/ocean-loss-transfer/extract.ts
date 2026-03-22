/**
 * @file extract.ts
 * @description 从论文+代码提取 Loss IR
 * @author Leizheng
 * @date 2026-03-22
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { z } from 'zod';

export const oceanLossTransferExtract = defineTool({
  name: 'ocean_loss_transfer_extract',
  description: '从论文 PDF 和代码仓库提取 Loss IR',
  
  parameters: z.object({
    paper_pdf_path: z.string().optional().describe('论文 PDF 路径'),
    code_repo_path: z.string().optional().describe('代码仓库路径'),
    output_yaml: z.string().describe('输出 YAML 路径'),
    manual_mode: z.boolean().optional().describe('手动模式(生成模板)')
  }),
  
  execute: async (args, ctx) => {
    const pythonArgs = ['scripts/ocean-loss-transfer/extract_loss_ir.py'];
    
    const result = await ctx.sandbox.exec({
      command: 'python',
      args: ['-c', `
import sys
sys.path.insert(0, '.')
from scripts.ocean_loss_transfer.extract_loss_ir import extract_loss_ir
output = extract_loss_ir(
  paper_pdf_path=${args.paper_pdf_path ? `'${args.paper_pdf_path}'` : 'None'},
  code_repo_path=${args.code_repo_path ? `'${args.code_repo_path}'` : 'None'},
  output_yaml_path='${args.output_yaml}',
  manual_mode=${args.manual_mode || false}
)
print(output)
`]
    });
    
    return { output_file: result.stdout.trim() };
  }
});
