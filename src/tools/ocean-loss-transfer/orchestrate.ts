/**
 * @file orchestrate.ts
 * @description 编排 5-trial 搜索
 * @author Leizheng
 * @date 2026-03-22
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { z } from 'zod';

export const oceanLossTransferOrchestrate = defineTool({
  name: 'ocean_loss_transfer_orchestrate',
  description: '编排 5-trial 搜索实验',
  
  parameters: z.object({
    loss_ir_yaml: z.string().describe('Loss IR YAML 路径'),
    paper_slug: z.string().describe('论文标识符')
  }),
  
  execute: async (args, ctx) => {
    const result = await ctx.sandbox.exec({
      command: 'python',
      args: ['-c', `
import sys, yaml
sys.path.insert(0, '.')
from scripts.ocean_loss_transfer.orchestrate_trials import orchestrate_trials
from scripts.ocean_loss_transfer.loss_ir_schema import LossIR

data = yaml.safe_load(open('${args.loss_ir_yaml}'))
loss_ir = LossIR(**data)
summary = orchestrate_trials(loss_ir, '${args.paper_slug}')
print(yaml.dump(summary))
`]
    });
    
    return { summary: result.stdout };
  }
});
