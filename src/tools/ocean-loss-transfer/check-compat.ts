/**
 * @file check-compat.ts
 * @description 检查 Loss IR 兼容性
 * @author Leizheng
 * @date 2026-03-22
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { z } from 'zod';

export const oceanLossTransferCheckCompat = defineTool({
  name: 'ocean_loss_transfer_check_compat',
  description: '检查 Loss IR 与目标接口的兼容性',
  
  parameters: z.object({
    loss_ir_yaml: z.string().describe('Loss IR YAML 文件路径')
  }),
  
  execute: async (args, ctx) => {
    const result = await ctx.sandbox.exec({
      command: 'python',
      args: ['-c', `
import sys, yaml
sys.path.insert(0, '.')
from scripts.ocean_loss_transfer.check_compatibility import check_compatibility
from scripts.ocean_loss_transfer.loss_ir_schema import LossIR

data = yaml.safe_load(open('${args.loss_ir_yaml}'))
loss_ir = LossIR(**data)
result = check_compatibility(loss_ir)
print(result)
`]
    });
    
    return eval(result.stdout);
  }
});
