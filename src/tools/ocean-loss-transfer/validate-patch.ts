/**
 * @file validate-patch.ts
 * @description 验证 loss 文件的 4 层渐进式检查
 * @author Leizheng
 * @date 2026-03-22
 */

import { defineTool } from '@shareai-lab/kode-sdk';
import { z } from 'zod';

export const oceanLossTransferValidate = defineTool({
  name: 'ocean_loss_transfer_validate',
  description: '验证 loss 文件 (4层: static/smoke/single/full)',
  
  parameters: z.object({
    loss_file_path: z.string().describe('Loss 文件路径'),
    mode: z.enum(['static', 'smoke', 'single', 'full']).describe('验证层级')
  }),
  
  execute: async (args, ctx) => {
    const result = await ctx.sandbox.exec({
      command: 'python',
      args: [
        'scripts/ocean-loss-transfer/validate_loss.py',
        '--loss_file', args.loss_file_path,
        '--mode', args.mode
      ]
    });
    
    return JSON.parse(result.stdout);
  }
});
