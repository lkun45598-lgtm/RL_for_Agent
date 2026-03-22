/**
 * @file index.ts
 * @description Ocean Loss Transfer 工具集导出
 * @author Leizheng
 * @date 2026-03-22
 */

import { oceanLossTransferValidate } from './validate-patch';
import { oceanLossTransferExtract } from './extract';
import { oceanLossTransferCheckCompat } from './check-compat';
import { oceanLossTransferOrchestrate } from './orchestrate';

export const oceanLossTransferTools = [
  oceanLossTransferValidate,
  oceanLossTransferExtract,
  oceanLossTransferCheckCompat,
  oceanLossTransferOrchestrate
];
