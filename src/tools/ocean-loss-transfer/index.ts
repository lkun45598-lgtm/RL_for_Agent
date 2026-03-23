/**
 * @file index.ts
 * @description Ocean Loss Transfer 工具集导出
 * @author Leizheng
 * @contributors Leizheng
 * @date 2026-03-22
 *
 * @changelog
 *   - 2026-03-23 Leizheng: 新增 prepare-context 和 write-ir 工具
 */

import { oceanLossTransferValidate } from './validate-patch';
import { oceanLossTransferExtract } from './extract';
import { oceanLossTransferCheckCompat } from './check-compat';
import { oceanLossTransferOrchestrate } from './orchestrate';
import { oceanLossTransferPrepareContext } from './prepare-context';
import { oceanLossTransferWriteIr } from './write-ir';

export const oceanLossTransferTools = [
  oceanLossTransferPrepareContext,
  oceanLossTransferWriteIr,
  oceanLossTransferValidate,
  oceanLossTransferExtract,
  oceanLossTransferCheckCompat,
  oceanLossTransferOrchestrate
];
