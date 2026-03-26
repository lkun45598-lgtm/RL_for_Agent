/**
 * @file index.ts
 * @description Ocean Loss Transfer 工具集导出
 * @author Leizheng
 * @contributors Leizheng
 * @date 2026-03-22
 *
 * @changelog
 *   - 2026-03-23 Leizheng: 新增 prepare-context 和 write-ir 工具
 *   - 2026-03-24 Leizheng: 新增 submit-code 工具 (Agent-Native 代码提交)
 *   - 2026-03-26 OpenAI Codex: 新增 generate-plan 工具
 *   - 2026-03-26 OpenAI Codex: 新增 build-benchmark-catalog 工具
 *   - 2026-03-26 OpenAI Codex: 新增 materialize-benchmark-entry 工具
 *   - 2026-03-26 OpenAI Codex: 新增 run-benchmark-batch 工具
 */

import { oceanLossTransferValidate } from './validate-patch';
import { oceanLossTransferExtract } from './extract';
import { oceanLossTransferExtractFormula } from './extract-formula';
import { oceanLossTransferBuildBenchmarkCatalog } from './build-benchmark-catalog';
import { oceanLossTransferGeneratePlan } from './generate-plan';
import { oceanLossTransferCheckCompat } from './check-compat';
import { oceanLossTransferMaterializeBenchmarkEntry } from './materialize-benchmark-entry';
import { oceanLossTransferOrchestrate } from './orchestrate';
import { oceanLossTransferPrepareContext } from './prepare-context';
import { oceanLossTransferRunBenchmarkBatch } from './run-benchmark-batch';
import { oceanLossTransferWriteIr } from './write-ir';
import { oceanLossTransferWriteFormula } from './write-formula';
import { oceanLossTransferSubmitCode } from './submit-code';

export const oceanLossTransferTools = [
  oceanLossTransferBuildBenchmarkCatalog,
  oceanLossTransferMaterializeBenchmarkEntry,
  oceanLossTransferRunBenchmarkBatch,
  oceanLossTransferPrepareContext,
  oceanLossTransferWriteIr,
  oceanLossTransferWriteFormula,
  oceanLossTransferExtractFormula,
  oceanLossTransferGeneratePlan,
  oceanLossTransferValidate,
  oceanLossTransferExtract,
  oceanLossTransferCheckCompat,
  oceanLossTransferOrchestrate,
  oceanLossTransferSubmitCode,
];
