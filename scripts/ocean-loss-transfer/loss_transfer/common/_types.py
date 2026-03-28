"""
@file _types.py
@description 类型定义模块 - 消除隐式/显式 Any，规范化工作流数据结构
@author kongzhiquan
@contributors Leizheng
@date 2026-03-23
@version 1.2.0

@changelog
  - 2026-03-23 kongzhiquan: v1.0.0 initial version
  - 2026-03-24 Leizheng: v1.1.0 add enhanced validation types (FailureCategory,
    TrainingCurve, GradientAnalysis, SmokeTestDetail, FixAttempt, EarlyStopInfo);
    extend ValidationResult, TrialResult, ExperimentSummary with new optional fields
  - 2026-03-28 kongzhiquan: v1.2.0 remove legacy innovations.yaml/index.json typed wrappers
"""

from typing import Dict, List, Optional, Union, Literal, TypedDict

# ============================================================
# Literal 枚举类型 - 字符串常量约束
# ============================================================

PixelVariant = Literal['rel_l2', 'abs_l1', 'smooth_l1']
GradientVariant = Literal['sobel_3x3', 'scharr_3x3']
FFTVariant = Literal['residual_rfft2_abs', 'amplitude_diff']
GenerationStrategy = Literal['faithful', 'creative']
CompatibilityStatus = Literal[
    'incompatible', 'partially_compatible', 'fully_compatible'
]
ValidationMode = Literal['static', 'smoke', 'single', 'full']
ValidationLayer = Literal['layer1', 'layer2', 'layer3', 'layer4']
ComponentType = Literal[
    'pixel_loss', 'gradient_loss', 'frequency_loss',
    'structural_loss', 'unknown'
]

# 失败分类 (改进3: 实验记录增强)
FailureCategory = Literal[
    'syntax_error', 'import_error', 'forbidden_import', 'missing_function',
    'invalid_signature', 'blocked_pattern',
    'nan_in_forward', 'inf_in_forward', 'nan_in_gradient', 'no_gradient',
    'gradient_vanish', 'gradient_explode', 'invalid_output_type', 'not_scalar',
    'mask_none_failed', 'invalid_zero_mask_output', 'detached_zero_mask_loss',
    'zero_mask_no_gradient', 'runtime_error', 'import_failed',
    'ssim_collapse', 'model_collapse', 'below_baseline',
    'timeout', 'oom', 'crash', 'parse_failed',
    'nan_during_training',
    'undefined_function', 'variable_scope_error', 'device_hardcoded',
    'shape_dependent_nan', 'shape_dependent_error',
    'boundary_instability', 'boundary_gradient_nan',
]

# 训练趋势 (改进6: 超时恢复)
TrainingTrend = Literal['improving', 'degrading', 'unstable', 'insufficient_data']


# ============================================================
# 代码片段
# ============================================================

class CodeSnippet(TypedDict):
    """代码仓库中扫描到的单个文件片段"""
    file: str
    content: str


class _CodeScanResultBase(TypedDict):
    files: List[str]


class CodeScanResult(_CodeScanResultBase, total=False):
    """scan_code_for_loss 的返回值"""
    snippets: List[CodeSnippet]


# ============================================================
# Loss IR 子结构 (用于 LossComponent / LossIR 字段精化)
# ============================================================

class InputTensorSpec(TypedDict, total=False):
    """输入张量规格"""
    name: str
    shape: str
    required: bool


class ComponentImplementation(TypedDict, total=False):
    """LossComponent.implementation 的结构"""
    reduction: str          # 'mean' | 'sum'
    operates_on: str        # 'pixel_space' | 'frequency_space'


class LossInterfaceDict(TypedDict, total=False):
    """LossIR.interface 的结构"""
    input_tensors: List[InputTensorSpec]


class MultiScaleConfig(TypedDict, total=False):
    """LossIR.multi_scale 的结构"""
    enabled: bool
    scales: List[int]


class CombinationConfig(TypedDict, total=False):
    """LossIR.combination 的结构"""
    method: str             # 'weighted_sum' | ...


class IncompatibilityFlags(TypedDict, total=False):
    """LossIR.incompatibility_flags 的结构"""
    requires_model_features: bool
    requires_pretrained_network: bool
    requires_adversarial: bool
    requires_multiple_forward_passes: bool


class ComponentDict(TypedDict, total=False):
    """LossComponent 的 dict 表示 (来自 vars() 或 YAML)"""
    name: str
    type: str
    weight: float
    implementation: ComponentImplementation
    required_tensors: List[str]
    required_imports: List[str]
    formula: str
    code_evidence: Dict[str, str]


class LossIRDict(TypedDict, total=False):
    """LossIR 的 dict 表示 (来自 YAML 反序列化)"""
    metadata: Dict[str, Union[str, List[str]]]
    interface: LossInterfaceDict
    components: List[ComponentDict]
    multi_scale: MultiScaleConfig
    combination: CombinationConfig
    incompatibility_flags: IncompatibilityFlags
    _raw_code_snippets: List[CodeSnippet]


class ComponentsByType(TypedDict):
    """按类型分组的组件"""
    pixel: List[ComponentDict]
    gradient: List[ComponentDict]
    frequency: List[ComponentDict]


# ============================================================
# 增强验证类型 (borrowing from AutoResearchClaw patterns)
# ============================================================

class EpochMetric(TypedDict, total=False):
    """单个 epoch 的训练指标"""
    epoch: int
    ssim: float
    psnr: float
    rmse: float
    train_loss: float
    valid_loss: float


class TrainingCurve(TypedDict, total=False):
    """从 stdout 提取的逐 epoch 训练曲线"""
    epochs: List[EpochMetric]
    trend: TrainingTrend
    last_epoch: int
    total_expected_epochs: int


class StaticWarning(TypedDict):
    """Layer 1 静态分析的非致命警告"""
    type: str       # 'device_hardcoded' etc.
    detail: str
    line: int


class GradientAnalysis(TypedDict, total=False):
    """Layer 2 梯度流分析结果"""
    grad_norm: float
    grad_min: float
    grad_max: float
    warning: str    # 'vanishing' | 'exploding'


class SmokeTestDetail(TypedDict, total=False):
    """Layer 2 扩展 smoke 测试详情"""
    shapes_tested: List[str]
    boundary_test_passed: bool
    zero_mask_test_passed: bool
    gradient_analysis: GradientAnalysis


class FixAttempt(TypedDict, total=False):
    """单次自动修复尝试的记录"""
    attempt_num: int
    original_error: str
    fix_applied: str
    result: str     # 'fixed' | 'same_error' | 'different_error'


class EarlyStopInfo(TypedDict, total=False):
    """Trial 早停信息"""
    reason: str     # 'systematic_incompatibility'
    consecutive_failures: int
    failure_pattern: str
    trials_completed: int
    trials_skipped: int


# ============================================================
# 验证结果
# ============================================================

class TrainingMetrics(TypedDict, total=False):
    """训练指标"""
    val_ssim: float
    val_psnr: float
    swinir: float
    edsr: float
    fno2d: float
    unet2d: float


class _ValidationResultBase(TypedDict):
    passed: bool


class ValidationResult(_ValidationResultBase, total=False):
    """validate_* 函数的返回值"""
    error: str
    detail: str
    fix_hint: str
    traceback: str
    metrics: TrainingMetrics
    loss_value: float
    grad_norm: float
    # 改进1: Layer 1 深度静态分析
    warnings: List[StaticWarning]
    # 改进2: Layer 2 梯度流分析
    smoke_detail: SmokeTestDetail
    # 改进3: 失败归因分类
    failure_category: FailureCategory
    # 改进6: 超时恢复
    training_curve: TrainingCurve
    partial_metrics: TrainingMetrics
    artifact_paths: Dict[str, str]


class ValidationIRResult(TypedDict):
    """validate_loss_ir 的返回值"""
    valid: bool
    errors: List[str]


# ============================================================
# Patch 规格
# ============================================================

class TemplatePatchSpec(TypedDict, total=False):
    """模板模式的 patch 规格"""
    name: str
    pixel_variant: PixelVariant
    gradient_variant: GradientVariant
    fft_variant: FFTVariant
    scales: List[int]
    scale_weights: List[float]
    alpha: float
    beta: float
    gamma: float


class _LLMPatchSpecBase(TypedDict):
    name: str
    mode: Literal['llm_generate']
    strategy: GenerationStrategy


class LLMPatchSpec(_LLMPatchSpecBase, total=False):
    """LLM 生成模式的 patch 规格"""
    pass


class _AgentPatchSpecBase(TypedDict):
    name: str
    mode: Literal['agent_generate']
    strategy: GenerationStrategy
    code: str


class AgentPatchSpec(_AgentPatchSpecBase, total=False):
    """Agent-Native 生成模式的 patch 规格"""
    pass


PatchSpec = Union[TemplatePatchSpec, LLMPatchSpec, AgentPatchSpec]


# ============================================================
# 代码生成结果
# ============================================================

class GenerateResult(TypedDict, total=False):
    """generate_loss_code 的返回值"""
    code: str
    passed_static: bool
    passed_smoke: bool
    repair_rounds: int
    error: Optional[str]
    passed_formula_alignment: bool
    formula_alignment_error: Optional[str]
    formula_alignment_warnings: List[str]


class LLMGenerateValidation(TypedDict, total=False):
    """LLM 生成模式的验证摘要 (存入 validation_results['llm_generate'])"""
    passed_static: bool
    passed_smoke: bool
    repair_rounds: int
    error: Optional[str]
    passed_formula_alignment: bool
    formula_alignment_error: Optional[str]
    formula_alignment_warnings: List[str]


# ============================================================
# Trial 结果
# ============================================================

class _TrialResultBase(TypedDict):
    passed: bool


class TrialResult(_TrialResultBase, total=False):
    """Single attempt execution result."""
    layer_stopped: Optional[ValidationLayer]
    validation: Dict[str, ValidationResult]
    metrics: TrainingMetrics
    trial_dir: str
    error: str
    # 改进4: 执行修复循环
    fix_attempts: List[FixAttempt]
    # 改进3: 基线对比
    baseline_delta: Optional[float]


class _TrialSummaryItemBase(TypedDict):
    trial_id: int
    name: str
    passed: bool


class TrialSummaryItem(_TrialSummaryItemBase, total=False):
    """Summary item for an executed attempt."""
    layer_stopped: Optional[str]
    metrics: TrainingMetrics


# ============================================================
# 基线阈值
# ============================================================

class BaselineThresholds(TypedDict, total=False):
    """基线性能阈值"""
    model: str
    n_runs: int
    ssim_mean: float
    ssim_std: float
    psnr_mean: float
    psnr_std: float
    viable_threshold: float
    improvement_threshold: float


# ============================================================
# 实验 Summary
# ============================================================

class _ExperimentSummaryBase(TypedDict):
    paper_slug: str
    baseline: BaselineThresholds
    trials: List[TrialSummaryItem]


class ExperimentSummary(_ExperimentSummaryBase, total=False):
    """Top-level summary for the agentic loop."""
    best_trial: Optional[int]
    best_ssim: float
    improvement: Optional[float]
    # 改进5: Trial 早停
    early_stop: Optional[EarlyStopInfo]
    # 改进3: 失败归因统计
    failure_summary: Dict[str, int]


# ============================================================
# 兼容性检查
# ============================================================

class _CompatibilityResultBase(TypedDict):
    status: CompatibilityStatus


class CompatibilityResult(_CompatibilityResultBase, total=False):
    """check_compatibility 的返回值"""
    reason: str
    issues: List[str]
    warnings: List[str]


class HardIncompatibilityResult(TypedDict):
    """check_hard_incompatibility 的返回值"""
    compatible: bool
    issues: List[str]


class BlockedPatternResult(TypedDict):
    """check_blocked_patterns 的返回值"""
    warnings: List[str]


# ============================================================
# 创新点 & 知识库
# ============================================================

class InnovationEvidence(TypedDict):
    """创新点的证据"""
    baseline_ssim: float
    new_ssim: float


class _InnovationBase(TypedDict):
    paper: str
    component_type: str
    key_idea: str
    why_works: str
    improvement: float
    confidence: float
    evidence: InnovationEvidence
    tags: List[str]


class Innovation(_InnovationBase, total=False):
    """知识库中的创新点记录"""
    id: str
    date: str


class LLMInnovationExtract(TypedDict, total=False):
    """LLM 提取的创新点原始结构"""
    component_type: str
    key_idea: str
    why_works: str
    tags: List[str]


# ============================================================
# Patch 生成结果
# ============================================================

class PatchWeights(TypedDict):
    """权重配置"""
    alpha: float
    beta: float
    gamma: float


class PatchSummary(TypedDict):
    """generate_patch_from_ir 返回的 summary"""
    pixel_variant: str
    gradient_variant: str
    fft_variant: str
    scales: List[int]
    weights: PatchWeights


class PatchGenerateResult(TypedDict):
    """generate_patch_from_ir 的返回值"""
    code: str
    summary: PatchSummary


# ============================================================
# 已知失败模式配置
# ============================================================

class BlockedComponent(TypedDict, total=False):
    """blocked_patterns.yaml 中的 blocked_components 元素"""
    name: str
    action: Literal['REJECT', 'WARN']
    reason: str
    fix_hint: str


class BlockedScale(TypedDict, total=False):
    """blocked_patterns.yaml 中的 blocked_scales 元素"""
    scales: List[int]
    action: Literal['REJECT', 'WARN']
    reason: str


class BlockedPatternsConfig(TypedDict, total=False):
    """blocked_patterns.yaml 的结构"""
    blocked_components: List[BlockedComponent]
    blocked_scales: List[BlockedScale]
