# 本地论文影响力评估系统汇报摘要

更新日期：2026-06-01

## 1. 思路来源

本项目的目标不是做一个不可解释的“影响力打分黑盒”，而是把公开文献中的影响力评估思想做成本地可运行、可审计、可交付的系统。

主要来源包括：

- `ImpactRankings` 的总体思路：先识别论文之间的关键影响关系，再聚合为论文、作者、机构等层级的影响力。
- citation-context bibliometrics：引用不是等权的，方法、数据、baseline、复用类引用应比背景引用更重要。
- citation graph ranking：PageRank、内部引用图、OpenCitations/BIP 类图证据可作为可审计的传统 baseline。
- field/year normalization 和 RCR-like 思路：不同领域、年份的引用规模差异很大，必须做归一化，不能直接比较原始引用数。
- SciImpact 类 benchmark：用成对影响力判断评估模型是否真的能区分更高影响力的对象。

由于公开 `ImpactRankings` 仓库没有完整公开 prompt、候选构造、推理参数、异常处理和评估协议，本项目不声称严格复现其结果；当前目标是复现其核心思想，并补上本地部署、证据融合、验证、报告和 claim boundary。

## 2. 系统设计

系统输出的是 **dataset-local paper impact index**：只评价用户给定论文集合内部的相对影响力，不直接宣称全球科学影响力。

当前主分数是 `final_impact_score`。它只使用两类证据：真实引用数和有效引用数。

核心设计：

- 本地模型负责辅助识别引用上下文和有效引用角色，默认推荐模型名为 `local-impact-judge-source-task-dpo`。
- 模型判断、图分数、外部 adoption/patent/policy/media/award/artifact 信号只作为诊断和消融字段，不直接进入最终影响力分数。
- 最终分数公式是 `0.95 * true_citation_score + 0.05 * effective_citation_score`，并输出 `1.00/0.00`、`0.95/0.05`、`0.90/0.10`、`0.80/0.20`、`0.70/0.30`、`0.50/0.50`、`0.30/0.70` 等敏感性分数。
- 每篇论文输出 `credible_evidence_tier`、`missing_high_value_evidence`、`reportability_status`、`validity_confidence` 和 claim certificate。
- 用户入口会生成 `report_decision`，直接说明结果是 `report_with_disclosure`、`exploratory_only`、`validated_report` 还是 `blocked`。
- 用户入口会生成 `model_credibility_card.md/json`，把当前模型的 SciImpact 跨 seed 证据、弱切片和可声明边界放进交付包；模型证据缺失、低可信或与实际 `MODEL` 不匹配时，`report_decision` 会降级为 `exploratory_only`。
- 用户入口会生成 `calibration_profile_card.md/json`，把 calibrated confidence 的样本量、ECE/MCE/Brier、兼容模型和 `calibration_model_identity_match` 放进交付包；校准证据缺失、质量不足或与实际 `MODEL` 不匹配时，`report_decision` 也会降级为 `exploratory_only`。

## 3. 当前可交付入口

真实用户数据入口：

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact
```

该脚本会执行：

1. 输入数据校验；
2. `model-score-preflight` 估算模型请求量；
3. 检查本地 OpenAI-compatible 模型服务；
4. 运行 `model-score-dataset`；
5. 生成 `impact_scores.csv/jsonl`；
6. 生成 `model_impact_report.md/json`；
7. 生成 `impact_claim_certificate.md/json`；
8. 生成 `impact_score_cards.md/json`；
9. 生成 `model_credibility_card.md/json`；
10. 生成 `calibration_profile_card.md/json`；
11. 刷新 `report_bundle/`，把模型可信度卡和校准可信度卡纳入交付包；
12. 生成 `user_impact_status.json` 和 `user_impact_summary.md`。

`user_impact_summary.md` 是面向汇报的短报告，包含：

- claim boundary；
- `report_decision`；
- `model_credibility_status`、`model_identity_match` 和 `model_credibility_card.md`；
- `calibration_status`、`calibration_model_identity_match` 和 `calibration_profile_card.md`；
- top final impact scores；
- evidence gaps；
- recommended next actions；
- 关键产物路径。

## 4. 当前实现进展

已实现的关键能力：

- 数据输入：JSONL、CSV 转换、输入校验、OpenAlex/OpenCitations/外部证据导入。
- 影响力评分：`final_impact_score` 两类引用分数，外加 `impact_index_v2/v3` legacy diagnostic 分数。
- 模型部署：支持本地 OpenAI-compatible vLLM 服务。
- 模型推荐：当前推荐 `local-impact-judge-source-task-dpo`，adapter 为 `qwen32b-source-task-sft16k4096-dpo16k1024-128step`。
- 长任务可靠性：pairwise replay、resume、pairwise journal。
- 公信力控制：impact claim certificate、impact validity evaluation、impact credibility scorecard、external evidence readiness。
- 模型可信度控制：当前推荐模型证据卡、cross-seed SciImpact profile、weak-slice disclosure、低可信模型自动降级。
- 校准可信度控制：SciImpact-derived confidence calibration profile、compatible model whitelist、ECE/MCE/Brier disclosure、错误校准 profile 自动降级。
- 交付包：report bundle、score cards、user summary、deliverable check。
- benchmark：SciImpact pairwise evaluation、SciImpact model credibility summary、candidate promotion gate、weak-slice error pack。

## 5. 当前验证

最新验证状态：

- 发布检查：`473 passed`，`Publish readiness checks passed.`
- no-GPU deliverable check：已通过，并包含 `user_impact=passed`。
- optional real-model deliverable smoke：已通过，`model_smoke=passed`。
- 用户入口 replay 验证：已通过，输出 `user_impact_summary.md` 和 `report_decision=report_with_disclosure`。
- GPU4 上推荐模型服务可用：`local-impact-judge-source-task-dpo`。

关键验证路径：

```bash
scripts/run_deliverable_check.sh runs/deliverable-check
CHECK_MODEL=1 BASE_URL=http://localhost:8010/v1 MODEL=local-impact-judge-source-task-dpo \
  scripts/run_deliverable_check.sh runs/deliverable-check-model
PYTHONPATH=src scripts/check_publish_ready.sh
```

## 6. 可信边界

当前系统可以汇报：

- 给定数据集内部的 dataset-local paper impact score；
- 每篇论文的 evidence-first final impact score；
- 分数来源、缺失证据、报告边界和建议补充数据；
- 本地模型在 SciImpact 类 benchmark 上的中等可信度证据。

当前系统不能直接宣称：

- not global scientific impact；
- 不是全局科学影响力排名；
- 没有 future-window/external label 时，不能宣称 validated impact accuracy；
- 没有高覆盖 DOI/OpenAlex/OpenCitations/BIP/Semantic Scholar/adoption 证据时，不能宣称 high credibility global impact。

模型侧结论也应保守：

- 当前推荐模型显著好于随机，但不是高可信 standalone impact oracle。
- 多次弱切片 DPO 继续训练未超过当前推荐 adapter，已通过 promotion gate 拒绝。
- 因此最终系统把模型作为辅助信号，把证据覆盖、外部证据、validity labels 和 claim certificate 作为公信力核心。

## 7. 一句话结论

项目已经从 MVP 进入可交付状态：用户给定论文 JSONL 数据集后，可以本地调用模型，生成每篇论文的 dataset-local 影响力分数、证据解释、可信边界、缺失证据清单、可汇报性结论和完整 report bundle。
