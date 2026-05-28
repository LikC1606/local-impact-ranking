# 本地论文影响力评估系统汇报摘要

## 1. 思路来源

本项目不是从零自定义一个“影响力分数”，而是把已有影响力评估思路组合成一个可本地运行、可审计的系统。

核心来源：

- `ImpactRankings`：采用“先识别关键参考文献，再聚合到论文、作者、机构”的总体框架。
- `Not all citations are equal`：说明引用不能简单等权，关键引用更能代表真实学术影响。
- `Identifying Meaningful Citations`：支持区分 meaningful citation 和 incidental citation。
- Content-aware bibliometrics：支持利用引用上下文、方法/数据/baseline 等语义线索给 citation edge 加权。
- Citation graph ranking / BIP! DB / OpenCitations：提供 citation count、PageRank、field/year normalization 等传统 baseline。

由于公开 `ImpactRankings` 仓库没有完整公开 prompt、候选构造、推理参数、匹配失败处理和评估协议，本项目不声称严格复现其结果，而是复现其核心方法思想，并补上本地数据、审计、评估和报告能力。

## 2. 方法设计

系统目标是生成 **dataset-local** 影响力排名：只评价用户给定论文集合内部的相对影响力，不宣称全局真实影响力。

主流程：

1. 输入论文数据，每篇论文包含自己的 references，可选 citation contexts。
2. 对每篇目标论文，模型只能从其 references 中选择 top-k 关键参考文献。
3. 将被选中的参考文献按 rank、上下文证据等规则计分。
4. 将分数聚合到 paper、author、institution、field、year。
5. 同时输出 citation count、PageRank、field/year-normalized citation baseline。
6. 输出 validation、evaluation、proxy evaluation、anomaly、stability、manifest 和 audit 文件，支持复查和汇报。

当前支持三种评分模式：

- `rank_weighted`：ImpactRankings-compatible 的 rank 权重。
- `context_weighted`：在 rank 权重上加入引用上下文证据。
- `flat`：每个有效选择等权，用于 ablation。

## 3. 当前实现进展

已实现命令：

- `validate`：数据质量检查，输出 JSON 和 Markdown。
- `convert-csv`：把 CSV 转成统一 JSONL 输入格式。
- `judge`：可恢复的独立模型判断阶段，输出 `audit/judge_outputs.jsonl`。
- `score`：从已有 judge audit 打分，不调用模型。
- `run`：端到端运行。
- `project-readiness`：汇总运行产物和质量门，判断 `reportable`、`needs_improvement` 或 `blocked`。
- `benchmarks`：列出 benchmark 并支持 SciCite 转换。
- `benchmarks convert-prescience`：把 PreScience JSONL/Parquet shard 转成项目 schema，并保留 gold positive reference IDs。

已实现输出：

- 排名：papers、authors、institutions、fields、years。
- 审计：judge outputs、完整 prompts、scoring edge ledger。
- 报告：validation、evaluation、gold evaluation、proxy evaluation、baseline comparison、data readiness、anomaly、stability、paper/entity explanations、MVP summary、project readiness。
- 复现：run manifest、artifact index。
- 微调准备：SFT、proxy-positive SFT、pairwise preference、DPO preference、LLaMAFactory config。

已接入公开 benchmark：

- PreScience adapter 已实现，可用于 key-reference selection gold evaluation。
- SciCite train/dev/test 已转换成本地 schema。
- SciImpact pairwise adapter 已实现，可把成对影响力标签转换为 `papers.jsonl` 和 `pairwise_labels.jsonl`，并用 `evaluate-pairwise-impact` 评估现有排名的 pairwise accuracy、coverage、tie rate 和分领域/维度表现。
- `scripts/run_benchmark_smoke.sh` 可在没有私有数据时跑一个无 GPU benchmark smoke。
- `scripts/run_sciimpact_smoke.sh` 可在没有下载完整 SciImpact 时验证成对影响力评估链路。

## 4. 当前验证

当前验证命令：

```bash
PYTHONPATH=src pytest -q
scripts/run_local_smoke.sh /tmp/local-impact-ranking-final-local-smoke
scripts/run_benchmark_smoke.sh /tmp/local-impact-ranking-final-benchmark-smoke
```

当前测试状态：

- 本轮优化前基线：`170 passed`
- 本轮优化后全量测试：`175 passed`
- 本地 smoke 通过：`/tmp/local-impact-ranking-full-optimization-smoke-final`
- SciCite benchmark smoke 通过：`/tmp/local-impact-ranking-full-optimization-benchmark-smoke-final`
- 未启动模型服务，未占用 GPU

## 5. 当前结论

项目已经达到可运行 MVP 状态：

- 能接收用户论文数据集，生成 dataset-local 影响力排名。
- 能使用启发式 judge 做无 GPU smoke，也能接入 OpenAI-compatible 本地模型服务。
- 能把模型选择、prompt、评分边、报告和运行环境全部落盘审计。
- 能在没有人工标注的情况下用 proxy evaluation、baseline comparison、anomaly、stability 和 project readiness 做质量检查。

需要明确的边界：

- 当前不能证明“真实全局影响力”。
- 没有人类标注时，评估结果是 proxy quality，不是 human-validated correctness。
- 更大规模正式使用前应先跑 100、1k、5k 递增 pilot。

## 6. 一句话总结

本项目基于 `ImpactRankings` 的核心思想，实现了一套本地可部署、可审计、可评估的 dataset-local 论文影响力排名系统；当前已具备从数据导入、模型判断、影响力聚合到报告汇报的完整 MVP 能力。
