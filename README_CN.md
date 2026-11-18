<!-- README_SYNC: source=c1cf7fddfa49326c543bacbe665fed5e7e1d03d0; updated=2026-11-18 -->
<p align="center"><a href="./README.md" lang="en">English</a> · 简体中文</p>

# Local Impact Ranking

把一批论文及其参考文献整理成一份能追溯证据的排名。它估计的是**论文在给定语料中的影响力**：每个分数都可以回看引用依据，证据不足的结果会单独标出。

## 跑一次离线示例

在仓库根目录使用 Python 3.11 或更高版本和 Bash。装好软件包后，示例只读取随仓库提供的论文与成对判断，不需要联网、GPU、模型权重或模型服务。脚本会重新创建输出目录，请选一个可覆盖的路径。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
scripts/run_impact_demo.sh runs/impact-demo
```

先看 `runs/impact-demo/demo_report.md`；`demo_status.json` 记录回放是否完成。`model-impact/` 下的几个文件分别回答：

| 文件 | 可以查什么 |
| --- | --- |
| `impact_scores.csv` | 排名、`final_impact_score` 及每篇论文的评分依据 |
| `impact_score_cards.md` | 单篇论文的证据、置信度与各影响维度的覆盖情况 |
| `impact_claim_certificate.md` | 现有结果能支持哪些结论，还缺哪些证据 |
| `report_bundle/` | 整理好的评分、报告与来源记录 |

完整离线流程见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。`scripts/reproduce_no_gpu.sh` 运行测试、发布前检查和本地示例。`scripts/run_quickstart.sh` 展示无需模型的启发式路径，`scripts/run_deliverable_check.sh` 检查报告与报告包。只有选择 `CHECK_MODEL=1` 才需要另行配置模型服务。

## 用自己的论文数据

输入用 JSONL，每行一篇论文，至少记录稳定的 `paper_id`、标题和参考文献。有引用上下文和来源标识符时也一并提供。完整格式见[输入规范](docs/input-schema.md)。系统会对同一篇论文的参考文献去重，并把映射写到 `metadata.reference_deduplication`。

本地已有兼容 OpenAI 接口的模型服务时，可执行下面的命令。脚本会先校验数据并估算调用量，再生成排名和报告包。`MODEL` 要填服务实际运行的模型；仓库不附带模型权重。

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact
```

先读 `user_impact_summary.md`，再用 `user_impact_status.json` 核对状态。对同一份输入和配对计划，可以通过 `PAIRWISE_JUDGMENTS` 指定现有的判断 JSONL，避免再次调用模型。报告会分别列出语料覆盖、模型匹配情况与验证证据。

## 怎样读分数

`model-score-dataset` 输出语料内的分数 `final_impact_score`（dataset-local）。默认把真实引用数和语料内部的有效引用数转成排名百分位信号，权重分别为 95% 和 5%。真实引用数在核对论文身份后，为每篇选择一个兼容来源；有效引用数按指南、临床应用、方法使用等引用上下文给入边加权。同一条边同时有上下文和模型证据时取较大值，避免重复计入。

`impact_index_v2`、`impact_index_v3`、`relative_citation_ratio_score` 和引用异常诊断仍保留供复核；启发式流程也可按作者和机构汇总。排名受输入覆盖范围约束，不能直接当作全局科研影响力。当前外部指标的一致性与未来时间窗内的有效性需要分别验证。详见[评分方法](docs/scoring-method.md)、[有效引用计算](docs/effective-citation-calculation.md)和[评估方法](docs/evaluation-method.md)。

<details>
<summary>命令与证据参考</summary>

- **预检与审计。** 用 `doctor --markdown` 查看环境，或加 `doctor --output` 保存 `doctor.json`。`model-score-preflight` 估算计划配对数 `planned_pair_count` 和模型请求数 `total_model_request_count`。`external-evidence-readiness` 把覆盖情况写入 `external_evidence_readiness.json`；`citation-completeness-audit` 生成 `citation_completeness_audit.json`，区分有证据支持的零引用与来源缺失或冲突。`score-card`、`entity-resolution-report`、`runs-index` 可查看各类产物，`report-bundle` 将它们整理成报告包。
- **回放与续跑判断。** `--pairwise-judgments` 读取完整判断并记为 `pairwise_judgment_source.type=file`。`--resume-pairwise-judgments` 只补齐缺失的配对，报告记录 `pairwise_judgment_source.type=resume`、已有条数 `existing_count` 和新增条数 `generated_count`。`--pairwise-journal` 在每条新判断完成时写入 JSONL 检查点。成对信号默认使用 `rank_centrality` 聚合。
- **补充引用证据。** `enrich-openalex`、`enrich-opencitations` 可生成 `papers.openalex.jsonl`、`papers.opencitations.jsonl` 等文件。其他入口包括 `enrich-semantic-scholar`、`enrich-europepmc`、`enrich-icite`、`enrich-ncbi-citations`、`enrich-crossref`、`enrich-crossref-data-citations`、`enrich-datacite`、`enrich-datacite-events`、`enrich-openaire`、`enrich-wikidata` 和 `enrich-plos-metrics`。`AUTO_ENRICH=1` 打开联网查询，并记录 `auto_enrich_status.json` 和实际用于打分的 `scoring_input`。详见[引用数据来源](docs/citation-data-sources.md)。
- **导入已有证据。** `external-evidence-template` 和 `import-external-evidence` 支持按论文 ID、DOI、PMID、PMCID 等标识匹配。保留 uCite/NIH-OCC 生物医学引用数据及 Scopus、Web of Science、Dimensions、Lens、Google Scholar 等来源导出的计数。`scripts/apply_high_recall_evidence.sh` 汇总导入审计，其中包括 `citation_completeness_delta` 和 `evidence_validation`。
- **检查可报告范围。** `evidence_quality_flag`、`credible_evidence_tier`、`missing_high_value_evidence` 标出每篇论文的证据缺口。将 `reportability_status` 和判定依据 `reportability_basis` 与 `legacy_reportability_status` 对照，可看出新旧判定的差异。在 `model_credibility_card.md` 查看 `model_credibility_status` 和 `model_identity_match`，在 `calibration_profile_card.md` 查看 `calibration_status` 和 `calibration_model_identity_match`。基于 SciImpact 的校准档案只适用于 `compatible_models` 所列的模型。模型不匹配或证据较弱时，`report_decision` 可能保持 `exploratory_only`。结论凭证用 `validated`、`evidence_supported`、`exploratory` 或 `blocked` 概括范围。
- **核对报告包。** 给 `model-score-dataset` 传入 `--report-bundle` 和 `--report-bundle-output`。主要文件有 `bundle_manifest.json`、`model_impact_report.json/md` 和 `impact_scores.csv/jsonl`；报告还可附带模型可信度、校准档案及 `bundle_decision`，对应的文件为 `model_credibility_card.json/md` 和 `calibration_profile_card.json/md`。

开启自动补全后，脚本默认按下面的顺序查询。在论文身份恢复后，Europe PMC 会再查询一次。

```bash
AUTO_ENRICH_SOURCES=openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire
```

内置档案：[模型可信度](configs/model_credibility/local-impact-judge-source-task-dpo.json)和[校准](configs/calibration/sciimpact-qwen32b-heldout-p50.json)。

</details>

<details>
<summary>评分公式与权重敏感性</summary>

```text
final_impact_score =
  0.95 * true_citation_score
+ 0.05 * effective_citation_score
```

两项各自对计数做 `log1p` 变换，再计算保留并列的排名百分位。缺失项按零计入，并记在 `final_score_missing_signals`。选用的真实引用来源写在 `true_citation_source`。把默认的 `score_95_true_05_effective` 与 `score_90_true_10_effective`、`score_70_true_30_effective`、`score_50_true_50_effective` 等列对照，可看出调整权重会怎样改变排名。

`impact_index_v2` 和 `impact_index_v3` 是保留的消融分数；其中 v2 将 `0.30 model_pairwise` 纳入加权。可以用包含 v3 分项的表运行 `calibrate-fusion-weights`，再把结果通过 `--fusion-weights` 传回评分命令做诊断。只含 `impact_index` 的旧版 SciImpact index-eval CSV 缺少校准所需的分项。诊断权重不改变 `final_impact_score`。

</details>

<details>
<summary>评测与训练</summary>

- **当前影响力。** `benchmarks plan` 列出 `ready_now` 的评测项。`build-opcitance-icite-current-benchmark` 把引用上下文与 NIH iCite 标签对齐，生成 `opcitance_icite_benchmark_summary.md`。`evaluate-current-impact-credibility` 用当前外部指标对照分数，生成 `current_impact_credibility.json/md`。
- **未来时间窗。** 打分前先运行 `apply-citation-cutoff` 和 `audit-citation-cutoff`。验证报告缺少 `citation_cutoff_audit.json` 时，`project-readiness` 会提示；审计未通过时，报告不能据此主张未来时间窗有效。用 `derive-future-impact-labels` 准备标签，再用 `evaluate-impact-validity` 或 `backtest-impact-validity` 做评估。跨学科回测可用 `--label-normalization field-year-percentile`。评分时可用 `--validity-labels` 提供标签；`impact_credibility_scorecard` 会列出匹配覆盖（`score_label_match_rate`、`coverage_examples`）、自助法置信区间、随机置换基线及异常值稳健性检查。
- **本地判断器。** [scripts/serve_recommended_impact_judge.sh](scripts/serve_recommended_impact_judge.sh) 可启动 `local-impact-judge-source-task-dpo`，需另备基础模型、适配器权重和 vLLM。历史 `credibility-report` 对照采用 `--min-seed-count 3`、`--allowed-seed-selected-regression 0.01` 和 `--allowed-slice-regression 0.06`。在这三个留出种子上，直接判断平均准确率由 0.634 升至 0.659，双向选择平均准确率由 0.662 升至 0.669。由于专利子集的回退，更严格的 0.05 容差会保留上一版适配器。
- **评估判断器。** `split-prescience` 划分数据，`model-eval` 评估判断器。选择模型前可检查 `pairwise_credibility_scorecard`、`sciimpact_model_credibility_scorecard`、`sciimpact_model_credibility_summary`、`sciimpact_model_candidate_gate` 和 `sciimpact_model_weak_slice_error_pack`。
- **可选训练。** 模型支持时，用 `inspect-sft-tokenization` 对照 `8192` token 与 `--max-length 16384` 下的截断情况。此前 1024 token 的 LoRA 实验只用于排查短上下文限制。错误分析包可以导出 `train_pairwise_sft.jsonl`、`train_pairwise_dpo.jsonl` 和 `weak_slice_source_records.jsonl`，后续训练入口见 [scripts/train_sciimpact_weak_slice_dpo.sh](scripts/train_sciimpact_weak_slice_dpo.sh)。

</details>

本地数据、模型权重和生成的运行产物不纳入 Git。需要分享报告或发布版本时，先查阅[发布清单](docs/publish-checklist.md)。这个私有仓库没有单独的许可证文件。[原技术指南](https://github.com/LikC1606/local-impact-ranking/blob/c1cf7fddfa49326c543bacbe665fed5e7e1d03d0/README.md)保留了详细命令示例和实验记录。
