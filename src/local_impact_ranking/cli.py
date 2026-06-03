from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer

from local_impact_ranking.audit import (
    append_judge_result,
    attach_judge_selections_to_papers,
    completed_paper_ids,
    judge_result_to_record,
    load_judge_results,
    write_prompt_audit,
)
from local_impact_ranking.anomalies import anomaly_report
from local_impact_ranking.artifacts import (
    OPTIONAL_RUN_ARTIFACTS,
    RUN_ARTIFACTS,
    artifact_index,
)
from local_impact_ranking.benchmark_compare import (
    write_benchmark_comparison,
    write_gold_target_comparison,
    write_model_eval,
    write_model_selection_decision,
)
from local_impact_ranking.benchmark_status import (
    benchmark_status_report,
    render_benchmark_status_markdown,
)
from local_impact_ranking.benchmarks import (
    benchmark_plan,
    get_benchmark,
    list_benchmarks,
    render_benchmark_plan_markdown,
)
from local_impact_ranking.baselines import (
    compute_bibliometric_baselines,
    export_baseline_rankings,
    summarize_baseline_comparison,
)
from local_impact_ranking.citation_cutoff import (
    write_citation_cutoff,
    write_citation_cutoff_audit,
)
from local_impact_ranking.citation_completeness_audit import (
    citation_completeness_audit_report,
    render_citation_completeness_audit_markdown,
    write_citation_completeness_audit,
)
from local_impact_ranking.contexts import context_coverage_report
from local_impact_ranking.credibility_report import write_credibility_report
from local_impact_ranking.current_impact_credibility import (
    write_current_impact_credibility,
)
from local_impact_ranking.crossref_data_citations_enrichment import (
    enrich_papers_with_crossref_data_citations,
)
from local_impact_ranking.crossref_enrichment import enrich_papers_with_crossref
from local_impact_ranking.data_readiness import data_readiness_report
from local_impact_ranking.datacite_enrichment import enrich_papers_with_datacite
from local_impact_ranking.datacite_events_enrichment import (
    enrich_papers_with_datacite_events,
)
from local_impact_ranking.doctor import doctor_report, render_doctor_markdown
from local_impact_ranking.evaluate import evaluate_judge_results, render_evaluation_markdown
from local_impact_ranking.error_analysis import (
    gold_error_analysis_report,
    render_gold_error_analysis_markdown,
)
from local_impact_ranking.entity_resolution import write_entity_resolution_report
from local_impact_ranking.external_evidence_readiness import (
    external_evidence_paper_coverage_rows,
    external_evidence_readiness_report,
    render_external_evidence_readiness_markdown,
    write_external_evidence_paper_coverage_csv,
)
from local_impact_ranking.external_evidence_import import (
    merge_external_impact_evidence,
    write_external_evidence_template,
)
from local_impact_ranking.external_identity_recovery import recover_external_identities
from local_impact_ranking.europepmc_enrichment import enrich_papers_with_europepmc
from local_impact_ranking.explanations import (
    entity_influence_explanations,
    export_paper_evidence_csv,
    paper_influence_explanations,
    paper_evidence_report,
)
from local_impact_ranking.export import (
    export_human_audit_sample,
    export_rankings_csv,
    export_scoring_edges_csv,
)
from local_impact_ranking.finetune_export import export_finetuning_datasets
from local_impact_ranking.future_impact_labels import write_future_impact_labels
from local_impact_ranking.icite_benchmark import (
    build_icite_current_metrics,
    enrich_papers_with_icite,
    evaluate_icite_current_benchmark,
)
from local_impact_ranking.dpo_lora_train import (
    load_preference_examples_with_report,
    run_dpo_lora_training,
)
from local_impact_ranking.sft_lora_train import (
    inspect_sft_tokenization,
    load_sft_examples_with_report,
    run_sft_lora_training,
)
from local_impact_ranking.gold_evaluation import (
    evaluate_gold_reference_selection,
    gold_baseline_comparison_report,
    gold_candidate_position_report,
    render_gold_baseline_comparison_markdown,
    render_gold_candidate_position_markdown,
    render_gold_evaluation_markdown,
)
from local_impact_ranking.io import convert_papers_csv_to_jsonl, load_papers_jsonl
from local_impact_ranking.impact_claim_certificate import (
    render_impact_claim_certificate_markdown,
    write_impact_claim_certificate,
)
from local_impact_ranking.impact_backtest_workflow import write_impact_backtest
from local_impact_ranking.impact_fusion_calibration import write_fusion_calibration
from local_impact_ranking.impact_validity_evaluation import (
    write_impact_validity_evaluation,
)
from local_impact_ranking.internal_citation_edges import (
    materialize_internal_citation_edges,
)
from local_impact_ranking.judge import (
    HeuristicJudge,
    Judge,
    JudgeResult,
    OpenAICompatibleJudge,
    error_result,
    with_judge_audit_metadata,
)
from local_impact_ranking.llamafactory_export import (
    export_llamafactory_configs,
    export_sciimpact_pairwise_sft_llamafactory_configs,
)
from local_impact_ranking.manifest import build_run_manifest, sha256_file
from local_impact_ranking.model_impact_index import (
    model_score_dataset,
    write_model_score_preflight,
)
from local_impact_ranking.ncbi_citation_enrichment import (
    enrich_papers_with_ncbi_citations,
)
from local_impact_ranking.normalize import normalize_paper
from local_impact_ranking.openalex_enrichment import enrich_papers_with_openalex
from local_impact_ranking.opencitations_enrichment import (
    enrich_papers_with_opencitations,
)
from local_impact_ranking.openaire_enrichment import enrich_papers_with_openaire
from local_impact_ranking.opcitance_benchmark import (
    build_opcitance_icite_current_benchmark,
)
from local_impact_ranking.semantic_scholar_enrichment import (
    enrich_papers_with_semantic_scholar,
)
from local_impact_ranking.wikidata_enrichment import enrich_papers_with_wikidata
from local_impact_ranking.pairwise_evaluation import write_pairwise_evaluation
from local_impact_ranking.plos_metrics_enrichment import enrich_papers_with_plos_metrics
from local_impact_ranking.proxy_evaluation import evaluate_proxy_quality
from local_impact_ranking.project_readiness import (
    render_project_readiness_markdown,
    write_project_readiness_report,
)
from local_impact_ranking.report_bundle import write_report_bundle
from local_impact_ranking.runs_index import write_runs_index
from local_impact_ranking.scoring import (
    add_fused_paper_scores,
    scoring_edges,
    score_influence,
    scoring_config,
)
from local_impact_ranking.scicite_adapter import convert_scicite_jsonl
from local_impact_ranking.sciimpact_adapter import (
    convert_sciimpact_jsonl,
    export_sciimpact_dpo_jsonl,
)
from local_impact_ranking.sciimpact_baseline import write_sciimpact_text_baseline_rankings
from local_impact_ranking.sciimpact_index_eval import write_sciimpact_index_evaluation
from local_impact_ranking.sciimpact_model_eval import (
    evaluate_sciimpact_flip_consistency,
    evaluate_sciimpact_predictions,
    generate_sciimpact_predictions,
    write_sciimpact_model_candidate_gate,
    write_sciimpact_model_credibility_summary,
    write_sciimpact_model_weak_slice_error_pack,
    write_sciimpact_calibration_split,
    write_sciimpact_flipped_sft,
    write_sciimpact_stratified_sample,
)
from local_impact_ranking.sciimpact_pairwise_sft import (
    export_sciimpact_pairwise_dpo,
    export_sciimpact_pairwise_sft,
    mine_sciimpact_hard_pairwise_examples,
)
from local_impact_ranking.score_cards import (
    render_score_cards_markdown,
    score_cards_from_evidence_report,
    write_score_cards,
)
from local_impact_ranking.prescience_adapter import (
    convert_prescience_file,
    write_prescience_splits,
)
from local_impact_ranking.splits import write_paper_splits
from local_impact_ranking.stability import (
    bootstrap_paper_score_report,
    export_bootstrap_paper_score_csv,
    export_paper_stability_csv,
    paper_stability_report,
    stability_report,
)
from local_impact_ranking.summary import (
    benchmark_summary_report,
    mvp_summary_report,
    ranking_preview_from_scores,
    render_benchmark_summary_markdown,
    render_mvp_summary_markdown,
)
from local_impact_ranking.validation import render_validation_markdown, validation_report

app = typer.Typer(help="Local ImpactRankings-style influence pipeline.")
benchmarks_app = typer.Typer(help="List benchmark options and mappings.")
app.add_typer(benchmarks_app, name="benchmarks")


@app.command()
def validate(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    papers = load_papers_jsonl(input)
    report = validation_report(papers)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "validation.md").write_text(
            render_validation_markdown(report),
            encoding="utf-8",
        )
    typer.echo(
        f"Loaded {len(papers)} papers. Validation issues: {report['summary']['issue_count']}"
    )


@app.command("convert-csv")
def convert_csv(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    count = convert_papers_csv_to_jsonl(input, output)
    typer.echo(f"Wrote {count} converted papers to {output}")


@app.command("doctor")
def doctor(
    root: Path = typer.Option(Path("."), "--root", exists=True, file_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    markdown: bool = typer.Option(False, "--markdown/--json"),
) -> None:
    report = doctor_report(root)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "doctor.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "doctor.md").write_text(
            render_doctor_markdown(report),
            encoding="utf-8",
        )
    if markdown:
        typer.echo(render_doctor_markdown(report))
    else:
        typer.echo(json.dumps(report, indent=2))


@app.command("runs-index")
def runs_index(
    runs_root: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    write_runs_index(runs_root, output)
    typer.echo(f"Wrote runs index to {output}")


@app.command("split")
def split_dataset(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    train_count: int = typer.Option(..., "--train-count", min=0),
    dev_count: int = typer.Option(..., "--dev-count", min=0),
    test_count: int = typer.Option(..., "--test-count", min=0),
) -> None:
    papers = load_papers_jsonl(input)
    try:
        manifest = write_paper_splits(
            papers,
            output,
            train_count=train_count,
            dev_count=dev_count,
            test_count=test_count,
            input_path=input,
            input_sha256=sha256_file(input),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        "Wrote paper splits to "
        f"{output} (train={manifest['splits']['train']['count']}, "
        f"dev={manifest['splits']['dev']['count']}, "
        f"test={manifest['splits']['test']['count']})"
    )


@app.command("train-dpo")
def train_dpo(
    model: str = typer.Option(..., "--model"),
    train_file: list[Path] = typer.Option(
        ...,
        "--train-file",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    max_steps: int = typer.Option(20, "--max-steps", min=1),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
    max_length: int = typer.Option(4096, "--max-length", min=1),
    beta: float = typer.Option(0.1, "--beta"),
    learning_rate: float = typer.Option(1e-6, "--learning-rate"),
    seed: int = typer.Option(13, "--seed"),
    gradient_accumulation_steps: int = typer.Option(
        8,
        "--gradient-accumulation-steps",
        min=1,
    ),
    lora_r: int = typer.Option(16, "--lora-r", min=1),
    lora_alpha: int = typer.Option(32, "--lora-alpha", min=1),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", min=0.0),
    device_map: str = typer.Option("auto", "--device-map"),
    base_adapter: str | None = typer.Option(None, "--base-adapter"),
    sampling_strategy: str = typer.Option("random", "--sampling-strategy"),
) -> None:
    run_dpo_lora_training(
        model_name_or_path=model,
        train_files=train_file,
        output_dir=output,
        max_steps=max_steps,
        max_examples=max_examples,
        max_length=max_length,
        beta=beta,
        learning_rate=learning_rate,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device_map=device_map,
        base_adapter=base_adapter,
        sampling_strategy=sampling_strategy.replace("-", "_"),
    )
    typer.echo(f"Wrote DPO adapter to {output}")


@app.command("train-sft")
def train_sft(
    model: str = typer.Option(..., "--model"),
    train_file: list[Path] = typer.Option(
        ...,
        "--train-file",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    max_steps: int = typer.Option(20, "--max-steps", min=1),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
    max_length: int = typer.Option(4096, "--max-length", min=1),
    learning_rate: float = typer.Option(2e-5, "--learning-rate"),
    seed: int = typer.Option(13, "--seed"),
    gradient_accumulation_steps: int = typer.Option(
        8,
        "--gradient-accumulation-steps",
        min=1,
    ),
    lora_r: int = typer.Option(16, "--lora-r", min=1),
    lora_alpha: int = typer.Option(32, "--lora-alpha", min=1),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", min=0.0),
    device_map: str = typer.Option("auto", "--device-map"),
    sampling_strategy: str = typer.Option("random", "--sampling-strategy"),
) -> None:
    run_sft_lora_training(
        model_name_or_path=model,
        train_files=train_file,
        output_dir=output,
        max_steps=max_steps,
        max_examples=max_examples,
        max_length=max_length,
        learning_rate=learning_rate,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device_map=device_map,
        sampling_strategy=sampling_strategy,
    )
    typer.echo(f"Wrote SFT adapter to {output}")


@app.command("validate-dpo-data")
def validate_dpo_data(
    train_file: list[Path] = typer.Option(
        ...,
        "--train-file",
        exists=True,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
) -> None:
    loaded = load_preference_examples_with_report(train_file, limit=max_examples)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "dpo_data_validation.json").write_text(
            json.dumps(loaded.report, indent=2) + "\n",
            encoding="utf-8",
        )
    typer.echo(
        "DPO data validation: "
        f"raw={loaded.report['raw_record_count']} "
        f"loaded={loaded.report['loaded_example_count']} "
        f"invalid={loaded.report['invalid_record_count']} "
        f"duplicates={loaded.report['duplicate_pair_count']} "
        f"reverse_conflicts={loaded.report['reverse_conflict_count']}"
    )


@app.command("validate-sft-data")
def validate_sft_data(
    train_file: list[Path] = typer.Option(
        ...,
        "--train-file",
        exists=True,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
) -> None:
    loaded = load_sft_examples_with_report(train_file, limit=max_examples)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "sft_data_validation.json").write_text(
            json.dumps(loaded.report, indent=2) + "\n",
            encoding="utf-8",
        )
    typer.echo(
        "SFT data validation: "
        f"raw={loaded.report['raw_record_count']} "
        f"loaded={loaded.report['loaded_example_count']} "
        f"invalid={loaded.report['invalid_record_count']}"
    )


@app.command("inspect-sft-tokenization")
def inspect_sft_tokenization_cmd(
    model: str = typer.Option(..., "--model"),
    train_file: list[Path] = typer.Option(
        ...,
        "--train-file",
        exists=True,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    max_length: int = typer.Option(4096, "--max-length", min=1),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
) -> None:
    tokenizer = _load_tokenizer_for_inspection(model)
    report = inspect_sft_tokenization(
        train_file,
        tokenizer=tokenizer,
        max_length=max_length,
        max_examples=max_examples,
    )
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "sft_tokenization_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "sft_tokenization_report.md").write_text(
            _render_sft_tokenization_report_markdown(report),
            encoding="utf-8",
        )
    summary = report["summary"]
    typer.echo(
        "SFT tokenization inspection: "
        f"loaded={summary['loaded_example_count']} "
        f"truncated={summary['truncated_example_count']} "
        f"prefix_retained_rate={summary['prefix_retained_rate']:.3f} "
        f"suffix_retained_rate={summary['suffix_retained_rate']:.3f} "
        f"response_fully_retained_rate={summary['response_fully_retained_rate']:.3f}"
    )


@app.command("finetuning-preflight")
def finetuning_preflight(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
) -> None:
    finetuning_dir = run_dir / "finetuning"
    readiness_path = finetuning_dir / "finetuning_readiness.json"
    dpo_path = finetuning_dir / "merged_dpo_preferences.jsonl"
    if not readiness_path.exists():
        raise typer.BadParameter(f"Missing {readiness_path}")
    if not dpo_path.exists():
        raise typer.BadParameter(f"Missing {dpo_path}")

    report = _write_finetuning_preflight(finetuning_dir)
    typer.echo(
        "Finetuning preflight: "
        f"recommended={report['decision']['recommended_next_step']} "
        f"can_start_dpo_smoke={report['decision']['can_start_dpo_smoke']}"
    )


@app.command("compare-benchmarks")
def compare_benchmarks(
    run_dirs: list[Path] = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    write_benchmark_comparison(run_dirs, output)
    typer.echo(f"Wrote benchmark comparison to {output}")


@app.command("compare-gold-targets")
def compare_gold_targets(
    base_run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    candidate_run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    write_gold_target_comparison(base_run_dir, candidate_run_dir, output)
    typer.echo(f"Wrote gold target comparison to {output}")


@app.command("model-selection-decision")
def model_selection_decision_command(
    comparison_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    max_broken_rate: float = typer.Option(0.05, "--max-broken-rate"),
    min_ndcg_delta: float = typer.Option(0.0, "--min-ndcg-delta"),
) -> None:
    write_model_selection_decision(
        comparison_dir,
        max_broken_rate=max_broken_rate,
        min_ndcg_delta=min_ndcg_delta,
    )
    typer.echo(f"Wrote model selection decision to {comparison_dir}")


@app.command("model-eval")
def model_eval(
    base_run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    candidate_run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    max_broken_rate: float = typer.Option(0.05, "--max-broken-rate"),
    min_ndcg_delta: float = typer.Option(0.0, "--min-ndcg-delta"),
) -> None:
    report = write_model_eval(
        base_run_dir,
        candidate_run_dir,
        output,
        max_broken_rate=max_broken_rate,
        min_ndcg_delta=min_ndcg_delta,
    )
    typer.echo(
        "Wrote model eval to "
        f"{output} (recommendation={report['decision']['recommendation']})"
    )


@app.command("score-card")
def score_card(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    top_n: int = typer.Option(20, "--top-n", min=1),
    max_edges: int = typer.Option(5, "--max-edges", min=0),
) -> None:
    report = write_score_cards(run_dir, output, top_n=top_n, max_edges=max_edges)
    typer.echo(f"Wrote score cards to {output} (cards={report['card_count']})")


@app.command("evaluate-pairwise-impact")
def evaluate_pairwise_impact(
    rankings: Path = typer.Option(..., "--rankings", exists=True, readable=True),
    labels: Path = typer.Option(..., "--labels", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    score_column: str | None = typer.Option(None, "--score-column"),
) -> None:
    report = write_pairwise_evaluation(
        rankings,
        labels,
        output,
        score_column=score_column,
    )
    summary = report["summary"]
    scorecard = report.get("credibility_scorecard", {})
    typer.echo(
        "Wrote pairwise impact evaluation to "
        f"{output} (pairwise_accuracy={summary['pairwise_accuracy']:.3f}, "
        f"coverage_rate={summary['coverage_rate']:.3f}, "
        f"credibility={scorecard.get('status', 'unknown')})"
    )


@app.command("evaluate-impact-validity")
def evaluate_impact_validity_cmd(
    rankings: Path = typer.Option(..., "--rankings", exists=True, readable=True),
    labels: Path = typer.Option(..., "--labels", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    score_column: str | None = typer.Option(None, "--score-column"),
    label_column: str = typer.Option("future_impact", "--label-column"),
    top_fraction: float = typer.Option(0.1, "--top-fraction", min=0.0, max=1.0),
    min_label_delta: float = typer.Option(0.0, "--min-label-delta", min=0.0),
    bootstrap_samples: int = typer.Option(200, "--bootstrap-samples", min=0),
    permutation_samples: int = typer.Option(200, "--permutation-samples", min=0),
    random_seed: int = typer.Option(0, "--random-seed"),
) -> None:
    report = write_impact_validity_evaluation(
        rankings,
        labels,
        output,
        score_column=score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    summary = report["summary"]
    significance = report.get("statistical_significance", {})
    scorecard = report.get("credibility_scorecard", {})
    typer.echo(
        "Wrote impact validity evaluation to "
        f"{output} (spearman={summary['spearman_correlation']:.3f}, "
        f"pairwise_accuracy={summary['pairwise_accuracy']:.3f}, "
        f"ndcg_at_top_k={summary['ndcg_at_top_k']:.3f}, "
        f"significant_vs_random={significance.get('significant_vs_random', False)}, "
        f"credibility={scorecard.get('status', 'unknown')})"
    )


@app.command("evaluate-current-impact-credibility")
def evaluate_current_impact_credibility_cmd(
    scores: Path = typer.Option(..., "--scores", exists=True, readable=True),
    labels: Path = typer.Option(..., "--labels", exists=True, readable=True),
    output: Path = typer.Option(..., "--output"),
    score_columns: str = typer.Option(
        "final_impact_score,score_95_true_05_effective,score_90_true_10_effective,score_80_true_20_effective,score_70_true_30_effective,score_50_true_50_effective,score_30_true_70_effective",
        "--score-columns",
        help="Comma-separated score columns to evaluate.",
    ),
    label_columns: str = typer.Option(
        "rcr,citations_per_year,apt,nih_percentile,icite_citation_count",
        "--label-columns",
        help="Comma-separated current external metric columns.",
    ),
    primary_label_column: str | None = typer.Option(
        None,
        "--primary-label-column",
        help="Primary current external metric used as the credibility gate. Defaults to the first available label column.",
    ),
    top_fraction: float = typer.Option(0.1, "--top-fraction"),
    bootstrap_samples: int = typer.Option(100, "--bootstrap-samples"),
    permutation_samples: int = typer.Option(100, "--permutation-samples"),
    random_seed: int = typer.Option(0, "--random-seed"),
) -> None:
    report = write_current_impact_credibility(
        scores,
        labels,
        output,
        score_columns=_comma_list(score_columns),
        label_columns=_comma_list(label_columns),
        primary_label_column=primary_label_column,
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    summary = report["summary"]
    typer.echo(
        "Wrote current impact credibility report to "
        f"{output} "
        f"(credibility={summary['credibility_status']}, "
        f"grade={summary['credibility_grade']})"
    )


@app.command("build-icite-current-benchmark")
def build_icite_current_benchmark_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    batch_size: int = typer.Option(200, "--batch-size", min=1),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
) -> None:
    summary = build_icite_current_metrics(
        input,
        output,
        summary_output=summary_output,
        batch_size=batch_size,
        timeout=timeout,
    )
    typer.echo(
        "Wrote NIH iCite current benchmark labels to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-icite")
def enrich_icite_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    batch_size: int = typer.Option(200, "--batch-size", min=1),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_icite(
        input,
        output,
        summary_output=summary_output,
        batch_size=batch_size,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote NIH iCite-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-europepmc")
def enrich_europepmc_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_europepmc(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote Europe PMC-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-crossref")
def enrich_crossref_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_crossref(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote Crossref-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-crossref-data-citations")
def enrich_crossref_data_citations_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_crossref_data_citations(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote Crossref Data Citations-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"positive={summary['positive_count']})"
    )


@app.command("enrich-datacite")
def enrich_datacite_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_datacite(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote DataCite-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-datacite-events")
def enrich_datacite_events_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_datacite_events(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote DataCite Events-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"positive={summary['positive_count']})"
    )


@app.command("enrich-openaire")
def enrich_openaire_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_openaire(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote OpenAIRE-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-plos-metrics")
def enrich_plos_metrics_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_plos_metrics(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote PLOS metrics-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"positive={summary['positive_count']})"
    )


@app.command("enrich-wikidata")
def enrich_wikidata_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    user_agent: str = typer.Option("local-impact-ranking/0.1", "--user-agent"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_wikidata(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        user_agent=user_agent,
        resume=resume,
    )
    typer.echo(
        "Wrote Wikidata-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"positive={summary['positive_count']})"
    )


@app.command("enrich-ncbi-citations")
def enrich_ncbi_citations_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_ncbi_citations(
        input,
        output,
        summary_output=summary_output,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote NCBI citation-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"skipped={summary['skipped_count']})"
    )


@app.command("recover-external-identities")
def recover_external_identities_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    min_title_similarity: float = typer.Option(0.92, "--min-title-similarity"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = recover_external_identities(
        input,
        output,
        summary_output=summary_output,
        min_title_similarity=min_title_similarity,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote identity-recovered papers to "
        f"{output} (recovered={summary['recovered_count']}, "
        f"unresolved={summary['unresolved_count']})"
    )


@app.command("evaluate-icite-current-benchmark")
def evaluate_icite_current_benchmark_cmd(
    papers: Path = typer.Option(..., "--papers", exists=True, readable=True),
    scores: Path = typer.Option(..., "--scores", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    batch_size: int = typer.Option(200, "--batch-size", min=1),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    top_fraction: float = typer.Option(0.1, "--top-fraction"),
    bootstrap_samples: int = typer.Option(100, "--bootstrap-samples"),
    permutation_samples: int = typer.Option(100, "--permutation-samples"),
    random_seed: int = typer.Option(0, "--random-seed"),
) -> None:
    report = evaluate_icite_current_benchmark(
        papers,
        scores,
        output,
        batch_size=batch_size,
        timeout=timeout,
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    credibility = report["credibility"]["summary"]
    typer.echo(
        "Wrote NIH iCite current benchmark evaluation to "
        f"{output} (matched={report['summary']['matched_count']}, "
        f"credibility={credibility['credibility_status']}, "
        f"grade={credibility['credibility_grade']})"
    )


@app.command("build-opcitance-icite-current-benchmark")
def build_opcitance_icite_current_benchmark_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    max_papers: int = typer.Option(1000, "--max-papers", min=1),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
    candidate_limit: int | None = typer.Option(None, "--candidate-limit", min=1),
    batch_size: int = typer.Option(200, "--batch-size", min=1),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    top_fraction: float = typer.Option(0.1, "--top-fraction"),
    bootstrap_samples: int = typer.Option(100, "--bootstrap-samples"),
    permutation_samples: int = typer.Option(100, "--permutation-samples"),
    random_seed: int = typer.Option(0, "--random-seed"),
) -> None:
    summary = build_opcitance_icite_current_benchmark(
        input,
        output,
        max_papers=max_papers,
        max_rows=max_rows,
        candidate_limit=candidate_limit,
        batch_size=batch_size,
        timeout=timeout,
        top_fraction=top_fraction,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    typer.echo(
        "Wrote OpCitance + iCite current benchmark to "
        f"{output} (papers={summary['paper_count']}, "
        f"internal_edges={summary['internal_reference_edges']}, "
        f"credibility={summary['credibility_status']}, "
        f"grade={summary['credibility_grade']})"
    )


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command("derive-future-impact-labels")
def derive_future_impact_labels_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    start_year: int = typer.Option(..., "--start-year"),
    end_year: int = typer.Option(..., "--end-year"),
    label_column: str = typer.Option("future_impact", "--label-column"),
    normalization: str = typer.Option("raw", "--normalization"),
) -> None:
    report = write_future_impact_labels(
        input_path,
        output,
        start_year=start_year,
        end_year=end_year,
        label_column=label_column,
        normalization=normalization,
    )
    summary = report["summary"]
    typer.echo(
        "Wrote future impact labels to "
        f"{output} (labels={summary['label_count']} "
        f"skipped={summary['skipped_count']} "
        f"normalization={summary['normalization']})"
    )


@app.command("apply-citation-cutoff")
def apply_citation_cutoff_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    cutoff_year: int = typer.Option(..., "--cutoff-year"),
) -> None:
    report = write_citation_cutoff(
        input_path,
        output,
        cutoff_year=cutoff_year,
    )
    summary = report["summary"]
    typer.echo(
        "Wrote citation-cutoff dataset to "
        f"{output} (cutoff_year={summary['cutoff_year']} "
        f"updated_histories={summary['updated_citation_history_count']})"
    )


@app.command("audit-citation-cutoff")
def audit_citation_cutoff_cmd(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    cutoff_year: int = typer.Option(..., "--cutoff-year"),
) -> None:
    report = write_citation_cutoff_audit(
        input_path,
        output,
        cutoff_year=cutoff_year,
    )
    decision = report["decision"]
    summary = report["summary"]
    typer.echo(
        "Wrote citation-cutoff audit to "
        f"{output} (status={decision['status']} "
        f"future_year_papers={summary['papers_with_future_years']})"
    )


@app.command("backtest-impact-validity")
def backtest_impact_validity_cmd(
    enriched_input: Path = typer.Option(
        ...,
        "--enriched-input",
        exists=True,
        readable=True,
    ),
    cutoff_input: Path = typer.Option(
        ...,
        "--cutoff-input",
        exists=True,
        readable=True,
    ),
    rankings: Path = typer.Option(..., "--rankings", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    cutoff_year: int = typer.Option(..., "--cutoff-year"),
    future_start_year: int = typer.Option(..., "--future-start-year"),
    future_end_year: int = typer.Option(..., "--future-end-year"),
    score_column: str | None = typer.Option(None, "--score-column"),
    label_column: str = typer.Option("future_impact", "--label-column"),
    label_normalization: str = typer.Option(
        "field-year-percentile",
        "--label-normalization",
    ),
    top_fraction: float = typer.Option(0.1, "--top-fraction", min=0.0, max=1.0),
    min_label_delta: float = typer.Option(0.0, "--min-label-delta", min=0.0),
    bootstrap_samples: int = typer.Option(200, "--bootstrap-samples", min=0),
    permutation_samples: int = typer.Option(200, "--permutation-samples", min=0),
    random_seed: int = typer.Option(0, "--random-seed"),
) -> None:
    manifest = write_impact_backtest(
        enriched_input=enriched_input,
        cutoff_input=cutoff_input,
        rankings=rankings,
        output_dir=output,
        cutoff_year=cutoff_year,
        future_start_year=future_start_year,
        future_end_year=future_end_year,
        score_column=score_column,
        label_column=label_column,
        label_normalization=label_normalization,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    summary = manifest["summary"]
    typer.echo(
        "Wrote impact backtest evidence to "
        f"{output} (cutoff_audit={summary['cutoff_audit_status']} "
        f"label_normalization={summary['label_normalization']} "
        f"labels={summary['future_label_count']} "
        f"validity={summary['validity_status']} "
        f"significant_vs_random={summary['validity_significant_vs_random']})"
    )


@app.command("calibrate-fusion-weights")
def calibrate_fusion_weights_cmd(
    rankings: Path = typer.Option(..., "--rankings", exists=True, readable=True),
    labels: Path = typer.Option(..., "--labels", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    grid_step: float = typer.Option(0.1, "--grid-step", min=0.01, max=1.0),
) -> None:
    report = write_fusion_calibration(
        rankings,
        labels,
        output,
        grid_step=grid_step,
    )
    summary = report["summary"]
    typer.echo(
        "Wrote fusion calibration to "
        f"{output} (best_pairwise_accuracy={summary['best_pairwise_accuracy']:.3f}, "
        f"covered_pairs={summary['covered_pairs']}, "
        f"profile={summary.get('weight_profile', 'unknown')})"
    )


@app.command("enrich-openalex")
def enrich_openalex_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    email: str | None = typer.Option(None, "--email"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_openalex(
        input,
        output,
        email=email,
        timeout=timeout,
        resume=resume,
    )
    typer.echo(
        "Wrote OpenAlex-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']})"
    )


@app.command("enrich-opencitations")
def enrich_opencitations_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    token: str | None = typer.Option(None, "--token"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    include_edges: bool = typer.Option(
        True,
        "--include-edges/--no-include-edges",
    ),
    max_incoming_edges_per_paper: int = typer.Option(
        500,
        "--max-incoming-edges-per-paper",
        min=0,
    ),
    materialize_internal_edges: bool = typer.Option(
        False,
        "--materialize-internal-edges/--no-materialize-internal-edges",
    ),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_opencitations(
        input,
        output,
        token=token,
        timeout=timeout,
        include_edges=include_edges,
        max_incoming_edges_per_paper=max_incoming_edges_per_paper,
        materialize_internal_edges=materialize_internal_edges,
        resume=resume,
    )
    typer.echo(
        "Wrote OpenCitations-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']}, "
        f"internal_edges={summary['internal_incoming_edge_count']}, "
        f"materialized_edges={summary['materialized_internal_edge_count']})"
    )


@app.command("enrich-semantic-scholar")
def enrich_semantic_scholar_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    api_key: str | None = typer.Option(None, "--api-key"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
    batch_size: int = typer.Option(100, "--batch-size", min=1),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    summary = enrich_papers_with_semantic_scholar(
        input,
        output,
        api_key=api_key,
        timeout=timeout,
        batch_size=batch_size,
        resume=resume,
    )
    typer.echo(
        "Wrote Semantic Scholar-enriched papers to "
        f"{output} (matched={summary['matched_count']}, "
        f"unmatched={summary['unmatched_count']}, "
        f"title_matches={summary['title_match_count']})"
    )


@app.command("materialize-internal-citations")
def materialize_internal_citations_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
    title_threshold: float = typer.Option(0.95, "--title-threshold", min=0.0, max=1.0),
    openalex_references: bool = typer.Option(
        False,
        "--openalex-references/--no-openalex-references",
    ),
    openalex_email: str | None = typer.Option(None, "--openalex-email"),
    openalex_timeout: float = typer.Option(30.0, "--openalex-timeout", min=1.0),
) -> None:
    summary = materialize_internal_citation_edges(
        input,
        output,
        summary_output=summary_output,
        title_threshold=title_threshold,
        openalex_references=openalex_references,
        openalex_email=openalex_email,
        openalex_timeout=openalex_timeout,
    )
    typer.echo(
        "Wrote papers with materialized internal citation edges to "
        f"{output} (materialized={summary['materialized_internal_edge_count']}, "
        f"existing={summary['existing_internal_edge_count']}, "
        f"unmatched_references={summary['unmatched_reference_count']})"
    )


@app.command("external-evidence-readiness")
def external_evidence_readiness_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    markdown: bool = typer.Option(False, "--markdown/--json"),
) -> None:
    papers = load_papers_jsonl(input)
    report = external_evidence_readiness_report(papers)
    rendered = render_external_evidence_readiness_markdown(report)
    coverage_rows = external_evidence_paper_coverage_rows(papers)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "external_evidence_readiness.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "external_evidence_readiness.md").write_text(
            rendered,
            encoding="utf-8",
        )
        write_external_evidence_paper_coverage_csv(
            coverage_rows,
            output / "external_evidence_paper_coverage.csv",
        )
    if markdown:
        typer.echo(rendered)
    else:
        typer.echo(json.dumps(report, indent=2))


@app.command("citation-completeness-audit")
def citation_completeness_audit_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    markdown: bool = typer.Option(False, "--markdown/--json"),
) -> None:
    papers = load_papers_jsonl(input)
    if output is not None:
        report = write_citation_completeness_audit(papers, output)
    else:
        report = citation_completeness_audit_report(papers)
    if markdown:
        typer.echo(render_citation_completeness_audit_markdown(report))
    else:
        typer.echo(json.dumps(report, indent=2))


@app.command("import-external-evidence")
def import_external_evidence_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    evidence: Path = typer.Option(..., "--evidence", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    summary_output: Path | None = typer.Option(None, "--summary-output"),
) -> None:
    summary = merge_external_impact_evidence(input, evidence, output)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    typer.echo(
        "Imported external evidence to "
        f"{output} (papers={summary['paper_count']}, matched={summary['matched_count']})"
    )


@app.command("external-evidence-template")
def external_evidence_template_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    summary = write_external_evidence_template(input, output)
    typer.echo(
        "Wrote external evidence template to "
        f"{output} (papers={summary['paper_count']}, "
        f"missing_query_ids={summary['papers_without_query_id']})"
    )


@app.command("model-score-preflight")
def model_score_preflight_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    max_pairs: int | None = typer.Option(None, "--max-pairs", min=1),
    pair_budget: str = typer.Option("manual", "--pair-budget"),
    target_paper_coverage: float = typer.Option(
        1.0,
        "--target-paper-coverage",
        min=0.0,
        max=1.0,
    ),
    comparison_rounds: int = typer.Option(1, "--comparison-rounds", min=1),
    shuffle_pairs: bool = typer.Option(False, "--shuffle-pairs/--no-shuffle-pairs"),
    pair_sampling: str = typer.Option("sequential", "--pair-sampling"),
    seed: int = typer.Option(0, "--seed"),
    direct_score: bool = typer.Option(False, "--direct-score/--no-direct-score"),
    bidirectional_pairwise: bool = typer.Option(
        False,
        "--bidirectional-pairwise/--single-direction-pairwise",
    ),
    resume_pairwise_judgments: Path | None = typer.Option(
        None,
        "--resume-pairwise-judgments",
        exists=True,
        readable=True,
    ),
) -> None:
    report = write_model_score_preflight(
        input,
        output,
        max_pairs=max_pairs,
        pair_budget=pair_budget,
        target_paper_coverage=target_paper_coverage,
        comparison_rounds=comparison_rounds,
        shuffle_pairs=shuffle_pairs,
        pair_sampling=pair_sampling,
        seed=seed,
        direct_score=direct_score,
        bidirectional_pairwise=bidirectional_pairwise,
        resume_pairwise_judgments_path=resume_pairwise_judgments,
    )
    typer.echo(
        "Wrote model score preflight to "
        f"{output} (papers={report['paper_count']}, "
        f"requests={report['request_estimate']['total_model_request_count']})"
    )


@app.command("model-score-dataset")
def model_score_dataset_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--base-url"),
    model: str = typer.Option(..., "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    timeout: float = typer.Option(120.0, "--timeout"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int | None = typer.Option(512, "--max-tokens", min=1),
    max_pairs: int | None = typer.Option(None, "--max-pairs", min=1),
    pair_budget: str = typer.Option("manual", "--pair-budget"),
    target_paper_coverage: float = typer.Option(
        1.0,
        "--target-paper-coverage",
        min=0.0,
        max=1.0,
    ),
    comparison_rounds: int = typer.Option(1, "--comparison-rounds", min=1),
    shuffle_pairs: bool = typer.Option(False, "--shuffle-pairs/--no-shuffle-pairs"),
    pair_sampling: str = typer.Option("sequential", "--pair-sampling"),
    seed: int = typer.Option(0, "--seed"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    direct_score: bool = typer.Option(False, "--direct-score/--no-direct-score"),
    calibration_report: Path | None = typer.Option(
        None,
        "--calibration-report",
        exists=True,
        readable=True,
    ),
    fusion_weights: Path | None = typer.Option(
        None,
        "--fusion-weights",
        exists=True,
        readable=True,
    ),
    bidirectional_pairwise: bool = typer.Option(
        False,
        "--bidirectional-pairwise/--single-direction-pairwise",
    ),
    bidirectional_confidence_margin: float = typer.Option(
        0.0,
        "--bidirectional-confidence-margin",
        min=0.0,
    ),
    pairwise_aggregation: str = typer.Option(
        "rank_centrality",
        "--pairwise-aggregation",
        help="Pairwise aggregation method: rank_centrality or elo.",
    ),
    pairwise_judgments: Path | None = typer.Option(
        None,
        "--pairwise-judgments",
        exists=True,
        readable=True,
        help="Replay an existing pairwise_judgments.jsonl file instead of calling the model for pairwise judgments.",
    ),
    resume_pairwise_judgments: Path | None = typer.Option(
        None,
        "--resume-pairwise-judgments",
        exists=True,
        readable=True,
        help="Load existing pairwise judgments and call the model only for missing planned pairs.",
    ),
    pairwise_journal: Path | None = typer.Option(
        None,
        "--pairwise-journal",
        help="Write each newly generated pairwise judgment to this JSONL checkpoint as soon as it completes.",
    ),
    validity_labels: Path | None = typer.Option(
        None,
        "--validity-labels",
        exists=True,
        readable=True,
    ),
    validity_label_column: str = typer.Option(
        "future_impact",
        "--validity-label-column",
    ),
    validity_bootstrap_samples: int = typer.Option(
        200,
        "--validity-bootstrap-samples",
        min=0,
    ),
    validity_permutation_samples: int = typer.Option(
        200,
        "--validity-permutation-samples",
        min=0,
    ),
    pairwise_stability_samples: int = typer.Option(
        0,
        "--pairwise-stability-samples",
        min=0,
        help="Bootstrap samples for pairwise judgment stability auditing. Use 0 to disable.",
    ),
    pairwise_stability_top_n: int = typer.Option(
        20,
        "--pairwise-stability-top-n",
        min=1,
        help="Number of top model-pairwise papers to track in the stability report.",
    ),
    pairwise_stability_seed: int = typer.Option(
        17,
        "--pairwise-stability-seed",
        help="Random seed for pairwise stability bootstrap.",
    ),
    final_score_stability_samples: int = typer.Option(
        0,
        "--final-score-stability-samples",
        min=0,
        help="Bootstrap samples for final fused score stability auditing. Use 0 to disable.",
    ),
    final_score_stability_top_n: int = typer.Option(
        20,
        "--final-score-stability-top-n",
        min=1,
        help="Number of top final-impact-score papers to track in the stability report.",
    ),
    final_score_stability_seed: int = typer.Option(
        23,
        "--final-score-stability-seed",
        help="Random seed for final score stability bootstrap.",
    ),
    report_bundle: bool = typer.Option(
        False,
        "--report-bundle/--no-report-bundle",
    ),
    report_bundle_output: Path | None = typer.Option(
        None,
        "--report-bundle-output",
    ),
) -> None:
    report = model_score_dataset(
        input,
        output,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        max_pairs=max_pairs,
        pair_budget=pair_budget,
        target_paper_coverage=target_paper_coverage,
        comparison_rounds=comparison_rounds,
        shuffle_pairs=shuffle_pairs,
        pair_sampling=pair_sampling,
        seed=seed,
        concurrency=concurrency,
        direct_score=direct_score,
        calibration_report_path=calibration_report,
        fusion_weights_path=fusion_weights,
        bidirectional_pairwise=bidirectional_pairwise,
        bidirectional_confidence_margin=bidirectional_confidence_margin,
        pairwise_aggregation_method=pairwise_aggregation,
        pairwise_judgments_path=pairwise_judgments,
        resume_pairwise_judgments_path=resume_pairwise_judgments,
        pairwise_journal_path=pairwise_journal,
        validity_labels_path=validity_labels,
        validity_label_column=validity_label_column,
        validity_bootstrap_samples=validity_bootstrap_samples,
        validity_permutation_samples=validity_permutation_samples,
        pairwise_stability_samples=pairwise_stability_samples,
        pairwise_stability_top_n=pairwise_stability_top_n,
        pairwise_stability_seed=pairwise_stability_seed,
        final_score_stability_samples=final_score_stability_samples,
        final_score_stability_top_n=final_score_stability_top_n,
        final_score_stability_seed=final_score_stability_seed,
    )
    typer.echo(
        "Wrote model impact index to "
        f"{output} (papers={report['summary']['paper_count']}, "
        f"judgments={report['summary']['judgment_count']})"
    )
    if report_bundle:
        bundle_output = report_bundle_output or (output / "report_bundle")
        write_report_bundle(output, bundle_output)
        typer.echo(f"Wrote model score report bundle to {bundle_output}")


@app.command("entity-resolution-report")
def entity_resolution_report_command(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    papers = load_papers_jsonl(input)
    report = write_entity_resolution_report(papers, output)
    typer.echo(
        "Wrote entity resolution report to "
        f"{output} (scope={report['quality_gate']['entity_rankings_scope']})"
    )


@app.command("benchmark-status")
def benchmark_status(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    markdown: bool = typer.Option(False, "--markdown/--json"),
) -> None:
    report = benchmark_status_report(run_dir)
    if markdown:
        typer.echo(render_benchmark_status_markdown(report))
    else:
        typer.echo(json.dumps(report, indent=2))


@app.command("credibility-report")
def credibility_report_command(
    baseline_run: list[Path] = typer.Option(
        ...,
        "--baseline-run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    candidate_run: list[Path] = typer.Option(
        ...,
        "--candidate-run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    baseline_model: str = typer.Option("baseline", "--baseline-model"),
    candidate_model: str = typer.Option("candidate", "--candidate-model"),
    seed_label: list[str] | None = typer.Option(None, "--seed-label"),
    min_selected_delta: float = typer.Option(0.0, "--min-selected-delta"),
    min_seed_count: int = typer.Option(1, "--min-seed-count", min=1),
    allowed_seed_selected_regression: float = typer.Option(
        0.0,
        "--allowed-seed-selected-regression",
        min=0.0,
    ),
    allowed_slice_regression: float | None = typer.Option(
        None,
        "--allowed-slice-regression",
        min=0.0,
    ),
) -> None:
    try:
        report = write_credibility_report(
            baseline_runs=baseline_run,
            candidate_runs=candidate_run,
            output_dir=output,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            seed_labels=seed_label,
            min_selected_delta=min_selected_delta,
            min_seed_count=min_seed_count,
            allowed_seed_selected_regression=allowed_seed_selected_regression,
            allowed_slice_regression=allowed_slice_regression,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        "Wrote credibility report to "
        f"{output} (recommendation={report['decision']['recommendation']})"
    )


@app.command("project-readiness")
def project_readiness(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    markdown: bool = typer.Option(False, "--markdown/--json"),
) -> None:
    report = _write_project_readiness_with_artifact_index(run_dir)
    if markdown:
        typer.echo(render_project_readiness_markdown(report))
    else:
        typer.echo(json.dumps(report, indent=2))


@app.command("impact-claim-certificate")
def impact_claim_certificate(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    markdown: bool = typer.Option(False, "--markdown/--json"),
    top_n: int = typer.Option(10, "--top-n", min=1),
) -> None:
    certificate = write_impact_claim_certificate(run_dir, output, top_n=top_n)
    if markdown:
        typer.echo(render_impact_claim_certificate_markdown(certificate))
    else:
        destination = output if output is not None else _impact_claim_default_output(run_dir)
        typer.echo(f"Wrote impact claim certificate to {destination}")


@app.command("report-bundle")
def report_bundle(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    _write_project_readiness_with_artifact_index(run_dir)
    write_impact_claim_certificate(run_dir)
    write_report_bundle(run_dir, output)
    typer.echo(f"Wrote report bundle to {output}")


@app.command("analyze-gold-errors")
def analyze_gold_errors(
    gold_evaluation: Path = typer.Option(
        ...,
        "--gold-evaluation",
        exists=True,
        readable=True,
    ),
    judge_audit: Path | None = typer.Option(
        None,
        "--judge-audit",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    max_failures: int = typer.Option(50, "--max-failures", min=1),
) -> None:
    gold_report = json.loads(gold_evaluation.read_text(encoding="utf-8"))
    candidate_ids_by_target = None
    if judge_audit is not None:
        candidate_ids_by_target = {
            result.target_paper_id: list(result.candidate_reference_ids)
            for result in load_judge_results(judge_audit)
        }
    report = gold_error_analysis_report(
        gold_report,
        candidate_ids_by_target=candidate_ids_by_target,
        max_failures=max_failures,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "gold_error_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "gold_error_analysis.md").write_text(
        render_gold_error_analysis_markdown(report),
        encoding="utf-8",
    )
    typer.echo(f"Wrote gold error analysis to {output}")


@benchmarks_app.command("list")
def benchmarks_list() -> None:
    for spec in list_benchmarks():
        typer.echo(
            f"{spec.recommended_order}. {spec.name} "
            f"[{spec.implementation_status}] - {spec.purpose}"
        )


@benchmarks_app.command("show")
def benchmarks_show(name: str = typer.Argument(...)) -> None:
    spec = get_benchmark(name)
    typer.echo(
        json.dumps(
            {
                "key": spec.key,
                "name": spec.name,
                "purpose": spec.purpose,
                "pipeline_layers": list(spec.pipeline_layers),
                "preferred_input": spec.preferred_input,
                "notes": spec.notes,
                "source_url": spec.source_url,
                "paper_url": spec.paper_url,
                "implementation_status": spec.implementation_status,
                "recommended_order": spec.recommended_order,
                "evaluation_metrics": list(spec.evaluation_metrics),
                "evaluation_role": spec.evaluation_role,
                "local_command": spec.local_command,
                "transfer_risk": spec.transfer_risk,
            },
            indent=2,
        )
    )


@benchmarks_app.command("plan")
def benchmarks_plan(markdown: bool = typer.Option(False, "--markdown/--json")) -> None:
    plan = benchmark_plan()
    if markdown:
        typer.echo(render_benchmark_plan_markdown(plan))
    else:
        typer.echo(json.dumps(plan, indent=2))


@benchmarks_app.command("convert-scicite")
def benchmarks_convert_scicite(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    count = convert_scicite_jsonl(input, output)
    typer.echo(f"Wrote {count} converted papers to {output}")


@benchmarks_app.command("convert-sciimpact")
def benchmarks_convert_sciimpact(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
) -> None:
    manifest = convert_sciimpact_jsonl(input, output, skip_invalid=not strict)
    typer.echo(
        "Wrote "
        f"{manifest['pair_count']} SciImpact pairs and "
        f"{manifest['paper_count']} papers to {output}"
    )
    if manifest["skipped_pair_count"]:
        typer.echo(f"Skipped {manifest['skipped_pair_count']} invalid SciImpact rows")


@benchmarks_app.command("export-sciimpact-dpo")
def benchmarks_export_sciimpact_dpo(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
) -> None:
    manifest = export_sciimpact_dpo_jsonl(input, output, skip_invalid=not strict)
    typer.echo(
        "Wrote "
        f"{manifest['dpo_record_count']} SciImpact DPO preference records to {output}"
    )
    if manifest["skipped_record_count"]:
        typer.echo(f"Skipped {manifest['skipped_record_count']} invalid SciImpact rows")


@benchmarks_app.command("convert-prescience")
def benchmarks_convert_prescience(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    max_candidates: int = typer.Option(50, "--max-candidates", min=1),
    shuffle_seed: int = typer.Option(13, "--shuffle-seed"),
) -> None:
    count = convert_prescience_file(
        input,
        output,
        limit=limit,
        max_candidates=max_candidates,
        shuffle_seed=shuffle_seed,
    )
    typer.echo(f"Wrote {count} converted papers to {output}")


@benchmarks_app.command("split-prescience")
def benchmarks_split_prescience(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    train_count: int = typer.Option(..., "--train-count", min=0),
    dev_count: int = typer.Option(..., "--dev-count", min=0),
    test_count: int = typer.Option(..., "--test-count", min=0),
    max_candidates: int = typer.Option(50, "--max-candidates", min=1),
    shuffle_seed: int = typer.Option(13, "--shuffle-seed"),
) -> None:
    try:
        manifest = write_prescience_splits(
            input,
            output,
            train_count=train_count,
            dev_count=dev_count,
            test_count=test_count,
            max_candidates=max_candidates,
            shuffle_seed=shuffle_seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        "Wrote PreScience splits to "
        f"{output} (train={manifest['splits']['train']['count']}, "
        f"dev={manifest['splits']['dev']['count']}, "
        f"test={manifest['splits']['test']['count']})"
    )


@app.command()
def judge(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    audit_output: Path = typer.Option(..., "--audit-output"),
    top_k: int = typer.Option(5, "--top-k", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    judge_name: str = typer.Option("heuristic", "--judge"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--base-url"),
    model: str = typer.Option("", "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    timeout: float = typer.Option(120.0, "--timeout"),
    temperature: float = typer.Option(0.0, "--temperature"),
    repair_attempts: int = typer.Option(1, "--repair-attempts", min=0),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1),
    json_mode: bool = typer.Option(True, "--json-mode/--no-json-mode"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_retries: int = typer.Option(0, "--max-retries", min=0),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input)]
    if limit is not None:
        papers = papers[:limit]
    done = completed_paper_ids(audit_output)
    selected_papers = [paper for paper in papers if paper.paper_id not in done]
    if progress:
        typer.echo(
            "Judge progress: "
            f"total={len(papers)} done={len(done)} skipped={len(done)} "
            f"to_run={len(selected_papers)}"
        )
    judge_impl = _build_judge(
        judge_name=judge_name,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        repair_attempts=repair_attempts,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )
    if concurrency == 1:
        for completed_count, paper in enumerate(selected_papers, start=1):
            result = _judge_one(paper, judge_impl, top_k, max_retries)
            append_judge_result(audit_output, result)
            if progress:
                typer.echo(
                    f"Judged {completed_count}/{len(selected_papers)}: "
                    f"{result.target_paper_id} -> {audit_output}"
                )
    else:
        for completed_count, result in enumerate(
            _judge_papers(
                selected_papers,
                judge_impl=judge_impl,
                top_k=top_k,
                concurrency=concurrency,
                max_retries=max_retries,
            ),
            start=1,
        ):
            append_judge_result(audit_output, result)
            if progress:
                typer.echo(
                    f"Judged {completed_count}/{len(selected_papers)}: "
                    f"{result.target_paper_id} -> {audit_output}"
                )
    typer.echo(f"Wrote {len(selected_papers)} judge records to {audit_output}")


@app.command("attach-judge-selections")
def attach_judge_selections(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    judge_audit: Path = typer.Option(
        ...,
        "--judge-audit",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    metadata_key: str = typer.Option(
        "impact_selected_references",
        "--metadata-key",
    ),
) -> None:
    manifest = attach_judge_selections_to_papers(
        input,
        judge_audit,
        output,
        metadata_key=metadata_key,
    )
    typer.echo(
        "Attached judge selections to "
        f"{manifest['matched_paper_count']}/{manifest['paper_count']} papers "
        f"({manifest['valid_selected_reference_count']}/"
        f"{manifest['selected_reference_count']} valid selections) -> {output}"
    )


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    top_k: int = typer.Option(5, "--top-k", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    credit_mode: str = typer.Option("rank_weighted", "--credit-mode"),
    judge_name: str = typer.Option("heuristic", "--judge"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--base-url"),
    model: str = typer.Option("", "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    timeout: float = typer.Option(120.0, "--timeout"),
    temperature: float = typer.Option(0.0, "--temperature"),
    repair_attempts: int = typer.Option(1, "--repair-attempts", min=0),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1),
    json_mode: bool = typer.Option(True, "--json-mode/--no-json-mode"),
    judge_audit: Path | None = typer.Option(None, "--judge-audit", exists=True, readable=True),
    strict_validation: bool = typer.Option(False, "--strict-validation"),
) -> None:
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input)]
    if limit is not None:
        papers = papers[:limit]
    reports_dir = output / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    current_validation_report = validation_report(papers)
    _write_validation_outputs(current_validation_report, reports_dir)
    if strict_validation and not current_validation_report["summary"]["can_run"]:
        typer.echo(
            "Validation failed; reports written to "
            f"{reports_dir / 'validation.json'} and {reports_dir / 'validation.md'}"
        )
        raise typer.Exit(code=1)

    if judge_audit is not None:
        selected_paper_ids = {paper.paper_id for paper in papers}
        judge_results = [
            result
            for result in load_judge_results(judge_audit)
            if result.target_paper_id in selected_paper_ids
        ]
    else:
        judge = _build_judge(
            judge_name=judge_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            repair_attempts=repair_attempts,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        judge_results = [judge.judge(paper, top_k=top_k) for paper in papers]
    raw_scores = score_influence(papers, judge_results, credit_mode=credit_mode)

    rankings_dir = output / "rankings"
    baselines_dir = output / "baselines"
    audit_dir = output / "audit"
    review_dir = output / "review"
    finetuning_dir = output / "finetuning"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    finetuning_dir.mkdir(parents=True, exist_ok=True)

    baselines = compute_bibliometric_baselines(papers)
    scores = add_fused_paper_scores(
        raw_scores,
        graph_scores=baselines.pagerank,
        paper_metadata=_paper_metadata_index(papers),
    )
    export_rankings_csv(scores, rankings_dir)
    export_baseline_rankings(baselines, baselines_dir)
    export_human_audit_sample(papers, judge_results, review_dir / "human_audit_sample.csv")
    _write_judge_audit(judge_results, audit_dir / "judge_outputs.jsonl")
    write_prompt_audit(papers, audit_dir / "prompts.jsonl", top_k=top_k)
    export_scoring_edges_csv(
        scoring_edges(papers, judge_results, credit_mode=credit_mode),
        audit_dir / "scoring_edges.csv",
    )
    evaluation_report = evaluate_judge_results(judge_results, top_k=top_k)
    gold_report = evaluate_gold_reference_selection(papers, judge_results, top_k=top_k)
    gold_position_report = gold_candidate_position_report(papers)
    gold_baseline_report = gold_baseline_comparison_report(
        papers,
        judge_results,
        top_k=top_k,
    )
    current_scoring_config = scoring_config(credit_mode)
    context_report = context_coverage_report(papers)
    readiness_report = data_readiness_report(papers)
    external_readiness_report = external_evidence_readiness_report(papers)
    external_paper_coverage_rows = external_evidence_paper_coverage_rows(papers)
    baseline_report = summarize_baseline_comparison(
        scores.paper_scores,
        baselines,
        final_scores=_paper_detail_scores(scores.paper_score_details, "final_score"),
        top_n=100,
    )
    proxy_report = evaluate_proxy_quality(papers, judge_results, baselines, top_n=100)
    current_anomaly_report = anomaly_report(
        papers,
        judge_results,
        scores,
        credit_mode=credit_mode,
    )
    current_stability_report = stability_report(
        papers,
        judge_results,
        primary_credit_mode=credit_mode,
        top_n=100,
    )
    current_paper_stability_report = paper_stability_report(
        papers,
        judge_results,
        credit_mode=credit_mode,
        top_n=100,
    )
    current_paper_score_uncertainty_report = bootstrap_paper_score_report(
        papers,
        judge_results,
        credit_mode=credit_mode,
        top_n=100,
        samples=200,
        seed=13,
    )
    paper_explanations_report = paper_influence_explanations(
        papers,
        judge_results,
        credit_mode=credit_mode,
        top_n=20,
        max_edges_per_paper=5,
    )
    entity_explanations_report = entity_influence_explanations(
        papers,
        judge_results,
        credit_mode=credit_mode,
        top_n=20,
        max_edges_per_entity=5,
    )
    current_paper_evidence_report = paper_evidence_report(
        papers,
        judge_results,
        paper_score_details=scores.paper_score_details,
        paper_score_uncertainty=current_paper_score_uncertainty_report,
        credit_mode=credit_mode,
        top_n=100,
        max_edges_per_paper=5,
    )
    current_mvp_summary = mvp_summary_report(
        data_readiness=readiness_report,
        evaluation=evaluation_report,
        proxy_evaluation=proxy_report,
        anomaly=current_anomaly_report,
        stability=current_stability_report,
        scoring_config=current_scoring_config,
        paper_stability=current_paper_stability_report,
        paper_score_uncertainty=current_paper_score_uncertainty_report,
        context_coverage=context_report,
        ranking_preview=ranking_preview_from_scores(
            paper_scores=scores.paper_scores,
            author_scores=scores.author_scores,
            institution_scores=scores.institution_scores,
            field_scores=scores.field_scores,
            year_scores=scores.year_scores,
            top_n=10,
        ),
    )
    current_benchmark_summary = benchmark_summary_report(
        evaluation=evaluation_report,
        gold_evaluation=gold_report,
        gold_baseline_comparison=gold_baseline_report,
        data_readiness=readiness_report,
        scoring_config=current_scoring_config,
    )
    (reports_dir / "evaluation.json").write_text(
        json.dumps(evaluation_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "evaluation.md").write_text(
        render_evaluation_markdown(evaluation_report),
        encoding="utf-8",
    )
    (reports_dir / "gold_evaluation.json").write_text(
        json.dumps(gold_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "gold_evaluation.md").write_text(
        render_gold_evaluation_markdown(gold_report),
        encoding="utf-8",
    )
    (reports_dir / "gold_candidate_position.json").write_text(
        json.dumps(gold_position_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "gold_candidate_position.md").write_text(
        render_gold_candidate_position_markdown(gold_position_report),
        encoding="utf-8",
    )
    (reports_dir / "gold_baseline_comparison.json").write_text(
        json.dumps(gold_baseline_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "gold_baseline_comparison.md").write_text(
        render_gold_baseline_comparison_markdown(gold_baseline_report),
        encoding="utf-8",
    )
    (reports_dir / "scoring_config.json").write_text(
        json.dumps(current_scoring_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "context_coverage.json").write_text(
        json.dumps(context_report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_validation_outputs(current_validation_report, reports_dir)
    (reports_dir / "data_readiness.json").write_text(
        json.dumps(readiness_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "external_evidence_readiness.json").write_text(
        json.dumps(external_readiness_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "external_evidence_readiness.md").write_text(
        render_external_evidence_readiness_markdown(external_readiness_report),
        encoding="utf-8",
    )
    write_external_evidence_paper_coverage_csv(
        external_paper_coverage_rows,
        reports_dir / "external_evidence_paper_coverage.csv",
    )
    write_citation_completeness_audit(papers, reports_dir)
    (reports_dir / "baseline_comparison.json").write_text(
        json.dumps(baseline_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "proxy_evaluation.json").write_text(
        json.dumps(proxy_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "anomaly_report.json").write_text(
        json.dumps(current_anomaly_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "stability_report.json").write_text(
        json.dumps(current_stability_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "paper_stability_report.json").write_text(
        json.dumps(current_paper_stability_report, indent=2) + "\n",
        encoding="utf-8",
    )
    export_paper_stability_csv(
        current_paper_stability_report,
        reports_dir / "paper_stability_report.csv",
    )
    (reports_dir / "paper_score_uncertainty_report.json").write_text(
        json.dumps(current_paper_score_uncertainty_report, indent=2) + "\n",
        encoding="utf-8",
    )
    export_bootstrap_paper_score_csv(
        current_paper_score_uncertainty_report,
        reports_dir / "paper_score_uncertainty_report.csv",
    )
    (reports_dir / "paper_explanations.json").write_text(
        json.dumps(paper_explanations_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "entity_explanations.json").write_text(
        json.dumps(entity_explanations_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "paper_evidence_report.json").write_text(
        json.dumps(current_paper_evidence_report, indent=2) + "\n",
        encoding="utf-8",
    )
    export_paper_evidence_csv(
        current_paper_evidence_report,
        reports_dir / "paper_evidence_report.csv",
    )
    current_score_cards = score_cards_from_evidence_report(
        current_paper_evidence_report,
        top_n=20,
        max_edges=5,
    )
    (reports_dir / "score_cards.json").write_text(
        json.dumps(current_score_cards, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "score_cards.md").write_text(
        render_score_cards_markdown(current_score_cards),
        encoding="utf-8",
    )
    write_entity_resolution_report(papers, reports_dir)
    (reports_dir / "mvp_summary.json").write_text(
        json.dumps(current_mvp_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "mvp_summary.md").write_text(
        render_mvp_summary_markdown(current_mvp_summary),
        encoding="utf-8",
    )
    (reports_dir / "benchmark_summary.json").write_text(
        json.dumps(current_benchmark_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "benchmark_summary.md").write_text(
        render_benchmark_summary_markdown(current_benchmark_summary),
        encoding="utf-8",
    )
    export_finetuning_datasets(
        papers,
        judge_results,
        proxy_report,
        finetuning_dir,
        top_k=top_k,
        gold_report=gold_report,
    )
    _write_finetuning_preflight(finetuning_dir)
    export_llamafactory_configs(
        finetuning_dir,
        model_name_or_path=model or "local-impact-judge",
        output_root=output / "checkpoints",
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            build_run_manifest(
                input,
                top_k=top_k,
                limit=limit,
                credit_mode=credit_mode,
                judge_name=judge_name,
                base_url=base_url,
                model=model,
                temperature=temperature,
                repair_attempts=repair_attempts,
                max_tokens=max_tokens,
                judge_audit=judge_audit,
                normalized_papers=papers,
                serving_backend=_serving_backend_name(judge_name),
                model_path=model or None,
                validation_report=current_validation_report,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_project_readiness_with_artifact_index(output)
    typer.echo(f"Wrote run outputs to {output}")


@app.command()
def score(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    judge_audit: Path = typer.Option(..., "--judge-audit", exists=True, readable=True),
    top_k: int = typer.Option(5, "--top-k", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    credit_mode: str = typer.Option("rank_weighted", "--credit-mode"),
    strict_validation: bool = typer.Option(False, "--strict-validation"),
) -> None:
    run(
        input=input,
        output=output,
        top_k=top_k,
        limit=limit,
        credit_mode=credit_mode,
        judge_name="audit",
        base_url="",
        model="",
        api_key="EMPTY",
        timeout=120.0,
        temperature=0.0,
        repair_attempts=0,
        max_tokens=None,
        json_mode=False,
        judge_audit=judge_audit,
        strict_validation=strict_validation,
    )


@app.command("sciimpact-text-baseline")
def sciimpact_text_baseline(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    manifest = write_sciimpact_text_baseline_rankings(input, output)
    typer.echo(
        "Wrote SciImpact text baseline rankings for "
        f"{manifest['paper_count']} papers to {output}"
    )


@app.command("evaluate-sciimpact-predictions")
def evaluate_sciimpact_predictions_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    predictions: Path = typer.Option(..., "--predictions", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    calibration_report: Path | None = typer.Option(
        None,
        "--calibration-report",
        exists=True,
        readable=True,
    ),
) -> None:
    report = evaluate_sciimpact_predictions(
        sft,
        predictions,
        output,
        limit=limit,
        calibration_report_path=calibration_report,
    )
    typer.echo(
        "Wrote SciImpact prediction evaluation to "
        f"{output} (accuracy={report['summary']['accuracy']:.3f}, "
        f"coverage_rate={report['summary']['coverage_rate']:.3f}, "
        f"credibility={report.get('credibility_scorecard', {}).get('status', 'unknown')})"
    )


@app.command("summarize-sciimpact-model-credibility")
def summarize_sciimpact_model_credibility_cmd(
    eval: list[Path] = typer.Option(  # noqa: A002
        ...,
        "--eval",
        exists=True,
        readable=True,
        help="Evaluation directory or sciimpact_model_eval.json file.",
    ),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    report = write_sciimpact_model_credibility_summary(eval, output)
    summary = report["summary"]
    scorecard = report["credibility_scorecard"]
    typer.echo(
        "Wrote SciImpact model credibility summary to "
        f"{output} (evaluations={summary['evaluation_count']}, "
        f"pooled_accuracy={summary['pooled_accuracy']:.3f}, "
        f"mean_accuracy={summary['mean_accuracy']:.3f}, "
        f"status={scorecard.get('status', 'unknown')})"
    )


@app.command("compare-sciimpact-model-candidate")
def compare_sciimpact_model_candidate_cmd(
    baseline_summary: Path = typer.Option(
        ...,
        "--baseline-summary",
        exists=True,
        readable=True,
        help="Baseline sciimpact_model_credibility_summary.json or its directory.",
    ),
    candidate_summary: Path = typer.Option(
        ...,
        "--candidate-summary",
        exists=True,
        readable=True,
        help="Candidate sciimpact_model_credibility_summary.json or its directory.",
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    min_pooled_accuracy_delta: float = typer.Option(
        0.005,
        "--min-pooled-accuracy-delta",
        help="Minimum pooled accuracy gain required for promotion.",
    ),
    min_evaluations: int = typer.Option(
        3,
        "--min-evaluations",
        help="Minimum candidate evaluation count required for promotion.",
    ),
    max_min_seed_accuracy_regression: float = typer.Option(
        0.0,
        "--max-min-seed-accuracy-regression",
        help="Allowed regression in the candidate's worst-seed accuracy.",
    ),
    max_accuracy_range_increase: float = typer.Option(
        0.02,
        "--max-accuracy-range-increase",
        help="Allowed increase in seed-to-seed accuracy range.",
    ),
    max_weak_slice_count_increase: int = typer.Option(
        0,
        "--max-weak-slice-count-increase",
        help="Allowed increase in weak dimension/item-type slice count.",
    ),
    max_worst_slice_regression: float = typer.Option(
        0.02,
        "--max-worst-slice-regression",
        help="Allowed regression in worst eligible slice accuracy.",
    ),
) -> None:
    report = write_sciimpact_model_candidate_gate(
        baseline_summary,
        candidate_summary,
        output,
        min_pooled_accuracy_delta=min_pooled_accuracy_delta,
        min_evaluations=min_evaluations,
        max_min_seed_accuracy_regression=max_min_seed_accuracy_regression,
        max_accuracy_range_increase=max_accuracy_range_increase,
        max_weak_slice_count_increase=max_weak_slice_count_increase,
        max_worst_slice_regression=max_worst_slice_regression,
    )
    comparison = report["comparison"]
    typer.echo(
        "Wrote SciImpact model candidate gate to "
        f"{output} (decision={report['decision']}, "
        f"pooled_delta={comparison['pooled_accuracy_delta']:.3f}, "
        f"min_seed_delta={comparison['min_seed_accuracy_delta']:.3f})"
    )


@app.command("export-sciimpact-model-weak-slice-errors")
def export_sciimpact_model_weak_slice_errors_cmd(
    eval: list[Path] = typer.Option(  # noqa: A002
        ...,
        "--eval",
        exists=True,
        readable=True,
        help="Evaluation directory or sciimpact_model_eval.json file.",
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    sft: list[Path] | None = typer.Option(
        None,
        "--sft",
        exists=True,
        readable=True,
        help="SFT JSONL aligned with each --eval, used to export training JSONL.",
    ),
    prompt_style: str = typer.Option(
        "source-task",
        "--prompt-style",
        help="Training prompt style for exported SFT/DPO rows.",
    ),
    weak_slice_threshold: float = typer.Option(0.60, "--weak-slice-threshold"),
    training_source: bool = typer.Option(
        False,
        "--training-source/--diagnostic-only",
        help="Mark the source as safe training-split diagnostics.",
    ),
) -> None:
    report = write_sciimpact_model_weak_slice_error_pack(
        eval,
        output,
        weak_slice_threshold=weak_slice_threshold,
        training_source=training_source,
        sft_paths=sft,
        prompt_style=prompt_style,
    )
    manifest = report["manifest"]
    typer.echo(
        "Wrote SciImpact weak-slice error pack to "
        f"{output} (weak_slices={manifest['weak_slice_count']}, "
        f"errors={manifest['error_count']}, "
        f"use={manifest['recommended_use']})"
    )


@app.command("split-sciimpact-calibration")
def split_sciimpact_calibration_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    predictions: Path = typer.Option(..., "--predictions", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    calibration_fraction: float = typer.Option(0.5, "--calibration-fraction"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    try:
        manifest = write_sciimpact_calibration_split(
            sft,
            predictions,
            output,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        "Wrote SciImpact calibration split to "
        f"{output} (calibration={manifest['calibration_count']}, "
        f"validation={manifest['validation_count']})"
    )


@app.command("flip-sciimpact-sft")
def flip_sciimpact_sft_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
) -> None:
    manifest = write_sciimpact_flipped_sft(
        sft,
        output,
        limit=limit,
        skip_invalid=not strict,
    )
    typer.echo(
        "Wrote "
        f"{manifest['flipped_record_count']} flipped SciImpact examples to {output} "
        f"(skipped={manifest['skipped_record_count']})"
    )


@app.command("evaluate-sciimpact-flip-consistency")
def evaluate_sciimpact_flip_consistency_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    predictions: Path = typer.Option(..., "--predictions", exists=True, readable=True),
    flipped_predictions: Path = typer.Option(
        ...,
        "--flipped-predictions",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    confidence_margin: float = typer.Option(0.0, "--confidence-margin", min=0.0),
) -> None:
    report = evaluate_sciimpact_flip_consistency(
        sft,
        predictions,
        flipped_predictions,
        output,
        limit=limit,
        confidence_margin=confidence_margin,
    )
    summary = report["summary"]
    typer.echo(
        "Wrote SciImpact flip-consistency evaluation to "
        f"{output} (selected_accuracy={summary['selected_accuracy']:.3f}, "
        f"consistency_rate={summary['consistency_rate']:.3f})"
    )


@app.command("evaluate-sciimpact-index")
def evaluate_sciimpact_index_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    predictions: Path = typer.Option(..., "--predictions", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    report = write_sciimpact_index_evaluation(sft, predictions, output)
    summary = report["pairwise_evaluation"]["summary"]
    typer.echo(
        "Wrote SciImpact index evaluation to "
        f"{output} (pairwise_accuracy={summary['pairwise_accuracy']:.3f}, "
        f"coverage_rate={summary['coverage_rate']:.3f})"
    )


@app.command("export-sciimpact-pairwise-sft")
def export_sciimpact_pairwise_sft_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
    prompt_style: str = typer.Option(
        "generic-impact",
        "--prompt-style",
        help="Training prompt style: generic-impact or source-task.",
    ),
) -> None:
    manifest = export_sciimpact_pairwise_sft(
        input,
        output,
        limit=limit,
        skip_invalid=not strict,
        prompt_style=prompt_style.replace("-", "_"),
    )
    typer.echo(
        "Wrote "
        f"{manifest['record_count']} SciImpact pairwise SFT records to {output} "
        f"(skipped={manifest['skipped_record_count']})"
    )


@app.command("export-sciimpact-pairwise-dpo")
def export_sciimpact_pairwise_dpo_cmd(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
) -> None:
    manifest = export_sciimpact_pairwise_dpo(
        input,
        output,
        limit=limit,
        skip_invalid=not strict,
    )
    typer.echo(
        "Wrote "
        f"{manifest['record_count']} SciImpact pairwise DPO records to {output} "
        f"(skipped={manifest['skipped_record_count']})"
    )


@app.command("mine-sciimpact-hard-examples")
def mine_sciimpact_hard_examples_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    flip_rows: Path = typer.Option(..., "--flip-rows", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    max_examples: int | None = typer.Option(None, "--max-examples", min=1),
    include_wrong: bool = typer.Option(True, "--include-wrong/--exclude-wrong"),
    include_inconsistent: bool = typer.Option(
        True,
        "--include-inconsistent/--exclude-inconsistent",
    ),
    strict: bool = typer.Option(False, "--strict/--skip-invalid"),
) -> None:
    manifest = mine_sciimpact_hard_pairwise_examples(
        sft,
        flip_rows,
        output,
        max_examples=max_examples,
        include_wrong=include_wrong,
        include_inconsistent=include_inconsistent,
        skip_invalid=not strict,
    )
    typer.echo(
        "Wrote "
        f"{manifest['record_count']} SciImpact hard pairwise examples to {output} "
        f"(skipped={manifest['skipped_record_count']})"
    )


@app.command("export-sciimpact-pairwise-sft-lora-config")
def export_sciimpact_pairwise_sft_lora_config_cmd(
    train_file: Path = typer.Option(..., "--train-file", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    model_name_or_path: str = typer.Option(..., "--model-name-or-path"),
    checkpoint_output: Path = typer.Option(..., "--checkpoint-output"),
    max_samples: int | None = typer.Option(None, "--max-samples", min=1),
) -> None:
    manifest = export_sciimpact_pairwise_sft_llamafactory_configs(
        train_file,
        output,
        model_name_or_path=model_name_or_path,
        output_dir=checkpoint_output,
        max_samples=max_samples,
    )
    typer.echo(
        "Wrote SciImpact pairwise SFT LoRA config to "
        f"{manifest['train_config_path']}"
    )


@app.command("sample-sciimpact-sft")
def sample_sciimpact_sft_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    per_dimension: int = typer.Option(100, "--per-dimension", min=1),
    seed: int = typer.Option(0, "--seed"),
    dimension: list[str] | None = typer.Option(None, "--dimension"),
    dimension_weight: list[str] | None = typer.Option(None, "--dimension-weight"),
    item_type_weight: list[str] | None = typer.Option(None, "--item-type-weight"),
) -> None:
    manifest = write_sciimpact_stratified_sample(
        sft,
        output,
        per_dimension=per_dimension,
        seed=seed,
        dimensions=dimension,
        dimension_weights=_parse_dimension_weights(dimension_weight or []),
        item_type_weights=_parse_item_type_weights(item_type_weight or []),
    )
    typer.echo(
        "Wrote "
        f"{manifest['sample_count']} stratified SciImpact examples to {output}"
    )


def _parse_dimension_weights(items: list[str]) -> dict[str, int]:
    weights: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(
                "--dimension-weight must use DIMENSION=WEIGHT, for example patent=3"
            )
        dimension, raw_weight = item.split("=", 1)
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid dimension weight for {dimension}: {raw_weight}"
            ) from exc
        if weight < 1:
            raise typer.BadParameter("dimension weights must be positive integers")
        weights[dimension] = weight
    return weights


def _parse_item_type_weights(items: list[str]) -> dict[str, int]:
    weights: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(
                "--item-type-weight must use ITEM_TYPE=WEIGHT, for example github=3"
            )
        item_type, raw_weight = item.split("=", 1)
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid item type weight for {item_type}: {raw_weight}"
            ) from exc
        if weight < 1:
            raise typer.BadParameter("item type weights must be positive integers")
        weights[item_type] = weight
    return weights


@app.command("generate-sciimpact-predictions")
def generate_sciimpact_predictions_cmd(
    sft: Path = typer.Option(..., "--sft", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--base-url"),
    model: str = typer.Option(..., "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    timeout: float = typer.Option(120.0, "--timeout"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int | None = typer.Option(32, "--max-tokens", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    resume: bool = typer.Option(True, "--resume/--overwrite"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    answer_format: str = typer.Option("freeform", "--answer-format"),
) -> None:
    manifest = generate_sciimpact_predictions(
        sft,
        output,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        answer_format=answer_format,
    )
    typer.echo(
        "Wrote "
        f"{manifest['written_prediction_count']} SciImpact predictions to {output} "
        f"(skipped_existing={manifest['skipped_existing_count']})"
    )


def _build_judge(
    judge_name: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    repair_attempts: int,
    max_tokens: int | None,
    json_mode: bool,
) -> Judge:
    if judge_name == "heuristic":
        return HeuristicJudge()
    if judge_name == "openai-compatible":
        if not model:
            raise typer.BadParameter("--model is required when --judge openai-compatible")
        return OpenAICompatibleJudge(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            repair_attempts=repair_attempts,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    raise typer.BadParameter("--judge must be 'heuristic' or 'openai-compatible'")


def _write_validation_outputs(report: dict, reports_dir: Path) -> None:
    (reports_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "validation.md").write_text(
        render_validation_markdown(report),
        encoding="utf-8",
    )


def _write_project_readiness_with_artifact_index(run_dir: Path) -> dict:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact_index(run_dir)
    report = write_project_readiness_report(run_dir)
    _write_artifact_index(run_dir)
    report = write_project_readiness_report(run_dir)
    _write_artifact_index(run_dir)
    return report


def _impact_claim_default_output(run_dir: Path) -> Path:
    if (run_dir / "reports").exists():
        return run_dir / "reports"
    return run_dir


def _write_artifact_index(run_dir: Path) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "artifact_index.json").write_text(
        json.dumps(
            artifact_index(
                run_dir,
                RUN_ARTIFACTS,
                optional_paths=OPTIONAL_RUN_ARTIFACTS,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_finetuning_preflight(finetuning_dir: Path) -> dict:
    readiness_path = finetuning_dir / "finetuning_readiness.json"
    dpo_path = finetuning_dir / "merged_dpo_preferences.jsonl"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    loaded = load_preference_examples_with_report([dpo_path])
    report = _finetuning_preflight_report(readiness, loaded.report)
    (finetuning_dir / "finetuning_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (finetuning_dir / "finetuning_preflight.md").write_text(
        _render_finetuning_preflight_markdown(report),
        encoding="utf-8",
    )
    return report


def _finetuning_preflight_report(
    readiness: dict,
    dpo_data_validation: dict,
) -> dict:
    readiness_gates = readiness.get("readiness", {})
    recommendation = readiness.get("recommendation", {})
    recommended_next_step = recommendation.get("next_step", "collect_more_examples")
    can_start_dpo_smoke = (
        recommended_next_step in {"dpo_smoke", "dpo_train"}
        and bool(readiness_gates.get("ready_for_dpo_smoke"))
        and not bool(readiness_gates.get("has_training_data_quality_blockers"))
        and dpo_data_validation.get("loaded_example_count", 0) >= 10
        and dpo_data_validation.get("invalid_record_count", 0) == 0
        and dpo_data_validation.get("reverse_conflict_count", 0) == 0
    )
    can_start_sft_smoke = (
        recommended_next_step in {"sft_smoke", "dpo_smoke", "dpo_train"}
        and bool(readiness_gates.get("ready_for_sft_smoke"))
        and not bool(readiness_gates.get("has_training_data_quality_blockers"))
    )
    return {
        "readiness_summary": readiness.get("summary", {}),
        "readiness_gates": readiness_gates,
        "readiness_recommendation": recommendation,
        "dpo_data_validation": dpo_data_validation,
        "decision": {
            "recommended_next_step": recommended_next_step,
            "can_start_sft_smoke": can_start_sft_smoke,
            "can_start_dpo_smoke": can_start_dpo_smoke,
            "can_start_full_dpo": (
                can_start_dpo_smoke
                and bool(readiness_gates.get("ready_for_full_dpo"))
            ),
        },
        "limitations": [
            "This preflight checks training data readiness only; it does not evaluate model quality.",
            "Run dev-set model comparison before promoting a trained adapter to test or production.",
        ],
    }


def _render_finetuning_preflight_markdown(report: dict) -> str:
    decision = report.get("decision", {})
    validation = report.get("dpo_data_validation", {})
    recommendation = report.get("readiness_recommendation", {})
    lines = [
        "# Finetuning Preflight",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| recommended_next_step | {decision.get('recommended_next_step', '')} |",
        f"| can_start_sft_smoke | {_yes_no_bool(decision.get('can_start_sft_smoke'))} |",
        f"| can_start_dpo_smoke | {_yes_no_bool(decision.get('can_start_dpo_smoke'))} |",
        f"| can_start_full_dpo | {_yes_no_bool(decision.get('can_start_full_dpo'))} |",
        f"| reason | {recommendation.get('reason', '')} |",
        "",
        "## DPO Data Validation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| raw_record_count | {validation.get('raw_record_count', 0)} |",
        f"| loaded_example_count | {validation.get('loaded_example_count', 0)} |",
        f"| invalid_record_count | {validation.get('invalid_record_count', 0)} |",
        f"| duplicate_pair_count | {validation.get('duplicate_pair_count', 0)} |",
        f"| reverse_conflict_count | {validation.get('reverse_conflict_count', 0)} |",
        "",
    ]
    return "\n".join(lines)


def _load_tokenizer_for_inspection(model: str):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on training env.
        raise typer.BadParameter(
            "transformers is required for inspect-sft-tokenization"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    return tokenizer


def _render_sft_tokenization_report_markdown(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        "# SFT Tokenization Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| loaded_example_count | {summary.get('loaded_example_count', 0)} |",
        f"| inspected_example_count | {summary.get('inspected_example_count', 0)} |",
        f"| truncated_example_count | {summary.get('truncated_example_count', 0)} |",
        f"| truncated_example_rate | {summary.get('truncated_example_rate', 0.0):.6f} |",
        f"| mean_prompt_retention_rate | {summary.get('mean_prompt_retention_rate', 0.0):.6f} |",
        f"| prefix_retained_rate | {summary.get('prefix_retained_rate', 0.0):.6f} |",
        f"| suffix_retained_rate | {summary.get('suffix_retained_rate', 0.0):.6f} |",
        f"| response_fully_retained_rate | {summary.get('response_fully_retained_rate', 0.0):.6f} |",
        "",
        "## Example Preview",
        "",
        "| # | Prompt tokens | Response tokens | Retained prompt | Truncated | Prefix | Suffix | Response |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in report.get("examples", []):
        lines.append(
            "| "
            f"{row.get('example_index')} | "
            f"{row.get('prompt_token_count')} | "
            f"{row.get('response_token_count')} | "
            f"{row.get('retained_prompt_token_count')} | "
            f"{_yes_no_bool(row.get('truncated'))} | "
            f"{_yes_no_bool(row.get('prefix_retained'))} | "
            f"{_yes_no_bool(row.get('suffix_retained'))} | "
            f"{_yes_no_bool(row.get('response_fully_retained'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _yes_no_bool(value: object) -> str:
    return "yes" if bool(value) else "no"


def _serving_backend_name(judge_name: str) -> str | None:
    if judge_name == "openai-compatible":
        return "openai-compatible"
    if judge_name == "heuristic":
        return None
    return judge_name


def _judge_papers(
    papers,
    judge_impl: Judge,
    top_k: int,
    concurrency: int,
    max_retries: int,
) -> list[JudgeResult]:
    if concurrency == 1:
        for paper in papers:
            yield _judge_one(paper, judge_impl, top_k, max_retries)
        return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_judge_one, paper, judge_impl, top_k, max_retries)
            for paper in papers
        ]
        for future in as_completed(futures):
            yield future.result()


def _judge_one(paper, judge_impl: Judge, top_k: int, max_retries: int) -> JudgeResult:
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            return judge_impl.judge(paper, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - per-paper isolation is intentional.
            last_error = exc
    return with_judge_audit_metadata(
        error_result(paper.paper_id, str(last_error) if last_error else "unknown error"),
        paper,
        top_k=top_k,
    )


def _write_judge_audit(results: list[JudgeResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(judge_result_to_record(result)) + "\n")


def _paper_metadata_index(papers) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for paper in papers:
        metadata[paper.paper_id] = {"field": paper.field, "year": paper.year}
        for reference in paper.references:
            if reference.reference_id:
                metadata[reference.reference_id] = {
                    "field": reference.field,
                    "year": reference.year,
                }
    return metadata


def _paper_detail_scores(
    paper_score_details: dict[str, dict[str, object]] | None,
    field: str,
) -> dict[str, float]:
    if not paper_score_details:
        return {}
    scores: dict[str, float] = {}
    for identifier, details in paper_score_details.items():
        value = details.get(field)
        if value is None:
            continue
        try:
            scores[identifier] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


if __name__ == "__main__":
    app()
