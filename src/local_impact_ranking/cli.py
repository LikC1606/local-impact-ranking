from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer

from local_impact_ranking.audit import (
    append_judge_result,
    completed_paper_ids,
    judge_result_to_record,
    load_judge_results,
    write_prompt_audit,
)
from local_impact_ranking.anomalies import anomaly_report
from local_impact_ranking.artifacts import RUN_ARTIFACTS, artifact_index
from local_impact_ranking.benchmark_compare import (
    write_benchmark_comparison,
    write_gold_target_comparison,
    write_model_selection_decision,
)
from local_impact_ranking.benchmark_status import (
    benchmark_status_report,
    render_benchmark_status_markdown,
)
from local_impact_ranking.benchmarks import get_benchmark, list_benchmarks
from local_impact_ranking.baselines import (
    compute_bibliometric_baselines,
    export_baseline_rankings,
    summarize_baseline_comparison,
)
from local_impact_ranking.contexts import context_coverage_report
from local_impact_ranking.data_readiness import data_readiness_report
from local_impact_ranking.evaluate import evaluate_judge_results, render_evaluation_markdown
from local_impact_ranking.error_analysis import (
    gold_error_analysis_report,
    render_gold_error_analysis_markdown,
)
from local_impact_ranking.explanations import (
    entity_influence_explanations,
    paper_influence_explanations,
)
from local_impact_ranking.export import (
    export_human_audit_sample,
    export_rankings_csv,
    export_scoring_edges_csv,
)
from local_impact_ranking.finetune_export import export_finetuning_datasets
from local_impact_ranking.dpo_lora_train import run_dpo_lora_training
from local_impact_ranking.gold_evaluation import (
    evaluate_gold_reference_selection,
    render_gold_evaluation_markdown,
)
from local_impact_ranking.io import convert_papers_csv_to_jsonl, load_papers_jsonl
from local_impact_ranking.judge import (
    HeuristicJudge,
    Judge,
    JudgeResult,
    OpenAICompatibleJudge,
    error_result,
    with_judge_audit_metadata,
)
from local_impact_ranking.llamafactory_export import export_llamafactory_configs
from local_impact_ranking.manifest import build_run_manifest, sha256_file
from local_impact_ranking.normalize import normalize_paper
from local_impact_ranking.proxy_evaluation import evaluate_proxy_quality
from local_impact_ranking.scoring import scoring_edges, score_influence, scoring_config
from local_impact_ranking.scicite_adapter import convert_scicite_jsonl
from local_impact_ranking.prescience_adapter import convert_prescience_file
from local_impact_ranking.splits import write_paper_splits
from local_impact_ranking.stability import stability_report
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
    max_length: int = typer.Option(1536, "--max-length", min=1),
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
    )
    typer.echo(f"Wrote DPO adapter to {output}")


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
        typer.echo(f"{spec.name} - {spec.purpose}")


@benchmarks_app.command("show")
def benchmarks_show(name: str = typer.Argument(...)) -> None:
    spec = get_benchmark(name)
    typer.echo(
        json.dumps(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "pipeline_layers": list(spec.pipeline_layers),
                "preferred_input": spec.preferred_input,
                "notes": spec.notes,
            },
            indent=2,
        )
    )


@benchmarks_app.command("convert-scicite")
def benchmarks_convert_scicite(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    count = convert_scicite_jsonl(input, output)
    typer.echo(f"Wrote {count} converted papers to {output}")


@benchmarks_app.command("convert-prescience")
def benchmarks_convert_prescience(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    max_candidates: int = typer.Option(50, "--max-candidates", min=1),
) -> None:
    count = convert_prescience_file(
        input,
        output,
        limit=limit,
        max_candidates=max_candidates,
    )
    typer.echo(f"Wrote {count} converted papers to {output}")


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
    scores = score_influence(papers, judge_results, credit_mode=credit_mode)

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

    export_rankings_csv(scores, rankings_dir)
    baselines = compute_bibliometric_baselines(papers)
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
    current_scoring_config = scoring_config(credit_mode)
    context_report = context_coverage_report(papers)
    readiness_report = data_readiness_report(papers)
    baseline_report = summarize_baseline_comparison(scores.paper_scores, baselines, top_n=100)
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
    current_mvp_summary = mvp_summary_report(
        data_readiness=readiness_report,
        evaluation=evaluation_report,
        proxy_evaluation=proxy_report,
        anomaly=current_anomaly_report,
        stability=current_stability_report,
        scoring_config=current_scoring_config,
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
    (reports_dir / "paper_explanations.json").write_text(
        json.dumps(paper_explanations_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "entity_explanations.json").write_text(
        json.dumps(entity_explanations_report, indent=2) + "\n",
        encoding="utf-8",
    )
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
    (reports_dir / "artifact_index.json").write_text(
        json.dumps(artifact_index(output, RUN_ARTIFACTS), indent=2) + "\n",
        encoding="utf-8",
    )
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


if __name__ == "__main__":
    app()
