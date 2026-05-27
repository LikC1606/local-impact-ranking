from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


REPORT_BUNDLE_FILES = [
    "rankings/papers.csv",
    "rankings/authors.csv",
    "rankings/institutions.csv",
    "rankings/fields.csv",
    "rankings/years.csv",
    "reports/mvp_summary.md",
    "reports/benchmark_summary.json",
    "reports/benchmark_summary.md",
    "reports/project_readiness.json",
    "reports/project_readiness.md",
    "reports/evaluation.md",
    "reports/gold_evaluation.json",
    "reports/gold_evaluation.md",
    "reports/gold_baseline_comparison.json",
    "reports/gold_baseline_comparison.md",
    "reports/baseline_comparison.json",
    "reports/paper_evidence_report.csv",
    "reports/score_cards.md",
    "reports/entity_resolution_report.md",
    "reports/paper_stability_report.csv",
    "reports/paper_score_uncertainty_report.csv",
    "reports/scoring_config.json",
    "reports/artifact_index.json",
    "run_manifest.json",
]


def write_report_bundle(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    missing_optional_files: list[str] = []
    for relative_path in REPORT_BUNDLE_FILES:
        source_path = source / relative_path
        destination_path = destination / relative_path
        if not source_path.exists() or not source_path.is_file():
            missing_optional_files.append(relative_path)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied_files.append(relative_path)

    readiness = _read_json(source / "reports" / "project_readiness.json")
    benchmark = _read_json(source / "reports" / "benchmark_summary.json")
    manifest = {
        "run_dir": str(source),
        "claim_scope": readiness.get(
            "claim_scope",
            "dataset-local content-aware impact ranking",
        ),
        "readiness": _readiness_summary(readiness),
        "benchmark": _benchmark_summary(benchmark),
        "warnings": readiness.get("warnings", []),
        "copied_files": copied_files,
        "missing_optional_files": missing_optional_files,
        "bundle_files": list(REPORT_BUNDLE_FILES),
    }
    (destination / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        render_report_bundle_readme(manifest),
        encoding="utf-8",
    )
    return manifest


def render_report_bundle_readme(manifest: dict[str, Any]) -> str:
    readiness = manifest.get("readiness", {})
    benchmark = manifest.get("benchmark", {})
    lines = [
        "# Local Impact Ranking Report Bundle",
        "",
        "This bundle contains report-facing artifacts for a single local impact ranking run.",
        "",
        "## Claim Boundary",
        "",
        f"- Claim scope: {manifest.get('claim_scope', 'dataset-local impact ranking')}",
        "- Scores and rankings are valid only inside the input dataset unless an external calibrated graph is added.",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| readiness_status | {_text(readiness.get('status'))} |",
        f"| can_report | {_yes_no(readiness.get('can_report'))} |",
        f"| evidence_level | {_text(readiness.get('evidence_level'))} |",
        f"| reason | {_text(readiness.get('reason'))} |",
        "",
        "## Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {benchmark.get('paper_count', 0)} |",
        f"| hit_rate | {_number(benchmark.get('hit_rate'))} |",
        f"| mean_ndcg | {_number(benchmark.get('mean_ndcg'))} |",
        f"| judge_ndcg_delta_vs_best_baseline | {_number(benchmark.get('judge_ndcg_delta_vs_best_baseline'))} |",
        f"| judge_ndcg_delta_ci95 | {_ci(benchmark.get('judge_ndcg_delta_ci95'))} |",
        f"| statistically_positive_ndcg_delta | {_yes_no(benchmark.get('statistically_positive_ndcg_delta'))} |",
        f"| needs_prompt_or_model_tuning | {_yes_no(benchmark.get('needs_prompt_or_model_tuning'))} |",
        "",
        "## Included Files",
        "",
    ]
    copied_files = manifest.get("copied_files", [])
    if copied_files:
        lines.extend(f"- `{path}`" for path in copied_files)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Optional Files", ""])
    missing_files = manifest.get("missing_optional_files", [])
    if missing_files:
        lines.extend(f"- `{path}`" for path in missing_files)
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _readiness_summary(readiness: dict[str, Any]) -> dict[str, Any]:
    decision = readiness.get("decision", {})
    quality = readiness.get("quality", {})
    return {
        "status": decision.get("status"),
        "can_report": decision.get("can_report"),
        "reason": decision.get("reason"),
        "evidence_level": quality.get("evidence_level"),
    }


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    gold = benchmark.get("gold_quality", {})
    baseline = benchmark.get("gold_baseline_comparison", {})
    decision = benchmark.get("decision", {})
    return {
        "paper_count": benchmark.get("benchmark", {}).get("paper_count", 0),
        "hit_rate": gold.get("hit_rate", 0.0),
        "mean_ndcg": gold.get("mean_ndcg", 0.0),
        "judge_ndcg_delta_vs_best_baseline": baseline.get(
            "judge_mean_ndcg_delta_vs_best_baseline",
            0.0,
        ),
        "judge_ndcg_delta_ci95": baseline.get("judge_ndcg_delta_ci95", [0.0, 0.0]),
        "statistically_positive_ndcg_delta": baseline.get(
            "statistically_positive_ndcg_delta",
            False,
        ),
        "needs_prompt_or_model_tuning": decision.get(
            "needs_prompt_or_model_tuning",
            False,
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _ci(value: Any) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
