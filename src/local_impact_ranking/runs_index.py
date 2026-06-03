from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row


FIELDNAMES = [
    "run",
    "paper_count",
    "judge_name",
    "judge_model",
    "readiness_status",
    "evidence_level",
    "hit_rate",
    "mean_ndcg",
    "judge_ndcg_delta_vs_best_baseline",
    "statistically_positive_ndcg_delta",
    "has_report_bundle",
]


def runs_index_report(runs_root: str | Path) -> dict[str, Any]:
    root = Path(runs_root)
    rows = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        row = _run_row(child)
        if row is not None:
            rows.append(row)
    return {"runs_root": str(root), "run_count": len(rows), "runs": rows}


def write_runs_index(runs_root: str | Path, output: str | Path) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    report = runs_index_report(runs_root)
    (destination / "index.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (destination / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sanitize_csv_row(row) for row in report["runs"])
    (destination / "index.md").write_text(
        render_runs_index_markdown(report),
        encoding="utf-8",
    )
    return report


def render_runs_index_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Runs Index",
        "",
        f"Runs root: `{report.get('runs_root', '')}`",
        "",
        "| Run | Papers | Judge | Model | Readiness | Evidence | Hit | NDCG | Baseline delta | Bundle |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    rows = report.get("runs", [])
    if not rows:
        lines.append("|  |  |  |  |  |  |  |  |  |  |")
    else:
        for row in rows:
            lines.append(
                f"| {row.get('run', '')} | "
                f"{row.get('paper_count', 0)} | "
                f"{_text(row.get('judge_name'))} | "
                f"{_text(row.get('judge_model'))} | "
                f"{_text(row.get('readiness_status'))} | "
                f"{_text(row.get('evidence_level'))} | "
                f"{_number(row.get('hit_rate'))} | "
                f"{_number(row.get('mean_ndcg'))} | "
                f"{_number(row.get('judge_ndcg_delta_vs_best_baseline'))} | "
                f"{_yes_no(row.get('has_report_bundle'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _run_row(run_dir: Path) -> dict[str, Any] | None:
    benchmark = _read_json(run_dir / "reports" / "benchmark_summary.json")
    readiness = _read_json(run_dir / "reports" / "project_readiness.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    if not benchmark and not readiness and not manifest:
        return None
    gold = benchmark.get("gold_quality", {})
    baseline = benchmark.get("gold_baseline_comparison", {})
    judge = manifest.get("judge", {})
    readiness_decision = readiness.get("decision", {})
    readiness_quality = readiness.get("quality", {})
    return {
        "run": run_dir.name,
        "paper_count": benchmark.get("benchmark", {}).get(
            "paper_count",
            manifest.get("input", {}).get("normalized_record_count", 0),
        ),
        "judge_name": judge.get("name", ""),
        "judge_model": judge.get("model", ""),
        "readiness_status": readiness_decision.get("status", ""),
        "evidence_level": readiness_quality.get("evidence_level", ""),
        "hit_rate": gold.get("hit_rate", 0.0),
        "mean_ndcg": gold.get("mean_ndcg", 0.0),
        "judge_ndcg_delta_vs_best_baseline": baseline.get(
            "judge_mean_ndcg_delta_vs_best_baseline",
            0.0,
        ),
        "statistically_positive_ndcg_delta": baseline.get(
            "statistically_positive_ndcg_delta",
            False,
        ),
        "has_report_bundle": (run_dir / "report_bundle" / "bundle_manifest.json").exists(),
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


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
