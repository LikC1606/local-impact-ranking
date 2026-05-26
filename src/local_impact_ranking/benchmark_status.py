from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def benchmark_status_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    input_path = root / "prescience.papers.jsonl"
    audit_path = root / "audit" / "judge_outputs.jsonl"
    summary_path = root / "reports" / "benchmark_summary.json"
    input_papers = _count_jsonl_records(input_path)
    audit_records = _count_jsonl_records(audit_path)
    remaining = max(input_papers - audit_records, 0)
    report: dict[str, Any] = {
        "run": root.name,
        "run_dir": str(root),
        "input_path": str(input_path),
        "audit_path": str(audit_path),
        "input_papers": input_papers,
        "audit_records": audit_records,
        "remaining": remaining,
        "completion_rate": (audit_records / input_papers) if input_papers else 0.0,
        "has_benchmark_summary": summary_path.exists(),
        "metrics": {},
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        judge_quality = summary.get("judge_quality", {})
        gold_quality = summary.get("gold_quality", {})
        decision = summary.get("decision", {})
        report["metrics"] = {
            "parse_success_rate": judge_quality.get("parse_success_rate", 0.0),
            "invalid_selection_rate": judge_quality.get("invalid_selection_rate", 0.0),
            "hit_rate": gold_quality.get("hit_rate", 0.0),
            "mean_ndcg": gold_quality.get("mean_ndcg", 0.0),
            "usable_for_benchmark_report": decision.get(
                "usable_for_benchmark_report",
                False,
            ),
        }
    return report


def render_benchmark_status_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# Benchmark Status",
        "",
        "## Progress",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| run | {report.get('run', '')} |",
        f"| input_papers | {report.get('input_papers', 0)} |",
        f"| audit_records | {report.get('audit_records', 0)} |",
        f"| remaining | {report.get('remaining', 0)} |",
        f"| completion_rate | {_number(report.get('completion_rate'))} |",
        f"| has_benchmark_summary | {_yes_no(report.get('has_benchmark_summary'))} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| parse_success_rate | {_number(metrics.get('parse_success_rate'))} |",
        f"| invalid_selection_rate | {_number(metrics.get('invalid_selection_rate'))} |",
        f"| hit_rate | {_number(metrics.get('hit_rate'))} |",
        f"| mean_ndcg | {_number(metrics.get('mean_ndcg'))} |",
        f"| usable_for_benchmark_report | {_yes_no(metrics.get('usable_for_benchmark_report'))} |",
        "",
    ]
    return "\n".join(lines)


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"
