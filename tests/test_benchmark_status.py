import json

from local_impact_ranking.benchmark_status import (
    benchmark_status_report,
    render_benchmark_status_markdown,
)


def write_papers(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "paper_id": f"p{index}",
                        "title": f"Paper {index}",
                        "references": [{"reference_id": f"r{index}", "title": "Reference"}],
                    }
                )
                + "\n"
            )


def write_audit(path, paper_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper_id in paper_ids:
            handle.write(
                json.dumps(
                    {
                        "target_paper_id": paper_id,
                        "parse_error": None,
                        "error": None,
                        "selected_references": [],
                        "raw_output": "{}",
                    }
                )
                + "\n"
            )


def test_benchmark_status_report_counts_input_audit_and_remaining(tmp_path):
    run_dir = tmp_path / "run"
    write_papers(run_dir / "prescience.papers.jsonl", 5)
    write_audit(run_dir / "audit" / "judge_outputs.jsonl", ["p0", "p1"])

    report = benchmark_status_report(run_dir)

    assert report["run"] == "run"
    assert report["input_papers"] == 5
    assert report["audit_records"] == 2
    assert report["remaining"] == 3
    assert report["completion_rate"] == 0.4
    assert report["has_benchmark_summary"] is False


def test_benchmark_status_report_includes_summary_metrics_when_available(tmp_path):
    run_dir = tmp_path / "run"
    write_papers(run_dir / "prescience.papers.jsonl", 2)
    write_audit(run_dir / "audit" / "judge_outputs.jsonl", ["p0", "p1"])
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "judge_quality": {
                    "parse_success_rate": 1.0,
                    "invalid_selection_rate": 0.0,
                },
                "gold_quality": {
                    "hit_rate": 0.86,
                    "mean_ndcg": 0.72,
                },
                "decision": {"usable_for_benchmark_report": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = benchmark_status_report(run_dir)

    assert report["has_benchmark_summary"] is True
    assert report["metrics"]["parse_success_rate"] == 1.0
    assert report["metrics"]["hit_rate"] == 0.86
    assert report["metrics"]["mean_ndcg"] == 0.72
    assert report["metrics"]["usable_for_benchmark_report"] is True


def test_render_benchmark_status_markdown_contains_progress_table():
    markdown = render_benchmark_status_markdown(
        {
            "run": "run",
            "input_papers": 5,
            "audit_records": 2,
            "remaining": 3,
            "completion_rate": 0.4,
            "has_benchmark_summary": True,
            "metrics": {
                "parse_success_rate": 1.0,
                "invalid_selection_rate": 0.0,
                "hit_rate": 0.86,
                "mean_ndcg": 0.72,
                "usable_for_benchmark_report": True,
            },
        }
    )

    assert markdown.startswith("# Benchmark Status\n")
    assert "| run | run |" in markdown
    assert "| completion_rate | 0.400 |" in markdown
    assert "| hit_rate | 0.860 |" in markdown
    assert "| usable_for_benchmark_report | yes |" in markdown
