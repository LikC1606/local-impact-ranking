import csv
import json

from local_impact_ranking.runs_index import (
    render_runs_index_markdown,
    runs_index_report,
    write_runs_index,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_run(root, name, *, papers, hit_rate, ndcg, status, has_bundle=False):
    run = root / name
    _write_json(
        run / "reports" / "benchmark_summary.json",
        {
            "benchmark": {"paper_count": papers, "judge_results": papers},
            "gold_quality": {"hit_rate": hit_rate, "mean_ndcg": ndcg},
            "gold_baseline_comparison": {
                "judge_mean_ndcg_delta_vs_best_baseline": 0.02,
                "judge_ndcg_delta_ci95": [0.01, 0.03],
                "statistically_positive_ndcg_delta": True,
            },
            "decision": {"needs_prompt_or_model_tuning": False},
        },
    )
    _write_json(
        run / "reports" / "project_readiness.json",
        {
            "decision": {"status": status, "can_report": status != "blocked"},
            "quality": {"evidence_level": "gold_validated"},
        },
    )
    _write_json(
        run / "run_manifest.json",
        {
            "judge": {"name": "heuristic", "model": "idf"},
            "scoring": {"credit_mode": "rank_weighted"},
        },
    )
    if has_bundle:
        _write_json(run / "report_bundle" / "bundle_manifest.json", {"ok": True})


def test_runs_index_report_summarizes_run_directories(tmp_path):
    _write_run(tmp_path, "run-a", papers=10, hit_rate=0.7, ndcg=0.5, status="reportable")
    _write_run(
        tmp_path,
        "run-b",
        papers=30,
        hit_rate=0.8,
        ndcg=0.6,
        status="needs_improvement",
        has_bundle=True,
    )
    (tmp_path / "not-a-run").mkdir()

    report = runs_index_report(tmp_path)

    assert report["run_count"] == 2
    assert [row["run"] for row in report["runs"]] == ["run-a", "run-b"]
    assert report["runs"][1]["paper_count"] == 30
    assert report["runs"][1]["has_report_bundle"] is True
    assert report["runs"][1]["readiness_status"] == "needs_improvement"


def test_write_runs_index_exports_csv_and_markdown(tmp_path):
    _write_run(tmp_path, "run-a", papers=10, hit_rate=0.7, ndcg=0.5, status="reportable")
    output = tmp_path / "index"

    write_runs_index(tmp_path, output)

    rows = list(csv.DictReader((output / "index.csv").read_text().splitlines()))
    markdown = (output / "index.md").read_text(encoding="utf-8")
    payload = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert rows[0]["run"] == "run-a"
    assert rows[0]["paper_count"] == "10"
    assert "# Runs Index" in markdown
    assert "| run-a | 10 | heuristic | idf | reportable |" in markdown
    assert payload["run_count"] == 1


def test_render_runs_index_markdown_handles_empty_index():
    markdown = render_runs_index_markdown({"run_count": 0, "runs": []})

    assert "# Runs Index" in markdown
    assert "|  |  |  |  |  |  |  |  |  |  |" in markdown
