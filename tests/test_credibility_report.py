import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.credibility_report import (
    credibility_report,
    render_credibility_report_markdown,
    write_credibility_report,
)


def write_model_run(root, model_name, seed, direct_accuracy, selected_accuracy, *, by_dimension=None, by_item_type=None):
    run_dir = root / model_name / f"seed{seed}"
    eval_dir = run_dir / "eval"
    flip_dir = run_dir / "flip"
    eval_dir.mkdir(parents=True)
    flip_dir.mkdir(parents=True)
    correct = round(direct_accuracy * 300)
    selected_correct = round(selected_accuracy * 300)
    eval_report = {
        "summary": {
            "total_examples": 300,
            "covered_examples": 300,
            "correct_examples": correct,
            "wrong_examples": 300 - correct,
            "coverage_rate": 1.0,
            "accuracy": direct_accuracy,
        }
    }
    flip_report = {
        "summary": {
            "total_examples": 300,
            "original_accuracy": direct_accuracy,
            "flipped_mapped_accuracy": selected_accuracy - 0.01,
            "selected_accuracy": selected_accuracy,
            "selected_covered_examples": 300,
            "consistency_rate": 0.62,
            "confidence_margin": 0.05,
        },
        "by_dimension": by_dimension or {"citation": {"selected_accuracy": selected_accuracy}},
        "by_item_type": by_item_type or {"paper": {"selected_accuracy": selected_accuracy}},
    }
    (eval_dir / "sciimpact_model_eval.json").write_text(
        json.dumps(eval_report) + "\n",
        encoding="utf-8",
    )
    (flip_dir / "sciimpact_flip_consistency.json").write_text(
        json.dumps(flip_report) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_credibility_report_promotes_candidate_with_multiseed_gain(tmp_path):
    baseline_23 = write_model_run(tmp_path, "dpo64", 23, 0.653, 0.690)
    baseline_47 = write_model_run(tmp_path, "dpo64", 47, 0.620, 0.623)
    candidate_23 = write_model_run(tmp_path, "candidate", 23, 0.663, 0.693)
    candidate_47 = write_model_run(tmp_path, "candidate", 47, 0.617, 0.633)

    report = credibility_report(
        baseline_runs=[baseline_23, baseline_47],
        candidate_runs=[candidate_23, candidate_47],
        baseline_model="DPO64",
        candidate_model="SFT16k+DPO16k",
        seed_labels=["seed23", "seed47"],
    )

    assert report["decision"]["recommendation"] == "promote_candidate_for_bidirectional_scoring"
    assert report["decision"]["pass"] is True
    assert report["aggregate"]["candidate"]["mean_selected_accuracy"] == 0.663
    assert report["aggregate_delta"]["mean_selected_accuracy_delta"] == 0.0065
    assert [row["seed"] for row in report["seed_comparisons"]] == ["seed23", "seed47"]
    assert "candidate_selected_accuracy_non_decreasing" in report["decision"]["passed_checks"]
    assert report["limitations"][0].startswith("SciImpact is an external benchmark")


def test_credibility_report_keeps_candidate_when_any_seed_regresses(tmp_path):
    baseline_23 = write_model_run(tmp_path, "dpo64", 23, 0.653, 0.690)
    baseline_47 = write_model_run(tmp_path, "dpo64", 47, 0.620, 0.623)
    candidate_23 = write_model_run(tmp_path, "candidate", 23, 0.663, 0.693)
    candidate_47 = write_model_run(tmp_path, "candidate", 47, 0.600, 0.610)

    report = credibility_report(
        baseline_runs=[baseline_23, baseline_47],
        candidate_runs=[candidate_23, candidate_47],
        baseline_model="DPO64",
        candidate_model="SFT16k+DPO16k",
    )

    assert report["decision"]["recommendation"] == "keep_baseline_or_collect_more_evidence"
    assert report["decision"]["pass"] is False
    assert "candidate_selected_accuracy_non_decreasing" in report["decision"]["failed_checks"]


def test_credibility_report_blocks_candidate_with_too_few_seeds(tmp_path):
    baseline = write_model_run(tmp_path, "dpo64", 23, 0.653, 0.690)
    candidate = write_model_run(tmp_path, "candidate", 23, 0.663, 0.700)

    report = credibility_report(
        baseline_runs=[baseline],
        candidate_runs=[candidate],
        baseline_model="DPO64",
        candidate_model="candidate",
        min_seed_count=3,
    )

    assert report["decision"]["pass"] is False
    assert report["decision"]["recommendation"] == "keep_baseline_or_collect_more_evidence"
    assert "minimum_seed_count_met" in report["decision"]["failed_checks"]
    assert report["decision"]["thresholds"]["min_seed_count"] == 3


def test_credibility_report_allows_small_seed_regression_but_blocks_large_one(tmp_path):
    baseline_23 = write_model_run(tmp_path, "dpo64", 23, 0.653, 0.690)
    baseline_47 = write_model_run(tmp_path, "dpo64", 47, 0.620, 0.623)
    baseline_71 = write_model_run(tmp_path, "dpo64", 71, 0.617, 0.653)
    candidate_23 = write_model_run(tmp_path, "candidate", 23, 0.663, 0.710)
    candidate_47 = write_model_run(tmp_path, "candidate", 47, 0.620, 0.643)
    candidate_71 = write_model_run(tmp_path, "candidate", 71, 0.617, 0.645)

    report = credibility_report(
        baseline_runs=[baseline_23, baseline_47, baseline_71],
        candidate_runs=[candidate_23, candidate_47, candidate_71],
        baseline_model="DPO64",
        candidate_model="candidate",
        min_seed_count=3,
        allowed_seed_selected_regression=0.01,
    )

    assert report["decision"]["pass"] is True
    assert "seed_selected_regression_within_tolerance" in report["decision"]["passed_checks"]
    assert report["aggregate_delta"]["min_seed_selected_accuracy_delta"] == -0.008

    worse_candidate_71 = write_model_run(tmp_path, "candidate_worse", 71, 0.617, 0.630)
    blocked = credibility_report(
        baseline_runs=[baseline_23, baseline_47, baseline_71],
        candidate_runs=[candidate_23, candidate_47, worse_candidate_71],
        baseline_model="DPO64",
        candidate_model="candidate",
        min_seed_count=3,
        allowed_seed_selected_regression=0.01,
    )

    assert blocked["decision"]["pass"] is False
    assert "seed_selected_regression_within_tolerance" in blocked["decision"]["failed_checks"]
    assert blocked["aggregate_delta"]["min_seed_selected_accuracy_delta"] == -0.023


def test_credibility_report_blocks_candidate_with_large_slice_regression(tmp_path):
    baseline = write_model_run(
        tmp_path,
        "dpo64",
        23,
        0.653,
        0.690,
        by_dimension={
            "citation": {"selected_accuracy": 0.70},
            "downloads": {"selected_accuracy": 0.70},
        },
    )
    candidate = write_model_run(
        tmp_path,
        "candidate",
        23,
        0.663,
        0.720,
        by_dimension={
            "citation": {"selected_accuracy": 0.62},
            "downloads": {"selected_accuracy": 0.73},
        },
    )

    report = credibility_report(
        baseline_runs=[baseline],
        candidate_runs=[candidate],
        baseline_model="DPO64",
        candidate_model="candidate",
        allowed_slice_regression=0.05,
    )

    assert report["decision"]["pass"] is False
    assert "slice_regression_within_tolerance" in report["decision"]["failed_checks"]
    assert report["aggregate_delta"]["worst_group_delta"] == -0.08


def test_write_credibility_report_outputs_json_and_markdown(tmp_path):
    baseline = write_model_run(tmp_path, "dpo64", 71, 0.617, 0.653)
    candidate = write_model_run(
        tmp_path,
        "candidate",
        71,
        0.623,
        0.660,
        by_dimension={
            "award": {"selected_accuracy": 0.86},
            "stars": {"selected_accuracy": 0.56},
        },
        by_item_type={
            "paper": {"selected_accuracy": 0.69},
            "github_repo": {"selected_accuracy": 0.56},
        },
    )
    output = tmp_path / "credibility"

    report = write_credibility_report(
        baseline_runs=[baseline],
        candidate_runs=[candidate],
        output_dir=output,
        baseline_model="DPO64",
        candidate_model="SFT16k+DPO16k",
    )

    assert (output / "credibility_report.json").exists()
    assert (output / "credibility_report.md").exists()
    markdown = (output / "credibility_report.md").read_text(encoding="utf-8")
    assert "# Impact System Credibility Report" in markdown
    assert "SFT16k+DPO16k" in markdown
    assert "## Claim Boundary" in markdown
    assert "## Quality Gate" in markdown
    assert report["seed_count"] == 1


def test_credibility_report_reads_selected_eval_and_legacy_flip_dirs(tmp_path):
    run_dir = tmp_path / "legacy"
    selected_eval = run_dir / "selected_eval"
    flip_dir = run_dir / "flip_consistency"
    selected_eval.mkdir(parents=True)
    flip_dir.mkdir(parents=True)
    (selected_eval / "sciimpact_model_eval.json").write_text(
        json.dumps(
            {
                "summary": {
                    "total_examples": 300,
                    "accuracy": 0.69,
                    "coverage_rate": 1.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (flip_dir / "sciimpact_flip_consistency.json").write_text(
        json.dumps(
            {
                "summary": {
                    "original_accuracy": 0.65,
                    "selected_accuracy": 0.69,
                    "consistency_rate": 0.64,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = credibility_report(
        baseline_runs=[run_dir],
        candidate_runs=[run_dir],
        baseline_model="DPO64",
        candidate_model="DPO64",
    )

    row = report["seed_comparisons"][0]
    assert row["baseline_direct_accuracy"] == 0.65
    assert row["baseline_selected_accuracy"] == 0.69
    assert row["candidate_direct_accuracy"] == 0.65
    assert row["candidate_selected_accuracy"] == 0.69


def test_credibility_report_prefers_margin005_flip_consistency(tmp_path):
    run_dir = tmp_path / "run"
    margin000 = run_dir / "flip_consistency"
    margin005 = run_dir / "flip_consistency_margin005"
    margin000.mkdir(parents=True)
    margin005.mkdir(parents=True)
    (margin000 / "sciimpact_flip_consistency.json").write_text(
        json.dumps(
            {
                "summary": {
                    "original_accuracy": 0.65,
                    "selected_accuracy": 0.61,
                    "consistency_rate": 0.70,
                    "confidence_margin": 0.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (margin005 / "sciimpact_flip_consistency.json").write_text(
        json.dumps(
            {
                "summary": {
                    "original_accuracy": 0.65,
                    "selected_accuracy": 0.69,
                    "consistency_rate": 0.64,
                    "confidence_margin": 0.05,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = credibility_report(
        baseline_runs=[run_dir],
        candidate_runs=[run_dir],
        baseline_model="same",
        candidate_model="same",
    )

    row = report["seed_comparisons"][0]
    assert row["baseline_selected_accuracy"] == 0.69
    assert row["candidate_selected_accuracy"] == 0.69


def test_cli_credibility_report_writes_report(tmp_path):
    baseline = write_model_run(tmp_path, "dpo64", 71, 0.617, 0.653)
    candidate = write_model_run(tmp_path, "candidate", 71, 0.623, 0.660)
    output = tmp_path / "credibility"

    result = CliRunner().invoke(
        app,
        [
            "credibility-report",
            "--baseline-run",
            str(baseline),
            "--candidate-run",
            str(candidate),
            "--baseline-model",
            "DPO64",
            "--candidate-model",
            "SFT16k+DPO16k",
            "--seed-label",
            "seed71",
            "--min-seed-count",
            "1",
            "--allowed-seed-selected-regression",
            "0.02",
            "--allowed-slice-regression",
            "0.05",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote credibility report" in result.stdout
    assert (output / "credibility_report.json").exists()
    assert (output / "credibility_report.md").exists()
    payload = json.loads((output / "credibility_report.json").read_text(encoding="utf-8"))
    assert payload["seed_comparisons"][0]["seed"] == "seed71"
    assert payload["decision"]["thresholds"]["allowed_seed_selected_regression"] == 0.02
    assert payload["decision"]["thresholds"]["allowed_slice_regression"] == 0.05


def test_render_credibility_report_markdown_contains_limits_and_evidence(tmp_path):
    baseline = write_model_run(tmp_path, "dpo64", 71, 0.617, 0.653)
    candidate = write_model_run(tmp_path, "candidate", 71, 0.623, 0.660)
    report = credibility_report(
        baseline_runs=[baseline],
        candidate_runs=[candidate],
        baseline_model="DPO64",
        candidate_model="SFT16k+DPO16k",
    )

    markdown = render_credibility_report_markdown(report)

    assert "dataset-local paper influence" in markdown
    assert "SciImpact" in markdown
    assert "mean_selected_accuracy_delta" in markdown
