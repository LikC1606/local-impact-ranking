import json
import threading
import time

import pytest
from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_model_eval import (
    evaluate_sciimpact_flip_consistency,
    evaluate_sciimpact_predictions,
    generate_sciimpact_predictions,
    write_sciimpact_model_candidate_gate,
    write_sciimpact_model_credibility_summary,
    write_sciimpact_model_weak_slice_error_pack,
    write_sciimpact_calibration_split,
    write_sciimpact_flipped_sft,
)


def write_sft(path):
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "Which paper has more citations?"},
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Low Abstract: Narrow. "
                        "Paper B: Title: High Abstract: Broad. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "Which paper won the Best Paper award?"},
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Winner Abstract: Strong. "
                        "Paper B: Title: Loser Abstract: Weak. "
                        "Which paper should win the Best Paper award?"
                    ),
                },
                {"from": "gpt", "value": "Paper A won the best paper award."},
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_model_eval_dir(
    output_dir,
    *,
    accuracy,
    covered=300,
    coverage=1.0,
    weak_slice_accuracy=0.52,
):
    output_dir.mkdir(parents=True)
    correct = round(accuracy * covered)
    report = {
        "summary": {
            "total_examples": covered,
            "covered_examples": covered,
            "uncovered_examples": round(covered * (1.0 - coverage)),
            "correct_examples": correct,
            "wrong_examples": covered - correct,
            "coverage_rate": coverage,
            "accuracy": accuracy,
        },
        "statistical_significance": {
            "status": "statistically_supported",
            "observed": accuracy,
            "p_value": 0.001,
            "significant_vs_random": True,
        },
        "credibility_scorecard": {
            "status": "moderate_credibility",
            "grade": "B",
            "can_claim_high_model_credibility": False,
        },
        "by_dimension": {
            "citation": {
                "total_examples": 50,
                "covered_examples": 50,
                "correct_examples": 36,
                "accuracy": 0.72,
                "coverage_rate": 1.0,
            },
            "stars": {
                "total_examples": 50,
                "covered_examples": 50,
                "correct_examples": round(weak_slice_accuracy * 50),
                "accuracy": weak_slice_accuracy,
                "coverage_rate": 1.0,
            },
        },
        "by_item_type": {
            "paper": {
                "total_examples": 200,
                "covered_examples": 200,
                "correct_examples": 140,
                "accuracy": 0.70,
                "coverage_rate": 1.0,
            },
            "dataset": {
                "total_examples": 25,
                "covered_examples": 25,
                "correct_examples": round(weak_slice_accuracy * 25),
                "accuracy": weak_slice_accuracy,
                "coverage_rate": 1.0,
            },
            "github_repo": {
                "total_examples": 50,
                "covered_examples": 50,
                "correct_examples": round(weak_slice_accuracy * 50),
                "accuracy": weak_slice_accuracy,
                "coverage_rate": 1.0,
            },
        },
    }
    (output_dir / "sciimpact_model_eval.json").write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )
    return output_dir


def _write_model_eval_rows(output_dir, rows):
    (output_dir / "sciimpact_model_eval_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_write_sciimpact_model_credibility_summary_blocks_high_claim_for_weak_slices(tmp_path):
    eval_dirs = [
        _write_model_eval_dir(
            tmp_path / "seed23" / "model_eval" / "eval",
            accuracy=0.68,
            weak_slice_accuracy=0.52,
        ),
        _write_model_eval_dir(
            tmp_path / "seed47" / "model_eval" / "eval",
            accuracy=0.647,
            weak_slice_accuracy=0.56,
        ),
        _write_model_eval_dir(
            tmp_path / "seed71" / "model_eval" / "eval",
            accuracy=0.65,
            weak_slice_accuracy=0.58,
        ),
    ]
    output = tmp_path / "summary"

    report = write_sciimpact_model_credibility_summary(eval_dirs, output)

    assert report["summary"]["evaluation_count"] == 3
    assert report["summary"]["pooled_accuracy"] == pytest.approx(0.659, abs=0.001)
    assert report["credibility_scorecard"]["status"] == "moderate_credibility"
    assert report["credibility_scorecard"]["can_claim_high_model_credibility"] is False
    assert report["credibility_scorecard"]["checks"]["slice_stability"]["status"] == "warn"
    assert [row["label"] for row in report["evaluations"]] == [
        "seed23",
        "seed47",
        "seed71",
    ]
    weak_slice_names = {row["slice"] for row in report["weak_slices"]}
    assert "dimension:stars" in weak_slice_names
    assert "item_type:dataset" in weak_slice_names
    assert (output / "sciimpact_model_credibility_summary.json").exists()
    assert (output / "sciimpact_model_credibility_summary.md").exists()


def test_write_sciimpact_model_weak_slice_error_pack_exports_actionable_errors(tmp_path):
    eval_dir = _write_model_eval_dir(
        tmp_path / "seed23" / "model_eval" / "eval",
        accuracy=0.68,
        weak_slice_accuracy=0.52,
    )
    sft = tmp_path / "sft.jsonl"
    write_sft(sft)
    _write_model_eval_rows(
        eval_dir,
        [
            {
                "source_line": 1,
                "impact_dimension": "stars",
                "item_type": "github_repo",
                "outcome": "wrong",
                "expected_side": "A",
                "predicted_side": "B",
                "confidence": 0.91,
            },
            {
                "source_line": 2,
                "impact_dimension": "citation",
                "item_type": "dataset",
                "outcome": "wrong",
                "expected_side": "A",
                "predicted_side": "B",
                "confidence": 0.7,
            },
            {
                "source_line": 3,
                "impact_dimension": "citation",
                "item_type": "paper",
                "outcome": "wrong",
                "expected_side": "B",
                "predicted_side": "A",
                "confidence": 0.82,
            },
            {
                "source_line": 4,
                "impact_dimension": "stars",
                "item_type": "github_repo",
                "outcome": "correct",
                "expected_side": "A",
                "predicted_side": "A",
                "confidence": 0.55,
            },
        ],
    )
    output = tmp_path / "error_pack"

    report = write_sciimpact_model_weak_slice_error_pack(
        [eval_dir],
        output,
        sft_paths=[sft],
        training_source=True,
    )

    assert report["manifest"]["recommended_use"] == "training_candidate"
    assert report["manifest"]["error_count"] == 2
    assert report["manifest"]["training_record_count"] == 2
    assert report["manifest"]["high_confidence_error_count"] == 1
    assert set(report["manifest"]["by_weak_slice"]) >= {
        "dimension:stars",
        "item_type:dataset",
        "item_type:github_repo",
    }
    rows = [
        json.loads(line)
        for line in (output / "sciimpact_model_weak_slice_errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_line"] for row in rows] == [1, 2]
    assert rows[0]["matched_weak_slices"] == [
        "dimension:stars",
        "item_type:github_repo",
    ]
    sft_rows = [
        json.loads(line)
        for line in (output / "train_pairwise_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    dpo_rows = [
        json.loads(line)
        for line in (output / "train_pairwise_dpo.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(sft_rows) == 2
    assert len(dpo_rows) == 2
    assert sft_rows[0]["mining_source"] == "sciimpact_model_weak_slice_error_pack"
    assert dpo_rows[0]["label_source"] == "sciimpact_model_weak_slice_error_pack"
    assert (output / "sciimpact_model_weak_slice_error_pack.json").exists()
    assert (output / "sciimpact_model_weak_slice_error_pack.md").exists()


def test_cli_export_sciimpact_model_weak_slice_errors_writes_pack(tmp_path):
    eval_dir = _write_model_eval_dir(tmp_path / "seed23", accuracy=0.68)
    _write_model_eval_rows(
        eval_dir,
        [
            {
                "source_line": 1,
                "impact_dimension": "stars",
                "item_type": "dataset",
                "outcome": "wrong",
                "expected_side": "A",
                "predicted_side": "B",
            }
        ],
    )
    output = tmp_path / "error_pack"

    result = CliRunner().invoke(
        app,
        [
            "export-sciimpact-model-weak-slice-errors",
            "--eval",
            str(eval_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "errors=1" in result.output
    assert (output / "sciimpact_model_weak_slice_errors.jsonl").exists()


def test_cli_summarize_sciimpact_model_credibility_writes_report(tmp_path):
    eval_dirs = [
        _write_model_eval_dir(tmp_path / "seed23", accuracy=0.68),
        _write_model_eval_dir(tmp_path / "seed47", accuracy=0.647),
        _write_model_eval_dir(tmp_path / "seed71", accuracy=0.65),
    ]
    output = tmp_path / "summary"

    result = CliRunner().invoke(
        app,
        [
            "summarize-sciimpact-model-credibility",
            "--eval",
            str(eval_dirs[0]),
            "--eval",
            str(eval_dirs[1]),
            "--eval",
            str(eval_dirs[2]),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "status=moderate_credibility" in result.output
    assert (output / "sciimpact_model_credibility_summary.json").exists()


def test_write_sciimpact_model_candidate_gate_rejects_cross_seed_regression(tmp_path):
    baseline_evals = [
        _write_model_eval_dir(tmp_path / "baseline-seed23", accuracy=0.680),
        _write_model_eval_dir(tmp_path / "baseline-seed47", accuracy=0.647),
        _write_model_eval_dir(tmp_path / "baseline-seed71", accuracy=0.650),
    ]
    candidate_evals = [
        _write_model_eval_dir(tmp_path / "candidate-seed23", accuracy=0.667),
        _write_model_eval_dir(tmp_path / "candidate-seed47", accuracy=0.657),
        _write_model_eval_dir(tmp_path / "candidate-seed71", accuracy=0.627),
    ]
    baseline_summary_dir = tmp_path / "baseline-summary"
    candidate_summary_dir = tmp_path / "candidate-summary"
    write_sciimpact_model_credibility_summary(baseline_evals, baseline_summary_dir)
    write_sciimpact_model_credibility_summary(candidate_evals, candidate_summary_dir)

    report = write_sciimpact_model_candidate_gate(
        baseline_summary_dir,
        candidate_summary_dir,
        tmp_path / "gate",
    )

    assert report["decision"] == "do_not_promote"
    assert report["checks"]["pooled_accuracy_delta"]["status"] == "fail"
    assert report["checks"]["min_seed_accuracy_delta"]["status"] == "fail"
    assert report["comparison"]["pooled_accuracy_delta"] == pytest.approx(
        -0.008889,
        abs=0.000001,
    )
    assert (tmp_path / "gate" / "sciimpact_model_candidate_gate.json").exists()
    assert (tmp_path / "gate" / "sciimpact_model_candidate_gate.md").exists()


def test_write_sciimpact_model_candidate_gate_promotes_clear_improvement(tmp_path):
    baseline_evals = [
        _write_model_eval_dir(tmp_path / "baseline-seed23", accuracy=0.640, weak_slice_accuracy=0.62),
        _write_model_eval_dir(tmp_path / "baseline-seed47", accuracy=0.650, weak_slice_accuracy=0.62),
        _write_model_eval_dir(tmp_path / "baseline-seed71", accuracy=0.660, weak_slice_accuracy=0.62),
    ]
    candidate_evals = [
        _write_model_eval_dir(tmp_path / "candidate-seed23", accuracy=0.680, weak_slice_accuracy=0.66),
        _write_model_eval_dir(tmp_path / "candidate-seed47", accuracy=0.690, weak_slice_accuracy=0.66),
        _write_model_eval_dir(tmp_path / "candidate-seed71", accuracy=0.700, weak_slice_accuracy=0.66),
    ]
    baseline_summary_dir = tmp_path / "baseline-summary"
    candidate_summary_dir = tmp_path / "candidate-summary"
    write_sciimpact_model_credibility_summary(baseline_evals, baseline_summary_dir)
    write_sciimpact_model_credibility_summary(candidate_evals, candidate_summary_dir)

    report = write_sciimpact_model_candidate_gate(
        baseline_summary_dir,
        candidate_summary_dir,
        tmp_path / "gate",
    )

    assert report["decision"] == "promote"
    assert all(check["status"] == "pass" for check in report["checks"].values())
    assert report["comparison"]["pooled_accuracy_delta"] == pytest.approx(0.04)


def test_cli_compare_sciimpact_model_candidate_writes_gate(tmp_path):
    baseline_evals = [_write_model_eval_dir(tmp_path / "baseline-seed23", accuracy=0.68)]
    candidate_evals = [_write_model_eval_dir(tmp_path / "candidate-seed23", accuracy=0.65)]
    baseline_summary_dir = tmp_path / "baseline-summary"
    candidate_summary_dir = tmp_path / "candidate-summary"
    write_sciimpact_model_credibility_summary(baseline_evals, baseline_summary_dir)
    write_sciimpact_model_credibility_summary(candidate_evals, candidate_summary_dir)

    output = tmp_path / "gate"
    result = CliRunner().invoke(
        app,
        [
            "compare-sciimpact-model-candidate",
            "--baseline-summary",
            str(baseline_summary_dir),
            "--candidate-summary",
            str(candidate_summary_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "decision=do_not_promote" in result.output
    assert (output / "sciimpact_model_candidate_gate.json").exists()


def test_write_sciimpact_calibration_split_keeps_predictions_aligned(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "split"
    rows = [
        {
            "target": "Answer: A",
            "messages": [{"role": "user", "content": "Prompt 1"}],
            "impact_dimension": "citation",
            "item_type": "paper",
        },
        {
            "target": "Answer: B",
            "messages": [{"role": "user", "content": "Prompt 2"}],
            "impact_dimension": "citation",
            "item_type": "paper",
        },
        {
            "target": "Answer: A",
            "messages": [{"role": "user", "content": "Prompt 3"}],
            "impact_dimension": "award",
            "item_type": "paper",
        },
        {
            "target": "Answer: B",
            "messages": [{"role": "user", "content": "Prompt 4"}],
            "impact_dimension": "award",
            "item_type": "paper",
        },
    ]
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions.write_text(
        "\n".join(
            json.dumps(
                {
                    "source_line": index,
                    "prediction": row["target"],
                    "confidence": 0.5 + index / 10,
                }
            )
            for index, row in enumerate(rows, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_sciimpact_calibration_split(
        sft,
        predictions,
        output,
        calibration_fraction=0.5,
        seed=7,
    )

    assert manifest["source_record_count"] == 4
    assert manifest["calibration_count"] == 2
    assert manifest["validation_count"] == 2
    assert manifest["by_dimension"] == {
        "award": {"calibration_count": 1, "validation_count": 1},
        "citation": {"calibration_count": 1, "validation_count": 1},
    }
    calibration_predictions = [
        json.loads(line)
        for line in (output / "calibration_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validation_predictions = [
        json.loads(line)
        for line in (output / "validation_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_line"] for row in calibration_predictions] == [1, 2]
    assert [row["source_line"] for row in validation_predictions] == [1, 2]
    assert (output / "calibration_sft.jsonl").exists()
    assert (output / "validation_sft.jsonl").exists()


def test_write_sciimpact_flipped_sft_swaps_prompt_sides_and_target(tmp_path):
    sft = tmp_path / "sft.jsonl"
    flipped = tmp_path / "flipped.jsonl"
    write_sft(sft)

    manifest = write_sciimpact_flipped_sft(sft, flipped)

    assert manifest["source_record_count"] == 2
    assert manifest["flipped_record_count"] == 2
    rows = [json.loads(line) for line in flipped.read_text(encoding="utf-8").splitlines()]
    first_human = rows[0]["conversations"][1]["value"]
    assert "Paper A: Title: High Abstract: Broad." in first_human
    assert "Paper B: Title: Low Abstract: Narrow." in first_human
    assert rows[0]["conversations"][2]["value"] == "Paper A has more citations."
    second_human = rows[1]["conversations"][1]["value"]
    assert "Paper A: Title: Loser Abstract: Weak." in second_human
    assert "Paper B: Title: Winner Abstract: Strong." in second_human
    assert rows[1]["conversations"][2]["value"] == "Paper B won the best paper award."
    assert rows[0]["flip_metadata"] == {
        "mode": "swap_ab",
        "source_line": 1,
        "original_expected_side": "B",
        "flipped_expected_side": "A",
    }


def test_evaluate_sciimpact_flip_consistency_maps_flipped_predictions(tmp_path):
    sft = tmp_path / "sft.jsonl"
    original_predictions = tmp_path / "original_predictions.jsonl"
    flipped_predictions = tmp_path / "flipped_predictions.jsonl"
    output = tmp_path / "flip_eval"
    write_sft(sft)
    original_predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_line": 1,
                        "prediction": "Answer: B",
                        "confidence": 0.6,
                    }
                ),
                json.dumps(
                    {
                        "source_line": 2,
                        "prediction": "Answer: B",
                        "confidence": 0.55,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    flipped_predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_line": 1,
                        "prediction": "Answer: A",
                        "confidence": 0.7,
                    }
                ),
                json.dumps(
                    {
                        "source_line": 2,
                        "prediction": "Answer: B",
                        "confidence": 0.8,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_flip_consistency(
        sft,
        original_predictions,
        flipped_predictions,
        output,
    )

    assert report["summary"]["total_examples"] == 2
    assert report["summary"]["original_accuracy"] == 0.5
    assert report["summary"]["flipped_mapped_accuracy"] == 1.0
    assert report["summary"]["consistency_rate"] == 0.5
    assert report["summary"]["selected_accuracy"] == 1.0
    assert report["summary"]["both_correct_examples"] == 1
    assert report["summary"]["original_only_correct_examples"] == 0
    assert report["summary"]["flipped_only_correct_examples"] == 1
    assert report["summary"]["both_wrong_examples"] == 0
    rows = [
        json.loads(line)
        for line in (output / "sciimpact_flip_consistency_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["flipped_mapped_side"] == "B"
    assert rows[0]["consistent"] is True
    assert rows[1]["flipped_mapped_side"] == "A"
    assert rows[1]["selected_source"] == "flipped_mapped_confidence"
    assert rows[1]["selected_outcome"] == "correct"
    selected = [
        json.loads(line)
        for line in (output / "sciimpact_flip_selected_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert selected == [
        {
            "source_line": 1,
            "prediction": "Answer: B",
            "confidence": 0.7,
            "confidence_method": "flip_consistency",
            "selected_source": "consistent",
            "original_prediction": "Answer: B",
            "flipped_prediction": "Answer: A",
            "flipped_mapped_side": "B",
        },
        {
            "source_line": 2,
            "prediction": "Answer: A",
            "confidence": 0.8,
            "confidence_method": "flip_consistency",
            "selected_source": "flipped_mapped_confidence",
            "original_prediction": "Answer: B",
            "flipped_prediction": "Answer: B",
            "flipped_mapped_side": "A",
        },
    ]
    markdown = (output / "sciimpact_flip_consistency.md").read_text(encoding="utf-8")
    assert "## Selection Summary" in markdown


def test_evaluate_sciimpact_flip_consistency_uses_confidence_margin(tmp_path):
    sft = tmp_path / "sft.jsonl"
    original_predictions = tmp_path / "original_predictions.jsonl"
    flipped_predictions = tmp_path / "flipped_predictions.jsonl"
    output = tmp_path / "flip_eval"
    write_sft(sft)
    original_predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Answer: A", "confidence": 0.72})
        + "\n",
        encoding="utf-8",
    )
    flipped_predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Answer: A", "confidence": 0.75})
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_flip_consistency(
        sft,
        original_predictions,
        flipped_predictions,
        output,
        confidence_margin=0.05,
    )

    assert report["summary"]["confidence_margin"] == 0.05
    row = json.loads(
        (output / "sciimpact_flip_consistency_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["original_side"] == "A"
    assert row["flipped_mapped_side"] == "B"
    assert row["selected_side"] == "A"
    assert row["selected_source"] == "original_margin_default"
    selected = json.loads(
        (output / "sciimpact_flip_selected_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert selected["prediction"] == "Answer: A"
    assert selected["selected_source"] == "original_margin_default"


def test_cli_flip_sciimpact_sft_writes_flipped_file(tmp_path):
    sft = tmp_path / "sft.jsonl"
    flipped = tmp_path / "flipped.jsonl"
    write_sft(sft)

    result = CliRunner().invoke(
        app,
        [
            "flip-sciimpact-sft",
            "--sft",
            str(sft),
            "--output",
            str(flipped),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 flipped SciImpact examples" in result.output
    assert flipped.exists()


def test_cli_evaluate_sciimpact_flip_consistency_writes_report(tmp_path):
    sft = tmp_path / "sft.jsonl"
    original_predictions = tmp_path / "original_predictions.jsonl"
    flipped_predictions = tmp_path / "flipped_predictions.jsonl"
    output = tmp_path / "flip_eval"
    write_sft(sft)
    original_predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Answer: B"}) + "\n",
        encoding="utf-8",
    )
    flipped_predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Answer: A"}) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-sciimpact-flip-consistency",
            "--sft",
            str(sft),
            "--predictions",
            str(original_predictions),
            "--flipped-predictions",
            str(flipped_predictions),
            "--output",
            str(output),
            "--limit",
            "1",
            "--confidence-margin",
            "0.05",
        ],
    )

    assert result.exit_code == 0
    assert "selected_accuracy=1.000" in result.output
    assert (output / "sciimpact_flip_consistency.json").exists()


def test_evaluate_sciimpact_predictions_counts_correct_wrong_and_missing(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"source_line": 1, "prediction": "Paper B has more citations."}),
                json.dumps({"source_line": 2, "prediction": "Paper B won the best paper award."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["total_examples"] == 2
    assert report["summary"]["covered_examples"] == 2
    assert report["summary"]["correct_examples"] == 1
    assert report["summary"]["wrong_examples"] == 1
    assert report["summary"]["accuracy"] == 0.5
    assert report["by_dimension"]["citation"]["accuracy"] == 1.0
    assert report["by_dimension"]["award"]["accuracy"] == 0.0
    assert (output / "sciimpact_model_eval.json").exists()
    assert (output / "sciimpact_model_eval.md").exists()
    assert (output / "sciimpact_model_credibility_scorecard.json").exists()
    assert (output / "sciimpact_model_credibility_scorecard.md").exists()
    assert (output / "sciimpact_model_eval_rows.jsonl").exists()
    assert (output / "sciimpact_model_eval_wrong.jsonl").exists()
    assert (output / "sciimpact_model_error_analysis.json").exists()
    assert (output / "sciimpact_model_error_analysis.md").exists()
    assert report["credibility_scorecard"]["status"] == "insufficient_validation_data"
    assert report["statistical_significance"]["status"] == "insufficient_validation_data"
    wrong_rows = [
        json.loads(line)
        for line in (output / "sciimpact_model_eval_wrong.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert wrong_rows == [
        {
            "source_line": 2,
            "expected_side": "A",
            "predicted_side": "B",
            "impact_dimension": "award",
            "item_type": "paper",
            "outcome": "wrong",
            "prediction": "Paper B won the best paper award.",
            "confidence": None,
            "confidence_bucket": "unscored",
            "calibrated_confidence": None,
            "calibrated_confidence_method": None,
            "confidence_method": None,
            "choice_logprob_margin": None,
        }
    ]
    analysis = json.loads(
        (output / "sciimpact_model_error_analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["summary"]["wrong_examples"] == 1
    assert analysis["by_dimension"]["award"]["wrong_examples"] == 1
    assert analysis["by_dimension"]["award"]["wrong_predicted_side_counts"] == {"B": 1}


def test_sciimpact_model_credibility_scorecard_marks_strong_model_supported(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    rows = []
    prediction_rows = []
    for index in range(40):
        expected = "Answer: A" if index % 2 == 0 else "Answer: B"
        rows.append(
            {
                "target": expected,
                "messages": [{"role": "user", "content": f"Prompt {index}"}],
                "impact_dimension": "citation" if index < 20 else "award",
                "item_type": "paper" if index < 20 else "dataset",
            }
        )
        prediction_rows.append({"source_line": index + 1, "prediction": expected})
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["accuracy"] == 1.0
    assert report["statistical_significance"]["significant_vs_random"] is True
    scorecard = report["credibility_scorecard"]
    assert scorecard["status"] == "high_credibility"
    assert scorecard["grade"] == "A"
    assert scorecard["checks"]["slice_stability"]["status"] == "pass"


def test_sciimpact_model_credibility_scorecard_rejects_random_like_model(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    rows = []
    prediction_rows = []
    for index in range(40):
        rows.append(
            {
                "target": "Answer: A",
                "messages": [{"role": "user", "content": f"Prompt {index}"}],
                "impact_dimension": "citation",
                "item_type": "paper",
            }
        )
        predicted = "Answer: A" if index % 2 == 0 else "Answer: B"
        prediction_rows.append({"source_line": index + 1, "prediction": predicted})
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["accuracy"] == 0.5
    assert report["statistical_significance"]["significant_vs_random"] is False
    scorecard = report["credibility_scorecard"]
    assert scorecard["status"] == "not_credible"
    assert scorecard["checks"]["statistical_support"]["status"] == "fail"


def test_evaluate_sciimpact_predictions_reports_confidence_buckets(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_line": 1,
                        "prediction": "Answer: B",
                        "confidence": 0.91,
                        "confidence_method": "choice_logprob_margin",
                        "choice_logprob_margin": 2.3,
                    }
                ),
                json.dumps(
                    {
                        "source_line": 2,
                        "prediction": "Answer: B",
                        "confidence": 0.55,
                        "confidence_method": "choice_logprob_margin",
                        "choice_logprob_margin": 0.2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["confidence_summary"] == {
        "scored_examples": 2,
        "mean_confidence": 0.73,
        "correct_mean_confidence": 0.91,
        "wrong_mean_confidence": 0.55,
        "high_confidence_accuracy": 1.0,
        "low_confidence_accuracy": 0.0,
    }
    assert report["by_confidence_bucket"]["high"]["accuracy"] == 1.0
    assert report["by_confidence_bucket"]["low"]["accuracy"] == 0.0
    rows = [
        json.loads(line)
        for line in (output / "sciimpact_model_eval_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["confidence_bucket"] == "high"
    assert rows[1]["confidence_bucket"] == "low"
    markdown = (output / "sciimpact_model_eval.md").read_text(encoding="utf-8")
    assert "## By Confidence Bucket" in markdown
    assert "Mean Confidence" in markdown


def test_evaluate_sciimpact_predictions_reports_calibration_metrics(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {"source_line": 1, "prediction": "Answer: B", "confidence": 0.9}
                ),
                json.dumps(
                    {"source_line": 2, "prediction": "Answer: B", "confidence": 0.7}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["calibration"]["scored_examples"] == 2
    assert report["calibration"]["bin_count"] == 5
    assert report["calibration"]["ece"] == 0.4
    assert report["calibration"]["mce"] == 0.7
    assert report["calibration"]["brier_score"] == 0.25
    assert report["calibration"]["overconfidence_gap"] == 0.3
    assert report["calibration"]["bins"] == [
        {
            "bin": "0.6-0.8",
            "lower": 0.6,
            "upper": 0.8,
            "count": 1,
            "accuracy": 0.0,
            "mean_confidence": 0.7,
            "calibration_gap": 0.7,
        },
        {
            "bin": "0.8-1.0",
            "lower": 0.8,
            "upper": 1.0,
            "count": 1,
            "accuracy": 1.0,
            "mean_confidence": 0.9,
            "calibration_gap": -0.1,
        },
    ]
    markdown = (output / "sciimpact_model_eval.md").read_text(encoding="utf-8")
    assert "## Calibration" in markdown
    assert "Expected Calibration Error" in markdown


def test_evaluate_sciimpact_predictions_applies_external_calibration_bins(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    calibration_report = tmp_path / "calibration.json"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        "\n".join(
            [
                json.dumps(
                    {"source_line": 1, "prediction": "Answer: B", "confidence": 0.91}
                ),
                json.dumps(
                    {"source_line": 2, "prediction": "Answer: B", "confidence": 0.55}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calibration_report.write_text(
        json.dumps(
            {
                "calibration": {
                    "bins": [
                        {
                            "lower": 0.4,
                            "upper": 0.6,
                            "count": 20,
                            "accuracy": 0.52,
                        },
                        {
                            "lower": 0.8,
                            "upper": 1.0,
                            "count": 30,
                            "accuracy": 0.76,
                        },
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(
        sft,
        predictions,
        output,
        calibration_report_path=calibration_report,
    )

    assert report["calibration_model"] == {
        "method": "empirical_bin_accuracy",
        "source_path": str(calibration_report),
        "bin_count": 2,
    }
    assert report["calibrated_confidence_summary"]["scored_examples"] == 2
    assert report["calibrated_confidence_summary"]["mean_confidence"] == 0.64
    assert report["calibrated_calibration"]["ece"] == 0.38
    assert report["calibrated_calibration"]["brier_score"] == 0.164
    rows = [
        json.loads(line)
        for line in (output / "sciimpact_model_eval_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["calibrated_confidence"] == 0.76
    assert rows[0]["calibrated_confidence_method"] == "empirical_bin_accuracy"
    assert rows[1]["calibrated_confidence"] == 0.52
    markdown = (output / "sciimpact_model_eval.md").read_text(encoding="utf-8")
    assert "## Calibrated Confidence Summary" in markdown


def test_evaluate_sciimpact_predictions_can_limit_source_rows(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Paper B has more citations."}) + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output, limit=1)

    assert report["summary"]["total_examples"] == 1
    assert report["summary"]["coverage_rate"] == 1.0
    assert report["summary"]["accuracy"] == 1.0


def test_evaluate_sciimpact_predictions_uses_last_explicit_side_in_reasoning_output(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps(
            {
                "source_line": 1,
                "prediction": (
                    "Paper A has a narrow contribution. Paper B looks broader. "
                    "Therefore, Paper B has more citations."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["covered_examples"] == 1
    assert report["summary"]["correct_examples"] == 1


def test_evaluate_sciimpact_predictions_prefers_explicit_final_answer(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps(
            {
                "source_line": 1,
                "prediction": (
                    "Paper B looks strong. Paper A has a broader contribution. "
                    "Answer: B"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["covered_examples"] == 1
    assert report["summary"]["correct_examples"] == 1


def test_cli_evaluate_sciimpact_predictions_writes_report(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Paper B has more citations."}) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-sciimpact-predictions",
            "--sft",
            str(sft),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "accuracy=1.000" in result.output
    assert "coverage_rate=0.500" in result.output


def test_generate_sciimpact_predictions_writes_openai_compatible_answers(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Paper B has more citations."}}
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
    )

    assert manifest["written_prediction_count"] == 1
    assert manifest["source_record_count"] == 1
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "source_line": 1,
            "prediction": "Paper B has more citations.",
            "model": "local-impact",
        }
    ]
    assert "Paper A:" in client.payloads[0]["messages"][1]["content"]
    assert client.payloads[0]["messages"][0]["role"] == "system"


def test_generate_sciimpact_predictions_can_request_final_answer_only(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "Answer: B"}}]}

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="final_answer",
    )

    assert manifest["answer_format"] == "final_answer"
    assert client.payloads[0]["messages"][-1]["content"].endswith(
        "Final output format: Answer: A or Answer: B"
    )


def test_generate_sciimpact_predictions_can_request_strict_answer_only(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "B"}}]}

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="strict_answer",
    )

    assert manifest["answer_format"] == "strict_answer"
    assert client.payloads[0]["max_tokens"] == 4
    assert client.payloads[0]["messages"][-1]["content"].endswith(
        "Output exactly: Answer: A or Answer: B. Do not explain."
    )


def test_generate_sciimpact_predictions_can_constrain_qwen_answer_tokens(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": " A"}}]}

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
            answer_format="constrained-answer",
        )

    assert manifest["answer_format"] == "constrained_answer"
    assert client.payloads[0]["max_tokens"] == 3
    assert client.payloads[0]["allowed_token_ids"] == [16141, 25, 362, 425]


def test_generate_sciimpact_predictions_records_logprob_confidence_for_constrained_answers(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {"content": "Answer: B"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " B",
                                            "logprob": -0.1,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -2.1},
                                                {"token": " B", "logprob": -0.1},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="constrained-answer",
    )

    assert manifest["confidence_enabled"] is True
    assert client.payloads[0]["logprobs"] is True
    assert client.payloads[0]["top_logprobs"] == 2
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "source_line": 1,
            "prediction": "Answer: B",
            "model": "local-impact",
            "confidence": 0.880797,
            "confidence_method": "choice_logprob_margin",
            "choice_logprob_margin": 2.0,
        }
    ]


def test_generate_sciimpact_predictions_can_use_single_token_choice_logprobs(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {"content": " B"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " B",
                                            "logprob": -0.2,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -1.7},
                                                {"token": " B", "logprob": -0.2},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="choice-logprob",
    )

    assert manifest["answer_format"] == "choice_logprob"
    assert manifest["confidence_enabled"] is True
    assert client.payloads[0]["max_tokens"] == 1
    assert client.payloads[0]["allowed_token_ids"] == [362, 425]
    assert client.payloads[0]["logprobs"] is True
    assert client.payloads[0]["top_logprobs"] == 2
    assert client.payloads[0]["messages"][-1]["content"].endswith(
        "Output only A or B. Do not explain."
    )
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "source_line": 1,
            "prediction": "B",
            "model": "local-impact",
            "confidence": 0.817574,
            "confidence_method": "choice_logprob_margin",
            "choice_logprob_margin": 1.5,
        }
    ]


def test_generate_sciimpact_predictions_can_use_structured_answer_choice(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {"content": "Answer: B"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " B",
                                            "logprob": -0.4,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -1.4},
                                                {"token": " B", "logprob": -0.4},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="structured-choice",
    )

    assert manifest["answer_format"] == "structured_choice"
    assert manifest["confidence_enabled"] is True
    assert client.payloads[0]["max_tokens"] == 4
    assert client.payloads[0]["structured_outputs"] == {
        "choice": ["Answer: A", "Answer: B"]
    }
    assert "allowed_token_ids" not in client.payloads[0]
    assert client.payloads[0]["logprobs"] is True
    assert client.payloads[0]["top_logprobs"] == 2
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "source_line": 1,
            "prediction": "Answer: B",
            "model": "local-impact",
            "confidence": 0.731059,
            "confidence_method": "choice_logprob_margin",
            "choice_logprob_margin": 1.0,
        }
    ]


def test_generate_sciimpact_predictions_accepts_hyphenated_final_answer_alias(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "Answer: A"}}]}

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
        answer_format="final-answer",
    )

    assert manifest["answer_format"] == "final_answer"
    assert "Final output format" in client.payloads[0]["messages"][-1]["content"]


def test_evaluate_sciimpact_predictions_accepts_bare_a_b_outputs(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "B"}) + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["covered_examples"] == 1
    assert report["summary"]["correct_examples"] == 1


def test_generate_sciimpact_predictions_can_run_requests_concurrently(tmp_path):
    class SlowClient:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def post(self, url, headers, json, timeout):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
            finally:
                with self.lock:
                    self.active -= 1

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Paper B has more citations."}}
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = SlowClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        concurrency=2,
        resume=False,
    )

    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert manifest["written_prediction_count"] == 2
    assert manifest["concurrency"] == 2
    assert sorted(row["source_line"] for row in rows) == [1, 2]
    assert client.max_active == 2


def test_cli_generate_sciimpact_predictions_writes_predictions_with_fake_client(
    monkeypatch,
    tmp_path,
):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)

    def fake_generate(*args, **kwargs):
        predictions.write_text(
            json.dumps(
                {
                    "source_line": 1,
                    "prediction": "Paper B has more citations.",
                    "model": kwargs["model"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"written_prediction_count": 1, "skipped_existing_count": 0}

    import local_impact_ranking.cli as cli

    monkeypatch.setattr(cli, "generate_sciimpact_predictions", fake_generate)

    result = CliRunner().invoke(
        app,
        [
            "generate-sciimpact-predictions",
            "--sft",
            str(sft),
            "--output",
            str(predictions),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-impact",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 SciImpact predictions" in result.output
