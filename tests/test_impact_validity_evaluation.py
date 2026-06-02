import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.impact_validity_evaluation import (
    evaluate_impact_validity,
    write_impact_validity_evaluation,
)


def write_rankings(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "final_impact_score", "confidence"],
        )
        writer.writeheader()
        writer.writerow({"paper_id": "p1", "final_impact_score": "90", "confidence": "0.9"})
        writer.writerow({"paper_id": "p2", "final_impact_score": "70", "confidence": "0.7"})
        writer.writerow({"paper_id": "p3", "final_impact_score": "80", "confidence": "0.8"})
        writer.writerow({"paper_id": "p4", "final_impact_score": "10", "confidence": "0.4"})


def write_labels(path):
    rows = [
        {"paper_id": "p1", "future_impact": 100, "field": "AI"},
        {"paper_id": "p2", "future_impact": 80, "field": "AI"},
        {"paper_id": "p3", "future_impact": 60, "field": "AI"},
        {"paper_id": "p4", "future_impact": 0, "field": "AI"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_impact_validity_reports_rank_and_topk_metrics(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.jsonl"
    write_rankings(rankings)
    write_labels(labels)

    report = evaluate_impact_validity(
        rankings,
        labels,
        label_column="future_impact",
        top_fraction=0.5,
    )

    summary = report["summary"]
    assert summary["matched_paper_count"] == 4
    assert summary["ranking_paper_count"] == 4
    assert summary["label_paper_count"] == 4
    assert summary["score_label_match_rate"] == 1.0
    assert summary["label_match_rate"] == 1.0
    assert summary["score_column"] == "final_impact_score"
    assert summary["label_column"] == "future_impact"
    assert summary["spearman_correlation"] == 0.8
    assert round(summary["kendall_tau_b"], 6) == 0.666667
    assert round(summary["pairwise_accuracy"], 6) == 0.833333
    assert round(summary["ndcg_at_top_k"], 6) == 0.916141
    assert summary["top_k"] == 2
    assert summary["top_k_precision"] == 0.5
    assert summary["top_k_recall"] == 0.5
    assert summary["decision"]["status"] == "insufficient_validation_data"
    assert "bootstrap" in report
    assert "permutation_baseline" in report
    assert "statistical_significance" in report
    assert report["statistical_significance"]["tested_metric"] == "pairwise_accuracy"


def test_write_impact_validity_evaluation_writes_json_and_markdown(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.jsonl"
    output = tmp_path / "validity"
    write_rankings(rankings)
    write_labels(labels)

    report = write_impact_validity_evaluation(
        rankings,
        labels,
        output,
        label_column="future_impact",
        top_fraction=0.5,
    )

    assert report["summary"]["pairwise_accuracy"] == 5 / 6
    assert (output / "impact_validity_evaluation.json").exists()
    assert (output / "impact_credibility_scorecard.json").exists()
    assert (output / "impact_credibility_scorecard.md").exists()
    markdown = (output / "impact_validity_evaluation.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Impact Validity Evaluation\n")
    assert "| spearman_correlation | 0.800 |" in markdown
    assert "| ndcg_at_top_k | 0.916 |" in markdown
    scorecard = json.loads(
        (output / "impact_credibility_scorecard.json").read_text(encoding="utf-8")
    )
    assert scorecard["checks"]["coverage"]["status"] == "fail"
    scorecard_md = (output / "impact_credibility_scorecard.md").read_text(
        encoding="utf-8"
    )
    assert scorecard_md.startswith("# Impact Credibility Scorecard\n")


def test_cli_evaluate_impact_validity_writes_report(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.jsonl"
    output = tmp_path / "validity"
    write_rankings(rankings)
    write_labels(labels)

    result = CliRunner().invoke(
        app,
        [
            "evaluate-impact-validity",
            "--rankings",
            str(rankings),
            "--labels",
            str(labels),
            "--label-column",
            "future_impact",
            "--top-fraction",
            "0.5",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "spearman=0.800" in result.output
    assert "credibility=" in result.output
    assert (output / "impact_validity_evaluation.json").exists()
    assert (output / "impact_credibility_scorecard.md").exists()


def test_evaluate_impact_validity_reports_bootstrap_and_permutation_evidence(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.csv"
    rows = []
    label_rows = []
    for index in range(30):
        paper_id = f"p{index:02d}"
        score = 100 - index
        label = 100 - index
        rows.append({"paper_id": paper_id, "final_impact_score": score})
        label_rows.append(
            {
                "paper_id": paper_id,
                "future_impact": label,
                "field": "AI" if index < 15 else "Bio",
            }
        )
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        writer.writerows(rows)
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "future_impact", "field"],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    report = evaluate_impact_validity(
        rankings,
        labels,
        label_column="future_impact",
        top_fraction=0.2,
        bootstrap_samples=80,
        permutation_samples=80,
        random_seed=7,
    )

    bootstrap = report["bootstrap"]
    assert bootstrap["sample_count"] == 80
    assert bootstrap["metrics"]["spearman_correlation"]["ci95"][0] > 0.99
    assert bootstrap["metrics"]["pairwise_accuracy"]["ci95"][0] > 0.99

    permutation = report["permutation_baseline"]
    assert permutation["sample_count"] == 80
    assert permutation["metrics"]["pairwise_accuracy"]["observed"] == 1.0
    assert 0.0 <= permutation["metrics"]["pairwise_accuracy"]["p_value"] <= 0.05

    significance = report["statistical_significance"]
    assert significance["significant_vs_random"] is True
    assert significance["status"] == "statistically_supported"
    assert report["summary"]["decision"]["status"] == "externally_supported"
    scorecard = report["credibility_scorecard"]
    assert scorecard["status"] == "high_credibility"
    assert scorecard["grade"] == "A"
    assert scorecard["can_claim_high_validation_credibility"] is True
    assert scorecard["checks"]["statistical_support"]["status"] == "pass"
    assert scorecard["checks"]["outlier_robustness"]["status"] == "pass"
    assert report["robustness"]["status"] == "stable"


def test_evaluate_impact_validity_flags_outlier_sensitive_top_quality(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.csv"
    score_rows = [{"paper_id": "p00", "final_impact_score": 1000}]
    label_rows = [{"paper_id": "p00", "future_impact": 10000, "field": "AI"}]
    for index in range(1, 31):
        paper_id = f"p{index:02d}"
        score_rows.append({"paper_id": paper_id, "final_impact_score": index})
        label_rows.append(
            {
                "paper_id": paper_id,
                "future_impact": 31 - index,
                "field": "AI",
            }
        )
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        writer.writerows(score_rows)
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "future_impact", "field"],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    report = evaluate_impact_validity(
        rankings,
        labels,
        label_column="future_impact",
        top_fraction=0.2,
        bootstrap_samples=20,
        permutation_samples=20,
        random_seed=13,
    )

    robustness = report["robustness"]
    assert robustness["status"] == "outlier_sensitive"
    assert robustness["checks"]["leave_top_label_out"]["status"] == "fail"
    scorecard = report["credibility_scorecard"]
    assert scorecard["checks"]["outlier_robustness"]["status"] == "fail"
    assert scorecard["status"] in {"low_credibility", "not_credible"}


def test_evaluate_impact_validity_marks_random_like_scores_not_significant(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.csv"
    rows = []
    label_rows = []
    for index in range(30):
        paper_id = f"p{index:02d}"
        rows.append({"paper_id": paper_id, "final_impact_score": index % 3})
        label_rows.append(
            {
                "paper_id": paper_id,
                "future_impact": 100 - index,
                "field": "AI",
            }
        )
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        writer.writerows(rows)
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "future_impact", "field"],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    report = evaluate_impact_validity(
        rankings,
        labels,
        label_column="future_impact",
        top_fraction=0.2,
        bootstrap_samples=40,
        permutation_samples=40,
        random_seed=11,
    )

    assert report["statistical_significance"]["significant_vs_random"] is False
    assert report["summary"]["decision"]["status"] == "not_validated"


def test_evaluate_impact_validity_reports_low_label_match_coverage(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.csv"
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "paper_id": f"p{index:02d}",
                    "final_impact_score": 100 - index,
                }
            )
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "future_impact", "field"],
        )
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "paper_id": f"p{index:02d}",
                    "future_impact": 100 - index,
                    "field": "AI",
                }
            )

    report = evaluate_impact_validity(
        rankings,
        labels,
        bootstrap_samples=0,
        permutation_samples=0,
    )

    summary = report["summary"]
    assert summary["ranking_paper_count"] == 10
    assert summary["label_paper_count"] == 3
    assert summary["matched_paper_count"] == 3
    assert summary["score_label_match_rate"] == 0.3
    assert summary["label_match_rate"] == 1.0
    coverage = report["coverage_examples"]
    assert coverage["unmatched_score_ids"] == [
        "p03",
        "p04",
        "p05",
        "p06",
        "p07",
        "p08",
        "p09",
    ]
    assert coverage["unmatched_label_ids"] == []


def test_evaluate_impact_validity_reports_unmatched_label_examples(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "future_labels.csv"
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_impact_score"])
        writer.writeheader()
        writer.writerow({"paper_id": "p1", "final_impact_score": 90})
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "future_impact", "field"],
        )
        writer.writeheader()
        writer.writerow({"paper_id": "p1", "future_impact": 10, "field": "AI"})
        writer.writerow({"paper_id": "missing_label", "future_impact": 20, "field": "AI"})

    report = evaluate_impact_validity(
        rankings,
        labels,
        bootstrap_samples=0,
        permutation_samples=0,
    )

    assert report["coverage_examples"]["unmatched_score_ids"] == []
    assert report["coverage_examples"]["unmatched_label_ids"] == ["missing_label"]
