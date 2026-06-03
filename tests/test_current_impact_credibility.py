import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.current_impact_credibility import (
    evaluate_current_impact_credibility,
    write_current_impact_credibility,
)


def write_current_scores(path):
    rows = []
    for index in range(30):
        score = 100 - index
        rows.append(
            {
                "paper_id": f"p{index:02d}",
                "final_impact_score": score,
                "score_80_true_20_effective": score,
                "score_70_true_30_effective": score - index * 0.01,
                "score_50_true_50_effective": score - index * 0.02,
                "score_30_true_70_effective": score - index * 0.03,
                "true_citation_raw": 1000 - index,
                "effective_citation_edge_count": 2 if index < 20 else 0,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_current_labels(path):
    rows = [
        {
            "paper_id": f"p{index:02d}",
            "rcr": 100 - index,
            "citations_per_year": 90 - index,
            "apt": 80 - index,
            "field": "biomedicine",
        }
        for index in range(30)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_current_impact_credibility_reports_scorecard(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    write_current_scores(scores)
    write_current_labels(labels)

    report = evaluate_current_impact_credibility(
        scores,
        labels,
        label_columns=["rcr", "citations_per_year", "apt"],
        bootstrap_samples=20,
        permutation_samples=20,
        random_seed=11,
    )

    assert report["summary"]["credibility_status"] == "high_credibility"
    assert report["summary"]["can_claim_current_impact_credible"] is True
    assert report["coverage"]["true_citation_coverage_rate"] == 1.0
    assert report["coverage"]["effective_citation_coverage_rate"] == 0.666667
    assert report["credibility_scorecard"]["checks"]["external_alignment"][
        "status"
    ] == "pass"
    assert report["sensitivity"]["min_top_k_overlap_with_final"] >= 0.7
    assert report["interpretation"]["not_a_prediction"]


def test_current_impact_credibility_uses_primary_label_not_best_cherry_pick(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    write_current_scores(scores)
    rows = []
    for index in range(30):
        rows.append(
            {
                "paper_id": f"p{index:02d}",
                "rcr": index,
                "icite_citation_count": 100 - index,
            }
        )
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = evaluate_current_impact_credibility(
        scores,
        labels,
        label_columns=["rcr", "icite_citation_count"],
        bootstrap_samples=20,
        permutation_samples=20,
    )

    scorecard = report["credibility_scorecard"]
    assert scorecard["primary_label_column"] == "rcr"
    assert scorecard["checks"]["primary_external_alignment"]["status"] == "fail"
    assert scorecard["checks"]["external_alignment"]["status"] == "pass"
    assert scorecard["status"] == "not_credible"


def test_current_impact_credibility_allows_partial_label_coverage(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    write_current_scores(scores)
    rows = []
    for index in range(30):
        rows.append(
            {
                "paper_id": f"p{index:02d}",
                "field_year_percentile": 100 - index if index < 20 else "",
                "citation_count": 100 - index,
            }
        )
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = evaluate_current_impact_credibility(
        scores,
        labels,
        label_columns=["field_year_percentile", "citation_count"],
        primary_label_column="field_year_percentile",
        bootstrap_samples=20,
        permutation_samples=20,
    )

    assert report["alignment"]["final_impact_score"]["field_year_percentile"][
        "matched_paper_count"
    ] == 20
    assert report["coverage"]["label_column_coverage"]["field_year_percentile"] == 20
    assert report["summary"]["primary_label_column"] == "field_year_percentile"


def test_current_impact_credibility_does_not_gate_on_component_diagnostic_columns(
    tmp_path,
):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    rows = []
    label_rows = []
    for index in range(30):
        final = 100 - index
        rows.append(
            {
                "paper_id": f"p{index:02d}",
                "final_impact_score": final,
                "true_citation_score": final,
                "effective_citation_score": index,
                "score_100_true_00_effective": final,
                "score_95_true_05_effective": final - index * 0.01,
                "score_90_true_10_effective": final - index * 0.02,
                "score_50_true_50_effective": 50,
                "true_citation_raw": 1000 - index,
                "effective_citation_edge_count": 1,
            }
        )
        label_rows.append({"paper_id": f"p{index:02d}", "rcr": final})
    with scores.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)

    report = evaluate_current_impact_credibility(
        scores,
        labels,
        score_columns=[
            "final_impact_score",
            "true_citation_score",
            "effective_citation_score",
            "score_100_true_00_effective",
            "score_95_true_05_effective",
            "score_90_true_10_effective",
            "score_50_true_50_effective",
        ],
        label_columns=["rcr"],
        bootstrap_samples=20,
        permutation_samples=20,
    )

    assert "effective_citation_score" in report["sensitivity"]["by_score"]
    assert report["sensitivity"]["gate_score_columns"] == [
        "score_100_true_00_effective",
        "score_95_true_05_effective",
        "score_90_true_10_effective",
    ]
    assert report["credibility_scorecard"]["checks"]["weight_sensitivity"][
        "status"
    ] == "pass"


def test_write_current_impact_credibility_writes_json_and_markdown(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "current"
    write_current_scores(scores)
    write_current_labels(labels)

    report = write_current_impact_credibility(
        scores,
        labels,
        output,
        label_columns=["rcr"],
        bootstrap_samples=20,
        permutation_samples=20,
    )

    assert report["summary"]["credibility_grade"] == "A"
    assert (output / "current_impact_credibility.json").exists()
    markdown = (output / "current_impact_credibility.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Current Impact Credibility\n")
    assert "not a future-impact prediction benchmark" in markdown


def test_cli_evaluate_current_impact_credibility_writes_report(tmp_path):
    scores = tmp_path / "scores.csv"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "current"
    write_current_scores(scores)
    write_current_labels(labels)

    result = CliRunner().invoke(
        app,
        [
            "evaluate-current-impact-credibility",
            "--scores",
            str(scores),
            "--labels",
            str(labels),
            "--label-columns",
            "rcr,citations_per_year",
            "--bootstrap-samples",
            "20",
            "--permutation-samples",
            "20",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "credibility=high_credibility" in result.output
    payload = json.loads(
        (output / "current_impact_credibility.json").read_text(encoding="utf-8")
    )
    assert payload["summary"]["label_columns"] == ["rcr", "citations_per_year"]
