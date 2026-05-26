import csv
import json

from local_impact_ranking.evaluate import evaluate_judge_results, render_evaluation_markdown
from local_impact_ranking.export import export_human_audit_sample, export_rankings_csv
from local_impact_ranking.judge import error_result, parse_judge_output
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import InfluenceScores


def sample_result():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [{"reference_id": "r1", "title": "Known"}],
        }
    )
    return parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {"reference_id": "r1", "rank": 1, "confidence": 0.9, "rationale": ""},
                    {"reference_id": "bad", "rank": 2, "confidence": 0.1, "rationale": ""},
                ],
            }
        ),
        paper,
    )


def test_evaluate_judge_results_counts_parse_and_invalid_rates():
    metrics = evaluate_judge_results([sample_result()])

    assert metrics["judge_results"] == 1
    assert metrics["parse_success_rate"] == 1.0
    assert metrics["invalid_selection_rate"] == 0.5
    assert metrics["average_selected_references"] == 2.0


def test_evaluate_judge_results_counts_error_rate():
    metrics = evaluate_judge_results([sample_result(), error_result("p2", "server down")])

    assert metrics["judge_results"] == 2
    assert metrics["error_rate"] == 0.5


def test_evaluate_judge_results_counts_empty_duplicate_and_top_k_compliance():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [{"reference_id": "r1", "title": "Known"}],
        }
    )
    duplicate_result = parse_judge_output(
        json.dumps(
            {
                "target_paper_id": "p1",
                "selected_references": [
                    {"reference_id": "r1", "rank": 1, "confidence": 0.9, "rationale": ""},
                    {"reference_id": "r1", "rank": 2, "confidence": 0.8, "rationale": ""},
                ],
            }
        ),
        paper,
    )
    empty_result = parse_judge_output(
        json.dumps({"target_paper_id": "p2", "selected_references": []}),
        Paper.model_validate(
            {"paper_id": "p2", "title": "Empty", "references": [{"reference_id": "r2", "title": "Known"}]}
        ),
    )

    metrics = evaluate_judge_results([duplicate_result, empty_result], top_k=1)

    assert metrics["empty_selection_rate"] == 0.5
    assert metrics["duplicate_selection_rate"] == 0.5
    assert metrics["top_k_compliance_rate"] == 0.5


def test_render_evaluation_markdown_contains_report_table():
    markdown = render_evaluation_markdown(
        {
            "judge_results": 10,
            "parse_success_rate": 1.0,
            "error_rate": 0.0,
            "invalid_selection_rate": 0.1,
            "average_selected_references": 4.8,
            "empty_selection_rate": 0.0,
            "duplicate_selection_rate": 0.0,
            "top_k_compliance_rate": 1.0,
        }
    )

    assert markdown.startswith("# Judge Evaluation Report\n")
    assert "| parse_success_rate | 1.000 |" in markdown
    assert "| judge_results | 10 |" in markdown
    assert "not a human correctness label" in markdown


def test_export_rankings_csv_writes_sorted_rankings(tmp_path):
    scores = InfluenceScores(
        paper_scores={"p2": 1.0, "p1": 2.0, "p3": 0.0},
        author_scores={"Author B": 0.5, "Author A": 3.0},
        institution_scores={"Institution Y": 4.0, "Institution X": 1.0},
        field_scores={"AI": 2.5},
        year_scores={"2024": 2.0, "2023": 1.0},
    )

    export_rankings_csv(scores, tmp_path)

    with (tmp_path / "papers.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0] == {
        "rank": "1",
        "id": "p1",
        "score": "2.000000",
        "normalized_score": "100.000000",
        "percentile": "100.000000",
    }
    assert rows[1] == {
        "rank": "2",
        "id": "p2",
        "score": "1.000000",
        "normalized_score": "50.000000",
        "percentile": "66.666667",
    }
    assert rows[2]["normalized_score"] == "0.000000"
    assert rows[2]["percentile"] == "33.333333"
    assert (tmp_path / "authors.csv").exists()
    assert (tmp_path / "institutions.csv").exists()
    assert (tmp_path / "fields.csv").exists()
    assert (tmp_path / "years.csv").exists()


def test_export_human_audit_sample_writes_selected_references_with_metadata(tmp_path):
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "abstract": "Target abstract",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "Known Reference",
                    "year": 2020,
                    "venue": "ICML",
                }
            ],
        }
    )

    export_human_audit_sample([paper], [sample_result()], tmp_path / "human_audit_sample.csv")

    with (tmp_path / "human_audit_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["target_paper_id"] == "p1"
    assert rows[0]["target_title"] == "Target"
    assert rows[0]["reference_id"] == "r1"
    assert rows[0]["reference_title"] == "Known Reference"
    assert rows[0]["human_importance_label"] == ""
