import csv
import json

import pytest
from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.impact_fusion_calibration import (
    calibrate_fusion_weights,
    write_fusion_calibration,
)


def write_rankings(path, rows):
    fieldnames = [
        "paper_id",
        "impact_index_v2",
        "model_pairwise_score",
        "field_year_normalized_citation_score",
        "relative_citation_ratio_score",
        "citation_graph_score",
        "citation_context_quality_score",
        "influential_citation_score",
        "temporal_momentum_score",
        "external_adoption_score",
        "fusion_weight_model_pairwise",
        "fusion_weight_field_year_normalized_citation",
        "fusion_weight_citation_graph",
        "fusion_weight_citation_context_quality",
        "fusion_weight_temporal_momentum",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in fieldnames})


def write_labels(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def calibration_fixture(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    available = {
        "fusion_weight_model_pairwise": 0.5,
        "fusion_weight_field_year_normalized_citation": 0.5,
        "fusion_weight_citation_graph": 0.0,
        "fusion_weight_citation_context_quality": 0.0,
        "fusion_weight_temporal_momentum": 0.0,
        "citation_graph_score": 0.0,
        "citation_context_quality_score": 0.0,
        "temporal_momentum_score": 0.0,
    }
    write_rankings(
        rankings,
        [
            {
                "paper_id": "winner-a",
                "impact_index_v2": 50,
                "model_pairwise_score": 0,
                "field_year_normalized_citation_score": 100,
                **available,
            },
            {
                "paper_id": "loser-a",
                "impact_index_v2": 50,
                "model_pairwise_score": 100,
                "field_year_normalized_citation_score": 0,
                **available,
            },
            {
                "paper_id": "winner-b",
                "impact_index_v2": 50,
                "model_pairwise_score": 10,
                "field_year_normalized_citation_score": 90,
                **available,
            },
            {
                "paper_id": "loser-b",
                "impact_index_v2": 50,
                "model_pairwise_score": 90,
                "field_year_normalized_citation_score": 10,
                **available,
            },
        ],
    )
    write_labels(
        labels,
        [
            {"pair_id": "a", "winner_id": "winner-a", "loser_id": "loser-a"},
            {"pair_id": "b", "winner_id": "winner-b", "loser_id": "loser-b"},
        ],
    )
    return rankings, labels


def test_calibrate_fusion_weights_selects_weights_that_improve_pairwise_accuracy(tmp_path):
    rankings, labels = calibration_fixture(tmp_path)

    report = calibrate_fusion_weights(rankings, labels, grid_step=0.5)

    assert report["summary"]["calibration_status"] == "calibrated"
    assert report["summary"]["effective_signal_count"] == 2
    assert report["summary"]["best_pairwise_accuracy"] == 1.0
    assert report["summary"]["default_pairwise_accuracy"] == 0.0
    assert report["best_component_weights"]["model_pairwise"] == 0.0
    assert report["best_weights"] == {
        "field_year_normalized_citation": 1.0,
        "relative_citation_ratio": 0.0,
        "citation_graph": 0.0,
        "citation_context_quality": 0.0,
        "influential_citation": 0.0,
        "temporal_momentum": 0.0,
        "external_adoption": 0.0,
    }
    assert report["metric_contract"]["task"] == "Calibrate ImpactIndex v3 final component weights against pairwise impact labels."
    assert report["top_candidates"][0]["pairwise_accuracy"] == 1.0


def test_calibrate_fusion_weights_outputs_v3_component_weights(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    write_rankings(
        rankings,
        [
            {
                "paper_id": "external-winner",
                "model_pairwise_score": 0,
                "external_adoption_score": 100,
            },
            {
                "paper_id": "model-winner",
                "model_pairwise_score": 100,
                "external_adoption_score": 0,
            },
        ],
    )
    write_labels(
        labels,
        [
            {
                "pair_id": "external",
                "winner_id": "external-winner",
                "loser_id": "model-winner",
            }
        ],
    )

    report = calibrate_fusion_weights(rankings, labels, grid_step=0.5)

    assert report["summary"]["weight_profile"] == "impact_index_v3_final_components"
    assert report["best_component_weights"]["external_adoption"] == 1.0
    assert report["best_component_weights"]["model_pairwise"] == 0.0
    assert report["best_impact_index_v3_weights"] == {
        "evidence_supported": 1.0,
        "model_supported": 0.0,
    }
    assert report["best_evidence_first_weights"]["external_adoption"] == 1.0
    assert report["best_evaluation"]["pairwise_accuracy"] == 1.0


def test_calibrate_fusion_weights_rejects_legacy_rankings_without_component_scores(tmp_path):
    rankings = tmp_path / "legacy_impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    rankings.write_text(
        "rank,paper_id,impact_index\n1,p1,100\n2,p2,0\n",
        encoding="utf-8",
    )
    write_labels(labels, [{"winner_id": "p1", "loser_id": "p2"}])

    with pytest.raises(ValueError, match="missing required component score columns"):
        calibrate_fusion_weights(rankings, labels)


def test_calibrate_fusion_weights_marks_single_signal_runs_as_diagnostic(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    write_rankings(
        rankings,
        [
            {
                "paper_id": "p1",
                "impact_index_v2": 100,
                "model_pairwise_score": 100,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
            {
                "paper_id": "p2",
                "impact_index_v2": 0,
                "model_pairwise_score": 0,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
        ],
    )
    write_labels(labels, [{"winner_id": "p1", "loser_id": "p2"}])

    report = calibrate_fusion_weights(rankings, labels, grid_step=0.5)

    assert report["summary"]["calibration_status"] == "single_signal_diagnostic"
    assert report["summary"]["effective_signal_count"] == 1
    assert "At least two component signals" in report["summary"]["calibration_status_reason"]
    assert "single active signal" in report["metric_contract"]["known_limitations"][-1]


def test_calibrate_fusion_weights_reports_dimension_diagnostics_and_errors(tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    write_rankings(
        rankings,
        [
            {
                "paper_id": "citation-winner",
                "impact_index_v2": 100,
                "model_pairwise_score": 100,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
            {
                "paper_id": "citation-loser",
                "impact_index_v2": 0,
                "model_pairwise_score": 0,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
            {
                "paper_id": "patent-winner",
                "impact_index_v2": 0,
                "model_pairwise_score": 0,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
            {
                "paper_id": "patent-loser",
                "impact_index_v2": 100,
                "model_pairwise_score": 100,
                "field_year_normalized_citation_score": 0,
                "citation_graph_score": 0,
                "citation_context_quality_score": 0,
                "temporal_momentum_score": 0,
            },
        ],
    )
    write_labels(
        labels,
        [
            {
                "winner_id": "citation-winner",
                "loser_id": "citation-loser",
                "impact_dimension": "citation",
            },
            {
                "winner_id": "patent-winner",
                "loser_id": "patent-loser",
                "impact_dimension": "patent",
            },
        ],
    )

    report = calibrate_fusion_weights(rankings, labels, grid_step=1.0)

    assert report["best_diagnostics"]["by_dimension"]["citation"]["pairwise_accuracy"] == 1.0
    assert report["best_diagnostics"]["by_dimension"]["patent"]["pairwise_accuracy"] == 0.0
    assert report["best_diagnostics"]["weakest_dimensions"][0]["dimension"] == "patent"
    assert report["best_diagnostics"]["wrong_examples"][0]["impact_dimension"] == "patent"
    assert report["best_diagnostics"]["wrong_examples"][0]["score_margin"] == -100.0


def test_write_fusion_calibration_writes_json_and_markdown(tmp_path):
    rankings, labels = calibration_fixture(tmp_path)
    output = tmp_path / "calibration"

    report = write_fusion_calibration(rankings, labels, output, grid_step=0.5)

    assert report["best_weights"]["field_year_normalized_citation"] == 1.0
    assert (output / "fusion_calibration.json").exists()
    markdown = (output / "fusion_calibration.md").read_text(encoding="utf-8")
    assert "Fusion Weight Calibration" in markdown
    assert "calibration_status" in markdown
    assert "By Impact Dimension" in markdown
    assert "Wrong Examples" in markdown
    assert "field_year_normalized_citation" in markdown


def test_cli_calibrate_fusion_weights_writes_report(monkeypatch, tmp_path):
    rankings = tmp_path / "impact_scores.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    output = tmp_path / "calibration"
    rankings.write_text("paper_id,impact_index_v2\np1,1\n", encoding="utf-8")
    labels.write_text(
        json.dumps({"winner_id": "p1", "loser_id": "p2"}) + "\n",
        encoding="utf-8",
    )

    def fake_write_fusion_calibration(*args, **kwargs):
        assert args[0] == rankings
        assert args[1] == labels
        assert args[2] == output
        assert kwargs["grid_step"] == 0.25
        output.mkdir(parents=True)
        return {"summary": {"best_pairwise_accuracy": 0.75, "covered_pairs": 4}}

    monkeypatch.setattr(cli, "write_fusion_calibration", fake_write_fusion_calibration)

    result = CliRunner().invoke(
        app,
        [
            "calibrate-fusion-weights",
            "--rankings",
            str(rankings),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--grid-step",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert "best_pairwise_accuracy=0.750" in result.output
    assert "profile=unknown" in result.output


def test_cli_calibrate_fusion_weights_reports_v3_profile(tmp_path):
    rankings, labels = calibration_fixture(tmp_path)
    output = tmp_path / "calibration"

    result = CliRunner().invoke(
        app,
        [
            "calibrate-fusion-weights",
            "--rankings",
            str(rankings),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--grid-step",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert "profile=impact_index_v3_final_components" in result.output
    payload = json.loads((output / "fusion_calibration.json").read_text(encoding="utf-8"))
    assert payload["summary"]["weight_profile"] == "impact_index_v3_final_components"
