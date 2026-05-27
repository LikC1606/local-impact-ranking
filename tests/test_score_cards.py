import json

from local_impact_ranking.score_cards import (
    render_score_cards_markdown,
    score_card_report,
    write_score_cards,
)


def test_score_card_report_uses_paper_evidence_for_reportable_cards(tmp_path):
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "paper_evidence_report.json").write_text(
        json.dumps(
            {
                "credit_mode": "context_weighted",
                "papers": [
                    {
                        "paper_id": "r1",
                        "paper_title": "Important Method",
                        "final_score": 92.5,
                        "confidence_flag": "stable",
                        "support_band": "strong",
                        "max_single_edge_share": 0.4,
                        "score_ci_low": 85.0,
                        "score_ci_high": 98.0,
                        "score_ci_width": 13.0,
                        "rank_span": 2,
                        "zero_score_rate": 0.0,
                        "uncertainty_band": "moderate",
                        "source_edge_count": 3,
                        "top_source_edges": [
                            {
                                "target_paper_id": "p1",
                                "target_title": "Target Paper",
                                "rank": 1,
                                "credit": 1.5,
                                "context_count": 2,
                                "context_weight": 1.5,
                            }
                        ],
                    },
                    {
                        "paper_id": "r2",
                        "paper_title": "Fragile Method",
                        "final_score": 40.0,
                        "confidence_flag": "fragile",
                        "support_band": "thin",
                        "max_single_edge_share": 1.0,
                        "score_ci_low": 0.0,
                        "score_ci_high": 50.0,
                        "score_ci_width": 50.0,
                        "rank_span": 10,
                        "zero_score_rate": 0.4,
                        "uncertainty_band": "unstable",
                        "source_edge_count": 1,
                        "top_source_edges": [],
                    },
                ],
                "limitations": ["Local score only."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = score_card_report(run_dir, top_n=1, max_edges=1)

    assert report["run"] == "run"
    assert report["top_n"] == 1
    assert report["cards"][0]["paper_id"] == "r1"
    assert report["cards"][0]["headline_score"] == 92.5
    assert report["cards"][0]["score_interval"] == [85.0, 98.0]
    assert report["cards"][0]["credibility"]["level"] == "high"
    assert report["cards"][0]["top_evidence"][0]["target_paper_id"] == "p1"
    assert report["limitations"] == ["Local score only."]


def test_write_score_cards_exports_json_and_markdown(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "cards"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "paper_evidence_report.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "r1",
                        "paper_title": "Important Method",
                        "final_score": 100.0,
                        "confidence_flag": "stable",
                        "support_band": "strong",
                        "max_single_edge_share": 0.2,
                        "score_ci_low": 95.0,
                        "score_ci_high": 100.0,
                        "source_edge_count": 5,
                        "top_source_edges": [],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_score_cards(run_dir, output_dir, top_n=5)

    assert report["card_count"] == 1
    assert json.loads((output_dir / "score_cards.json").read_text())["cards"][0][
        "paper_id"
    ] == "r1"
    markdown = (output_dir / "score_cards.md").read_text(encoding="utf-8")
    assert "# Paper Score Cards" in markdown
    assert "Important Method" in markdown


def test_render_score_cards_markdown_handles_empty_cards():
    markdown = render_score_cards_markdown({"cards": [], "run": "empty"})

    assert markdown.startswith("# Paper Score Cards\n")
    assert "No score cards available." in markdown
