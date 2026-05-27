import csv

from local_impact_ranking.baselines import (
    compute_bibliometric_baselines,
    export_baseline_rankings,
    summarize_baseline_comparison,
)
from local_impact_ranking.models import Paper


def sample_papers():
    return [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Paper One",
                "year": 2020,
                "field": "AI",
                "citation_count": 10,
                "references": [
                    {"reference_id": "p2", "title": "Paper Two"},
                    {"reference_id": "p3", "title": "Paper Three"},
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Paper Two",
                "year": 2020,
                "field": "AI",
                "metadata": {"citation_count": 5},
                "references": [{"reference_id": "p3", "title": "Paper Three"}],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Paper Three",
                "year": 2021,
                "field": "AI",
                "citation_count": 2,
                "references": [],
            }
        ),
    ]


def test_compute_bibliometric_baselines_reads_citation_counts_and_normalizes_by_field_year():
    baselines = compute_bibliometric_baselines(sample_papers())

    assert baselines.citation_count["p1"] == 10.0
    assert baselines.citation_count["p2"] == 5.0
    assert baselines.field_year_normalized_citation["p1"] == 10.0 / 7.5
    assert baselines.field_year_normalized_citation["p2"] == 5.0 / 7.5
    assert baselines.field_year_normalized_citation["p3"] == 1.0


def test_compute_bibliometric_baselines_page_rank_rewards_incoming_reference_edges():
    baselines = compute_bibliometric_baselines(sample_papers(), pagerank_iterations=50)

    assert baselines.pagerank["p3"] > baselines.pagerank["p2"]
    assert baselines.pagerank["p2"] > baselines.pagerank["p1"]


def test_export_baseline_rankings_writes_one_csv_per_baseline(tmp_path):
    baselines = compute_bibliometric_baselines(sample_papers())

    export_baseline_rankings(baselines, tmp_path)

    assert (tmp_path / "citation_count.csv").exists()
    assert (tmp_path / "pagerank.csv").exists()
    assert (tmp_path / "field_year_normalized_citation.csv").exists()
    with (tmp_path / "citation_count.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "p1"
    assert rows[0]["score"] == "10.000000"


def test_summarize_baseline_comparison_reports_overlap_with_llm_scores():
    baselines = compute_bibliometric_baselines(sample_papers())

    summary = summarize_baseline_comparison(
        llm_scores={"p3": 2.0, "p1": 1.0},
        baselines=baselines,
        top_n=2,
    )

    assert summary["top_n"] == 2
    assert summary["baselines"]["citation_count"]["top_n_overlap"] == 1
    assert summary["baselines"]["pagerank"]["top_n_overlap"] == 1


def test_summarize_baseline_comparison_reports_final_score_rank_alignment():
    baselines = compute_bibliometric_baselines(sample_papers())

    summary = summarize_baseline_comparison(
        llm_scores={"p3": 2.0, "p1": 1.0},
        baselines=baselines,
        final_scores={"p1": 90.0, "p2": 50.0, "p3": 10.0},
        top_n=2,
    )

    citation_count = summary["baselines"]["citation_count"]
    assert citation_count["raw_top_n_overlap"] == 1
    assert citation_count["final_top_n_overlap"] == 2
    assert citation_count["final_spearman"] == 1.0
    assert citation_count["final_kendall_tau"] == 1.0
    pagerank = summary["baselines"]["pagerank"]
    assert pagerank["final_spearman"] < 0
    assert pagerank["final_kendall_tau"] < 0
