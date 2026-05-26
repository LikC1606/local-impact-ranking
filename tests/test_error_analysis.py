from local_impact_ranking.error_analysis import (
    gold_error_analysis_report,
    render_gold_error_analysis_markdown,
)


def test_gold_error_analysis_report_counts_hit_at_k_and_rank_buckets():
    gold_report = {
        "labeled_targets": 4,
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["r1"],
                "selected_reference_ids": ["r1", "r2", "r3", "r4", "r5"],
                "selected_positive_reference_ids": ["r1"],
                "hit": True,
            },
            {
                "target_paper_id": "p2",
                "positive_reference_ids": ["r7"],
                "selected_reference_ids": ["r3", "r4", "r7", "r8", "r9"],
                "selected_positive_reference_ids": ["r7"],
                "hit": True,
            },
            {
                "target_paper_id": "p3",
                "positive_reference_ids": ["r9"],
                "selected_reference_ids": ["r3", "r4", "r5", "r8", "r9"],
                "selected_positive_reference_ids": ["r9"],
                "hit": True,
            },
            {
                "target_paper_id": "p4",
                "positive_reference_ids": ["r10"],
                "selected_reference_ids": ["r3", "r4", "r5", "r8", "r9"],
                "selected_positive_reference_ids": [],
                "hit": False,
            },
        ],
    }

    report = gold_error_analysis_report(gold_report, max_failures=2)

    assert report["labeled_targets"] == 4
    assert report["hit_at"]["hit@1"] == 0.25
    assert report["hit_at"]["hit@3"] == 0.5
    assert report["hit_at"]["hit@5"] == 0.75
    assert report["first_positive_rank_buckets"] == {
        "rank_1": 1,
        "rank_2_to_3": 1,
        "rank_4_to_5": 1,
        "miss": 1,
    }
    assert report["failures"][0]["target_paper_id"] == "p4"


def test_gold_error_analysis_report_counts_candidate_coverage_and_miss_causes():
    gold_report = {
        "labeled_targets": 3,
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["r1"],
                "selected_reference_ids": ["r1", "r2"],
                "selected_positive_reference_ids": ["r1"],
                "hit": True,
            },
            {
                "target_paper_id": "p2",
                "positive_reference_ids": ["r3"],
                "selected_reference_ids": ["r4", "r5"],
                "selected_positive_reference_ids": [],
                "hit": False,
            },
            {
                "target_paper_id": "p3",
                "positive_reference_ids": ["r6"],
                "selected_reference_ids": ["r4", "r5"],
                "selected_positive_reference_ids": [],
                "hit": False,
            },
        ],
    }

    report = gold_error_analysis_report(
        gold_report,
        candidate_ids_by_target={
            "p1": ["r1", "r2"],
            "p2": ["r3", "r4", "r5"],
            "p3": ["r7", "r8"],
        },
        max_failures=5,
    )

    assert report["candidate_coverage"] == {
        "available": True,
        "targets_with_candidate_data": 3,
        "gold_references": 3,
        "gold_in_candidates": 2,
        "gold_candidate_recall": 2 / 3,
    }
    assert report["miss_decomposition"] == {
        "miss_count": 2,
        "miss_gold_absent_from_candidates": 1,
        "miss_gold_present_but_not_selected": 1,
        "miss_candidate_status_unknown": 0,
    }
    assert report["candidate_position_buckets"] == {
        "gold_position_1": 2,
        "gold_position_2_to_5": 0,
        "gold_position_6_to_10": 0,
        "gold_position_11_to_20": 0,
        "gold_position_21_plus": 0,
        "gold_position_missing": 1,
    }
    assert report["miss_candidate_position_buckets"] == {
        "gold_position_1": 1,
        "gold_position_2_to_5": 0,
        "gold_position_6_to_10": 0,
        "gold_position_11_to_20": 0,
        "gold_position_21_plus": 0,
        "gold_position_missing": 1,
    }
    assert report["failures"][0]["target_paper_id"] == "p2"
    assert report["failures"][0]["candidate_reference_ids"] == ["r3", "r4", "r5"]
    assert report["failures"][0]["miss_reason"] == "gold_present_but_not_selected"
    assert report["failures"][0]["gold_candidate_positions"] == {"r3": 1}
    assert report["failures"][1]["target_paper_id"] == "p3"
    assert report["failures"][1]["candidate_reference_ids"] == ["r7", "r8"]
    assert report["failures"][1]["miss_reason"] == "gold_absent_from_candidates"
    assert report["failures"][1]["gold_candidate_positions"] == {"r6": None}


def test_gold_error_analysis_report_marks_miss_cause_unknown_without_candidate_data():
    gold_report = {
        "labeled_targets": 1,
        "targets": [
            {
                "target_paper_id": "p1",
                "positive_reference_ids": ["r1"],
                "selected_reference_ids": ["r2"],
                "selected_positive_reference_ids": [],
                "hit": False,
            }
        ],
    }

    report = gold_error_analysis_report(gold_report, max_failures=1)

    assert report["candidate_coverage"]["available"] is False
    assert report["miss_decomposition"] == {
        "miss_count": 1,
        "miss_gold_absent_from_candidates": 0,
        "miss_gold_present_but_not_selected": 0,
        "miss_candidate_status_unknown": 1,
    }
    assert report["failures"][0]["miss_reason"] == "candidate_status_unknown"


def test_render_gold_error_analysis_markdown_contains_tables():
    markdown = render_gold_error_analysis_markdown(
        {
            "labeled_targets": 4,
            "hit_at": {"hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.75},
            "first_positive_rank_buckets": {
                "rank_1": 1,
                "rank_2_to_3": 1,
                "rank_4_to_5": 1,
                "miss": 1,
            },
            "failures": [
                {
                    "target_paper_id": "p4",
                    "positive_reference_ids": ["r10"],
                    "selected_reference_ids": ["r3", "r4"],
                    "candidate_reference_ids": ["r3", "r4", "r10"],
                    "miss_reason": "gold_present_but_not_selected",
                }
            ],
            "candidate_coverage": {
                "available": True,
                "targets_with_candidate_data": 4,
                "gold_references": 4,
                "gold_in_candidates": 3,
                "gold_candidate_recall": 0.75,
            },
            "miss_decomposition": {
                "miss_count": 1,
                "miss_gold_absent_from_candidates": 0,
                "miss_gold_present_but_not_selected": 1,
                "miss_candidate_status_unknown": 0,
            },
            "candidate_position_buckets": {
                "gold_position_1": 0,
                "gold_position_2_to_5": 1,
                "gold_position_6_to_10": 0,
                "gold_position_11_to_20": 0,
                "gold_position_21_plus": 0,
                "gold_position_missing": 0,
            },
            "miss_candidate_position_buckets": {
                "gold_position_1": 0,
                "gold_position_2_to_5": 1,
                "gold_position_6_to_10": 0,
                "gold_position_11_to_20": 0,
                "gold_position_21_plus": 0,
                "gold_position_missing": 0,
            },
        }
    )

    assert markdown.startswith("# Gold Error Analysis\n")
    assert "| hit@1 | 0.250 |" in markdown
    assert "| miss | 1 |" in markdown
    assert "| gold_candidate_recall | 0.750 |" in markdown
    assert "| miss_gold_present_but_not_selected | 1 |" in markdown
    assert "## Gold Candidate Positions" in markdown
    assert "| gold_position_2_to_5 | 1 |" in markdown
    assert "| p4 | r10 | r3, r4 | r3, r4, r10 | gold_present_but_not_selected |" in markdown
