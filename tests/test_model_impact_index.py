import csv
import json
import pytest
import threading
import time
from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
import local_impact_ranking.model_impact_index as model_impact_index_module
from local_impact_ranking.cli import app
from local_impact_ranking.impact_fusion import compute_impact_index_v2
from local_impact_ranking.model_impact_index import (
    aggregate_pairwise_impact_index,
    build_direct_impact_score_prompt,
    build_pairwise_impact_prompt,
    final_score_bootstrap_stability_report,
    generate_model_direct_scores,
    generate_model_pairwise_judgments,
    model_score_dataset,
    model_score_preflight_report,
    pair_budget_plan,
    pairwise_comparison_plan,
    pairwise_bootstrap_stability_report,
    pairwise_ranking_quality_report,
    parse_direct_impact_score,
    render_final_score_stability_markdown,
    render_pairwise_stability_markdown,
    write_model_score_preflight,
)
from local_impact_ranking.models import Paper


class PairwiseClient:
    def __init__(self, winners):
        self.winners = winners
        self.payloads = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)
        content = json["messages"][0]["content"]
        pair = None
        for candidate in self.winners:
            if f"Paper A ID: {candidate[0]}" in content and f"Paper B ID: {candidate[1]}" in content:
                pair = candidate
                break
        winner = self.winners[pair]

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": f"Answer: {winner}"}}]}

        return Response()


class DirectScoreClient:
    def __init__(self, scores):
        self.scores = scores
        self.payloads = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)
        content = json["messages"][0]["content"]
        paper_id = None
        for candidate in self.scores:
            if f"Paper ID: {candidate}" in content:
                paper_id = candidate
                break
        score = self.scores[paper_id]

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": f"Score: {score}"}}]}

        return Response()


def paper(paper_id, title):
    return Paper.model_validate({"paper_id": paper_id, "title": title, "references": []})


def test_generate_model_direct_scores_calls_model_and_parses_numeric_scores():
    papers = [paper("p1", "One"), paper("p2", "Two")]
    client = DirectScoreClient({"p1": 91, "p2": 37})

    rows = generate_model_direct_scores(
        papers,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
    )

    assert rows == [
        {
            "paper_id": "p1",
            "title": "One",
            "direct_score": 91.0,
            "parseable": True,
            "prediction": "Score: 91",
            "model": "local-impact",
        },
        {
            "paper_id": "p2",
            "title": "Two",
            "direct_score": 37.0,
            "parseable": True,
            "prediction": "Score: 37",
            "model": "local-impact",
        },
    ]
    assert "Return exactly one line using one of these calibrated bins" in client.payloads[0]["messages"][0]["content"]
    assert client.payloads[0]["structured_outputs"] == {
        "choice": [
            "Score: 0",
            "Score: 20",
            "Score: 40",
            "Score: 60",
            "Score: 80",
            "Score: 100",
        ]
    }


def test_parse_direct_impact_score_rejects_missing_or_out_of_range_values():
    assert parse_direct_impact_score("Score: 88.5") == 88.5
    assert parse_direct_impact_score("88/100") == 88.0
    assert parse_direct_impact_score("Score: 140") is None
    assert parse_direct_impact_score("Score: -5") is None
    assert parse_direct_impact_score("high impact") is None


def test_generate_model_pairwise_judgments_calls_model_and_parses_winners():
    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]
    client = PairwiseClient(
        {
            ("p1", "p2"): "B",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    judgments = generate_model_pairwise_judgments(
        papers,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
    )

    assert [judgment["winner_id"] for judgment in judgments] == ["p2", "p1", "p2"]
    assert all(judgment["parseable"] for judgment in judgments)
    assert "Which paper has higher scholarly impact" in client.payloads[0]["messages"][0]["content"]
    assert client.payloads[0]["structured_outputs"] == {
        "choice": ["Answer: A", "Answer: B"]
    }
    assert client.payloads[0]["logprobs"] is True
    assert client.payloads[0]["top_logprobs"] == 2


def test_pairwise_prompt_truncates_long_abstracts_for_local_context_budget():
    long_text = "A" * 9000
    prompt = build_pairwise_impact_prompt(
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": long_text,
                "abstract": long_text,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": long_text,
                "abstract": long_text,
                "references": [],
            }
        ),
    )

    assert len(prompt) < 5000
    assert "[truncated]" in prompt
    assert "Answer: A or Answer: B" in prompt


def test_direct_prompt_truncates_long_abstract_for_local_context_budget():
    prompt = build_direct_impact_score_prompt(
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "A" * 9000,
                "abstract": "A" * 9000,
                "references": [],
            }
        )
    )

    assert len(prompt) < 4000
    assert "[truncated]" in prompt
    assert "Score: 100" in prompt


def test_generate_model_pairwise_judgments_records_logprob_margin_confidence():
    class LogprobPairwiseClient:
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
                                "message": {"content": "Answer: A"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " A",
                                            "logprob": -0.2,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -0.2},
                                                {"token": " B", "logprob": -2.2},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    judgments = generate_model_pairwise_judgments(
        [paper("p1", "One"), paper("p2", "Two")],
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=LogprobPairwiseClient(),
    )

    assert judgments[0]["winner_id"] == "p1"
    assert judgments[0]["confidence_method"] == "choice_logprob_margin"
    assert judgments[0]["choice_logprob_margin"] == 2.0
    assert judgments[0]["confidence"] == 0.880797
    assert judgments[0]["model"] == "local-impact"


def test_generate_model_pairwise_judgments_can_use_bidirectional_consistency():
    class BidirectionalClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)
            content = json["messages"][0]["content"]
            if "Paper A ID: p1" in content and "Paper B ID: p2" in content:
                answer = "Answer: A"
            elif "Paper A ID: p2" in content and "Paper B ID: p1" in content:
                answer = "Answer: B"
            else:
                raise AssertionError(content)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": answer}}]}

            return Response()

    client = BidirectionalClient()

    judgments = generate_model_pairwise_judgments(
        [paper("p1", "One"), paper("p2", "Two")],
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        bidirectional_pairwise=True,
    )

    assert len(client.payloads) == 2
    assert judgments == [
        {
            "pair_index": 1,
            "round": 1,
            "paper_a_id": "p1",
            "paper_b_id": "p2",
            "winner_side": "A",
            "winner_id": "p1",
            "loser_id": "p2",
            "parseable": True,
            "prediction": "Answer: A",
            "model": "local-impact",
            "confidence": 0.5,
            "confidence_method": "bidirectional_consistent",
            "choice_logprob_margin": None,
            "bidirectional": True,
            "decisive": True,
            "forward_winner_side": "A",
            "reverse_winner_side": "B",
            "reverse_mapped_winner_side": "A",
            "position_consistency": "consistent",
            "selected_source": "consistent",
            "forward_prediction": "Answer: A",
            "reverse_prediction": "Answer: B",
            "forward_confidence": 0.5,
            "reverse_confidence": 0.5,
        }
    ]


def test_generate_model_pairwise_judgments_marks_bidirectional_conflicts_nondecisive():
    class ConflictingBidirectionalClient:
        def post(self, url, headers, json, timeout):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "Answer: A"}}]}

            return Response()

    judgments = generate_model_pairwise_judgments(
        [paper("p1", "One"), paper("p2", "Two")],
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=ConflictingBidirectionalClient(),
        bidirectional_pairwise=True,
        bidirectional_confidence_margin=0.05,
    )

    assert judgments[0]["winner_side"] is None
    assert judgments[0]["winner_id"] is None
    assert judgments[0]["loser_id"] is None
    assert judgments[0]["parseable"] is False
    assert judgments[0]["decisive"] is False
    assert judgments[0]["forward_winner_side"] == "A"
    assert judgments[0]["reverse_winner_side"] == "A"
    assert judgments[0]["reverse_mapped_winner_side"] == "B"
    assert judgments[0]["position_consistency"] == "conflict"
    assert judgments[0]["selected_source"] == "conflict_tie"
    assert judgments[0]["confidence"] == 0.0
    assert judgments[0]["confidence_method"] == "bidirectional_conflict"


def test_model_score_dataset_applies_empirical_confidence_calibration(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    calibration_report = tmp_path / "calibration.json"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    calibration_report.write_text(
        json.dumps(
            {
                "calibration": {
                    "bins": [
                        {
                            "lower": 0.8,
                            "upper": 1.0,
                            "count": 30,
                            "accuracy": 0.76,
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class LogprobPairwiseClient:
        def post(self, url, headers, json, timeout):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {"content": "Answer: A"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " A",
                                            "logprob": -0.2,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -0.2},
                                                {"token": " B", "logprob": -2.2},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=LogprobPairwiseClient(),
        calibration_report_path=calibration_report,
    )

    assert report["summary"]["calibration_model"] == {
        "method": "empirical_bin_accuracy",
        "source_path": str(calibration_report),
        "bin_count": 1,
    }
    judgments = [
        json.loads(line)
        for line in (output_dir / "pairwise_judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert judgments[0]["confidence"] == 0.880797
    assert judgments[0]["calibrated_confidence"] == 0.76
    assert judgments[0]["calibrated_confidence_method"] == "empirical_bin_accuracy"
    by_id = {row["paper_id"]: row for row in report["rankings"]}
    assert by_id["p1"]["mean_pairwise_confidence"] == 0.880797
    assert by_id["p1"]["mean_calibrated_pairwise_confidence"] == 0.76
    assert by_id["p1"]["confidence"] == 0.76
    assert by_id["p1"]["confidence_source"] == "calibrated_pairwise"
    impact_csv = (output_dir / "impact_scores.csv").read_text(encoding="utf-8")
    assert "mean_calibrated_pairwise_confidence" in impact_csv
    assert "confidence_source" in impact_csv


def test_builtin_sciimpact_calibration_profile_can_calibrate_dataset_scores(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    calibration_report = Path(
        "configs/calibration/sciimpact-qwen32b-heldout-p50.json"
    )
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    class LogprobPairwiseClient:
        def post(self, url, headers, json, timeout):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {"content": "Answer: A"},
                                "logprobs": {
                                    "content": [
                                        {
                                            "token": " A",
                                            "logprob": -0.2,
                                            "top_logprobs": [
                                                {"token": " A", "logprob": -0.2},
                                                {"token": " B", "logprob": -2.2},
                                            ],
                                        }
                                    ]
                                },
                            }
                        ]
                    }

            return Response()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=LogprobPairwiseClient(),
        calibration_report_path=calibration_report,
    )

    assert report["summary"]["calibration_model"] == {
        "method": "empirical_bin_accuracy",
        "source_path": str(calibration_report),
        "bin_count": 3,
    }
    assert report["rankings"][0]["mean_calibrated_pairwise_confidence"] == 0.754717
    assert report["rankings"][0]["confidence_source"] == "calibrated_pairwise"


def test_generate_model_pairwise_judgments_can_run_concurrently():
    class SlowPairwiseClient(PairwiseClient):
        def __init__(self, winners):
            super().__init__(winners)
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def post(self, url, headers, json, timeout):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return super().post(url, headers, json, timeout)
            finally:
                with self.lock:
                    self.active -= 1

    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]
    client = SlowPairwiseClient(
        {
            ("p1", "p2"): "B",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    judgments = generate_model_pairwise_judgments(
        papers,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        concurrency=2,
    )

    assert client.max_active == 2
    assert [judgment["pair_index"] for judgment in judgments] == [1, 2, 3]
    assert [judgment["winner_id"] for judgment in judgments] == ["p2", "p1", "p2"]


def test_aggregate_pairwise_impact_index_combines_coverage_and_pairwise_confidence():
    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]
    rankings = aggregate_pairwise_impact_index(
        papers,
        [
            {
                "winner_id": "p1",
                "loser_id": "p2",
                "parseable": True,
                "confidence": 0.9,
            },
            {
                "winner_id": "p1",
                "loser_id": "p3",
                "parseable": True,
                "confidence": 0.7,
            },
        ],
    )

    by_id = {row["paper_id"]: row for row in rankings}
    assert by_id["p1"]["comparisons"] == 2
    assert by_id["p1"]["coverage_confidence"] == 1.0
    assert by_id["p1"]["mean_pairwise_confidence"] == 0.8
    assert by_id["p1"]["confidence"] == 0.8
    assert by_id["p2"]["coverage_confidence"] == 0.5
    assert by_id["p2"]["mean_pairwise_confidence"] == 0.9
    assert by_id["p2"]["confidence"] == 0.45


def test_aggregate_pairwise_impact_index_is_order_independent_by_default():
    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]
    judgments = [
        {"winner_id": "p1", "loser_id": "p2", "parseable": True, "confidence": 0.9},
        {"winner_id": "p2", "loser_id": "p3", "parseable": True, "confidence": 0.9},
        {"winner_id": "p3", "loser_id": "p1", "parseable": True, "confidence": 0.9},
    ]

    forward = aggregate_pairwise_impact_index(papers, judgments)
    reversed_order = aggregate_pairwise_impact_index(papers, list(reversed(judgments)))

    forward_scores = {row["paper_id"]: row["impact_index"] for row in forward}
    reversed_scores = {row["paper_id"]: row["impact_index"] for row in reversed_order}

    assert forward_scores == reversed_scores
    assert {row["pairwise_aggregation_method"] for row in forward} == {
        "rank_centrality"
    }
    assert all("elo_rating" in row for row in forward)


def test_pairwise_bootstrap_stability_report_tracks_rank_sensitivity():
    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]
    judgments = [
        {"winner_id": "p1", "loser_id": "p2", "parseable": True, "confidence": 0.9},
        {"winner_id": "p1", "loser_id": "p3", "parseable": True, "confidence": 0.9},
        {"winner_id": "p2", "loser_id": "p3", "parseable": True, "confidence": 0.9},
    ]
    base_rankings = aggregate_pairwise_impact_index(papers, judgments)

    report = pairwise_bootstrap_stability_report(
        papers,
        judgments,
        base_rankings,
        samples=30,
        seed=3,
        top_n=2,
        extra_tracked_paper_ids=["p3"],
    )

    assert report["summary"]["method"] == "pairwise_judgment_bootstrap"
    assert report["summary"]["samples"] == 30
    assert report["summary"]["tracked_paper_count"] == 3
    assert 0.0 <= report["summary"]["mean_top_n_retention_rate"] <= 1.0
    assert report["papers"][0]["paper_id"] == base_rankings[0]["paper_id"]
    assert {row["paper_id"] for row in report["papers"]} == {"p1", "p2", "p3"}
    assert report["papers"][0]["score_ci_low"] <= report["papers"][0]["score_ci_high"]
    assert report["papers"][0]["stability_band"] in {"stable", "moderate", "unstable"}
    markdown = render_pairwise_stability_markdown(report)
    assert "# Pairwise Stability" in markdown
    assert "mean_top_n_retention_rate" in markdown


def test_final_score_bootstrap_stability_report_tracks_fused_score_sensitivity():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "High citation method",
                "citation_count": 100,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Model preferred benchmark",
                "citation_count": 30,
                "references": [{"reference_id": "p1", "title": "High citation method"}],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Narrow follow-up",
                "citation_count": 1,
                "references": [
                    {"reference_id": "p2", "title": "Model preferred benchmark"}
                ],
            }
        ),
    ]
    judgments = [
        {"winner_id": "p1", "loser_id": "p2", "parseable": True, "confidence": 0.9},
        {"winner_id": "p1", "loser_id": "p3", "parseable": True, "confidence": 0.9},
        {"winner_id": "p2", "loser_id": "p3", "parseable": True, "confidence": 0.9},
    ]
    pairwise_rankings = aggregate_pairwise_impact_index(papers, judgments)
    base_rankings = compute_impact_index_v2(papers, pairwise_rankings)["rankings"]

    report = final_score_bootstrap_stability_report(
        papers,
        judgments,
        base_rankings,
        samples=30,
        seed=5,
        top_n=2,
    )

    assert report["summary"]["method"] == "final_score_pairwise_bootstrap"
    assert report["summary"]["samples"] == 30
    assert report["summary"]["tracked_paper_count"] == 2
    assert 0.0 <= report["summary"]["mean_top_n_retention_rate"] <= 1.0
    assert report["papers"][0]["paper_id"] == base_rankings[0]["paper_id"]
    assert report["papers"][0]["base_final_impact_score"] >= 0.0
    assert report["papers"][0]["score_ci_low"] <= report["papers"][0]["score_ci_high"]
    assert report["papers"][0]["stability_band"] in {"stable", "moderate", "unstable"}
    markdown = render_final_score_stability_markdown(report)
    assert "# Final Score Stability" in markdown
    assert "base_final_impact_score" in markdown
    assert "citation and external evidence are held fixed" in markdown


def test_pairwise_comparison_plan_supports_rounds_shuffle_and_max_pairs():
    papers = [paper("p1", "One"), paper("p2", "Two"), paper("p3", "Three")]

    plan = pairwise_comparison_plan(
        papers,
        comparison_rounds=2,
        shuffle_pairs=True,
        seed=11,
        max_pairs=4,
    )

    assert len(plan) == 4
    assert {item["round"] for item in plan} == {1, 2}
    assert all(item["paper_a"].paper_id != item["paper_b"].paper_id for item in plan)
    assert [item["pair_index"] for item in plan] == [1, 2, 3, 4]


def test_pairwise_comparison_plan_can_prioritize_coverage_under_pair_budget():
    papers = [paper(f"p{i}", f"Paper {i}") for i in range(1, 7)]

    plan = pairwise_comparison_plan(
        papers,
        max_pairs=3,
        pair_sampling="coverage",
    )

    compared_ids = {
        item["paper_a"].paper_id
        for item in plan
    } | {
        item["paper_b"].paper_id
        for item in plan
    }
    assert len(plan) == 3
    assert compared_ids == {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert [(item["paper_a"].paper_id, item["paper_b"].paper_id) for item in plan] == [
        ("p1", "p2"),
        ("p3", "p4"),
        ("p5", "p6"),
    ]


def test_pairwise_comparison_plan_shuffles_coverage_without_losing_early_coverage():
    papers = [paper(f"p{i}", f"Paper {i}") for i in range(1, 7)]

    plan = pairwise_comparison_plan(
        papers,
        max_pairs=3,
        pair_sampling="coverage",
        shuffle_pairs=True,
        seed=7,
    )

    compared_ids = {
        item["paper_a"].paper_id
        for item in plan
    } | {
        item["paper_b"].paper_id
        for item in plan
    }
    assert len(plan) == 3
    assert compared_ids == {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert [(item["paper_a"].paper_id, item["paper_b"].paper_id) for item in plan] != [
        ("p1", "p2"),
        ("p3", "p4"),
        ("p5", "p6"),
    ]


def test_pairwise_comparison_plan_balances_coverage_prefix_after_initial_pass():
    papers = [paper(f"p{i}", f"Paper {i}") for i in range(1, 11)]

    plan = pairwise_comparison_plan(
        papers,
        max_pairs=20,
        pair_sampling="coverage",
    )

    comparison_counts = Counter()
    for item in plan:
        comparison_counts[item["paper_a"].paper_id] += 1
        comparison_counts[item["paper_b"].paper_id] += 1

    assert [(item["paper_a"].paper_id, item["paper_b"].paper_id) for item in plan[:5]] == [
        ("p1", "p2"),
        ("p3", "p4"),
        ("p5", "p6"),
        ("p7", "p8"),
        ("p9", "p10"),
    ]
    assert len(plan) == 20
    assert min(comparison_counts.values()) >= 3
    assert max(comparison_counts.values()) - min(comparison_counts.values()) <= 2


def test_pairwise_comparison_plan_connects_coverage_prefix_before_balancing():
    papers = [paper(f"p{i}", f"Paper {i}") for i in range(1, 11)]

    plan = pairwise_comparison_plan(
        papers,
        max_pairs=9,
        pair_sampling="coverage",
    )

    assert len(plan) == 9
    assert _connected_component_count_from_plan(plan) == 1


def test_pairwise_comparison_plan_respects_budget_without_all_pair_materialization(
    monkeypatch,
):
    papers = [paper(f"p{i}", f"Paper {i}") for i in range(1, 101)]

    def fail_if_all_pairs_are_requested(*_args, **_kwargs):
        raise AssertionError("pairwise plan must not materialize all paper pairs")

    monkeypatch.setattr(
        model_impact_index_module,
        "combinations",
        fail_if_all_pairs_are_requested,
        raising=False,
    )

    plan = pairwise_comparison_plan(
        papers,
        max_pairs=50,
        pair_sampling="coverage",
    )

    compared_ids = {
        item["paper_a"].paper_id for item in plan
    } | {
        item["paper_b"].paper_id for item in plan
    }
    assert len(plan) == 50
    assert len(compared_ids) == 100
    assert [item["pair_index"] for item in plan] == list(range(1, 51))


def _connected_component_count_from_plan(plan):
    adjacency = {}
    for item in plan:
        left = item["paper_a"].paper_id
        right = item["paper_b"].paper_id
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    visited = set()
    component_count = 0
    for paper_id in adjacency:
        if paper_id in visited:
            continue
        component_count += 1
        stack = [paper_id]
        visited.add(paper_id)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
    return component_count


def test_pairwise_comparison_plan_rejects_unknown_sampling_strategy():
    try:
        pairwise_comparison_plan(
            [paper("p1", "One"), paper("p2", "Two")],
            pair_sampling="unknown",
        )
    except ValueError as exc:
        assert "pair_sampling" in str(exc)
    else:
        raise AssertionError("Expected invalid pair_sampling to fail")


def test_pair_budget_plan_handles_tiny_datasets_and_rejects_invalid_options():
    assert pair_budget_plan(1, pair_budget="auto")["effective_max_pairs"] == 0
    try:
        pair_budget_plan(3, pair_budget="unsupported")
    except ValueError as exc:
        assert "pair_budget" in str(exc)
    else:
        raise AssertionError("Expected invalid pair_budget to fail")

    try:
        pair_budget_plan(3, pair_budget="auto", target_paper_coverage=0.0)
    except ValueError as exc:
        assert "target_paper_coverage" in str(exc)
    else:
        raise AssertionError("Expected invalid target_paper_coverage to fail")


def test_model_score_dataset_writes_direct_scores_when_enabled(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "abstract": "Middle", "references": []},
        {"paper_id": "p2", "title": "Two", "abstract": "Best", "references": []},
        {"paper_id": "p3", "title": "Three", "abstract": "Lowest", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    class HybridClient:
        def __init__(self):
            self.direct = DirectScoreClient({"p1": 55, "p2": 94, "p3": 12})
            self.pairwise = PairwiseClient(
                {
                    ("p1", "p2"): "B",
                    ("p1", "p3"): "A",
                    ("p2", "p3"): "A",
                }
            )

        def post(self, url, headers, json, timeout):
            content = json["messages"][0]["content"]
            if "Paper ID:" in content:
                return self.direct.post(url, headers, json, timeout)
            return self.pairwise.post(url, headers, json, timeout)

    client = HybridClient()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        direct_score=True,
    )

    assert report["summary"]["paper_count"] == 3
    assert report["summary"]["direct_scoring_enabled"] is True
    assert report["summary"]["model_pairwise_scoring_method"] == "rank_centrality_v1"
    assert report["summary"]["parseable_direct_score_count"] == 3
    assert report["summary"]["direct_score_quality"]["status"] == "usable"
    assert report["summary"]["judgment_count"] == 3
    assert report["summary"]["parseable_rate"] == 1.0
    by_id = {row["paper_id"]: row for row in report["rankings"]}
    assert report["rankings"][0]["paper_id"] == "p1"
    assert by_id["p2"]["impact_index"] == 100.0
    assert by_id["p2"]["direct_score"] == 94.0
    assert by_id["p2"]["final_impact_score"] == 0.0
    assert by_id["p2"]["final_score_missing_signals"] == [
        "true_citation",
        "effective_citation",
    ]
    assert "coverage_confidence" in by_id["p2"]
    assert "mean_pairwise_confidence" in by_id["p2"]
    assert by_id["p3"]["direct_score"] == 12.0
    assert (output_dir / "direct_scores.csv").exists()
    assert (output_dir / "direct_scores.jsonl").exists()
    assert (output_dir / "impact_scores.csv").exists()
    assert (output_dir / "impact_scores.jsonl").exists()
    assert (output_dir / "pairwise_judgments.jsonl").exists()
    assert (output_dir / "model_impact_report.md").exists()
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "Coverage Confidence" in report_markdown
    assert "Mean Pairwise Confidence" in report_markdown


def test_model_score_dataset_sanitizes_formula_like_csv_cells(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": '=HYPERLINK("http://example.test","paper")',
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Ordinary paper",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=DirectScoreClient({"p1": 55, "p2": 44}),
        direct_score=True,
        max_pairs=0,
    )

    with (output_dir / "impact_scores.csv").open(newline="", encoding="utf-8") as handle:
        impact_rows = list(csv.DictReader(handle))
    with (output_dir / "direct_scores.csv").open(newline="", encoding="utf-8") as handle:
        direct_rows = list(csv.DictReader(handle))

    assert impact_rows[0]["title"].startswith("'=HYPERLINK")
    assert direct_rows[0]["title"].startswith("'=HYPERLINK")
    assert impact_rows[1]["title"] == "Ordinary paper"


def test_model_score_dataset_writes_fused_impact_index_v2_components(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    (output_dir / "reports").mkdir(parents=True)
    (output_dir / "impact_claim_certificate.json").write_text(
        json.dumps({"recommended_next_actions": ["stale certificate"]}) + "\n",
        encoding="utf-8",
    )
    records = [
        {
            "paper_id": "p1",
            "title": "Foundational method",
            "year": 2020,
            "field": "AI",
            "citation_count": 100,
            "references": [],
        },
        {
            "paper_id": "p2",
            "title": "Adopted benchmark",
            "year": 2020,
            "field": "AI",
            "metadata": {"citation_count": 30},
            "references": [{"reference_id": "p1", "title": "Foundational method"}],
            "citation_contexts": [
                {
                    "reference_id": "p1",
                    "sentence": "We build on this method as a baseline.",
                    "section": "Methods",
                }
            ],
        },
        {
            "paper_id": "p3",
            "title": "Narrow follow-up",
            "year": 2020,
            "field": "AI",
            "citation_count": 1,
            "references": [{"reference_id": "p2", "title": "Adopted benchmark"}],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "B",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        pairwise_stability_samples=30,
        pairwise_stability_top_n=2,
        final_score_stability_samples=30,
        final_score_stability_top_n=2,
    )

    assert report["summary"]["fusion_scoring_enabled"] is True
    assert report["summary"]["scoring_method"] == "true_effective_citation_impact_v1"
    assert report["summary"]["legacy_scoring_method"] == "impact_index_v3_evidence_first"
    assert report["summary"]["model_pairwise_scoring_method"] == "rank_centrality_v1"
    assert report["summary"]["pairwise_stability"]["samples"] == 30
    assert report["summary"]["pairwise_stability"]["tracked_paper_count"] == 2
    assert report["summary"]["final_score_stability"]["samples"] == 30
    assert report["summary"]["final_score_stability"]["tracked_paper_count"] == 2
    assert report["summary"]["fusion"]["version"] == "true_effective_citation_impact_v1"
    assert report["summary"]["fusion"]["legacy_fusion_version"] == "impact_index_v3_evidence_first"
    assert "evidence_signal_audit" in report["summary"]["fusion"]["evidence_first"]
    assert "impact_index_v3" in report["rankings"][0]
    assert "impact_index_v2" in report["rankings"][0]
    assert "true_citation_score" in report["rankings"][0]
    assert "true_citation_raw" in report["rankings"][0]
    assert "true_citation_source" in report["rankings"][0]
    assert "true_citation_openalex_raw" in report["rankings"][0]
    assert "true_citation_semantic_scholar_raw" in report["rankings"][0]
    assert "true_citation_crossref_raw" in report["rankings"][0]
    assert "true_citation_crossref_data_citations_raw" in report["rankings"][0]
    assert "true_citation_europepmc_raw" in report["rankings"][0]
    assert "true_citation_ncbi_pubmed_raw" in report["rankings"][0]
    assert "true_citation_datacite_raw" in report["rankings"][0]
    assert "true_citation_datacite_events_raw" in report["rankings"][0]
    assert "true_citation_openaire_raw" in report["rankings"][0]
    assert "true_citation_wikidata_raw" in report["rankings"][0]
    assert "true_citation_source_count" in report["rankings"][0]
    assert "true_citation_source_conflict_flag" in report["rankings"][0]
    assert "true_citation_identity_status" in report["rankings"][0]
    assert "true_citation_identity_similarity" in report["rankings"][0]
    assert "true_citation_identity_external_title" in report["rankings"][0]
    assert "true_citation_identity_reason" in report["rankings"][0]
    assert "effective_citation_score" in report["rankings"][0]
    assert "final_score_method" in report["rankings"][0]
    assert "score_95_true_05_effective" in report["rankings"][0]
    assert "score_90_true_10_effective" in report["rankings"][0]
    assert "score_50_true_50_effective" in report["rankings"][0]
    assert "score_70_true_30_effective" in report["rankings"][0]
    assert "score_80_true_20_effective" in report["rankings"][0]
    assert "score_30_true_70_effective" in report["rankings"][0]
    assert "credibility_adjusted_impact_score" in report["rankings"][0]
    assert "credibility_adjustment_factor" in report["rankings"][0]
    assert "evidence_supported_score" in report["rankings"][0]
    assert "model_supported_score" in report["rankings"][0]
    assert "positive_evidence_signal_count" in report["rankings"][0]
    assert "zero_evidence_signal_count" in report["rankings"][0]
    assert "external_adoption_score" in report["rankings"][0]
    assert "bip_impact_score" in report["rankings"][0]
    assert "credible_evidence_tier" in report["rankings"][0]
    assert "missing_high_value_evidence" in report["rankings"][0]
    assert "field_year_normalized_citation_score" in report["rankings"][0]
    assert "relative_citation_ratio_score" in report["rankings"][0]
    assert "influential_citation_score" in report["rankings"][0]
    assert "relative_citation_ratio_raw" in report["rankings"][0]
    assert "influential_citation_count_raw" in report["rankings"][0]
    assert "citation_count_raw" in report["rankings"][0]
    assert "citation_graph_score" in report["rankings"][0]
    assert "citation_context_quality_score" in report["rankings"][0]
    assert "effective_citation_score" in report["rankings"][0]
    assert "effective_citation_count" in report["rankings"][0]
    assert "life_science_translation_score" in report["rankings"][0]
    assert "score_ci_low" in report["rankings"][0]
    assert "score_ci_high" in report["rankings"][0]
    assert "uncertainty_band" in report["rankings"][0]
    assert "reportability_status" in report["rankings"][0]
    assert "reportability_basis" in report["rankings"][0]
    assert "reportability_gap_count" in report["rankings"][0]
    assert "reportability_gap_actions" in report["rankings"][0]
    assert "legacy_reportability_status" in report["rankings"][0]
    assert "max_signal_contribution_share" in report["rankings"][0]
    assert "pairwise_stability_band" in report["rankings"][0]
    assert "pairwise_stability_rank_span" in report["rankings"][0]
    assert "final_score_stability_band" in report["rankings"][0]
    assert "final_score_stability_rank_span" in report["rankings"][0]
    assert report["rankings"][0]["model_pairwise_rank"] >= 1
    assert "reportability" in report["summary"]["fusion"]
    impact_csv = (output_dir / "impact_scores.csv").read_text(encoding="utf-8")
    assert "impact_index_v3" in impact_csv
    assert "impact_index_v2" in impact_csv
    assert "score_95_true_05_effective" in impact_csv
    assert "score_90_true_10_effective" in impact_csv
    assert "score_50_true_50_effective" in impact_csv
    assert "score_70_true_30_effective" in impact_csv
    assert "score_80_true_20_effective" in impact_csv
    assert "score_30_true_70_effective" in impact_csv
    assert "credibility_adjusted_impact_score" in impact_csv
    assert "credibility_adjustment_factor" in impact_csv
    assert "evidence_supported_score" in impact_csv
    assert "positive_evidence_signal_count" in impact_csv
    assert "zero_evidence_signal_count" in impact_csv
    assert "relative_citation_ratio_score" in impact_csv
    assert "effective_citation_score" in impact_csv
    assert "effective_citation_count" in impact_csv
    assert "influential_citation_score" in impact_csv
    assert "external_adoption_score" in impact_csv
    assert "life_science_translation_score" in impact_csv
    assert "bip_impact_score" in impact_csv
    assert "pairwise_aggregation_method" in impact_csv
    assert "credible_evidence_tier" in impact_csv
    assert "true_citation_openalex_raw" in impact_csv
    assert "true_citation_semantic_scholar_raw" in impact_csv
    assert "true_citation_crossref_raw" in impact_csv
    assert "true_citation_crossref_data_citations_raw" in impact_csv
    assert "true_citation_europepmc_raw" in impact_csv
    assert "true_citation_ncbi_pubmed_raw" in impact_csv
    assert "true_citation_datacite_raw" in impact_csv
    assert "true_citation_datacite_events_raw" in impact_csv
    assert "true_citation_openaire_raw" in impact_csv
    assert "true_citation_wikidata_raw" in impact_csv
    assert "true_citation_source_count" in impact_csv
    assert "true_citation_source_conflict_flag" in impact_csv
    assert "true_citation_identity_status" in impact_csv
    assert "true_citation_identity_similarity" in impact_csv
    assert "true_citation_identity_external_title" in impact_csv
    assert "true_citation_identity_reason" in impact_csv
    assert "fusion_weight_model_pairwise" in impact_csv
    assert "score_ci_low" in impact_csv
    assert "reportability_status" in impact_csv
    assert "reportability_basis" in impact_csv
    assert "reportability_gap_count" in impact_csv
    assert "reportability_gap_actions" in impact_csv
    assert "legacy_reportability_status" in impact_csv
    assert "pairwise_stability_band" in impact_csv
    assert "pairwise_stability_score_ci_low" in impact_csv
    assert "final_score_stability_band" in impact_csv
    assert "final_score_stability_score_ci_low" in impact_csv
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "Impact Index v3" in report_markdown
    assert "only two categories" in report_markdown
    assert "true_citation_score" in report_markdown
    assert "rank_centrality_v1" in report_markdown
    assert "True Citation" in report_markdown
    assert "Relative Citation" in report_markdown
    assert "Influential Citation" in report_markdown
    assert "External Adoption" in report_markdown
    assert "fusion.version" in report_markdown
    assert "fusion.evidence_first.high_or_better_rate" in report_markdown
    assert "pairwise_stability.status" in report_markdown
    assert "Pairwise Stability" in report_markdown
    assert "final_score_stability.status" in report_markdown
    assert "Final Score Stability" in report_markdown
    assert "fusion.evidence_signal_audit.mean_positive_evidence_signal_count" in report_markdown
    assert "Evidence Signal Audit" in report_markdown
    assert "Positive Evidence" in report_markdown
    assert "Zero Evidence" in report_markdown
    assert "Evidence Gaps" in report_markdown
    assert "missing_high_value_evidence" in report_markdown
    assert "Uncertainty" in report_markdown
    assert "Reportability" in report_markdown
    assert (output_dir / "impact_score_cards.json").exists()
    assert (output_dir / "impact_score_cards.md").exists()
    assert (output_dir / "pairwise_stability.json").exists()
    assert (output_dir / "pairwise_stability.csv").exists()
    assert (output_dir / "pairwise_stability.md").exists()
    assert (output_dir / "final_score_stability.json").exists()
    assert (output_dir / "final_score_stability.csv").exists()
    assert (output_dir / "final_score_stability.md").exists()
    assert "Pairwise Stability" in (output_dir / "pairwise_stability.md").read_text(
        encoding="utf-8"
    )
    assert "Final Score Stability" in (
        output_dir / "final_score_stability.md"
    ).read_text(encoding="utf-8")
    score_cards = json.loads(
        (output_dir / "impact_score_cards.json").read_text(encoding="utf-8")
    )
    assert score_cards["cards"][0]["paper_id"] == report["rankings"][0]["paper_id"]
    assert "relative_citation_ratio" in score_cards["cards"][0]["component_scores"]
    assert "effective_citation" in score_cards["cards"][0]["component_scores"]
    assert "life_science_translation" in score_cards["cards"][0]["component_scores"]
    assert "influential_citation" in score_cards["cards"][0]["component_scores"]
    assert "effective_citation_count" in score_cards["cards"][0]["raw_evidence"]
    assert "missing_high_value_evidence" in score_cards["cards"][0]
    assert "reportability_gap_count" in score_cards["cards"][0]
    assert "reportability_gap_actions" in score_cards["cards"][0]
    assert "positive_evidence_signal_count" in score_cards["cards"][0]
    assert "zero_evidence_signal_count" in score_cards["cards"][0]
    assert "Relative Citation" in (output_dir / "impact_score_cards.md").read_text(
        encoding="utf-8"
    )
    assert "Effective Citation" in (output_dir / "impact_score_cards.md").read_text(
        encoding="utf-8"
    )
    assert "positive_evidence_signal_count" in (
        output_dir / "impact_score_cards.md"
    ).read_text(encoding="utf-8")
    assert "reportability_gap_actions" in (
        output_dir / "impact_score_cards.md"
    ).read_text(encoding="utf-8")
    certificate = json.loads(
        (output_dir / "impact_claim_certificate.json").read_text(encoding="utf-8")
    )
    assert certificate["recommended_next_actions"] != ["stale certificate"]
    assert (output_dir / "reports" / "impact_claim_certificate.json").exists()


def test_model_score_dataset_can_replay_pairwise_judgments_without_model_calls(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    judgments_path = tmp_path / "pairwise_judgments.jsonl"
    records = [
        {
            "paper_id": "p1",
            "title": "Foundational method",
            "year": 2020,
            "field": "AI",
            "citation_count": 100,
            "references": [],
        },
        {
            "paper_id": "p2",
            "title": "Narrow follow-up",
            "year": 2021,
            "field": "AI",
            "citation_count": 5,
            "references": [{"reference_id": "p1", "title": "Foundational method"}],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    judgments_path.write_text(
        json.dumps(
            {
                "pair_index": 1,
                "round": 1,
                "paper_a_id": "p1",
                "paper_b_id": "p2",
                "winner_side": "A",
                "winner_id": "p1",
                "loser_id": "p2",
                "parseable": True,
                "decisive": True,
                "prediction": "Answer: A",
                "model": "cached-local-impact",
                "confidence": 0.9,
                "confidence_method": "cached",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FailingClient:
        def post(self, *args, **kwargs):
            raise AssertionError("model should not be called when judgments are replayed")

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=FailingClient(),
        pairwise_judgments_path=judgments_path,
    )

    assert report["summary"]["pairwise_judgment_source"] == {
        "type": "file",
        "path": str(judgments_path),
        "count": 1,
    }
    assert report["summary"]["judgment_count"] == 1
    assert report["rankings"][0]["paper_id"] == "p1"
    replayed = (output_dir / "pairwise_judgments.jsonl").read_text(encoding="utf-8")
    assert "cached-local-impact" in replayed


def test_model_score_dataset_can_resume_pairwise_judgments_and_generate_missing_pairs(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    resume_path = tmp_path / "partial_pairwise_judgments.jsonl"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    resume_path.write_text(
        json.dumps(
            {
                "pair_index": 1,
                "round": 1,
                "paper_a_id": "p1",
                "paper_b_id": "p2",
                "winner_side": "A",
                "winner_id": "p1",
                "loser_id": "p2",
                "parseable": True,
                "decisive": True,
                "prediction": "Answer: A",
                "model": "cached-local-impact",
                "confidence": 0.8,
                "confidence_method": "cached",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class MissingPairClient:
        def __init__(self):
            self.prompts = []

        def post(self, url, headers, json, timeout):
            self.prompts.append(json["messages"][0]["content"])

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "Answer: A"}}]}

            return Response()

    client = MissingPairClient()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        resume_pairwise_judgments_path=resume_path,
    )

    assert len(client.prompts) == 2
    assert "Paper A ID: p1" not in client.prompts[0] or "Paper B ID: p2" not in client.prompts[0]
    assert report["summary"]["pairwise_judgment_source"] == {
        "type": "resume",
        "path": str(resume_path),
        "existing_count": 1,
        "generated_count": 2,
        "count": 3,
    }
    replayed = (output_dir / "pairwise_judgments.jsonl").read_text(encoding="utf-8")
    assert "cached-local-impact" in replayed
    assert replayed.count('"pair_index"') == 3


def test_model_score_dataset_writes_pairwise_journal_during_generation(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    journal_path = tmp_path / "pairwise_journal.jsonl"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    class JournalCheckingClient:
        def __init__(self):
            self.call_count = 0

        def post(self, url, headers, json, timeout):
            self.call_count += 1
            if self.call_count == 2:
                assert journal_path.exists()
                assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "Answer: A"}}]}

            return Response()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=JournalCheckingClient(),
        pairwise_journal_path=journal_path,
    )

    assert report["summary"]["pairwise_journal"] == {
        "path": str(journal_path),
        "append": False,
    }
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 3
    assert "pairwise_journal" in (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )


def test_model_score_preflight_estimates_requests_and_resume_gap(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    resume_path = tmp_path / "partial_pairwise_judgments.jsonl"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
        {"paper_id": "p4", "title": "Four", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    resume_path.write_text(
        json.dumps(
            {
                "round": 1,
                "paper_a_id": "p1",
                "paper_b_id": "p2",
                "winner_id": "p1",
                "loser_id": "p2",
                "parseable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = model_score_preflight_report(
        input_path,
        max_pairs=3,
        pair_sampling="coverage",
        direct_score=True,
        bidirectional_pairwise=True,
        resume_pairwise_judgments_path=resume_path,
    )

    assert report["paper_count"] == 4
    assert report["pair_plan"]["planned_pair_count"] == 3
    assert report["pair_plan_quality"]["status"] == "limited"
    assert report["pair_plan_quality"]["connected_component_count"] == 1
    assert report["pair_plan_quality"]["largest_connected_component_coverage"] == 1.0
    assert report["pair_plan_quality"]["min_planned_comparisons_per_paper"] == 1
    assert report["pair_plan_quality"]["minimum_recommended_comparisons_per_paper"] == 2
    assert report["resume"]["completed_planned_pair_count"] == 1
    assert report["resume"]["missing_pair_count"] == 2
    assert report["request_estimate"] == {
        "pairwise_request_multiplier": 2,
        "pairwise_model_request_count": 4,
        "direct_model_request_count": 4,
        "total_model_request_count": 8,
    }
    assert report["recommendations"][0].startswith("Run model-score-dataset")
    assert "recommended minimum planned comparisons" in report["recommendations"][1]


def test_model_score_preflight_marks_disconnected_or_uncovered_plan_exploratory(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    records = [
        {"paper_id": f"p{i}", "title": f"Paper {i}", "references": []}
        for i in range(1, 5)
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = model_score_preflight_report(
        input_path,
        max_pairs=2,
        pair_sampling="sequential",
    )

    assert report["pair_plan_quality"]["status"] == "exploratory"
    assert report["pair_plan_quality"]["paper_coverage_rate"] < 1.0
    assert report["pair_plan_quality"]["connected_component_count"] > 1
    assert "planned pair graph" in report["recommendations"][1]


def test_write_model_score_preflight_writes_json_and_markdown(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "preflight"
    input_path.write_text(
        "\n".join(
            json.dumps({"paper_id": f"p{index}", "title": str(index), "references": []})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_model_score_preflight(
        input_path,
        output_dir,
        max_pairs=2,
    )

    assert report["pair_plan"]["planned_pair_count"] == 2
    assert "pair_plan_quality" in report
    assert (output_dir / "model_score_preflight.json").exists()
    markdown = (output_dir / "model_score_preflight.md").read_text(encoding="utf-8")
    assert "# Model Score Preflight" in markdown
    assert "total_model_request_count" in markdown
    assert "Pair Plan Quality" in markdown
    assert "connected_component_count" in markdown


def test_cli_model_score_preflight_writes_request_estimate(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "preflight"
    input_path.write_text(
        "\n".join(
            json.dumps({"paper_id": f"p{index}", "title": str(index), "references": []})
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "model-score-preflight",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-pairs",
            "3",
            "--pair-sampling",
            "coverage",
            "--bidirectional-pairwise",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote model score preflight" in result.output
    report = json.loads((output_dir / "model_score_preflight.json").read_text())
    assert report["request_estimate"]["total_model_request_count"] == 6
    assert report["pair_plan_quality"]["status"] == "limited"
    assert report["pair_plan_quality"]["connected_component_count"] == 1
    assert (output_dir / "model_score_preflight.md").exists()


def test_model_score_dataset_writes_external_evidence_readiness_artifacts(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {
            "paper_id": "p1",
            "title": "Externally supported",
            "doi": "10.1000/a",
            "citation_count": 20,
            "references": [],
            "metadata": {
                "openalex_enriched": True,
                "opencitations_enriched": True,
                "external_ids": {"openalex": "https://openalex.org/W1"},
                "external_impact": {
                    "field_year_percentile": 0.9,
                    "opencitations_cited_by_count": 20,
                    "opencitations_reference_count": 30,
                },
            },
        },
        {
            "paper_id": "p2",
            "title": "Citing paper",
            "doi": "10.1000/b",
            "citation_count": 5,
            "references": [
                {
                    "reference_id": "p1",
                    "title": "Externally supported",
                    "raw_citation": "OpenCitations internal DOI edge",
                }
            ],
            "metadata": {"opencitations_materialized_edge_count": 1},
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient({("p1", "p2"): "A"})

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
    )

    readiness_json = output_dir / "external_evidence_readiness.json"
    readiness_md = output_dir / "external_evidence_readiness.md"
    coverage_csv = output_dir / "external_evidence_paper_coverage.csv"
    citation_completeness_json = output_dir / "citation_completeness_audit.json"
    citation_completeness_md = output_dir / "citation_completeness_audit.md"
    citation_completeness_csv = output_dir / "citation_completeness_papers.csv"
    assert readiness_json.exists()
    assert readiness_md.exists()
    assert coverage_csv.exists()
    assert citation_completeness_json.exists()
    assert citation_completeness_md.exists()
    assert citation_completeness_csv.exists()
    readiness = json.loads(readiness_json.read_text(encoding="utf-8"))
    citation_completeness = json.loads(
        citation_completeness_json.read_text(encoding="utf-8")
    )
    assert report["summary"]["external_evidence_readiness"]["decision"]["status"] in {
        "ready_for_high_credibility_scoring",
        "usable_with_disclosure",
    }
    assert report["summary"]["citation_completeness_audit"]["summary"]["paper_count"] == 2
    assert readiness["graph_evidence"]["materialized_internal_edge_count"] == 1
    assert citation_completeness["summary"]["citation_evidence_coverage"] == 1.0
    report_payload = json.loads(
        (output_dir / "model_impact_report.json").read_text(encoding="utf-8")
    )
    assert "external_evidence_readiness" in report_payload["summary"]
    assert "citation_completeness_audit" in report_payload["summary"]
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "External Evidence Readiness" in report_markdown
    assert "Citation Completeness Audit" in report_markdown
    assert "external_query_id_coverage" in report_markdown
    assert "papers_with_any_missing_evidence" in report_markdown
    assert "missing_evidence" in coverage_csv.read_text(encoding="utf-8")
    assert "completeness_grade" in citation_completeness_csv.read_text(encoding="utf-8")


def test_model_score_dataset_can_write_impact_validity_evaluation(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    labels_path = tmp_path / "future_labels.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    labels = [
        {"paper_id": "p1", "future_impact": 10, "field": "AI"},
        {"paper_id": "p2", "future_impact": 90, "field": "AI"},
        {"paper_id": "p3", "future_impact": 20, "field": "AI"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "\n".join(json.dumps(record) for record in labels) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "B",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        validity_labels_path=labels_path,
        validity_bootstrap_samples=10,
        validity_permutation_samples=10,
    )

    validity = report["summary"]["impact_validity_evaluation"]
    assert validity["status"] in {
        "validated",
        "needs_more_evidence",
        "not_validated",
        "insufficient_validation_data",
    }
    assert validity["label_path"] == str(labels_path)
    assert (output_dir / "impact_validity" / "impact_validity_evaluation.json").exists()
    report_payload = json.loads(
        (output_dir / "model_impact_report.json").read_text(encoding="utf-8")
    )
    assert "impact_validity_evaluation" in report_payload["summary"]
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "Impact Validity" in report_markdown
    assert "validated_impact_claim" in report["summary"]
    assert "validity_confidence" in report["rankings"][0]


def test_model_score_dataset_marks_unlabeled_scores_as_unvalidated(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient({("p1", "p2"): "A"})

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
    )

    claim = report["summary"]["validated_impact_claim"]
    assert claim["status"] == "unvalidated"
    assert claim["can_claim_validated_impact_accuracy"] is False
    assert claim["reason"] == "No future-window or external validity labels were supplied."
    assert all(row["validity_status"] == "unvalidated" for row in report["rankings"])
    assert all(row["validity_confidence"] <= 0.25 for row in report["rankings"])
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "validated_impact_claim.status" in report_markdown


def test_model_score_dataset_can_load_calibrated_fusion_weights(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    weights_path = tmp_path / "fusion_calibration.json"
    records = [
        {
            "paper_id": "p1",
            "title": "Citation winner",
            "year": 2024,
            "field": "AI",
            "citation_count": 100,
            "references": [],
        },
        {
            "paper_id": "p2",
            "title": "Model winner",
            "year": 2024,
            "field": "AI",
            "citation_count": 1,
            "references": [],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    weights_path.write_text(
        json.dumps(
            {
                "best_weights": {
                    "model_pairwise": 0.0,
                    "field_year_normalized_citation": 1.0,
                    "citation_graph": 0.0,
                    "citation_context_quality": 0.0,
                    "temporal_momentum": 0.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient({("p1", "p2"): "B"})

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        fusion_weights_path=weights_path,
    )

    assert report["summary"]["fusion"]["weight_source"]["source_path"] == str(weights_path)
    assert report["summary"]["fusion"]["base_weights"] == {
        "model_pairwise": 0.0,
        "field_year_normalized_citation": 1.0,
        "citation_graph": 0.0,
        "citation_context_quality": 0.0,
        "temporal_momentum": 0.0,
    }
    assert (
        report["summary"]["fusion"]["evidence_first_weights"][
            "field_year_normalized_citation"
        ]
        == 1.0
    )
    assert report["summary"]["fusion"]["evidence_first_weights"]["influential_citation"] == 0.0
    assert report["summary"]["fusion"]["evidence_first_weights"]["external_adoption"] == 0.20
    assert report["rankings"][0]["paper_id"] == "p1"
    assert report["rankings"][0]["fusion_weight_field_year_normalized_citation"] == 1.0
    persisted_report = json.loads(
        (output_dir / "model_impact_report.json").read_text(encoding="utf-8")
    )
    assert (
        persisted_report["summary"]["fusion"]["weight_source"]["source"]
        == "fusion_calibration.best_weights"
    )


def test_model_score_dataset_can_load_v3_component_calibration(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    weights_path = tmp_path / "fusion_calibration.json"
    records = [
        {
            "paper_id": "p1",
            "title": "Externally validated paper",
            "year": 2024,
            "field": "AI",
            "metadata": {
                "external_impact": {
                    "bip_impact_percentile": 0.99,
                }
            },
            "references": [],
        },
        {
            "paper_id": "p2",
            "title": "Model preferred paper",
            "year": 2024,
            "field": "AI",
            "metadata": {
                "external_impact": {
                    "bip_impact_percentile": 0.10,
                }
            },
            "references": [],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    weights_path.write_text(
        json.dumps(
            {
                "weight_profile": "impact_index_v3_final_components",
                "best_component_weights": {
                    "model_pairwise": 0.0,
                    "field_year_normalized_citation": 0.0,
                    "relative_citation_ratio": 0.0,
                    "citation_graph": 0.0,
                    "citation_context_quality": 0.0,
                    "influential_citation": 0.0,
                    "temporal_momentum": 0.0,
                    "external_adoption": 1.0,
                },
                "best_impact_index_v3_weights": {
                    "evidence_supported": 1.0,
                    "model_supported": 0.0,
                },
                "best_evidence_first_weights": {
                    "field_year_normalized_citation": 0.0,
                    "relative_citation_ratio": 0.0,
                    "citation_graph": 0.0,
                    "citation_context_quality": 0.0,
                    "influential_citation": 0.0,
                    "temporal_momentum": 0.0,
                    "external_adoption": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient({("p1", "p2"): "B"})

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        fusion_weights_path=weights_path,
    )

    assert report["rankings"][0]["paper_id"] == "p1"
    assert report["summary"]["fusion"]["weight_source"]["source"] == (
        "fusion_calibration.best_component_weights"
    )
    assert report["summary"]["fusion"]["impact_index_v3_weights"] == {
        "evidence_supported": 1.0,
        "model_supported": 0.0,
    }
    assert report["summary"]["fusion"]["evidence_first_weights"]["external_adoption"] == 1.0
    assert report["rankings"][0]["external_adoption_score"] > report["rankings"][1][
        "external_adoption_score"
    ]


def test_model_score_dataset_v3_uses_external_evidence_over_model_only_preference(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {
            "paper_id": "p1",
            "title": "Externally adopted paper",
            "year": 2024,
            "field": "AI",
            "citation_count": 120,
            "metadata": {
                "external_impact": {
                    "field_year_percentile": 0.96,
                    "patent_citations": 4,
                    "policy_citations": 2,
                    "software_mentions": 20,
                    "dataset_reuses": 10,
                }
            },
            "references": [],
        },
        {
            "paper_id": "p2",
            "title": "Model preferred paper",
            "year": 2024,
            "field": "AI",
            "citation_count": 5,
            "metadata": {"external_impact": {"field_year_percentile": 0.30}},
            "references": [],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=PairwiseClient({("p1", "p2"): "B"}),
    )

    assert report["rankings"][0]["paper_id"] == "p1"
    assert report["rankings"][0]["impact_index_v3"] > report["rankings"][1]["impact_index_v3"]
    assert report["rankings"][0]["model_pairwise_score"] < report["rankings"][1]["model_pairwise_score"]
    assert report["rankings"][0]["credible_evidence_tier"] in {"high", "very_high"}
    assert report["rankings"][0]["dominant_score_family"] == "evidence_supported"


def test_model_score_dataset_rejects_invalid_fusion_weights(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    weights_path = tmp_path / "fusion_calibration.json"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "One", "references": []}) + "\n",
        encoding="utf-8",
    )
    weights_path.write_text(
        json.dumps({"best_weights": {"model_pairwise": 1.0}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing signals"):
        model_score_dataset(
            input_path,
            output_dir,
            base_url="http://localhost:8000/v1",
            model="local-impact",
            client=PairwiseClient({}),
            fusion_weights_path=weights_path,
        )


def test_model_score_dataset_marks_degenerate_direct_scores_as_diagnostic_only(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    class DegenerateHybridClient:
        def __init__(self):
            self.direct = DirectScoreClient({"p1": 0, "p2": 0, "p3": 0})
            self.pairwise = PairwiseClient(
                {
                    ("p1", "p2"): "A",
                    ("p1", "p3"): "A",
                    ("p2", "p3"): "A",
                }
            )

        def post(self, url, headers, json, timeout):
            content = json["messages"][0]["content"]
            if "Paper ID:" in content:
                return self.direct.post(url, headers, json, timeout)
            return self.pairwise.post(url, headers, json, timeout)

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=DegenerateHybridClient(),
        direct_score=True,
    )

    assert report["summary"]["direct_score_quality"] == {
        "status": "degenerate",
        "unique_score_count": 1,
        "score_range": 0.0,
        "reason": "Direct model scores collapsed to a single value; treat as diagnostic only.",
    }


def test_model_score_dataset_records_comparison_rounds(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "B",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        comparison_rounds=2,
    )

    assert report["summary"]["comparison_rounds"] == 2
    assert report["summary"]["judgment_count"] == 6
    judgments = [
        json.loads(line)
        for line in (output_dir / "pairwise_judgments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {judgment["round"] for judgment in judgments} == {1, 2}


def test_model_score_dataset_records_bidirectional_summary(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"paper_id": "p1", "title": "One", "references": []}),
                json.dumps({"paper_id": "p2", "title": "Two", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class ConsistentBidirectionalClient:
        def post(self, url, headers, json, timeout):
            content = json["messages"][0]["content"]
            answer = (
                "Answer: A"
                if "Paper A ID: p1" in content
                else "Answer: B"
            )

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": answer}}]}

            return Response()

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=ConsistentBidirectionalClient(),
        bidirectional_pairwise=True,
        bidirectional_confidence_margin=0.05,
    )

    assert report["summary"]["bidirectional_pairwise"] is True
    assert report["summary"]["bidirectional_confidence_margin"] == 0.05
    assert report["summary"]["decisive_judgment_count"] == 1
    assert report["summary"]["position_consistency_counts"] == {"consistent": 1}
    assert report["rankings"][0]["paper_id"] == "p1"
    judgment = json.loads(
        (output_dir / "pairwise_judgments.jsonl").read_text(encoding="utf-8")
    )
    assert judgment["bidirectional"] is True
    assert judgment["position_consistency"] == "consistent"


def test_model_score_dataset_marks_sparse_pair_budget_as_exploratory(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": f"p{i}", "title": f"Paper {i}", "references": []}
        for i in range(1, 7)
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "A",
            ("p3", "p4"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        max_pairs=2,
        pair_sampling="coverage",
    )

    assert report["summary"]["ranking_quality"] == {
        "status": "exploratory",
        "reason": "Only 4 of 6 papers have decisive pairwise comparisons; increase --max-pairs or --comparison-rounds before treating the index as a full ranking.",
        "compared_paper_count": 4,
        "uncompared_paper_count": 2,
        "paper_coverage_rate": 0.666667,
        "min_comparisons_per_paper": 0,
        "max_comparisons_per_paper": 1,
        "mean_comparisons_per_paper": 0.666667,
        "decisive_judgment_count": 2,
        "minimum_recommended_decisive_judgments": 6,
        "minimum_recommended_comparisons_per_paper": 2,
        "connected_component_count": 4,
        "largest_connected_component_paper_count": 2,
        "largest_connected_component_coverage": 0.333333,
        "isolated_paper_count": 2,
    }
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "| ranking_quality.status | exploratory |" in report_markdown
    assert "| ranking_quality.paper_coverage_rate | 0.666667 |" in report_markdown
    impact_csv = (output_dir / "impact_scores.csv").read_text(encoding="utf-8")
    assert "ranking_quality_status" in impact_csv
    assert "exploratory" in impact_csv


def test_model_score_dataset_marks_complete_pairwise_comparisons_reportable(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": "p1", "title": "One", "references": []},
        {"paper_id": "p2", "title": "Two", "references": []},
        {"paper_id": "p3", "title": "Three", "references": []},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "A",
            ("p1", "p3"): "A",
            ("p2", "p3"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
    )

    assert report["summary"]["ranking_quality"]["status"] == "reportable"
    assert report["summary"]["ranking_quality"]["paper_coverage_rate"] == 1.0
    assert report["summary"]["ranking_quality"]["min_comparisons_per_paper"] == 2
    assert report["summary"]["ranking_quality"]["connected_component_count"] == 1
    assert report["summary"]["ranking_quality"]["largest_connected_component_coverage"] == 1.0


def test_pairwise_ranking_quality_marks_disconnected_full_coverage_as_exploratory():
    quality = pairwise_ranking_quality_report(
        [paper(f"p{i}", f"Paper {i}") for i in range(1, 5)],
        [
            {"winner_id": "p1", "loser_id": "p2", "parseable": True},
            {"winner_id": "p2", "loser_id": "p1", "parseable": True},
            {"winner_id": "p3", "loser_id": "p4", "parseable": True},
            {"winner_id": "p4", "loser_id": "p3", "parseable": True},
        ],
    )

    assert quality["paper_coverage_rate"] == 1.0
    assert quality["min_comparisons_per_paper"] == 2
    assert quality["connected_component_count"] == 2
    assert quality["largest_connected_component_coverage"] == 0.5
    assert quality["status"] == "exploratory"
    assert "connected components" in quality["reason"]


def test_model_score_dataset_auto_pair_budget_targets_paper_coverage(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": f"p{i}", "title": f"Paper {i}", "references": []}
        for i in range(1, 11)
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "A",
            ("p3", "p4"): "A",
            ("p5", "p6"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        pair_sampling="coverage",
        pair_budget="auto",
        target_paper_coverage=0.6,
    )

    assert report["summary"]["pair_budget"] == {
        "mode": "auto",
        "requested_max_pairs": None,
        "effective_max_pairs": 3,
        "target_paper_coverage": 0.6,
        "paper_count": 10,
        "reason": "Auto pair budget selected 3 pairs to cover about 60.0% of 10 papers.",
    }
    report_markdown = (output_dir / "model_impact_report.md").read_text(
        encoding="utf-8"
    )
    assert "| pair_budget.mode | auto |" in report_markdown
    assert "| pair_budget.effective_max_pairs | 3 |" in report_markdown
    assert report["summary"]["judgment_count"] == 3
    assert report["summary"]["ranking_quality"]["compared_paper_count"] == 6
    assert report["summary"]["ranking_quality"]["paper_coverage_rate"] == 0.6


def test_model_score_dataset_manual_max_pairs_records_budget_source(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    records = [
        {"paper_id": f"p{i}", "title": f"Paper {i}", "references": []}
        for i in range(1, 7)
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    client = PairwiseClient(
        {
            ("p1", "p2"): "A",
            ("p3", "p4"): "A",
        }
    )

    report = model_score_dataset(
        input_path,
        output_dir,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        max_pairs=2,
        pair_sampling="coverage",
    )

    assert report["summary"]["pair_budget"]["mode"] == "manual"
    assert report["summary"]["pair_budget"]["requested_max_pairs"] == 2
    assert report["summary"]["pair_budget"]["effective_max_pairs"] == 2


def test_cli_model_score_dataset_writes_impact_index(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    calibration_report = tmp_path / "calibration.json"
    fusion_weights = tmp_path / "fusion_calibration.json"
    validity_labels = tmp_path / "future_labels.jsonl"
    bundle_dir = tmp_path / "impact_bundle"
    pairwise_judgments = tmp_path / "cached_pairwise_judgments.jsonl"
    pairwise_journal = tmp_path / "pairwise_journal.jsonl"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "One", "references": []}) + "\n",
        encoding="utf-8",
    )
    calibration_report.write_text('{"calibration": {"bins": []}}\n', encoding="utf-8")
    fusion_weights.write_text('{"best_weights": {}}\n', encoding="utf-8")
    validity_labels.write_text('{"paper_id": "p1", "future_impact": 1}\n', encoding="utf-8")
    pairwise_judgments.write_text('{"winner_id":"p1","loser_id":"p2"}\n', encoding="utf-8")

    def fake_model_score_dataset(*args, **kwargs):
        assert kwargs["calibration_report_path"] == calibration_report
        assert kwargs["fusion_weights_path"] == fusion_weights
        assert kwargs["pair_sampling"] == "coverage"
        assert kwargs["pair_budget"] == "auto"
        assert kwargs["target_paper_coverage"] == 0.75
        assert kwargs["bidirectional_pairwise"] is True
        assert kwargs["bidirectional_confidence_margin"] == 0.05
        assert kwargs["pairwise_aggregation_method"] == "rank_centrality"
        assert kwargs["validity_labels_path"] == validity_labels
        assert kwargs["validity_label_column"] == "future_impact"
        assert kwargs["pairwise_judgments_path"] == pairwise_judgments
        assert kwargs["pairwise_journal_path"] == pairwise_journal
        assert kwargs["pairwise_stability_samples"] == 25
        assert kwargs["pairwise_stability_top_n"] == 7
        assert kwargs["pairwise_stability_seed"] == 11
        assert kwargs["final_score_stability_samples"] == 35
        assert kwargs["final_score_stability_top_n"] == 9
        assert kwargs["final_score_stability_seed"] == 13
        output_dir.mkdir(parents=True)
        (output_dir / "impact_scores.csv").write_text("rank,paper_id,impact_index\n1,p1,50\n")
        return {"summary": {"paper_count": 1, "judgment_count": 0}, "rankings": []}

    monkeypatch.setattr(cli, "model_score_dataset", fake_model_score_dataset)

    def fake_write_report_bundle(run_dir, output):
        assert run_dir == output_dir
        assert output == bundle_dir
        output.mkdir(parents=True)
        (output / "bundle_manifest.json").write_text('{"copied_files":[]}\n')
        (output / "README.md").write_text("# Local Impact Ranking Report Bundle\n")
        return {"copied_files": []}

    monkeypatch.setattr(cli, "write_report_bundle", fake_write_report_bundle)

    result = CliRunner().invoke(
        app,
        [
            "model-score-dataset",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-impact",
            "--direct-score",
            "--calibration-report",
            str(calibration_report),
            "--fusion-weights",
            str(fusion_weights),
            "--pair-sampling",
            "coverage",
            "--pair-budget",
            "auto",
            "--target-paper-coverage",
            "0.75",
            "--bidirectional-pairwise",
            "--bidirectional-confidence-margin",
            "0.05",
            "--pairwise-aggregation",
            "rank_centrality",
            "--validity-labels",
            str(validity_labels),
            "--validity-label-column",
            "future_impact",
            "--pairwise-judgments",
            str(pairwise_judgments),
            "--pairwise-journal",
            str(pairwise_journal),
            "--pairwise-stability-samples",
            "25",
            "--pairwise-stability-top-n",
            "7",
            "--pairwise-stability-seed",
            "11",
            "--final-score-stability-samples",
            "35",
            "--final-score-stability-top-n",
            "9",
            "--final-score-stability-seed",
            "13",
            "--report-bundle",
            "--report-bundle-output",
            str(bundle_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote model impact index" in result.output
    assert "Wrote model score report bundle" in result.output


def test_cli_model_score_dataset_can_resume_cached_pairwise_judgments(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "impact"
    resume_judgments = tmp_path / "partial_pairwise_judgments.jsonl"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "One", "references": []}) + "\n",
        encoding="utf-8",
    )
    resume_judgments.write_text(
        '{"paper_a_id":"p1","paper_b_id":"p2","winner_id":"p1","loser_id":"p2"}\n',
        encoding="utf-8",
    )

    def fake_model_score_dataset(*args, **kwargs):
        assert kwargs["pairwise_judgments_path"] is None
        assert kwargs["resume_pairwise_judgments_path"] == resume_judgments
        output_dir.mkdir(parents=True)
        (output_dir / "impact_scores.csv").write_text("rank,paper_id,impact_index\n1,p1,50\n")
        return {"summary": {"paper_count": 1, "judgment_count": 1}, "rankings": []}

    monkeypatch.setattr(cli, "model_score_dataset", fake_model_score_dataset)

    result = CliRunner().invoke(
        app,
        [
            "model-score-dataset",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--model",
            "local-impact",
            "--resume-pairwise-judgments",
            str(resume_judgments),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote model impact index" in result.output
