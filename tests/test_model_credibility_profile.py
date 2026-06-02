import json

from local_impact_ranking.model_credibility_profile import (
    model_credibility_card,
    render_model_credibility_card_markdown,
    report_decision_with_model_credibility,
    write_model_credibility_card,
)


def _moderate_profile():
    return {
        "profile": {
            "name": "local-impact-judge-source-task-dpo",
            "source_benchmark": "SciImpact",
            "source_model": "local-impact-judge-source-task-dpo",
            "source_adapter": "qwen32b-source-task-sft16k4096-dpo16k1024-128step",
        },
        "summary": {
            "evaluation_count": 3,
            "total_examples": 900,
            "pooled_accuracy": 0.658889,
            "min_accuracy": 0.646667,
        },
        "pooled_statistical_significance": {
            "significant_vs_random": True,
            "wilson_ci95": [0.627299, 0.689128],
        },
        "credibility_scorecard": {
            "status": "moderate_credibility",
            "grade": "B",
            "can_claim_high_model_credibility": False,
            "reason": "Cross-seed SciImpact model evidence is supported, with warnings to disclose.",
        },
        "weak_slices": [
            {
                "slice": "dimension:stars",
                "pooled_accuracy": 0.58,
                "min_seed_accuracy": 0.52,
                "covered_examples": 150,
            }
        ],
    }


def test_model_credibility_card_allows_moderate_model_only_with_disclosure():
    card = model_credibility_card(
        _moderate_profile(),
        profile_path="configs/model_credibility/local-impact-judge-source-task-dpo.json",
    )

    assert card["decision"]["status"] == "moderate_credibility"
    assert card["decision"]["can_use_for_report_with_disclosure"] is True
    assert card["decision"]["can_claim_high_model_credibility"] is False
    assert card["decision"]["standalone_model_claim_allowed"] is False
    assert "dimension:stars" in card["weak_slices"][0]["slice"]
    assert any("weak slices" in item for item in card["required_disclosures"])

    markdown = render_model_credibility_card_markdown(card)
    assert "# Model Credibility Card" in markdown
    assert "| model_credibility_status | moderate_credibility |" in markdown
    assert "dimension:stars" in markdown


def test_model_credibility_card_blocks_profile_model_mismatch():
    card = model_credibility_card(
        _moderate_profile(),
        profile_path="configs/model_credibility/local-impact-judge-source-task-dpo.json",
        served_model="unverified-local-impact-judge",
    )

    assert card["decision"]["status"] == "mismatched_profile"
    assert card["decision"]["can_use_for_report_with_disclosure"] is False
    assert card["decision"]["model_identity_match"] is False
    assert card["served_model"] == "unverified-local-impact-judge"
    assert card["profile"]["name"] == "local-impact-judge-source-task-dpo"
    assert "does not match" in card["decision"]["reason"]
    assert any("does not match" in item for item in card["required_disclosures"])

    decision = report_decision_with_model_credibility(
        {
            "status": "report_with_disclosure",
            "reason": "Dataset-local scores can be reported.",
        },
        card,
    )
    assert decision["status"] == "exploratory_only"


def test_report_decision_is_downgraded_when_model_credibility_is_missing():
    base_decision = {
        "status": "report_with_disclosure",
        "reason": "Dataset-local scores can be reported.",
    }
    card = model_credibility_card({})

    decision = report_decision_with_model_credibility(base_decision, card)

    assert decision["status"] == "exploratory_only"
    assert "Model credibility evidence is not sufficient" in decision["reason"]


def test_write_model_credibility_card_outputs_json_and_markdown(tmp_path):
    profile_path = tmp_path / "profile.json"
    output_dir = tmp_path / "out"
    profile_path.write_text(json.dumps(_moderate_profile()) + "\n", encoding="utf-8")

    card = write_model_credibility_card(
        profile_path,
        output_dir,
        served_model="local-impact-judge-source-task-dpo",
    )

    assert card["decision"]["status"] == "moderate_credibility"
    assert card["decision"]["model_identity_match"] is True
    assert (output_dir / "model_credibility_card.json").exists()
    assert (output_dir / "model_credibility_card.md").exists()
    assert "Model Credibility Card" in (
        output_dir / "model_credibility_card.md"
    ).read_text(encoding="utf-8")
