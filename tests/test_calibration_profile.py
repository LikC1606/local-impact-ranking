import json

from local_impact_ranking.calibration_profile import (
    calibration_profile_card,
    render_calibration_profile_card_markdown,
    report_decision_with_calibration_profile,
    write_calibration_profile_card,
)


def _supported_profile():
    return {
        "profile": {
            "name": "sciimpact-qwen32b-heldout-p50",
            "method": "empirical_bin_accuracy",
            "source_benchmark": "SciImpact",
            "source_model": "local-impact-judge",
            "compatible_models": [
                "local-impact-judge",
                "local-impact-judge-source-task-dpo",
            ],
        },
        "summary": {
            "total_examples": 150,
            "covered_examples": 150,
            "accuracy": 0.66,
        },
        "calibration": {
            "scored_examples": 150,
            "bin_count": 5,
            "ece": 0.086021,
            "mce": 0.160367,
            "brier_score": 0.221465,
            "overconfidence_gap": 0.08602,
            "bins": [{"lower": 0.8, "upper": 1.0, "accuracy": 0.75, "count": 53}],
        },
    }


def test_calibration_profile_card_allows_compatible_served_model():
    card = calibration_profile_card(
        _supported_profile(),
        profile_path="configs/calibration/sciimpact-qwen32b-heldout-p50.json",
        served_model="local-impact-judge-source-task-dpo",
    )

    assert card["decision"]["status"] == "calibration_supported"
    assert card["decision"]["can_use_calibrated_confidence"] is True
    assert card["decision"]["model_identity_match"] is True
    assert card["summary"]["ece"] == 0.086021
    assert "local-impact-judge-source-task-dpo" in card["decision"]["compatible_models"]

    markdown = render_calibration_profile_card_markdown(card)
    assert "# Calibration Profile Card" in markdown
    assert "| calibration_status | calibration_supported |" in markdown
    assert "| model_identity_match | yes |" in markdown


def test_calibration_profile_card_blocks_model_mismatch():
    card = calibration_profile_card(
        _supported_profile(),
        served_model="unverified-local-impact-judge",
    )

    assert card["decision"]["status"] == "mismatched_profile"
    assert card["decision"]["can_use_calibrated_confidence"] is False
    assert card["decision"]["model_identity_match"] is False

    decision = report_decision_with_calibration_profile(
        {
            "status": "report_with_disclosure",
            "reason": "Dataset-local scores can be reported.",
        },
        card,
    )
    assert decision["status"] == "exploratory_only"
    assert "Calibration profile evidence is not sufficient" in decision["reason"]


def test_write_calibration_profile_card_outputs_json_and_markdown(tmp_path):
    profile_path = tmp_path / "calibration.json"
    output_dir = tmp_path / "out"
    profile_path.write_text(json.dumps(_supported_profile()) + "\n", encoding="utf-8")

    card = write_calibration_profile_card(
        profile_path,
        output_dir,
        served_model="local-impact-judge-source-task-dpo",
    )

    assert card["decision"]["status"] == "calibration_supported"
    assert (output_dir / "calibration_profile_card.json").exists()
    assert (output_dir / "calibration_profile_card.md").exists()
