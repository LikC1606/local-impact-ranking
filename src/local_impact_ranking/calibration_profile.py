from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MIN_CALIBRATION_EXAMPLES = 100
MAX_ECE = 0.12
MAX_MCE = 0.25
MAX_BRIER = 0.25


def load_calibration_profile(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not str(path) or not source.exists():
        return {}
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def calibration_profile_card(
    profile: dict[str, Any],
    *,
    profile_path: str | Path | None = None,
    served_model: str | None = None,
) -> dict[str, Any]:
    profile_info = profile.get("profile", {})
    summary = profile.get("summary", {})
    calibration = profile.get("calibration", {})
    compatible_models = _compatible_model_names(profile_info)
    model_identity_match = _model_identity_match(served_model, compatible_models)
    scored_examples = int(calibration.get("scored_examples") or 0)
    ece = _float_or_none(calibration.get("ece"))
    mce = _float_or_none(calibration.get("mce"))
    brier = _float_or_none(calibration.get("brier_score"))
    quality_checks = {
        "enough_examples": scored_examples >= MIN_CALIBRATION_EXAMPLES,
        "ece_within_threshold": ece is not None and ece <= MAX_ECE,
        "mce_within_threshold": mce is not None and mce <= MAX_MCE,
        "brier_within_threshold": brier is not None and brier <= MAX_BRIER,
        "has_bins": bool(calibration.get("bins")),
    }
    quality_supported = all(quality_checks.values())

    if not profile:
        status = "not_provided"
    elif model_identity_match is False:
        status = "mismatched_profile"
    elif not quality_supported:
        status = "weak_calibration"
    else:
        status = "calibration_supported"

    can_use = status == "calibration_supported"
    disclosures = [
        "Calibrated confidence is a reporting trust prior, not domain-specific ground truth.",
        "Scores remain dataset-local and require external/future labels for validated impact accuracy claims.",
    ]
    if model_identity_match is False:
        disclosures.append(
            "The served model does not match the calibration profile; calibrated confidence should not support reportable claims."
        )
    if not quality_supported:
        disclosures.append(
            "Calibration quality thresholds are not fully satisfied; disclose calibration limits."
        )

    return {
        "profile_path": str(profile_path) if profile_path is not None else None,
        "served_model": served_model,
        "profile": {
            "name": profile_info.get("name"),
            "method": profile_info.get("method"),
            "source_benchmark": profile_info.get("source_benchmark"),
            "source_model": profile_info.get("source_model"),
            "source_base_model": profile_info.get("source_base_model"),
            "source_run": profile_info.get("source_run"),
            "scope": profile_info.get("scope"),
            "limitations": profile_info.get("limitations"),
        },
        "summary": {
            "total_examples": summary.get("total_examples"),
            "covered_examples": summary.get("covered_examples"),
            "accuracy": summary.get("accuracy"),
            "scored_examples": scored_examples,
            "bin_count": calibration.get("bin_count"),
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
            "overconfidence_gap": calibration.get("overconfidence_gap"),
        },
        "decision": {
            "status": status,
            "can_use_calibrated_confidence": can_use,
            "model_identity_match": model_identity_match,
            "compatible_models": sorted(compatible_models),
            "quality_supported": quality_supported,
            "quality_checks": quality_checks,
            "reason": _decision_reason(
                status=status,
                served_model=served_model,
                compatible_models=compatible_models,
            ),
        },
        "required_disclosures": disclosures,
    }


def write_calibration_profile_card(
    profile_path: str | Path | None,
    output_dir: str | Path,
    *,
    served_model: str | None = None,
) -> dict[str, Any]:
    profile = load_calibration_profile(profile_path)
    card = calibration_profile_card(
        profile,
        profile_path=profile_path,
        served_model=served_model,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "calibration_profile_card.json").write_text(
        json.dumps(card, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "calibration_profile_card.md").write_text(
        render_calibration_profile_card_markdown(card),
        encoding="utf-8",
    )
    return card


def report_decision_with_calibration_profile(
    decision: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    calibration_decision = card.get("decision", {})
    if calibration_decision.get("can_use_calibrated_confidence"):
        return decision
    if decision.get("status") in {"validated_report", "report_with_disclosure"}:
        updated = dict(decision)
        updated["status"] = "exploratory_only"
        updated["reason"] = (
            f"{decision.get('reason', '')} Calibration profile evidence is not "
            "sufficient for reportable calibrated-confidence scoring; treat the "
            "output as exploratory until an applicable calibration profile is supplied."
        ).strip()
        return updated
    return decision


def render_calibration_profile_card_markdown(card: dict[str, Any]) -> str:
    profile = card.get("profile", {})
    summary = card.get("summary", {})
    decision = card.get("decision", {})
    lines = [
        "# Calibration Profile Card",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| profile_name | {_text(profile.get('name'))} |",
        f"| method | {_text(profile.get('method'))} |",
        f"| served_model | {_text(card.get('served_model'))} |",
        f"| compatible_models | {_text(', '.join(decision.get('compatible_models') or []))} |",
        f"| calibration_status | {_text(decision.get('status'))} |",
        f"| model_identity_match | {_identity_match_text(decision.get('model_identity_match'))} |",
        f"| can_use_calibrated_confidence | {_yes_no(decision.get('can_use_calibrated_confidence'))} |",
        f"| quality_supported | {_yes_no(decision.get('quality_supported'))} |",
        f"| reason | {_text(decision.get('reason'))} |",
        "",
        "## Calibration Evidence",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| scored_examples | {_text(summary.get('scored_examples'))} |",
        f"| bin_count | {_text(summary.get('bin_count'))} |",
        f"| ece | {_number(summary.get('ece'))} |",
        f"| mce | {_number(summary.get('mce'))} |",
        f"| brier_score | {_number(summary.get('brier_score'))} |",
        f"| overconfidence_gap | {_number(summary.get('overconfidence_gap'))} |",
        "",
        "## Required Disclosures",
        "",
    ]
    disclosures = card.get("required_disclosures", [])
    lines.extend(f"- {_text(item)}" for item in disclosures) if disclosures else lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _compatible_model_names(profile_info: dict[str, Any]) -> set[str]:
    values = [profile_info.get("source_model")]
    compatible = profile_info.get("compatible_models", [])
    if isinstance(compatible, list):
        values.extend(compatible)
    return {str(value).strip() for value in values if str(value or "").strip()}


def _model_identity_match(
    served_model: str | None,
    compatible_models: set[str],
) -> bool | None:
    if served_model is None or not str(served_model).strip():
        return None
    if not compatible_models:
        return False
    return str(served_model).strip() in compatible_models


def _decision_reason(
    *,
    status: str,
    served_model: str | None,
    compatible_models: set[str],
) -> str:
    if status == "not_provided":
        return "No calibration profile was provided."
    if status == "mismatched_profile":
        expected = ", ".join(sorted(compatible_models)) or "unknown"
        return (
            f"The served model '{served_model}' does not match the calibration "
            f"profile compatible model name(s): {expected}."
        )
    if status == "weak_calibration":
        return "Calibration profile quality checks do not meet reportable thresholds."
    return "Calibration profile is applicable to the served model and passes quality thresholds."


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _identity_match_text(value: Any) -> str:
    if value is None:
        return "not_checked"
    return _yes_no(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")
