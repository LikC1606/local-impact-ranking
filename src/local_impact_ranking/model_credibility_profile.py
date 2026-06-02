from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORTABLE_MODEL_STATUSES = {"moderate_credibility", "high_credibility"}


def load_model_credibility_profile(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not str(path) or not source.exists():
        return {}
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def model_credibility_card(
    profile: dict[str, Any],
    *,
    profile_path: str | Path | None = None,
    served_model: str | None = None,
) -> dict[str, Any]:
    profile_info = profile.get("profile", {})
    summary = profile.get("summary", {})
    scorecard = profile.get("credibility_scorecard", {})
    significance = profile.get("pooled_statistical_significance", {})
    weak_slices = list(profile.get("weak_slices", []))
    profile_model_names = _profile_model_names(profile_info)
    model_identity_match = _model_identity_match(served_model, profile_model_names)
    status = str(scorecard.get("status") or "not_provided")
    if model_identity_match is False:
        status = "mismatched_profile"
    significant_vs_random = bool(significance.get("significant_vs_random", False))
    can_use_for_report = (
        status in REPORTABLE_MODEL_STATUSES
        and significant_vs_random
        and model_identity_match is not False
    )
    can_claim_high = (
        bool(scorecard.get("can_claim_high_model_credibility", False))
        and model_identity_match is not False
    )

    required_disclosures = [
        "Scores are dataset-local paper influence, not global scientific impact.",
        "Model evidence comes from a public benchmark profile and must be disclosed with results.",
    ]
    if not can_claim_high:
        required_disclosures.append(
            "The model cannot be claimed as a high-credibility standalone impact oracle."
        )
    if model_identity_match is False:
        required_disclosures.append(
            "The served model does not match the supplied credibility profile; do not reuse this profile for reportable claims."
        )
    if weak_slices:
        required_disclosures.append(
            "Known weak slices must be disclosed: "
            + ", ".join(str(row.get("slice", "")) for row in weak_slices[:10] if row.get("slice"))
            + "."
        )
    if not can_use_for_report:
        required_disclosures.append(
            "Model credibility evidence is not sufficient for reportable model-supported scoring."
        )

    return {
        "profile_path": str(profile_path) if profile_path is not None else None,
        "served_model": served_model,
        "profile": {
            "name": profile_info.get("name"),
            "source_benchmark": profile_info.get("source_benchmark"),
            "source_model": profile_info.get("source_model"),
            "source_base_model": profile_info.get("source_base_model"),
            "source_adapter": profile_info.get("source_adapter"),
            "source_run": profile_info.get("source_run"),
            "scope": profile_info.get("scope"),
            "limitations": profile_info.get("limitations"),
        },
        "summary": {
            "evaluation_count": summary.get("evaluation_count"),
            "total_examples": summary.get("total_examples"),
            "pooled_accuracy": summary.get("pooled_accuracy"),
            "min_seed_accuracy": summary.get("min_accuracy"),
            "max_seed_accuracy": summary.get("max_accuracy"),
            "accuracy_range": summary.get("accuracy_range"),
            "coverage_rate": summary.get("mean_coverage_rate"),
            "wilson_ci95": significance.get("wilson_ci95"),
        },
        "decision": {
            "status": status,
            "grade": scorecard.get("grade"),
            "significant_vs_random": significant_vs_random,
            "model_identity_match": model_identity_match,
            "profile_model_names": sorted(profile_model_names),
            "can_use_for_report_with_disclosure": can_use_for_report,
            "can_claim_high_model_credibility": can_claim_high,
            "standalone_model_claim_allowed": can_claim_high,
            "reason": _decision_reason(
                status=status,
                significant_vs_random=significant_vs_random,
                reason=scorecard.get("reason"),
                served_model=served_model,
                profile_model_names=profile_model_names,
            ),
        },
        "weak_slices": weak_slices,
        "required_disclosures": required_disclosures,
    }


def write_model_credibility_card(
    profile_path: str | Path | None,
    output_dir: str | Path,
    *,
    served_model: str | None = None,
) -> dict[str, Any]:
    profile = load_model_credibility_profile(profile_path)
    card = model_credibility_card(
        profile,
        profile_path=profile_path,
        served_model=served_model,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_credibility_card.json").write_text(
        json.dumps(card, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "model_credibility_card.md").write_text(
        render_model_credibility_card_markdown(card),
        encoding="utf-8",
    )
    return card


def report_decision_with_model_credibility(
    decision: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    model_decision = card.get("decision", {})
    if model_decision.get("can_use_for_report_with_disclosure"):
        if model_decision.get("status") != "high_credibility" and decision.get(
            "status"
        ) == "validated_report":
            updated = dict(decision)
            updated["status"] = "report_with_disclosure"
            updated["reason"] = (
                f"{decision.get('reason', '')} Model credibility is "
                f"{model_decision.get('status')}; disclose benchmark limits before "
                "claiming validated model-supported impact."
            ).strip()
            return updated
        return decision

    if decision.get("status") in {"validated_report", "report_with_disclosure"}:
        updated = dict(decision)
        updated["status"] = "exploratory_only"
        updated["reason"] = (
            f"{decision.get('reason', '')} Model credibility evidence is not sufficient "
            "for reportable model-supported scoring; treat the output as exploratory "
            "until a reportable model profile is supplied."
        ).strip()
        return updated
    return decision


def render_model_credibility_card_markdown(card: dict[str, Any]) -> str:
    profile = card.get("profile", {})
    summary = card.get("summary", {})
    decision = card.get("decision", {})
    lines = [
        "# Model Credibility Card",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| model_name | {_text(profile.get('name'))} |",
        f"| served_model | {_text(card.get('served_model'))} |",
        f"| source_benchmark | {_text(profile.get('source_benchmark'))} |",
        f"| source_adapter | {_text(profile.get('source_adapter'))} |",
        f"| model_credibility_status | {_text(decision.get('status'))} |",
        f"| grade | {_text(decision.get('grade'))} |",
        f"| significant_vs_random | {_yes_no(decision.get('significant_vs_random'))} |",
        f"| model_identity_match | {_identity_match_text(decision.get('model_identity_match'))} |",
        f"| can_use_for_report_with_disclosure | {_yes_no(decision.get('can_use_for_report_with_disclosure'))} |",
        f"| can_claim_high_model_credibility | {_yes_no(decision.get('can_claim_high_model_credibility'))} |",
        f"| standalone_model_claim_allowed | {_yes_no(decision.get('standalone_model_claim_allowed'))} |",
        f"| reason | {_text(decision.get('reason'))} |",
        "",
        "## Benchmark Evidence",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| evaluation_count | {_text(summary.get('evaluation_count'))} |",
        f"| total_examples | {_text(summary.get('total_examples'))} |",
        f"| pooled_accuracy | {_number(summary.get('pooled_accuracy'))} |",
        f"| min_seed_accuracy | {_number(summary.get('min_seed_accuracy'))} |",
        f"| max_seed_accuracy | {_number(summary.get('max_seed_accuracy'))} |",
        f"| accuracy_range | {_number(summary.get('accuracy_range'))} |",
        "",
        "## Weak Slices",
        "",
        "| Slice | Pooled Accuracy | Min Seed Accuracy | Covered Examples |",
        "| --- | ---: | ---: | ---: |",
    ]
    weak_slices = card.get("weak_slices", [])
    if weak_slices:
        for row in weak_slices[:10]:
            lines.append(
                "| "
                f"{_text(row.get('slice'))} | "
                f"{_number(row.get('pooled_accuracy'))} | "
                f"{_number(row.get('min_seed_accuracy'))} | "
                f"{_text(row.get('covered_examples'))} |"
            )
    else:
        lines.append("| none_reported |  |  |  |")
    lines.extend(["", "## Required Disclosures", ""])
    disclosures = card.get("required_disclosures", [])
    if disclosures:
        lines.extend(f"- {_text(item)}" for item in disclosures)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _decision_reason(
    *,
    status: str,
    significant_vs_random: bool,
    reason: Any,
    served_model: str | None = None,
    profile_model_names: set[str] | None = None,
) -> str:
    if status == "not_provided":
        return "No model credibility profile was provided."
    if status == "mismatched_profile":
        expected = ", ".join(sorted(profile_model_names or set())) or "unknown"
        return (
            f"The served model '{served_model}' does not match the supplied "
            f"credibility profile model name(s): {expected}."
        )
    if not significant_vs_random:
        return "The model profile does not show statistically supported performance above random."
    if reason:
        return str(reason)
    if status == "high_credibility":
        return "Model benchmark evidence supports high credibility."
    if status == "moderate_credibility":
        return "Model benchmark evidence is reportable with disclosure."
    return "Model benchmark evidence is not sufficient for reportable use."


def _profile_model_names(profile_info: dict[str, Any]) -> set[str]:
    return {
        str(value).strip()
        for value in (profile_info.get("name"), profile_info.get("source_model"))
        if str(value or "").strip()
    }


def _model_identity_match(
    served_model: str | None,
    profile_model_names: set[str],
) -> bool | None:
    if served_model is None or not str(served_model).strip():
        return None
    if not profile_model_names:
        return False
    return str(served_model).strip() in profile_model_names


def _identity_match_text(value: Any) -> str:
    if value is None:
        return "not_checked"
    return _yes_no(value)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")
