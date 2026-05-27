from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_card_report(
    run_dir: str | Path,
    *,
    top_n: int = 20,
    max_edges: int = 5,
) -> dict[str, Any]:
    root = Path(run_dir)
    evidence = json.loads(
        (root / "reports" / "paper_evidence_report.json").read_text(encoding="utf-8")
    )
    return score_cards_from_evidence_report(
        evidence,
        run=root.name,
        top_n=top_n,
        max_edges=max_edges,
    )


def score_cards_from_evidence_report(
    evidence: dict[str, Any],
    *,
    run: str = "current",
    top_n: int = 20,
    max_edges: int = 5,
) -> dict[str, Any]:
    cards = [
        _score_card(row, max_edges=max_edges)
        for row in evidence.get("papers", [])[:top_n]
    ]
    return {
        "run": run,
        "top_n": top_n,
        "max_edges": max_edges,
        "card_count": len(cards),
        "cards": cards,
        "limitations": evidence.get(
            "limitations",
            [
                "Scores are dataset-local influence estimates, not global scientific impact.",
                "Cards inherit the available citation metadata, judge selections, and benchmark coverage.",
            ],
        ),
    }


def write_score_cards(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 20,
    max_edges: int = 5,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = score_card_report(run_dir, top_n=top_n, max_edges=max_edges)
    (destination / "score_cards.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "score_cards.md").write_text(
        render_score_cards_markdown(report),
        encoding="utf-8",
    )
    return report


def render_score_cards_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Score Cards",
        "",
        f"Run: `{report.get('run', '')}`",
        "",
    ]
    cards = report.get("cards", [])
    if not cards:
        lines.extend(["No score cards available.", ""])
        return "\n".join(lines)

    for index, card in enumerate(cards, start=1):
        credibility = card.get("credibility", {})
        lines.extend(
            [
                f"## {index}. {card.get('paper_title') or card.get('paper_id', '')}",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| paper_id | `{card.get('paper_id', '')}` |",
                f"| headline_score | {_number(card.get('headline_score'))} |",
                f"| score_interval | {_interval(card.get('score_interval'))} |",
                f"| credibility | {credibility.get('level', '')} |",
                f"| credibility_reason | {credibility.get('reason', '')} |",
                f"| source_edge_count | {card.get('source_edge_count', 0)} |",
                f"| max_single_edge_share | {_number(card.get('max_single_edge_share'))} |",
                f"| support_band | {card.get('support_band', '')} |",
                "",
                "Top evidence:",
                "",
                "| Target | Rank | Credit | Contexts |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        edges = card.get("top_evidence", [])
        if not edges:
            lines.append("|  |  |  |  |")
        else:
            for edge in edges:
                lines.append(
                    f"| {edge.get('target_title') or edge.get('target_paper_id', '')} | "
                    f"{edge.get('rank', '')} | "
                    f"{_number(edge.get('credit'))} | "
                    f"{edge.get('context_count', 0)} |"
                )
        lines.append("")

    if report.get("limitations"):
        lines.extend(["## Limitations", ""])
        for limitation in report.get("limitations", []):
            lines.append(f"- {limitation}")
        lines.append("")
    return "\n".join(lines)


def _score_card(row: dict[str, Any], *, max_edges: int) -> dict[str, Any]:
    return {
        "paper_id": row.get("paper_id", ""),
        "paper_title": row.get("paper_title", ""),
        "headline_score": _float(row.get("final_score")),
        "confidence_flag": row.get("confidence_flag", ""),
        "support_band": row.get("support_band", ""),
        "source_edge_count": int(row.get("source_edge_count", 0) or 0),
        "max_single_edge_share": _float(row.get("max_single_edge_share")),
        "score_interval": [
            _float(row.get("score_ci_low")),
            _float(row.get("score_ci_high")),
        ],
        "score_ci_width": _float(row.get("score_ci_width")),
        "rank_span": int(row.get("rank_span", 0) or 0),
        "zero_score_rate": _float(row.get("zero_score_rate")),
        "uncertainty_band": row.get("uncertainty_band", ""),
        "credibility": _credibility(row),
        "top_evidence": list(row.get("top_source_edges", []))[:max_edges],
    }


def _credibility(row: dict[str, Any]) -> dict[str, str]:
    confidence = str(row.get("confidence_flag", ""))
    uncertainty = str(row.get("uncertainty_band", ""))
    edge_count = int(row.get("source_edge_count", 0) or 0)
    max_edge_share = _float(row.get("max_single_edge_share"))
    zero_score_rate = _float(row.get("zero_score_rate"))
    if confidence == "stable" and edge_count >= 2 and max_edge_share <= 0.7:
        return {"level": "high", "reason": "stable score with multiple supporting edges"}
    if (
        confidence == "low_evidence"
        or uncertainty == "unstable"
        or edge_count <= 1
        or max_edge_share >= 0.9
        or zero_score_rate >= 0.25
    ):
        return {"level": "low", "reason": "thin, unstable, or concentrated evidence"}
    return {"level": "medium", "reason": "usable score with some evidence caveats"}


def _number(value: Any) -> str:
    return f"{_float(value):.3f}"


def _interval(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
