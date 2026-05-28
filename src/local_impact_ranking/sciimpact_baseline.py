from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from local_impact_ranking.io import load_papers_jsonl


IMPACT_CUE_WEIGHTS = {
    "benchmark": 1.4,
    "dataset": 1.3,
    "open-source": 1.2,
    "open source": 1.2,
    "state-of-the-art": 1.1,
    "state of the art": 1.1,
    "large-scale": 1.0,
    "large scale": 1.0,
    "scalable": 0.8,
    "reproducible": 0.8,
    "evaluation": 0.7,
    "method": 0.6,
    "framework": 0.6,
    "survey": 0.5,
    "novel": 0.4,
    "introduce": 0.4,
}

DIMENSION_CUE_WEIGHTS = {
    "citation": {
        "benchmark": 0.6,
        "dataset": 0.5,
        "method": 0.4,
        "survey": 0.4,
    },
    "media": {
        "climate": 0.5,
        "health": 0.5,
        "public": 0.4,
        "policy": 0.4,
    },
    "patent": {
        "device": 0.5,
        "material": 0.5,
        "system": 0.3,
        "method": 0.3,
    },
    "award": {
        "novel": 0.5,
        "first": 0.4,
        "new": 0.3,
    },
}


def score_sciimpact_text_baseline(
    *,
    title: str,
    abstract: str | None,
    item_type: str = "paper",
    impact_dimension: str = "unknown",
) -> float:
    text = f"{title} {abstract or ''}".casefold()
    token_count = len(re.findall(r"[a-z0-9]+", text))
    length_score = min(math.log1p(token_count) / math.log(401), 1.0) * 2.0
    cue_score = _cue_score(text, IMPACT_CUE_WEIGHTS)
    dimension_score = _cue_score(text, DIMENSION_CUE_WEIGHTS.get(impact_dimension, {}))
    item_type_bonus = 0.2 if item_type == "paper" else 0.0
    return length_score + cue_score + dimension_score + item_type_bonus


def write_sciimpact_text_baseline_rankings(
    papers_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    papers = load_papers_jsonl(papers_path)
    rows = []
    for paper in papers:
        item_type = _first_metadata_value(paper.metadata, "sciimpact_item_types", "paper")
        impact_dimension = _first_metadata_value(
            paper.metadata,
            "sciimpact_impact_dimensions",
            "unknown",
        )
        score = score_sciimpact_text_baseline(
            title=paper.title,
            abstract=paper.abstract,
            item_type=item_type,
            impact_dimension=impact_dimension,
        )
        rows.append(
            {
                "id": paper.paper_id,
                "score": score,
                "final_score": score,
                "baseline_name": "sciimpact_text_prior",
                "item_type": item_type,
                "impact_dimension": impact_dimension,
            }
        )
    rows.sort(key=lambda row: (-float(row["final_score"]), str(row["id"])))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "id",
                "score",
                "final_score",
                "baseline_name",
                "item_type",
                "impact_dimension",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "id": row["id"],
                    "score": f"{float(row['score']):.6f}",
                    "final_score": f"{float(row['final_score']):.6f}",
                    "baseline_name": row["baseline_name"],
                    "item_type": row["item_type"],
                    "impact_dimension": row["impact_dimension"],
                }
            )

    manifest = {
        "baseline_name": "sciimpact_text_prior",
        "input_path": str(papers_path),
        "output_path": str(destination),
        "paper_count": len(rows),
    }
    (destination.parent / "sciimpact_text_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _cue_score(text: str, cue_weights: dict[str, float]) -> float:
    score = 0.0
    for cue, weight in cue_weights.items():
        if cue in text:
            score += weight
    return score


def _first_metadata_value(metadata: dict[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value
    return fallback
