from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from local_impact_ranking.models import Paper


class PaperSplits(TypedDict):
    train: list[Paper]
    dev: list[Paper]
    test: list[Paper]


def split_papers(
    papers: list[Paper],
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
) -> PaperSplits:
    total = len(papers)
    requested_total = train_count + dev_count + test_count
    if requested_total != total:
        raise ValueError(
            "Split counts must sum to "
            f"{total}; got train={train_count}, dev={dev_count}, test={test_count} "
            f"(sum={requested_total})"
        )
    if min(train_count, dev_count, test_count) < 0:
        raise ValueError("Split counts must be non-negative")

    train_end = train_count
    dev_end = train_end + dev_count
    return {
        "train": papers[:train_end],
        "dev": papers[train_end:dev_end],
        "test": papers[dev_end:],
    }


def write_paper_splits(
    papers: list[Paper],
    output_dir: str | Path,
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    input_path: str | Path | None = None,
    input_sha256: str | None = None,
) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    splits = split_papers(
        papers,
        train_count=train_count,
        dev_count=dev_count,
        test_count=test_count,
    )

    for split_name, split_papers_ in splits.items():
        _write_papers_jsonl(destination / f"{split_name}.papers.jsonl", split_papers_)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path) if input_path is not None else None,
            "sha256": input_sha256,
            "record_count": len(papers),
        },
        "method": {
            "type": "ordered_prefix",
            "note": "Input order is preserved; no record is shuffled across splits.",
        },
        "splits": {
            split_name: {
                "path": f"{split_name}.papers.jsonl",
                "count": len(split_papers_),
                "paper_ids": [paper.paper_id for paper in split_papers_],
            }
            for split_name, split_papers_ in splits.items()
        },
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_papers_jsonl(path: Path, papers: list[Paper]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.model_dump(mode="json"), ensure_ascii=False) + "\n")
