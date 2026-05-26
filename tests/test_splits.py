import json

import pytest

from local_impact_ranking.models import Paper
from local_impact_ranking.splits import split_papers, write_paper_splits


def test_split_papers_by_counts_preserves_order_and_has_no_overlap():
    papers = [_paper(f"p{i}") for i in range(1, 11)]

    splits = split_papers(papers, train_count=6, dev_count=2, test_count=2)

    assert [paper.paper_id for paper in splits["train"]] == [
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
    ]
    assert [paper.paper_id for paper in splits["dev"]] == ["p7", "p8"]
    assert [paper.paper_id for paper in splits["test"]] == ["p9", "p10"]
    split_ids = [
        paper.paper_id for split_papers in splits.values() for paper in split_papers
    ]
    assert set(split_ids) == {paper.paper_id for paper in papers}
    assert len(split_ids) == len(papers)


def test_split_papers_rejects_counts_that_do_not_cover_dataset():
    papers = [_paper(f"p{i}") for i in range(1, 6)]

    with pytest.raises(ValueError, match="sum to 5"):
        split_papers(papers, train_count=3, dev_count=1, test_count=0)


def test_write_paper_splits_exports_jsonl_and_manifest(tmp_path):
    papers = [_paper(f"p{i}") for i in range(1, 6)]

    manifest = write_paper_splits(
        papers,
        tmp_path,
        train_count=3,
        dev_count=1,
        test_count=1,
        input_path="papers.jsonl",
        input_sha256="abc123",
    )

    train_records = _read_jsonl(tmp_path / "train.papers.jsonl")
    dev_records = _read_jsonl(tmp_path / "dev.papers.jsonl")
    test_records = _read_jsonl(tmp_path / "test.papers.jsonl")
    written_manifest = json.loads((tmp_path / "split_manifest.json").read_text())

    assert [record["paper_id"] for record in train_records] == ["p1", "p2", "p3"]
    assert [record["paper_id"] for record in dev_records] == ["p4"]
    assert [record["paper_id"] for record in test_records] == ["p5"]
    assert manifest == written_manifest
    assert written_manifest["input"] == {
        "path": "papers.jsonl",
        "sha256": "abc123",
        "record_count": 5,
    }
    assert written_manifest["splits"]["train"]["count"] == 3
    assert written_manifest["splits"]["train"]["paper_ids"] == ["p1", "p2", "p3"]
    assert written_manifest["splits"]["dev"]["paper_ids"] == ["p4"]
    assert written_manifest["splits"]["test"]["paper_ids"] == ["p5"]


def _paper(paper_id: str) -> Paper:
    return Paper.model_validate(
        {
            "paper_id": paper_id,
            "title": f"Target {paper_id}",
            "references": [{"reference_id": f"{paper_id}-r1", "title": "Reference"}],
        }
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
