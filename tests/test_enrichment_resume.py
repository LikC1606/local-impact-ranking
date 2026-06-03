import json

import pytest

from local_impact_ranking.enrichment_resume import (
    resume_rows_by_paper_id,
    write_jsonl_atomic,
)


def test_resume_rows_by_paper_id_ignores_truncated_final_line(tmp_path):
    resume_path = tmp_path / "partial.jsonl"
    resume_path.write_text(
        json.dumps({"paper_id": "p1", "title": "complete"}) + "\n"
        '{"paper_id": "p2", "title":',
        encoding="utf-8",
    )

    rows = resume_rows_by_paper_id(resume_path)

    assert rows == {"p1": {"paper_id": "p1", "title": "complete"}}


def test_resume_rows_by_paper_id_rejects_invalid_middle_line(tmp_path):
    resume_path = tmp_path / "corrupt.jsonl"
    resume_path.write_text(
        json.dumps({"paper_id": "p1", "title": "complete"}) + "\n"
        '{"paper_id": "broken"\n'
        + json.dumps({"paper_id": "p2", "title": "complete"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid resume JSON on line 2"):
        resume_rows_by_paper_id(resume_path)


def test_write_jsonl_atomic_preserves_existing_file_on_failure(tmp_path):
    output_path = tmp_path / "rows.jsonl"
    output_path.write_text(
        json.dumps({"paper_id": "old", "title": "existing"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        write_jsonl_atomic(output_path, [{"paper_id": "new", "bad": object()}])

    assert output_path.read_text(encoding="utf-8") == (
        json.dumps({"paper_id": "old", "title": "existing"}) + "\n"
    )


def test_write_jsonl_atomic_writes_complete_jsonl(tmp_path):
    output_path = tmp_path / "rows.jsonl"

    write_jsonl_atomic(output_path, [{"paper_id": "p1"}, {"paper_id": "p2"}])

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows == [{"paper_id": "p1"}, {"paper_id": "p2"}]
