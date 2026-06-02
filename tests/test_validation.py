import json

from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.validation import (
    render_validation_markdown,
    validation_report,
)


def test_validation_report_flags_duplicate_ids_empty_references_and_bad_years(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "year": 2024,
                        "references": [
                            {"reference_id": "r1", "title": "Reference One"},
                            {"reference_id": "r1", "title": "Reference One Duplicate"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "",
                        "year": 1700,
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    papers = load_papers_jsonl(path)

    report = validation_report(papers)

    assert report["record_count"] == 2
    assert report["summary"]["issue_count"] == 5
    assert report["summary"]["severity_counts"] == {"error": 3, "warning": 2}
    assert sorted(report["issues"], key=lambda issue: (issue["code"], issue["message"])) == sorted([
        {
            "severity": "error",
            "code": "duplicate_paper_id",
            "paper_id": "p1",
            "message": "paper_id appears 2 times",
        },
        {
            "severity": "warning",
            "code": "empty_reference_list",
            "paper_id": "p1",
            "message": "paper has no candidate references; evidence coverage may be weaker",
        },
        {
            "severity": "warning",
            "code": "missing_title",
            "paper_id": "p1",
            "message": "paper title is empty",
        },
        {
            "severity": "error",
            "code": "invalid_year",
            "paper_id": "p1",
            "message": "paper year 1700 is outside supported range 1800-2100",
        },
        {
            "severity": "error",
            "code": "duplicate_reference_id",
            "paper_id": "p1",
            "message": "reference_id r1 appears 2 times in one paper",
        },
    ], key=lambda issue: (issue["code"], issue["message"]))


def test_render_validation_markdown_summarizes_issues():
    markdown = render_validation_markdown(
        {
            "record_count": 1,
            "summary": {
                "issue_count": 1,
                "severity_counts": {"error": 1},
                "can_run": False,
            },
            "issues": [
                {
                    "severity": "warning",
                    "code": "empty_reference_list",
                    "paper_id": "p1",
                    "message": "paper has no candidate references; evidence coverage may be weaker",
                }
            ],
        }
    )

    assert markdown.startswith("# Validation Report\n")
    assert "| Can run | no |" in markdown
    assert (
        "| warning | empty_reference_list | p1 | "
        "paper has no candidate references; evidence coverage may be weaker |"
    ) in markdown
