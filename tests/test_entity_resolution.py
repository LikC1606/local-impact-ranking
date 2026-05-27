import json

from local_impact_ranking.entity_resolution import (
    entity_resolution_report,
    render_entity_resolution_markdown,
    write_entity_resolution_report,
)
from local_impact_ranking.io import load_papers_jsonl


def test_entity_resolution_report_quantifies_missing_and_ambiguous_entities(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "authors": [{"name": "Ada Lovelace", "ids": {"s2": "a1"}}],
                        "references": [
                            {
                                "reference_id": "r1",
                                "title": "Reference One",
                                "authors": [
                                    {
                                        "name": "Grace Hopper",
                                        "affiliations": ["Navy"],
                                        "ids": {"s2": "g1"},
                                    },
                                    {
                                        "name": "Grace Hopper",
                                        "affiliations": ["Navy"],
                                        "ids": {"s2": "g2"},
                                    },
                                ],
                            },
                            {
                                "reference_id": "r2",
                                "title": "Reference Two",
                                "authors": [{"name": "  ", "affiliations": []}],
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Target Two",
                        "references": [
                            {
                                "reference_id": "r3",
                                "title": "Reference Three",
                                "authors": [
                                    {
                                        "name": "Grace Hopper",
                                        "affiliations": ["US Navy"],
                                        "ids": {"s2": "g1"},
                                    }
                                ],
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    papers = load_papers_jsonl(input_path)

    report = entity_resolution_report(papers)

    assert report["summary"]["paper_count"] == 2
    assert report["summary"]["reference_count"] == 3
    assert report["summary"]["author_mention_count"] == 5
    assert report["summary"]["reference_author_mention_count"] == 4
    assert report["summary"]["author_id_coverage"] == 0.75
    assert report["summary"]["affiliation_coverage"] == 0.75
    assert report["summary"]["same_name_multiple_id_count"] == 1
    assert report["summary"]["same_id_multiple_name_count"] == 0
    assert report["summary"]["duplicate_reference_title_count"] == 0
    assert report["same_name_multiple_ids"][0]["name"] == "grace hopper"
    assert sorted(report["same_name_multiple_ids"][0]["ids"]) == ["g1", "g2"]
    assert report["institution_name_variants"][0]["normalized_name"] == "navy"
    assert report["quality_gate"]["entity_rankings_scope"] == "exploratory"


def test_write_entity_resolution_report_exports_json_and_markdown(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "entity-report"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Reference",
                        "authors": [
                            {
                                "name": "Author A",
                                "affiliations": ["Institution A"],
                                "ids": {"s2": "a1"},
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_entity_resolution_report(load_papers_jsonl(input_path), output_dir)

    assert report["quality_gate"]["entity_rankings_scope"] == "reportable"
    assert (output_dir / "entity_resolution_report.json").exists()
    markdown = (output_dir / "entity_resolution_report.md").read_text(encoding="utf-8")
    assert "# Entity Resolution Report" in markdown
    assert "| entity_rankings_scope | reportable |" in markdown


def test_render_entity_resolution_markdown_handles_empty_report():
    markdown = render_entity_resolution_markdown(
        {
            "summary": {},
            "same_name_multiple_ids": [],
            "same_id_multiple_names": [],
            "institution_name_variants": [],
            "duplicate_reference_titles": [],
            "quality_gate": {"entity_rankings_scope": "exploratory"},
        }
    )

    assert markdown.startswith("# Entity Resolution Report\n")
    assert "| entity_rankings_scope | exploratory |" in markdown
