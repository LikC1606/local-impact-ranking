import json

from local_impact_ranking.citation_role_classifier import (
    ALLOWED_CITATION_ROLES,
    build_citation_role_prompt,
    parse_citation_role_output,
)
from local_impact_ranking.models import CitationContext, Paper


def test_build_citation_role_prompt_contains_fixed_role_schema():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Citing paper",
            "abstract": "Uses a protocol.",
            "references": [{"reference_id": "r1", "title": "Protocol paper"}],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We used this protocol for the assay.",
                    "section": "Methods",
                }
            ],
        }
    )

    prompt = build_citation_role_prompt(
        paper,
        paper.citation_contexts[0],
        context_index=0,
    )

    assert "Return JSON only" in prompt
    for role in ALLOWED_CITATION_ROLES:
        assert role in prompt
    assert "We used this protocol" in prompt
    assert "Protocol paper" in prompt


def test_build_citation_role_prompt_defaults_context_index_to_zero():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Citing paper",
            "references": [{"reference_id": "r1", "title": "Protocol paper"}],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We used this protocol for the assay.",
                }
            ],
        }
    )

    prompt = build_citation_role_prompt(paper, paper.citation_contexts[0])

    assert "Context index: 0" in prompt


def test_parse_citation_role_output_accepts_valid_role():
    parsed = parse_citation_role_output(
        json.dumps(
            {
                "role": "method",
                "confidence": 0.86,
                "rationale": "The sentence says a protocol was used.",
            }
        )
    )

    assert parsed["role"] == "method"
    assert parsed["confidence"] == 0.86
    assert parsed["validation_status"] == "valid"


def test_parse_citation_role_output_invalid_json_becomes_unknown():
    parsed = parse_citation_role_output("not json")

    assert parsed["role"] == "unknown"
    assert parsed["confidence"] == 0.0
    assert parsed["validation_status"] == "invalid_json"
    assert "raw output was not valid JSON" in parsed["rationale"]


def test_parse_citation_role_output_illegal_role_becomes_unknown():
    parsed = parse_citation_role_output(
        json.dumps({"role": "important", "confidence": 0.7, "rationale": "bad"})
    )

    assert parsed["role"] == "unknown"
    assert parsed["confidence"] == 0.0
    assert parsed["validation_status"] == "invalid_role"


def test_parse_citation_role_output_clamps_confidence():
    parsed = parse_citation_role_output(
        json.dumps({"role": "supporting", "confidence": 7, "rationale": "ok"})
    )

    assert parsed["confidence"] == 1.0
