import json

import httpx

from local_impact_ranking.judge import (
    HeuristicJudge,
    OpenAICompatibleJudge,
    build_judge_prompt,
    parse_judge_output,
)
from local_impact_ranking.models import Paper


def sample_paper():
    return Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target Paper",
            "abstract": "This paper builds on representation learning.",
            "references": [
                {"reference_id": "r1", "title": "Representation Learning", "year": 2019},
                {"reference_id": "r2", "title": "Unrelated Systems Work", "year": 2020},
            ],
        }
    )


def test_build_judge_prompt_lists_candidate_reference_ids():
    prompt = build_judge_prompt(sample_paper(), top_k=2)

    assert "p1" in prompt
    assert "r1" in prompt
    assert "r2" in prompt
    assert "JSON" in prompt


def test_build_judge_prompt_includes_available_citation_contexts():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target Paper",
            "references": [{"reference_id": "r1", "title": "Representation Learning"}],
            "citation_contexts": [
                {
                    "reference_id": "r1",
                    "sentence": "We build directly on Representation Learning.",
                    "section": "Method",
                }
            ],
        }
    )

    prompt = build_judge_prompt(paper, top_k=1)

    assert "Citation contexts" in prompt
    assert "section: Method" in prompt
    assert "We build directly on Representation Learning." in prompt


def test_build_judge_prompt_requires_single_candidate_selection():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target Paper",
            "references": [{"reference_id": "r1", "title": "Representation Learning"}],
        }
    )

    prompt = build_judge_prompt(paper, top_k=1)

    assert "single candidate" in prompt.lower()
    assert "must select that candidate" in prompt.lower()


def test_parse_judge_output_accepts_valid_selected_references():
    raw = json.dumps(
        {
            "target_paper_id": "p1",
            "selected_references": [
                {
                    "reference_id": "r1",
                    "rank": 1,
                    "confidence": 0.9,
                    "rationale": "Foundational to the target method.",
                }
            ],
        }
    )

    parsed = parse_judge_output(raw, sample_paper())

    assert parsed.target_paper_id == "p1"
    assert parsed.selected_references[0].reference_id == "r1"
    assert parsed.selected_references[0].valid is True
    assert parsed.parse_error is None


def test_parse_judge_output_preserves_input_paper_id_when_model_returns_title_like_id():
    raw = json.dumps(
        {
            "target_paper_id": "SciCite citing paper p1",
            "selected_references": [
                {
                    "reference_id": "r1",
                    "rank": 1,
                    "confidence": 0.9,
                    "rationale": "ok",
                }
            ],
        }
    )

    parsed = parse_judge_output(raw, sample_paper())

    assert parsed.target_paper_id == "p1"


def test_parse_judge_output_extracts_json_object_after_reasoning_text():
    raw = (
        "thinking text before json\n"
        '{"target_paper_id":"p1","selected_references":['
        '{"reference_id":"r1","rank":1,"confidence":0.9,"rationale":"ok"}]}'
    )

    parsed = parse_judge_output(raw, sample_paper())

    assert parsed.parse_error is None
    assert parsed.selected_references[0].reference_id == "r1"


def test_parse_judge_output_marks_unknown_reference_invalid():
    raw = json.dumps(
        {
            "target_paper_id": "p1",
            "selected_references": [
                {
                    "reference_id": "not-a-candidate",
                    "rank": 1,
                    "confidence": 0.5,
                    "rationale": "The model invented this ID.",
                }
            ],
        }
    )

    parsed = parse_judge_output(raw, sample_paper())

    assert parsed.selected_references[0].valid is False
    assert parsed.invalid_selection_count == 1


def test_heuristic_judge_returns_parseable_top_k_json():
    result = HeuristicJudge().judge(sample_paper(), top_k=1)

    assert len(result.selected_references) == 1
    assert result.selected_references[0].reference_id == "r1"
    assert result.selected_references[0].valid is True
    assert len(result.prompt_sha256) == 64
    assert result.candidate_reference_ids == ("r1", "r2")


def test_heuristic_judge_uses_target_title_when_scoring_references():
    paper = Paper.model_validate(
        {
            "paper_id": "p-title",
            "title": "Graph Neural Networks",
            "abstract": "We introduce a scalable training recipe.",
            "references": [
                {"reference_id": "r1", "title": "Scalable Training Recipe"},
                {"reference_id": "r2", "title": "Graph Neural Networks"},
            ],
        }
    )

    result = HeuristicJudge().judge(paper, top_k=1)

    assert result.selected_references[0].reference_id == "r2"


def test_heuristic_judge_prefers_specific_terms_over_many_generic_terms():
    paper = Paper.model_validate(
        {
            "paper_id": "p-specific",
            "title": "Sparse Transformer Retrieval",
            "abstract": "Sparse retrieval improves matching.",
            "references": [
                {"reference_id": "r1", "title": "Retrieval Matching Improves"},
                {"reference_id": "r2", "title": "Sparse Transformer"},
            ],
        }
    )

    result = HeuristicJudge().judge(paper, top_k=1)

    assert result.selected_references[0].reference_id == "r2"


def test_openai_compatible_judge_repairs_malformed_json_output():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        content = (
            "not json"
            if len(calls) == 1
            else json.dumps(
                {
                    "target_paper_id": "p1",
                    "selected_references": [
                        {
                            "reference_id": "r1",
                            "rank": 1,
                            "confidence": 0.8,
                            "rationale": "repair",
                        }
                    ],
                }
            )
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    judge = OpenAICompatibleJudge(
        base_url="http://test/v1",
        model="local-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        repair_attempts=1,
    )

    result = judge.judge(sample_paper(), top_k=1)

    assert len(calls) == 2
    assert "Repair" in calls[1]["messages"][0]["content"]
    assert result.parse_error is None
    assert result.selected_references[0].reference_id == "r1"


def test_openai_compatible_judge_repairs_invalid_reference_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        reference_id = "invented" if len(calls) == 1 else "r1"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "target_paper_id": "p1",
                                    "selected_references": [
                                        {
                                            "reference_id": reference_id,
                                            "rank": 1,
                                            "confidence": 0.8,
                                            "rationale": "repair",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        base_url="http://test/v1",
        model="local-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        repair_attempts=1,
    )

    result = judge.judge(sample_paper(), top_k=1)

    assert len(calls) == 2
    assert result.invalid_selection_count == 0
    assert result.selected_references[0].reference_id == "r1"


def test_openai_compatible_judge_sends_max_tokens_when_configured():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "target_paper_id": "p1",
                                    "selected_references": [
                                        {
                                            "reference_id": "r1",
                                            "rank": 1,
                                            "confidence": 0.8,
                                            "rationale": "ok",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        base_url="http://test/v1",
        model="local-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_tokens=256,
    )

    judge.judge(sample_paper(), top_k=1)

    assert calls[0]["max_tokens"] == 256


def test_openai_compatible_judge_requests_json_object_response_by_default():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "target_paper_id": "p1",
                                    "selected_references": [
                                        {
                                            "reference_id": "r1",
                                            "rank": 1,
                                            "confidence": 0.8,
                                            "rationale": "ok",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        base_url="http://test/v1",
        model="local-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    judge.judge(sample_paper(), top_k=1)

    assert calls[0]["response_format"] == {"type": "json_object"}


def test_openai_compatible_judge_can_disable_json_object_response_format():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "target_paper_id": "p1",
                                    "selected_references": [
                                        {
                                            "reference_id": "r1",
                                            "rank": 1,
                                            "confidence": 0.8,
                                            "rationale": "ok",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    judge = OpenAICompatibleJudge(
        base_url="http://test/v1",
        model="local-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        json_mode=False,
    )

    judge.judge(sample_paper(), top_k=1)

    assert "response_format" not in calls[0]
