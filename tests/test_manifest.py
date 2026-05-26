import hashlib
import json

from local_impact_ranking.manifest import build_run_manifest
from local_impact_ranking.models import Paper


def test_build_run_manifest_records_normalized_input_serving_backend_and_hardware(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text('{"paper_id":"p1","title":" Raw ","references":[]}\n', encoding="utf-8")
    normalized_papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Raw",
                "references": [{"reference_id": "r1", "title": "Reference"}],
            }
        )
    ]

    manifest = build_run_manifest(
        input_path,
        top_k=5,
        limit=100,
        credit_mode="rank_weighted",
        judge_name="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="local-impact-judge",
        temperature=0.0,
        repair_attempts=1,
        max_tokens=1024,
        judge_audit="runs/example/audit/judge_outputs.jsonl",
        normalized_papers=normalized_papers,
        serving_backend="vllm",
        serving_backend_version="0.8.5",
        model_path="models/deepseek",
        validation_report={
            "summary": {
                "issue_count": 2,
                "severity_counts": {"error": 1, "warning": 1},
                "can_run": False,
            }
        },
    )

    normalized_jsonl = (
        json.dumps(normalized_papers[0].model_dump(mode="json"), sort_keys=True) + "\n"
    )
    expected_sha = hashlib.sha256(normalized_jsonl.encode("utf-8")).hexdigest()
    assert manifest["input"]["normalized_sha256"] == expected_sha
    assert manifest["input"]["normalized_record_count"] == 1
    assert manifest["run"]["limit"] == 100
    assert manifest["judge"]["serving_backend"] == "vllm"
    assert manifest["judge"]["serving_backend_version"] == "0.8.5"
    assert manifest["judge"]["model_path"] == "models/deepseek"
    assert manifest["scoring"]["version"] == "score_v1_rank_weighted"
    assert manifest["scoring"]["credit_mode"] == "rank_weighted"
    assert manifest["scoring"]["rank_weights"] == {
        "1": 1.0,
        "2": 0.8,
        "3": 0.6,
        "4": 0.4,
        "5": 0.2,
    }
    assert manifest["scoring"]["normalization"] == "none"
    assert manifest["validation"] == {
        "can_run": False,
        "issue_count": 2,
        "error_count": 1,
        "warning_count": 1,
    }
    assert manifest["hardware"]["cpu_count"] is not None
    assert "gpu_summary" in manifest["hardware"]
