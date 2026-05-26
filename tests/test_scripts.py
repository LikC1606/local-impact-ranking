from pathlib import Path


def test_run_pilot_script_documents_required_arguments_and_defaults():
    script = Path("scripts/run_pilot.sh").read_text(encoding="utf-8")

    assert 'INPUT="${1:?Usage:' in script
    assert 'OUTPUT="${2:-runs/pilot-100}"' in script
    assert 'LIMIT="${LIMIT:-100}"' in script
    assert 'TOP_K="${TOP_K:-5}"' in script
    assert 'CONCURRENCY="${CONCURRENCY:-1}"' in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge}"' in script
    assert 'MAX_TOKENS="${MAX_TOKENS:-512}"' in script


def test_run_pilot_script_checks_server_judges_then_scores_from_audit():
    script = Path("scripts/run_pilot.sh").read_text(encoding="utf-8")

    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "--limit" in script
    assert '"${LIMIT}"' in script
    assert "--judge-audit" in script
    assert '--judge-audit "${AUDIT_PATH}" \\\n  --limit "${LIMIT}"' in script
    assert "--strict-validation" in script
    assert "--max-tokens" in script
    assert '"${AUDIT_PATH}"' in script
    assert 'test -s "${OUTPUT}/audit/prompts.jsonl"' in script
    assert 'test -s "${OUTPUT}/audit/scoring_edges.csv"' in script
    assert 'test -s "${OUTPUT}/rankings/fields.csv"' in script
    assert 'test -s "${OUTPUT}/rankings/years.csv"' in script
    assert 'test -s "${OUTPUT}/reports/validation.json"' in script
    assert 'test -s "${OUTPUT}/reports/validation.md"' in script
    assert 'test -s "${OUTPUT}/reports/artifact_index.json"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.md"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/proxy_evaluation.json"' in script
    assert 'test -f "${OUTPUT}/run_manifest.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_explanations.json"' in script
    assert 'test -s "${OUTPUT}/reports/entity_explanations.json"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.json"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script


def test_run_local_smoke_script_runs_heuristic_examples_and_checks_context_weight():
    script = Path("scripts/run_local_smoke.sh").read_text(encoding="utf-8")

    assert 'OUTPUT="${1:-runs/local-smoke}"' in script
    assert 'examples/tiny_papers.jsonl' in script
    assert 'examples/context_papers.jsonl' in script
    assert "--strict-validation" in script
    assert "--credit-mode context_weighted" in script
    assert 'test -s "${OUTPUT}/context/audit/prompts.jsonl"' in script
    assert 'test -s "${OUTPUT}/context/audit/scoring_edges.csv"' in script
    assert 'test -s "${OUTPUT}/context/rankings/fields.csv"' in script
    assert 'test -s "${OUTPUT}/context/rankings/years.csv"' in script
    assert 'test -s "${OUTPUT}/context/reports/evaluation.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/context/reports/mvp_summary.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/artifact_index.json"' in script
    assert 'grep -q "1,r1,1.500000" "${OUTPUT}/context/rankings/papers.csv"' in script


def test_run_benchmark_smoke_script_runs_limited_scicite_pipeline():
    script = Path("scripts/run_benchmark_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/scicite/dev.papers.jsonl}"' in script
    assert "--limit" in script
    assert '"${LIMIT}"' in script
    assert "--credit-mode context_weighted" in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script
    assert 'test -s "${OUTPUT}/reports/proxy_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/rankings/papers.csv"' in script


def test_download_prescience_sample_script_streams_from_mirror():
    script = Path("scripts/download_prescience_sample.sh").read_text(encoding="utf-8")

    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in script
    assert 'TARGETS="${TARGETS:-100}"' in script
    assert "load_dataset('allenai/prescience'" in script
    assert "os._exit(0)" in script


def test_run_prescience_smoke_script_downloads_converts_and_runs_gold_eval():
    script = Path("scripts/run_prescience_smoke.sh").read_text(encoding="utf-8")

    assert 'scripts/download_prescience_sample.sh "${SAMPLE_PATH}"' in script
    assert "benchmarks convert-prescience" in script
    assert "--max-candidates" in script
    assert "--strict-validation" in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.md"' in script


def test_run_prescience_gpu_benchmark_script_judges_incrementally_then_scores():
    script = Path("scripts/run_prescience_gpu_benchmark.sh").read_text(encoding="utf-8")

    assert 'SAMPLE_PATH="${SAMPLE_PATH:-data/prescience/sample.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-100}"' in script
    assert 'MAX_CANDIDATES="${MAX_CANDIDATES:-32}"' in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge}"' in script
    assert 'AUDIT_PATH="${OUTPUT}/audit/judge_outputs.jsonl"' in script
    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "benchmarks convert-prescience" in script
    assert "-m local_impact_ranking.cli judge" in script
    assert "--audit-output" in script
    assert '"${AUDIT_PATH}"' in script
    assert "-m local_impact_ranking.cli score" in script
    assert '--judge-audit "${AUDIT_PATH}"' in script
    assert 'test -s "${OUTPUT}/audit/judge_outputs.jsonl"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script
    assert 'test -f "${OUTPUT}/run_manifest.json"' in script


def test_download_scicite_script_downloads_extracts_and_converts_splits():
    script = Path("scripts/download_scicite.sh").read_text(encoding="utf-8")

    assert 'SCICITE_URL="${SCICITE_URL:-https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz}"' in script
    assert 'DEST_DIR="${1:-data/scicite}"' in script
    assert 'curl -L "${SCICITE_URL}" -o "${ARCHIVE_PATH}"' in script
    assert 'tar -xzf "${ARCHIVE_PATH}" -C "${DEST_DIR}/raw"' in script
    assert 'for split in train dev test; do' in script
    assert 'benchmarks convert-scicite' in script
    assert '${split}.papers.jsonl' in script
