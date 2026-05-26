import json

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.judge import HeuristicJudge


def test_cli_validate_writes_validation_reports(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "validation"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "references": [{"reference_id": "r1", "title": "Reference One"}],
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

    result = CliRunner().invoke(
        app,
        ["validate", "--input", str(input_path), "--output", str(output_dir)],
    )

    assert result.exit_code == 0
    report = json.loads((output_dir / "validation.json").read_text())
    markdown = (output_dir / "validation.md").read_text()
    assert report["summary"]["can_run"] is False
    assert "Validation issues: 4" in result.stdout
    assert "# Validation Report" in markdown


def test_cli_convert_csv_writes_jsonl_that_validate_can_read(tmp_path):
    input_path = tmp_path / "papers.csv"
    output_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                "paper_id,title,references_json",
                '"p1","Target","[{""reference_id"":""r1"",""title"":""Reference""}]"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["convert-csv", "--input", str(input_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert "Wrote 1 converted papers" in result.stdout
    assert json.loads(output_path.read_text())["paper_id"] == "p1"


def test_cli_train_dpo_delegates_to_lora_trainer(monkeypatch, tmp_path):
    train_file_one = tmp_path / "train-one.jsonl"
    train_file_two = tmp_path / "train-two.jsonl"
    train_file_one.write_text("{}\n", encoding="utf-8")
    train_file_two.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "adapter"
    captured = {}

    def fake_run_dpo_lora_training(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_dpo_lora_training", fake_run_dpo_lora_training)

    result = CliRunner().invoke(
        app,
        [
            "train-dpo",
            "--model",
            "/models/base",
            "--train-file",
            str(train_file_one),
            "--train-file",
            str(train_file_two),
            "--output",
            str(output_dir),
            "--max-steps",
            "3",
            "--max-examples",
            "2",
            "--beta",
            "0.2",
            "--learning-rate",
            "2e-6",
            "--seed",
            "42",
            "--gradient-accumulation-steps",
            "4",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.1",
            "--device-map",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_name_or_path"] == "/models/base"
    assert captured["train_files"] == [train_file_one, train_file_two]
    assert captured["output_dir"] == output_dir
    assert captured["max_steps"] == 3
    assert captured["max_examples"] == 2
    assert captured["beta"] == 0.2
    assert captured["learning_rate"] == 2e-6
    assert captured["seed"] == 42
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["lora_r"] == 8
    assert captured["lora_alpha"] == 16
    assert captured["lora_dropout"] == 0.1
    assert captured["device_map"] == "cpu"


def test_cli_split_writes_train_dev_test_files_and_manifest(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "splits"
    records = [
        {
            "paper_id": f"p{i}",
            "title": f"Target {i}",
            "references": [{"reference_id": f"r{i}", "title": f"Reference {i}"}],
        }
        for i in range(1, 6)
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "split",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--train-count",
            "3",
            "--dev-count",
            "1",
            "--test-count",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote paper splits" in result.stdout
    assert (output_dir / "train.papers.jsonl").exists()
    assert (output_dir / "dev.papers.jsonl").exists()
    assert (output_dir / "test.papers.jsonl").exists()
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["splits"]["train"]["paper_ids"] == ["p1", "p2", "p3"]
    assert manifest["splits"]["dev"]["paper_ids"] == ["p4"]
    assert manifest["splits"]["test"]["paper_ids"] == ["p5"]


def test_cli_run_writes_rankings_and_audit_outputs(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target Representation Paper",
                "abstract": "representation learning",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Representation Learning",
                        "authors": [{"name": "Author A", "affiliations": ["Institution X"]}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(input_path), "--output", str(output_dir), "--top-k", "1"],
    )

    assert result.exit_code == 0
    assert (output_dir / "rankings" / "papers.csv").exists()
    assert (output_dir / "rankings" / "fields.csv").exists()
    assert (output_dir / "rankings" / "years.csv").exists()
    assert (output_dir / "audit" / "judge_outputs.jsonl").exists()
    assert (output_dir / "audit" / "prompts.jsonl").exists()
    assert (output_dir / "audit" / "scoring_edges.csv").exists()
    assert (output_dir / "reports" / "evaluation.json").exists()
    assert (output_dir / "reports" / "evaluation.md").exists()
    assert (output_dir / "reports" / "gold_evaluation.json").exists()
    assert (output_dir / "reports" / "gold_evaluation.md").exists()
    assert (output_dir / "review" / "human_audit_sample.csv").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "baselines" / "citation_count.csv").exists()
    assert (output_dir / "reports" / "baseline_comparison.json").exists()
    assert (output_dir / "reports" / "context_coverage.json").exists()
    assert (output_dir / "reports" / "validation.json").exists()
    assert (output_dir / "reports" / "validation.md").exists()
    assert (output_dir / "reports" / "artifact_index.json").exists()
    assert (output_dir / "reports" / "proxy_evaluation.json").exists()
    assert (output_dir / "reports" / "data_readiness.json").exists()
    assert (output_dir / "reports" / "anomaly_report.json").exists()
    assert (output_dir / "reports" / "scoring_config.json").exists()
    assert (output_dir / "reports" / "stability_report.json").exists()
    assert (output_dir / "reports" / "mvp_summary.json").exists()
    assert (output_dir / "reports" / "mvp_summary.md").exists()
    assert (output_dir / "reports" / "benchmark_summary.json").exists()
    assert (output_dir / "reports" / "benchmark_summary.md").exists()
    assert (output_dir / "reports" / "paper_explanations.json").exists()
    assert (output_dir / "reports" / "entity_explanations.json").exists()
    assert (output_dir / "finetuning" / "format_sft.jsonl").exists()
    assert (output_dir / "finetuning" / "proxy_positive_sft.jsonl").exists()
    assert (output_dir / "finetuning" / "pairwise_preferences.jsonl").exists()
    assert (output_dir / "finetuning" / "dpo_preferences.jsonl").exists()
    assert (output_dir / "finetuning" / "gold_hard_negative_dpo.jsonl").exists()
    assert (output_dir / "finetuning" / "llamafactory_dataset_info.json").exists()
    assert (output_dir / "finetuning" / "llamafactory_sft.yaml").exists()
    assert (output_dir / "finetuning" / "llamafactory_dpo.yaml").exists()
    artifact_index = json.loads((output_dir / "reports" / "artifact_index.json").read_text())
    artifact_paths = {artifact["path"] for artifact in artifact_index["artifacts"]}
    assert "run_manifest.json" in artifact_paths
    assert "rankings/papers.csv" in artifact_paths
    assert "rankings/fields.csv" in artifact_paths
    assert "reports/evaluation.md" in artifact_paths
    assert "reports/gold_evaluation.json" in artifact_paths
    assert "reports/benchmark_summary.md" in artifact_paths
    benchmark_summary = json.loads(
        (output_dir / "reports" / "benchmark_summary.json").read_text()
    )
    assert benchmark_summary["benchmark"]["paper_count"] == 1
    assert benchmark_summary["judge_quality"]["parse_success_rate"] == 1.0
    prompt_record = json.loads((output_dir / "audit" / "prompts.jsonl").read_text())
    assert len(prompt_record["prompt_sha256"]) == 64
    assert prompt_record["candidate_reference_ids"] == ["r1"]
    assert "Candidate references" in prompt_record["prompt"]


def test_cli_run_strict_validation_stops_on_validation_errors(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "references": [{"reference_id": "r1", "title": "Reference One"}],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target Duplicate",
                        "references": [{"reference_id": "r2", "title": "Reference Two"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--strict-validation",
        ],
    )

    assert result.exit_code != 0
    assert (output_dir / "reports" / "validation.json").exists()
    assert not (output_dir / "rankings" / "papers.csv").exists()
    assert "Validation failed" in result.stdout


def test_cli_run_limit_scores_only_limited_prefix(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "abstract": "alpha",
                        "references": [{"reference_id": "r1", "title": "Alpha Reference"}],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Target Two",
                        "abstract": "beta",
                        "references": [{"reference_id": "r2", "title": "Beta Reference"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--top-k",
            "1",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    papers_csv = (output_dir / "rankings" / "papers.csv").read_text()
    evaluation = json.loads((output_dir / "reports" / "evaluation.json").read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert "r1" in papers_csv
    assert "r2" not in papers_csv
    assert evaluation["judge_results"] == 1
    assert manifest["run"]["limit"] == 1
    assert manifest["input"]["normalized_record_count"] == 1


def test_cli_run_reports_data_readiness_and_anomalies(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target One",
                        "references": [
                            {
                                "reference_id": "r1",
                                "title": "Reference One",
                                "authors": [
                                    {"name": "Author A", "affiliations": ["Institution X"]}
                                ],
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Target Two",
                        "references": [
                            {
                                "reference_id": "r1",
                                "title": "Reference One",
                                "authors": [
                                    {"name": "Author A", "affiliations": ["Institution X"]}
                                ],
                            },
                            {
                                "reference_id": "r2",
                                "title": "Reference Two",
                                "authors": [
                                    {"name": "Author B", "affiliations": ["Institution Y"]}
                                ],
                            },
                            {
                                "reference_id": "r3",
                                "title": "Reference Three",
                                "authors": [
                                    {"name": "Author C", "affiliations": ["Institution Z"]}
                                ],
                            },
                            {
                                "reference_id": "r4",
                                "title": "Reference Four",
                                "authors": [
                                    {"name": "Author D", "affiliations": ["Institution W"]}
                                ],
                            },
                            {
                                "reference_id": "r5",
                                "title": "Reference Five",
                                "authors": [
                                    {"name": "Author E", "affiliations": ["Institution V"]}
                                ],
                            },
                            {
                                "reference_id": "r6",
                                "title": "Reference Six",
                                "authors": [
                                    {"name": "Author F", "affiliations": ["Institution U"]}
                                ],
                            },
                            {
                                "reference_id": "r7",
                                "title": "Reference Seven",
                                "authors": [
                                    {"name": "Author G", "affiliations": ["Institution T"]}
                                ],
                            },
                            {
                                "reference_id": "r8",
                                "title": "Reference Eight",
                                "authors": [
                                    {"name": "Author H", "affiliations": ["Institution S"]}
                                ],
                            },
                            {
                                "reference_id": "r9",
                                "title": "Reference Nine",
                                "authors": [
                                    {"name": "Author I", "affiliations": ["Institution R"]}
                                ],
                            },
                            {
                                "reference_id": "r10",
                                "title": "",
                                "authors": [
                                    {"name": "Author J", "affiliations": ["Institution Q"]}
                                ],
                            },
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(input_path), "--output", str(output_dir), "--top-k", "1"],
    )

    assert result.exit_code == 0
    readiness = json.loads((output_dir / "reports" / "data_readiness.json").read_text())
    anomaly = json.loads((output_dir / "reports" / "anomaly_report.json").read_text())

    assert readiness["ranking_scope"]["paper_ranking"] == "formal"
    assert readiness["ranking_scope"]["author_ranking"] == "formal"
    assert readiness["ranking_scope"]["institution_ranking"] == "formal"
    assert readiness["coverage"]["reference_title_coverage"] < 1.0
    assert anomaly["score_dominance"]["paper_single_edge_dominance"][0]["id"] == "r1"
    assert anomaly["missing_metadata"]["references_missing_title_count"] == 1


def test_cli_run_writes_reproducibility_manifest(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target Representation Paper",
                "abstract": "representation learning",
                "references": [{"reference_id": "r1", "title": "Representation Learning"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--top-k",
            "1",
            "--credit-mode",
            "flat",
            "--judge",
            "heuristic",
        ],
    )

    assert result.exit_code == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["path"] == str(input_path)
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["run"]["top_k"] == 1
    assert manifest["run"]["credit_mode"] == "flat"
    assert manifest["judge"]["name"] == "heuristic"
    assert manifest["prompt"]["version"]
    assert manifest["validation"]["can_run"] is True
    assert manifest["validation"]["issue_count"] == 0


def test_cli_run_records_context_weighted_scoring_config(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target Representation Paper",
                "references": [{"reference_id": "r1", "title": "Representation Learning"}],
                "citation_contexts": [
                    {
                        "reference_id": "r1",
                        "sentence": "We build on this method.",
                        "section": "Methods",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--top-k",
            "1",
            "--credit-mode",
            "context_weighted",
        ],
    )

    assert result.exit_code == 0
    config = json.loads((output_dir / "reports" / "scoring_config.json").read_text())
    stability = json.loads((output_dir / "reports" / "stability_report.json").read_text())
    summary = json.loads((output_dir / "reports" / "mvp_summary.json").read_text())
    summary_md = (output_dir / "reports" / "mvp_summary.md").read_text()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert config["version"] == "score_v2_context_weighted"
    assert config["context_weighting"]["method_or_data_or_baseline_supported"] == 1.5
    assert stability["primary_credit_mode"] == "context_weighted"
    assert summary["scoring"]["version"] == "score_v2_context_weighted"
    assert summary["claim_scope"] == "dataset-local content-aware impact ranking"
    assert summary["ranking_preview"]["papers"][0]["id"] == "r1"
    assert "# Local Impact Ranking MVP Summary" in summary_md
    assert "## Top Papers" in summary_md
    assert manifest["scoring"]["version"] == "score_v2_context_weighted"


def test_cli_run_can_select_openai_compatible_judge(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target Representation Paper",
                "abstract": "representation learning",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Representation Learning",
                        "authors": [{"name": "Author A", "affiliations": ["Institution X"]}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeOpenAIJudge(HeuristicJudge):
        def __init__(
            self,
            base_url,
            model,
            api_key,
            timeout,
            temperature,
            repair_attempts,
            max_tokens,
            json_mode,
        ):
            captured.update(
                {
                    "base_url": base_url,
                    "model": model,
                    "api_key": api_key,
                    "timeout": timeout,
                    "temperature": temperature,
                    "repair_attempts": repair_attempts,
                    "max_tokens": max_tokens,
                    "json_mode": json_mode,
                }
            )

    monkeypatch.setattr(cli, "OpenAICompatibleJudge", FakeOpenAIJudge)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--judge",
            "openai-compatible",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "Qwen3",
            "--api-key",
            "token",
            "--temperature",
            "0.2",
            "--repair-attempts",
            "2",
            "--max-tokens",
            "256",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen3",
        "api_key": "token",
        "timeout": 120.0,
        "temperature": 0.2,
        "repair_attempts": 2,
        "max_tokens": 256,
        "json_mode": True,
    }


def test_cli_openai_compatible_judge_can_disable_json_mode(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target",
                "references": [{"reference_id": "r1", "title": "Known"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeOpenAIJudge(HeuristicJudge):
        def __init__(
            self,
            base_url,
            model,
            api_key,
            timeout,
            temperature,
            repair_attempts,
            max_tokens,
            json_mode,
        ):
            captured["json_mode"] = json_mode

    monkeypatch.setattr(cli, "OpenAICompatibleJudge", FakeOpenAIJudge)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--judge",
            "openai-compatible",
            "--model",
            "local-model",
            "--no-json-mode",
        ],
    )

    assert result.exit_code == 0
    assert captured["json_mode"] is False


def test_cli_judge_writes_audit_jsonl_and_skips_existing_records(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Target Representation Paper",
                        "abstract": "representation learning",
                        "references": [{"reference_id": "r1", "title": "Representation Learning"}],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Target Systems Paper",
                        "abstract": "distributed systems",
                        "references": [{"reference_id": "r2", "title": "Distributed Systems"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = CliRunner().invoke(
        app,
        [
            "judge",
            "--input",
            str(input_path),
            "--audit-output",
            str(audit_path),
            "--top-k",
            "1",
            "--limit",
            "1",
        ],
    )
    second = CliRunner().invoke(
        app,
        [
            "judge",
            "--input",
            str(input_path),
            "--audit-output",
            str(audit_path),
            "--top-k",
            "1",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["target_paper_id"] for record in records] == ["p1", "p2"]
    assert len(records[0]["prompt_sha256"]) == 64
    assert records[0]["candidate_reference_ids"] == ["r1"]
    assert "Judge progress: total=1 done=0 skipped=0 to_run=1" in first.output
    assert "Judged 1/1: p1" in first.output
    assert "Judge progress: total=2 done=1 skipped=1 to_run=1" in second.output
    assert "Judged 1/1: p2" in second.output


def test_cli_judge_can_disable_progress_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target Representation Paper",
                "abstract": "representation learning",
                "references": [{"reference_id": "r1", "title": "Representation Learning"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "judge",
            "--input",
            str(input_path),
            "--audit-output",
            str(audit_path),
            "--top-k",
            "1",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert "Judge progress" not in result.output
    assert "Judged 1/1" not in result.output
    assert "Wrote 1 judge records" in result.output


def test_cli_judge_limit_applies_before_resume_skip(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "paper_id": f"p{index}",
                    "title": f"Paper {index}",
                    "references": [{"reference_id": f"r{index}", "title": "Reference"}],
                }
            )
            for index in range(1, 5)
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p1",
                "parse_error": None,
                "selected_references": [
                    {
                        "reference_id": "r1",
                        "rank": 1,
                        "confidence": 1.0,
                        "rationale": "existing",
                        "valid": True,
                    }
                ],
                "raw_output": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "judge",
            "--input",
            str(input_path),
            "--audit-output",
            str(audit_path),
            "--top-k",
            "1",
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["target_paper_id"] for record in records] == ["p1", "p2", "p3"]
    assert "Judge progress: total=3 done=1 skipped=1 to_run=2" in result.output
    assert "Wrote 2 judge records" in result.output


def test_cli_run_can_reuse_existing_judge_audit(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    output_dir = tmp_path / "run"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Known",
                        "authors": [{"name": "Author A", "affiliations": ["Institution X"]}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p1",
                "parse_error": None,
                "invalid_selection_count": 0,
                "selected_references": [
                    {
                        "reference_id": "r1",
                        "rank": 1,
                        "confidence": 0.9,
                        "rationale": "precomputed",
                        "valid": True,
                    }
                ],
                "raw_output": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--judge-audit",
            str(audit_path),
        ],
    )

    assert result.exit_code == 0
    records = (output_dir / "audit" / "judge_outputs.jsonl").read_text(encoding="utf-8")
    assert "precomputed" in records
    assert "Institution X" in (output_dir / "rankings" / "institutions.csv").read_text(
        encoding="utf-8"
    )


def test_cli_score_command_scores_from_existing_judge_audit(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    output_dir = tmp_path / "score"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Target",
                "references": [
                    {
                        "reference_id": "r1",
                        "title": "Known",
                        "authors": [{"name": "Author A", "affiliations": ["Institution X"]}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p1",
                "parse_error": None,
                "invalid_selection_count": 0,
                "selected_references": [
                    {
                        "reference_id": "r1",
                        "rank": 1,
                        "confidence": 0.9,
                        "rationale": "precomputed",
                        "valid": True,
                    }
                ],
                "raw_output": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--judge-audit",
            str(audit_path),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "rankings" / "papers.csv").exists()
    assert (output_dir / "reports" / "mvp_summary.md").exists()
    assert "precomputed" in (output_dir / "audit" / "judge_outputs.jsonl").read_text(
        encoding="utf-8"
    )


def test_cli_compare_benchmarks_writes_csv_and_markdown(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    output_dir = tmp_path / "comparison"
    for run_dir, paper_count, hit_rate in [(run_a, 32, 0.81), (run_b, 100, 0.86)]:
        reports = run_dir / "reports"
        reports.mkdir(parents=True)
        (reports / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "benchmark": {
                        "paper_count": paper_count,
                        "reference_count": paper_count * 10,
                        "judge_results": paper_count,
                        "credit_mode": "rank_weighted",
                    },
                    "judge_quality": {
                        "parse_success_rate": 1.0,
                        "invalid_selection_rate": 0.0,
                        "empty_selection_rate": 0.0,
                        "top_k_compliance_rate": 1.0,
                    },
                    "gold_quality": {
                        "labeled_targets": paper_count,
                        "hit_rate": hit_rate,
                        "selection_precision": 0.3,
                        "selection_recall": 0.7,
                        "mean_reciprocal_rank": 0.6,
                        "mean_ndcg": 0.7,
                    },
                    "decision": {
                        "usable_for_benchmark_report": True,
                        "needs_prompt_or_model_tuning": False,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    result = CliRunner().invoke(
        app,
        [
            "compare-benchmarks",
            str(run_b),
            str(run_a),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "benchmark_comparison.csv").exists()
    markdown = (output_dir / "benchmark_comparison.md").read_text(encoding="utf-8")
    assert "| run-a | 32 |" in markdown
    assert "| run-b | 100 |" in markdown
    assert "Wrote benchmark comparison" in result.output


def test_cli_compare_gold_targets_writes_target_level_report(tmp_path):
    base_dir = tmp_path / "base"
    candidate_dir = tmp_path / "candidate"
    output_dir = tmp_path / "comparison"
    (base_dir / "reports").mkdir(parents=True)
    (candidate_dir / "reports").mkdir(parents=True)
    (base_dir / "reports" / "gold_evaluation.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["g1"],
                        "selected_reference_ids": ["r1"],
                        "selected_positive_reference_ids": [],
                        "hit": False,
                        "reciprocal_rank": 0.0,
                        "ndcg": 0.0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "reports" / "gold_evaluation.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["g1"],
                        "selected_reference_ids": ["g1"],
                        "selected_positive_reference_ids": ["g1"],
                        "hit": True,
                        "reciprocal_rank": 1.0,
                        "ndcg": 1.0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "compare-gold-targets",
            str(base_dir),
            str(candidate_dir),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "target_comparison.json").exists()
    assert (output_dir / "target_changes.csv").exists()
    assert "# Gold Target Comparison" in (
        output_dir / "target_comparison.md"
    ).read_text(encoding="utf-8")
    assert "Wrote gold target comparison" in result.output


def test_cli_benchmark_status_prints_json_or_markdown(tmp_path):
    run_dir = tmp_path / "run"
    input_path = run_dir / "prescience.papers.jsonl"
    audit_path = run_dir / "audit" / "judge_outputs.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "paper_id": f"p{index}",
                    "title": f"Paper {index}",
                    "references": [{"reference_id": f"r{index}", "title": "Reference"}],
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p0",
                "parse_error": None,
                "error": None,
                "selected_references": [],
                "raw_output": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    json_result = CliRunner().invoke(app, ["benchmark-status", str(run_dir)])
    md_result = CliRunner().invoke(app, ["benchmark-status", str(run_dir), "--markdown"])

    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["input_papers"] == 2
    assert payload["audit_records"] == 1
    assert payload["remaining"] == 1
    assert md_result.exit_code == 0
    assert "# Benchmark Status" in md_result.output
    assert "| remaining | 1 |" in md_result.output


def test_cli_analyze_gold_errors_writes_report(tmp_path):
    output_dir = tmp_path / "analysis"
    gold_path = tmp_path / "gold_evaluation.json"
    gold_path.write_text(
        json.dumps(
            {
                "labeled_targets": 1,
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["r1"],
                        "selected_reference_ids": ["r2", "r3"],
                        "selected_positive_reference_ids": [],
                        "hit": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "analyze-gold-errors",
            "--gold-evaluation",
            str(gold_path),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "gold_error_analysis.json").exists()
    markdown = (output_dir / "gold_error_analysis.md").read_text(encoding="utf-8")
    assert "# Gold Error Analysis" in markdown
    assert "| hit@5 | 0.000 |" in markdown
    assert "Wrote gold error analysis" in result.output


def test_cli_analyze_gold_errors_uses_judge_audit_for_candidate_coverage(tmp_path):
    output_dir = tmp_path / "analysis"
    gold_path = tmp_path / "gold_evaluation.json"
    audit_path = tmp_path / "judge_outputs.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "labeled_targets": 1,
                "targets": [
                    {
                        "target_paper_id": "p1",
                        "positive_reference_ids": ["r1"],
                        "selected_reference_ids": ["r2", "r3"],
                        "selected_positive_reference_ids": [],
                        "hit": False,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "target_paper_id": "p1",
                "candidate_reference_ids": ["r1", "r2", "r3"],
                "selected_references": [
                    {
                        "reference_id": "r2",
                        "rank": 1,
                        "confidence": 0.9,
                        "rationale": "selected",
                        "valid": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "analyze-gold-errors",
            "--gold-evaluation",
            str(gold_path),
            "--judge-audit",
            str(audit_path),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    report = json.loads((output_dir / "gold_error_analysis.json").read_text(encoding="utf-8"))
    assert report["candidate_coverage"]["gold_candidate_recall"] == 1.0
    assert report["miss_decomposition"]["miss_gold_present_but_not_selected"] == 1
    markdown = (output_dir / "gold_error_analysis.md").read_text(encoding="utf-8")
    assert "| gold_candidate_recall | 1.000 |" in markdown


def test_cli_judge_records_per_paper_errors_without_stopping(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    audit_path = tmp_path / "judge_outputs.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Fails",
                        "references": [{"reference_id": "r1", "title": "Reference One"}],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Succeeds",
                        "references": [{"reference_id": "r2", "title": "Reference Two"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class SometimesFailingJudge(HeuristicJudge):
        def judge(self, paper, top_k=5):
            if paper.paper_id == "p1":
                raise RuntimeError("model server unavailable")
            return super().judge(paper, top_k=top_k)

    monkeypatch.setattr(cli, "HeuristicJudge", SometimesFailingJudge)

    result = CliRunner().invoke(
        app,
        [
            "judge",
            "--input",
            str(input_path),
            "--audit-output",
            str(audit_path),
            "--top-k",
            "1",
            "--concurrency",
            "2",
            "--max-retries",
            "1",
        ],
    )

    assert result.exit_code == 0
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["target_paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["error"] == "model server unavailable"
    assert records[1]["error"] is None


def test_judge_papers_yields_results_as_completed(monkeypatch):
    class SlowFirstJudge(HeuristicJudge):
        def judge(self, paper, top_k=5):
            if paper.paper_id == "p1":
                import time

                time.sleep(0.05)
            return super().judge(paper, top_k=top_k)

    papers = [
        cli.normalize_paper(
            cli.load_papers_jsonl.__globals__["Paper"].model_validate(
                {
                    "paper_id": "p1",
                    "title": "Slow",
                    "references": [{"reference_id": "r1", "title": "Reference One"}],
                }
            )
        ),
        cli.normalize_paper(
            cli.load_papers_jsonl.__globals__["Paper"].model_validate(
                {
                    "paper_id": "p2",
                    "title": "Fast",
                    "references": [{"reference_id": "r2", "title": "Reference Two"}],
                }
            )
        ),
    ]

    results = list(
        cli._judge_papers(
            papers,
            judge_impl=SlowFirstJudge(),
            top_k=1,
            concurrency=2,
            max_retries=0,
        )
    )

    assert [result.target_paper_id for result in results] == ["p2", "p1"]
