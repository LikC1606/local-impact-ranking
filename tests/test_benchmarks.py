import json

import pytest
from typer.testing import CliRunner

from local_impact_ranking.benchmarks import BENCHMARKS, get_benchmark, list_benchmarks
from local_impact_ranking.cli import app


def test_list_benchmarks_returns_named_specs():
    benchmarks = list_benchmarks()

    assert {benchmark.name for benchmark in benchmarks} >= {
        "PreScience",
        "SCIDOCA 2025",
        "SciCite / ACL-ARC / MultiCite",
        "SciImpact",
        "BIP! DB / OpenCitations",
    }


def test_get_benchmark_accepts_common_aliases():
    assert get_benchmark("PreScience").name == "PreScience"
    assert get_benchmark("scidoca-2025").name == "SCIDOCA 2025"
    assert get_benchmark("ACL-ARC").name == "SciCite / ACL-ARC / MultiCite"
    assert get_benchmark("opencitations").name == "BIP! DB / OpenCitations"


def test_get_benchmark_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_benchmark("missing")


def test_cli_benchmarks_list_prints_registry():
    result = CliRunner().invoke(app, ["benchmarks", "list"])

    assert result.exit_code == 0
    assert "PreScience" in result.output
    assert "SCIDOCA 2025" in result.output


def test_cli_benchmarks_show_prints_json():
    result = CliRunner().invoke(app, ["benchmarks", "show", "SciCite"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "SciCite / ACL-ARC / MultiCite"
    assert "finetuning" in payload["pipeline_layers"]


def test_benchmark_specs_include_evaluation_metadata():
    prescience = get_benchmark("prescience")
    sciimpact = get_benchmark("sciimpact")

    assert prescience.recommended_order == 1
    assert prescience.implementation_status == "ready"
    assert "hit@k" in prescience.evaluation_metrics
    assert "https://huggingface.co/datasets/allenai/prescience" in prescience.source_url
    assert sciimpact.implementation_status == "planned"
    assert "pairwise_accuracy" in sciimpact.evaluation_metrics


def test_cli_benchmarks_plan_prints_ordered_json():
    result = CliRunner().invoke(app, ["benchmarks", "plan"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["recommended_order"][0] == "prescience"
    assert payload["ready_now"] == ["prescience", "scicite"]
    assert payload["benchmarks"][0]["name"] == "PreScience"
    assert payload["benchmarks"][0]["metrics"] == ["hit@k", "mrr", "ndcg@k", "recall@k"]


def test_cli_benchmarks_plan_prints_markdown():
    result = CliRunner().invoke(app, ["benchmarks", "plan", "--markdown"])

    assert result.exit_code == 0
    assert result.output.startswith("# Benchmark Evaluation Plan\n")
    assert "| 1 | PreScience | ready |" in result.output
    assert "pairwise_accuracy" in result.output
