import json

import pytest
from typer.testing import CliRunner

from local_impact_ranking.benchmarks import BENCHMARKS, get_benchmark, list_benchmarks
from local_impact_ranking.cli import app


def test_list_benchmarks_returns_named_specs():
    benchmarks = list_benchmarks()

    assert {benchmark.name for benchmark in benchmarks} >= {
        "NIH iCite / RCR Current Impact",
        "OpCitance + NIH iCite Current Impact",
        "PreScience",
        "SCIDOCA 2025",
        "SciCite / ACL-ARC / MultiCite",
        "SciImpact",
        "BIP! DB / OpenCitations",
    }


def test_get_benchmark_accepts_common_aliases():
    assert get_benchmark("RCR").name == "NIH iCite / RCR Current Impact"
    assert get_benchmark("OpCitance").name == "OpCitance + NIH iCite Current Impact"
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
    icite = get_benchmark("icite")
    opcitance = get_benchmark("opcitance")
    prescience = get_benchmark("prescience")
    sciimpact = get_benchmark("sciimpact")

    assert icite.recommended_order == 1
    assert icite.implementation_status == "ready"
    assert "spearman" in icite.evaluation_metrics
    assert "https://icite.od.nih.gov/" in icite.source_url
    assert opcitance.recommended_order == 2
    assert opcitance.implementation_status == "ready"
    assert "citation_context_count" in opcitance.evaluation_metrics
    assert "s41597-023-02134-x" in opcitance.paper_url
    assert prescience.recommended_order == 3
    assert prescience.implementation_status == "ready"
    assert "hit@k" in prescience.evaluation_metrics
    assert "https://huggingface.co/datasets/allenai/prescience" in prescience.source_url
    assert sciimpact.implementation_status == "adapter_ready"
    assert "pairwise_accuracy" in sciimpact.evaluation_metrics


def test_cli_benchmarks_plan_prints_ordered_json():
    result = CliRunner().invoke(app, ["benchmarks", "plan"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["recommended_order"][0] == "icite_current"
    assert payload["recommended_order"][1] == "opcitance_icite"
    assert payload["ready_now"] == [
        "icite_current",
        "opcitance_icite",
        "prescience",
        "scicite",
        "sciimpact",
    ]
    assert payload["primary_benchmark"] == "icite_current"
    assert payload["benchmarks"][0]["name"] == "NIH iCite / RCR Current Impact"
    assert payload["benchmarks"][0]["metrics"] == [
        "spearman",
        "kendall_tau",
        "pairwise_accuracy",
        "ndcg@k",
        "coverage_rate",
        "weight_sensitivity",
    ]


def test_cli_benchmarks_plan_prints_markdown():
    result = CliRunner().invoke(app, ["benchmarks", "plan", "--markdown"])

    assert result.exit_code == 0
    assert result.output.startswith("# Benchmark Evaluation Plan\n")
    assert "| 1 | NIH iCite / RCR Current Impact | ready |" in result.output
    assert "| 2 | OpCitance + NIH iCite Current Impact | ready |" in result.output
    assert "build-opcitance-icite-current-benchmark" in result.output
    assert "pairwise_accuracy" in result.output
