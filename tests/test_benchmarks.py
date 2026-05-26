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
