import csv
import json

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.opcitance_benchmark import (
    build_opcitance_icite_current_benchmark,
)


class ICiteClient:
    def __init__(self, rows):
        self.rows = {str(row["pmid"]): row for row in rows}
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        requested = str((params or {}).get("pmids", "")).split(",")
        return ICiteResponse([self.rows[pmid] for pmid in requested if pmid in self.rows])

    def close(self):
        self.closed = True


class ICiteResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self.rows}


def write_opcitance_fixture(path):
    fieldnames = [
        "pmcid",
        "pmid",
        "location",
        "IMRaD",
        "sentence_id",
        "total_sentences",
        "intxt_id",
        "intxt_pmid",
        "intxt_pmid_source",
        "intxt_mark",
        "best_id",
        "best_source",
        "best_id_diff",
        "citation",
        "progression",
    ]
    rows = [
        {
            "pmcid": "PMC1",
            "pmid": "2002",
            "location": "body",
            "IMRaD": "M",
            "sentence_id": "1",
            "total_sentences": "10",
            "intxt_id": "PMC1_sref1",
            "intxt_pmid": "1001",
            "intxt_pmid_source": "pmc",
            "intxt_mark": "Prior assay>|sref1|",
            "best_id": "1001",
            "best_source": "pmc",
            "best_id_diff": "INSERT_PMID",
            "citation": "We used the biomarker assay protocol described by the prior study.",
            "progression": "10.0",
        },
        {
            "pmcid": "PMC2",
            "pmid": "2003",
            "location": "body",
            "IMRaD": "I",
            "sentence_id": "2",
            "total_sentences": "12",
            "intxt_id": "PMC2_sref1",
            "intxt_pmid": "1001",
            "intxt_pmid_source": "pmc",
            "intxt_mark": "Prior assay>|sref1|",
            "best_id": "1001",
            "best_source": "pmc",
            "best_id_diff": "INSERT_PMID",
            "citation": "This disease biomarker has been investigated in previous studies.",
            "progression": "16.0",
        },
        {
            "pmcid": "PMC3",
            "pmid": "2004",
            "location": "body",
            "IMRaD": "R",
            "sentence_id": "3",
            "total_sentences": "9",
            "intxt_id": "PMC3_sref1",
            "intxt_pmid": "2002",
            "intxt_pmid_source": "pmc",
            "intxt_mark": "Trial>|sref1|",
            "best_id": "2002",
            "best_source": "pmc",
            "best_id_diff": "INSERT_PMID",
            "citation": "The clinical trial supports using the assay in patient selection.",
            "progression": "33.0",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_build_opcitance_icite_current_benchmark_outputs_scores_and_labels(tmp_path):
    opcitance = tmp_path / "opcitance.tsv"
    output = tmp_path / "opcitance-benchmark"
    write_opcitance_fixture(opcitance)
    client = ICiteClient(
        [
            {
                "pmid": 1001,
                "title": "Reusable biomarker assay",
                "year": 2018,
                "journal": "Biomedicine",
                "citation_count": 100,
                "relative_citation_ratio": 10.0,
                "citations_per_year": 12.5,
                "nih_percentile": 95.0,
                "apt": 0.8,
            },
            {
                "pmid": 2002,
                "title": "Trial using assay",
                "year": 2020,
                "journal": "Biomedicine",
                "citation_count": 40,
                "relative_citation_ratio": 4.0,
                "citations_per_year": 5.0,
                "nih_percentile": 80.0,
                "apt": 0.6,
            },
            {
                "pmid": 2003,
                "title": "Background review",
                "year": 2020,
                "journal": "Biomedicine",
                "citation_count": 5,
                "relative_citation_ratio": 0.5,
                "citations_per_year": 0.8,
                "nih_percentile": 20.0,
                "apt": 0.2,
            },
            {
                "pmid": 2004,
                "title": "Clinical follow-up",
                "year": 2021,
                "journal": "Biomedicine",
                "citation_count": 10,
                "relative_citation_ratio": 1.0,
                "citations_per_year": 1.6,
                "nih_percentile": 40.0,
                "apt": 0.3,
            },
        ]
    )

    summary = build_opcitance_icite_current_benchmark(
        opcitance,
        output,
        max_papers=4,
        candidate_limit=4,
        bootstrap_samples=10,
        permutation_samples=10,
        client=client,
    )

    papers = [
        json.loads(line)
        for line in (output / "opcitance_icite_papers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    score_rows = list(
        csv.DictReader((output / "impact_scores.csv").open(encoding="utf-8"))
    )
    label_rows = list(
        csv.DictReader((output / "icite_current_metrics.csv").open(encoding="utf-8"))
    )
    assay = next(row for row in score_rows if row["paper_id"] == "1001")
    trial = next(row for row in score_rows if row["paper_id"] == "2002")

    assert summary["paper_count"] == 4
    assert summary["icite_matched_paper_count"] == 4
    assert summary["internal_reference_edges"] == 3
    assert summary["citation_context_count"] == 3
    assert any(paper["references"] for paper in papers if paper["paper_id"] == "2002")
    assert len(label_rows) == 4
    assert assay["true_citation_source"] == "pubmed_icite"
    assert float(assay["effective_citation_edge_count"]) == 2
    assert float(trial["final_impact_score"]) == float(
        trial["score_95_true_05_effective"]
    )
    assert float(trial["final_impact_score"]) != float(
        trial["score_50_true_50_effective"]
    )
    assert (output / "current_impact_credibility" / "current_impact_credibility.md").exists()
    assert (output / "opcitance_icite_benchmark_summary.md").exists()


def test_cli_build_opcitance_icite_current_benchmark_invokes_builder(
    monkeypatch,
    tmp_path,
):
    opcitance = tmp_path / "opcitance.tsv"
    output = tmp_path / "out"
    opcitance.write_text("pmcid\tpmid\n", encoding="utf-8")
    calls = {}

    def fake_builder(input_path, output_dir, **kwargs):
        calls["input_path"] = input_path
        calls["output_dir"] = output_dir
        calls["kwargs"] = kwargs
        return {
            "paper_count": 12,
            "internal_reference_edges": 34,
            "credibility_status": "moderate_credibility",
            "credibility_grade": "B",
        }

    monkeypatch.setattr(cli, "build_opcitance_icite_current_benchmark", fake_builder)

    result = CliRunner().invoke(
        app,
        [
            "build-opcitance-icite-current-benchmark",
            "--input",
            str(opcitance),
            "--output",
            str(output),
            "--max-papers",
            "12",
            "--candidate-limit",
            "100",
            "--bootstrap-samples",
            "10",
            "--permutation-samples",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert "credibility=moderate_credibility" in result.output
    assert calls["input_path"] == opcitance
    assert calls["output_dir"] == output
    assert calls["kwargs"]["max_papers"] == 12
    assert calls["kwargs"]["candidate_limit"] == 100
