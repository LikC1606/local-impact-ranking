import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.icite_benchmark import (
    ICITE_API_URL,
    build_icite_current_metrics,
    enrich_papers_with_icite,
    evaluate_icite_current_benchmark,
)


class ICiteClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        return ICiteResponse(self.payloads)

    def close(self):
        self.closed = True


class ICiteResponse:
    def __init__(self, payloads):
        self.payloads = payloads
        self.status_code = 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        requested = str(self.payloads.get("requested_pmids", ""))
        rows = [
            row
            for row in self.payloads.get("data", [])
            if str(row.get("pmid")) in set(requested.split(","))
        ]
        return {"data": rows}


def test_build_icite_current_metrics_writes_public_benchmark_labels(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    labels_path = tmp_path / "icite_current_metrics.csv"
    summary_path = tmp_path / "icite_summary.json"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "PubMed paper",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "12345"}},
                    }
                ),
                json.dumps({"paper_id": "p2", "title": "No PMID", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = ICiteClient(
        {
            "requested_pmids": "12345",
            "data": [
                {
                    "pmid": 12345,
                    "doi": "10.1000/icite",
                    "title": "PubMed paper",
                    "year": 2020,
                    "journal": "Benchmark Journal",
                    "citation_count": 42,
                    "citations_per_year": 7.0,
                    "relative_citation_ratio": 2.5,
                    "nih_percentile": 91.0,
                    "apt": 0.75,
                    "human": 0.2,
                    "animal": 0.1,
                    "molecular_cellular": 0.7,
                    "field_citation_rate": 2.8,
                    "expected_citations_per_year": 2.0,
                    "is_research_article": True,
                    "provisional": False,
                    "citedByClinicalArticle": True,
                }
            ],
        }
    )

    summary = build_icite_current_metrics(
        input_path,
        labels_path,
        summary_output=summary_path,
        client=client,
    )

    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    assert summary["source"] == "nih_icite"
    assert summary["paper_count"] == 2
    assert summary["pmid_query_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 1
    assert rows[0]["paper_id"] == "p1"
    assert rows[0]["pmid"] == "12345"
    assert rows[0]["rcr"] == "2.5"
    assert rows[0]["icite_citation_count"] == "42"
    assert rows[0]["cited_by_clinical_article"] == "true"
    assert rows[1]["paper_id"] == "p2"
    assert rows[1]["pmid"] == ""
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1
    assert client.requests[0]["url"] == ICITE_API_URL


def test_build_icite_current_metrics_does_not_infer_local_numeric_paper_id_as_pmid(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    labels_path = tmp_path / "icite_current_metrics.csv"
    input_path.write_text(
        json.dumps({"paper_id": "12345", "title": "Local numeric ID", "references": []})
        + "\n",
        encoding="utf-8",
    )
    client = ICiteClient(
        {
            "requested_pmids": "12345",
            "data": [
                {
                    "pmid": 12345,
                    "relative_citation_ratio": 3.0,
                    "citation_count": 10,
                }
            ],
        }
    )

    summary = build_icite_current_metrics(input_path, labels_path, client=client)

    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    assert summary["pmid_query_count"] == 0
    assert summary["matched_count"] == 0
    assert rows[0]["pmid"] == ""
    assert rows[0]["rcr"] == ""


def test_enrich_papers_with_icite_adds_pubmed_counts_and_rcr(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.icite.jsonl"
    summary_path = tmp_path / "icite_enrichment_summary.json"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "PubMed paper",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "12345"}},
                    }
                ),
                json.dumps({"paper_id": "p2", "title": "No PMID", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = ICiteClient(
        {
            "requested_pmids": "12345",
            "data": [
                {
                    "pmid": 12345,
                    "doi": "10.1000/icite",
                    "citation_count": 42,
                    "citations_per_year": 7.0,
                    "relative_citation_ratio": 2.5,
                    "nih_percentile": 91.0,
                    "apt": 0.75,
                    "human": 0.2,
                    "animal": 0.1,
                    "molecular_cellular": 0.7,
                    "field_citation_rate": 2.8,
                    "expected_citations_per_year": 2.0,
                    "is_research_article": True,
                    "provisional": False,
                    "citedByClinicalArticle": True,
                }
            ],
        }
    )

    summary = enrich_papers_with_icite(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    enriched = records[0]
    metadata = enriched["metadata"]
    external_impact = metadata["external_impact"]
    assert summary["source"] == "nih_icite"
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 1
    assert enriched["citation_count"] == 42
    assert metadata["external_ids"]["pmid"] == "12345"
    assert metadata["external_ids"]["doi"] == "10.1000/icite"
    assert metadata["icite_enriched"] is True
    assert external_impact["pubmed_citation_count"] == 42
    assert external_impact["icite_citation_count"] == 42
    assert external_impact["relative_citation_ratio"] == 2.5
    assert external_impact["nih_percentile"] == 91.0
    assert external_impact["apt"] == 0.75
    assert external_impact["cited_by_clinical_article"] is True
    assert "external_impact" not in records[1].get("metadata", {})
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_icite_can_resume_without_refetching_existing_pmids(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.icite.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "111"}},
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs enrichment",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "222"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Already enriched",
                "references": [],
                "citation_count": 9,
                "metadata": {"icite_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = ICiteClient(
        {
            "requested_pmids": "222",
            "data": [{"pmid": 222, "citation_count": 12}],
        }
    )

    summary = enrich_papers_with_icite(
        input_path,
        output_path,
        client=client,
        resume=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert summary["resume"] is True
    assert summary["resumed_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 0
    assert client.requests[0]["params"]["pmids"] == "222"
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12


def test_evaluate_icite_current_benchmark_writes_credibility_report(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    scores_path = tmp_path / "impact_scores.csv"
    output = tmp_path / "icite-benchmark"
    paper_rows = []
    score_rows = []
    icite_rows = []
    for index in range(30):
        pmid = str(20000 + index)
        score = 100 - index
        paper_rows.append(
            {
                "paper_id": f"p{index:02d}",
                "title": f"Paper {index}",
                "references": [],
                "metadata": {"external_ids": {"pmid": pmid}},
            }
        )
        score_rows.append(
            {
                "paper_id": f"p{index:02d}",
                "final_impact_score": score,
                "score_80_true_20_effective": score,
                "score_70_true_30_effective": score,
                "score_50_true_50_effective": score,
                "score_30_true_70_effective": score,
                "true_citation_raw": score,
                "effective_citation_edge_count": 1,
            }
        )
        icite_rows.append(
            {
                "pmid": int(pmid),
                "relative_citation_ratio": score,
                "citations_per_year": score,
                "apt": score,
                "nih_percentile": score,
                "citation_count": score,
            }
        )
    papers_path.write_text(
        "\n".join(json.dumps(row) for row in paper_rows) + "\n",
        encoding="utf-8",
    )
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    client = ICiteClient(
        {
            "requested_pmids": ",".join(str(20000 + index) for index in range(30)),
            "data": icite_rows,
        }
    )

    report = evaluate_icite_current_benchmark(
        papers_path,
        scores_path,
        output,
        bootstrap_samples=20,
        permutation_samples=20,
        client=client,
    )

    assert report["summary"]["benchmark_source"] == "nih_icite"
    assert report["credibility"]["summary"]["credibility_status"] == "high_credibility"
    assert (output / "icite_current_metrics.csv").exists()
    assert (output / "icite_current_metrics_summary.json").exists()
    assert (output / "current_impact_credibility.md").exists()


def test_evaluate_icite_current_benchmark_handles_missing_rcr(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    scores_path = tmp_path / "impact_scores.csv"
    output = tmp_path / "icite-benchmark"
    paper_rows = []
    score_rows = []
    icite_rows = []
    for index in range(30):
        pmid = str(30000 + index)
        paper_rows.append(
            {
                "paper_id": pmid,
                "title": f"Very recent paper {index}",
                "references": [],
                "metadata": {"external_ids": {"pmid": pmid}},
            }
        )
        score_rows.append(
            {
                "paper_id": pmid,
                "final_impact_score": 100 - index,
                "score_80_true_20_effective": 100 - index,
                "score_70_true_30_effective": 100 - index,
                "score_50_true_50_effective": 100 - index,
                "score_30_true_70_effective": 100 - index,
                "true_citation_raw": 0,
                "effective_citation_edge_count": 1,
            }
        )
        icite_rows.append(
            {
                "pmid": int(pmid),
                "citation_count": 0,
                "citations_per_year": 0.0,
                "apt": 0.05,
            }
        )
    papers_path.write_text(
        "\n".join(json.dumps(row) for row in paper_rows) + "\n",
        encoding="utf-8",
    )
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    client = ICiteClient(
        {
            "requested_pmids": ",".join(str(30000 + index) for index in range(30)),
            "data": icite_rows,
        }
    )

    report = evaluate_icite_current_benchmark(
        papers_path,
        scores_path,
        output,
        bootstrap_samples=20,
        permutation_samples=20,
        client=client,
    )

    assert report["summary"]["matched_count"] == 30
    assert report["summary"]["primary_label_column"] == "citations_per_year"
    assert report["summary"]["primary_label_fallback_reason"]
    assert report["credibility"]["summary"]["credibility_status"] == "not_credible"


def test_cli_evaluate_icite_current_benchmark_writes_report(monkeypatch, tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    scores_path = tmp_path / "scores.csv"
    output = tmp_path / "out"
    papers_path.write_text("", encoding="utf-8")
    scores_path.write_text("paper_id,final_impact_score\n", encoding="utf-8")

    def fake_evaluate(*args, **kwargs):
        output_dir = args[2]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "current_impact_credibility.md").write_text(
            "# Current Impact Credibility\n",
            encoding="utf-8",
        )
        return {
            "summary": {"matched_count": 3},
            "credibility": {
                "summary": {
                    "credibility_status": "moderate_credibility",
                    "credibility_grade": "B",
                }
            },
        }

    monkeypatch.setattr(
        "local_impact_ranking.cli.evaluate_icite_current_benchmark",
        fake_evaluate,
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-icite-current-benchmark",
            "--papers",
            str(papers_path),
            "--scores",
            str(scores_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "credibility=moderate_credibility" in result.output


def test_cli_enrich_icite_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.icite.jsonl"
    summary_path = tmp_path / "icite_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["batch_size"] == 123
        assert kwargs["timeout"] == 9.0
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(json.dumps({"matched_count": 1}) + "\n", encoding="utf-8")
        return {"matched_count": 1, "unmatched_count": 0}

    monkeypatch.setattr(
        "local_impact_ranking.cli.enrich_papers_with_icite",
        fake_enrich,
    )

    result = CliRunner().invoke(
        app,
        [
            "enrich-icite",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--batch-size",
            "123",
            "--timeout",
            "9",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "NIH iCite-enriched" in result.output
    assert output_path.exists()
