from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    name: str
    purpose: str
    pipeline_layers: tuple[str, ...]
    preferred_input: str
    notes: str
    source_url: str
    paper_url: str
    implementation_status: str
    recommended_order: int
    evaluation_metrics: tuple[str, ...]
    evaluation_role: str
    local_command: str
    transfer_risk: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "prescience": BenchmarkSpec(
        key="prescience",
        name="PreScience",
        purpose="Prior work selection and impact prediction on AI papers.",
        pipeline_layers=("judge", "contexts", "baselines", "proxy_evaluation", "finetuning"),
        preferred_input="AI-paper citation graph with prior-work references and citation spans.",
        notes="Best first benchmark when no private dataset is available.",
        source_url="https://huggingface.co/datasets/allenai/prescience",
        paper_url="https://arxiv.org/abs/2602.20459",
        implementation_status="ready",
        recommended_order=1,
        evaluation_metrics=("hit@k", "mrr", "ndcg@k", "recall@k"),
        evaluation_role=(
            "Closest executable gold benchmark for whether the judge selects "
            "known important prior work among candidate references."
        ),
        local_command="scripts/run_prescience_smoke.sh runs/prescience-smoke",
        transfer_risk=(
            "AI-paper prior-work selection is not identical to global impact; "
            "report it as gold reference-selection evidence."
        ),
    ),
    "scidoca2025": BenchmarkSpec(
        key="scidoca2025",
        name="SCIDOCA 2025",
        purpose="Citation discovery, citation sentence prediction, and masked citation prediction.",
        pipeline_layers=("judge", "contexts", "proxy_evaluation", "finetuning"),
        preferred_input="S2ORC-derived citation spans and masked-citation samples.",
        notes="Good for testing citation-context-aware retrieval and selection.",
        source_url="https://github.com/SCIDOCA-2025/SCIDOCA-2025",
        paper_url="",
        implementation_status="planned",
        recommended_order=3,
        evaluation_metrics=("recall@k", "mrr", "masked_citation_accuracy"),
        evaluation_role=(
            "Citation-discovery stress test for context-aware candidate selection "
            "and missing-reference recovery."
        ),
        local_command="planned: add converter after selecting a stable official release",
        transfer_risk=(
            "Citation discovery measures likely-citation recovery, not direct "
            "impact score calibration."
        ),
    ),
    "scicite": BenchmarkSpec(
        key="scicite",
        name="SciCite / ACL-ARC / MultiCite",
        purpose="Citation intent classification and meaning-aware citation labeling.",
        pipeline_layers=("contexts", "proxy_evaluation", "finetuning"),
        preferred_input="Citation sentences with background/method/result labels.",
        notes="Useful auxiliary benchmark for intent and support signals.",
        source_url="https://github.com/allenai/scicite",
        paper_url="https://aclanthology.org/N19-1361/",
        implementation_status="ready",
        recommended_order=2,
        evaluation_metrics=("context_support_rate", "method_context_rate", "label_coverage"),
        evaluation_role=(
            "Auxiliary proxy check that influential selections are supported by "
            "meaningful citation contexts rather than bare citation counts."
        ),
        local_command="scripts/download_scicite.sh data/scicite && scripts/run_benchmark_smoke.sh runs/benchmark-smoke",
        transfer_risk=(
            "Citation intent labels are proxy evidence; they do not define true "
            "paper impact."
        ),
    ),
    "sciimpact": BenchmarkSpec(
        key="sciimpact",
        name="SciImpact",
        purpose="Broader scientific impact prediction and pairwise comparison supervision.",
        pipeline_layers=("judge", "proxy_evaluation", "finetuning"),
        preferred_input="Pairwise impact examples across citation and non-citation impact dimensions.",
        notes="Broader than citation-only influence; use as a later-stage benchmark.",
        source_url="https://flypig23.github.io/sciimpact-homepage/",
        paper_url="https://arxiv.org/abs/2604.17141",
        implementation_status="adapter_ready",
        recommended_order=4,
        evaluation_metrics=(
            "pairwise_accuracy",
            "non_tie_pairwise_accuracy",
            "coverage_rate",
            "tie_rate",
            "field_accuracy",
        ),
        evaluation_role=(
            "Direct pairwise scientific-impact benchmark for checking whether "
            "the scoring model orders higher-impact papers above lower-impact papers."
        ),
        local_command=(
            "scripts/run_sciimpact_smoke.sh runs/sciimpact-smoke; "
            "for full data, run benchmarks convert-sciimpact then evaluate-pairwise-impact"
        ),
        transfer_risk=(
            "Pairwise labels transfer well to ranking, but their impact dimensions "
            "must be reported separately from dataset-local score magnitude."
        ),
    ),
    "bib_bases": BenchmarkSpec(
        key="bib_bases",
        name="BIP! DB / OpenCitations",
        purpose="Traditional bibliometric baseline and citation-graph comparison.",
        pipeline_layers=("baselines", "proxy_evaluation", "finetuning"),
        preferred_input="Citation graph and metadata with paper IDs or DOI links.",
        notes="Best for sanity checks and classical baseline comparison.",
        source_url="https://zenodo.org/records/10804822",
        paper_url="https://arxiv.org/abs/2101.12001",
        implementation_status="external_baseline",
        recommended_order=5,
        evaluation_metrics=("spearman", "kendall_tau", "top_n_overlap", "field_year_normalized_correlation"),
        evaluation_role=(
            "External bibliometric calibration against citation count, PageRank, "
            "and field/year-normalized impact indicators."
        ),
        local_command="planned: ingest DOI/citation graph export, then run baseline comparison",
        transfer_risk=(
            "Citation graph baselines are popularity and age sensitive; use for "
            "calibration and anomaly detection, not as the sole target."
        ),
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return sorted(BENCHMARKS.values(), key=lambda spec: spec.recommended_order)


def get_benchmark(name: str) -> BenchmarkSpec:
    key = name.lower().replace("-", "").replace("_", "")
    lookup = {
        "prescience": "prescience",
        "scidoca2025": "scidoca2025",
        "scicite": "scicite",
        "aclarc": "scicite",
        "multicite": "scicite",
        "sciimpact": "sciimpact",
        "bibbases": "bib_bases",
        "bipdb": "bib_bases",
        "opencitations": "bib_bases",
    }
    try:
        return BENCHMARKS[lookup[key]]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark: {name}") from exc


def benchmark_plan() -> dict[str, Any]:
    specs = list_benchmarks()
    return {
        "recommended_order": [spec.key for spec in specs],
        "ready_now": [
            spec.key
            for spec in specs
            if spec.implementation_status in {"ready", "adapter_ready"}
        ],
        "primary_benchmark": "prescience",
        "direct_impact_benchmark": "sciimpact",
        "baseline_anchor": "bib_bases",
        "benchmarks": [_spec_to_plan_item(spec) for spec in specs],
    }


def render_benchmark_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Evaluation Plan",
        "",
        "## Recommended Order",
        "",
        "| Order | Benchmark | Status | Metrics | Role |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for item in plan.get("benchmarks", []):
        lines.append(
            "| "
            f"{item['order']} | "
            f"{item['name']} | "
            f"{item['status']} | "
            f"{', '.join(item['metrics'])} | "
            f"{item['evaluation_role']} |"
        )
    lines.extend(
        [
            "",
            "## Ready Now",
            "",
            ", ".join(plan.get("ready_now", [])) or "none",
            "",
            "## Commands",
            "",
        ]
    )
    for item in plan.get("benchmarks", []):
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Source: {item['source_url']}",
                f"- Paper: {item['paper_url'] or 'not recorded'}",
                f"- Local command: `{item['local_command']}`",
                f"- Transfer risk: {item['transfer_risk']}",
                "",
            ]
        )
    return "\n".join(lines)


def _spec_to_plan_item(spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "order": spec.recommended_order,
        "name": spec.name,
        "status": spec.implementation_status,
        "purpose": spec.purpose,
        "metrics": list(spec.evaluation_metrics),
        "pipeline_layers": list(spec.pipeline_layers),
        "preferred_input": spec.preferred_input,
        "evaluation_role": spec.evaluation_role,
        "source_url": spec.source_url,
        "paper_url": spec.paper_url,
        "local_command": spec.local_command,
        "transfer_risk": spec.transfer_risk,
        "notes": spec.notes,
    }
