from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    purpose: str
    pipeline_layers: tuple[str, ...]
    preferred_input: str
    notes: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "prescience": BenchmarkSpec(
        name="PreScience",
        purpose="Prior work selection and impact prediction on AI papers.",
        pipeline_layers=("judge", "contexts", "baselines", "proxy_evaluation", "finetuning"),
        preferred_input="AI-paper citation graph with prior-work references and citation spans.",
        notes="Best first benchmark when no private dataset is available.",
    ),
    "scidoca2025": BenchmarkSpec(
        name="SCIDOCA 2025",
        purpose="Citation discovery, citation sentence prediction, and masked citation prediction.",
        pipeline_layers=("judge", "contexts", "proxy_evaluation", "finetuning"),
        preferred_input="S2ORC-derived citation spans and masked-citation samples.",
        notes="Good for testing citation-context-aware retrieval and selection.",
    ),
    "scicite": BenchmarkSpec(
        name="SciCite / ACL-ARC / MultiCite",
        purpose="Citation intent classification and meaning-aware citation labeling.",
        pipeline_layers=("contexts", "proxy_evaluation", "finetuning"),
        preferred_input="Citation sentences with background/method/result labels.",
        notes="Useful auxiliary benchmark for intent and support signals.",
    ),
    "sciimpact": BenchmarkSpec(
        name="SciImpact",
        purpose="Broader scientific impact prediction and pairwise comparison supervision.",
        pipeline_layers=("judge", "proxy_evaluation", "finetuning"),
        preferred_input="Pairwise impact examples across citation and non-citation impact dimensions.",
        notes="Broader than citation-only influence; use as a later-stage benchmark.",
    ),
    "bib_bases": BenchmarkSpec(
        name="BIP! DB / OpenCitations",
        purpose="Traditional bibliometric baseline and citation-graph comparison.",
        pipeline_layers=("baselines", "proxy_evaluation", "finetuning"),
        preferred_input="Citation graph and metadata with paper IDs or DOI links.",
        notes="Best for sanity checks and classical baseline comparison.",
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return list(BENCHMARKS.values())


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
