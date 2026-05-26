from __future__ import annotations

from collections import defaultdict

from local_impact_ranking.models import CitationContext, Paper


def citation_contexts_by_reference(paper: Paper) -> dict[str, list[CitationContext]]:
    grouped: defaultdict[str, list[CitationContext]] = defaultdict(list)
    reference_ids = {reference.reference_id for reference in paper.references}
    for context in paper.citation_contexts:
        if context.reference_id in reference_ids:
            grouped[context.reference_id].append(context)
    return dict(grouped)


def context_coverage_report(papers: list[Paper]) -> dict[str, float | int]:
    paper_count = len(papers)
    papers_with_contexts = sum(1 for paper in papers if paper.citation_contexts)
    reference_count = sum(len(paper.references) for paper in papers)
    references_with_contexts = 0
    context_count = 0
    for paper in papers:
        grouped = citation_contexts_by_reference(paper)
        references_with_contexts += len(grouped)
        context_count += sum(len(contexts) for contexts in grouped.values())
    return {
        "papers": paper_count,
        "papers_with_contexts": papers_with_contexts,
        "paper_context_coverage_rate": papers_with_contexts / paper_count
        if paper_count
        else 0.0,
        "references": reference_count,
        "references_with_contexts": references_with_contexts,
        "reference_context_coverage_rate": references_with_contexts / reference_count
        if reference_count
        else 0.0,
        "citation_contexts": context_count,
    }
