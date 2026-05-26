from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import scoring_config

PROMPT_VERSION = "impact-reference-topk-v1"


def build_run_manifest(
    input_path: str | Path,
    *,
    top_k: int,
    limit: int | None = None,
    credit_mode: str,
    judge_name: str,
    base_url: str,
    model: str,
    temperature: float,
    repair_attempts: int,
    max_tokens: int | None = None,
    judge_audit: str | Path | None = None,
    normalized_papers: list[Paper] | None = None,
    serving_backend: str | None = None,
    serving_backend_version: str | None = None,
    model_path: str | Path | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict:
    source = Path(input_path)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "normalized_sha256": normalized_papers_sha256(normalized_papers)
            if normalized_papers is not None
            else None,
            "normalized_record_count": len(normalized_papers)
            if normalized_papers is not None
            else None,
        },
        "run": {
            "top_k": top_k,
            "limit": limit,
            "credit_mode": credit_mode,
            "judge_audit": str(judge_audit) if judge_audit is not None else None,
        },
        "judge": {
            "name": judge_name,
            "base_url": base_url if judge_name != "heuristic" else None,
            "model": model or None,
            "temperature": temperature,
            "repair_attempts": repair_attempts,
            "max_tokens": max_tokens,
            "serving_backend": serving_backend,
            "serving_backend_version": serving_backend_version,
            "model_path": str(model_path) if model_path is not None else None,
        },
        "prompt": {
            "version": PROMPT_VERSION,
        },
        "validation": validation_summary(validation_report),
        "scoring": scoring_config(credit_mode),
        "code": {
            "git_revision": git_revision(Path.cwd()),
            "git_dirty": git_dirty(Path.cwd()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "hardware": hardware_summary(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_papers_sha256(papers: list[Paper]) -> str:
    digest = hashlib.sha256()
    for paper in papers:
        line = json.dumps(paper.model_dump(mode="json"), sort_keys=True) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def validation_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    summary = report.get("summary", {})
    severity_counts = summary.get("severity_counts", {})
    return {
        "can_run": summary.get("can_run"),
        "issue_count": summary.get("issue_count", 0),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
    }


def hardware_summary() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "gpu_summary": gpu_summary(),
    }


def gpu_summary() -> list[dict[str, str]] | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return None
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[2],
                }
            )
    return gpus


def git_revision(cwd: Path) -> str | None:
    return _git_output(["git", "rev-parse", "HEAD"], cwd)


def git_dirty(cwd: Path) -> bool | None:
    output = _git_output(["git", "status", "--porcelain"], cwd)
    if output is None:
        return None
    return bool(output.strip())


def _git_output(command: list[str], cwd: Path) -> str | None:
    return _command_output(command, cwd=cwd)


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
