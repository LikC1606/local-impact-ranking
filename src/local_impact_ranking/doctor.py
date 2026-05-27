from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_PROJECT_FILES = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".github/workflows/tests.yml",
    "docs/input-schema.md",
    "docs/scoring-method.md",
    "docs/evaluation-method.md",
    "docs/publish-checklist.md",
]

LOCAL_ARTIFACT_DIRS = ["data", "models", "runs"]


def doctor_report(
    root: str | Path,
    *,
    command_names: tuple[str, ...] = ("git", "gh"),
) -> dict[str, Any]:
    base = Path(root)
    command_status = {name: shutil.which(name) is not None for name in command_names}
    git_status = _git_status(base)
    required = _required_files(base)
    publish = _publish_status(base)

    passed_checks: list[str] = []
    failed_checks: list[str] = []
    warning_checks: list[str] = []
    warnings: list[str] = []

    _record(
        passed_checks,
        failed_checks,
        "required_files_present",
        required["missing_count"] == 0,
    )
    _record(
        passed_checks,
        failed_checks,
        "local_artifacts_ignored",
        publish["ignored_local_artifacts"],
    )
    _record(
        passed_checks,
        failed_checks,
        "local_artifacts_not_tracked",
        not publish["tracked_local_artifacts"],
    )
    _record(
        passed_checks,
        failed_checks,
        "git_worktree_available",
        git_status["inside_worktree"],
    )
    if "gh" in command_status and not command_status["gh"]:
        warning_checks.append("optional_command_gh_available")
        warnings.append("Optional command not found: gh")

    return {
        "decision": _decision(failed_checks, warning_checks),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "commands": command_status,
        "git": git_status,
        "required_files": required,
        "publish": publish,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "warnings": warnings,
    }


def render_doctor_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    commands = report.get("commands", {})
    git = report.get("git", {})
    required = report.get("required_files", {})
    publish = report.get("publish", {})
    lines = [
        "# Project Doctor",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {_text(decision.get('status'))} |",
        f"| reason | {_text(decision.get('reason'))} |",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| python_version | {_text(report.get('python', {}).get('version'))} |",
        f"| command_git | {_yes_no(commands.get('git'))} |",
        f"| command_gh | {_yes_no(commands.get('gh'))} |",
        f"| inside_git_worktree | {_yes_no(git.get('inside_worktree'))} |",
        f"| branch | {_text(git.get('branch'))} |",
        f"| remote_origin | {_text(git.get('remote_origin'))} |",
        "",
        "## Publish Checks",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| missing_required_files | {required.get('missing_count', 0)} |",
        f"| ignored_local_artifacts | {_yes_no(publish.get('ignored_local_artifacts'))} |",
        f"| tracked_local_artifacts | {len(publish.get('tracked_local_artifacts', []))} |",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _required_files(base: Path) -> dict[str, Any]:
    files = [
        {"path": relative_path, "exists": (base / relative_path).exists()}
        for relative_path in REQUIRED_PROJECT_FILES
    ]
    return {
        "missing_count": sum(1 for item in files if not item["exists"]),
        "files": files,
    }


def _publish_status(base: Path) -> dict[str, Any]:
    ignored_paths = [f"{directory}/.publish-check" for directory in LOCAL_ARTIFACT_DIRS]
    ignored = _run_git(base, ["check-ignore", *ignored_paths]).returncode == 0
    tracked = _run_git(base, ["ls-files", *LOCAL_ARTIFACT_DIRS])
    tracked_paths = [
        line for line in tracked.stdout.splitlines() if line.strip()
    ] if tracked.returncode == 0 else []
    return {
        "ignored_local_artifacts": ignored,
        "tracked_local_artifacts": tracked_paths,
    }


def _git_status(base: Path) -> dict[str, Any]:
    inside = _run_git(base, ["rev-parse", "--is-inside-work-tree"])
    branch = _run_git(base, ["branch", "--show-current"])
    remote = _run_git(base, ["remote", "get-url", "origin"])
    return {
        "inside_worktree": inside.returncode == 0 and inside.stdout.strip() == "true",
        "branch": branch.stdout.strip() if branch.returncode == 0 else "",
        "remote_origin": remote.stdout.strip() if remote.returncode == 0 else "",
    }


def _run_git(base: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=base,
        capture_output=True,
        text=True,
        check=False,
    )


def _record(
    passed_checks: list[str],
    failed_checks: list[str],
    name: str,
    condition: bool,
) -> None:
    if condition:
        passed_checks.append(name)
    else:
        failed_checks.append(name)


def _decision(
    failed_checks: list[str],
    warning_checks: list[str],
) -> dict[str, Any]:
    if failed_checks:
        return {
            "status": "blocked",
            "reason": "Required project or publish checks failed.",
        }
    if warning_checks:
        return {
            "status": "warning",
            "reason": "Core checks passed, but optional tooling is missing.",
        }
    return {"status": "ok", "reason": "Project checks passed."}


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
