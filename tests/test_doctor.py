import subprocess

from local_impact_ranking.doctor import doctor_report, render_doctor_markdown


def _write(path, content="placeholder\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_doctor_report_flags_missing_required_files_and_unignored_artifacts(tmp_path):
    report = doctor_report(tmp_path, command_names=())

    assert report["decision"]["status"] == "blocked"
    assert report["required_files"]["missing_count"] > 0
    assert report["publish"]["ignored_local_artifacts"] is False
    assert "required_files_present" in report["failed_checks"]
    assert "local_artifacts_ignored" in report["failed_checks"]


def test_doctor_report_accepts_publish_ready_layout_without_optional_gh(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    for relative_path in [
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".github/workflows/tests.yml",
        "docs/input-schema.md",
        "docs/scoring-method.md",
        "docs/evaluation-method.md",
        "docs/publish-checklist.md",
    ]:
        _write(tmp_path / relative_path)
    (tmp_path / ".gitignore").write_text(
        "\n".join(["data/", "models/", "runs/"]) + "\n",
        encoding="utf-8",
    )

    report = doctor_report(tmp_path, command_names=("git",))

    assert "required_files_present" in report["passed_checks"]
    assert "local_artifacts_ignored" in report["passed_checks"]
    assert "local_artifacts_not_tracked" in report["passed_checks"]
    assert report["required_files"]["missing_count"] == 0
    assert report["publish"]["tracked_local_artifacts"] == []
    assert report["decision"]["status"] in {"ok", "warning"}


def test_render_doctor_markdown_includes_publish_checks():
    markdown = render_doctor_markdown(
        {
            "decision": {"status": "warning", "reason": "Optional tool missing."},
            "python": {"version": "3.11.0"},
            "commands": {"git": True, "gh": False},
            "git": {"inside_worktree": True, "branch": "master", "remote_origin": "origin"},
            "required_files": {"missing_count": 0},
            "publish": {
                "ignored_local_artifacts": True,
                "tracked_local_artifacts": [],
            },
            "passed_checks": ["required_files_present"],
            "failed_checks": [],
            "warning_checks": ["optional_command_gh_available"],
            "warnings": ["Optional command not found: gh"],
        }
    )

    assert "# Project Doctor" in markdown
    assert "| status | warning |" in markdown
    assert "| command_gh | no |" in markdown
    assert "| ignored_local_artifacts | yes |" in markdown
    assert "- Optional command not found: gh" in markdown
