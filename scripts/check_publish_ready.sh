#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

test -f README.md
test -f pyproject.toml
test -f .gitignore
test -f .github/workflows/tests.yml
test -f docs/input-schema.md
test -f docs/scoring-method.md
test -f docs/evaluation-method.md
test -f docs/publish-checklist.md

git check-ignore data models runs >/dev/null

if git ls-files data models runs | grep -q .; then
  echo "Refusing to publish: data/, models/, or runs/ are tracked by git." >&2
  git ls-files data models runs >&2
  exit 1
fi

grep -q "report-bundle" README.md
grep -q "dataset-local" README.md
grep -q "data/" docs/publish-checklist.md
grep -q "models/" docs/publish-checklist.md
grep -q "runs/" docs/publish-checklist.md

PYTHONNOUSERSITE=1 PYTHONPATH=src "${PYTHON_BIN}" -m pytest -q

echo "Publish readiness checks passed."
