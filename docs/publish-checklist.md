# Publish Checklist

Use this before uploading the repository to GitHub or preparing a release.

## Include

- Source code under `src/`.
- Tests under `tests/`.
- Examples under `examples/`.
- Scripts under `scripts/`.
- Documentation under `docs/`.
- Small reusable configuration profiles under `configs/`, including `configs/calibration/`.
- GitHub Actions workflow under `.github/workflows/tests.yml`.

## Exclude

These paths are local artifacts and must not be committed:

- `data/`: local datasets and downloaded benchmark shards.
- `models/`: local model weights and adapters.
- `runs/`: generated experiment outputs, report bundles, audits, and rankings.

They are intentionally ignored by `.gitignore`.

## Verify

Run:

```bash
PYTHON_BIN=/root/miniconda3/envs/air/bin/python3.11 scripts/check_publish_ready.sh
```

The script checks required docs, GitHub Actions, ignore rules, tracked local-output paths, README references to `report-bundle`, and then runs:

```bash
pytest -q
```

## Report Boundary

Keep the public claim conservative:

- The project computes dataset-local paper, author, institution, field, and year influence.
- It does not claim global scientific impact without external graph calibration.
- Benchmark evidence such as PreScience evaluates reference-selection quality, not universal impact truth.
- Bundled calibration profiles under `configs/calibration/` are reusable reporting priors. They are not replacements for domain-specific validation.
- `report-bundle` creates shareable run artifacts, but generated bundles remain under `runs/` by default and should not be committed unless intentionally published separately.
