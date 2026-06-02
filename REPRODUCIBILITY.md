# Reproducibility

This repository is designed so another user can clone it and reproduce the public, no-GPU deliverable without private data, model weights, or generated run outputs.

## What Is Reproducible From The Repository Alone

The built-in reproduction path uses only tracked source code, tests, configs, and small examples:

- package installation
- full pytest suite
- publish-safety checks
- no-GPU quickstart
- no-GPU impact demo with replayed pairwise judgments
- user-impact workflow on the bundled example dataset
- SciImpact fixture smoke test
- impact backtest/validity fixture
- report bundle generation
- impact claim certificate generation

It does not require a GPU or local model server.

## Fresh Clone Reproduction

```bash
git clone https://github.com/LikC1606/local-impact-ranking.git
cd local-impact-ranking

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

scripts/reproduce_no_gpu.sh runs/reproduce-no-gpu
```

Expected final line:

```text
No-GPU reproduction complete: runs/reproduce-no-gpu
```

Expected status file:

```bash
cat runs/reproduce-no-gpu/reproduce_status.json
```

The most important reproduced artifacts are:

- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/impact_scores.csv`
- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/model_impact_report.md`
- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/impact_score_cards.md`
- `runs/reproduce-no-gpu/deliverable-check/impact-demo/model-impact/impact_claim_certificate.md`
- `runs/reproduce-no-gpu/deliverable-check/user-impact/user_impact_summary.md`
- `runs/reproduce-no-gpu/deliverable-check/user-impact/model-impact/report_bundle/`

## Faster Local Checks

If the package is already installed:

```bash
SKIP_INSTALL=1 scripts/reproduce_no_gpu.sh runs/reproduce-no-gpu
```

Run only the core tests and publish-safety checks:

```bash
scripts/check_publish_ready.sh
```

Run only the public impact demo:

```bash
scripts/run_impact_demo.sh runs/impact-demo
```

## Optional Local Model Reproduction

The repository does not include model weights. To reproduce the model-backed path, run an OpenAI-compatible local model server separately, then execute:

```bash
CHECK_MODEL=1 \
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_deliverable_check.sh runs/deliverable-check-model
```

For a user dataset:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=local-impact-judge-source-task-dpo \
  scripts/run_user_impact.sh papers.external.jsonl runs/my-impact
```

Use `PAIRWISE_JUDGMENTS=...` to replay saved pairwise judgments without calling a model server.

## Optional Training Reproduction

Training is not part of the no-GPU public reproduction path because it requires external model weights and GPU dependencies. The tracked scripts are:

- `scripts/train_sciimpact_qwen32b_long_context_sft.sh`
- `scripts/train_sciimpact_qwen32b_long_context_dpo.sh`
- `scripts/train_sciimpact_weak_slice_dpo.sh`

Before training, run:

```bash
PYTHONPATH=src python -m local_impact_ranking.cli finetuning-preflight runs/my-training-run
```

## Data And Artifact Policy

The repository intentionally excludes local datasets, generated run directories, and model weights:

- `data/`
- `models/`
- `runs/`
- compressed local archives such as `*.zip` and `*.tar.gz`

The public examples under `examples/` are small, tracked fixtures for reproducibility. Full deployment data should be supplied by the user and should not be committed by default.

## Claim Boundary

The reproduced system computes dataset-local paper influence. It should not be reported as global scientific impact unless the input corpus includes calibrated external citation/adoption evidence and validation labels that support that stronger claim.
