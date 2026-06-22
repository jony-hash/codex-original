# Experiment Configuration Contract

Scope: `configs/**`. Read the repository-root `AGENTS.md` first. This file adds configuration constraints and may not relax the root contract.

## Configuration Rules

- YAML must remain compatible with `load_training_config`; use existing keys and naming patterns before adding a new option.
- Explicit CLI seed, checkpoint, dataset, geometry, and output arguments take precedence. Never guess a missing path or silently change geometry.
- Keep paired train/eval configs aligned on optical geometry, SLM semantics, readout model, target source, and checkpoint contract.
- Audit and formal configurations must state their diagnostic or acceptance role in the filename and run metadata.

## Optical Path Policy

- `relay_identity` is audit/feasibility-only and may appear only in `audit_*` configurations.
- Formal one-shot train/eval configs must use an audited non-ideal path such as `relay_coherent_psf` or `relay_psf`; they must not use `relay_identity`.
- Do not switch a failed physical path to a readability-only or identity path without explicit approval and a `diagnostic-only` label.
- Keep amplitude SLM control, single-exposure inference, and camera-primary output unchanged across config variants.

## Runs And Artifacts

- Diagnostic and pre-acceptance configs write to `/tmp`; do not treat their checkpoints as managed acceptance artifacts.
- Accepted/configured runs write to `outputs/runs/...`; never overwrite or backfill an existing run.
- Long or full-dataset training is not a routine verification step. Respect pre-registered gates before extending epochs or promoting a config.
