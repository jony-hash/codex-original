# Experiment Configuration Contract

Scope: `configs/**`. Read the repository-root `AGENTS.md` first. This file adds configuration constraints and may not relax the root contract.

Only Golden Path, teacher/smoke, v52/v55/v56 and current decision-gate configs belong here. Historical YAML lives in `experiments/archive/` and is reference-only.

## Configuration Rules

- YAML must remain compatible with `load_training_config`; use existing keys and naming patterns before adding a new option.
- Explicit CLI seed, checkpoint, dataset, geometry, and output arguments take precedence. Never guess a missing path or silently change geometry.
- Keep paired train/eval configs aligned on optical geometry, SLM semantics, readout model, target source, and checkpoint contract.
- Audit and formal configurations must state their diagnostic or acceptance role in the filename and run metadata.

## Optical Path Policy

- `relay_identity`, `relay_coherent_psf`, and `relay_psf` are diagnostic relay models. They may not be used as the current v55/v56 physical-forward contract.
- The current physical-forward configuration must explicitly use `front_path_model: coherent_imaging_4f`, `back_path_model: free_space`, `physical_forward_contract: true`, and `ols_kind: identity` with the documented sampling geometry. The legacy v52/v53 free-space-plus-aperture path is historical diagnostic evidence only.
- v55.1/v55.2/v58.2 的旧 reference report 仅是历史自洽或 invalid harness 证据；完整 FOV 的 v58.3 B=0 relay reference 必须通过，才可签发新的 formal-valid status。
- The front `coherent_imaging_4f` relay maps the complete 500 µm SLM FOV to the complete 200 µm encoder FOV with magnification `-0.4`, `A=-0.4`, `B≈0`, and front clear aperture at least 23.75 mm. Central aperture sampling and generic complex resize are forbidden in the formal path.
- The backend `free_space` name means two free-space segments around the configured thin lens, not a no-lens path: chip → 100 mm free space → `f=71.4285714 mm` lens → 250 mm free space → CMOS. The contract must validate `B≈0`, magnification `-2.5`, CMOS FOV 500 µm, and clear aperture at least 26.60 mm.
- Do not switch a failed physical path to a readability-only or identity path without explicit approval and a `diagnostic-only` label.
- Keep amplitude SLM control, single-exposure inference, and camera-primary output unchanged across config variants.

## Runs And Artifacts

- Diagnostic and pre-acceptance configs write to persistent `outputs/YYYYMMDD_<version>/...` paths by default; keep them labeled `diagnostic-only` and do not treat their checkpoints as accepted artifacts.
- Accepted/configured runs write to the real prefixed directory `outputs/YYYYMMDD_<version>/runs/...`; never overwrite an existing run, and do not reintroduce `outputs/runs/`.
- Long or full-dataset training is not a routine verification step. Respect pre-registered gates before extending epochs or promoting a config.
