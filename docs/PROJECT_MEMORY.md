# Project Memory

This file is a short navigation aid, not an execution authorization. The local
repository source, evidence and root [`AGENTS.md`](../AGENTS.md) take priority.

## Runtime Boundaries

- v1 remains the LDM-teacher distillation line behind `train.py`.
- v2 remains the teacher-free conditional runtime behind `python -m
  ldm_project.v2`; its formal path is amplitude SLM → encoder metasurface →
  identity OLS → decoder metasurface → one CMOS exposure.
- LightGen OLS-native research is separate from both formal runtime contracts.
  Its current question, evidence, and stop point are maintained in
  [`CURRENT_STATUS.md`](CURRENT_STATUS.md), not in this navigation file.
- `raw_intensity` is the production raw-CMOS accessor. `IntensityFrame` stores
  its image in `.tensor`.

## Project Navigation

| Work | Location |
| --- | --- |
| v1 runtime | `src/ldm_project/v1/` |
| v2 runtime and optics | `src/ldm_project/v2/` |
| LightGen research | `scripts/research/lightgen/`, `configs/research/lightgen/`, `tests/research/lightgen/` |
| Scaled Chen research | matching `scaled_chen/` family directories |
| Optical manifold research | matching `optical_manifold/` family directories |
| Capacity research | matching `capacity/` family directories |
| Shared research lifecycle | `src/ldm_project/v2/application/research/shared/` |
| Current status | `docs/CURRENT_STATUS.md` |
| Current handoff | `docs/status/current_handoff.md` |
| Full historical record | `docs/archive/20261001_project_refactor/experiment_history_full.md` |
| Windows development and CPU verification | `docs/operations/windows_development.md` |

## Reading Results

An engineering failure (`INVALID_HARNESS`) is not a scientific negative
result. A valid failed measurement applies only to its stated samples, model,
objective and budget. E_D is a frozen confirmation evaluator, not part of the
optical inference path. TRAIN-neighbor samples are not independent-prior
evidence. A loss decrease alone does not establish readable semantic images.

Current LightGen status and the next authorized action, if any, are recorded in
`docs/CURRENT_STATUS.md`; this memory file does not create an experiment or
grant training permission. Historical detailed memory remains in the dated
archive so the fixed bridge filename continues to resolve.
