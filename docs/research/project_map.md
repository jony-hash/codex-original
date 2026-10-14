# Research Code Map

Research code, configuration and tests use the same subject families. Formal
v1/v2 runtime paths were not moved.

| Before | Current location | Notes |
| --- | --- | --- |
| `scripts/research/*.py` | `scripts/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | Command entry points grouped by subject; cross-family helpers stay small. |
| `src/ldm_project/v2/application/research/*.py` | `src/ldm_project/v2/application/research/{lightgen,capacity,optical_manifold,shared}/` | Runtime helpers grouped by family; `research_run_control.py` is in `shared/`. |
| E_D / E_C model definitions in evaluator training commands | `src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py` | Training commands keep their original CLI and re-export the shared classes; parameter names and checkpoint state keys are unchanged. |
| `configs/research/*` | `configs/research/{scaled_chen,optical_manifold,capacity,lightgen}/` | LightGen now includes a standalone zero-update Gaussian S12/LP01 G3 audit config; it is diagnostic-only and does not select a production default. |
| Research tests under `tests/v2/` | `tests/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | Formal v2 runtime tests remain under `tests/v2/`. |
| `docs/CURRENT_STATUS.md`, `docs/status/current_handoff.md`, `docs/PROJECT_MEMORY.md`, `docs/status/experiment_history.md` | `docs/archive/20261001_project_refactor/` snapshots | Original long contents are preserved; stable bridge filenames remain as short current indexes. |
| Stopped M0 executors and their historical tests | `scripts/research/archive/project_refactor_20261001/lightgen/m0/` and `tests/archive/project_refactor_20261001/lightgen/m0/` | M0 checkpoints and reports stay under `outputs/`; archived runners do not create a current authorization or science result. |
| Closed N3 executors and their historical tests | `scripts/research/archive/project_refactor_20261001/lightgen/n3/` and `tests/archive/project_refactor_20261001/lightgen/n3/` | The N3 line has no current consumer; original output evidence remains under `outputs/`. |

## Entry-point status

The family roots are lookup locations, not an execution queue. Current commands
are identified by `docs/CURRENT_STATUS.md` and its current handoff. Frozen replay
commands are usable only to reproduce named historical evidence under their
documented inputs. `archive/` trees are read-only history and never grant
execution authority. A directory name, manifest, or old authorization does not
make a command current. Shared lifecycle code lives in the `shared/` runtime
package rather than being copied into each family.

| Kind | Entry or implementation | How to use it |
| --- | --- | --- |
| Maintained formal runtimes | `train.py` → `src/ldm_project/v1/`; `python -m ldm_project.v2` → `src/ldm_project/v2/application/current/` | Follow the selected runtime's contract and explicit inputs. |
| Maintained BOGT execution boundary | `scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py` | `--preflight` is CPU engineering; `--production-preflight --checkpoint <explicit> --output-dir <new>` runs a diagnostic-only production CUDA check and exits; `--execute` performs its own preflight before a scientific attempt. Neither diagnostic mode grants successor authority. |
| Physical OLS G3 and dimension re-audit | v5 zero-update evidence: `docs/ols_window_dimension_power_v5.md`, `artifacts/ols_window_dimension_power_v5.json`, and `outputs/20261014_ols_window_dimension_power_v5/runs/autodl_backup_attempt_02/`; v4/v3 scripts remain named historical inputs | Current evidence is `VALID_COMPLETED`: expanded 3072/4096/6144 windows pass all nine production-complex64 common-ROI comparisons at the original `1e-4`, and fixed-phase K/H response spectra plus absolute fiber-power measurements completed at 3072. Historical v4 2048-window `G3_FAIL` remains unchanged. The 125-point reference checks do not explain the full old-window mechanism. Zero-phase 10×10 remains concentrated in four channels (`N_eff≈4`, 4% utilization at −10 to −40 dB); amplitude-response spectrum depends on Gaussian/LP01 and threshold, with no noise calibration. This supports planning a separate phase-only diagnostic, not choosing an OLS dimension, adopting a production default, or training. |
| Current three-workstream closeout | `docs/status/lightgen_three_workstreams_closeout_20261014.md`; current status and handoff in `docs/CURRENT_STATUS.md` and `docs/status/current_handoff.md` | Identity evidence is mixed and the specific stroke interpretation remains incomplete; OLS/G3 v5 is complete only for its stated simulation; Generator recovery remains `INVALID_HARNESS / INCOMPLETE` at 39/45 engineering updates. An independently authorized OLS v6 task is tracked by its own session and evidence. |
| Current frozen research evidence | Class-1/7 diagnostic, Generator-only reconstruction, and [existing 400×400 step600 frozen replay](../../outputs/20261002_m0_existing_400_frozen_replay_validity_v1/runs/attempt_01/report.json), linked from `../CURRENT_STATUS.md` | Read the saved reports; their completion is not a new execution queue. |
| Earlier frozen executors in family roots | Prior attribution, recovery, capacity and evaluator comparison scripts | Historical file-hash checks and task constants remain in some files. Replay their named evidence with the corresponding old Git version; do not treat them as current admission rules. |
| Archived executors | `scripts/research/archive/` | Preserve the original text and records. Directory membership does not authorize execution. |

This remediation reuses the BOGT update/reload/rollback implementation for CPU
and CUDA preflight. It does not split every large entrypoint or migrate every
historical family; file length alone is not the refactoring criterion.
