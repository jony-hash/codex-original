# Research Code Map

Research code, configuration and tests use the same subject families. Formal
v1/v2 runtime paths were not moved.

| Before | Current location | Notes |
| --- | --- | --- |
| `scripts/research/*.py` | `scripts/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | Command entry points grouped by subject; cross-family helpers stay small. |
| `src/ldm_project/v2/application/research/*.py` | `src/ldm_project/v2/application/research/{lightgen,capacity,optical_manifold,shared}/` | Runtime helpers grouped by family; `research_run_control.py` is in `shared/`. |
| E_D / E_C model definitions in evaluator training commands | `src/ldm_project/v2/application/research/lightgen/confirmation_evaluators.py` | Training commands keep their original CLI and re-export the shared classes; parameter names and checkpoint state keys are unchanged. |
| `configs/research/*` | `configs/research/{scaled_chen,optical_manifold,capacity}/` | These are the families with tracked research configs; LightGen currently has no standalone research config files. |
| Research tests under `tests/v2/` | `tests/research/{lightgen,scaled_chen,optical_manifold,capacity}/` | Formal v2 runtime tests remain under `tests/v2/`. |
| `docs/CURRENT_STATUS.md`, `docs/status/current_handoff.md`, `docs/PROJECT_MEMORY.md`, `docs/status/experiment_history.md` | `docs/archive/20261001_project_refactor/` snapshots | Original long contents are preserved; stable bridge filenames remain as short current indexes. |
| Stopped M0 executors and their historical tests | `scripts/research/archive/project_refactor_20261001/lightgen/m0/` and `tests/archive/project_refactor_20261001/lightgen/m0/` | M0 checkpoints and reports stay under `outputs/`; archived runners do not create a current authorization or science result. |
| Closed N3 executors and their historical tests | `scripts/research/archive/project_refactor_20261001/lightgen/n3/` and `tests/archive/project_refactor_20261001/lightgen/n3/` | The N3 line has no current consumer; original output evidence remains under `outputs/`. |

The research family roots are the lookup entrypoints. Do not infer that an
experiment is active from its folder or filename; use `docs/CURRENT_STATUS.md`
and the evidence it names. Shared lifecycle code lives in the `shared/`
runtime package rather than being copied into each family.
