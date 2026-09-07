# Repository taxonomy

## Active runtimes

- `src/ldm_project/v1/` is the retained v1 runtime behind `train.py`.
- `src/ldm_project/v2/` is the contract-first runtime behind `python -m ldm_project.v2`.
- `src/ldm_project/v2/application/current/` owns formal identity execution;
  `application/diagnostic/`, `application/research/`, and `application/legacy/`
  are isolated non-promotion namespaces.
- Root-level Python modules in `src/ldm_project/` are compatibility facades only. New code must import a versioned runtime.

## Active assets

- `configs/v1/formal/`, `configs/v1/teacher/`, `configs/v1/smoke/`, and `configs/v1/diagnostic/` contain runnable v1 configurations.
- `configs/v2/` contains v2 configurations.
- `configs/research/` contains active research-only lifecycle configurations.
  They cannot authorize the formal identity runtime or hardware promotion.
- `scripts/maintenance/` and `scripts/visualization/` contain version-neutral utilities.
- `outputs/` is a managed artifact tree with separate roles: current
  Windows-native runs, protected `outputs/final_pt/`, and the materialized view
  of restored historical outputs. The restored view is bounded by
  `outputs/_wsl_restore_metadata_v4/restore-plan.json`, `restore-state.json`,
  and `path-map.jsonl`; it is provenance material, not an active runtime input.
- `outputs/version_index.json` and `outputs/version_index.md` are historical
  WSL-generated indexes. Their original absolute paths may be retained for
  provenance and must not be used by current Windows path resolution.

## Research reference assets

- `research_assets/lightgen/science_adv7434/` is an immutable, read-only
  evidence bundle containing the user-provided Science supplement and extracted
  figure-source CSV files. Its `manifest.json` is consumed by the LightGen
  paper-parameter and figure-source audit; a missing or tampered file rejects
  claims based on that bundle.
- Research reference assets are not runtime inputs, checkpoints, training data,
  or formal acceptance evidence. They must not be auto-resolved by a CLI.

## Historical assets

`experiments/archive/` is reference-only. Its `manifest.yaml` and
`path_migration_manifest.yaml` provide the source path, target path, baseline
commit, and content verification method for each archival era. Historical
configurations and tools are never active entrypoints.

## Documentation

- `docs/architecture/`: stable v1/v2 design contracts.
- `docs/operations/`: commands, branching, and output handling.
- `docs/status/`: concise current handoff, the mutable hypothesis ledger, dated
  experiment facts, and specialized diagnostic status records.
- `docs/archive/`: immutable historical handoffs, reports, and acceptance JSON.

Current navigation:

- [Capacity-map hypothesis ledger](status/capacity_map_hypothesis_ledger.md):
  electronic O0 hypotheses and the only permitted next action.
- [Fiber image-generation status](status/v2_fiber_image_generation_v1.md) and
  [corrected fiber OLS status](status/v2_fiber_ols_corrected_status.md): fiber
  research boundaries and closed evidence.
- [Long-training preflight](operations/long-training-preflight.md): required
  gates before long runs, model selection, or scientific claims.

The existing experiment conclusions remain unchanged by this classification.
