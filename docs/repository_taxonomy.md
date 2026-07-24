# Repository taxonomy

## Active runtimes

- `src/ldm_project/v1/` is the retained v1 runtime behind `train.py`.
- `src/ldm_project/v2/` is the contract-first runtime behind `python -m ldm_project.v2`.
- Root-level Python modules in `src/ldm_project/` are compatibility facades only. New code must import a versioned runtime.

## Active assets

- `configs/v1/formal/`, `configs/v1/teacher/`, `configs/v1/smoke/`, and `configs/v1/diagnostic/` contain runnable v1 configurations.
- `configs/v2/` contains v2 configurations.
- `scripts/maintenance/` and `scripts/visualization/` contain version-neutral utilities.

## Historical assets

`experiments/archive/` is reference-only. Its `manifest.yaml` and
`path_migration_manifest.yaml` provide the source path, target path, baseline
commit, and content verification method for each archival era. Historical
configurations and tools are never active entrypoints.

## Documentation

- `docs/architecture/`: stable v1/v2 design contracts.
- `docs/operations/`: commands, branching, and output handling.
- `docs/status/`: current handoff and experiment history.
- `docs/archive/`: immutable historical handoffs, reports, and acceptance JSON.

The existing experiment conclusions remain unchanged by this classification.
