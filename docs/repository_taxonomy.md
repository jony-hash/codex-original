# Repository taxonomy

本分类按“真实消费者和拒绝决定”而不是文件数量、日期或磁盘占用划分。活动、冻结和历史是生命周期标签；它们不是 Git 目录层级，也不改变已有 artifact 的 provenance。

## Active runtimes

- `src/ldm_project/v1/` is the retained v1 runtime behind `train.py`.
- `src/ldm_project/v2/` is the contract-first runtime behind `python -m ldm_project.v2`.
- `src/ldm_project/v2/application/current/` owns formal identity execution;
  `application/diagnostic/`, `application/research/`, and `application/legacy/`
  are isolated non-promotion namespaces.
- `research/lightgen-ols-native-semantic-generation-v1` is the active OLS-native
  research branch; it is not a third formal runtime or a v2 contract replacement.
- Root-level Python modules in `src/ldm_project/` are compatibility facades only. New code must import a versioned runtime.

## Runtime and lifecycle assets

- `configs/v1/formal/`, `configs/v1/teacher/`, `configs/v1/smoke/`, and `configs/v1/diagnostic/` contain retained/runnable v1 configurations; runnable does not mean current acceptance-authorized.
- `configs/v2/` contains v2 configurations.
- `configs/research/` contains research-only lifecycle configurations. They
  cannot authorize the formal identity runtime or hardware promotion.
- `scripts/maintenance/` and `scripts/visualization/` contain shared utilities;
  `scripts/research/` contains evidence-bound research consumers and is not a
  generic current-runtime API.

## Managed outputs

- **Current runs:** Windows-native training, diagnostic and pre-acceptance runs
  under `outputs/YYYYMMDD_<version>/runs/<run>/`. They are current only when a
  named consumer and evidence status say so.
- **Protected checkpoints:** `outputs/final_pt/` and explicitly bound current
  checkpoints. They are never replaced by an automatic “latest output”.
- **Restored historical view:** visible historical checkpoints, images, JSON
  evidence and diagnostics materialized from the WSL archive. Restore membership
  is authoritative only from `outputs/_wsl_restore_metadata_v4/`.
- **Restore metadata:** `restore-plan.json`, `restore-state.json` and
  `path-map.jsonl` define membership, completion and Linux-to-Windows mapping;
  they are governance evidence, not runtime inputs.
- **External immutable archive:** the tar, index and manifest under
  `D:\LDM-Archive\codex-original\outputs` remain outside the current runtime
  and must not be modified.

Restored assets may be inspected for provenance but cannot overwrite, delete or
automatically feed current checkpoint resolution or formal acceptance. A new
run must use a fresh child path not listed as restored, even if it shares a
version-date parent.

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

## Research lifecycle

| Label | Meaning | Permitted use |
| --- | --- | --- |
| Active | Has a current evidence consumer and one declared next decision | Execute only within that consumer's authorization |
| Frozen reproducible | Closed or qualified result with replay value | Compare or replay by explicit artifact identity |
| Historical reference | Dated lineage, failed attempt or superseded plan | Provenance and learning only; never current acceptance input |

An evidence item without a clear consumer is not promoted to “active” merely
because a script or configuration still exists.

## Documentation

- `docs/architecture/`: stable v1/v2 design contracts.
- `docs/operations/`: commands, branching, and output handling.
- `docs/status/`: current handoff, the mutable hypothesis ledger, dated
  experiment facts, and specialized diagnostic status records. The first
  current snapshot is authoritative; dated entries are provenance.
- `docs/archive/`: immutable historical handoffs, reports, and acceptance JSON.

`PROJECT_MEMORY.md` stores durable navigation, `CURRENT_STATUS.md` stores the
current high-level state, `status/current_handoff.md` stores the latest delta
and any next allowed planning action (or an explicit stop), and
`experiment_history.md` stores dated facts. Do not copy full machine hash
tables into all four files.

Current navigation:

- [Capacity-map hypothesis ledger](status/capacity_map_hypothesis_ledger.md):
  closed electronic O0 hypotheses and historical attribution boundaries.
- [Current Handoff](status/current_handoff.md): active OLS-native N3 state and
  the current allowed action or explicit stop boundary.
- [Fiber image-generation status](status/v2_fiber_image_generation_v1.md) and
  [corrected fiber OLS status](status/v2_fiber_ols_corrected_status.md): fiber
  research boundaries and closed evidence.
- [Long-training preflight](operations/long-training-preflight.md): required
  gates before long runs, model selection, or scientific claims.

The existing experiment conclusions remain unchanged by this classification.
