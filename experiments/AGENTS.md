# Archived Experiment Contract

`experiments/archive/**` is reference-only source. It is `diagnostic-only`, is
not an active CLI/config entrypoint, and must not be promoted to a formal
result or repaired merely to run from its archived path.

- Keep historical files immutable except for manifest maintenance.
- Use the named safety snapshot and `manifest.yaml` to reproduce an era.
- Do not move, delete, or rewrite `outputs/`, `data/raw/`, `.tmp/`, or `legacy/`
  while working with archive material.
- A new active experiment must be recreated as an explicit current config/tool;
  it may not be promoted by moving an archived file back to the root.
