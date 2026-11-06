# Documentation

Scope: `docs/**`. The root `AGENTS.md` defines the repository contract; this file adds placement rules.

## Placement

- `docs/` root holds only the entry files: `README.md`, `CURRENT_STATUS.md`, `PROJECT_MEMORY.md`, `repository_taxonomy.md`, and this contract.
- New topic documents go to:
  - `architecture/` — stable runtime contracts;
  - `operations/` — commands, branching, output and environment procedures;
  - `status/` — current handoff and dated state records;
  - `hardware/` — physical device and measurement documents;
  - `research/<family>/` — research reports (`lightgen`, `ols`, `scaled_chen`, `optical_manifold`, `capacity`);
  - `archive/<era>/` — superseded material.
- Do not add a new top-level directory under `docs/` without updating this file and `README.md`.

## Naming and links

- Use dated or versioned names only when the document is bound to one task; otherwise use a stable topic name.
- Links must be relative to the containing file and must resolve. Do not add absolute `D:/...` paths in new documents.
- When moving a document, update every inbound link in the same change.

## Write set

- `CURRENT_STATUS.md` records the latest scientific conclusions; `status/current_handoff.md` records the current stop point and allowed actions. Organizational work moves links only; it does not rewrite their dated entries.

## Verification

Run `python scripts/run_windows_cpu_checks.py` after changing repository structure; the focused lane includes the architecture-boundary tests.