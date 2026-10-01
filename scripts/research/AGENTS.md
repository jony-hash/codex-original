# Research Scripts

Research command-line entry points are grouped by subject: `lightgen`,
`scaled_chen`, `optical_manifold`, and `capacity`. Put cross-family runtime
helpers in `src/ldm_project/v2/application/research/shared/`; keep a script as
a small argument, load, compute, measure, and save layer.

Use explicit checkpoint, data, config, and output paths. Check the loaded
model step and optimizer state, the selected sample rows and labels, data
roles, trainable groups, evaluator isolation, numerical validity, and actual
hardware safety. Durations and source digests may be recorded as provenance;
they do not grant or prevent execution. The repository root
[`AGENTS.md`](../../AGENTS.md) defines the research authorization and stop
rules.
