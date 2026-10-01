# Configurations

Keep formal runtime configs under their existing `v1/` and `v2/` trees.
Research configs are grouped under `configs/research/{lightgen,scaled_chen,
optical_manifold,capacity}`. Update the matching command/test paths when a
config moves; see [`docs/research/project_map.md`](../docs/research/project_map.md).

Use the loader for the selected runtime, keep physical geometry and data roles
explicit, and do not silently substitute a missing path or diagnostic model.
The repository root [`AGENTS.md`](../AGENTS.md) defines shared research rules.
