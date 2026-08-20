# Corrected fiber OLS status

The corrected operator is implemented on `research/v2-fiber-ols-v1`, but the
research lifecycle is not yet closed:

- LP01 V=2.2 is solved independently in float64. The current root is
  `u=1.5910592824373024`, `w=1.5193848622946382`; closure is below `1e-15`
  and boundary derivative mismatch is below `2e-15`.
- The primary model is a cell-partitioned 100-port LP01 overlap operator on
  the formal 100x100 encoder plane. It records input/output energy and rejects
  non-passive transport.
- Matrix records include exact realized phase/loss statistics, independent
  signed y/x offsets, and hashes over the matrix realization plus provenance.
- `validate` now consumes the offline fixture, runs an independent finite-volume
  radial check, measures the actual Lowdin overlap, and records the transport
  pool provenance.
- `preflight`, `screen`, and `select` are still being hardened to bind the
  research-only generator initialization, dual evaluator, resume replay, B0,
  identity, bootstrap, and automatic closeout gates.

The old CUDA preflight was blocked by lineage, not by a model result. The
available 20260812 checkpoint carries config hash
`77db9b7fcfe47cb53f2d089c40454e811a898a329a8f694b8370b8d17d81706d`, while
the current formal config hashes to
`c585bc683dc9af347699c24c79431f3fab398efc533a17778c7c9103dcc7e987`.
The checkpoint is therefore accepted only as an explicitly labelled research
initialization: its generator tensors are imported into a freshly constructed
current identity system, while its old relay/system state is never loaded or
re-signed. Until the resulting clean-branch CUDA preflight passes, no screen,
locked selection, tag movement, or scientific closeout is authorized.
