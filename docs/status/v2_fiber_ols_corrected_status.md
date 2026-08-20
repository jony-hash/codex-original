# Corrected fiber OLS status

The corrected operator and synthetic diagnostic lifecycle are implemented on
`research/v2-fiber-ols-v1`; formal identity promotion remains closed:

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
- `preflight`, `screen`, and `select` bind research-only initialization, dual
  evaluators, resume replay, B0/identity controls, bootstrap, and closeout.

The old CUDA preflight was blocked by lineage, not by a model result. The
available 20260812 checkpoint carries config hash
`77db9b7fcfe47cb53f2d089c40454e811a898a329a8f694b8370b8d17d81706d`, while
the current formal config hashes to
`c585bc683dc9af347699c24c79431f3fab398efc533a17778c7c9103dcc7e987`.
The checkpoint is therefore accepted only as an explicitly labelled research
initialization: its generator tensors are imported into a freshly constructed
current identity system, while its old relay/system state is never loaded or
re-signed. Until the resulting clean-branch CUDA preflight passes, no screen,
locked selection, tag movement, or scientific closeout was authorized until the
corrected lifecycle recorded below completed.

## Synthetic ensemble closeout (2026-08-21)

The corrected lifecycle completed on the clean research runtime:

- validation: `37d9d5edf3d82f2a6fe868b2c3e744e16c09efc0dd815c8137263f249237fbf`;
- CUDA preflight: `9c94e8d5dd7e0f35634af21f96012278fb7af9c522b63e4193f593cf405faa75`;
- three-seed screen: `a38325ac4cc8b25b7fba1dd56ab34a294e30edf0c10777d8cf202db9e71a09e8`;
- locked selection: `d69d28c2331a43bf18f4973e5d57ac0f296824b925a1e1e459c6e39937eb59fc`;
- closeout: `4247abf6ae2386952591ab3c33c9c16a4401551eeeb45404c2e5d238a5b7e2ca`.

All measurements were finite and passive, but secondary median gains were
approximately zero with bootstrap lower bounds not above zero. The automatic
conclusion is `no_reproducible_generator_gain` with `valid_failed`; no formal
promotion or additional training is permitted. The result is synthetic,
diagnostic-only, and does not establish hardware or calibrated-fiber behavior.

The historical invalid archive target remains recorded as `b873eba`. The local
archive tag may now be moved to the final closeout documentation commit while
preserving that old target in this audit record; it is not pushed.
