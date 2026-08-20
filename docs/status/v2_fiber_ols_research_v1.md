# v2 synthetic fiber OLS research v1

This branch is diagnostic-only. The formal v2 path remains
`ols_kind: identity` and has no fiber selector. The research path injects a
replaceable `OpticalLatentOperator` and a deterministic 100-port diagonal SMF
transport pool.

## Frozen nominal history

The prior nominal LP01 CUDA screen selected step 50 for seeds 5201, 5202 and
5203. Those are training/data seeds, not three fiber matrices. Its conclusion
was `seed_sensitive_generator_gain`; the worktree was dirty and the result is
only a historical numerical comparison. It does not authorize formal or
hardware claims.

## Research contract

`FiberTransportPool` contains nominal identity plus 24 mild training matrices,
four validation matrices, eight locked mild holdouts, four stress matrices and
four full-phase negative controls. Every matrix is diagonal 100x100, passive,
content-hashed and provenance-bound. Training schedules begin with a complete
pass over the training pool before deterministic seeded draws.

`research fiber-robustness validate|preflight|screen|select|closeout` is the
single command family. Validation evidence is consumed by preflight; a dirty
patch hash is an `invalid_harness` stop. All outputs are synthetic
diagnostic-only and cannot promote the formal identity line.

No real NA, MFD, refractive index, length or measured complex matrix is
available. A future calibrated-fiber study must use a separate contract.
