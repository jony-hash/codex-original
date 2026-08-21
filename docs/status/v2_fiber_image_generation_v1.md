# v2 fiber image-generation lane status

The research lane is implemented on `research/v2-fiber-image-generation-v1`.
It is diagnostic-only and does not alter the formal identity OLS contract.

The clean validation evidence and paired baseline export are in
`outputs/20260821_v2/runs/20260821_fiber_image_v1/`. The baseline produced
paired I0/S0/S1a/S1b/S1c/N0 CMOS tensors, optical traces, port coefficients,
fixed-exposure contact sheets, and identity difference sheets. The N0/S0
baseline response was observable in both domains (CMOS normalized L2 about
0.848; port normalized L2 about 1.380).

CUDA S0 preflight reached all physical, finite-gradient, freeze, negative
control, full-phase, memory, and schedule checks. It is nevertheless
`invalid_harness` because the RTX 3060/PyTorch CUDA path cannot reproduce the
three-update interrupted/resumed generator within the preregistered
`rtol=atol=1e-5`: the recorded maximum error was `2.3228e-4` and optimizer
state comparison failed. The warning comes from the unsupported deterministic
`adaptive_avg_pool2d_backward_cuda` implementation; the tolerance was not
relaxed.

Consequently no isolated screen, locked select, E1 shared-matrix training, or
scientific closeout was authorized. There is no claim of fiber-dependent or
fiber-specific generator gain, and no archive tag was moved.

## Deterministic replay repair and completed paired rerun (2026-08-21)

The stale S0 screen above remains historical diagnostic evidence only. The
research lane now uses a mathematically equivalent fixed linear pooling
operator for the 200-to-28 exposure view and 28-to-14 Sinkhorn view. The
native-versus-research forward error in the new validation is
`5.55e-16`; CUDA interrupted/resumed replay passed for all four functions with
generator maximum error `0.0` and equal optimizer state.

The new run is
`outputs/20260821_v2/runs/20260821_fiber_image_v2/`. Its validation and paired
baseline are bound to commit `914206e`; the baseline contains I0, S0, S1a,
S1b, S1c and N0 tensors, optical trace amplitude/phase artifacts, field
provenance, port coefficients and contact sheets. The baseline reproduces the
main attribution: I0 primary/secondary accuracy is `1.00/0.40`, while S0 is
`0.60/0.16`; N0/S0 remains observable in both port and CMOS domains.

All twelve isolated screens passed their structural gates. Locked selection
was completed on three 500-sample panels with 10,000-replicate stratified
paired bootstrap (`90210`). The largest panel median secondary adaptation gain
was approximately `0.006`, below the preregistered `0.015` threshold; every
cross-function confidence interval included zero. The automatic closeout is
`valid_failed / no_reproducible_fiber_gain`. No function passed the isolated
scientific gates, so E1 shared-matrix training was correctly not run. This is a
synthetic diagnostic negative result, not a formal identity or hardware
conclusion, and no archive tag or formal checkpoint was changed.

The two WSL interruptions occurred outside the Python evidence path. The WSL
kernel log recorded unclean journal restart, DXG adapter query failures and
DNS `getaddrinfo -5`; no model exception or OOM record was present. The run
was resumed from the last completed stage using independent output paths, not
from a partial checkpoint.

## v1 closeout invalidation

The v1 numerical measurements remain useful diagnostic observations, but the
`valid_failed` closeout is not an authoritative lifecycle result. Its
measurement bundle did not bind the complete dataset/evaluator/source/pool
lineage, preflight did not persist all upstream hashes, and locked select did
not persist the required 500-sample field/image artifacts. The v1 closeout is
therefore treated as `invalid_harness diagnostic`; it is not eligible for an
archive tag or downstream promotion. The strict repair continues on
`research/v2-fiber-image-generation-v2` and writes only to a new v3 run.
