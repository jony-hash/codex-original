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

## Strict v3 evidence closeout (2026-08-21)

The repaired lifecycle completed on clean commit `cb5260f` in
`outputs/20260821_v2/runs/20260821_fiber_image_v3/`. Validation, paired
baseline, four function-specific CUDA preflights, all twelve isolated screens,
and locked selection completed with current lineage. At closeout time the
selection persisted 36 locked artifacts (four functions × three training seeds
× three panels), including content-hashed tensors, optical traces, port
coefficients, fixed exposure CMOS tensors, contact sheets, and paired
difference sheets. Its selection hash is
`0e5a41fef5c697d2e2da17049ba0e0a3a35645e54ebc1ae03d5c8b1ac281d7c6`.

The structural and measurement validity gates passed. The scientific closeout
is `valid_failed / no_reproducible_fiber_gain`:

- S0 median secondary adaptation gain reached at most `0.006`, S1a at most
  `0.004`, and S1b/S1c were `0` on the locked panels; all 95% bootstrap lower
  bounds were `0`.
- No function passed the preregistered `0.015` gain and semantic gates. The
  observed identity gaps were approximately `0.272` (S0), `0.314` (S1a),
  `0.326` (S1b), and `0.320` (S1c), while primary minimum recall remained
  zero.
- Cross-function differences were small (`0.002–0.006`) and every paired
  10,000-replicate confidence interval included zero.

Accordingly E1 shared-matrix training was gated off. This is a structurally
valid, synthetic diagnostic negative result: it does not promote a fiber
operator, generator, checkpoint, or hardware claim, and the formal identity
mainline remains unchanged. The v3 closeout is the first lifecycle result
eligible for local archival consideration; no tag was moved or pushed.

## No-update attribution and presentation bundle (2026-08-21)

The follow-up branch `research/v2-fiber-image-attribution-v1` completed a
no-update attribution run from the same research initialization. It evaluated
I0, nominal S0, phase-only, loss-only, offset-only, complete mild S1a, and
decoder-neutral S0/M1 controls on the same three 500-sample panels. The compact
measurement output is 655MB and is bounded by a 2GiB storage gate; the prior
91G v3 raw evidence remains unchanged and is not consumed as a writable
training source.

The observed secondary-evaluator drop is dominated by the LP01 coupling and
re-emission boundary: coupling effect `0.2413` with paired bootstrap 95% CI
`[0.2200, 0.2633]`. Loss contribution is approximately `0.0493` in the
registered direction, offset is `0.0067`, and phase/decoder effects are not
separated from zero at this panel resolution. The automatic attribution label
is `coupling_dominant_observed_drop`. This is a synthetic observation, not a
physical causal claim about a calibrated fiber.

The presentation bundle contains a labeled pipeline diagram, representative
amplitude/phase trace, fixed-exposure function comparison, v3 training-gain
plot, attribution waterfall, decoder-neutral comparison, CSV metrics, and a
slide outline. Closeout is `valid` with `formal_promotion=false`; no training
or E1 shared-matrix run was authorized.

## Evidence storage pruning and replay boundary (2026-08-21)

The raw v3 evidence was pruned under the approved Plan B storage contract.
The atomic manifest is
`outputs/20260821_v2/runs/20260821_fiber_image_v3/storage_pruning_manifest.v1.json`
with SHA-256
`ab05c5ded07d2af17bfbd0dd228d4307455f90a24c7fc993738fb7698aa4741b`.
It records 1,756 explicitly hashed files totalling 93.181 GiB: 756 locked
raw tensor files totalling 88.78 GiB and 4.40 GiB of invalid or superseded
historical runs and duplicate presentation artifacts. All listed targets were
removed; no unlisted path was deleted. The v3 run is now approximately 1.8 GiB
and the attribution run approximately 694 MiB.

The pruning does not alter any scientific value or closeout conclusion. It
does mean that v3 full raw optical-trace replay and raw-artifact integrity
revalidation are intentionally unavailable. The retained consumers are the
closeout/selection summaries, aggregate metrics, checkpoints, fixed-exposure
images, provenance, contact sheets, and the valid attribution/PPT bundle.
Training resume and formal promotion remain rejected. The 0–9 S0 pipeline
manifest and 20 per-digit images remain available.

`df` inside WSL reports the ext4 VHD logical capacity, not the Windows host's
physical free space. Before compaction, PowerShell measured
`D:\WSL\Ubuntu\ext4.vhdx` at approximately 324.42 GiB and D: with 50.97 GiB
free; the VHD was not sparse. The non-fiber portion of `outputs/` remains
approximately 210 GiB and was outside this cleanup authorization. Windows
VHDX compaction and its before/after measurements are a separate final step
and are not claimed until an administrator completes `diskpart compact vdisk`.
