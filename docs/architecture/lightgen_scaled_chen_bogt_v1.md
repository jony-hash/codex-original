# Scaled Chen-reference BOGT–OLS lifecycle

This document defines the research-only `scaled_chen_bogt` path. It is not a
replacement for the formal v2 `label + 8D` runtime and cannot change the
existing SMF/FMF evidence or authorize promotion.

> **Scope:** `HISTORICAL_RECORD` / `SUPERSEDED` for the current project route. Historical facts in this document remain valid for the Scaled-Chen line; its old current/next actions are superseded by the M0 authority in `docs/CURRENT_STATUS.md` and `docs/status/current_handoff.md`.

The optical chain is `MNIST -> sqrt-intensity amplitude SLM -> front 4f ->
2 encoder phase layers -> fixed 10x10 LP01 OLS -> 2 decoder phase layers ->
inferred 3 mm coherent imaging relay -> CMOS`. The metasurface is 100x100 at
2 µm pitch, the field of view is 200 µm, the wavelength is 532 nm, and the
inter-layer spacing is 1.50 mm. Every phase layer has a fixed field factor
`sqrt(0.889)`. The OLS has 100 fixed ports and identity passive transport using
the existing scaled `V=2.2`, 7.111 µm waist, fill 0.8 mode bank.

BOGT is aggregate and physical: `z_j = |c_j|²/s_j`, with `s_j` measured from
1000 independent train-only zero-phase calibration samples. The fixed prior
is a non-negative truncated normal with mean 1 and variance 0.25. The training
KL is its diagonal Gaussian surrogate over a class-balanced logical batch of
ten samples. No electronic `mu/logvar`, reparameterization, active latent
injector, software decoder, or trainable OLS is present.

The 3 mm backend is an explicitly labeled inference: for magnification -2.5,
`u=0.857142857 mm`, `v=2.142857143 mm`, and `f=0.612244898 mm`, with no extra
pupil clipping. The real Chen fiber length and outer diameter remain paper
reference metadata only and are not used in scaled coordinates.

The lifecycle is fail-closed. Validation records the geometry and lineage;
CUDA preflight requires a three-step real update, a 2+1 resume check and
negative controls; only then may the four-arm 100-step pilot run. The pilot is
reviewed before continuation, prior-only generation, capacity audits, or any
promotion. Source details are labeled `paper_reported`,
`user_locked_scaled_deviation`, `high_confidence_inference`, or `unknown` in
the research configuration.
