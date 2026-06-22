# Optical Implementation Contract

Scope: `src/ldm_project/**`. Read the repository-root `AGENTS.md` first. This file adds implementation constraints and may not relax the root contract.

## Physical Model

- `generator_e` produces the actual amplitude SLM command. Do not silently reinterpret it as phase-only or complex-amplitude control.
- Encoder and decoder metasurfaces are thin complex modulation layers whose trainability and fabrication constraints must remain explicit.
- OLS stays behind `complex field in -> OLS/fiber module -> complex field out`; proxy, fiber, calibrated-matrix, and camera-in-the-loop implementations share that interface.
- CMOS/camera intensity is the primary student output. Digital-twin paths must represent the physical `SLM command -> CMOS image` contract.
- Preserve single-exposure inference: no iterative software denoising, latent sampling, or optimization in the final path.

## Field And Visualization Semantics

- `amplitude` means field magnitude `|E|`.
- `intensity` means `|E|^2` or a detector-normalized image.
- `phase` means `arg(E)` from an available complex field.
- Intensity-only `relay_psf` exposes camera intensity but no camera field amplitude or phase. Respect explicit availability metadata; never infer availability from an all-zero tensor.
- Propagation-field width plots must not contain D2NN mask parameters. Save mask transmittance `|t|` and mask phase `arg(t)` only in device diagnostics.
- Epoch, sample, eval, teacher, and audit outputs must use the same physical naming contract; do not restore misleading amplitude aliases or duplicate phase aliases.

## Compatibility And Verification

- Preserve documented optical ablation/slicing compatibility when loading checkpoints.
- Generator architecture or structure mismatches must fail explicitly; never silently map MLP weights into a different generator architecture.
- For one-shot changes, verify shapes, finite values, device placement, camera field availability, visualization splits, and checkpoint compatibility with focused tests.
