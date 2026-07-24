# Optical Implementation Contract

Scope: `src/ldm_project/**`. Read the repository-root `AGENTS.md` first. This file adds implementation constraints and may not relax the root contract. Canonical v1 code lives in `v1/`, canonical v2 code in `v2/`; root modules are compatibility facades only.

## Domain Routing

- `optics/` owns formal field/provenance, propagation, relays, devices and phase-only D2NN; it imports no teacher, student, application or diagnostic code.
- `student/` owns the single-exposure forward/training/evaluation path; it may consume neutral `generation/`, `distillation/` and `evaluation/` modules but not teacher internals or `diagnostics`.
- `teacher/` owns VAE/diffusion and target export; it does not import student.
- `application/` owns CLI and lazy stage construction. Compatibility files at this directory level are import-only shims; root package model exports are lazy so data-only stages do not load execution networks.
- `student/formal_contract.py` validates the current formal model before construction; `student/checkpoints.py` owns checkpoint loading contracts. `student/generator.py` is the canonical `ElectronicGeneratorE`; `student/layers.py` is a compatibility facade and diagnostic implementations belong under `student/experimental/`.
- `configuration/` owns scoped config schema/loading/validation modules behind the public `config.py` facade. `teacher/distill.py` is a compatibility facade over scoped teacher distillation modules.

## Physical Model

- `generator_e` produces the actual amplitude SLM command. Do not silently reinterpret it as phase-only or complex-amplitude control.
- Under the current formal contract, encoder and decoder metasurfaces are trainable phase-only thin modulation layers. Complex or amplitude modulation is allowed only in an explicitly named diagnostic or separately validated future hardware variant; never infer it from a phase-only formal config.
- The formal macro path is a complete SLM-FOV coherent 4f front relay, encoder/decoder D2NN, identity OLS, and finite-conjugate backend relay. Preserve each relay's physical FOV, sampling, magnification, pupil, and ABCD provenance.
- OLS stays behind `complex field in -> linear identity OLS -> complex field out` for the current formal contract. Fiber, calibrated-matrix, and camera-in-the-loop implementations are future variants and must share that interface.
- CMOS/camera intensity is the primary student output. Digital-twin paths must represent the physical `SLM command -> CMOS image` contract.
- Preserve single-exposure inference: no iterative software denoising, latent sampling, or optimization in the final path.
- Formal relay provenance must expose the named SLM→BS→M3→L1→L2→Encoder segments; an aggregate distance alone is insufficient.

## Field And Visualization Semantics

- `amplitude` means field magnitude `|E|`.
- `intensity` means `|E|^2` or a detector-normalized image.
- `phase` means `arg(E)` from an available complex field.
- Intensity-only `relay_psf` exposes camera intensity but no camera field amplitude or phase. Respect explicit availability metadata; never infer availability from an all-zero tensor.
- Propagation-field width plots must not contain D2NN mask parameters. Save mask transmittance `|t|` and mask phase `arg(t)` only in device diagnostics.
- Formal planes must carry physical coordinates and sampling metadata; a tensor shape match does not justify a resize or a FOV change.
- The backend `free_space` path includes its configured thin lens and B=0 relay segments; it must not be treated as a no-lens propagation shortcut.
- Epoch, sample, eval, teacher, and audit outputs must use the same physical naming contract; do not restore misleading amplitude aliases or duplicate phase aliases.

## Compatibility And Verification

- Preserve documented optical ablation/slicing compatibility when loading checkpoints.
- Generator architecture or structure mismatches must fail explicitly; never silently map MLP weights into a different generator architecture.
- For one-shot changes, verify shapes, finite values, device placement, camera field availability, visualization splits, and checkpoint compatibility with focused tests.
