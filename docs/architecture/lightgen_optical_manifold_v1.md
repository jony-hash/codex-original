# LightGen-like optical manifold research v1

This document registers `H-LOMA-I0`: a diagnostic-only Identity optical
manifold autoencoder. It is an independent v2 research line; it is not a
resumption of Package B3 and it does not replace the formal
`label + 8D code` current runtime.

## Pinned paper evidence

The user-provided Science supplement and extracted figure-source CSV files are
pinned in [`research_assets/lightgen/science_adv7434/`](../../research_assets/lightgen/science_adv7434/).
Read its `manifest.json` before using the files for a paper comparison. The
bundle is reference-only: it does not change this research line's source-status
ledger, formal v2 geometry, or training authorization. The CSV files contain
figure-level A.U. summaries and two-dimensional OLS display coordinates; they
are not raw complex fields or training targets.

## Scientific question and contract

The question is whether the complete formal 2+2 optical path can learn a
reconstruction mapping from real MNIST images when the OLS is the identity:

```text
MNIST intensity [N,1,28,28]
  -> real bilinear rendering to the 200x200 SLM field of view
  -> sqrt(clamp(intensity, 0, 1)) amplitude command
  -> formal coherent front relay
  -> two phase-only encoder layers
  -> complex identity OLS
  -> two phase-only decoder layers
  -> formal backend relay
  -> raw CMOS intensity
  -> fixed global exposure
  -> 200x200 reconstruction MSE
```

Only `system.encoder.phase` and `system.decoder.phase` are trainable. The SLM
command, exposure, relay, OLS, and target are fixed. Labels select one
deterministic representative per class for the preflight batch; labels are not
model inputs and do not enter the loss. There is no evaluator, classifier, KL,
diversity, teacher, digital decoder, SMF, FMF, latent sampling, or formal
training command in this first implementation.

The reconstruction target is the same real-valued MNIST intensity rendered to
the same 200x200 CMOS field of view. Complex fields are created only at the
SLM boundary and are never resized. The `sqrt` input transform is a
project-physics choice: public LightGen material reports a photonic encoder and
amplitude/phase OLS but does not expose exact SLM preprocessing or source code.
The source ledger therefore distinguishes `paper_reported`,
`project_physics_choice`, and `unknown` rather than attributing this transform
to LightGen.

## Evidence and stopping rules

`validate` produces schema
`lightgen_optical_manifold_identity_validation.v1`. It verifies clean lineage,
the independent branch, the locked formal geometry, Identity OLS, MNIST
manifest, coordinate/target contract, fixed exposure, and the exact trainable
parameter set before an optimizer or preflight run exists.

`preflight` consumes matching validation evidence and performs three real CUDA
optimizer updates with microbatch one, activation checkpointing, the 2048
finite-crop grid, and Adam at `1e-3`. It also verifies finite camera/loss/
gradients, non-zero updates for all four phase layers, frozen-state isolation,
loss decrease, zero evaluator calls, and exact uninterrupted 3-step versus
interrupted 2+1-step replay. The resource gates are at least 4 GiB free before
execution, 5 GiB peak allocation, 300 seconds wall time, and 2 GiB run
storage.

Every failed execution gate is `invalid_harness`; it cannot support a model
capacity conclusion. Passing evidence still records
`training_authorized=false` and `formal_promotion=false`. Its sole next
action is `manual_review_identity_reconstruction_training_contract`.
No formal reconstruction training is supplied by this first route.

## Source-status ledger

| Field | Status | Basis |
| --- | --- | --- |
| photonic encoder and OLS architecture | `paper_reported` | LightGen, DOI `10.1126/science.adv7434` |
| amplitude and phase at OLS | `paper_reported` | LightGen public article |
| exact LightGen SLM preprocessing | `unknown` | not disclosed in public article/material |
| `sqrt(intensity)` amplitude encoding | `project_physics_choice` | preserves this repository's `intensity=|E|^2` semantics |
| 532 nm, 200x200 @ 2.5 µm formal bench | `project_physics_choice` | current formal v2 contract |
| 100x100 @ 2 µm and 2+2 phase-only layers | `project_physics_choice` | current formal v2 contract |
| Identity OLS and complete relays | `project_physics_choice` | current formal v2 contract |

LightGen comparison remains explanatory context only. This route cannot
authorize formal v2 promotion or any fiber/SMF/FMF stage.
