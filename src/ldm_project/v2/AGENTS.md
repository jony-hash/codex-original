# v2 Runtime Contract

Scope: `src/ldm_project/v2/**`. v2 is the current teacher-free conditional
optical-generation runtime.

- The current architecture is
  `conditional_identity_logit_amplitude_v2`: `label + 8D code` produces a
  global amplitude SLM command, then the unchanged formal finite-crop,
  identity-OLS, phase-only `2+2` optical system produces raw CMOS intensity.
- The frozen MNIST evaluator is a training and acceptance instrument only. It
  is not an inference-time optical component.
- Current v2 code must not import v1 teacher modules, teacher targets,
  prototype/capacity basis artifacts, direct-template artifacts, or diagnostic
  modules. Legacy v2 code is compatibility/reference-only and cannot authorize
  current acceptance.
- Formal optical fields retain plane metadata, physical coordinates and
  provenance. Electronic MNIST carrier initialization is real-valued input
  preparation; it must never be implemented as complex-field propagation or a
  generic complex resize.
- `student train`, `sample`, and `evaluate` remain public generic commands, but
  architecture dispatch must reject missing context before creating an
  optimizer or run artifact.

## Research Namespaces

`application/research/{lightgen,capacity,optical_manifold,scaled_chen}/` are
research-only, non-promotion namespaces. They may consume formal v2 code but may
not become part of it: `application/current/` and the formal student modules must
not import them, enforced by `tests/test_architecture_boundaries.py`. A research
namespace with no current consumer stays frozen; moving its files out of the
runtime package is a separate, explicitly authorized change.

## Shared Research Lifecycle

`application/research/shared/research_run_control.py` is research execution
governance, not part of formal v2 inference. Scientific producers must provide
a task-authorized READY receipt with explicit passing correctness, resource,
and implementation groups; execution and resume use the same receipt through
`open_scientific_attempt`. Do not restore the no-receipt
`begin_scientific_attempt` path. Source digests and Git state remain provenance
under the root policy, not admission or resume locks.
