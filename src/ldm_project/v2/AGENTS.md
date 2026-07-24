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
