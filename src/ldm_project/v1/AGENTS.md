# v1 Runtime Contract

Scope: `src/ldm_project/v1/**`. v1 is the preserved LDM-teacher distillation
runtime and remains the implementation behind `train.py`.

- Teacher, distillation targets, one-shot student training and v1 evaluation
  may use the v1 optics and neutral modules according to the repository
  dependency direction.
- v1 checkpoints and historical state-key signatures are compatibility
  artifacts. Do not change them while reorganizing v2.
- v1 code must not import `ldm_project.v2` conditional generators, evaluators,
  or v2 evidence records.
- New v1 changes must preserve the documented `train.py --stage` interface.
