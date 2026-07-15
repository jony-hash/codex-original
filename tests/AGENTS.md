# Test And Acceptance Contract

Scope: `tests/**` and verification for repository changes. Read the repository-root `AGENTS.md` first. This file adds test constraints and may not relax the root contract.

## Test Selection

- Start with tests nearest the changed behavior.
- One-shot model, optics, visualization, config, or checkpoint changes normally require `tests/student/`, `tests/teacher/`, `tests/application/`, `tests/diagnostics/`, the compatibility smoke files, and `tests/test_repo_layout.py` as applicable.
- Teacher/first-stage/diffusion changes normally require the matching conditioned, diffusion, data, or optical-path tests rather than an unrelated fixed bundle.
- Docs-only and repository-instruction changes require `git diff --check`; do not run Python or CUDA tests unless behavior-bearing files also changed.
- Run `python -m compileall src tests train.py` for shared Python changes and `ruff check src tests *.py` when lint-relevant files change.
- Run the full suite only after targeted checks pass for shared/final changes. If a corrective edit can affect the result, rerun the invalidated check.
- Keep architecture-boundary and lazy-stage assertions separate from physics/numerical assertions; legacy imports must resolve to canonical package implementations.

## Acceptance Semantics

- CPU smoke proves only local execution. It cannot replace CUDA, full-geometry, hardware, or strict-readability acceptance.
- A valid failed gate stops downstream training or promotion. A malformed dataset, wrong checkpoint, invalid geometry, or broken evaluator invalidates the gate and must be repaired first.
- Tests for formal train/eval layouts must keep `relay_identity` audit-only.
- Use temporary directories for generated test artifacts. Do not write tests into existing `outputs/` runs.
- Report skipped or unavailable verification precisely; do not convert it into a passing result.
