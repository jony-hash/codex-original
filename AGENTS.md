# AGENTS.md

- **Codex only**: In the first assistant message of each new task, the first line must be exactly `【AGENTS effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.
- Claude Code 请参见 [CLAUDE.md](CLAUDE.md)，以 CLAUDE.md 为首要行为约束。

Primary execution contract for this repository. This file overrides `docs/`; scoped files may add constraints but may not relax this contract.

## 1. Scope Router

Before changing a scoped path, read its local contract:

- `src/ldm_project/**` -> `src/ldm_project/AGENTS.md`
- `configs/**` -> `configs/AGENTS.md`
- `tests/**` -> `tests/AGENTS.md`
- Cross-scope work -> read every applicable local contract.

This explicit routing is required when Codex starts at the repository root, because child instructions are not assumed to be loaded automatically.

## 2. Execution Rules

1. Start with the answer or action; do not fabricate.
2. Resolve ambiguity by reading the local source of truth. Ask only when different interpretations materially change the result.
3. Touch only requested scope, follow existing patterns, and prefer the smallest correct change.
4. Before editing, state the intended change and inspect only the necessary files and callers.
5. No silent fallback. State the blocker, fallback, and lost guarantee; label weaker CPU/smoke/readability substitutes `diagnostic-only`.
6. Missing required checkpoints, datasets, CUDA, hardware, or external resources are blockers, not permission to substitute another task.
7. Use focused searches and the smallest sufficient verification. Do not reread large logs or rerun fresh green checks without a relevant change.
8. If a valid acceptance gate fails, stop downstream training or promotion. If the gate input or harness is invalid, repair and revalidate the gate before drawing a model conclusion.
9. Do not claim success without verification. State exactly what could not be verified.

## 3. Superpowers Policy

- Superpowers is opt-in in this repository. Invoke a skill only when the user names `$skill-name` or explicitly asks to use Superpowers.
- Ordinary questions, small edits, and documentation changes must not trigger the full Superpowers workflow.
- A skill must not create worktrees, branches, commits, subagents, or long CUDA jobs unless the user explicitly requests that action.
- Root and scoped AGENTS contracts take precedence over skill workflow suggestions. Do not copy Superpowers workflow text into repository instructions.

## 4. Project Contract

This project distills an LDM teacher into a single-exposure optical generative student.

Final inference path: `noise/code + optional label -> generator_e -> amplitude SLM command -> encoder metasurface -> OLS/fiber -> decoder metasurface -> CMOS single exposure -> generated image`.

- LDM is teacher infrastructure for priors, targets, exports, and baselines.
- Final inference must not require diffusion denoising, iterative latent sampling, software-only VAE decoding, or inference-time optimization.
- Main flow: `teacher_first_stage/teacher_diffusion -> teacher_distill_export -> one_shot_distill -> one_shot_sample`.
- Direct MNIST `one_shot` is baseline/smoke only. Keep teacher, student, and hardware-calibration responsibilities separate.
- Design source: `docs/ldm_migration/single_shot_ldm_distillation.md`; stage commands: `docs/ldm_migration/stage_guide.md`.

## 5. Repository And Artifacts

- Python requirement comes from `pyproject.toml` (`>=3.11`). Conda env: `optical-generative-models`.
- Source: `src/ldm_project/`; tests: `tests/`; entrypoint: `train.py`; configs: `configs/`.
- Explicit CLI checkpoint paths win. Omitted teacher checkpoints resolve first from `outputs/final_pt/`.
- One-shot sampling requires an explicit student checkpoint unless a documented resolver exists.
- Diagnostic and pre-acceptance runs write to `/tmp`. Accepted/configured managed runs write to `outputs/runs/...`.
- Existing `outputs/` artifacts are immutable: do not edit, delete, or backfill them. Do not modify `data/raw/`, repository `.tmp/`, or `legacy/`.

## 6. Commands And Verification

- Prefix: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models &&`
- Build: `python -m compileall src tests train.py`
- Targeted baseline: `pytest tests/test_train_script.py tests/test_one_shot.py tests/test_repo_layout.py -q`
- Full: `pytest -q`; lint: `ruff check src tests *.py`; docs/meta: `git diff --check`.
- Run targeted checks during iteration. Run the full suite after targeted checks for shared/final changes; rerun only when a corrective edit can affect its result.
- Use CUDA for formal training and acceptance, CPU only for labeled smoke tests, and `auto` only for compatibility checks.
- Long CUDA and hardware jobs are normally prepared by Codex and launched by the user. If explicitly delegated, avoid duplicate evaluations and unnecessary polling.

## 7. Boundaries

- Do not hardcode interpreter, Conda, Windows, or machine-specific paths; do not mutate `sys.path`, use `pip install -e .`.
- Do not mix `legacy` optical VAE code into the LDM core package.
- Do not present teacher image quality as student hardware readiness or call iterative software inference single-exposure.
- Ask before an unapproved change to stages, public CLI, teacher/student boundaries, SLM control, metasurface trainability, OLS behavior, CMOS calibration, checkpoint policy, or managed output policy.
