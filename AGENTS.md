# AGENTS.md

- **Codex only**: In the first assistant message of each new task, the first line must be exactly `【AGENTS effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.
- Claude Code 请参见 [CLAUDE.md](CLAUDE.md) 获取 Claude 专用行为规则；项目合同以本文件和适用的 scoped `AGENTS.md` 为准。

Primary execution contract for this repository. This file overrides `docs/`; scoped files may add constraints but may not relax this contract.

## 1. Scope Router

Before changing a scoped path, read its local contract:

- `src/ldm_project/**` -> `src/ldm_project/AGENTS.md`
- `configs/**` -> `configs/AGENTS.md`
- `tests/**`, or repository verification work -> `tests/AGENTS.md`
- `experiments/**` -> `experiments/AGENTS.md`
- Cross-scope work -> read every applicable local contract.

This explicit routing is required when Codex starts at the repository root, because child instructions are not assumed to be loaded automatically.

## 2. Execution Rules

1. Start with the answer or action; do not fabricate.
2. Resolve ambiguity from the local source of truth; ask only when reasonable interpretations materially change the result.
3. Touch only requested scope, follow existing patterns, and inspect necessary files and callers before editing.
4. Missing required inputs, checkpoints, datasets, CUDA, hardware, external resources, or valid gates block the formal task. Do not substitute another path without explicit user approval; report the blocker, smallest unblock action, and lost guarantee. An approved weaker run is `diagnostic-only` and cannot satisfy acceptance.
5. Use focused searches and the smallest sufficient verification. Do not reread large logs or rerun fresh green checks without a relevant change.
6. A valid failed acceptance gate stops downstream training or promotion; repair and revalidate an invalid gate before drawing a model conclusion.
7. Do not claim success without verification. State exactly what could not be verified.

For long optical-training or one-shot readability debugging, follow the **Experimental Debugging Decision Policy** in `docs/ldm_migration/stage_guide.md`.

## 3. Skill Policy

- Use matching built-in skills; do not install, enable, or depend on third-party skills or plugins for repository work.
- Skills must not create worktrees, branches, commits, subagents, long CUDA jobs, external calls, or broad rewrites unless the user explicitly requests that action.
- Root and scoped AGENTS contracts take precedence over skill workflow suggestions.

## 4. Project Contract

This project distills an LDM teacher into a single-exposure optical generative student.

Final inference path: `noise/code + optional label -> generator_e -> amplitude SLM command -> encoder metasurface -> linear identity OLS -> decoder metasurface -> CMOS single exposure -> generated image`. The current formal contract has no fiber; a future fiber variant must preserve the complex-field interface and be separately validated.

Dependency direction is `optics -> (none)`, `teacher/student -> neutral generation/distillation/evaluation`, and `application -> teacher/student`. The formal student entry is `FormalOneShotOpticalGenerator`; compatibility variants and root shims must not be imported by formal code. `distillation/targets.py` is neutral and must not import student. Archived files are reference-only.

Formal optical-field artifacts must carry shape, pixel pitch, physical FOV, wavelength, upstream plane, propagation distance, and operation. Tensor equality alone is not physical-coordinate validation; generic complex resize is forbidden in the formal path.

- LDM is teacher infrastructure; final inference must not require diffusion denoising, iterative latent sampling, software-only VAE decoding, or inference-time optimization.
- Direct MNIST `one_shot` is baseline/smoke only. Keep teacher, student, and hardware-calibration responsibilities separate.
- Design source: `docs/ldm_migration/single_shot_ldm_distillation.md`; stage commands and debugging policy: `docs/ldm_migration/stage_guide.md`.

## 5. Repository And Artifacts

- Python requirement comes from `pyproject.toml` (`>=3.11`). Conda env: `optical-generative-models`; source: `src/ldm_project/`; tests: `tests/`; entrypoint: `train.py`; configs: `configs/`.
- Explicit CLI checkpoint paths win. Omitted teacher checkpoints resolve first from `outputs/final_pt/`; one-shot sampling requires an explicit student checkpoint unless a documented resolver exists.
- Generated runs write to `outputs/YYYYMMDD_<version>/runs/...`, with `<version>` derived from the run name or config filename and falling back to `unversioned`; new runs remain in an existing real prefixed version directory.
- Diagnostic and pre-acceptance runs use persistent versioned `outputs/` paths but remain `diagnostic-only`. Do not modify `outputs/final_pt/`, non-run root files under `outputs/`, `data/raw/`, repository `.tmp/`, or `legacy/`; `outputs/runs/` must not be reintroduced.

## 6. Commands And Verification

- Prefix: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models &&`
- Build: `python -m compileall src tests train.py`; targeted baseline: `pytest tests/student tests/application tests/diagnostics tests/test_one_shot.py tests/test_train_script.py tests/test_repo_layout.py -q`.
- Full: `pytest -q`; lint: `ruff check src tests *.py`; docs/meta: `git diff --check`.
- Run targeted checks during iteration; run the full suite after targeted checks for shared/final changes, and rerun only checks invalidated by a corrective edit.
- Use CUDA for formal training and acceptance, CPU only for labeled smoke tests, and `auto` only for compatibility checks. Codex prepares long CUDA/hardware jobs for the user unless explicitly delegated.

## 7. Boundaries

- Do not hardcode interpreter, Conda, Windows, or machine-specific paths; do not mutate `sys.path`; use `pip install -e .`.
- Do not mix `legacy` optical VAE code into the LDM core package, present teacher image quality as student hardware readiness, or call iterative software inference single-exposure.
- Ask before an unapproved change to stages, public CLI, teacher/student boundaries, SLM control, metasurface trainability, OLS behavior, CMOS calibration, checkpoint policy, or managed output policy.
