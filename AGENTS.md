# AGENTS.md

- **Codex only**: In the first assistant message of each new task, the first line must be exactly `【AGENTS effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.
- Claude Code 请参见 [CLAUDE.md](CLAUDE.md) 获取 Claude 专用行为规则；项目合同以本文件和适用的 scoped `AGENTS.md` 为准。

Primary execution contract for this repository. This file overrides `docs/`; scoped files may add constraints but may not relax this contract.

## 1. Scope Router

Before changing a scoped path, read its local contract:

- `src/ldm_project/v1/**`, `src/ldm_project/v2/**`, and root compatibility facades -> `src/ldm_project/AGENTS.md`
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

For long optical-training or one-shot readability debugging, follow the **Experimental Debugging Decision Policy** in `docs/operations/stage_guide.md`.

## 3. Skill Policy

- Use matching built-in skills; do not install, enable, or depend on third-party skills or plugins for repository work.
- Skills must not create worktrees, branches, commits, subagents, long CUDA jobs, external calls, or broad rewrites unless the user explicitly requests that action.
- Root and scoped AGENTS contracts take precedence over skill workflow suggestions.

## 4. Project Contract

This repository contains two intentionally separate runtimes. v1 preserves the
LDM-teacher distillation line behind `train.py`; v2 is the current teacher-free
conditional optical-generation line behind `python -m ldm_project.v2`. v1
history remains reproducible, but it does not define the v2 training contract.

The current v2 inference path is `label + 8D code -> identity-anchored
electronic amplitude generator -> amplitude SLM command -> encoder metasurface
-> linear identity OLS -> decoder metasurface -> CMOS single exposure`. The
formal optical contract has no fiber; a future fiber variant must preserve the
complex-field interface and be separately validated. The frozen evaluator is a
training/acceptance instrument only and is not part of final optical inference.

Dependency direction is `optics -> (none)`, v1 teacher/student -> optics and
neutral modules, v2 conditional student -> optics and neutral evaluation, and
application -> the selected runtime. v2 current code must not import v1 teacher,
teacher targets, prototype basis, capacity artifacts, or diagnostics. Root shims
are compatibility facades only. Archived files are reference-only.

Formal optical-field artifacts must carry shape, pixel pitch, physical FOV, wavelength, upstream plane, propagation distance, and operation. Tensor equality alone is not physical-coordinate validation; generic complex resize is forbidden in the formal path.

The LightGen paper is an architecture reference only. Its exact parameters are
not silently substituted into the formal v2 contract. v68-v79, A2/B,
prototype/capacity, direct-template, and LightGen measurements belong in status
or archive documents and cannot authorize a current v2 run.

The formal D2NN boundary is `finite_crop`, meaning a propagation-after-crop ideal hard-aperture computation contract; it is not a claim that printed glass is opaque. `transparent_identity` is diagnostic-only: it models a zero-thickness phase surface with an identity exterior on a full propagation grid and must pass the v67.1 convergence gate before any paired diagnostic training.

- Neither runtime may require diffusion denoising, iterative latent sampling,
  software-only VAE decoding, or inference-time optimization at final inference.
- Direct MNIST `one_shot` is a v1 baseline/smoke. v2 conditional MNIST is a
  separate teacher-free training contract and must not be routed through v1.
- v1 design and stages are documented in `docs/architecture/v1_single_shot_distillation.md`;
  v2 current stages are documented in `docs/architecture/v2_runtime.md`.

## 5. Repository And Artifacts

- Python requirement comes from `pyproject.toml` (`>=3.11`). Conda env: `optical-generative-models`; canonical source: `src/ldm_project/v1/` and `src/ldm_project/v2/`; root modules are compatibility facades; tests: `tests/`; entrypoint: `train.py`; configs: `configs/`.
- Explicit CLI checkpoint paths win. Omitted teacher checkpoints resolve first from `outputs/final_pt/`; one-shot sampling requires an explicit student checkpoint unless a documented resolver exists.
- Generated runs write to `outputs/YYYYMMDD_<version>/runs/...`, with `<version>` derived from the run name or config filename and falling back to `unversioned`; new runs remain in an existing real prefixed version directory.
- Diagnostic and pre-acceptance runs use persistent versioned `outputs/` paths but remain `diagnostic-only`. Do not modify `outputs/final_pt/`, non-run root files under `outputs/`, `data/raw/`, repository `.tmp/`, or `legacy/`; `outputs/runs/` must not be reintroduced.

## 6. Commands And Verification

- Prefix: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models &&`
- Build: `python -m compileall src tests train.py`; targeted baseline: `pytest tests/student tests/teacher tests/application tests/diagnostics tests/test_one_shot.py tests/test_train_script.py tests/test_repo_layout.py -q`.
- Full: `pytest -q`; lint: `ruff check src tests *.py`; docs/meta: `git diff --check`.
- Run targeted checks during iteration; run the full suite after targeted checks for shared/final changes, and rerun only checks invalidated by a corrective edit.
- Use CUDA for formal training and acceptance, CPU only for labeled smoke tests, and `auto` only for compatibility checks. Codex prepares long CUDA/hardware jobs for the user unless explicitly delegated.

## 7. Boundaries

- Do not hardcode interpreter, Conda, Windows, or machine-specific paths; do not mutate `sys.path`; use `pip install -e .`.
- Do not mix `legacy` optical VAE code into the LDM core package, present teacher image quality as student hardware readiness, or call iterative software inference single-exposure.
- Ask before an unapproved change to stages, public CLI, teacher/student boundaries, SLM control, metasurface trainability, OLS behavior, CMOS calibration, checkpoint policy, or managed output policy.

## 8. Capacity-map Research Lifecycle

- `python -m ldm_project.v2 research capacity-map` is diagnostic-only. Its
  `research` failures close a single declared variant; only formal
  `acceptance` evidence is global fail-closed.
- Capacity-map evidence is immutable and content-hashed. A repaired harness
  must use an evidence revision; invalid-harness records are excluded from
  Pareto ranking and causal attribution.
- `docs/status/capacity_map_hypothesis_ledger.md` is the single current
  hypothesis ledger. Keep handoff notes to current status, blocker, and next
  command; move completed history to immutable evidence/report artifacts.
