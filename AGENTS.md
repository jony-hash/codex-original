# AGENTS.md

- Claude Code 请参见 [CLAUDE.md](CLAUDE.md) 获取 Claude 专用行为规则；项目合同以本文件和适用的 scoped `AGENTS.md` 为准。

Primary execution contract for this repository. This file overrides `docs/`; scoped files may add constraints but may not relax this contract.

## 1. Scope Router

Before changing a scoped path, read its local contract:

- `src/ldm_project/v1/**` -> read `src/ldm_project/AGENTS.md` and `src/ldm_project/v1/AGENTS.md`
- `src/ldm_project/v2/**` -> read `src/ldm_project/AGENTS.md` and `src/ldm_project/v2/AGENTS.md`
- root compatibility facades -> `src/ldm_project/AGENTS.md`
- `configs/**` -> `configs/AGENTS.md`
- `tests/**`, or repository verification work -> `tests/AGENTS.md`
- `experiments/**` -> `experiments/AGENTS.md`
- Cross-scope work -> read every applicable local contract.

This explicit routing is required when Codex starts at the repository root, because child instructions are not assumed to be loaded automatically.

## 2. Execution Rules

1. Start with the answer or action; do not fabricate.
2. Resolve ambiguity from the local source of truth; state material uncertainty and tradeoffs, and ask when interpretations materially change the result.
3. Make the smallest change that satisfies the request. Touch only required files, add no speculative behavior, and clean only problems introduced by the current change.
4. Missing required inputs, checkpoints, datasets, CUDA, hardware, external resources, or valid gates block the formal task. Do not substitute another path without explicit user approval; report the blocker, smallest unblock action, and lost guarantee. An approved weaker run is `diagnostic-only` and cannot satisfy acceptance.
5. Use focused searches and the smallest sufficient verification. Do not reread large logs or rerun fresh green checks without a relevant change.
6. A valid failed acceptance gate stops downstream training or promotion; repair and revalidate an invalid gate before drawing a model conclusion.
7. Do not claim success without verification. State exactly what could not be verified.

8. Governance artifacts must identify a real consumer, rejection decision, or
   compatibility contract. Test counts, line counts, digest counts, and archive
   entry counts are measurements only and cannot independently define success.

9. Before a medium- or high-risk modification, record its substantive
   consequence, real consumer or rejection decision, affected scope, preserved
   invariants, smallest adequate verification, and stop condition. Classify by
   behavioral impact: a local implementation fix that changes no public contract,
   runtime semantics, or scientific decision is medium risk even if it is under
   training or evaluation. High-risk changes are public interfaces, acceptance
   boundaries, training/evaluation semantics, optical or hardware behavior,
   checkpoint/output policy, or scientific claims. If no substantive loss or
   rejected decision exists, stop without modifying the repository. High-risk
   work requires a no-update preflight, independent reference or negative control,
   explicit gates, and fail-closed stops. If verification is unavailable, report
   the blocker and lost guarantee; do not silently downgrade the result.

### Long-Training Preflight And Counterfactual Gate

This rule applies to every repository training path, including v1, v2,
research, diagnostics, one-off scripts, and newly added trainers. A run is
long when it is expected to use at least 15 GPU-minutes or 30 wall-clock
minutes, or when its result will select a model/objective, support a scientific
claim, or authorize a downstream stage.

- Before starting a long run, execute a short real-update trial through the
  same entrypoint, model, loss, optimizer, data path, evaluator, device, and
  essential configuration. By default use three optimizer steps; use the
  smallest representative batch that exercises every registered trainable
  parameter group. Verify finite inputs/loss/gradients/parameters, a real
  update in every intended group, no update in frozen/evaluator parameters,
  expected tensor/physical semantics, and sufficient memory headroom.
- Before a comparative, objective-selection, or scientific run, also execute
  the smallest domain-appropriate counterfactual that should fail the claimed
  direction, such as shuffled labels/codes, a disabled control, a frozen
  objective term, or a deliberately mismatched lineage input. State the
  expected rejection before running it. If the negative control passes the
  intended gate, classify the harness as invalid and do not start the long run.
- When checkpoint recovery will be relied on, the short trial must include one
  interrupted/resumed segment and prove that it continues at the next batch
  with compatible model, optimizer, RNG, schedule, and history state. If this
  cannot be demonstrated, either run without claiming resumability or stop;
  never advertise or silently attempt an unverified resume.
- Keep the entire ordinary preflight within the smaller of five minutes and
  5% of the planned run time. For high-risk scientific or acceptance runs,
  the cap is the smaller of ten minutes and 10%. If even one representative
  update cannot fit the applicable cap, stop and report the compute blocker;
  do not replace it with a non-representative CPU or reduced-model success.
- Record the short-trial command, immutable inputs, elapsed time, peak memory,
  observed gates, counterfactual result, and the exact long command it
  authorizes. A source/config/data/optimizer/evaluator change that can affect
  these observations invalidates the authorization; documentation-only or
  unrelated changes do not.
- A valid short trial authorizes only the predeclared long run. It does not
  predict final model quality. A valid scientific failure still stops the
  registered branch, while an invalid harness is repaired and rechecked.
- Satisfy this rule with the existing trainer's smallest safe controls. Do not
  introduce a generic contract framework, new public CLI, or evidence schema
  solely to comply unless the user explicitly requests that engineering work.

### Minimal Implementation And Failure Semantics

- Trust documented internal types, framework guarantees, and established
  invariants. Validate user input, CLI/file/checkpoint/evidence inputs,
  external APIs, network, hardware, and other process boundaries. Inside
  trusted code, use direct access and let invariant violations fail immediately.
- Use fail-closed checks at real acceptance or external boundaries. Use
  fail-fast behavior for internal programming and execution errors.
- Do not catch broad exceptions or convert execution, configuration, or
  programming errors into scientific/model gate failures. A failed scientific
  evidence record requires a complete, valid measurement whose explicit gate
  failed.
- Do not add test-only production branches, `hasattr`/`getattr` compatibility
  fallbacks, silent defaults, null substitutes, automatic latest-output
  resolution, or impossible-state recovery solely to keep callers or tests
  working.
- Do not create a helper, registry, schema, manifest, wrapper, or compatibility
  layer for one call site unless it represents an existing public boundary.
- Add a digest only when a mutable artifact crosses a process or run boundary
  and a named consumer compares it to make a rejection decision. Compute it
  once per operation. Do not add digest inventories, digest-count gates,
  duplicate bindings, nested hash-of-hashes, or speculative component
  fingerprints.
- Tests must exercise the production contract; production code must not
  contain a weaker path solely for tests. For one rejection mechanism, prefer
  one valid case and one representative negative case unless different inputs
  cause materially different decisions.

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

Diagnostic variants, LightGen comparisons, and historical experiment numbers
are status or archive material only; they cannot authorize a current v2 run.

The formal D2NN boundary is `finite_crop`, meaning a propagation-after-crop ideal hard-aperture computation contract; it is not a claim that printed glass is opaque. `transparent_identity` is diagnostic-only: it models a zero-thickness phase surface with an identity exterior on a full propagation grid and requires its applicable convergence gate before paired diagnostic training.

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

- Verification commands for repository tests live in `tests/AGENTS.md`.
- Run targeted checks during iteration; run the full suite after targeted checks
  for shared/final changes, and rerun only checks invalidated by a corrective
  edit.
- Use CUDA for formal training and acceptance, CPU only for labeled smoke tests, and `auto` only for compatibility checks. Codex prepares long CUDA/hardware jobs for the user unless explicitly delegated.

## 7. Boundaries

- Do not hardcode interpreter, Conda, Windows, or machine-specific paths; do not mutate `sys.path`; use `pip install -e .`.
- Do not mix `legacy` optical VAE code into the LDM core package, present teacher image quality as student hardware readiness, or call iterative software inference single-exposure.
- Ask before an unapproved change to stages, public CLI, teacher/student boundaries, SLM control, metasurface trainability, OLS behavior, CMOS calibration, checkpoint policy, or managed output policy.

## 8. Capacity-map Research Lifecycle

Capacity-map lifecycle rules and the current hypothesis ledger live in
`docs/status/capacity_map_hypothesis_ledger.md`; research failures remain
variant-local and cannot authorize formal acceptance.
