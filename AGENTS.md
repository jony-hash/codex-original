# AGENTS.md

- Claude Code 请参见 [CLAUDE.md](CLAUDE.md) 获取 Claude 专用行为规则；项目合同以本文件和适用的 scoped `AGENTS.md` 为准。

Primary execution contract for this repository. This file overrides `docs/`; scoped files may add constraints but may not relax this contract.

## Current LightGen Research Policy (2026-10-01)

The user's simplification instruction supersedes older research timing, hash-lock,
and repair-quota rules in scoped contracts, plans, and documentation.

- No elapsed-time upper limit applies to preflight, training, measurement,
  recovery, or finalization. Record durations and estimates as observations;
  keep the approved update counts, populations, and scientific thresholds.
- Do not require hash/digest, source-manifest, or dirty-patch equality for
  admission or resume. Use explicit input paths and validate the loaded model,
  configuration, checkpoint step, optimizer state, sample rows, and data roles.
  Preserve historical digest fields without using them as execution locks.
- Keep one final scientific decision. Ordinary engineering repairs stay within
  the authorized task; investigate repeated defects and recheck the affected
  production path without imposing a retry quota or requesting another token.
- Retain numerical validity, evaluator isolation, frozen/update scope, actual
  hardware safety, checkpoint recovery, and protection of existing artifacts.
- GPU memory readings are observations, not fixed free-memory or utilization
  quotas. Verify that the representative production preflight completes; actual
  CUDA unavailability or allocation failure is an engineering error.
- Keep plans short: question, inputs, action, final measurement, and stop point.
  This policy does not authorize new scientific updates or successor experiments.
- Reserve human review for decision-changing milestones such as adopting a model,
  validating sampled generation, physical deployment, or formal readability
  acceptance. Routine engineering fixes, local diagnostics, and intermediate
  training points do not trigger review by default; each important review must
  set its sample size and decision use in advance rather than inheriting a prior
  large review package.

## 1. Scope Router

Before changing a scoped path, read its local contract: v1 paths require
`src/ldm_project/AGENTS.md` and `src/ldm_project/v1/AGENTS.md`; v2 paths require
the parent and `v2/AGENTS.md`; root facades require the parent; `configs/`,
`tests/` or verification, and `experiments/` require their respective contracts.
Cross-scope work reads every applicable contract.

This explicit routing is required when Codex starts at the repository root, because child instructions are not assumed to be loaded automatically.

## 2. Execution Rules

1. Start with the answer or action; do not fabricate. Resolve ambiguity from the
   local source of truth, state material uncertainty or tradeoffs, and ask when
   the interpretation materially changes the result.
2. Make the smallest scoped change: touch only required files, add no speculative
   behavior, and clean only problems introduced by the current change.
3. Missing inputs, checkpoints, data, hardware, resources or valid gates block the
   formal task. Do not substitute a weaker path without approval; report the
   smallest unblock action and lost guarantee. An approved weaker run is
   `diagnostic-only` and cannot satisfy acceptance.
4. Use focused searches and the smallest sufficient verification; do not reread
   large logs or rerun unchanged green checks. A valid failed gate blocks downstream
   training/promotion; repair invalid harnesses before drawing conclusions.
5. Never claim success without verification; state what remains unverified.
   Governance artifacts need a real consumer, rejection decision or compatibility
   contract; counts and digests are measurements, not success criteria.
6. Classify risk by behavioral impact, not file location. For medium/high-risk
   changes record consequence, consumer, scope, invariants, verification and stop
   condition. High-risk work also needs a no-update preflight, independent
   reference or negative control, explicit gates and fail-closed stops; with no
   substantive loss or rejected decision, do not modify the repository.
   If verification is unavailable, report the blocker and lost guarantee; never
   silently downgrade the result.
7. Before finalizing a plan or writing consequential code, verify the quantity
   and its consumer against the production path. Ask: What exact quantity is
   being measured? Does its implementation include every term in the definition?
   What independent counterexample could falsify it? Does the final decision
   consume the verified production result? For numerical or gradient claims,
   check values, derivatives, and decision signs separately; use an analytic,
   finite-difference, or otherwise independent reference rather than a copy of
   the implementation. Regression tests must exercise the production producer
   through the decision that consumes its output; tests of hand-supplied values
   alone do not establish measurement validity. A check marked `checked`,
   `passed`, or `READY` must come from an executed check, and the readiness
   consumer must require that result. Prove that a regression fails against the
   faulty behavior and passes after the fix, then run a representative
   production preflight before the full measurement. Scale checks to the risk;
   do not add a generic framework, extra scientific gate, digest lock, or
   unrelated reruns.
8. Plan-driven or cross-product work follows the existing
   [Delta + One Gate + Short Codex Prompt](docs/operations/delta_one_gate_short_prompt.md):
   keep one scientific decision, verify the required inputs, run the relevant
   preflight without an elapsed-time cap, execute only the named action, and
   return acceptance or stop results. Its documentation write-set rule is mandatory: machine evidence
   does not by itself authorize edits to every memory/status entry. This is
   provenance control, not a second scientific gate.
   Research plans must explain how each possible result changes the decision.
   Research and plan-driven engineering closeout reports must use that guide's
   fixed five-part feedback order and explicitly state scientific learning,
   the remaining goal gap, and the boundary between advice and authorization.
   Recheck the relevant guide sections when preparing a plan or closeout,
   including changes made during an ongoing conversation; before delivery,
   verify the report's meaning against the evidence, not just file/test checks.
   Brief factual answers and routine progress updates do not require this format.

### Turn Opening Compliance

At the beginning of every repository task that may inspect, modify, or run code,
write exactly `Agent effective: AGENTS.md`; do not expand it into a scope or tier
inventory. For LightGen research, keep digest fields as provenance and do not
use them to block admission or resume. Any retained digest needs a named
reporting consumer; a digest alone is never a success criterion.

### Long-Training Gate

Every training path uses the smallest safe preflight before a run of at least 15
GPU-minutes, 30 wall-clock minutes, or one that selects a model, supports a claim
or authorizes a downstream stage. Run static/lineage checks, a representative
three-step real update through the same entrypoint, and a domain-appropriate
negative control when direction matters. The update must exercise declared
trainable groups and frozen/evaluator isolation; reduced surrogates and
automatic-latest shortcuts are invalid. A failed preflight is `invalid_harness`;
an explicit scientific gate failure is `valid_failed` and blocks promotion.
Use [long-training-preflight.md](docs/operations/long-training-preflight.md) for
startup checks, evidence, checkpoint recovery and counterfactual details.

### Ordinary Research Fast Path

Ordinary `R0/R1/R2` research uses one Outer Gate and one authorization carried
from `READY` preflight into execution. Authority, lineage, runtime/data roles,
tiny updates, checkpoint reload and similar `CHECK_*`/`INVARIANT_*` items are
internal checks, not additional gates; ordinary engineering failures are fixed
and rechecked under the same task identity. Historical gate/status artifacts
are consumed only when explicitly named as `INPUT_AUTHORITIES`. The shared
heartbeat, progress, recovery and resume contract is defined in
[delta_one_gate_short_prompt.md](docs/operations/delta_one_gate_short_prompt.md);
this does not relax formal/high-risk or long-training rules above.

### Minimal Implementation And Failure Semantics

Validate user input, CLI/file/checkpoint/evidence inputs and other process
boundaries; trust documented internal invariants. Use fail-closed checks at
acceptance/external boundaries and fail-fast behavior for internal errors. Only a
complete, valid measurement may produce a scientific gate failure. Do not turn
broad execution/configuration/programming exceptions into scientific gates,
or add test-only branches, compatibility fallbacks, silent defaults,
automatic-latest resolution or impossible-state recovery. Do not create one-call
helpers/registries/schemas/wrappers without a public boundary. Do not add digest
inventories or hash-based execution locks to research tasks.
Tests exercise the production contract; prefer one valid and one representative
negative case per rejection mechanism unless inputs cause different decisions.

For long optical-training or one-shot readability debugging, follow the **Experimental Debugging Decision Policy** in `docs/operations/stage_guide.md`.

## 3. Skill Policy

Use matching built-in skills; do not install, enable or depend on third-party skills
or plugins for repository work. Skills may not create worktrees, branches, commits,
subagents, long CUDA jobs, external calls or broad rewrites unless explicitly
requested. Root and scoped AGENTS contracts take precedence over skill suggestions.

## 4. Project Contract

Two intentionally separate runtimes are maintained: v1 preserves the LDM-teacher
line behind `train.py`; v2 is the teacher-free conditional runtime behind
`python -m ldm_project.v2`. v1 history does not define v2. Canonical stages and
boundaries live in [v1_single_shot_distillation.md](docs/architecture/v1_single_shot_distillation.md)
and [v2_runtime.md](docs/architecture/v2_runtime.md).

The v2 formal chain is `label + 8D code -> amplitude SLM -> encoder metasurface
-> identity OLS -> decoder metasurface -> CMOS single exposure`; it has no fiber,
and its frozen evaluator is not final optical inference. Keep lower-level optics/
neutral modules below the selected runtime and application; v2 must not import v1
teacher/target or prototype/capacity/diagnostic artifacts, and root shims are
compatibility facades.

Formal field artifacts require physical-coordinate metadata; tensor equality is
insufficient and generic complex resize is forbidden. `finite_crop` is the formal
D2NN boundary; `transparent_identity`, LightGen and historical variants are
diagnostic/reference only. Neither runtime may require diffusion denoising,
iterative latent sampling, software-only VAE decoding or inference-time optimization.

## 5. Repository And Artifacts

Python is `>=3.11` from `pyproject.toml`; the supported Windows environment is
`D:\ANACONDA\envs\optical-generative-models-win`. Canonical source is
`src/ldm_project/v1/` and `src/ldm_project/v2/`; root modules are facades,
tests are in `tests/`, and `train.py` is the entrypoint. Explicit CLI checkpoint
paths win; omitted teachers resolve first from `outputs/final_pt/`, while one-shot
sampling needs an explicit student unless a documented resolver exists.

New runs use `outputs/YYYYMMDD_<version>/runs/...`; diagnostic and pre-acceptance
runs remain `diagnostic-only`, and promotion requires a named consumer. Never
modify `outputs/final_pt/`, non-run roots, `data/raw/`, `.tmp/` or legacy
namespaces; never restore `outputs/runs/`.
See [output_artifact_notes.md](docs/operations/output_artifact_notes.md).

## 6. Commands And Verification

Verification commands live in `tests/AGENTS.md`. Run targeted checks during
iteration and the full suite for shared/final changes, rerunning only invalidated
checks. Use CUDA for formal training/acceptance, CPU only for labeled smoke and
`auto` only for compatibility; Codex prepares long hardware jobs unless delegated.

## 7. Boundaries

Do not hardcode interpreter/Conda/machine paths in runtime code or mutate `sys.path`;
use `pip install -e .`. Do not mix legacy optical VAE code into LDM core, present
teacher quality as hardware readiness, or call iterative software inference
single-exposure. Ask before changing stages, public CLI, teacher/student boundaries,
SLM/metasurface behavior, OLS/CMOS calibration, checkpoint or output policy.

## 8. Capacity-map Research Lifecycle

Lifecycle labels and the current hypothesis ledger live in
`docs/status/capacity_map_hypothesis_ledger.md`; variant failures never authorize
formal acceptance.

## Windows-native migration operating rules

Use the supported Windows-native environment for development, tests, training
and Git; runtime code remains path-portable. Exact setup, lock installation,
verification commands, and migration details live in [Windows development](docs/operations/windows_development.md)
and [Windows-native migration](docs/migration_windows_native_20260830.md).

`outputs/` combines Windows runs with restored historical views. Restore membership
and Linux-to-Windows mappings come only from
`outputs/_wsl_restore_metadata_v4/{restore-plan.json,restore-state.json,path-map.jsonl}`.
Restored entries, metadata, indexes and external archives are immutable and never
auto-resolved as current artifacts; `version_index` is provenance only. New runs
need a distinct non-restored child path; see [output_artifact_notes.md](docs/operations/output_artifact_notes.md).
The AutoDL Remote-SSH, `screen`, status, and explicit-result-backup workflow is documented in [AutoDL workflow](docs/operations/autodl_workflow.md).

### Fixed cloud execution and Codex network permissions (2026-10-14)

- The user has selected `autodl-monthly` as the execution host for subsequent
  training and long numerical validation. Windows remains the source, preparation,
  review and backup authority. This overrides the Windows-local training default
  above; each scientific run still needs its authorized scope and valid preflight.
- In Codex, run authorized AutoDL SSH/SCP operations, including `autodl_run.ps1`
  and `autodl_pull.ps1`, with `exec_command.sandbox_permissions=require_escalated`.
  On this host, the identical environment check failed with SSH DNS error 255
  inside the restricted sandbox and succeeded outside it on 2026-10-14.
- A sandbox DNS/proxy error is not evidence of a cloud-server DNS failure or a
  missing server. Verify the same read-only operation outside the sandbox before
  asking the user to repair SSH/DNS or supply another endpoint. Do not change DNS,
  SSH host keys, credentials or provider endpoint to work around sandbox isolation.
- Existing user authorization for the named cloud task carries through routine
  connection checks, screen execution and exact-file transfers. If automatic
  approval review rejects external execution, report that rejection distinctly;
  do not mislabel it as server/network failure or silently execute locally.

Keep `docs/CURRENT_STATUS.md` current, preserve the D: safety floor, and change
the migration record only for migration, environment, restore or bridge events.

## Codex history and project memory operating rules

Before a cross-product task, read `docs/PROJECT_MEMORY.md`; local Git, source and
evidence outrank Sources. Historical WSL paths are never current roots, and the
archive/SQLite must not be edited. Snapshot-first migration and old-path handling
live in [Codex history migration](docs/codex_history_migration.md) and
[Windows-native migration](docs/migration_windows_native_20260830.md).

Bridge refresh/status and the controlled six-file Sources contract live in
[chatgpt_codex_workflow.md](docs/chatgpt_codex_workflow.md). Cloud and local projects
remain separate; Codex writes authoritative memory only after verification, and no
automatic upload, database/index edit or conversion of discussion into
training/promotion authority is permitted.
