# AGENTS.md

Claude Code: see [CLAUDE.md](CLAUDE.md). This file is the primary execution contract
and overrides `docs/`; scoped files may add constraints but never relax it.

## 1. Scope Router

Read the applicable local contract before changing a scoped path: `src/ldm_project/`
(parent), `v1/` and `v2/` (their own), `configs/`, `tests/`, `experiments/`,
`scripts/research/`, `docs/`. Root facades use the parent contract. Cross-scope work
reads every applicable file; child instructions are not assumed to load automatically.

At the beginning of every repository task that may inspect, modify, or run code, write
exactly `Agent effective: AGENTS.md`; do not expand it into a scope or tier inventory.

## 2. Write And Commit

- One writer at a time: report the uncommitted count and confirm no other session is
  writing before modifying source, tests, configs, or committing.
- Default writer and sole `git add/commit/push` authority is Codex; DSH is read-only
  unless the user explicitly authorizes otherwise. AutoDL SSH, `screen`, upload and
  cleanup run only through Codex.
- New run directories carry the session origin; mixed-origin runs are not inputs.

## 3. Non-Negotiables

- Never modify `outputs/final_pt/`, non-run `outputs/` roots, `data/raw/`, `.tmp/`,
  legacy namespaces, or the D: safety floor; never restore `outputs/runs/`.
- No hash/digest, manifest, or dirty-patch equality for admission or resume. Validate
  explicit inputs, loaded model/config/step/optimizer state, sample rows and data
  roles; digests are provenance only.
- Preserve numerical validity, evaluator isolation, frozen/update scope, hardware
  safety, checkpoint recovery, and existing artifacts. GPU memory readings are
  observations, not quotas; CUDA unavailability is an engineering error.
- One final scientific decision per task; no retry quota, no elapsed-time cap, no
  automatic successor authorization.
- Human review only at decision-changing milestones (model adoption, sampled
  generation, physical deployment, formal readability acceptance).

## 4. Execution Rules

1. Act or answer first; do not fabricate. Resolve ambiguity from the local source of
   truth; state material uncertainty and ask when it changes the result.
2. Smallest scoped change; no speculative behavior; clean only what you introduced.
3. Missing inputs, checkpoints, data, hardware, or valid gates block the formal task.
   Do not silently downgrade; report the smallest unblock and lost guarantee. An
   approved weaker run is `diagnostic-only` and cannot satisfy acceptance.
4. Smallest sufficient verification; do not rerun unchanged green checks. A valid
   failed gate blocks downstream promotion; an invalid harness is repaired, not
   interpreted.
5. No success claim without an executed check. State what remains unverified; a
   `checked`/`passed`/`READY` label must come from the check and be required by its
   consumer.
6. Risk follows behavioral impact, not location. Medium/high-risk changes record
   consequence, consumer, scope, invariants, verification and stop condition;
   high-risk work adds a no-update preflight, independent reference or negative
   control, and fail-closed stops.
7. Verify quantity and consumer against the production path: exact definition, every
   term implemented, an independent counterexample, and a decision that consumes the
   production result. Prove a regression fails before the fix and passes after.
8. Plan-driven work follows [Delta + One Gate](docs/operations/delta_one_gate_short_prompt.md):
   one decision, verified inputs, no time cap, only the named action, acceptance or
   stop. Plans stay short and state how each result changes the decision; closeouts
   use the five-part order and separate advice from authorization. Machine evidence
   does not authorize edits to every memory/status entry.

## 5. Failure Semantics

Validate process boundaries (CLI, file, checkpoint, evidence); trust documented
internal invariants. Fail closed at acceptance boundaries, fail fast internally. Only
a complete, valid measurement may produce a scientific gate failure. Do not turn
execution/configuration exceptions into gates, add test-only branches, compatibility
fallbacks, silent defaults, automatic-latest resolution, or one-call wrappers without
a public boundary. One valid and one representative negative case per rejection path.
For long optical training or readability debugging follow the **Experimental Debugging
Decision Policy** in [stage_guide.md](docs/operations/stage_guide.md).

### Literature And Evidence

Recent primary sources only (current year −2; 2026 → 2024–2026) from the listed venues;
bind each claim to paper/DOI and sample/process/wavelength/quantity/units; label proxy
models as proxies; reviews are not primary evidence. See
[literature_quality.md](docs/operations/literature_quality.md).

### Long Training

Before ≥15 GPU-min or ≥30 wall-min, model selection, a claim, or a downstream stage:
static/lineage checks, a three-step real update through the same entrypoint, and a
negative control when direction matters. See
[long-training-preflight.md](docs/operations/long-training-preflight.md).

### Ordinary Research Fast Path

One Outer Gate and one authorization carried from `READY` into execution; other
`CHECK_*`/`INVARIANT_*` items are internal checks. Historical gates are consumed only
as named `INPUT_AUTHORITIES`. Capacity-map labels live in
[capacity_map_hypothesis_ledger.md](docs/status/capacity_map_hypothesis_ledger.md).

## 6. Project Contract

v1 (LDM teacher distillation, `train.py`) and v2 (teacher-free conditional,
`python -m ldm_project.v2`) are separate; v1 history does not define v2. Contracts:
[v1](docs/architecture/v1_single_shot_distillation.md),
[v2](docs/architecture/v2_runtime.md). The v2 chain is `label + 8D code -> amplitude
SLM -> encoder metasurface -> identity OLS -> decoder metasurface -> CMOS single
exposure`; no fiber, and the frozen evaluator is not inference. v2 must not import v1
teacher/target or prototype/capacity/diagnostic artifacts; root shims are facades.

Formal fields carry physical coordinates; tensor-shape equality does not justify a
resize. `finite_crop` is the formal D2NN boundary; `transparent_identity`, LightGen and
historical variants are diagnostic/reference only. No diffusion denoising, iterative
latent sampling, software-only VAE decode, or inference-time optimization.

## 7. Repository And Artifacts

Python `>=3.11`; supported Windows environment
`D:\ANACONDA\envs\optical-generative-models-win`. Canonical source
`src/ldm_project/{v1,v2}/`; tests in `tests/`; `train.py` is the v1 entrypoint.
Explicit checkpoint paths win; omitted teachers resolve from `outputs/final_pt/`;
one-shot sampling needs an explicit student.

New runs: `outputs/YYYYMMDD_<version>/runs/...`; pre-acceptance runs are
`diagnostic-only`; promotion needs a named consumer
([output_artifact_notes.md](docs/operations/output_artifact_notes.md)). Non-run output
(assets, reports) stays under the same dated version directory; no new non-dated
top-level output directories. Restore
membership comes only from `outputs/_wsl_restore_metadata_v4/`; restored entries are
immutable and never auto-resolved.

## 8. Commands And Verification

Single entry: `python scripts/run_windows_cpu_checks.py` (`--full` for the whole
`tests/` tree). Targeted checks during iteration; full suite for shared changes. CUDA
for formal training/acceptance; CPU only for labeled smoke; `auto` only for
compatibility. See [tests/AGENTS.md](tests/AGENTS.md) and
[windows_development.md](docs/operations/windows_development.md).

## 9. Boundaries

Do not hardcode interpreter/Conda/machine paths in runtime code or mutate `sys.path`;
use `pip install -e .`. Do not mix legacy optical VAE into LDM core, present teacher
quality as hardware readiness, or call iterative software inference single-exposure.
Ask before changing stages, public CLI, teacher/student boundaries, SLM/metasurface
behavior, OLS/CMOS calibration, checkpoint or output policy.

## 10. Skills, Routing, Cloud, Memory

Use built-in skills; no third-party skills/plugins for repository work. Skills may not
create worktrees, branches, commits, subagents, long CUDA jobs, external calls or broad
rewrites unless requested. Project skills under `~/.codex/memories/skills/` are
read-only on demand. Model routing: GPT-6.1 Sol plans, judges science and reviews;
GPT-6 Luna implements within authorization; high-risk changes need review separated
from implementation. See the [Delta guide](docs/operations/delta_one_gate_short_prompt.md).

`autodl-monthly` is the training host; Windows remains source, preparation, review and
backup authority. Run authorized AutoDL SSH/SCP outside the restricted sandbox; a
sandbox DNS error is not server evidence ([AutoDL workflow](docs/operations/autodl_workflow.md)).
Use the supported Windows-native environment; runtime code stays path-portable; change
the migration record only for migration/environment/restore/bridge events
([Windows development](docs/operations/windows_development.md),
[migration](docs/operations/migration_windows_native_20260830.md)).

Before cross-product work read [PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md); local Git,
source and evidence outrank external Sources. The archive/SQLite is read-only;
historical WSL paths are never current roots
([Codex history migration](docs/operations/codex_history_migration.md),
[ChatGPT ↔ Codex workflow](docs/operations/chatgpt_codex_workflow.md)).