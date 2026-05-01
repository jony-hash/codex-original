# AGENTS.md

- In the first assistant message of each new task, the first line must be exactly `【AGENTS effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.

Primary execution contract for this repo. If this file conflicts with any document under `docs/`, follow this file.

## 1. Non-negotiables

1. No filler. Start with the answer or the action.
2. Never fabricate. Read files, run commands, or say you do not know.
3. Ask only when ambiguity materially changes the output.
4. Touch only the requested scope. No unrelated refactors or cleanup.
5. Do not claim success without verification.

## 2. Working Rules

- Briefly state the plan before editing.
- Read the files you will change and the code that calls them.
- Match existing project patterns unless the user explicitly asks to replace them.
- Prefer the smallest change that solves the stated problem.
- Surface assumptions explicitly.
- When generating or modifying code, add concise Chinese comments where they improve readability or explain non-obvious logic. Do not add redundant comments for obvious code.

## 3. Verification

- Define success in a verifiable way before changing code.
- Prefer running a single test, smoke script, or single-file check during iteration.
- Use full-suite verification only for the final pass.
- Fix root causes, not symptoms.

## 4. Project Context

- Python 3.11
- Stack: PyTorch, torchvision, PyYAML, matplotlib
- Environment: WSL2 Linux shell, Conda + pip, VS Code Remote / terminal friendly
- Default Conda environment: `optical-generative-models`
- Verified PyTorch baseline: CUDA build `torch 2.5.1+cu121`
- Training contract: activate `optical-generative-models` first, then run training with explicit `--device cuda`
- Commands must be bash-compatible inside WSL2; do not use PowerShell or CMD syntax
- Imports must use `pip install -e .`; do not mutate `sys.path`
- Source: `src/ldm_project/`
- Tests: `tests/`
- Entry scripts: `train.py`
- Detailed design docs: `docs/ldm_migration/`
- Do not modify: `outputs/`, `data/raw/`, `.tmp/`, `legacy/`, generated checkpoints, generated plots, cached artifacts
- New `legacy` references must stay outside the LDM core package

## 5. Commands

- Install: `conda create -n optical-generative-models python=3.11 -y && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && pip install -r requirements.txt && pip install -e .`
- Build: `python -m compileall src tests`
- Test all: `pytest -q`
- Test single file: `pytest tests/test_scheduler.py -q`
- Lint: `ruff check src tests *.py`
- Typecheck: `mypy src`
- Run first stage: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage first_stage --device cuda`
- Run diffusion stage: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage diffusion --device cuda --first-stage-checkpoint outputs/runs/.../checkpoints/first_stage_latest.pt`
- Run full pipeline: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && python train.py --stage full --device cuda`

Use `--device auto` only for compatibility checks or temporary debugging. Use `--device cpu` only for local smoke tests; it is not the default training path.

Prefer single-test or single-file verification during iteration. Use the full suite only for the final pass.

## 6. Forbidden

- Do not hardcode interpreter paths, Conda paths, Windows paths, or machine-specific directories.
- Do not rely on editor settings as the only runtime configuration source.
- Do not write outputs or checkpoints outside managed `outputs/runs/...` directories.
- Do not import project modules by mutating `sys.path`.
- Do not mix `legacy` optical VAE code directly into the new LDM core package.
- Do not use full-dataset or full-epoch runs for routine iteration.
- Do not use PowerShell-only or Windows-only shell commands in examples, scripts, or instructions unless the user explicitly asks for them.

## 7. When To Ask

Ask before proceeding when:

- there are two plausible interpretations with materially different outputs
- the change affects load-bearing paths, migration boundaries, or public interfaces
- the user's stated goal conflicts with the literal request
- a secret, credential, or inaccessible external resource is required

Proceed without asking when:

- the task is trivial and reversible
- the ambiguity can be resolved by reading code or running a command
- the user already answered the same question in the current session

## 8. Project Learnings

- Use simple Python one-liners for file reads when needed; avoid shell-specific file-reading tricks that are fragile across terminals.
- Every new terminal session must activate the `optical-generative-models` Conda environment inside WSL2 before running project commands.
- Every new terminal session should verify `torch.cuda.is_available()` before formal training; if it is `False`, first debug WSL2 / driver / CUDA visibility instead of silently falling back to CPU long runs.
- In Codex, sandboxed checks may report `torch.cuda.is_available() == False` or NVML errors even when the user's Anaconda training environment has working CUDA. Treat sandbox results as sandbox visibility only, not as the final environment verdict. Before telling the user CUDA is unavailable, re-check in the same Conda environment with a non-sandbox command and state clearly whether the issue is sandbox-only or a real environment problem.
- Prefer bash-compatible commands and Linux paths in examples, scripts, and debugging steps.
- Long CUDA training or other long-running experiments should be prepared by Codex but launched by the user. Codex should provide the exact command, expected output directory pattern, and post-run verification script/checklist instead of keeping the chat session open for the whole training run, unless the user explicitly asks Codex to run it.
