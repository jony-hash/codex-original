# AGENTS.md

- **Codex only**: In the first assistant message of each new task, the first line must be exactly `【AGENTS effective】`.
- Do not repeat this marker later unless the user explicitly asks for it.
- Claude Code 请参见 [CLAUDE.md](CLAUDE.md)，以 CLAUDE.md 为首要行为约束。

Primary execution contract for this repo. If this file conflicts with any document under `docs/`, follow this file.

## 1. Core Rules

1. No filler. Start with the answer or the action.
2. Never fabricate. Read files, run commands, or say you do not know.
3. Ask only when ambiguity materially changes the output.
4. Touch only the requested scope. No unrelated refactors or cleanup.
5. Do not claim success without verification.
6. Briefly state the plan before editing.
7. Read the files you will change and the code that calls them.
8. Match existing patterns; prefer the smallest correct change.
9. Add concise Chinese comments only when they explain non-obvious logic.

## 2. Project Contract

This project is a single-exposure optical generative model distilled from an LDM teacher.

Final inference path: `noise/code + optional label -> generator_e -> amplitude SLM command -> encoder metasurface -> OLS/fiber -> decoder metasurface -> CMOS single exposure -> generated image`.
- LDM is teacher infrastructure: trained prior, teacher targets, distillation export, and baseline comparison.
- Final optical inference must not require diffusion U-Net denoising, iterative latent sampling, or software-only VAE decoding.
- Main flow: `teacher_first_stage/teacher_diffusion -> teacher_distill_export -> one_shot_distill -> one_shot_sample`.
- Single-shot design source: `docs/ldm_migration/single_shot_ldm_distillation.md`.
- Direct MNIST `one_shot` is baseline/smoke only, not the main research path.
- Keep teacher LDM, one-shot student, and future hardware calibration responsibilities separate.

## 3. Repo Map

- Python 3.11; stack: PyTorch, torchvision, PyYAML, matplotlib.
- Environment: WSL2 Linux shell, Conda + pip, VS Code Remote friendly; Conda env: `optical-generative-models`.
- PyTorch baseline: CUDA build `torch 2.5.1+cu121`.
- Source: `src/ldm_project/`; tests: `tests/`; entry: `train.py`; configs: `configs/`; docs: `docs/ldm_migration/`.
- Key files: `src/ldm_project/one_shot.py`, `src/ldm_project/train_utils.py`, `src/ldm_project/stage_runners.py`
- Teacher stages: `teacher_first_stage`, `teacher_diffusion`, `teacher_full`, `teacher_sample`, `teacher_distill_export`
- Student stages: `one_shot_distill`, `one_shot_sample`; baseline: `one_shot`.
- Compatibility aliases: `first_stage`, `diffusion`, `full`, `sample`.

## 4. Optical Model

- `generator_e` is an electronic generator that maps `noise/code + label` to an amplitude SLM command.
- The SLM is amplitude-controlled. Do not silently reinterpret its command as phase-only or complex-amplitude control.
- Encoder and decoder metasurfaces are thin complex modulation layers: trainable in simulation, manufacturability-constrained, or fixed after fabrication.
- OLS must stay behind a replaceable interface: `complex field in -> OLS/fiber module -> complex field out`.
- GELU proxy, single-mode fiber, multimode fiber, calibrated matrix, and camera-in-the-loop modules must occupy the same OLS interface.
- CMOS output is the primary student output; digital-twin code must represent real `SLM command -> CMOS image` behavior, not just ideal equations.

## 5. Checkpoints And Artifacts

- Explicit CLI checkpoint paths always win.
- `outputs/final_pt/` is the highest-priority source for omitted teacher checkpoints: `first_stage_best_optical.pt`, `diffusion_best.pt`.
- One-shot sampling requires an explicit student checkpoint unless a documented resolver exists.
- Preferred student checkpoints: `one_shot_distill_best.pt` for distilled runs, `one_shot_best.pt` for baseline runs.
- New run artifacts should be written under managed `outputs/runs/...` directories.
- Do not modify `outputs/` except configured final acceptance artifacts; do not modify `data/raw/`, `.tmp/`, `legacy/`, generated checkpoints/plots/caches outside final acceptance.

## 6. Commands

- Install: `conda create -n optical-generative-models python=3.11 -y && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models && pip install -r requirements.txt && pip install -e .`
- Build: `python -m compileall src tests`
- Test all: `pytest -q`
- Targeted tests: `pytest tests/test_train_script.py tests/test_one_shot.py -q`
- Lint: `ruff check src tests *.py`
- Typecheck: `mypy src`
- Command prefix: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate optical-generative-models &&`
- Teacher first stage: `<prefix> python train.py --stage teacher_first_stage --device cuda`
- Teacher diffusion: `<prefix> python train.py --stage teacher_diffusion --device cuda`
- Teacher sample: `<prefix> python train.py --stage teacher_sample --device cuda`
- Export distill data: `<prefix> python train.py --train-config configs/export_teacher_distill_mnist_3k_fast200.yaml --stage teacher_distill_export --device cuda --run-name teacher_distill_mnist_3k_fast200`
- Train distilled student: `<prefix> python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml --stage one_shot_distill --device cuda --distill-dataset outputs/runs/.../artifacts/teacher_distill_dataset.pt --run-name one_shot_distill_d2nn_mnist_3k_fast200`
- Sample student: `<prefix> python train.py --train-config configs/train_one_shot_distill_d2nn_mnist_3k_fast200.yaml --stage one_shot_sample --device cuda --one-shot-checkpoint outputs/runs/.../checkpoints/one_shot_distill_best.pt`
- Baseline smoke: `<prefix> python train.py --train-config configs/train_one_shot_d2nn_smoke.yaml --stage one_shot --device cuda`
- Use `--device auto` only for compatibility checks; use `--device cpu` only for local smoke tests.

## 7. Verification And CUDA

- Docs-only verification: `git diff --check`.
- Prefer targeted tests during iteration; use full suite only for final/shared changes.
- For training changes, verify arguments, stage routing, checkpoint selection, and artifact paths.
- For one-shot changes, verify shapes, finite values, device placement, and checkpoint compatibility.
- Before formal training, activate Conda and verify `torch.cuda.is_available()` in that environment.
- Codex sandbox CUDA failures may be sandbox-only; state that clearly and provide the exact WSL2 CUDA command.
- Long CUDA training or hardware experiments should usually be prepared by Codex and launched by the user.

## 8. Forbidden

- Do not hardcode interpreter paths, Conda paths, Windows paths, or machine-specific directories.
- Do not rely on editor settings as the only runtime config; do not mutate `sys.path`, use `pip install -e .`.
- Do not mix `legacy` optical VAE code directly into the LDM core package.
- Do not write outputs/checkpoints outside `outputs/runs/...`, except configured final acceptance artifacts.
- Do not use full-dataset/full-epoch runs for routine iteration; do not use PowerShell/Windows shell syntax unless explicitly asked.
- Do not present teacher LDM image quality as final student hardware readiness.
- Do not call an optical result single-exposure if it still depends on iterative software denoising or optimization at inference time.

## 9. When To Ask

Ask before proceeding when:

- two plausible interpretations materially change the output
- the change affects stages, checkpoints, output dirs, public CLI, or teacher/student boundaries
- the change affects SLM type, metasurface trainability, OLS/fiber behavior, CMOS calibration, or hardware feedback
- the user's stated goal conflicts with the literal request
- a secret, credential, hardware access, or inaccessible external resource is required

Proceed without asking when ambiguity is resolvable by reading/running, the change is trivial and reversible, or the user already answered.
