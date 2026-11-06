# Windows development and verification

This guide covers internal installation, the supported Windows package snapshot,
and local CPU checks. The project supports Python 3.11 or later. The Windows
explicit lock records 169 package URLs, builds, and SHA-256 values. It preserves
the supported `win-64` snapshot except that SymPy is pinned to `1.13.1` to meet
the installed PyTorch `2.5.1` Python dependency metadata; `mpmath` is pinned to
`1.3.0` to satisfy that SymPy version's declared range. The pip-only Windows
lock carries hashes for packages outside Conda, including PyTorch's missing
`fsspec` dependency. Shiboken6 remains owned by the Conda PySide6 package and is
checked against that package's file record. `environment-win-lock.json` records
package versions and CUDA build information without credentials or an
installed-machine path. `environment.yml` remains the channel-based starting
point for a new solve.

`requirements-win-cpu.lock.txt` is the separate Actions lock for Windows
CPython 3.11 x64. It pins the 34 third-party packages from the validated CPU
venv, including Torch `2.5.1+cpu`, torchvision `0.20.1+cpu`, pytest `9.0.3`,
Ruff `0.15.12`, and setuptools `79.0.1`; every selected wheel is hash-locked.
The lock verifier checks hashes, coverage of `pyproject.toml`, and—when asked—the
installed package records. Conda explicit URL fragments follow the Conda text
specification for MD5/SHA-256 checksums: [text spec](https://conda.org/learn/specifications/exchange/text-spec/).

## Create a clean environment from the Windows lock

The supported development environment remains
`D:\ANACONDA\envs\optical-generative-models-win`. Keep it unchanged. The
accepted full-lock environment is `D:\lg-envs\win`, the CPU CI environment is
`D:\lg-envs\cpu`, and micromamba is `D:\lg-envs\tools\micromamba.exe` with
private root `D:\lg-envs\mr`. Installation caches, `TEMP`, and test scratch
belong under a unique `D:\lg-envs\scratch\<run-id>` directory, never under
`C:\TEMP` or the repository. The accepted prefixes already exist on this host;
do not overwrite them. For a rebuild, choose a new, empty D: prefix and remove
it only after the replacement passes acceptance.

The verified micromamba release is `2.9.0-0`, SHA-256
`a6d804394b2418991c4e29562853eaace2f2ce9d9da661a98e74e02e8dbb44b0`. Obtain
it from the official release over HTTPS and verify this hash before running it.
Use `--no-rc`, an explicit root prefix, and process-local environment variables;
do not edit global Conda/micromamba configuration or machine `PATH`/`TEMP`.

With Conda available, keep the new environment and package cache on D:. Set
`$envPrefix` to a new, empty prefix; do not overwrite `D:\lg-envs\win` or the
supported development environment:

```powershell
$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$tempRoot = Join-Path 'D:\lg-envs\scratch' ('conda-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:CONDA_PKGS_DIRS = Join-Path $tempRoot 'pkgs'
$env:PIP_CACHE_DIR = Join-Path $tempRoot 'pip-cache'
$envPrefix = 'D:\lg-envs\win-rebuild-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
if (Test-Path -LiteralPath $envPrefix) { throw 'Refusing to reuse an environment prefix' }
New-Item -ItemType Directory -Path $env:CONDA_PKGS_DIRS,$env:PIP_CACHE_DIR -Force | Out-Null
conda create --prefix $envPrefix --file environment-win-explicit.lock.txt --yes
if ($LASTEXITCODE -ne 0) { throw 'The explicit Conda lock installation failed' }
$py = Join-Path $envPrefix 'python.exe'
& $py -m pip install --require-hashes --no-deps -r requirements-win-pip.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'The pip-only Windows lock installation failed' }
& $py scripts/verify_environment_locks.py --verify-installed-conda-lock --verify-installed-windows-pip-lock
if ($LASTEXITCODE -ne 0) { throw 'Installed package records do not match the locks' }
& $py -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable project installation failed' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'The installed dependency set is inconsistent' }
& $py -c "import sys, torch, torchvision; x=torch.ones((2,2),device='cpu'); assert torch.isfinite(x@x).all().item(); print({'python':sys.version.split()[0], 'torch':torch.__version__, 'torchvision':torchvision.__version__, 'cuda_build':torch.version.cuda, 'cuda_available':torch.cuda.is_available(), 'cpu_smoke':'PASS'})"
```

When Conda is unavailable, use the verified D: micromamba executable, private
root prefix, and a short D: cache path. Put only small logs and acceptance
records under a new `outputs/.../runs/diagnostic-only_*` evidence directory.
The former long cache path caused a Windows MAX_PATH error; the accepted D:
install used a short scratch path.

```powershell
$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$evidence = Join-Path $repo ("outputs/" + (Get-Date -Format 'yyyyMMdd') + "_environment_check/runs/diagnostic-only_windows_lock_install_" + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $evidence) { throw 'Refusing to overwrite diagnostic evidence' }
New-Item -ItemType Directory -Path $evidence | Out-Null
$tempRoot = Join-Path 'D:\lg-envs\scratch' ('mm-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
if (Test-Path -LiteralPath $tempRoot) { throw 'Refusing to reuse a micromamba scratch directory' }
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$rootPrefix = 'D:\lg-envs\mr'
$envPrefix = 'D:\lg-envs\win-rebuild-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
if (Test-Path -LiteralPath $envPrefix) { throw 'Refusing to reuse an environment prefix' }
$micromamba = 'D:\lg-envs\tools\micromamba.exe'
$python = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'
$savedRuntimeVariables = @{}
Get-ChildItem Env: | Where-Object { $_.Name -match '^(CONDA|MAMBA|PYTHON|CUDA|TORCH|CUDNN|CUBLAS|PIP)' } | ForEach-Object {
  $savedRuntimeVariables[$_.Name] = $_.Value
  Remove-Item ("Env:" + $_.Name)
}
$env:MAMBA_ROOT_PREFIX = $rootPrefix
$env:CONDA_PKGS_DIRS = Join-Path $tempRoot 'pkgs'
$env:MAMBA_NO_BANNER = '1'
$pipCache = Join-Path $tempRoot 'pip-cache'
New-Item -ItemType Directory -Path $pipCache | Out-Null
$env:PIP_CACHE_DIR = $pipCache
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
try {
try {
  $expectedMicromamba = 'a6d804394b2418991c4e29562853eaace2f2ce9d9da661a98e74e02e8dbb44b0'
  $actualMicromamba = (Get-FileHash -LiteralPath $micromamba -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualMicromamba -ne $expectedMicromamba) { throw 'micromamba release SHA-256 mismatch' }
  & $micromamba --no-rc --root-prefix $rootPrefix --version
  if ($LASTEXITCODE -ne 0) { throw 'micromamba failed its version probe' }
} catch {
  $_ | Out-String | Set-Content -LiteralPath (Join-Path $evidence 'micromamba-tool-failure.log') -Encoding utf8
  throw
}
$networkXPath = Join-Path $tempRoot 'pkgs\https\conda.anaconda.org\conda-forge\noarch\networkx-3.6.1-pyhcf101f3_0\site-packages\networkx\algorithms\centrality\tests\test_current_flow_betweenness_centrality_subset.py'
if ($networkXPath.Length -ge 260) { throw 'Micromamba cache path may exceed Windows MAX_PATH; choose a shorter TEMP path' }
$installLog = Join-Path $tempRoot 'conda-install.json.log'
& $micromamba --no-rc --root-prefix $rootPrefix create --prefix $envPrefix --file (Join-Path $repo 'environment-win-explicit.lock.txt') --yes 2>&1 | Tee-Object -FilePath $installLog
$condaCode = $LASTEXITCODE
Copy-Item -LiteralPath $installLog -Destination $evidence
if ($condaCode -ne 0) { throw 'The explicit Windows lock installation failed; preserve the log and stop' }
$py = Join-Path $envPrefix 'python.exe'
$pipLog = Join-Path $tempRoot 'pip-install.log'
& $micromamba --no-rc --root-prefix $rootPrefix run --prefix $envPrefix $py -m pip install --require-hashes --no-deps --no-input --cache-dir $pipCache -r (Join-Path $repo 'requirements-win-pip.lock.txt') *> $pipLog
if ($LASTEXITCODE -ne 0) { Copy-Item $pipLog $evidence; throw 'The pip-only Windows lock installation failed' }
& $micromamba --no-rc --root-prefix $rootPrefix run --prefix $envPrefix $py (Join-Path $repo 'scripts/verify_environment_locks.py') --verify-installed-conda-lock --verify-installed-windows-pip-lock
if ($LASTEXITCODE -ne 0) { Copy-Item $pipLog $evidence; throw 'Installed package records do not match the locks' }
& $micromamba --no-rc --root-prefix $rootPrefix run --prefix $envPrefix $py -m pip install --no-deps --no-build-isolation -e $repo *>> $pipLog
if ($LASTEXITCODE -ne 0) { Copy-Item $pipLog $evidence; throw 'Editable project installation failed' }
& $micromamba --no-rc --root-prefix $rootPrefix run --prefix $envPrefix $py -c "import sys, torch, torchvision, ldm_project; x=torch.ones((2,2),device='cpu'); assert torch.isfinite(x@x).all().item(); print({'python':sys.version.split()[0], 'torch':torch.__version__, 'torchvision':torchvision.__version__, 'cuda_build':torch.version.cuda, 'project_import':'PASS', 'cpu_smoke':'PASS'})"
if ($LASTEXITCODE -ne 0) { throw 'The minimal CPU smoke failed' }
Copy-Item -LiteralPath $pipLog -Destination $evidence
$checkLog = Join-Path $tempRoot 'pip-check.log'
& $micromamba --no-rc --root-prefix $rootPrefix run --prefix $envPrefix $py -m pip check *> $checkLog
$checkCode = $LASTEXITCODE
Copy-Item -LiteralPath $checkLog -Destination $evidence
if ($checkCode -ne 0) { throw 'pip check found an environment dependency conflict; preserve the output and do not report acceptance' }
} finally {
  Remove-Item Env:MAMBA_ROOT_PREFIX,Env:CONDA_PKGS_DIRS,Env:MAMBA_NO_BANNER,Env:PIP_CACHE_DIR,Env:PIP_DISABLE_PIP_VERSION_CHECK -ErrorAction SilentlyContinue
  foreach ($name in $savedRuntimeVariables.Keys) { Set-Item ("Env:" + $name) $savedRuntimeVariables[$name] }
}
```

## Minimum CPU verification

Create a separate Python 3.11 venv with `include-system-site-packages=false`.
The accepted CPU CI environment is `D:\lg-envs\cpu`. The base interpreter may
be the supported Windows Python; all locked packages are installed into the
new venv and checked independently. The CPU lock file specifies the PyPI and
official PyTorch CPU indexes. Keep `TEMP`, `TMP`, `RUNNER_TEMP`, and pip cache on
D:. Do not replace the inherited process `PATH`; the shared workflow invokes
Git, so verify that `Get-Command git` succeeds before running it.

```powershell
$ErrorActionPreference = 'Stop'
$cpuRoot = 'D:\lg-envs\cpu'
if (Test-Path -LiteralPath $cpuRoot) { throw 'Refusing to reuse the accepted CPU environment' }
$scratch = Join-Path 'D:\lg-envs\scratch' ('cpu-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
$env:TEMP = $scratch
$env:TMP = $scratch
$env:RUNNER_TEMP = $scratch
$basePython = 'D:\ANACONDA\envs\optical-generative-models-win\python.exe'
$venv = $cpuRoot
& $basePython -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated CPU venv' }
$py = Join-Path $venv 'Scripts/python.exe'
$env:PIP_CACHE_DIR = Join-Path $scratch 'pip-cache'
New-Item -ItemType Directory -Path $env:PIP_CACHE_DIR | Out-Null
& $py -m pip install --require-hashes --no-deps -r requirements-win-cpu.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Hash-locked CPU dependency installation failed' }
& $py scripts/verify_environment_locks.py --verify-installed-cpu-lock
if ($LASTEXITCODE -ne 0) { throw 'Installed CPU packages do not match the lock' }
& $py -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable project installation failed' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'CPU environment dependency check failed' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is missing from this process PATH; preserve or repair PATH for this process before running checks' }
& $py scripts/run_windows_cpu_checks.py
if ($LASTEXITCODE -ne 0) { throw 'The shared Windows CPU workflow failed' }
git diff --check
```

The GitHub Actions job runs the same `scripts/run_windows_cpu_checks.py` with
this same hash lock. It runs the architecture and import-boundary, lifecycle,
physical field type, canonical-gradient, BOGT, and frozen-400 preflight tests,
then the workflow Ruff file set and `compileall`. The CPU pass checks software
contracts only; it does not validate CUDA or scientific outcomes. The locked
Windows Conda environment and the CPU CI environment are separate targets.

GPU memory, utilization, and elapsed time are observations, not admission
thresholds. The CUDA build information records the export host only. Actual
CUDA unavailability, numerical errors, and failed restoration remain engineering
failures and do not count as scientific results.

For the full default test suite after shared or cross-package changes, use the
supported Windows interpreter and an external temporary directory:

```powershell
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$py = Join-Path $envRoot 'python.exe'
$testTemp = Join-Path 'D:\lg-envs\scratch' ("ldm-project-full-pytest-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
& $py -B -m pytest -q --basetemp $testTemp
git diff --check
```

Ruff uses `python -m ruff check --select E4,E9,F821` on the workflow's named
file list. This focused gate checks invalid import placement, syntax-level
failures, and undefined names; it does not claim a clean full-style Ruff
baseline for older research scripts.

## Production CUDA preflight

`--production-preflight` is diagnostic-only. It requires an explicit
`--checkpoint` and a new `--output-dir`; it does not accept `--resume`, open a
scientific attempt, or grant science authority. Use each selected source
checkpoint separately, with the actual 2048 grid and TRAIN population:

```powershell
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$py = Join-Path $envRoot 'python.exe'
$env:PATH = @((Join-Path $envRoot 'bin'), (Join-Path $envRoot 'Library\bin'), (Join-Path $envRoot 'Scripts'), (Join-Path $envRoot 'Lib\site-packages\torch\lib'), $env:PATH) -join ';'
& $py -B scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py `
  --production-preflight `
  --checkpoint outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/scientific_step_000000.pt `
  --output-dir outputs/YYYYMMDD_engineering/runs/diagnostic-only_bogt_preflight_step000000
& $py -B scripts/research/lightgen/execute_lightgen_2d_chen_bogt_complex_ols_training_v1.py `
  --production-preflight `
  --checkpoint outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/scientific_step_000100.pt `
  --output-dir outputs/YYYYMMDD_engineering/runs/diagnostic-only_bogt_preflight_step000100
```

The process-local `PATH` above adds the supported environment's CUDA runtime
folders, including `torch/lib`, for NVRTC DLL search. It does not edit the
supported environment or system configuration. When investigating an NVRTC
error, check the actual loaded file and process search path; DLL presence alone
does not prove that the runtime resolved it. The diagnostic must record three
real updates, the negative control, midpoint checkpoint reload, step/Adam/
history/sampler consistency, and complete restoration of model, frozen state,
weights, references, Adam, gradients, RNG, sampler, and progress. A failed CUDA,
numerical, or restoration stage ends that CUDA acceptance; preserve the report
and do not continue to the second input after a real failure.

## Local acceptance record for 2026-10-03

Both explicit BOGT inputs completed `READY` on the production CUDA path with
three real updates, frozen-parameter negative control rejected, midpoint reload
comparison passed, and all ten restoration checks passed. Step0 loaded zero
history/Adam entries/cursor and verified updates 1–3. Step100 loaded 100 history
rows, two Adam parameter states, and cursor100; it used completed update steps
101–103 and restored cursor100. Each report records the first TRAIN batch row
indices and each complete rollback comparison. Both runs left the scientific
attempt closed and scientific update count at zero. The report-level CUDA state
uses 2048 propagation grid and 55,000 TRAIN rows; DEV was not measured. See the
[step0 report](../../outputs/20261003_engineering_governance/runs/diagnostic-only_bogt_preflight_step000000_pathfix_v4/preflight_report.json)
and [step100 report](../../outputs/20261003_engineering_governance/runs/diagnostic-only_bogt_preflight_step000100_pathfix_v1/preflight_report.json).

The NVRTC DLLs were present in the supported environment. The initial error was
a runtime lookup failure; adding the environment's CUDA and `torch/lib`
folders to this process's `PATH` allowed the actual CUDA kernel and production
preflight to run. No package or global path was changed. Recorded free-memory
values and elapsed times are observations only; no fixed threshold was applied.

The full Windows lock was installed into a new environment with official
micromamba `2.9.0`; its release SHA-256, all 169 Conda records, both pip hashes,
editable install, project import, and CPU tensor smoke passed. The first install
attempt exposed a Windows path-length failure: a NetworkX cache path was 262
characters. The retry used a fresh short root and installed successfully; the
failed prefix and its evidence were retained.

The first complete-lock acceptance reproduced the original environment's
dependency conflicts. The follow-up lock pins SymPy `1.13.1`, its required
`mpmath 1.3.0`, and hashed `fsspec`; it keeps PyTorch and torchvision builds
unchanged. A fresh environment now passes `pip check`, the complete installed
lock verifier, editable install, project import, and CPU tensor smoke. The
[Windows lock repair evidence](../../outputs/20261003_engineering_governance/runs/diagnostic-only_windows_lock_repair_20261003/acceptance_record.json)
records both the failed original snapshot and the successful repaired install.

A separate CPython 3.11.16 venv installed all 34 CPU packages from the SHA-256
lock. Its installed versions, project-dependency coverage, editable install,
`pip check`, and CPU tensor smoke passed. The final Windows Actions-equivalent
local run passed **164 selected tests**, Ruff `E4/E9/F821`, and `compileall`.
This is local software evidence,
not a remote Actions result. The [CPU acceptance record](../../outputs/20261003_engineering_governance/runs/diagnostic-only_windows_cpu_ci_hash_locked/acceptance_record.json)
records the temporary environment location. Remote CI remains pending because
this task does not push changes.

## Migration and evidence boundaries

The supported Windows-native environment is the local development/test
authority. Restored WSL outputs and their restore metadata remain immutable and
are never resolved as current artifacts. Transfer large checkpoints, weights,
datasets, and tensors through the project's artifact-transfer process rather
than Git. Do not commit, push, or refresh Sources automatically as part of
environment setup. See [output artifact notes](output_artifact_notes.md) and
[Windows-native migration history](../operations/migration_windows_native_20260830.md).

## Code and binary delivery

Git carries source, configuration and environment locks. Existing tracked
weights stay as they are; do not rewrite history or convert them to LFS during
this cleanup. Transfer new large checkpoints, weights, datasets and tensors
separately. Copy a needed checkpoint together with its optimizer, RNG and
sampler records when the receiving task promises resume; a parameter-only
checkpoint does not establish resumability.

For an inbound artifact, choose an explicit new, non-restored run child and
check for an existing destination before copying. This PowerShell template
does not overwrite an existing file; replace the example values first:

```powershell
$remoteHost = 'user@research-host'
$remotePort = 22
$remoteArtifact = '/absolute/remote/path/checkpoint.pt'
$destination = Join-Path (Get-Location) 'outputs/YYYYMMDD_v2/runs/diagnostic-only-transfer/checkpoint.pt'
if (Test-Path -LiteralPath $destination) { throw 'Destination already exists' }
$destinationDir = Split-Path -Parent $destination
New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
scp -P $remotePort "${remoteHost}:${remoteArtifact}" $destination
if ($LASTEXITCODE -ne 0) { throw 'Artifact transfer failed' }
```

Record the explicit source and destination and the artifact's role in the
receiving task. Loaders must check its model/configuration, completed step,
optimizer and sampling state. Transport digests can record copy provenance;
they do not replace these compatibility checks or authorize a new experiment.
Do not copy into `outputs/final_pt/`, restored run children, restore metadata,
`data/raw/`, or legacy roots. Dataset access remains governed by the receiving
task's declared TRAIN/DEV/protected/test roles.
