# Tests

Keep tests beside their runtime family. Research regressions live under
`tests/research/{lightgen,scaled_chen,optical_manifold,capacity}`; shared
research lifecycle tests live under `tests/research/lightgen` when their
behavior is family-neutral. Formal v1/v2 tests remain in their current areas.

Use the smallest tests that prove the changed behavior, then run the full suite
once for shared or cross-package changes. Tests should validate actual inputs,
model/config/step/optimizer state, data roles, frozen groups, and saved payloads;
do not require source-digest or dirty-patch equality.

In Windows-native work, use the project environment and keep pytest temporary
files outside the checkout:

```powershell
$envRoot = 'D:\ANACONDA\envs\optical-generative-models-win'
$env:Path = "$envRoot;$envRoot\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\Lib\site-packages\torch\lib;$env:Path"
$py = Join-Path $envRoot 'python.exe'
& $py -m pytest <focused-tests> -o cache_dir="$env:TEMP\codex-project-refactor-pytest-cache" --basetemp "$env:TEMP\codex-project-refactor-pytest-tmp"
& $py -m compileall -q src tests scripts train.py
& (Join-Path $envRoot 'Scripts\ruff.exe') check <changed-python-files>
git diff --check
& $py -m pytest -q -o cache_dir="$env:TEMP\codex-project-refactor-pytest-cache" --basetemp "$env:TEMP\codex-project-refactor-pytest-tmp"
```

A CPU smoke verifies software behavior only; it does not replace hardware or
scientific acceptance.
