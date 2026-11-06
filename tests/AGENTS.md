# Tests

Keep tests beside their runtime family. Research regressions live under
`tests/research/{lightgen,scaled_chen,optical_manifold,capacity}`; shared
research lifecycle tests live under `tests/research/lightgen` when their
behavior is family-neutral. Formal v1/v2 tests remain in their current areas.

Use the smallest tests that prove the changed behavior, then run the full suite
once for shared or cross-package changes. Tests should validate actual inputs,
model/config/step/optimizer state, data roles, frozen groups, and saved payloads;
do not require source-digest or dirty-patch equality.

Shared research lifecycle tests must cover a valid receipt and representative
rejections for missing credentials, failed check groups, inconsistent engine
selection, and terminal restart. Architecture boundary tests scan the canonical
v1/v2 source trees, resolve absolute and relative imports, and fail if a required
scan root is missing or empty.

Use the supported Windows interpreter, and put pytest `--basetemp` outside the
checkout. The single verification entry is
`python scripts/run_windows_cpu_checks.py` (focused lane) or
`python scripts/run_windows_cpu_checks.py --full` (whole `tests/` tree). It runs
the regression tests, repository-wide Ruff (`E9, F82`), the focused Ruff rules,
and `compileall`. Installation and environment lock commands are maintained in
[`docs/operations/windows_development.md`](../docs/operations/windows_development.md).

A CPU smoke verifies software behavior only; it does not replace hardware or
scientific acceptance.

CI (`.github/workflows/windows-cpu.yml`) runs the focused lane on every push and
pull request; `--full` is not part of CI. Treat a full-suite result as manually
attested evidence and record its failures rather than silently fixing them.
