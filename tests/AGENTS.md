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
checkout. Installation, minimum CPU checks, focused Ruff rules, and full-suite
commands are maintained in
[`docs/operations/windows_development.md`](../docs/operations/windows_development.md).

A CPU smoke verifies software behavior only; it does not replace hardware or
scientific acceptance.
