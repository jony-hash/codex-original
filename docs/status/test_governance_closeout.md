# Test governance closeout audit

This audit records engineering verification for the test-governance and
evidence-chain work. It is diagnostic project status, not CUDA, hardware, or
scientific acceptance evidence.

## Reverse verification

- Purpose: strengthen provenance, initialization/evidence binding, real
  optimizer updates, safe loading, and historical-test governance while
  reducing structural test weight.
- Preserved invariants: v1/v2 boundaries, formal physical numerical references,
  checkpoint/evidence fail-closed behavior, public CLI/output policy, and
  immutable historical archives.
- Stop conditions: an active behavior test disappearing without a specific
  archive disposition or collected replacement, an unsafe current-v2 load, a
  changed evidence schema/history, a failed active regression, or a dirty
  acceptance worktree.

## Measurements

- Active test source: 18,785 lines (target <=19,100).
- Largest active test file: 1,739 lines (target <2,000).
- Final clean-commit full-suite measurement: 524 tests and 25 subtests passed
  in 88.57s;
  no repository warnings were emitted and the 113s budget was met.
- `pytest --collect-only` reports 524 active tests and no archive nodeids;
  compileall, Ruff, `git diff --check`, and Hash inventory `--check` pass.
- Archive mapping contains 111/111 archived nodeids with unique entries. Each
  entry now has a reason code and historical reference; replacement entries
  must be collected by the mapping test. This is semantic governance coverage,
  not a fixed subtest-count requirement.
- The AST Hash inventory contains 1,097 current calls versus 1,098 at `HEAD`.
  The inventory has explicit syntactic consumers and validation decisions;
  see `hash_consumer_audit.md` and `hash_consumer_inventory.tsv`.
- Current, diagnostic, and research v2 paths contain no direct
  `weights_only=False`; the only unsafe loader is the explicit legacy boundary.
- Final clean provenance check returned `None` from both the repository root and
  `src/ldm_project/` subdirectory; `git status --short` was empty.

## Governance status

The former fixed 148-subtest number is not an acceptance criterion. The current
criterion is: every archived nodeid has a specific disposition, every claimed
replacement is collected, and archive files remain outside default pytest
collection. Historical `experiment_history.md` entries are not rewritten.

The engineering clean-commit gate is complete. CUDA, hardware, promotion and
scientific acceptance remain separate follow-up gates and are not authorized by
this audit.
