# v2 Hash Consumer Audit

This is a diagnostic governance report. It does not change the v2 evidence
schema or rewrite historical evidence.

## Unified inventory

`tools/audit_hash_consumers.py` uses an AST call-site definition over
`src/ldm_project/v2` and records one row for every direct call to
`canonical_sha256`, `file_sha256`, `tensor_sha256`, or `state_content_hash`.
The inventory is reproducible with:

```text
python tools/audit_hash_consumers.py
python tools/audit_hash_consumers.py --check
```

The current worktree contains 1,097 call sites; the pre-change `HEAD` contains
1,098. One verified duplicate call was removed. The inventory now records the
owning function, syntactic consumer, validation decision, repeat-input key,
category, and explicit retain/delete conclusion. No generic
"conservative default" review remains.

| Category | Current calls |
|---|---:|
| contract binding | 245 |
| content integrity | 263 |
| evidence input binding | 589 |
| **total** | **1,097** |

The static pass identifies 156 same-function/input candidate groups (197
additional call sites). These are candidates for review, not automatic
duplicates: repeated calls are retained unless runtime behavior proves that a
single digest can be passed through without changing a validation decision.
The current checkpoint tests verify one digest check followed by one safe
deserialization for a single operation.

## Retention policy

Retain hashes that participate in content integrity, contract binding, or
evidence input binding. Delete only a call with a verified absence of a
consumer, complete parent-manifest coverage, same-operation duplicate input,
or report-only use. Every deletion must be recorded with its call site and
reason; historical digest values and archive evidence are never regenerated.

The static audit is complete and reproducible. It is not evidence that all
1,097 hashes are scientifically necessary; future reductions must preserve the
same call-site and repeated-read accounting.
