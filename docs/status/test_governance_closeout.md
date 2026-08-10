# Test governance and evidence-chain status

This document records engineering status only. CPU tests do not authorize
CUDA, hardware, promotion, or scientific acceptance.

## Verified engineering properties

- Current identity initialization is bound to the configured seed before model
  construction; checkpoint and evidence consumers validate the same lineage.
- Current training/evaluation, generation export, candidate audit/utility, and
  dataset admission consume their declared evidence actions before producing
  downstream artifacts; dataset admission also compares evaluator and split
  lineage rather than only the checkpoint.
- Current training checks optimizer coverage, real parameter updates, frozen
  parameters, and the no-step negative control.
- Non-legacy v2 loading uses the safe weights-only boundary. Unsafe loading is
  isolated to the private legacy serialization module.
- Workspace provenance is clean-detecting from the repository root and nested
  source directories, including tracked, staged, and untracked changes.
- Formal optical numerical references and current/v1/diagnostic import-boundary
  tests remain separate from historical archive material.

## Test lanes

The default lane contains current/formal regressions and lightweight public
diagnostic contracts. The two slower diagnostic workflows are marked
`scheduled_diagnostic` and run explicitly with:

```text
pytest -q -o addopts='' -m scheduled_diagnostic
```

Archive files are reference-only and remain outside the default pytest
`testpaths`. Their presence is historical preservation, not a reduction of
total repository test source.

## Hash governance

The generated all-retain Hash inventory was removed because syntactic call-site
classification did not establish actual consumers. The current audit keeps
only content-integrity, contract-binding, and evidence-rejection decisions;
see `hash_consumer_audit.md`. Historical digest values and evidence artifacts
were not recomputed.

## Acceptance boundary

The engineering chain is ready for clean-commit review when default and
scheduled tests, static checks, safe-loading isolation, provenance checks, and
consumer-level evidence tamper negative controls pass. A clean provenance
result is required before acceptance-tier commands can create evidence. This
status does not replace CUDA, hardware, strict readability, promotion, or
scientific acceptance gates.
