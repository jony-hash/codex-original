# v2 Hash decision audit

This is a compact engineering audit. It records only summaries that affect a
content-integrity check, a contract binding, or an evidence rejection decision.
It does not claim that every derived digest is scientifically necessary.

## Retained decisions

- `file_sha256` protects checkpoint, tensor artifact, evidence, and workspace
  files before they are consumed.
- `tensor_sha256` and `state_content_hash` bind tensor/state contents
  independently of serialization bytes.
- `canonical_sha256` binds configuration, contracts, manifests, and evidence
  records to their canonical representation.

## This closeout

- Removed the generated all-retain AST inventory; its syntactic classification
  did not prove a business consumer.
- Removed verified same-operation duplicate reads in ArtifactRef validation,
  evaluation tensor admission, and training-completion checkpoint evidence.
- Kept the existing v2 evidence schema and historical digest values unchanged.
- Retained the explicit legacy unsafe-loader boundary and its static isolation
  test.

## Open, non-blocking item

Absolute paths can still affect identity in future manifest migrations. That is
recorded as a separate schema concern; this closeout does not add a new digest
type, cache, service, or manifest version.
