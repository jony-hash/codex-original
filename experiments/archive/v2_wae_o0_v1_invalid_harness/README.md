# WAE O0 v1 invalid-harness correction capsule

This capsule preserves the original O0-I v1 JSON evidence byte-for-byte for
audit. A read-only recheck found that the reconstruction target used random
prior codes instead of encoded image latents and that several acceptance
metrics were placeholders or contract-incompatible. The recorded v1
classification is therefore superseded by `invalid_harness` and cannot
authorize O0-G, SMF, training, or promotion.

The original outputs and checkpoints remain in their original ignored output
paths and are not modified or moved. This capsule contains JSON evidence only.
The corrected lifecycle uses a new `.v2` harness revision, a new clean commit,
and a new output root; it must not warm-start from v1.
