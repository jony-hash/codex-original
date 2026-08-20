# Invalidated synthetic fiber ensemble snapshot

`research/v2-fiber-ols-v1` commit `b873eba` and the local tag
`archive/v2/fiber-ols-synthetic-ensemble-v1` are invalidated as
`invalid_harness`. The tag was created before a real optical update, LP01
reference, or matrix-ensemble measurement existed.

The invalid snapshot had an LP01 fixture generated from the same hard-coded
constants used by the implementation, treated a 10x10 tensor as ports instead
of coupling a 100x100 complex field, omitted matrix offsets from transport
realizations, and wrote preflight/screen results without executing those
operations. Its historical nominal three-seed results remain numerical
context only and cannot be reused after the operator changes.

The tag is local and has not been pushed. It will only be moved to a corrected
closeout commit after the complete research lifecycle passes.
