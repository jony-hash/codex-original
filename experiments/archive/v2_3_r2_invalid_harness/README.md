# v2.3-r2 invalid-harness capsule

This read-only capsule preserves the v2.3-r2 run facts while correcting their
scientific status.  The D0 probe was not a valid negative result: model
initialization was not bound to the evidence seed and auxiliary classifier/Q
gradients were not stepped by an optimizer.  No smoke, confirm, locked-final,
C0--C3, optical, teacher, or external-data branch ran.

The checkpoint remains in its managed `outputs/` location and is represented
only by its original path and SHA-256.  Nothing in this capsule is runnable or
authorizes promotion.
