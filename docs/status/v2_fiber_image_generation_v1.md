# v2 fiber image-generation lane status

The research lane is implemented on `research/v2-fiber-image-generation-v1`.
It is diagnostic-only and does not alter the formal identity OLS contract.

The clean validation evidence and paired baseline export are in
`outputs/20260821_v2/runs/20260821_fiber_image_v1/`. The baseline produced
paired I0/S0/S1a/S1b/S1c/N0 CMOS tensors, optical traces, port coefficients,
fixed-exposure contact sheets, and identity difference sheets. The N0/S0
baseline response was observable in both domains (CMOS normalized L2 about
0.848; port normalized L2 about 1.380).

CUDA S0 preflight reached all physical, finite-gradient, freeze, negative
control, full-phase, memory, and schedule checks. It is nevertheless
`invalid_harness` because the RTX 3060/PyTorch CUDA path cannot reproduce the
three-update interrupted/resumed generator within the preregistered
`rtol=atol=1e-5`: the recorded maximum error was `2.3228e-4` and optimizer
state comparison failed. The warning comes from the unsupported deterministic
`adaptive_avg_pool2d_backward_cuda` implementation; the tolerance was not
relaxed.

Consequently no isolated screen, locked select, E1 shared-matrix training, or
scientific closeout was authorized. There is no claim of fiber-dependent or
fiber-specific generator gain, and no archive tag was moved.
