# Corrected fiber OLS status

The corrected operator and synthetic diagnostic lifecycle are implemented on
`research/v2-fiber-ols-v1`; formal identity promotion remains closed:

- LP01 V=2.2 is solved independently in float64. The current root is
  `u=1.5910592824373024`, `w=1.5193848622946382`; closure is below `1e-15`
  and boundary derivative mismatch is below `2e-15`.
- The primary model is a cell-partitioned 100-port LP01 overlap operator on
  the formal 100x100 encoder plane. It records input/output energy and rejects
  non-passive transport.
- Matrix records include exact realized phase/loss statistics, independent
  signed y/x offsets, and hashes over the matrix realization plus provenance.
- `validate` now consumes the offline fixture, runs an independent finite-volume
  radial check, measures the actual Lowdin overlap, and records the transport
  pool provenance.
- `preflight`, `screen`, and `select` bind research-only initialization, dual
  evaluators, resume replay, B0/identity controls, bootstrap, and closeout.

The old CUDA preflight was blocked by lineage, not by a model result. The
available 20260812 checkpoint carries config hash
`77db9b7fcfe47cb53f2d089c40454e811a898a329a8f694b8370b8d17d81706d`, while
the current formal config hashes to
`c585bc683dc9af347699c24c79431f3fab398efc533a17778c7c9103dcc7e987`.
The checkpoint is therefore accepted only as an explicitly labelled research
initialization: its generator tensors are imported into a freshly constructed
current identity system, while its old relay/system state is never loaded or
re-signed. Until the resulting clean-branch CUDA preflight passes, no screen,
locked selection, tag movement, or scientific closeout was authorized until the
corrected lifecycle recorded below completed.

## Historical synthetic ensemble attempt (2026-08-21; invalidated)

The following artifact set is retained as a numerical diagnostic record only.
It is not a valid scientific closeout. The attempt at `b7e6108` used
non-replayable `/tmp` checkpoint paths, did not consume source/evaluator
qualification and measurement evidence, did not bind the research harness
commit through selection, omitted the full-phase port gate, and mixed
measurement validity with scientific quality gates. Its existing
`valid_failed` label is therefore superseded by `invalid_harness`.

The old archive targets are preserved for audit:

- invalid preliminary target: `b873eba`;
- invalid corrected-attempt target: `b7e6108`.

No checkpoint from this attempt may be used for the next run.

## Historical synthetic ensemble result (diagnostic only)

The corrected lifecycle executed numerically on the research runtime:

- validation: `37d9d5edf3d82f2a6fe868b2c3e744e16c09efc0dd815c8137263f249237fbf`;
- CUDA preflight: `9c94e8d5dd7e0f35634af21f96012278fb7af9c522b63e4193f593cf405faa75`;
- three-seed screen: `a38325ac4cc8b25b7fba1dd56ab34a294e30edf0c10777d8cf202db9e71a09e8`;
- locked selection: `d69d28c2331a43bf18f4973e5d57ac0f296824b925a1e1e459c6e39937eb59fc`;
- closeout: `4247abf6ae2386952591ab3c33c9c16a4401551eeeb45404c2e5d238a5b7e2ca`.

The persistent research artifacts are under
`outputs/20260821_v2/runs/20260821_robustness_fiber_ols_v1_closeout/`.

All measurements were finite and passive, but secondary median gains were
approximately zero with bootstrap lower bounds not above zero. These values
remain useful diagnostic observations, but the evidence chain is invalid and
does not authorize a `valid_failed` scientific conclusion. No formal promotion
or additional training is permitted.

The next lifecycle must generate a research-only measurement manifest,
persist all checkpoints inside one run directory, bind the research harness
commit, and rerun `validate → preflight → screen → select → closeout` from a
clean commit.

## Strict replacement lifecycle (2026-08-21)

The replacement run was executed from clean commit `fcbee0f` with a research
measurement manifest and a fresh generator initialization. Its persistent
artifacts are under
`outputs/20260821_v2/runs/20260821_robustness_fiber_ols_v2_closeout/`:

- validation: `ea244c82cbbdef9fea7a0b4c1620a24b6bf1a1325411493c02678bef3a7862bc`;
- measurement bundle: `542942df9505ce5f5a39f41405ff3ef767ad1a49a08d97ba7781cbff01764085`;
- measurement evidence: `a14c2807222839fe97996ab900046bac9ab238ea2c0a69f9844f3671aa39bef9`;
- CUDA preflight: `1b589da05cb34dfb062a0a95a96ac4c30ad3515d52cdd8285db76654dddd92d9`;
- screen: `aeee0de2d2ec1582b56b0189a4cf4db49bab2323da9494b8a81b1cf7feb7bca6`;
- locked selection: `f3ecb443f50b6038cfbd36103aaa83017711c5fb31ad086eda422433bfc325ec`;
- closeout: `c495986c3811457d986ceb2a798e70ac394ec98c9acf64a98743bf44e2d1e37e`.

The lifecycle passed structural and measurement validity, including the
full-phase port/CMOS preflight gates. The scientific conclusion is
`no_reproducible_generator_gain` with `valid_failed`: all three locked-panel
bootstrap median gains are zero and their 95% CI lower bounds are zero;
identity gap and seed-consistency gates also fail. This remains synthetic
diagnostic-only, does not authorize additional training or formal promotion,
and does not establish calibrated hardware behavior.

The local final archive tag must not be treated as valid until the replacement
lifecycle completes. It is not pushed.
