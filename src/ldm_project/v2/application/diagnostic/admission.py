# ruff: noqa: F401,F403,F405,F821
"""Diagnostic lifecycle bindings for neutral conditional admission workflows."""
from __future__ import annotations

from .._conditional_admission_shared import ConditionalRuntimeBindings, bind_admission
from . import runtime as _runtime

globals().update(bind_admission(ConditionalRuntimeBindings(vars(_runtime))))


def sample_conditional(args, config: V2Config) -> int:
    """Diagnostic sampling retains its broad conditional-architecture support."""
    _require_conditional(config)
    device = torch.device(config.run.device)
    bundle, _ = _load_bundle(config, args.checkpoint, device=device)
    codes = torch.load(args.codes, map_location=device, weights_only=False)
    labels = torch.load(args.labels, map_location=device, weights_only=False).long()
    if not torch.is_tensor(codes) or not torch.is_tensor(labels):
        raise ValueError("codes and labels must be tensor files")
    raw = raw_from_generator_batched(
        bundle.system,
        bundle.generator,
        codes,
        labels,
        microbatch_size=config.execution.evaluation_microbatch_size,
        activation_checkpoint=False,
    ).detach().cpu()
    atomic_torch_save(args.output, raw)
    return 0


def _conditional_parameter_groups(
    bundle: ConditionalBundle,
) -> dict[str, tuple[nn.Parameter, ...]]:
    if getattr(bundle.generator, "architecture_kind", None) == MULTI_ANCHOR_ARCHITECTURE:
        return {
            "anchor_logits": (bundle.generator.anchor_logits,),
            "local_modes": (bundle.generator.local_modes,),
            "encoder_phase": (bundle.system.encoder.phase,),
            "decoder_phase": (bundle.system.decoder.phase,),
        }
    return {
        "class_carrier": (bundle.generator.class_carrier_logits,),
        "global_modes": (bundle.generator.global_modes,),
        "code_projection": (bundle.generator.code_projection.weight,),
        "class_heads": (bundle.generator.class_heads,),
        "encoder_phase": (bundle.system.encoder.phase,),
        "decoder_phase": (bundle.system.decoder.phase,),
    }
