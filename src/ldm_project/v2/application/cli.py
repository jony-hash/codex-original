"""Small, side-effect-free v2 CLI dispatcher.

The static registry lives in :mod:`cli_handlers` so this public lifecycle
boundary stays small and imports execution targets only after parsing.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .cli_handlers import (
    IDENTITY_ARCHITECTURE,
    _DIAGNOSTIC_HANDLERS,
    _IDENTITY_HANDLERS,
    _LEGACY_HANDLERS,
)

_CONTROL_ORACLE_STEPS = 400
_CONTROL_ORACLE_LR = 1.0e-2
_CONTROL_ORACLE_PROGRESS_INTERVAL = 40


def _parser() -> argparse.ArgumentParser:
    from .cli_schema import _parser as build_parser

    return build_parser()


def _load_config(args: argparse.Namespace):
    if not hasattr(args, "config"):
        return None
    from ..config import load_config

    return load_config(args.config)


def _resolve(spec):
    module_name, separator, name = spec.target.partition(":")
    if not separator:
        raise ValueError(f"invalid CLI target: {spec.target}")
    return getattr(importlib.import_module(module_name, __package__), name)


def _summarize(args: argparse.Namespace) -> int:
    print(Path(args.run).read_text(encoding="utf-8"))
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.domain == "research":
        if args.research_program == "o0-feasibility":
            from .research.o0_feasibility import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "o0-identifiability":
            from .research.o0_identifiability import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "objective-o0":
            from .research.objective_o0 import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-robustness":
            from .research.fiber_robustness import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-image":
            from .research.fiber_image_generation import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-attribution":
            from .research.fiber_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-interface-attribution":
            from .research.fiber_interface_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-loss-exposure-attribution":
            from .research.fiber_loss_exposure_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "fiber-decoder-controllability-attribution":
            from .research.fiber_decoder_controllability_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-wae":
            from .research.electronic_o0_wae import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-latent-transport":
            from .research.electronic_o0_latent_transport import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-mixture-transport":
            from .research.electronic_o0_mixture_transport import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-prior-native-swae":
            from .research.electronic_o0_prior_native_swae import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-prior-native-sced":
            from .research.electronic_o0_prior_native_sced import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-prior-gradient-conflict":
            from .research.electronic_o0_prior_gradient_conflict import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-prior-trajectory-curvature":
            from .research.electronic_o0_prior_trajectory_curvature import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "electronic-o0-prior-dense-local-attribution":
            from .research.electronic_o0_prior_dense_local_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program == "smf-modal-density-attribution":
            from .research.smf_modal_density_attribution import dispatch

            return int(dispatch(args) or 0)
        if args.research_program != "capacity-map":
            raise RuntimeError(f"unknown research program {args.research_program}")
        from .research.capacity_map import dispatch

        return int(dispatch(args) or 0)
    config = _load_config(args)
    spec = _IDENTITY_HANDLERS.get((args.domain, args.action))
    if config is not None and config.architecture == IDENTITY_ARCHITECTURE:
        if spec is None:
            raise RuntimeError(
                f"identity architecture has no current handler for "
                f"{args.domain}/{args.action}"
            )
        return int(_resolve(spec)(args, config) or 0)
    spec = _DIAGNOSTIC_HANDLERS.get((args.domain, args.action))
    if spec is None:
        spec = _LEGACY_HANDLERS.get((args.domain, args.action))
    if spec is not None and spec.call_kind == "args":
        return int(_resolve(spec)(args) or 0)
    if spec is not None and config is not None:
        target = _resolve(spec)
        module = importlib.import_module(spec.target.partition(":")[0], __package__)
        for name in ("_CONTROL_ORACLE_STEPS", "_CONTROL_ORACLE_PROGRESS_INTERVAL", "_CONTROL_ORACLE_LR"):
            if hasattr(module, name):
                setattr(module, name, globals()[name])
        return int(target(args, config) or 0)
    raise RuntimeError(f"no static v2 command handler for {args.domain}/{args.action}")


def __getattr__(name: str):
    if name == "_lightgen_seeds":
        value = getattr(importlib.import_module(".legacy.teacher", __package__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    return _dispatch(args)
