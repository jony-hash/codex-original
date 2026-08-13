from __future__ import annotations

import argparse


def register_research(root: argparse._SubParsersAction) -> None:
    research = root.add_parser("research", help="diagnostic-only pre-registered research programs")
    research_programs = research.add_subparsers(dest="research_program", required=True)
    capacity_map = research_programs.add_parser("capacity-map")
    capacity_actions = capacity_map.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "screen", "train-smoke", "confirm", "summarize", "diagnose", "qualify-shape", "closeout"):
        command = capacity_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist", "controlled_shapes", "fashion_mnist"), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cpu" if action == "closeout" else "cuda")
        if action not in {"validate", "summarize", "qualify-shape", "closeout"}:
            command.add_argument("--variant", required=True)
        else:
            command.add_argument("--variant")
        if action in {"screen", "train-smoke", "confirm", "diagnose", "qualify-shape"}:
            command.add_argument("--seed", type=int, default=7319 if action == "qualify-shape" else 1234)
        if action == "screen":
            command.add_argument("--preflight-only", action="store_true")
            command.add_argument("--stage", choices=("audit", "probe"), default="audit")
            command.add_argument("--prior-evidence")
        if action in {"train-smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "diagnose":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--source-evidence", required=True)
            command.add_argument("--shape-evaluator-checkpoint")
            command.add_argument("--shape-qualification-evidence")
        if action == "summarize":
            command.add_argument("--evidence-root")
        if action == "closeout":
            command.add_argument("--comparison-config", required=True)
            command.add_argument("--evidence-root", required=True)
    o0 = research_programs.add_parser("o0-feasibility")
    o0_actions = o0.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "controls", "probe", "smoke", "confirm", "final", "summarize", "closeout"):
        command = o0_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0", "C0", "C1", "C2", "C3"), required=action in {"preflight", "probe", "smoke", "confirm", "final"})
        if action in {"preflight", "probe", "smoke", "confirm"}:
            command.add_argument("--seed", type=int, default=1234)
        if action in {"probe", "smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "final":
            command.add_argument("--checkpoint", action="append", required=True)
            command.add_argument("--source-evidence", action="append", required=True)
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    ident = research_programs.add_parser("o0-identifiability")
    ident_actions = ident.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "audit-update", "probe", "summarize", "closeout"):
        command = ident_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0",), default="D0")
        if action in {"preflight", "audit-update", "probe"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "probe":
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    o0e = research_programs.add_parser("objective-o0")
    o0e_actions = o0e.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "calibrate", "screen", "confirm", "locked-final", "closeout"):
        command = o0e_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if action != "validate":
            command.add_argument("--evaluator-checkpoint", required=True)
            command.add_argument("--primary-qualification-evidence", required=True)
            command.add_argument("--secondary-evaluator-checkpoint", required=True)
            command.add_argument("--secondary-training-evidence", required=True)
            command.add_argument("--measurement-bundle", required=True)
            command.add_argument("--measurement-evidence", required=True)
        if action in {"preflight", "calibrate", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "preflight":
            command.add_argument("--validate-evidence", required=True)
            command.add_argument("--o0-i-evidence")
        if action == "calibrate":
            command.add_argument("--preflight-evidence", required=True)
        if action == "screen":
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
        if action == "locked-final":
            command.add_argument("--confirm-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
        if action == "closeout":
            command.add_argument("--source-evidence", required=True)
