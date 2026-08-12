from __future__ import annotations

import argparse


def register_evidence(root: argparse._SubParsersAction) -> None:
    evidence = root.add_parser("evidence").add_subparsers(dest="action", required=True)
    summarize = evidence.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    rebind = evidence.add_parser("rebind-manifest")
    rebind.add_argument("--config", required=True)
    rebind.add_argument(
        "--kind", choices=("clean-target", "teacher-target"), required=True
    )
    rebind.add_argument("--source-manifest", required=True)
    rebind.add_argument("--output", required=True)
    rebind.add_argument("--dataset", default="")
    rebind.add_argument("--teacher-manifest", default="")
    rebind.add_argument("--legacy-train-config", default="")
    supersede = evidence.add_parser("supersede-reachability")
    supersede.add_argument("--config", required=True)
    supersede.add_argument("--old-evidence", required=True)

