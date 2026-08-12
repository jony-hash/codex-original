from __future__ import annotations

import argparse


def register_teacher(root: argparse._SubParsersAction) -> None:
    teacher = root.add_parser("teacher").add_subparsers(dest="action", required=True)
    for action, stage in (
        ("train-first-stage", "first_stage"),
        ("train-diffusion", "diffusion"),
    ):
        command = teacher.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--legacy-train-config", required=True)
    export = teacher.add_parser("export-targets")
    for name in (
        "config",
        "legacy_train_config",
        "teacher_manifest",
        "output",
        "target_manifest",
    ):
        export.add_argument("--" + name.replace("_", "-"), required=True)
    manifest = teacher.add_parser("manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--split", default="train")

