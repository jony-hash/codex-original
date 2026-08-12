from __future__ import annotations

import argparse

from ._cli_schema_audit import register_audit
from ._cli_schema_evidence import register_evidence
from ._cli_schema_research import register_research
from ._cli_schema_student import register_student
from ._cli_schema_teacher import register_teacher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ldm_project.v2")
    root = parser.add_subparsers(dest="domain", required=True)
    register_research(root)
    register_teacher(root)
    register_student(root)
    register_audit(root)
    register_evidence(root)
    return parser
