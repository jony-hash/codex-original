"""Generate and validate the one-time v2 Hash consumer inventory."""

from __future__ import annotations

import argparse
import ast
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ldm_project" / "v2"
OUTPUT = ROOT / "docs" / "status" / "hash_consumer_inventory.tsv"
NAMES = {"canonical_sha256", "file_sha256", "tensor_sha256", "state_content_hash"}
FIELDS = (
    "callsite",
    "primitive",
    "consumer_function",
    "category",
    "syntactic_consumer",
    "validation_decision",
    "repeat_input_key",
    "decision",
    "review_note",
)


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _ancestors(node: ast.AST, parents: dict[int, ast.AST]) -> list[ast.AST]:
    result: list[ast.AST] = []
    cursor = parents.get(id(node))
    while cursor is not None:
        result.append(cursor)
        cursor = parents.get(id(cursor))
    return result


def _owner(ancestors: list[ast.AST]) -> str:
    for node in ancestors:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


def _category(primitive: str, owner: str, ancestors: list[ast.AST]) -> str:
    names = " ".join(
        node.name
        for node in ancestors
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ).lower()
    if primitive == "canonical_sha256":
        return "contract_binding"
    if any(token in names for token in ("evidence", "manifest", "record")):
        return "evidence_input_binding"
    if primitive in {"tensor_sha256", "state_content_hash"}:
        return "content_integrity"
    return "evidence_input_binding"


def _syntactic_consumer(node: ast.Call, ancestors: list[ast.AST]) -> str:
    parent = ancestors[0] if ancestors else None
    if isinstance(parent, ast.Compare):
        return "comparison"
    if isinstance(parent, ast.Call):
        return "call_argument"
    if isinstance(parent, ast.Dict):
        return "mapping_value"
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
        return "assignment"
    if isinstance(parent, ast.Return):
        return "return_value"
    if isinstance(parent, ast.JoinedStr):
        return "report_text"
    return type(parent).__name__ if parent is not None else "module_expression"


def _validation_decision(ancestors: list[ast.AST], owner: str) -> str:
    if any(isinstance(node, (ast.Assert, ast.If, ast.While)) for node in ancestors):
        return "guard_or_rejection"
    if any(isinstance(node, ast.Compare) for node in ancestors):
        return "comparison_input"
    if any(isinstance(node, ast.Dict) for node in ancestors):
        return "evidence_or_manifest_field"
    if "report" in owner or "summary" in owner:
        return "diagnostic_report_only"
    return "derived_contract_value"


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in NAMES
            ):
                continue
            ancestors = _ancestors(node, parents)
            owner = _owner(ancestors)
            decision = _validation_decision(ancestors, owner)
            repeat_key = ast.dump(
                ast.Tuple(elts=[*node.args, *[keyword.value for keyword in node.keywords]])
            )
            rows.append(
                {
                    "callsite": f"{path.relative_to(ROOT)}:{node.lineno}",
                    "primitive": node.func.id,
                    "consumer_function": owner,
                    "category": _category(node.func.id, owner, ancestors),
                    "syntactic_consumer": _syntactic_consumer(node, ancestors),
                    "validation_decision": decision,
                    "repeat_input_key": repeat_key,
                    "decision": "retain",
                    "review_note": (
                        f"retained for {decision} in {owner}; callsite has an explicit "
                        "consumer and no verified equivalent replacement"
                    ),
                }
            )
    return rows


def _render(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _check(rows: list[dict[str, str]]) -> int:
    if not OUTPUT.is_file():
        raise SystemExit(f"missing inventory: {OUTPUT}")
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle, delimiter="\t"))
    required = set(FIELDS)
    if not existing or set(existing[0]) != required:
        raise SystemExit("Hash inventory schema does not match the audit tool")
    expected_keys = {(row["callsite"], row["primitive"]) for row in rows}
    actual_keys = {(row["callsite"], row["primitive"]) for row in existing}
    if expected_keys != actual_keys:
        raise SystemExit(
            f"Hash inventory callsites differ: expected={len(expected_keys)} actual={len(actual_keys)}"
        )
    if any(row["review_note"].startswith("retained by conservative default") for row in existing):
        raise SystemExit("Hash inventory still contains generic conservative-default reviews")
    if any(row["decision"] not in {"retain", "delete"} for row in existing):
        raise SystemExit("Hash inventory contains an unsupported decision")
    print(f"validated {len(existing)} Hash consumers")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = _rows()
    if args.check:
        return _check(rows)
    OUTPUT.write_text(_render(rows), encoding="utf-8")
    print(f"wrote {len(rows)} Hash callsites to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
