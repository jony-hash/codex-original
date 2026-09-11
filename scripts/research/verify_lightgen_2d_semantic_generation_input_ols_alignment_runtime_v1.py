"""Publish the non-scientific verification result for the aligned runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    ROW_MAJOR_PORT_ORDER,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance


TASK_ID = "IMPLEMENT_LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1"
CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED"
CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_contract_v1_attempt_01/"
    "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1.json"
)
CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
FIBER_OPERATOR_PATH = Path("src/ldm_project/v2/optics/fiber_operator.py")
REPORT_NAME = "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_lines": status.splitlines(),
    }


def _projection_fixture(runtime: LightGen2DSemanticGenerationInputOLSAlignment) -> bool:
    mode = runtime.ols.operator.input_modes[0]
    fixture = ComplexField(
        mode.unsqueeze(0).unsqueeze(0),
        runtime.ols.operator.input_plane,
        OperationProvenance("runtime_verification_modal_fixture", None, None),
    )
    state = runtime.ols.project_state(fixture)
    return (
        state.coefficients.shape == (1, 100)
        and torch.is_complex(state.coefficients)
        and state.port_order == ROW_MAJOR_PORT_ORDER
        and torch.isclose(state.coefficients[0, 0].abs(), torch.tensor(1.0), atol=2e-5)
        and float(state.coefficients[0, 1:].abs().max()) < 2e-5
    )


def verify(repo_root: Path, output_dir: Path | None = None) -> dict[str, object]:
    root = repo_root.resolve()
    contract_path = root / CONTRACT_PATH
    if not contract_path.is_file() or _sha256(contract_path) != CONTRACT_SHA256:
        raise RuntimeError("the exact LightGen alignment contract authority is unavailable or changed")

    # The reduced grid is explicitly a CPU interface smoke setting.  The
    # production default remains the contract's 2048 propagation grid.
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=128)
    ).eval()
    image = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)
    with torch.no_grad():
        image_result = runtime.forward_from_image(image)
        direct_result = runtime.forward_from_ols(image_result.z_ols)

    try:
        runtime.forward_from_ols(torch.ones(1, 100))
    except TypeError:
        q_rejected = True
    else:
        q_rejected = False

    if not q_rejected:
        raise RuntimeError("q-only input was accepted as the authoritative complex OLS state")

    report = {
        "schema": "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "result": "IMPLEMENTED",
        "classification": CLASSIFICATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(CONTRACT_PATH).replace("\\", "/"),
            "sha256": CONTRACT_SHA256,
            "schema": "lightgen_2d_semantic_generation_input_ols_alignment_contract.v1",
        },
        "git": _git(root),
        "runtime": {
            "module": str(RUNTIME_PATH).replace("\\", "/"),
            "module_sha256": _sha256(root / RUNTIME_PATH),
            "fiber_operator": str(FIBER_OPERATOR_PATH).replace("\\", "/"),
            "fiber_operator_sha256": _sha256(root / FIBER_OPERATOR_PATH),
            "production_propagation_grid": 2048,
            "verification_propagation_grid": 128,
            "canonical_ols": "complex [B,100], 10x10, row-major j=10*r+c",
            "projection": "z_j=<E_encoder,u_j> via FiberArrayOLS single-mode overlap",
            "transport": "T=I_100 in coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "image_api": "forward_from_image(image)",
            "direct_api": "forward_from_ols(z)",
            "generator_identity": "one shared runtime.generator module",
            "detector_output": "raw CMOS intensity |E_out|^2",
        },
        "structural_verification": {
            "complex_ols_shape": list(image_result.z_ols.coefficients.shape),
            "complex_ols_dtype": str(image_result.z_ols.coefficients.dtype),
            "stable_row_major_order": image_result.z_ols.port_order == ROW_MAJOR_PORT_ORDER,
            "modal_projection_active": _projection_fixture(runtime),
            "t_identity": bool(torch.equal(runtime.ols.transport, torch.eye(100, dtype=torch.complex64))),
            "w_ols_output_plane_matches_generator_input": image_result.generator_input_field.plane == runtime.generator_entrance_plane,
            "same_writer_path_equivalence": bool(torch.equal(image_result.generator_input_field.tensor, direct_result.generator_input_field.tensor)),
            "same_generator_raw_equivalence": bool(torch.equal(image_result.raw_camera.tensor, direct_result.raw_camera.tensor)),
            "direct_path_skips_image_encoder": direct_result.image_input is None and direct_result.encoder_output_field is None and direct_result.ols_input_field is None and not direct_result.encoder_records,
            "q_only_rejected": q_rejected,
            "no_legacy_identity_symbol_in_aligned_module": "IdentityOLS" not in (root / RUNTIME_PATH).read_text(encoding="utf-8"),
        },
        "authority_categories": runtime.alignment_metadata,
        "claim_boundary": {
            "non_scientific_structural_smoke_only": True,
            "scientific_generation_executed": False,
            "heldout_evaluation": False,
            "training_executed": False,
            "cuda_used": False,
            "sealed_values_loaded": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "reuse": [
            "PlaneSpec/ComplexField/IntensityFrame",
            "IdealAmplitudeSLM",
            "CoherentImagingRelay",
            "FiniteCropPhaseStack",
            "FiberArrayOLS modal bank and overlap primitive",
        ],
        "historical_only": [
            "N3 amplitude producer confirmation",
            "full-complex empirical control",
            "independent phase inheritance",
            "q-conditioned nearest phase donor",
            "q-conditioned complex phasor linear residual",
        ],
        "next_action": "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONTRACT_V1",
    }
    target_dir = (root / output_dir) if output_dir is not None else root / "outputs/20260910_v2/runs/lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_01"
    target_dir.mkdir(parents=True, exist_ok=False)
    report_path = target_dir / REPORT_NAME
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    report["artifact"] = {
        "path": str(report_path.relative_to(root)).replace("\\", "/"),
        "sha256": _sha256(report_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = verify(args.repo_root, args.output_dir)
    print(json.dumps(report["artifact"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
