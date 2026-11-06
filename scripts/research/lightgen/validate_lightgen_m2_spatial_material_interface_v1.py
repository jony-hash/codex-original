"""Frozen B-route material interface diagnostic. No fitting or optimizer exists here."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.lightgen.image_readout import resize_view
from ldm_project.v2.application.research.lightgen.spatial_material_interface import (
    SpatialMaterialInterface, SyntheticResponse, WeightedModeProjector, diagnostic_decision,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl, ResearchRunIdentity,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from scripts.research.lightgen import execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref


class DiagnosticStop(RuntimeError):
    """A user-specified numerical stop, rather than an arbitrary software exception."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticStop(message)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def relative(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a-b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-30)).double()


def summary(x: torch.Tensor) -> dict:
    x = x.detach().double().cpu()
    return {"mean": x.mean().item(), "min": x.min().item(), "max": x.max().item(), "rows": x.tolist()}


def make_model(cp: dict, grid: int, device: torch.device):
    model = base.SmallMatchedAdapter(grid, base.GAUSSIAN_OLS_MODEL).to(device)
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    return model


def frozen_logits(evaluator, views: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Preserve the saved cohort's inference batch shape, including floating arithmetic."""
    return torch.cat([evaluator(chunk) for chunk in views.split(batch_size)])


def frozen_reemit(operator, coefficients: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Match the old separable contractions' batch shape, not only their algebra."""
    return torch.cat([operator.reemit_coefficients(chunk) for chunk in coefficients.split(batch_size)])


def independent_qr_projection(modes: torch.Tensor, field: torch.Tensor, area: float) -> np.ndarray:
    """Independent reference: convert before weighting so float32 cannot quantize W^0.5 E."""
    weighted = modes.detach().cpu().numpy().astype(np.complex128)*math.sqrt(area)
    q, _ = np.linalg.qr(weighted)
    rhs = field.detach().cpu().numpy().astype(np.complex128).ravel()*math.sqrt(area)
    return q @ (q.conj().T @ rhs)/math.sqrt(area)


def detector(model, field: torch.Tensor, batch: int) -> torch.Tensor:
    runtime = model.runtime
    require(tuple(field.shape[1:]) == (1, 100, 100), "GENERATOR_PLANE_SHAPE_MISMATCH")
    results = []
    for chunk in field.split(batch):
        incoming = ComplexField(chunk.to(torch.complex64), runtime.generator_entrance_plane,
                                OperationProvenance("synthetic_material_Q3", "Q2", None,
                                                    {"P_post": "identity_simulation_assumption"}))
        output, _, _ = runtime.generator(incoming, "matched_generated_generator")
        output = ComplexField(output.tensor, runtime.generator_exit_plane,
                              OperationProvenance("lightgen_generator_exit", output.plane.plane_id,
                                                  None, {"generator_layers": 2}))
        camera = runtime.output_relay(output)
        area = math.prod(runtime.camera_plane.pixel_pitch_m)
        results.append(camera.tensor.abs().square()*area)
    return torch.cat(results)


def fine_basis(device: torch.device) -> torch.Tensor:
    # Independent analytic Gaussian sampling at 1 um on the same 200 um FOV.
    coord = (torch.arange(200, device=device, dtype=torch.float64)+.5)*1e-6-100e-6
    centers = (torch.arange(10, device=device, dtype=torch.float64)-4.5)*20e-6
    axis = (2/(math.pi*(3.5e-6)**2))**.25 * torch.exp(-((coord[None]-centers[:, None])/3.5e-6)**2)
    return torch.einsum("ay,bx->yxab", axis, axis).reshape(40000, 100).to(torch.complex128)


def run(config: dict, output: Path) -> dict:
    started = time.time()
    result = {"schema": "m2_spatial_material_diagnostics.v1", "config": config,
              "status": "PREFLIGHT", "gamma": "NOT_YET_IMPLEMENTED",
              "optimizer_updates": 0, "parameter_selection_by_metrics": False,
              "scientific_attempt_opened": False, "stages_completed": [],
              "phase_proxy": "NOT_IMPLEMENTED_NOT_ASSUMED"}
    staging = output.parent / (output.name+"_preflight")
    save_json(staging/"m2_spatial_material_diagnostics.json", result)
    control = None
    try:
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        device = torch.device("cuda")
        cp = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        pack = torch.load(config["cohort"], map_location="cpu", weights_only=False)
        require(cp["completed_global_step"] == 45000 and cp["completed_updates"] == 10000
                and cp["branch"] == "C", "BASELINE_MODEL_IDENTITY_MISMATCH")
        require(pack["global_step"] == 45000 and pack["method"] == "G4"
                and pack["stream"] == 0 and pack["model"] == "C_seed_2026101501"
                and pack["classifier_based_sample_selection"] is False
                and pack["per_image_normalization"] is False, "COHORT_IDENTITY_MISMATCH")
        optimizer_steps = sorted({int(s["step"]) for s in cp["optimizer"]["state"].values() if "step" in s})
        require(optimizer_steps == [45000], "BASELINE_OPTIMIZER_LINEAGE_MISMATCH")
        selected = [int(i) for k in range(10) for i in
                    torch.where(pack["requested_labels"] == k)[0][:config["population"]["per_class"]]]
        require(len(selected) == 40, "FROZEN_POPULATION_MISMATCH")
        labels = pack["requested_labels"][selected].to(device)
        z = pack["ols_coefficients"][selected].to(device)
        require(z.shape == (40, 100) and z.dtype == torch.complex64, "OLS_INPUT_FORMAT_MISMATCH")
        model = make_model(cp, config["propagation_grid"], device)
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        gain = cp["camera_gain"].to(device=device, dtype=torch.float32)
        ed = base._load_evaluator(Path(config["ed_checkpoint"]), device)
        ec, ec_meta = decoder_ref._load_ec_evaluator(Path(config["ec_checkpoint"]), device)
        require(all(not p.requires_grad for obj in (model, ed, ec) for p in obj.parameters()),
                "FROZEN_MODEL_OR_EVALUATOR_VIOLATION")
        runtime = model.runtime
        plane = runtime.generator_entrance_plane
        require(plane.pixel_pitch_m == (2e-6, 2e-6), "GENERATOR_PLANE_SAMPLING_MISMATCH")
        area = math.prod(plane.pixel_pitch_m)
        operator = runtime.ols.operator
        source_plane = operator.input_plane
        require(all(getattr(source_plane, key) == getattr(plane, key)
                    for key in ("shape", "pixel_pitch_m", "wavelength_m", "z_m", "origin_m")),
                "GENERATOR_PLANE_COORDINATE_MISMATCH")
        coefficients = torch.cat([operator.transmit_coefficients(chunk) for chunk in z.split(config["batch_size"])])
        incoming = frozen_reemit(operator, coefficients, config["batch_size"]).unsqueeze(1)
        old_raw = torch.cat([model.forward_from_ols(chunk)["physical_w_per_pixel"]
                             for chunk in z.split(config["batch_size"])])
        replay_view = resize_view(old_raw*gain)
        replay_ed = frozen_logits(ed, replay_view, config["batch_size"])
        replay_ec = frozen_logits(lambda view: decoder_ref._ec_logits(ec, view), replay_view, config["batch_size"])
        tol = config["tolerances"]
        replay = {}
        for name, actual, saved in (
            ("view28", replay_view, pack["images"][selected].to(device)),
            ("ed_logits", replay_ed, pack["ed_logits"][selected].to(device)),
            ("ec_logits", replay_ec, pack["ec_logits"][selected].to(device))):
            passed = torch.allclose(actual, saved, rtol=tol["replay_rtol"], atol=tol["replay_atol"])
            replay[name] = {"passed": passed, "max_abs_error": (actual-saved).abs().max().item()}
        native_index = {int(v): i for i, v in enumerate(pack["native_capture_indices"])}
        native_positions = [i for i, source in enumerate(selected) if source in native_index]
        if native_positions:
            actual = (old_raw*gain).clamp(0, 1)[native_positions]
            saved = pack["native_capture_images"][[native_index[selected[i]] for i in native_positions]].to(device)
            replay["native200"] = {"passed": torch.allclose(actual, saved, rtol=tol["replay_rtol"],
                                                          atol=tol["replay_atol"]),
                                   "count": len(native_positions),
                                   "max_abs_error": (actual-saved).abs().max().item()}
        result["baseline"] = {"id": config["baseline_id"], "checkpoint_model_identity": cp["model_identity"],
                              "completed_global_step": cp["completed_global_step"],
                              "optimizer_steps": optimizer_steps, "selected_positions": selected,
                              "requested_labels": labels.cpu().tolist(),
                              "component_ids": pack["component_ids"][selected].tolist(),
                              "proposal_ids": pack["proposal_ids"][selected].tolist(),
                              "sample_identity": pack["sample_identity"], "basis": runtime.ols.mode_identity(),
                              "camera_gain": gain.item(), "replay": replay,
                              "ec_metadata": ec_meta, "legacy_mfd_m": runtime.config.physical_optics.fiber_mfd_m,
                              "plane": str(plane)}
        require(all(item["passed"] for item in replay.values()), "BASELINE_REPLAY_FAILED")
        require(torch.equal(replay_ed.argmax(1), pack["ed_predictions"][selected].to(device))
                and torch.equal(replay_ec.argmax(1), pack["ec_predictions"][selected].to(device)),
                "BASELINE_PREDICTION_REPLAY_FAILED")
        new_bare = detector(model, incoming, config["batch_size"])
        result["entrypoint_equivalence"] = {"bitwise_equal": torch.equal(new_bare, old_raw),
                                            "max_abs_power_w_per_pixel": (new_bare-old_raw).abs().max().item(),
                                            "relative_max": relative(new_bare, old_raw).max().item()}
        require(torch.equal(new_bare, old_raw), "OLD_B_ENTRYPOINT_EQUIVALENCE_FAILED")
        result["stages_completed"] += ["S0", "S1"]
        intensity = incoming.abs().square().double()
        reference = (intensity.square().sum()/intensity.sum()).item()
        interface = SpatialMaterialInterface(reference, runtime.config.physical_optics.fiber_mfd_m,
                                             lambda c: frozen_reemit(operator, c, config["batch_size"]))
        modes = operator.reemit_coefficients(torch.eye(100, device=device, dtype=torch.complex128)).flatten(1).T
        projector = WeightedModeProjector(modes, area)
        s4_response = SyntheticResponse(**config["s4_response"])
        m1 = interface.apply("M1", incoming, coefficients, s4_response)
        projection_checks = {}
        for name, field in (("original", incoming),
                            ("uniform_M0", interface.apply("M0", incoming, coefficients,
                             SyntheticResponse("linear", proxy_linear_transmission=.8))), ("M1", m1)):
            metrics = projector.metrics(field)
            projection_checks[name] = {"eta_max": metrics["eta_perp"].max().item(),
                                       "closure_max": metrics["closure_relative"].max().item()}
        # Independent QR projection, including an out-of-span material response.
        probe = interface.apply("M2", incoming[:1], coefficients[:1], s4_response)
        projected_probe, _ = projector.project(probe)
        qr = independent_qr_projection(modes, probe, area)
        qr_error = float(np.linalg.norm(qr-projected_probe.cpu().numpy().ravel())/np.linalg.norm(qr))
        result["projection_validation"] = {"checks": projection_checks, "independent_qr_relative": qr_error,
                                           "gram_condition": torch.linalg.cond(projector.gram).item(),
                                           "gram_identity_max_error": (projector.gram-torch.eye(100, device=device)).abs().max().item(),
                                           "status": "PENDING"}
        require(all(c["eta_max"] <= tol["subspace_eta_max"] and c["closure_max"] <= tol["power_closure_max"]
                    for c in projection_checks.values()) and qr_error <= tol["qr_relative"],
                "PROJECTION_VALIDATION_FAILED")
        result["projection_validation"]["status"] = "PASSED"
        result["reference_intensity_w_m2"] = reference
        result["stages_completed"] += ["S2", "S3"]
        pin = projector.metrics(incoming)["power_w"]
        fine_v = fine_basis(device)
        fine_projector = WeightedModeProjector(fine_v, 1e-12)
        fine_in = (coefficients.to(torch.complex128) @ fine_v.T).reshape(40, 1, 200, 200)
        reference_positions = list(range(0, 40, 4))
        result["reference_positions_in_selected"] = reference_positions
        # Real CUDA optical preflight completes before the scientific directory opens.
        first_full = interface.apply("M2", incoming[:1], coefficients[:1], s4_response)
        first_projection, _ = projector.project(first_full)
        first_raw = detector(model, torch.cat((first_full, first_projection.to(torch.complex64))), 1)
        require(torch.isfinite(first_raw).all().item(), "NONFINITE_PRODUCTION_PREFLIGHT")
        identity = ResearchRunIdentity(
            task_id=config["task_id"], run_id=uuid.uuid4().hex, trajectory_id=config["baseline_id"],
            semantic_authority_id="USER_APPROVED_M2_INTERFACE_PACKAGE_20261026",
            data_role=config["data_role"], config_identity=config,
            model_identity=cp["model_identity"], optimizer_identity="NO_OPTIMIZER_NO_UPDATES",
            sampler_identity={"saved_positions": selected, "sample_identity": pack["sample_identity"]},
            execution_source_semantics_id="M2_STATIC_LOCAL_AMPLITUDE_FROZEN_GENERATOR_V1")
        control = ResearchRunControl(identity, authorization_id=identity.semantic_authority_id)
        receipt = control.make_ready_receipt(
            correctness={"passed": True, "baseline_replay": replay, "projection": result["projection_validation"]},
            resource={"passed": True, "cuda_forward_finite": True, "gpu": torch.cuda.get_device_name(0)},
            implementation={"passed": True, "selected_engine": "frozen_gaussian_m2_v1"},
            selected_engine="frozen_gaussian_m2_v1")
        control.mark_preflight_ready(receipt)
        control.open_scientific_attempt(output, "attempt_01", receipt)
        result.update({"scientific_attempt_opened": True, "run_identity": control.identity.to_dict(),
                       "ready_receipt": receipt.to_dict(), "gpu": torch.cuda.get_device_name(0),
                       "status": "RUNNING", "cases": []})
        save_json(output/"m2_spatial_material_diagnostics.json", result)
        reference_model = make_model(cp, config["reference_grid"], device)
        saved_tensors = {"selected_positions": selected, "requested_labels": labels.cpu(),
                        "incoming": incoming.cpu(), "bare_view": replay_view.cpu()}

        def evaluate(field: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor]:
            raw = detector(model, field, config["batch_size"])
            view = resize_view(raw*gain)
            ed_logits = frozen_logits(ed, view, config["batch_size"])
            ec_logits = frozen_logits(lambda chunk: decoder_ref._ec_logits(ec, chunk), view, config["batch_size"])
            diversity = [torch.pdist(view[labels == k].flatten(1)).mean()/math.sqrt(784) for k in range(10)]
            pred = ed_logits.argmax(1)
            accuracy = (pred == labels).double().mean().item()
            other = ed_logits.clone()
            other[torch.arange(40, device=device), labels] = -torch.inf
            margin = ed_logits[torch.arange(40, device=device), labels]-other.max(1).values
            metrics = {"semantic_accuracy_ED": accuracy, "conditional_obedience_ED": accuracy,
                       "semantic_accuracy_EC": (ec_logits.argmax(1) == labels).double().mean().item(),
                       "requested_margin_ED": summary(margin), "predictions_ED": pred.cpu().tolist(),
                       "predictions_EC": ec_logits.argmax(1).cpu().tolist(),
                       "within_class_pairwise_RMS": [float(d) for d in diversity],
                       "diversity_mean": torch.stack(diversity).mean().item(),
                       "capture_power_w": summary(raw.double().flatten(1).sum(1)),
                       "raw_detector_vs_bare": summary(relative(raw, old_raw)),
                       "view_vs_bare": summary(relative(view, replay_view))}
            return metrics, raw, view

        def measure(response: SyntheticResponse, stage: str) -> dict:
            print(json.dumps({"stage": stage, "response": asdict(response)}, ensure_ascii=False), flush=True)
            full = interface.apply("M2", incoming, coefficients, response)
            legacy = interface.apply("M1", incoming, coefficients, response)
            parallel, _ = projector.project(full)
            field_metrics = projector.metrics(full)
            legacy_metrics = projector.metrics(legacy)
            require((field_metrics["power_w"] <= pin*(1+tol["passive_relative"])).all().item(), "PASSIVE_ENERGY_GAIN")
            require(legacy_metrics["eta_perp"].max().item() <= tol["subspace_eta_max"], "PROJECTION_VALIDATION_FAILED")
            require(field_metrics["closure_relative"].max().item() <= tol["power_closure_max"], "POWER_CLOSURE_FAILED")
            fine_out = interface.apply("M2", fine_in, coefficients, response)
            fine_metrics = fine_projector.metrics(fine_out)
            power_error = ((fine_metrics["power_w"]-field_metrics["power_w"]).abs()/fine_metrics["power_w"]).max().item()
            eta_error = (fine_metrics["eta_perp"]-field_metrics["eta_perp"]).abs().max().item()
            result["latest_numerical_check"] = {"response": asdict(response),
                                                "power_relative_max": power_error,
                                                "eta_absolute_max": eta_error}
            require(power_error <= tol["quadrature_power_relative"] and eta_error <= tol["quadrature_eta_absolute"],
                    "INTERFACE_NOT_YET_NUMERICALLY_STABLE_QUADRATURE")
            linear_t = (field_metrics["power_w"].sum()/pin.sum()).item()
            linear = interface.apply("M0", incoming, coefficients,
                                     SyntheticResponse("linear", proxy_linear_transmission=linear_t))
            variants = {"Bare": incoming, "Linear_equal_loss": linear, "M1": legacy,
                        "M2_full": full, "M2_projected": parallel.to(torch.complex64)}
            # Supplementary control: isolates field shape from projection's power removal.
            matched = parallel * torch.sqrt(field_metrics["power_w"]/field_metrics["parallel_power_w"])[:, None, None, None]
            variants["projected_equal_power_supplement"] = matched.to(torch.complex64)
            measurements, raws, views = {}, {}, {}
            for name, field in variants.items():
                measurements[name], raws[name], views[name] = evaluate(field)
            refs = reference_positions
            full_ref = detector(reference_model, full[refs], 1)
            proj_ref = detector(reference_model, parallel[refs], 1)
            matched_ref = detector(reference_model, matched[refs], 1)
            full_error = relative(full_ref, raws["M2_full"][refs]).max().item()
            proj_error = relative(proj_ref, raws["M2_projected"][refs]).max().item()
            matched_error = relative(matched_ref, raws["projected_equal_power_supplement"][refs]).max().item()
            stable = max(full_error, proj_error, matched_error) <= tol["reference_output_relative"]
            output_delta = relative(raws["M2_full"], raws["M2_projected"])
            reference_delta = relative(full_ref, proj_ref)
            case = {"stage": stage, "response": asdict(response),
                    "input_power_w": summary(pin), "output_power_w": summary(field_metrics["power_w"]),
                    "eta_perp": summary(field_metrics["eta_perp"]),
                    "projected_power_w": summary(field_metrics["parallel_power_w"]),
                    "M1_eta_max": legacy_metrics["eta_perp"].max().item(),
                    "M1_M2_field_relative": summary(relative(legacy, full)),
                    "M1_M2_intensity_relative": summary(relative(legacy.abs().square(), full.abs().square())),
                    "generator_input_full_projected_relative": summary(relative(full, parallel)),
                    "projection_cast_relative": summary(relative(parallel.to(torch.complex64), parallel)),
                    "full_projected_raw_detector_relative": summary(output_delta),
                    "full_projected_view_relative": summary(relative(views["M2_full"], views["M2_projected"])),
                    "full_projected_ED_prediction_changes": int((torch.tensor(measurements["M2_full"]["predictions_ED"])
                                                                 != torch.tensor(measurements["M2_projected"]["predictions_ED"])).sum()),
                    "full_equal_power_projected_detector_relative": summary(relative(raws["M2_full"], raws["projected_equal_power_supplement"])),
                    "equal_loss_power_transmission": linear_t,
                    "quadrature": {"power_relative_max": power_error, "eta_absolute_max": eta_error,
                                   "native_pitch_m": 2e-6, "reference_pitch_m": 1e-6},
                    "padding_reference": {"full_relative_max": full_error, "projected_relative_max": proj_error,
                                          "equal_power_projected_relative_max": matched_error,
                                          "full_projected_delta": summary(reference_delta), "stable": stable},
                    "variants": measurements}
            result["latest_measured_case"] = case
            require(stable, "INTERFACE_NOT_YET_NUMERICALLY_STABLE_PADDING")
            response_result, gate = diagnostic_decision(field_metrics["eta_perp"].mean().item(),
                                                       case["M1_M2_field_relative"]["mean"],
                                                       min(output_delta.mean().item(), reference_delta.mean().item()), stable,
                                                       difference_tolerance=tol["output_difference"])
            case.update({"result": response_result, "gate": gate})
            result["cases"].append(case)
            saved_tensors[str(len(result["cases"]))] = {name: view.cpu() for name, view in views.items()}
            if stage == "S4":
                saved_tensors["S4_full"] = full.cpu()
                saved_tensors["S4_projected"] = parallel.cpu()
            save_json(output/"m2_spatial_material_diagnostics.json", result)
            return case

        measure(s4_response, "S4")
        result["stages_completed"].append("S4")
        for spec in config["sweep"]:
            response = SyntheticResponse(**spec)
            if response != s4_response:
                measure(response, "S5")
        result["stages_completed"].append("S5")
        require(all(torch.equal(v.detach().cpu(), snapshot[k]) for k, v in model.state_dict().items()),
                "OLD_B_MODEL_STATE_MUTATED")
        final_raw = torch.cat([model.forward_from_ols(chunk)["physical_w_per_pixel"] for chunk in z.split(config["batch_size"])])
        require(torch.equal(final_raw, old_raw), "OLD_B_OUTPUT_MUTATED")
        result.update({"status": "COMPLETED", "baseline_unchanged": True,
                       "gate": ("M2_INTERFACE_VALID_TASK_RELEVANT" if any(c["result"] == "C" for c in result["cases"])
                                else "M2_INTERFACE_VALID_FIELD_EFFECT_ONLY" if any(c["result"] == "B" for c in result["cases"])
                                else "M2_INTERFACE_VALID_NO_MATERIAL_DISTINCTION"),
                       "elapsed_seconds": time.time()-started,
                       "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()})
        control.complete()
        torch.save(saved_tensors, output/"m2_frozen_diagnostic_tensors.pt")
        save_json(output/"m2_spatial_material_diagnostics.json", result)
        return result
    except DiagnosticStop as exc:
        result.update({"status": "STOPPED", "stop_reason": str(exc),
                       "gate": "M2_RESULTS_INCONCLUSIVE" if "NUMERICALLY_STABLE" in str(exc) else "M2_INTERFACE_INVALID",
                       "elapsed_seconds": time.time()-started})
        if control is not None and result["scientific_attempt_opened"]:
            control.invalid_harness()
        save_json((output if output.exists() else staging)/"m2_spatial_material_diagnostics.json", result)
        print(json.dumps({"status": result["status"], "stop_reason": str(exc), "gate": result["gate"]}), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError(f"new diagnostic requires a fresh attempt: {args.output}")
    with torch.inference_mode():
        result = run(config, args.output)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}), flush=True)


if __name__ == "__main__":
    main()
