"""Read-only frozen-model M1* controls and existing-Gamma A-route forward audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.lightgen.image_readout import resize_view
from ldm_project.v2.application.research.lightgen.matched_m1_diagnostic import matched_mode_response
from ldm_project.v2.application.research.lightgen.material_nonlinearity import MaterialNonlinearity, MaterialParameters
from ldm_project.v2.application.research.lightgen.spatial_material_interface import (
    SpatialMaterialInterface, SyntheticResponse, WeightedModeProjector,
)
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity
from ldm_project.v2.application.research.shared.research_run_control import PreflightReceipt
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import validate_lightgen_m2_spatial_material_interface_v1 as previous
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot


def conditional_gate(mapping_present: bool, path_valid: bool, acceptance_defined: bool,
                     acceptance_passed: bool) -> str:
    if not mapping_present:
        return "A_ROUTE_BLOCKED_GAMMA_NOT_SPECIFIED"
    if not path_valid:
        return "A_ROUTE_NUMERICALLY_INVALID"
    if acceptance_defined and acceptance_passed:
        return "A_ROUTE_CONDITIONAL_REACHABILITY_SUPPORTED"
    return "A_ROUTE_FORWARD_PATH_VALID_CONDITIONAL_CONTROL_UNPROVEN"


def validate_a_trace(trace: dict, count: int) -> None:
    previous.require(all(trace.get(name) == count for name in ("SLM", "Encoder", "OLS_projection")),
                     "A_FRONTEND_NOT_ACTUALLY_EXECUTED")
    previous.require(trace.get("external_ols_injection", 0) == 0, "A_CONTAINS_EXTERNAL_OLS_BYPASS")


def independent_matched(coefficients, modes, area, reference, response) -> np.ndarray:
    z = coefficients.detach().cpu().numpy().astype(np.complex128)
    v = modes.detach().cpu().numpy().astype(np.complex128)
    factors = []
    for j in range(z.shape[1]):
        density = np.abs(v[:, j])**2
        x = np.abs(z[:, j, None])**2*density[None]/reference/response.proxy_saturation_scale
        if response.family == "saturation":
            t = np.sqrt(1-response.proxy_strength+response.proxy_strength*x/(1+x))
        elif response.family == "suppression":
            t = 1/np.sqrt(1+response.proxy_strength*x)
        elif response.family == "linear":
            t = np.full_like(x, math.sqrt(response.proxy_linear_transmission))
        else:
            t = np.ones_like(x)
        factors.append((density[None]*t*area).sum(1)/(density*area).sum())
    return np.stack(factors, axis=1)


def task_metrics(raw, labels, gain, ed, ec, batch=10) -> tuple[dict, torch.Tensor]:
    view = resize_view(raw*gain).float()
    logits_d = previous.frozen_logits(ed, view, batch)
    logits_c = previous.frozen_logits(lambda x: previous.decoder_ref._ec_logits(ec, x), view, batch)
    result = {}
    for name, logits in (("ED", logits_d), ("EC", logits_c)):
        pred = logits.argmax(1)
        matrix = torch.bincount(labels*10+pred, minlength=100).reshape(10, 10)
        result[name] = {"predictions": pred.cpu().tolist(), "logits": logits.cpu().tolist(),
                        "accuracy": (pred == labels).double().mean().item(),
                        "confusion_matrix": matrix.cpu().tolist(),
                        "recall_by_class": (matrix.diag()/matrix.sum(1)).cpu().tolist()}
    distances = [torch.pdist(view[labels == k].flatten(1)).double()/28 for k in range(10)]
    result["within_class_RMS"] = [previous.summary(d) for d in distances]
    result["diversity_mean"] = torch.stack([d.mean() for d in distances]).mean().item()
    result["capture_power_w"] = previous.summary(raw.double().flatten(1).sum(1))
    return result, view


def coeff_statistics(coefficients, labels) -> dict:
    z = coefficients.detach().cpu().to(torch.complex128)
    y = labels.detach().cpu()
    powers = z.abs().square()
    features = torch.view_as_real(z).reshape(len(z), -1)
    centered = features-features.mean(0)
    spectrum = torch.linalg.svdvals(centered).square()
    effective = spectrum.sum().square()/spectrum.square().sum().clamp_min(1e-300)
    class_rows, centers = [], []
    for k in range(10):
        values = z[y == k]
        p = powers[y == k]
        centers.append(values.mean(0))
        dist = torch.pdist(torch.view_as_real(values).reshape(len(values), -1))
        class_rows.append({"class": k, "coefficient_norm": previous.summary(values.norm(dim=1)),
                           "total_modal_power_w": previous.summary(p.sum(1)),
                           "mean_mode_power_w": p.mean(0).tolist(),
                           "pairwise_complex_distance_sqrt_w": previous.summary(dist)})
    centers = torch.stack(centers)
    between = torch.pdist(torch.view_as_real(centers).reshape(10, -1))
    share = powers.mean(0)/powers.mean(0).sum()
    return {"classes": class_rows, "centered_real_effective_dimension_participation_ratio": effective.item(),
            "between_class_centroid_distance_sqrt_w": previous.summary(between),
            "mean_mode_power_fraction": share.tolist(),
            "mode_occupancy_entropy_effective_count": torch.exp(-(share*share.clamp_min(1e-300).log()).sum()).item(),
            "modes_with_strictly_positive_mean_power": int((powers.mean(0) > 0).sum()),
            "distance_gauge": "complex Euclidean in saved coordinates, no global-phase alignment"}


def preserve_precision_detector(model, field, batch=10):
    runtime = model.runtime
    output = []
    for chunk in field.split(batch):
        incoming = ComplexField(chunk, runtime.generator_entrance_plane,
                                OperationProvenance("audited_M2_Q3", "Q2", None,
                                                    {"P_post": "identity_simulation_assumption"}))
        result, _, _ = runtime.generator(incoming, "matched_generated_generator")
        result = ComplexField(result.tensor, runtime.generator_exit_plane,
                              OperationProvenance("lightgen_generator_exit", result.plane.plane_id,
                                                  None, {"generator_layers": 2}))
        camera = runtime.output_relay(result)
        output.append(camera.tensor.abs().square()*math.prod(runtime.camera_plane.pixel_pitch_m))
    return torch.cat(output)


def run_matched(config, old, output, cp, cohort, ed, ec, control):
    device = torch.device("cuda")
    model = previous.make_model(cp, 2048, device)
    frozen = pilot._snapshot_state(model)
    indices = old["baseline"]["selected_positions"]
    z = cohort["ols_coefficients"][indices].to(device)
    labels = cohort["requested_labels"][indices].to(device)
    operator = model.runtime.ols.operator
    z_t = torch.cat([operator.transmit_coefficients(v) for v in z.split(10)])
    incoming = previous.frozen_reemit(operator, z_t, 10).unsqueeze(1)
    area = math.prod(model.runtime.generator_entrance_plane.pixel_pitch_m)
    modes = operator.reemit_coefficients(torch.eye(100, device=device, dtype=torch.complex128)).flatten(1).T
    projector = WeightedModeProjector(modes, area)
    reference = old["reference_intensity_w_m2"]
    interface = SpatialMaterialInterface(reference, old["baseline"]["legacy_mfd_m"],
                                         lambda c: previous.frozen_reemit(operator, c, 10))
    gain = cp["camera_gain"].to(device=device, dtype=torch.float32)
    bare_raw = torch.cat([model.forward_from_ols(chunk)["physical_w_per_pixel"] for chunk in z.split(10)])
    bare_metrics, bare_view = task_metrics(bare_raw, labels, gain, ed, ec)
    replay = {}
    for key, actual, saved in (("view28", bare_view, cohort["images"][indices].to(device)),
                               ("ED", torch.tensor(bare_metrics["ED"]["logits"], device=device), cohort["ed_logits"][indices].to(device)),
                               ("EC", torch.tensor(bare_metrics["EC"]["logits"], device=device), cohort["ec_logits"][indices].to(device))):
        replay[key] = {"passed": torch.allclose(actual, saved, rtol=2e-6, atol=5e-8),
                       "max_abs_error": (actual-saved).abs().max().item()}
    native_map = {int(row): j for j, row in enumerate(cohort["native_capture_indices"])}
    saved_native = cohort["native_capture_images"][[native_map[i] for i in indices]].to(device)
    replay["native200"] = {"passed": torch.equal((bare_raw*gain).clamp(0, 1), saved_native),
                           "max_abs_error": ((bare_raw*gain).clamp(0, 1)-saved_native).abs().max().item()}
    previous.require(all(v["passed"] for v in replay.values()), "B_BASELINE_REPLAY_FAILED")
    previous.require(torch.equal(previous.detector(model, incoming, 10), bare_raw), "B_ENTRYPOINT_NOT_IDENTICAL")
    first_response = SyntheticResponse(**config["matched_responses"][0])
    changed, factors = matched_mode_response(z_t, modes, (100, 100), area, interface, first_response)
    reference_a = independent_matched(z_t, modes, area, reference, first_response)
    formula_error = np.max(np.abs(reference_a-factors.cpu().numpy()))
    previous.require(formula_error <= 1e-10, "M1_MATCHED_FORMULA_FAILED")
    first_field = previous.frozen_reemit(operator, changed.to(z.dtype), 10).unsqueeze(1)
    previous.require(projector.metrics(first_field)["eta_perp"].max().item() <= 1e-10, "M1_MATCHED_PROJECTION_FAILED")
    tiny = previous.detector(model, first_field[:10], 10)
    previous.require(torch.isfinite(tiny).all().item(), "M1_MATCHED_CUDA_PREFLIGHT_FAILED")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "historical_B_replay": replay, "formula_max_abs_error": float(formula_error),
                     "matched_projection_eta_max": projector.metrics(first_field)["eta_perp"].max().item()},
        resource={"passed": True, "actual_cuda_forward_finite": True, "gpu": torch.cuda.get_device_name(0)},
        implementation={"passed": True, "selected_engine": "isolated_matched_m1_existing_gamma_v1"},
        selected_engine="isolated_matched_m1_existing_gamma_v1")
    control.mark_preflight_ready(receipt)
    control.open_scientific_attempt(output, "attempt_01", receipt)
    result = {"schema": "matched_m1_control_results.v1", "status": "RUNNING", "previous_result": config["previous_result"],
              "reference_intensity_w_m2": reference, "sample_positions": indices, "labels": labels.cpu().tolist(),
              "baseline_replay": replay, "run_identity": control.identity.to_dict(), "ready_receipt": receipt.to_dict(),
              "optimizer_updates": 0, "cases": []}
    tensors = {"labels": labels.cpu(), "input_coefficients": z.cpu(), "incoming": incoming.cpu(), "modes": modes.cpu()}
    pin = projector.metrics(incoming)["power_w"]
    old_tensors = torch.load(Path(config["previous_result"]).parent/"m2_frozen_diagnostic_tensors.pt",
                             map_location="cpu", weights_only=False)
    specs = [config["linear_negative_control"]]+config["matched_responses"]
    for index, spec in enumerate(specs):
        response = SyntheticResponse(**spec)
        print(json.dumps({"stage": "M1_MATCHED", "response": spec}), flush=True)
        full = interface.apply("M2", incoming, z_t, response)
        legacy = interface.apply("M1", incoming, z_t, response)
        matched_coeff, factors = matched_mode_response(z_t, modes, (100, 100), area, interface, response)
        matched = previous.frozen_reemit(operator, matched_coeff.to(z.dtype), 10).unsqueeze(1)
        expected_a = independent_matched(z_t, modes, area, reference, response)
        error = float(np.max(np.abs(expected_a-factors.cpu().numpy())))
        previous.require(error <= 1e-10 and bool((factors >= 0).all()) and bool((factors <= 1+1e-12).all()),
                         "M1_MATCHED_FORMULA_OR_PASSIVITY_FAILED")
        # Validate the legacy wrapper against an explicit call of the unchanged old function.
        scale = reference/1e4*response.proxy_saturation_scale
        peak = 2*z_t.abs().square()/(math.pi*(old["baseline"]["legacy_mfd_m"]/2)**2)/1e4
        if response.family == "saturation":
            params = MaterialParameters("saturable_absorption", 1-response.proxy_strength,
                                        response.proxy_strength, scale)
        elif response.family == "suppression":
            params = MaterialParameters("saturable_absorption_plus_RSA", 1, 0, scale, scale/response.proxy_strength)
        else:
            params = MaterialParameters("saturable_absorption", response.proxy_linear_transmission, 0, scale)
        legacy_coeff = MaterialNonlinearity().forward(z_t, peak, params)
        legacy_reference = previous.frozen_reemit(operator, legacy_coeff, 10).unsqueeze(1)
        previous.require(torch.equal(legacy, legacy_reference), "LEGACY_M1_WRAPPER_CHANGED")
        metrics = {name: projector.metrics(field) for name, field in
                   (("Bare", incoming), ("M1_legacy", legacy), ("M1_matched", matched), ("M2_full", full))}
        for name in ("Bare", "M1_legacy", "M1_matched"):
            previous.require(metrics[name]["eta_perp"].max().item() <= 1e-10, "PROJECTION_NEGATIVE_CONTROL_FAILED")
        for m in metrics.values():
            previous.require(m["closure_relative"].max().item() <= 1e-8, "POWER_CLOSURE_FAILED")
            previous.require((m["power_w"] <= pin*(1+1e-6)).all().item(), "PASSIVE_FIELD_POWER_FAILED")
        projected, _ = projector.project(full)
        t0 = (metrics["M2_full"]["power_w"].sum()/pin.sum()).item()
        linear = interface.apply("M0", incoming, z_t, SyntheticResponse("linear", proxy_linear_transmission=t0))
        previous.require(projector.metrics(linear)["eta_perp"].max().item() <= 1e-10, "M0_PROJECTION_FAILED")
        equal_power_matched = matched*torch.sqrt(metrics["M2_full"]["power_w"]/metrics["M1_matched"]["power_w"])[:, None, None, None]
        fields = {"Bare": incoming, "M0_equal_loss": linear, "M1_legacy": legacy, "M1_matched": matched,
                  "M2_full": full, "M2_projected": projected.to(torch.complex64),
                  "M1_matched_equal_power_supplement": equal_power_matched}
        rows, raws, views = {}, {}, {}
        for name, field in fields.items():
            raw = previous.detector(model, field, 10)
            rows[name], view = task_metrics(raw, labels, gain, ed, ec)
            m = projector.metrics(field)
            rows[name].update({"input_power_w": previous.summary(pin), "output_power_w": previous.summary(m["power_w"]),
                               "eta_perp": previous.summary(m["eta_perp"])})
            raws[name], views[name] = raw, view
        projected_coeff = (full.to(torch.complex128).flatten(1)@modes.conj()*area)@projector.gram_pinv.T
        differences = {}
        for name, field in (("M1_legacy", legacy), ("M1_matched", matched), ("M2_projected", projected)):
            differences[name+"_vs_M2_full"] = {
                "field_relative": previous.summary(previous.relative(field, full)),
                "intensity_relative": previous.summary(previous.relative(field.abs().square(), full.abs().square())),
                "detector_relative": previous.summary(previous.relative(raws[name], raws["M2_full"])),
                "view_relative": previous.summary(previous.relative(views[name], views["M2_full"])),
                "ED_prediction_changes": sum(a != b for a,b in zip(rows[name]["ED"]["predictions"], rows["M2_full"]["ED"]["predictions"])),
                "EC_prediction_changes": sum(a != b for a,b in zip(rows[name]["EC"]["predictions"], rows["M2_full"]["EC"]["predictions"]))}
        old_case = next(c for c in old["cases"] if c["response"]["family"] == response.family
                        and c["response"]["proxy_strength"] == response.proxy_strength
                        and c["response"]["proxy_linear_transmission"] == response.proxy_linear_transmission)
        unchanged_m2 = np.allclose(metrics["M2_full"]["eta_perp"].cpu().numpy(), old_case["eta_perp"]["rows"], rtol=1e-10, atol=1e-14)
        previous.require(unchanged_m2, "M2_PREVIOUS_RESULT_CHANGED")
        if response.family == "saturation":
            previous.require(torch.equal(full.cpu(), old_tensors["S4_full"]), "M2_S4_FIELD_NOT_IDENTICAL")
        result["cases"].append({"response": spec, "formula_max_abs_error": error,
                                "a_min": factors.min().item(), "a_max": factors.max().item(),
                                "legacy_function_exact": True, "M2_previous_reproduced": True,
                                "M0_equal_loss_power_transmission": t0,
                                "matched_vs_projected_coeff_relative": previous.summary(previous.relative(matched_coeff, projected_coeff)),
                                "legacy_vs_projected_coeff_relative": previous.summary(previous.relative(legacy_coeff, projected_coeff)),
                                "matched_vs_projected_field_relative": previous.summary(previous.relative(matched, projected)),
                                "equal_power_matched_vs_full_detector_relative": previous.summary(previous.relative(
                                    raws["M1_matched_equal_power_supplement"], raws["M2_full"])),
                                "differences": differences, "variants": rows})
        tensors[str(index)] = {"matched_coefficients": matched_coeff.cpu(), "a": factors.cpu(),
                               "full_field": full.cpu(), "matched_field": matched.cpu(),
                               "projected_field": projected.cpu(), "raws": {k:v.cpu() for k,v in raws.items()},
                               "views": {k:v.cpu() for k,v in views.items()}}
        previous.save_json(output/"matched_m1_control_results.json", result)
    pilot._assert_state_unchanged(model, frozen, "phase1_C")
    previous.require(torch.equal(bare_raw, torch.cat([model.forward_from_ols(chunk)["physical_w_per_pixel"] for chunk in z.split(10)])),
                     "HISTORICAL_B_MUTATED")
    result.update({"status": "PASSED", "baseline_unchanged": True, "M1_matched_validation_passed": True})
    previous.save_json(output/"matched_m1_control_results.json", result)
    torch.save(tensors, output/"matched_m1_tensors.pt")
    return result


def replay_saved_noise(population, prior, scale, chosen) -> tuple[torch.Tensor, torch.Tensor, dict]:
    part = population["validation"]
    errors, noise_rows, reconstructed = [], [], []
    for k in range(10):
        selected = [i for i in chosen if int(part["labels"][i]) == k]
        ids = part["proposal_ids"][selected]
        generator = torch.Generator(device="cpu").manual_seed(part["seeds"][k])
        blocks = [torch.randn((256, 200), generator=generator, dtype=torch.float64)
                  for _ in range(int(ids.max())//256+1)]
        eps = torch.cat(blocks)[ids]
        # Reproduce the original 256-proposal GEMM shape before selecting rows.
        all_noise = torch.cat(blocks)
        real = prior[k]["mean"].double()+all_noise@prior[k]["cholesky"].double().T
        coeff = (torch.view_as_complex(real.reshape(-1,100,2).contiguous())*scale.sqrt()[None]).to(torch.complex64)
        restored = coeff[ids]
        saved = part["coefficients"][selected]
        errors.append((restored-saved).abs().max().item())
        previous.require(torch.equal(restored, saved), "GAMMA_NOISE_AND_POPULATION_IDENTITY_FAILED")
        noise_rows.append(eps)
        reconstructed.append(restored)
    return torch.cat(noise_rows), torch.cat(reconstructed), {"max_coefficient_error": max(errors), "passed": True}


def run_a(config, old, output, cp, ed, ec, optical_microbatch):
    device = torch.device("cuda")
    mapper_cp = torch.load(config["mapper_checkpoint"], map_location="cpu", weights_only=False)
    population = torch.load(config["mapper_population"], map_location="cpu", weights_only=False)
    previous.require(mapper_cp["schema"] == "lightgen_g1_slm_input_mapping_checkpoint.v1"
                     and mapper_cp["checkpoint_step"] == 5000 and mapper_cp["completed_updates"] == 5000,
                     "FROZEN_COMPLETED_GAMMA_IDENTITY_FAILED")
    previous.require(population["schema"] == "lightgen_g1_slm_mapping_population.v1", "GAMMA_POPULATION_IDENTITY_FAILED")
    part = population["validation"]
    chosen = [int(i) for k in range(10) for i in torch.where(part["labels"] == k)[0][:32]]
    previous.require(len(chosen) == 320 and part["seeds"] == list(range(2026102630,2026102640)),
                     "GAMMA_SAMPLE_PROTOCOL_CHANGED")
    prior_path = yaml.safe_load(Path(config["mapper_config"]).read_text())["source"]["fitted_gaussian_prior"]
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    epsilon, control_vectors, replay = replay_saved_noise(population, prior, cp["scale"].double(), chosen)
    labels = part["labels"][chosen].to(device)
    mapper = pilot.AmplitudeMapper().to(device).eval().requires_grad_(False)
    mapper.load_state_dict(mapper_cp["mapper_state"], strict=True)
    mapper_state = pilot._snapshot_state(mapper)
    pilot._set_numerics()
    model = pilot._new_optical_model(cp, 2048, device)
    optical_state = pilot._snapshot_state(model)
    runtime = model.runtime
    trace = {"SLM": 0, "Encoder": 0, "OLS_projection": 0, "external_ols_injection": 0}
    original_slm = runtime.input_adapter.slm
    original_projection = runtime.ols.project_state

    def slm_call(command):
        trace["SLM"] += len(command.tensor)
        return original_slm(command)

    def project_call(field):
        trace["OLS_projection"] += len(field.tensor)
        return original_projection(field)

    def encoder_hook(_module, inputs, _output):
        trace["Encoder"] += len(inputs[0].tensor)

    runtime.input_adapter.slm = slm_call
    runtime.ols.project_state = project_call
    hook = runtime.encoder.register_forward_hook(encoder_hook)
    # Independent front-end qualification, no update and no direct target injection.
    probe_amp = mapper(pilot._feature_batch(control_vectors[:10], cp["scale"], device)).double()
    probe = runtime.forward_from_amplitude(probe_amp[:1])
    batch_probe = runtime.forward_from_amplitude(probe_amp[:2])
    batch_coefficient_error = previous.relative(probe.z_ols.coefficients, batch_probe.z_ols.coefficients[:1]).max().item()
    batch_detector_error = previous.relative(probe.physical_sensor_output_w_per_pixel,
                                             batch_probe.physical_sensor_output_w_per_pixel[:1]).max().item()
    previous.require(max(batch_coefficient_error, batch_detector_error) <= 1e-6,
                     "A_MICROBATCH_PRODUCTION_CONSISTENCY_FAILED")
    previous.require(probe.z_ols.source == "real_encoded" and probe.image_input is not None,
                     "GAMMA_PREFLIGHT_BYPASSED_FRONTEND")
    validate_a_trace(trace, 3)
    rejected_bypass = False
    try:
        validate_a_trace({**trace, "SLM": 0}, 3)
    except previous.DiagnosticStop:
        rejected_bypass = True
    previous.require(rejected_bypass, "A_TRACE_CONSUMER_ACCEPTS_BYPASS")
    for key in trace:
        trace[key] = 0
    interface = SpatialMaterialInterface(old["reference_intensity_w_m2"], old["baseline"]["legacy_mfd_m"],
                                         runtime.ols.operator.reemit_coefficients)
    response = SyntheticResponse(**config["material_for_A"])
    zs, commands, bare_raw, material_raw, records = [], [], [], [], []
    field_samples = []
    print(json.dumps({"stage": "A_REAL_FRONTEND", "samples": 320, "mapper_checkpoint_step": 5000}), flush=True)
    for start in range(0, 320, 10):
        w = control_vectors[start:start+10]
        logical_commands = mapper(pilot._feature_batch(w, cp["scale"], device))
        for q in logical_commands.split(optical_microbatch):
            result = runtime.forward_from_amplitude(q.double())
            previous.require(result.z_ols.source == "real_encoded" and result.ols_input_field is not None
                             and result.encoder_output_field is not None and result.image_input is not None,
                             "A_MISSING_ACTUAL_FRONTEND_FIELDS")
            previous.require(torch.equal(result.image_input.slm_amplitude, q.double())
                             and torch.equal(result.image_input.slm_intensity, q.double().square()),
                             "A_AMPLITUDE_INTENSITY_ENCODING_FAILED")
            z = result.z_ols.coefficients
            previous.require(z.shape == (len(q), 100) and z.dtype == torch.complex128
                             and torch.isfinite(z).all().item(), "A_OLS_FORMAT_FAILED")
            entry = result.generator_input_field.tensor
            material = interface.apply("M2", entry, z, response)
            pin = entry.abs().square().flatten(1).sum(1)*4e-12
            pout = material.abs().square().flatten(1).sum(1)*4e-12
            previous.require((pout <= pin*(1+1e-6)).all().item(), "A_MATERIAL_PASSIVITY_FAILED")
            material_output = preserve_precision_detector(model, material, optical_microbatch)
            raw = result.physical_sensor_output_w_per_pixel
            previous.require(raw is not None and raw.shape == (len(q),1,200,200), "A_DETECTOR_INTERFACE_FAILED")
            commands.append(q.cpu()); zs.append(z.cpu()); bare_raw.append(raw.cpu()); material_raw.append(material_output.cpu())
            record = {k:v.detach().cpu() for k,v in result.physical_power_trace.items()}
            record["M2_input_power_w"] = pin.cpu(); record["M2_output_power_w"] = pout.cpu()
            records.append(record)
            if not field_samples:
                field_samples.append({"input_slm_field": result.image_input.slm_field.tensor.cpu(),
                                      "encoder_output": result.encoder_output_field.tensor.cpu(),
                                      "ols_input": result.ols_input_field.tensor.cpu(),
                                      "material_input": entry.cpu(), "material_output": material.cpu()})
        if (start+10)%80 == 0:
            print(json.dumps({"stage": "A_REAL_FRONTEND", "completed": start+10, "total": 320}), flush=True)
    validate_a_trace(trace, 320)
    hook.remove(); runtime.input_adapter.slm = original_slm; runtime.ols.project_state = original_projection
    z_a = torch.cat(zs)
    commands = torch.cat(commands)
    raw_a = torch.cat(bare_raw).to(device)
    raw_a_m2 = torch.cat(material_raw).to(device)
    gain = cp["camera_gain"].to(device=device, dtype=torch.float64)
    a_metrics, a_view = task_metrics(raw_a, labels, gain, ed, ec)
    am_metrics, am_view = task_metrics(raw_a_m2, labels, gain, ed, ec)
    # Clearly isolated B reference with the same G1 controls, never consumed by A's encoder input.
    raw_b = torch.cat([model.forward_from_ols(chunk.to(device=device,dtype=torch.complex128))["physical_w_per_pixel"]
                       for chunk in control_vectors.split(optical_microbatch)])
    b_entry = previous.frozen_reemit(runtime.ols.operator, control_vectors.to(device=device,dtype=torch.complex128), 10).unsqueeze(1)
    b_material = interface.apply("M2", b_entry, control_vectors.to(device), response)
    raw_b_m2 = preserve_precision_detector(model, b_material, optical_microbatch)
    b_metrics, b_view = task_metrics(raw_b, labels, gain, ed, ec)
    bm_metrics, bm_view = task_metrics(raw_b_m2, labels, gain, ed, ec)
    power_records = {k:torch.cat([row[k] for row in records]) for k in records[0]}
    pilot._assert_state_unchanged(mapper, mapper_state, "frozen_Gamma_v1")
    pilot._assert_state_unchanged(model, optical_state, "frozen_A_optics")
    result = {"schema": "a_route_reachability_diagnostics.v1", "status": "COMPLETED",
              "gate": conditional_gate(True, True, False, False), "mapping_case": "A_EXISTING_FROZEN_MAPPING",
              "gamma_status": "EXISTING_COMPOSED_G1_SAMPLER_AND_FROZEN_MAPPER_V1",
              "source_mapper_checkpoint": config["mapper_checkpoint"], "source_mapper_completed_updates": 5000,
              "source_mapper_historical_status": "VALID_FAILED", "current_training_v2_consumed": False,
              "selection": "first 32 saved validation entries per class", "sample_positions": chosen,
              "labels": labels.cpu().tolist(), "sample_rows": [part["rows"][i] for i in chosen],
              "noise_replay": replay, "actual_frontend_trace": trace, "trace_bypass_negative_rejected": rejected_bypass,
              "optical_microbatch": optical_microbatch, "mapper_logical_batch": 10,
              "microbatch_preflight": {"coefficient_relative": batch_coefficient_error,
                                       "detector_relative": batch_detector_error, "passed": True},
              "input_contract": {"amplitude_shape": list(commands.shape), "amplitude_dtype": str(commands.dtype),
                                 "amplitude_min": commands.min().item(), "amplitude_max": commands.max().item(),
                                 "slm_plane": str(runtime.slm_plane), "ols_plane": str(runtime.ols_plane),
                                 "generator_plane": str(runtime.generator_entrance_plane),
                                 "camera_plane": str(runtime.camera_plane)},
              "power_trace_w": {k:previous.summary(v) if v.ndim==1 else {"shape":list(v.shape),"rows":v.tolist()}
                                for k,v in power_records.items()},
              "ols_A": coeff_statistics(z_a, labels), "ols_B_paired_G1": coeff_statistics(control_vectors, labels),
              "A_bare": a_metrics, "A_M2": am_metrics, "B_paired_G1_bare": b_metrics, "B_paired_G1_M2": bm_metrics,
              "paired_comparison": {"coefficient_relative_no_alignment": previous.summary(previous.relative(z_a,control_vectors.to(torch.complex128))),
                                    "detector_A_vs_B_relative": previous.summary(previous.relative(raw_a,raw_b)),
                                    "view_A_vs_B_relative": previous.summary(previous.relative(a_view,b_view)),
                                    "view_A_M2_vs_B_M2_relative": previous.summary(previous.relative(am_view,bm_view)),
                                    "A_material_vs_bare_view_relative": previous.summary(previous.relative(am_view,a_view))},
              "optimizer_updates": 0, "prior_refits": 0, "test_reads": 0, "new_population_samples": 0,
              "acceptance": {"conditional_control_threshold_defined": False, "new_threshold_invented": False,
                             "old_target_replication_threshold_used_as_gate": False},
              "frozen_model_state_unchanged": True,
              "separate_previous_G4_population": {"count":40,"role":"different G4 prior; not population-matched to this G1 A cohort"}}
    previous.save_json(output/"a_route_reachability_diagnostics.json", result)
    torch.save({"commands": commands, "epsilon": epsilon, "control_vectors": control_vectors, "z_A": z_a,
                "labels": labels.cpu(), "raw_A": raw_a.cpu(), "raw_A_M2": raw_a_m2.cpu(),
                "raw_B_paired": raw_b.cpu(), "raw_B_paired_M2": raw_b_m2.cpu(),
                "view_A": a_view.cpu(), "view_A_M2": am_view.cpu(), "view_B_paired": b_view.cpu(),
                "view_B_paired_M2": bm_view.cpu(), "power_trace":power_records, "field_samples":field_samples},
               output/"a_route_tensors.pt")
    return result


def main():
    # Set before phase 1 creates CuBLAS handles, not only in the later pilot setup.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-a", action="store_true", help="Continue after a saved, passing matched-M1 phase only")
    parser.add_argument("--a-optical-microbatch", type=int, default=1, choices=(1,2,5,10))
    args = parser.parse_args()
    if args.output.exists() and not args.resume_a:
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    old = json.loads(Path(config["previous_result"]).read_text())
    previous.require(old["status"] == "COMPLETED" and old["baseline_unchanged"], "PREVIOUS_M2_PACKAGE_NOT_VALID")
    previous.require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    cp = torch.load(old["config"]["checkpoint"], map_location="cpu", weights_only=False)
    previous.require(cp["completed_global_step"] == 45000 and cp["branch"] == "C", "C_CHECKPOINT_IDENTITY_FAILED")
    cohort = torch.load(old["config"]["cohort"], map_location="cpu", weights_only=False)
    ed = previous.base._load_evaluator(Path(old["config"]["ed_checkpoint"]), torch.device("cuda"))
    ec, _ = previous.decoder_ref._load_ec_evaluator(Path(old["config"]["ec_checkpoint"]), torch.device("cuda"))
    frozen_ed, frozen_ec = pilot._snapshot_state(ed), pilot._snapshot_state(ec)
    identity = ResearchRunIdentity(task_id=config["task_id"], run_id=uuid.uuid4().hex,
                                  trajectory_id=old["baseline"]["id"],
                                  semantic_authority_id="USER_AUTHORIZED_MATCHED_M1_A_REACHABILITY_20261027",
                                  data_role="SAVED_G4_DIAGNOSTIC_AND_G1_VALIDATION_NO_TEST",
                                  config_identity=config, model_identity=cp["model_identity"],
                                  optimizer_identity="NO_OPTIMIZER_NO_UPDATES",
                                  sampler_identity="saved G4 first4 and G1 validation first32 per class",
                                  execution_source_semantics_id="M1_MATCHED_AND_FROZEN_EXISTING_GAMMA_V1")
    started = time.time()
    with torch.inference_mode():
        if args.resume_a:
            matched = json.loads((args.output/"matched_m1_control_results.json").read_text())
            previous.require(matched["status"] == "PASSED" and matched["M1_matched_validation_passed"]
                             and (args.output/"matched_m1_tensors.pt").is_file()
                             and not (args.output/"a_route_reachability_diagnostics.json").exists(),
                             "RESUME_REQUIRES_CLOSED_PASSING_M1_AND_UNFINISHED_A")
            saved_identity = ResearchRunIdentity.from_mapping(matched["run_identity"])
            previous.require(saved_identity.config_identity == config and saved_identity.model_identity == cp["model_identity"],
                             "RESUME_FROZEN_INPUT_IDENTITY_CHANGED")
            control = ResearchRunControl(saved_identity, authorization_id=saved_identity.semantic_authority_id)
            receipt = PreflightReceipt.from_mapping(matched["ready_receipt"])
            control.mark_preflight_ready(receipt)
            control.resume_scientific_attempt(args.output, "attempt_01", receipt)
        else:
            control = ResearchRunControl(identity, authorization_id=identity.semantic_authority_id)
            matched = run_matched(config, old, args.output, cp, cohort, ed, ec, control)
        torch.cuda.empty_cache()
        route = run_a(config, old, args.output, cp, ed, ec, args.a_optical_microbatch)
        pilot._assert_state_unchanged(ed, frozen_ed, "ED")
        pilot._assert_state_unchanged(ec, frozen_ec, "EC")
    control.complete()
    previous.save_json(args.output/"package_execution.json", {"status":"COMPLETED", "gate":route["gate"],
                        "M1_matched_validation_passed":matched["M1_matched_validation_passed"],
                        "elapsed_seconds":time.time()-started,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),
                        "gpu":torch.cuda.get_device_name(0),"evaluators_unchanged":True,"optimizer_updates":0,
                        "resumed_A_only": args.resume_a, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    print(json.dumps({"status":"COMPLETED","gate":route["gate"]}), flush=True)


if __name__ == "__main__":
    main()
