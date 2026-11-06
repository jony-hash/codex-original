"""Frozen A-route accounting: direct overlaps versus field synthesis and LS only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.lightgen.power_budget_projection_diagnostics import (
    OLSProjectionPowerAudit, area_power, validate_production_handoff, validate_projection_accounting,
)
from ldm_project.v2.application.research.lightgen.spatial_material_interface import SpatialMaterialInterface, SyntheticResponse
from ldm_project.v2.application.research.lightgen.image_readout import resize_view
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import validate_lightgen_m2_spatial_material_interface_v1 as old_m2


ROOT = Path(__file__).resolve().parents[3]
SCALARS = ("P_encoder_w", "P_coefficient_norm_metric", "P_match_w", "P_match_gram_w", "P_LS_w", "P_LS_gram_w",
           "P_residual_w", "ratio_coefficient_norm", "ratio_match", "ratio_LS", "ratio_residual",
           "production_overlap_relative_error", "match_vs_LS_coefficient_relative", "match_vs_LS_field_relative",
           "match_gram_closure_relative", "LS_gram_closure_relative", "LS_pythagorean_closure_relative", "LS_residual_orthogonality")


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"std_population":float(x.std()),
            "min":float(x.min()),"max":float(x.max()),
            "P05":float(np.quantile(x,.05)),"P25":float(np.quantile(x,.25)),
            "P75":float(np.quantile(x,.75)),"P95":float(np.quantile(x,.95))}


def projection_checked(audit, field, coefficients):
    result = audit.analyze(field, coefficients)
    try:
        validate_projection_accounting(result)
    except ValueError as exc:
        raise old_m2.DiagnosticStop(f"PROJECTION_POWER_CLOSURE_FAILED: {exc}") from exc
    old_m2.require(result["production_overlap_relative_error"].max().item() <= 1e-10,
                   "PRODUCTION_COEFFICIENTS_ARE_NOT_THE_AUDITED_OVERLAPS")
    return result


def relay_breakdown(relay, input_power, output_power):
    source = relay.last_diagnostic["source_pupil_retention"].reshape(-1).double()
    target = relay.last_diagnostic["target_pupil_retention"].reshape(-1).double()
    old_m2.require((target > 0).all().item(), "RELAY_TARGET_RETENTION_UNDEFINED")
    return {"source_pupil_after":input_power*source,"pre_target_pupil":output_power/target,
            "source_retention":source,"target_retention":target}


def one_forward(model, q, interface, response, audit):
    runtime = model.runtime
    captured = {}
    def capture_encoder_input(_module, args):
        captured["encoder_entrance"] = args[0]
    handle = runtime.encoder.register_forward_pre_hook(capture_encoder_input)
    try:
        result = runtime.forward_from_amplitude(q.to(device=runtime.encoder.phase.device,dtype=torch.float64))
    finally:
        handle.remove()
    old_m2.require(result.z_ols.source == "real_encoded" and result.ols_input_field is not None,
                   "REPLAY_BYPASSED_ENCODER_OR_OLS")
    entry = result.generator_input_field.tensor
    source = result.ols_input_field.tensor
    coeff = result.z_ols.coefficients
    geometry = ("shape","pixel_pitch_m","wavelength_m","z_m","origin_m")
    old_m2.require(all(getattr(result.ols_input_field.plane,key)==getattr(runtime.generator_entrance_plane,key)
                       for key in geometry), "OLS_REEMISSION_REFERENCE_PLANE_MISMATCH")
    projection = projection_checked(audit, source, coeff)
    native_match_error = old_m2.relative(projection["field_match"],entry).max().item()
    old_m2.require(native_match_error <= 1e-10, "NATIVE_MATCH_SYNTHESIS_DIFFERS_FROM_DIAGNOSTIC")
    validate_production_handoff(entry,result.generator_input_field.tensor)
    power = {"source":result.physical_power_trace["P_input_W"],
             "slm_after":area_power(result.image_input.slm_field.tensor,6.25e-12),
             "encoder_entrance":area_power(captured["encoder_entrance"].tensor,4e-12),
             "OLS_input":area_power(source,4e-12),"match_synthesis":area_power(entry,4e-12),
             "M2_input":area_power(entry,4e-12)}
    relay = relay_breakdown(runtime.input_relay,power["slm_after"],power["encoder_entrance"])
    power["input_relay_source_pupil_after"] = relay["source_pupil_after"]
    power["input_relay_pre_target_pupil"] = relay["pre_target_pupil"]
    encoder_layers=result.crop_statistics["encoder_layer_power"]
    for layer in range(2):
        for pos,name in enumerate(("input","device","propagated","aperture")):
            power[f"encoder_{layer+1}_{name}"]=encoder_layers[:,layer,pos]
        incoming=captured["encoder_entrance"].tensor if layer==0 else result.encoder_records[layer-1].tensor
        transfer=torch.polar(torch.full_like(runtime.encoder.phase[layer],runtime.encoder.layer_power_transmission**.5),
                             runtime.encoder.phase[layer]).to(incoming)
        measured=area_power(incoming*transfer,4e-12)
        old_m2.require(old_m2.relative(measured[:,None],power[f"encoder_{layer+1}_device"][:,None]).max().item()<=1e-10,
                       "ENCODER_DEVICE_POWER_MISMATCH")
    bare_raw=result.physical_sensor_output_w_per_pixel
    bare_gen=result.crop_statistics["generator_layer_power"]
    power["bare_generator_exit"]=area_power(result.generator_output_field.tensor,4e-12)
    power["bare_detector"]=bare_raw.flatten(1).sum(1)
    bare_relay=relay_breakdown(runtime.output_relay,power["bare_generator_exit"],power["bare_detector"])
    power["bare_output_relay_source_pupil_after"]=bare_relay["source_pupil_after"]
    power["bare_output_relay_pre_target_pupil"]=bare_relay["pre_target_pupil"]
    material=interface.apply("M2",entry,coeff,response)
    power["M2_output"]=area_power(material,4e-12)
    incoming=ComplexField(material,runtime.generator_entrance_plane,
                           OperationProvenance("audited_M2_Q3","Q2",None,{"P_post":"identity_simulation_assumption"}))
    output,records,gen_stats=runtime.generator(incoming,"matched_generated_generator")
    output=ComplexField(output.tensor,runtime.generator_exit_plane,
                        OperationProvenance("lightgen_generator_exit",output.plane.plane_id,None,{"generator_layers":2}))
    camera=runtime.output_relay(output)
    area=math.prod(runtime.camera_plane.pixel_pitch_m)
    m2_raw=camera.tensor.abs().square()*area
    power["M2_generator_input"]=area_power(incoming.tensor,4e-12)
    power["M2_generator_exit"]=area_power(output.tensor,4e-12)
    power["M2_detector"]=area_power(camera.tensor,area)
    for tag,layers in (("bare_generator",bare_gen),("M2_generator",gen_stats["layer_power"])):
        for layer in range(2):
            for pos,name in enumerate(("input","device","propagated","aperture")):
                power[f"{tag}_{layer+1}_{name}"]=layers[:,layer,pos]
    out_relay=relay_breakdown(runtime.output_relay,power["M2_generator_exit"],power["M2_detector"])
    power["M2_output_relay_source_pupil_after"]=out_relay["source_pupil_after"]
    power["M2_output_relay_pre_target_pupil"]=out_relay["pre_target_pupil"]
    old_m2.require((power["M2_output"]<=power["M2_input"]*(1+1e-5)+1e-12).all().item(),"M2_PASSIVITY_FAILED")
    old_m2.require(torch.equal(material,incoming.tensor),"M2_GENERATOR_HANDOFF_CHANGED")
    trace_error=max(abs(power[k].item()-result.physical_power_trace[trace].item())/
                    max(abs(result.physical_power_trace[trace].item()),1e-300)
                    for k,trace in (("slm_after","P_after_slm_W"),("OLS_input","P_before_fiber_W"),
                                    ("match_synthesis","P_generator_entry_W"),("bare_detector","P_camera_W")))
    old_m2.require(trace_error<=1e-8,"POWER_TRACE_UNIT_MISMATCH")
    return {"power":{k:v.detach().cpu() for k,v in power.items()},
            "projection":{k:v.detach().cpu() for k,v in projection.items()},
            "encoder_field":source.detach().cpu(),"production_coefficients":coeff.detach().cpu(),
            "bare_raw":bare_raw.detach().cpu(),"M2_raw":m2_raw.detach().cpu(),
            "material_input":entry.detach().cpu(),"material_output":material.detach().cpu(),
            "trace_error":trace_error,"native_match_error":native_match_error,
            "input_relay_resampling_power_ratio":(relay["pre_target_pupil"]/relay["source_pupil_after"]).item()}


def make_audit(model,rtol):
    operator=model.runtime.ols.operator
    device=model.runtime.encoder.phase.device
    old_m2.require(torch.equal(operator.transport,torch.eye(100,device=device,dtype=torch.complex128)),"OLS_TRANSPORT_NOT_FROZEN_IDENTITY")
    v=operator.reemit_coefficients(torch.eye(100,device=device,dtype=torch.complex128)).flatten(1).T
    return OLSProjectionPowerAudit(v,math.prod(operator.input_plane.pixel_pitch_m),rtol)


def source_snapshots(output):
    names=["src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py",
           "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
           "src/ldm_project/v2/application/research/lightgen/spatial_material_interface.py",
           "src/ldm_project/v2/optics/d2nn.py","src/ldm_project/v2/optics/relays.py",
           "src/ldm_project/v2/optics/propagation.py","src/ldm_project/v2/application/research/lightgen/image_readout.py",
           "src/ldm_project/v2/application/research/lightgen/power_budget_projection_diagnostics.py",
           "scripts/research/lightgen/validate_lightgen_a_route_power_budget_v1.py"]
    target=output/"source_snapshot"
    target.mkdir()
    for name in names:
        shutil.copyfile(ROOT/name,target/Path(name).name)
    return {"files":names,"consumer":"read-only formula and reference-plane audit; never admission equality", "git_head_provenance_only":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}


def run(config,output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    pilot._set_numerics()
    old_m2.require(torch.cuda.is_available(),"CUDA_UNAVAILABLE")
    started=time.time()
    source=Path(config["source_run"])
    old=json.loads((source/"a_route_reachability_diagnostics.json").read_text())
    matched=json.loads((source/"matched_m1_control_results.json").read_text())
    data=torch.load(source/"a_route_tensors.pt",map_location="cpu",weights_only=False)
    old_m2.require(old["status"]=="COMPLETED" and matched["status"]=="PASSED" and len(data["labels"])==320,
                   "SOURCE_PACKAGE_NOT_THE_FROZEN_COMPLETED_POPULATION")
    cp=torch.load(config["optical_checkpoint"],map_location="cpu",weights_only=False)
    mapper_cp=torch.load(config["mapper_checkpoint"],map_location="cpu",weights_only=False)
    old_m2.require(cp["branch"]=="C" and cp["completed_global_step"]==45000 and cp["seed"]==2026101501,
                   "FROZEN_C_IDENTITY_FAILED")
    old_m2.require(mapper_cp["checkpoint_step"]==5000 and mapper_cp["completed_updates"]==5000,"MAPPER_IDENTITY_FAILED")
    old_m2.require(data["commands"].shape==(320,1,200,200) and data["commands"].dtype==torch.float32,
                   "SLM_COMMAND_FORMAT_CHANGED")
    labels=data["labels"]
    # Choose only by old proposal IDs, before calculating any new forward result.
    probes=[]
    for k in range(10):
        positions=torch.where(labels==k)[0].tolist()
        old_m2.require(len(positions)==32,"FROZEN_CLASS_POPULATION_CHANGED")
        probes.extend(sorted(positions,key=lambda j:old["sample_rows"][j]["proposal_id"])[:3])
    staging=output.parent/(output.name+"_preflight")
    old_m2.save_json(staging/"fixed_probe_selection.json",{"indices":probes,"rows":[old["sample_rows"][j] for j in probes]})
    device=torch.device("cuda")
    model=pilot._new_optical_model(cp,config["main_grid"],device)
    state=pilot._snapshot_state(model)
    audit=make_audit(model,config["gram_rtol"])
    basis=audit.basis_diagnostics()
    old_m2.require(basis["basis_status"]!="OLS_BASIS_ILL_CONDITIONED","BASIS_ILL_CONDITIONED_CANNOT_CERTIFY_LS")
    interface=SpatialMaterialInterface(matched["reference_intensity_w_m2"],model.runtime.physical_optics.fiber_mfd_m,model.runtime.ols.operator.reemit_coefficients)
    response=SyntheticResponse(**config["material"])
    first=one_forward(model,data["commands"][:1],interface,response,audit)
    gain=cp["camera_gain"].double().cpu()
    tolerance=config["tolerances"]

    def replay(row,index):
        errors={"coefficient_relative":old_m2.relative(row["production_coefficients"],data["z_A"][index:index+1]).item(),
                "bare_detector_relative":old_m2.relative(row["bare_raw"],data["raw_A"][index:index+1]).item(),
                "M2_detector_relative":old_m2.relative(row["M2_raw"],data["raw_A_M2"][index:index+1]).item()}
        view=resize_view(row["bare_raw"]*gain).float()
        view_m2=resize_view(row["M2_raw"]*gain).float()
        errors["bare_view_max_abs"]=(view-data["view_A"][index:index+1]).abs().max().item()
        errors["M2_view_max_abs"]=(view_m2-data["view_A_M2"][index:index+1]).abs().max().item()
        old_m2.require(max(errors[k] for k in ("coefficient_relative","bare_detector_relative","M2_detector_relative"))<=tolerance["replay_relative"],"BASELINE_REPLAY_FAILED")
        old_m2.require(torch.allclose(view,data["view_A"][index:index+1],rtol=2e-6,atol=5e-8)
                       and torch.allclose(view_m2,data["view_A_M2"][index:index+1],rtol=2e-6,atol=5e-8),"SOFTWARE_READOUT_REPLAY_FAILED")
        return errors

    first_replay=replay(first,0)
    old_m2.require(old_m2.relative(first["encoder_field"],data["field_samples"][0]["ols_input"]).item()<=2e-6,
                   "SAVED_ENCODER_FIELD_REPLAY_FAILED")
    v=audit.v.cpu().numpy()
    gram_numpy=v.conj().T@v*4e-12
    gram_error=np.linalg.norm(gram_numpy-audit.gram.cpu().numpy())/np.linalg.norm(gram_numpy)
    q,_=np.linalg.qr(v*np.sqrt(4e-12))
    e=first["encoder_field"].numpy().ravel().astype(np.complex128)
    qr=q@(q.conj().T@e)
    qr_error=np.linalg.norm(qr-first["projection"]["field_LS"].numpy().ravel())/np.linalg.norm(qr)
    old_m2.require(max(gram_error,qr_error)<=1e-10,"INDEPENDENT_GRAM_OR_QR_FAILED")
    rejected_ls=False
    try:
        validate_production_handoff(first["projection"]["field_LS"],first["material_input"])
    except ValueError:
        rejected_ls=True
    old_m2.require(rejected_ls,"LS_HANDOFF_NEGATIVE_CONTROL_NOT_REJECTED")
    identity=ResearchRunIdentity(task_id=config["task_id"],run_id=uuid.uuid4().hex,
              trajectory_id="frozen_G1_mapper5000_C45000_saved_A320",
              semantic_authority_id="USER_AUTHORIZED_A_POWER_BUDGET_20261027",
              data_role="SAVED_VALIDATION_320_NO_TEST_NO_NEW_SAMPLES",config_identity=config,
              model_identity=cp["model_identity"],optimizer_identity="NO_OPTIMIZER",
              sampler_identity={"source_run":config["source_run"],"rows":old["sample_rows"],"grid_probes":probes},
              execution_source_semantics_id="AREA_POWER_OVERLAP_GRAM_LS_DIAGNOSTIC_ONLY_V1")
    control=ResearchRunControl(identity,authorization_id=identity.semantic_authority_id)
    receipt=control.make_ready_receipt(correctness={"passed":True,"replay":first_replay,"numpy_gram_relative":gram_error,
                   "numpy_qr_relative":qr_error,"LS_handoff_rejected":rejected_ls},
                   resource={"passed":True,"representative_full_cuda_forward_completed":True,"gpu":torch.cuda.get_device_name(0)},
                   implementation={"passed":True,"selected_engine":"frozen_A_power_audit_v1"},selected_engine="frozen_A_power_audit_v1")
    control.mark_preflight_ready(receipt)
    control.open_scientific_attempt(output,"attempt_01",receipt)
    sources=source_snapshots(output)
    runtime=model.runtime
    metadata={name:getattr(runtime,name).to_dict() for name in ("slm_plane","encoder_entrance_plane","ols_plane","generator_entrance_plane","generator_exit_plane","camera_plane")}
    basis.update({"schema":"ols_basis_diagnostics.v1","plane":runtime.ols_plane.to_dict(),"numpy_gram_relative":gram_error,
                  "numpy_qr_relative":qr_error,"LS_generator_input_forbidden":True})
    old_m2.save_json(output/"ols_basis_diagnostics.json",basis)
    rows=[]; encoder_fields=[]; coefficients=[]; ls_coefficients=[]; probe_results={}
    for index in range(320):
        row=first if index==0 else one_forward(model,data["commands"][index:index+1],interface,response,audit)
        replay_errors=replay(row,index)
        powers={k:v.item() for k,v in row["power"].items()}
        projection={k:row["projection"][k].item() for k in SCALARS}
        readout={tag:{"scaled_sum_dimensionless":(row[key]*gain).sum().item(),
                       "clamp_fraction":((row[key]*gain)>1).double().mean().item(),
                       "view28_mean_dimensionless":resize_view(row[key]*gain).mean().item()}
                 for tag,key in (("bare","bare_raw"),("M2","M2_raw"))}
        rows.append({"sample_index":index,"class":int(labels[index]),"source_identity":old["sample_rows"][index],
                     "powers_w":powers,"projection":projection,"replay":replay_errors,"software_readout":readout,
                     "input_relay_resampling_power_ratio":row["input_relay_resampling_power_ratio"],
                     "native_synthesis_relative_error":row["native_match_error"],"source_trace_relative_error":row["trace_error"]})
        encoder_fields.append(row["encoder_field"]); coefficients.append(row["production_coefficients"])
        ls_coefficients.append(row["projection"]["coefficients_LS"])
        if index in probes:
            probe_results[index]=row
        if (index+1)%80==0:
            print(json.dumps({"stage":"MAIN_2048","completed":index+1,"total":320}),flush=True)
    pilot._assert_state_unchanged(model,state,"power_audit_C2048")
    reference_model=pilot._new_optical_model(cp,config["reference_grid"],device)
    reference_state=pilot._snapshot_state(reference_model)
    reference_audit=make_audit(reference_model,config["gram_rtol"])
    reference_interface=SpatialMaterialInterface(matched["reference_intensity_w_m2"],reference_model.runtime.physical_optics.fiber_mfd_m,reference_model.runtime.ols.operator.reemit_coefficients)
    old_m2.require(all(getattr(reference_model.runtime.ols_plane,key)==getattr(model.runtime.ols_plane,key)
                       for key in ("shape","pixel_pitch_m","wavelength_m","z_m","coordinate_frame","origin_m")),"CROSS_GRID_PHYSICAL_PLANE_CHANGED")
    grid_rows=[]; reference_fields=[]; reference_coefficients=[]
    for index in probes:
        result=one_forward(reference_model,data["commands"][index:index+1],reference_interface,response,reference_audit)
        main=probe_results[index]
        comparisons={key:old_m2.relative(result[key],main[key]).item() for key in ("encoder_field","production_coefficients","bare_raw","M2_raw")}
        power_comparisons={k:abs(result["projection"][k].item()-main["projection"][k].item())/abs(main["projection"][k].item())
                           for k in ("P_encoder_w","P_coefficient_norm_metric","P_match_w","P_LS_w","P_residual_w")}
        grid_rows.append({"sample_index":index,"source_identity":old["sample_rows"][index],"relative_differences":comparisons,
                          "projection_power_relative_differences":power_comparisons,
                          "reference_projection":{k:result["projection"][k].item() for k in SCALARS}})
        reference_fields.append(result["encoder_field"]); reference_coefficients.append(result["production_coefficients"])
    pilot._assert_state_unchanged(reference_model,reference_state,"power_audit_C4096")
    gram_cross_error=(reference_audit.gram-audit.gram).abs().max().item()
    robust=all(max(row["relative_differences"][key] for key in ("production_coefficients","bare_raw","M2_raw"))<=1e-4 for row in grid_rows)
    aggregated={key:stats([row["projection"][key] for row in rows]) for key in SCALARS}
    grouped={str(k):{key:stats([row["projection"][key] for row in rows if row["class"]==k]) for key in SCALARS} for k in range(10)}
    power_summary={key:stats([row["powers_w"][key] for row in rows]) for key in rows[0]["powers_w"]}
    mean_input=aggregated["P_encoder_w"]["mean"]
    ratio_of_means={key:aggregated[key]["mean"]/mean_input for key in ("P_coefficient_norm_metric","P_match_w","P_LS_w","P_residual_w")}
    transitions=[("source","slm_after","SLM amplitude selection"),
                 ("slm_after","input_relay_source_pupil_after","input relay source pupil"),
                 ("input_relay_source_pupil_after","input_relay_pre_target_pupil","physical resampling incl magnification"),
                 ("input_relay_pre_target_pupil","encoder_entrance","input relay target pupil")]
    for tag in ("encoder","bare_generator","M2_generator"):
        for layer in (1,2):
            transitions.extend([(f"{tag}_{layer}_input",f"{tag}_{layer}_device","configured layer transmission"),
                                (f"{tag}_{layer}_device",f"{tag}_{layer}_propagated","padded ASM numerical propagation"),
                                (f"{tag}_{layer}_propagated",f"{tag}_{layer}_aperture","finite hard aperture")])
    transitions += [("OLS_input","match_synthesis","mode reduction / not measured absorption"),
                    ("M2_input","M2_output","synthetic passive amplitude response"),
                    ("bare_generator_exit","bare_output_relay_source_pupil_after","bare output relay source pupil"),
                    ("bare_output_relay_source_pupil_after","bare_output_relay_pre_target_pupil","bare physical resampling"),
                    ("bare_output_relay_pre_target_pupil","bare_detector","bare output relay target pupil"),
                    ("M2_generator_exit","M2_output_relay_source_pupil_after","M2 output relay source pupil"),
                    ("M2_output_relay_source_pupil_after","M2_output_relay_pre_target_pupil","M2 physical resampling"),
                    ("M2_output_relay_pre_target_pupil","M2_detector","M2 output relay target pupil")]
    transition_summary=[]
    for before,after,mechanism in transitions:
        deltas=[row["powers_w"][before]-row["powers_w"][after] for row in rows]
        ratios=[row["powers_w"][after]/row["powers_w"][before] for row in rows]
        transition_summary.append({"input":before,"output":after,"mechanism":mechanism,"delta_power_w":stats(deltas),
                                   "per_sample_ratio":stats(ratios),"ratio_of_mean_powers":power_summary[after]["mean"]/power_summary[before]["mean"]})
        for row,delta,ratio in zip(rows,deltas,ratios):
            row.setdefault("stage_changes",[]).append({"input":before,"output":after,"mechanism":mechanism,"delta_w":delta,"ratio":ratio})
    summary={"schema":"power_budget_summary.v1","status":"COMPLETED",
             "gate":"POWER_BUDGET_VALID_OLS_CAPTURE_RATIO_QUANTIFIED" if robust else "POWER_BUDGET_INCONCLUSIVE",
             "basis_status":basis["basis_status"],"sample_count":320,"config":config,
             "source_identity":{"optical_step":cp["completed_global_step"],"optical_model":cp["model_identity"],
                                "mapper_step":mapper_cp["checkpoint_step"],"mapper_run_id":mapper_cp["run_id"],
                                "source_rows":old["sample_rows"],"source_package_gate":old["gate"]},
             "reference_planes":metadata,"powers_w":power_summary,"projection":aggregated,"by_class_projection":grouped,
             "by_class_powers_w":{str(k):{key:stats([row["powers_w"][key] for row in rows if row["class"]==k])
                                         for key in power_summary} for k in range(10)},
             "ratio_of_mean_powers":ratio_of_means,"stage_changes":transition_summary,
             "sampling":{"fixed_probe_indices":probes,"count":30,"rows":grid_rows,"passed_existing_cross_grid_tolerance":robust,
                         "gram_max_abs_difference":gram_cross_error,"all_population_4096":False},
             "replay_max_errors":{key:max(row["replay"][key] for row in rows) for key in rows[0]["replay"]},
             "normal_equation_max":max(row["projection"]["LS_residual_orthogonality"] for row in rows),
             "LS_never_injected":True,"frozen_model_state_unchanged":True,"optimizer_updates":0,"test_reads":0,
             "prior_refits":0,"new_population_samples":0,"ready_receipt":receipt.to_dict(),"run_identity":control.identity.to_dict(),
             "source_versions":sources,"elapsed_seconds":time.time()-started,"gpu":torch.cuda.get_device_name(0),
             "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"material_reference_intensity_w_m2":matched["reference_intensity_w_m2"]}
    old_m2.save_json(output/"power_budget_per_sample.json",{"schema":"power_budget_per_sample.v1","rows":rows})
    old_m2.save_json(output/"power_budget_summary.json",summary)
    torch.save({"encoder_fields":torch.cat(encoder_fields),"production_coefficients":torch.cat(coefficients),
                "LS_coefficients_diagnostic_only":torch.cat(ls_coefficients),"modes":audit.v.cpu(),"labels":labels,
                "reference_indices":probes,"reference_encoder_fields":torch.cat(reference_fields),
                "reference_production_coefficients":torch.cat(reference_coefficients)},output/"projection_evidence.pt")
    control.complete()
    print(json.dumps({"status":"COMPLETED","gate":summary["gate"],"basis_status":basis["basis_status"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config=yaml.safe_load(args.config.read_text())
    with torch.inference_mode():
        try:
            run(config,args.output)
        except old_m2.DiagnosticStop as exc:
            gate="POWER_BUDGET_INCONCLUSIVE" if "ILL_CONDITIONED" in str(exc) else "POWER_BUDGET_NUMERICALLY_INVALID"
            target=args.output if args.output.exists() else args.output.parent/(args.output.name+"_preflight")
            old_m2.save_json(target/"power_budget_summary.json",{"status":"STOPPED","gate":gate,"stop_reason":str(exc)})
            print(json.dumps({"status":"STOPPED","gate":gate,"stop_reason":str(exc)}),flush=True)


if __name__=="__main__":
    main()
