"""Formal student output/provenance contract.

This module owns optical-field artifact assembly for the fixed physical path.  It
contains no diagnostic intervention and never imports the experimental package.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..config import PhysicsConfig
from ..optics.fields import build_optical_field_record
from ..optics.sampling import PlaneSampling

def build_formal_optical_fields(
    *,
    physics: PhysicsConfig,
    d2nn: Mapping[str, Any],
    slm_transfer: torch.Tensor,
    slm_field: torch.Tensor,
    front_field: torch.Tensor,
    front_path_diagnostics: Mapping[str, object],
    encoder_input_field: torch.Tensor,
    encoder_grid: torch.Tensor,
    encoder_layer_fields: torch.Tensor,
    encoded_field: torch.Tensor,
    ols_field: torch.Tensor,
    decoder_input_field: torch.Tensor,
    decoder_layer_fields: torch.Tensor,
    chip_grid: torch.Tensor,
    raw_optical_readout: dict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    """Capture model-internal complex fields with provenance, never display proxies."""
    front_phase_available = physics.front_path_model != "relay_intensity_psf"
    decoder_phase_available = front_phase_available
    slm_sampling = PlaneSampling(
        "slm",
        height=int(slm_transfer.shape[-2]),
        width=int(slm_transfer.shape[-1]),
        pixel_size_y=physics.optics.pixel_size,
        pixel_size_x=physics.optics.pixel_size,
    ).to_dict()
    front_sampling = PlaneSampling(
        "pre_encoder_relay_image"
        if physics.front_path_model == "coherent_imaging_4f"
        else "lightgen_propagated_field",
        height=int(front_field.shape[-2]),
        width=int(front_field.shape[-1]),
        pixel_size_y=d2nn["pixel_size"]
        if physics.front_path_model == "coherent_imaging_4f"
        else physics.optics.pixel_size,
        pixel_size_x=d2nn["pixel_size"]
        if physics.front_path_model == "coherent_imaging_4f"
        else physics.optics.pixel_size,
    ).to_dict()
    d2nn_sampling = PlaneSampling(
        "lightgen_d2nn",
        height=int(encoder_input_field.shape[-2]),
        width=int(encoder_input_field.shape[-1]),
        pixel_size_y=d2nn["pixel_size"],
        pixel_size_x=d2nn["pixel_size"],
    ).to_dict()

    def add(
        fields: dict[str, dict[str, object]],
        name: str,
        field: torch.Tensor,
        *,
        upstream: str | None,
        operation: str,
        phase_available: bool,
        source_kind: str = "propagated_complex_field",
        sampling: Mapping[str, object] | None = None,
    ) -> None:
        fields[name] = build_optical_field_record(
            field,
            upstream=upstream,
            operation=operation,
            phase_available=phase_available,
            source_kind=source_kind,
            sampling=sampling,
        )

    fields: dict[str, dict[str, object]] = {}
    front_source_kind = (
        "propagated_complex_field"
        if front_phase_available
        else "intensity_relay_embedding"
    )
    add(
        fields,
        "slm_output",
        slm_field,
        upstream=None,
        operation="slm_output_field",
        phase_available=True,
        sampling=slm_sampling,
    )
    front_operation = str(
        front_path_diagnostics.get("operation", physics.front_path_model)
    )
    front_name = (
        "pre_encoder_relay_image"
        if physics.front_path_model == "coherent_imaging_4f"
        else "lightgen_input"
    )
    add(
        fields,
        front_name,
        front_field,
        upstream="slm_output",
        operation=front_operation,
        phase_available=front_phase_available,
        source_kind=front_source_kind,
        sampling=front_sampling,
    )
    if physics.front_path_model == "coherent_imaging_4f":
        fields[front_name]["relay_trace"] = {
            "front_relay": front_path_diagnostics.get("front_relay"),
            "segments_m": front_path_diagnostics.get("segments_m"),
            "components": front_path_diagnostics.get("relay_trace"),
        }
    add(
        fields,
        "encoder_input",
        encoder_input_field,
        upstream=front_name,
        operation="coherent_imaging_4f"
        if physics.front_path_model == "coherent_imaging_4f"
        else (
            "physical_aperture_sampling"
            if physics.physical_forward_contract
            else physics.front_path_model
        ),
        phase_available=front_phase_available,
        source_kind=front_source_kind,
        sampling=d2nn_sampling,
    )
    add(
        fields,
        "encoder_grid",
        encoder_grid,
        upstream="encoder_input",
        operation=(
            "coherent_relay_alias"
            if physics.front_path_model == "coherent_imaging_4f"
            else "physical_aperture_sampling"
        ),
        phase_available=front_phase_available,
        source_kind=front_source_kind,
        sampling=d2nn_sampling,
    )
    encoder_source_kind = (
        "propagated_complex_field"
        if front_phase_available
        else "downstream_of_intensity_relay"
    )
    previous = "encoder_grid"
    for layer_index in range(int(encoder_layer_fields.shape[1])):
        name = f"encoder_layer_{layer_index + 1}"
        add(
            fields,
            name,
            encoder_layer_fields[:, layer_index],
            upstream=previous,
            operation="encoder_diffractive_layer",
            phase_available=front_phase_available,
            source_kind=encoder_source_kind,
            sampling=d2nn_sampling,
        )
        previous = name
    add(
        fields,
        "encoder_output",
        encoded_field,
        upstream=previous,
        operation="encoder_output",
        phase_available=front_phase_available,
        source_kind=encoder_source_kind,
        sampling=d2nn_sampling,
    )
    add(
        fields,
        "ols_input",
        encoded_field,
        upstream="encoder_output",
        operation="ols_input_alias",
        phase_available=front_phase_available,
        source_kind=encoder_source_kind,
        sampling=d2nn_sampling,
    )
    add(
        fields,
        "ols_output",
        ols_field,
        upstream="ols_input",
        operation="complex_ols:identity",
        phase_available=front_phase_available,
        source_kind=encoder_source_kind,
        sampling=d2nn_sampling,
    )
    decoder_source_kind = (
        "propagated_complex_field"
        if front_phase_available
        else "downstream_of_intensity_relay"
    )
    add(
        fields,
        "decoder_input",
        decoder_input_field,
        upstream="ols_output",
        operation="decoder_input",
        phase_available=decoder_phase_available,
        source_kind=decoder_source_kind,
        sampling=d2nn_sampling,
    )
    previous = "decoder_input"
    for layer_index in range(int(decoder_layer_fields.shape[1])):
        name = f"decoder_layer_{layer_index + 1}"
        add(
            fields,
            name,
            decoder_layer_fields[:, layer_index],
            upstream=previous,
            operation="decoder_diffractive_layer",
            phase_available=decoder_phase_available,
            source_kind=decoder_source_kind,
            sampling=d2nn_sampling,
        )
        previous = name
    add(
        fields,
        "decoder_output",
        chip_grid,
        upstream=previous,
        operation="decoder_output",
        phase_available=decoder_phase_available,
        source_kind=decoder_source_kind,
        sampling=d2nn_sampling,
    )
    add(
        fields,
        "chip",
        chip_grid,
        upstream="decoder_output",
        operation="chip_plane",
        phase_available=decoder_phase_available,
        source_kind=decoder_source_kind,
        sampling=d2nn_sampling,
    )
    if bool(raw_optical_readout.get("camera_field_available", False)):
        real = raw_optical_readout.get("camera_field_real")
        imag = raw_optical_readout.get("camera_field_imag")
        if torch.is_tensor(real) and torch.is_tensor(imag):
            add(
                fields,
                "camera_field",
                torch.complex(real, imag),
                upstream="chip",
                operation="coherent_camera_propagation",
                phase_available=decoder_phase_available,
                source_kind=decoder_source_kind,
                sampling=raw_optical_readout.get("camera_sampling"),
            )
    return fields


__all__ = ["build_formal_optical_fields"]
