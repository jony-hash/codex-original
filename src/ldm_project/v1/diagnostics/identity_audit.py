from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from ..student.model import load_one_shot_checkpoint_subset
from ..student.layers import _resize_complex_field, _resize_real_intensity
from ..optics import (
    angular_spectrum_propagate,
    apply_thin_lens,
    fresnel_propagate,
    normalize_real_intensity,
    _gaussian_blur_complex_field,
    _gaussian_blur_intensity,
)
from ..visualization_core import labels_for_visual_rows


IDENTITY_AUDIT_VARIANTS = (
    # Original chain variants (keep for backward compat).
    "generator_direct",
    "front_path_only_intensity",
    # v9.3 front-path decomposition (inserted after front_path_only_intensity
    # so they are grouped with the front-path results visually).
    "slm_transfer_intensity",
    "front_after_pre_slm_plane",
    "front_after_slm_only",
    "front_after_slm_to_bs",
    "front_after_bs_to_m3",
    "front_after_m3_to_lg",
    "front_path_current_fresnel",
    "front_path_current_angular_spectrum",
    "front_path_zero_distance_bypass",
    # v9.5 relay-identity front path.
    "front_path_relay_identity_intensity",
    # v9.4 padded front-path variants.
    "front_after_slm_to_bs_padded2x",
    "front_after_slm_to_bs_padded4x",
    "front_after_bs_to_m3_padded2x",
    "front_after_bs_to_m3_padded4x",
    "front_after_m3_to_lg_padded2x",
    "front_after_m3_to_lg_padded4x",
    "front_path_current_angular_spectrum_padded2x",
    "front_path_current_angular_spectrum_padded4x",
    "front_path_current_fresnel_padded2x",
    "front_path_current_fresnel_padded4x",
    # Original chain variants continued.
    "back_path_only_clean_chip",
    "back_path_only_clean_chip_probe",
    # v9.6b back-path / camera readout audit variants.
    "back_current_clean_chip_probe",
    "back_current_clean_chip_camera",
    "back_current_zero_mask_chip_probe",
    "back_current_zero_mask_chip_camera",
    "back_identity_clean_chip_camera",
    "back_identity_zero_mask_chip_camera",
    "back_padded4x_clean_chip_camera",
    "back_padded4x_zero_mask_chip_camera",
    "back_lens_only_clean_chip_camera",
    "back_camera_normalization_only_clean_chip",
    # v9.7+ coherent front PSF relay variants.
    "front_relay_coherent_psf_sigma0",
    "front_relay_coherent_psf_sigma1",
    "front_relay_coherent_psf_sigma2",
    # v9.7+ back PSF relay (intensity-domain) variants.
    "back_relay_psf_sigma0",
    "back_relay_psf_sigma1",
    "back_relay_psf_sigma2",
    # Pre-existing D2NN variants.
    "front_plus_back_no_d2nn",
    "front_plus_chip_no_d2nn",
    "zero_mask_d2nn_1x1",
    "zero_mask_d2nn_1x1_chip_intensity",
    "zero_mask_d2nn_1x1_probe",
    "small_phase_amp_d2nn_1x1",
    "small_phase_amp_d2nn_1x1_chip_intensity",
    "small_phase_amp_d2nn_1x1_probe",
)
IDENTITY_AUDIT_METRIC_KEYS = (
    "mse_to_slm",
    "corr_to_slm",
    "foreground_dice_to_prototype",
    "background_speckle_ratio",
    "readable_contrast",
    "pairwise_mse",
    "stripe_energy_ratio",
)


def _normalize_plane(tensor: torch.Tensor, percentile: float) -> torch.Tensor:
    return normalize_real_intensity(
        tensor.detach().float().clamp_min(0.0), percentile, clamp=False
    ).clamp(0.0, 1.0)


def _match_image_size(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor.detach().float()
    return _resize_real_intensity(tensor.detach().float(), (image_size, image_size))


def _foreground_mask(reference: torch.Tensor) -> torch.Tensor:
    threshold = reference.detach().float().flatten(start_dim=1).quantile(0.70, dim=1)
    threshold = threshold.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))
    return (reference.detach().float() >= threshold).float()


def _foreground_dice(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    pred_mask = _foreground_mask(prediction)
    ref_mask = _foreground_mask(reference)
    intersection = (pred_mask * ref_mask).flatten(start_dim=1).sum(dim=1)
    denom = pred_mask.flatten(start_dim=1).sum(dim=1) + ref_mask.flatten(
        start_dim=1
    ).sum(dim=1)
    return ((2.0 * intersection + 1.0e-6) / denom.clamp_min(1.0e-6)).mean()


def _corr_to_reference(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    pred = prediction.detach().float().flatten(start_dim=1)
    ref = reference.detach().float().flatten(start_dim=1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    denom = pred.norm(dim=1) * ref.norm(dim=1)
    return (pred * ref).sum(dim=1).div(denom.clamp_min(1.0e-8)).mean()


def _background_speckle_ratio(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    background = 1.0 - _foreground_mask(reference)
    pred = prediction.detach().float().clamp_min(0.0)
    bg_mean = (pred * background).flatten(start_dim=1).sum(dim=1) / background.flatten(
        start_dim=1
    ).sum(dim=1).clamp_min(1.0)
    fg = _foreground_mask(prediction)
    fg_mean = (pred * fg).flatten(start_dim=1).sum(dim=1) / fg.flatten(start_dim=1).sum(
        dim=1
    ).clamp_min(1.0)
    return (bg_mean / fg_mean.clamp_min(1.0e-6)).mean()


def _readable_contrast(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    foreground = _foreground_mask(reference)
    background = 1.0 - foreground
    pred = prediction.detach().float().clamp_min(0.0)
    fg_mean = (pred * foreground).flatten(start_dim=1).sum(dim=1) / foreground.flatten(
        start_dim=1
    ).sum(dim=1).clamp_min(1.0)
    bg_mean = (pred * background).flatten(start_dim=1).sum(dim=1) / background.flatten(
        start_dim=1
    ).sum(dim=1).clamp_min(1.0)
    return (fg_mean / bg_mean.clamp_min(1.0e-6)).mean()


def _pairwise_mse(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().flatten(start_dim=1)
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    deltas = flat.unsqueeze(0) - flat.unsqueeze(1)
    pair_mse = deltas.square().mean(dim=-1)
    mask = ~torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    return pair_mse[mask].mean()


def _stripe_energy_ratio(tensor: torch.Tensor) -> torch.Tensor:
    raw = tensor.detach().float()
    dx = raw[..., :, 1:] - raw[..., :, :-1]
    dy = raw[..., 1:, :] - raw[..., :-1, :]
    stripe_energy = dx.square().mean() + dy.square().mean()
    image_energy = raw.square().mean().clamp_min(1.0e-8)
    return stripe_energy / image_energy


def _propagate_center_padded(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_size: float,
    pad_factor: int,
    method: str = "angular_spectrum",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Propagate with center zero-padding to mitigate FFT circular wraparound.

    Returns (cropped_field, diagnostics_dict).
    """
    propagator = (
        angular_spectrum_propagate
        if method == "angular_spectrum"
        else fresnel_propagate
    )
    h, w = field.shape[-2:]
    pH = h * pad_factor
    pW = w * pad_factor
    pad_h = (pH - h) // 2
    pad_w = (pW - w) // 2

    padded = F.pad(field.real, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
    padded_imag = F.pad(
        field.imag, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0
    )
    padded_field = torch.complex(padded, padded_imag)

    propagated = propagator(padded_field, distance, wavelength, pixel_size)
    cropped = propagated[..., pad_h : pad_h + h, pad_w : pad_w + w]

    total_energy = float(
        propagated.abs().pow(2).flatten(start_dim=1).sum(dim=1).mean().item()
    )
    crop_energy = float(
        cropped.abs().pow(2).flatten(start_dim=1).sum(dim=1).mean().item()
    )
    return cropped, {
        "pad_factor": pad_factor,
        "propagation_method": method,
        "distance_m": distance,
        "full_field_energy_mean": total_energy,
        "center_crop_energy_mean": crop_energy,
        "crop_retention_mean": crop_energy / max(total_energy, 1e-10),
    }


def _audit_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    output = output.detach().float().clamp(0.0, 1.0)
    reference = reference.detach().float().clamp(0.0, 1.0)
    return {
        "mse_to_slm": float((output - reference).square().mean().item()),
        "corr_to_slm": float(_corr_to_reference(output, reference).item()),
        "foreground_dice_to_prototype": float(
            _foreground_dice(output, reference).item()
        ),
        "background_speckle_ratio": float(
            _background_speckle_ratio(output, reference).item()
        ),
        "readable_contrast": float(_readable_contrast(output, reference).item()),
        "pairwise_mse": float(_pairwise_mse(output).item()),
        "stripe_energy_ratio": float(_stripe_energy_ratio(output).item()),
    }


@contextmanager
def _temporary_unit_d2nn_masks(model):
    stacks = [getattr(model, name, None) for name in ("encoder", "decoder")]
    saved: list[tuple[object, torch.Tensor | None, torch.Tensor | None]] = []
    try:
        with torch.no_grad():
            for stack in stacks:
                if stack is None or not hasattr(stack, "phase"):
                    continue
                saved.append(
                    (
                        stack,
                        stack.phase.detach().clone(),
                        stack.amplitude_logits.detach().clone()
                        if hasattr(stack, "amplitude_logits")
                        else None,
                    )
                )
                stack.phase.zero_()
                if hasattr(stack, "amplitude_logits"):
                    stack.amplitude_logits.fill_(20.0)
        yield
    finally:
        with torch.no_grad():
            for stack, phase, amplitude_logits in saved:
                stack.phase.copy_(phase)
                if amplitude_logits is not None and hasattr(stack, "amplitude_logits"):
                    stack.amplitude_logits.copy_(amplitude_logits)


def build_one_shot_optical_identity_audit(
    torch_module,
    *,
    one_shot_model,
    codes: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    one_shot_model.eval()
    image_size = int(one_shot_model.image_size)
    normalize_percentile = float(one_shot_model.physics.readout.normalize_percentile)
    with torch_module.no_grad():
        slm_command, generator_diagnostics = (
            one_shot_model.generator_e.forward_with_diagnostics(codes, labels)
        )
        slm_outputs = one_shot_model.slm(slm_command)
        slm_reference = slm_outputs["slm_command"].detach().float().clamp(0.0, 1.0)
        slm_transfer = slm_outputs["slm_transfer"]

        variants: dict[str, torch.Tensor] = {
            "generator_direct": slm_reference,
        }

        front_field = one_shot_model.front_path.forward_field(
            slm_transfer,
            propagation_method="angular_spectrum",
        )
        variants["front_path_only_intensity"] = _normalize_plane(
            front_field.abs().pow(2), normalize_percentile
        )

        # --- v9.3 front-path decomposition ---
        macro = one_shot_model.physics.macro_optics
        optics_cfg = one_shot_model.physics.optics
        wl, ps = optics_cfg.wavelength, optics_cfg.pixel_size

        # SLM transfer intensity (the amplitude-only pattern before any propagation).
        variants["slm_transfer_intensity"] = _normalize_plane(
            slm_transfer.abs().pow(2),
            normalize_percentile,
        )

        # Manually step through FrontOpticalPath.forward_field to capture intermediates.
        _propa = angular_spectrum_propagate
        probe_field = torch.ones_like(slm_transfer, dtype=torch.complex64)

        probe_field = _propa(probe_field, macro.dist_laser_to_m1, wl, ps)
        probe_field = one_shot_model.front_path.m1(probe_field)
        probe_field = _propa(probe_field, macro.dist_m1_to_bs, wl, ps)
        probe_field = one_shot_model.front_path.bs_in(probe_field)
        probe_field = _propa(probe_field, macro.dist_bs_to_slm, wl, ps)
        variants["front_after_pre_slm_plane"] = _normalize_plane(
            probe_field.abs().pow(2),
            normalize_percentile,
        )

        probe_field = one_shot_model.front_path.slm(slm_transfer, probe_field)
        variants["front_after_slm_only"] = _normalize_plane(
            probe_field.abs().pow(2),
            normalize_percentile,
        )
        # Keep a clone at the SLM output for padded isolation tests below.
        slm_modulated_field = probe_field.clone()

        probe_field = _propa(probe_field, macro.dist_slm_to_bs, wl, ps)
        variants["front_after_slm_to_bs"] = _normalize_plane(
            probe_field.abs().pow(2),
            normalize_percentile,
        )
        probe_field = one_shot_model.front_path.bs_out(probe_field)
        probe_field = _propa(probe_field, macro.dist_bs_to_m3, wl, ps)
        variants["front_after_bs_to_m3"] = _normalize_plane(
            probe_field.abs().pow(2),
            normalize_percentile,
        )

        probe_field = one_shot_model.front_path.m3(probe_field)
        probe_field = _propa(probe_field, macro.dist_m3_to_lg, wl, ps)
        variants["front_after_m3_to_lg"] = _normalize_plane(
            probe_field.abs().pow(2),
            normalize_percentile,
        )

        # Fresnel variant of the full front path.
        front_fresnel = one_shot_model.front_path.forward_field(
            slm_transfer, propagation_method="fresnel"
        )
        variants["front_path_current_fresnel"] = _normalize_plane(
            front_fresnel.abs().pow(2),
            normalize_percentile,
        )

        # Alias: current front_path angular_spectrum result.
        variants["front_path_current_angular_spectrum"] = _normalize_plane(
            front_field.abs().pow(2),
            normalize_percentile,
        )

        # Zero-distance bypass: SLM modulation only, no free-space propagation.
        zero_field = torch.ones_like(slm_transfer, dtype=torch.complex64)
        zero_field = one_shot_model.front_path.slm(slm_transfer, zero_field)
        variants["front_path_zero_distance_bypass"] = _normalize_plane(
            zero_field.abs().pow(2),
            normalize_percentile,
        )

        # v9.5 relay-identity front path: ideal SLM-to-encoder imaging.
        relay_field = torch.ones_like(slm_transfer, dtype=torch.complex64)
        relay_field = one_shot_model.front_path.slm(slm_transfer, relay_field)
        variants["front_path_relay_identity_intensity"] = _normalize_plane(
            relay_field.abs().pow(2),
            normalize_percentile,
        )

        # --- v9.4 padded front-path variants ---
        variant_diagnostics: dict[str, list[dict[str, object]]] = {}

        # Per-segment padded: propagate each distance in isolation from the SLM output.
        # This tells us if padding rescues THAT segment without cumulative degradation.
        for segment_dist, segment_name in [
            (macro.dist_slm_to_bs, "slm_to_bs"),
            (macro.dist_bs_to_m3, "bs_to_m3"),
            (macro.dist_m3_to_lg, "m3_to_lg"),
        ]:
            for pf in (2, 4):
                variant_name = f"front_after_{segment_name}_padded{pf}x"
                padded, diag = _propagate_center_padded(
                    slm_modulated_field,
                    segment_dist,
                    wl,
                    ps,
                    pf,
                    method="angular_spectrum",
                )
                variants[variant_name] = _normalize_plane(
                    padded.abs().pow(2), normalize_percentile
                )
                variant_diagnostics[variant_name] = [diag]

        # Full-chain padded front path: replicate all steps with padded propagation.
        for method in ("angular_spectrum", "fresnel"):
            for pf in (2, 4):
                variant_name = f"front_path_current_{method}_padded{pf}x"
                fp = torch.ones_like(slm_transfer, dtype=torch.complex64)
                step_diags: list[dict[str, object]] = []
                # laser → m1
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_laser_to_m1, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                fp = one_shot_model.front_path.m1(fp)
                # m1 → bs_in
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_m1_to_bs, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                fp = one_shot_model.front_path.bs_in(fp)
                # bs_in → slm
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_bs_to_slm, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                # SLM modulation
                fp = one_shot_model.front_path.slm(slm_transfer, fp)
                # slm → bs_out
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_slm_to_bs, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                fp = one_shot_model.front_path.bs_out(fp)
                # bs_out → m3
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_bs_to_m3, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                fp = one_shot_model.front_path.m3(fp)
                # m3 → lg (output)
                fp, sd = _propagate_center_padded(
                    fp, macro.dist_m3_to_lg, wl, ps, pf, method=method
                )
                step_diags.append(sd)
                variants[variant_name] = _normalize_plane(
                    fp.abs().pow(2), normalize_percentile
                )
                variant_diagnostics[variant_name] = step_diags

        clean_chip = torch.complex(slm_reference, torch.zeros_like(slm_reference))
        back_clean = one_shot_model.back_path(clean_chip)
        variants["back_path_only_clean_chip"] = _normalize_plane(
            _match_image_size(back_clean["camera_intensity_raw"], image_size),
            normalize_percentile,
        )
        variants["back_path_only_clean_chip_probe"] = _match_image_size(
            back_clean["probe_plane_loss"], image_size
        ).clamp(0.0, 1.0)

        # --- v9.6b back-path / camera readout audit ---
        # Clear aliases for the existing clean-chip back-path results.
        variants["back_current_clean_chip_probe"] = variants[
            "back_path_only_clean_chip_probe"
        ]
        variants["back_current_clean_chip_camera"] = variants[
            "back_path_only_clean_chip"
        ]

        # Zero-mask D2NN → chip output for back-path comparison.
        with _temporary_unit_d2nn_masks(one_shot_model):
            _zero_forward = one_shot_model._forward_diffractive(slm_outputs, labels)
        zero_mask_chip_intensity = (
            _zero_forward["chip_intensity_primary"].detach().float()
        )
        zero_mask_chip_field = torch.complex(
            zero_mask_chip_intensity.sqrt().clamp_min(0.0),
            torch.zeros_like(zero_mask_chip_intensity),
        )
        back_zero = one_shot_model.back_path(zero_mask_chip_field)
        variants["back_current_zero_mask_chip_probe"] = _match_image_size(
            back_zero["probe_plane_loss"], image_size
        ).clamp(0.0, 1.0)
        variants["back_current_zero_mask_chip_camera"] = _normalize_plane(
            _match_image_size(back_zero["camera_intensity_raw"], image_size),
            normalize_percentile,
        )

        # Identity bypass: chip intensity directly as camera (no back path).
        variants["back_identity_clean_chip_camera"] = _normalize_plane(
            _match_image_size(clean_chip.abs().pow(2), image_size),
            normalize_percentile,
        )
        variants["back_identity_zero_mask_chip_camera"] = _normalize_plane(
            _match_image_size(zero_mask_chip_intensity, image_size),
            normalize_percentile,
        )

        # Padded back-path replication: chip → probe → lens → camera, each
        # segment propagated with center zero-padding.
        macro = one_shot_model.physics.macro_optics
        _wl = optics_cfg.wavelength
        _ps = optics_cfg.pixel_size

        def _back_audit_variant(
            chip_field: torch.Tensor,
            *,
            pad_factor: int = 1,
            skip_lens_propagation: bool = False,
        ) -> torch.Tensor:
            """Simulate back-path readout for audit, with optional padding."""
            # chip → probe plane
            if pad_factor > 1:
                probe_field, _ = _propagate_center_padded(
                    chip_field,
                    macro.dist_lg_to_relay_lens,
                    _wl,
                    _ps,
                    pad_factor,
                    method="angular_spectrum",
                )
            else:
                probe_field = angular_spectrum_propagate(
                    chip_field,
                    macro.dist_lg_to_relay_lens,
                    _wl,
                    _ps,
                )
            # Thin lens
            lens_field = apply_thin_lens(probe_field, macro.f_relay, _wl, _ps)
            if skip_lens_propagation:
                camera_intensity = lens_field.abs().pow(2)
            elif pad_factor > 1:
                camera_field, _ = _propagate_center_padded(
                    lens_field,
                    macro.dist_relay_lens_to_cmos,
                    _wl,
                    _ps,
                    pad_factor,
                    method="angular_spectrum",
                )
                camera_intensity = camera_field.abs().pow(2)
            else:
                camera_field = angular_spectrum_propagate(
                    lens_field,
                    macro.dist_relay_lens_to_cmos,
                    _wl,
                    _ps,
                )
                camera_intensity = camera_field.abs().pow(2)
            return _normalize_plane(
                _match_image_size(camera_intensity, image_size),
                normalize_percentile,
            )

        variants["back_padded4x_clean_chip_camera"] = _back_audit_variant(
            clean_chip,
            pad_factor=4,
        )
        variants["back_padded4x_zero_mask_chip_camera"] = _back_audit_variant(
            zero_mask_chip_field,
            pad_factor=4,
        )
        # Lens-only: apply thin lens but skip free-space propagation.
        variants["back_lens_only_clean_chip_camera"] = _back_audit_variant(
            clean_chip,
            pad_factor=1,
            skip_lens_propagation=True,
        )
        # Normalization-only: no propagation at all, just normalize chip intensity.
        variants["back_camera_normalization_only_clean_chip"] = _normalize_plane(
            _match_image_size(clean_chip.abs().pow(2), image_size),
            normalize_percentile,
        )
        # --- end v9.6b ---

        # --- v9.7+ coherent front PSF relay ---
        # Build from the SLM-modulated field (same base as relay_identity) then
        # apply complex-field Gaussian blur.  sigma=0 must be strictly identical.
        slm_field = torch.ones_like(slm_transfer, dtype=torch.complex64)
        slm_field = one_shot_model.front_path.slm(slm_transfer, slm_field)
        for sigma_px, label in [(0.0, "sigma0"), (1.0, "sigma1"), (2.0, "sigma2")]:
            blurred = _gaussian_blur_complex_field(slm_field, sigma_px)
            variants[f"front_relay_coherent_psf_{label}"] = _normalize_plane(
                blurred.abs().pow(2),
                normalize_percentile,
            )

        # --- v9.7+ back PSF relay (intensity-domain) ---
        # Apply intensity-domain Gaussian blur to the identity relay chip.
        chip_intensity = clean_chip.abs().pow(2)
        for sigma_px, label in [(0.0, "sigma0"), (1.0, "sigma1"), (2.0, "sigma2")]:
            blurred_intensity = _gaussian_blur_intensity(chip_intensity, sigma_px)
            variants[f"back_relay_psf_{label}"] = _normalize_plane(
                _match_image_size(blurred_intensity, image_size),
                normalize_percentile,
            )

        # Compute relay PSF acceptance metrics.
        for sigma_key in ("sigma0", "sigma1", "sigma2"):
            front_key = f"front_relay_coherent_psf_{sigma_key}"
            back_key = f"back_relay_psf_{sigma_key}"
            if front_key not in variants or back_key not in variants:
                continue
        # --- end v9.7+ ---

        grid_size = int(one_shot_model._d2nn["grid_size"])
        front_grid = _resize_complex_field(front_field, (grid_size, grid_size))
        front_back = one_shot_model.back_path(front_grid)
        variants["front_plus_back_no_d2nn"] = _normalize_plane(
            _match_image_size(front_back["camera_intensity_raw"], image_size),
            normalize_percentile,
        )
        variants["front_plus_chip_no_d2nn"] = _normalize_plane(
            _match_image_size(front_grid.abs().pow(2), image_size),
            normalize_percentile,
        )

        with _temporary_unit_d2nn_masks(one_shot_model):
            zero_outputs = one_shot_model._forward_diffractive(slm_outputs, labels)
            variants["zero_mask_d2nn_1x1"] = (
                zero_outputs["physical_camera_plane"].detach().float()
            )
            variants["zero_mask_d2nn_1x1_chip_intensity"] = (
                zero_outputs["chip_intensity_primary"].detach().float()
            )
            variants["zero_mask_d2nn_1x1_probe"] = (
                zero_outputs["probe_plane"].detach().float()
            )
            zero_mask_checks = {
                "encoder_mask_amplitude_allclose_one": bool(
                    torch_module.allclose(
                        zero_outputs["diffractive_encoder_mask_amplitude"],
                        torch_module.ones_like(
                            zero_outputs["diffractive_encoder_mask_amplitude"]
                        ),
                        atol=1.0e-5,
                    )
                ),
                "decoder_mask_amplitude_allclose_one": bool(
                    torch_module.allclose(
                        zero_outputs["diffractive_decoder_mask_amplitude"],
                        torch_module.ones_like(
                            zero_outputs["diffractive_decoder_mask_amplitude"]
                        ),
                        atol=1.0e-5,
                    )
                ),
                "encoder_mask_phase_max_abs": float(
                    zero_outputs["diffractive_encoder_mask_phase"].abs().max().item()
                ),
                "decoder_mask_phase_max_abs": float(
                    zero_outputs["diffractive_decoder_mask_phase"].abs().max().item()
                ),
            }

        small_outputs = one_shot_model._forward_diffractive(slm_outputs, labels)
        variants["small_phase_amp_d2nn_1x1"] = (
            small_outputs["physical_camera_plane"].detach().float()
        )
        variants["small_phase_amp_d2nn_1x1_chip_intensity"] = (
            small_outputs["chip_intensity_primary"].detach().float()
        )
        variants["small_phase_amp_d2nn_1x1_probe"] = (
            small_outputs["probe_plane"].detach().float()
        )

    metrics = {
        name: _audit_metrics(output, slm_reference) for name, output in variants.items()
    }

    # --- relay PSF acceptance summary ---
    relay_psf_acceptance: dict[str, object] = {}
    for sigma_label in ("sigma0", "sigma1", "sigma2"):
        front_key = f"front_relay_coherent_psf_{sigma_label}"
        back_key = f"back_relay_psf_{sigma_label}"
        entry: dict[str, object] = {"front": {}, "back": {}}
        if front_key in metrics:
            for mk in ("corr_to_slm", "foreground_dice_to_prototype", "mse_to_slm"):
                entry["front"][mk] = metrics[front_key][mk]
        if back_key in metrics:
            for mk in ("corr_to_slm", "foreground_dice_to_prototype", "mse_to_slm"):
                entry["back"][mk] = metrics[back_key][mk]
        relay_psf_acceptance[sigma_label] = entry
    # Auto-judge: sigma0 variants must have corr >= 0.95 and dice >= 0.95.
    f0 = relay_psf_acceptance["sigma0"]
    front0 = f0["front"]
    back0 = f0["back"]
    relay_psf_acceptance["sigma0_acceptance"] = bool(
        front0["corr_to_slm"] >= 0.95
        and back0["corr_to_slm"] >= 0.95
        and front0["foreground_dice_to_prototype"] >= 0.95
        and back0["foreground_dice_to_prototype"] >= 0.95
    )
    # --- end relay PSF acceptance ---

    return {
        "labels": labels.detach().cpu(),
        "codes": codes.detach().cpu(),
        "slm_reference": slm_reference.detach().cpu(),
        "generator_diagnostics": {
            key: value.detach().cpu() for key, value in generator_diagnostics.items()
        },
        "variants": {key: value.detach().cpu() for key, value in variants.items()},
        "metrics": metrics,
        "zero_mask_checks": zero_mask_checks,
        "variant_diagnostics": variant_diagnostics,
        "relay_psf_acceptance": relay_psf_acceptance,
    }


def _shared_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        tensor.detach()
        .float()
        .clamp_min(0.0)
        .div(scale.clamp_min(1.0e-6))
        .clamp(0.0, 1.0)
    )


def _save_variant_artifacts(
    run_manager, *, variant: str, output: torch.Tensor, reference: torch.Tensor, labels
) -> dict[str, str]:
    row_labels = labels_for_visual_rows(labels)
    variant_dir = run_manager.images_dir / "one_shot_identity_audit"
    image_path = run_manager.save_tensor_grid(
        {variant: output},
        variant_dir / f"{variant}_intensity.png",
        title=f"one-shot identity audit {variant}",
        max_items=int(output.shape[0]),
        row_labels=row_labels,
    )
    shared_scale = (
        torch.cat([reference, output], dim=0).flatten(start_dim=1).quantile(0.995)
    )
    shared_path = run_manager.save_tensor_grid(
        {
            "slm reference": _shared_scale(reference, shared_scale),
            variant: _shared_scale(output, shared_scale),
        },
        variant_dir / f"{variant}_shared_scale_intensity.png",
        title=f"one-shot identity audit {variant} shared scale",
        max_items=int(output.shape[0]),
        row_labels=row_labels,
    )
    return {
        "intensity": str(image_path.relative_to(run_manager.run_dir)),
        "shared_scale_intensity": str(shared_path.relative_to(run_manager.run_dir)),
    }


def run_one_shot_optical_identity_audit_stage(
    torch_module,
    *,
    one_shot_model,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_generator_checkpoint: str | None,
) -> None:
    if one_shot_generator_checkpoint is None:
        raise ValueError(
            "one_shot_optical_identity_audit requires --one-shot-generator-checkpoint."
        )
    payload = torch_module.load(
        one_shot_generator_checkpoint, map_location=device, weights_only=False
    )
    summary = load_one_shot_checkpoint_subset(
        one_shot_model,
        payload,
        include_prefixes=("generator_e.",),
        description="one-shot generator identity audit checkpoint",
    )
    one_shot_model.to(device)

    class_labels = tuple(range(int(train_cfg.one_shot.num_classes)))
    labels = torch_module.tensor(class_labels, dtype=torch_module.long, device=device)
    generator = torch_module.Generator(device=device)
    generator.manual_seed(int(train_cfg.distillation.code_seed))
    codes = torch_module.randn(
        len(class_labels),
        int(train_cfg.one_shot.code_dim),
        generator=generator,
        device=device,
    )

    audit = build_one_shot_optical_identity_audit(
        torch_module,
        one_shot_model=one_shot_model,
        codes=codes,
        labels=labels,
    )
    reference = audit["slm_reference"]
    labels_cpu = audit["labels"]
    image_paths = {
        variant: _save_variant_artifacts(
            run_manager,
            variant=variant,
            output=output,
            reference=reference,
            labels=labels_cpu,
        )
        for variant, output in audit["variants"].items()
    }
    tensor_paths = {
        variant: str(
            run_manager.save_tensor_bundle(
                f"one_shot_identity_audit/{variant}_tensors.pt",
                {
                    "variant": variant,
                    "labels": labels_cpu,
                    "codes": audit["codes"],
                    "slm_reference": reference,
                    "output": output,
                },
            ).relative_to(run_manager.run_dir)
        )
        for variant, output in audit["variants"].items()
    }
    relay_psf_acceptance = audit.get("relay_psf_acceptance", {})
    metrics_path = run_manager.save_artifact_json(
        "one_shot_identity_audit_metrics",
        {
            "variants": audit["metrics"],
            "variant_diagnostics": audit["variant_diagnostics"],
            "zero_mask_checks": audit["zero_mask_checks"],
            "relay_psf_acceptance": relay_psf_acceptance,
            "metric_keys": list(IDENTITY_AUDIT_METRIC_KEYS),
            "variant_order": list(IDENTITY_AUDIT_VARIANTS),
        },
    )
    run_manager.save_config_snapshot(
        "one_shot_identity_audit_config",
        {
            "class_labels": list(class_labels),
            "code_seed": int(train_cfg.distillation.code_seed),
            "one_shot_generator_checkpoint": one_shot_generator_checkpoint,
            "generator_warm_start_summary": summary,
        },
    )
    run_meta.update(
        {
            "one_shot_identity_audit_generator_checkpoint": one_shot_generator_checkpoint,
            "one_shot_identity_audit_generator_warm_start_summary": summary,
            "one_shot_identity_audit_metric_source": str(
                metrics_path.relative_to(run_manager.run_dir)
            ),
            "one_shot_identity_audit_images": image_paths,
            "one_shot_identity_audit_tensor_sources": tensor_paths,
        }
    )
    run_manager.write_meta(run_meta)
    print(
        "[one_shot_optical_identity_audit] "
        f"metrics={metrics_path.relative_to(run_manager.run_dir)} variants={list(audit['variants'].keys())}"
    )
