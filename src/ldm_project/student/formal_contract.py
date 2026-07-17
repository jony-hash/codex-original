"""Validation for the current formal single-exposure student contract."""
from __future__ import annotations


def validate_formal_student_config(physics, config) -> None:
    """Raise before construction when a diagnostic or compatibility switch is active."""
    invalid = []
    if not physics.physical_forward_contract:
        invalid.append("physics.physical_forward_contract")
    if config.optical_core_arch != "diffractive_phase_only":
        invalid.append("one_shot.optical_core_arch")
    if config.ols_kind != "identity":
        invalid.append("one_shot.ols_kind")
    if config.decoder_diagnostic_mode != "normal":
        invalid.append("one_shot.decoder_diagnostic_mode")
    if config.decoder_diagnostic_modulation_override != "none":
        invalid.append("one_shot.decoder_diagnostic_modulation_override")
    if config.decoder_diagnostic_mask_component != "trained":
        invalid.append("one_shot.decoder_diagnostic_mask_component")
    if config.target_code_diagnostic_mode != "none":
        invalid.append("one_shot.target_code_diagnostic_mode")
    if config.target_code_bridge_source != "none":
        invalid.append("one_shot.target_code_bridge_source")
    if config.ols_transfer_calibration != "none":
        invalid.append("one_shot.ols_transfer_calibration")
    if config.d2nn is None or config.d2nn.modulation != "phase_only":
        invalid.append("one_shot.d2nn.modulation")
    if config.d2nn is None or config.d2nn.exterior_boundary != "finite_crop":
        invalid.append("one_shot.d2nn.exterior_boundary")
    if config.d2nn is None or config.d2nn.transparent_propagation_method != "periodic_asm":
        invalid.append("one_shot.d2nn.transparent_propagation_method")
    if config.d2nn is None or config.d2nn.transparent_fft_grid_size is not None:
        invalid.append("one_shot.d2nn.transparent_fft_grid_size")
    if config.d2nn is None or config.d2nn.transparent_edge_guard_fraction != 0.05:
        invalid.append("one_shot.d2nn.transparent_edge_guard_fraction")
    if config.d2nn is None or config.d2nn.transparent_memory_mode != "legacy_full_capture":
        invalid.append("one_shot.d2nn.transparent_memory_mode")
    if invalid:
        raise ValueError(
            "FormalOneShotOpticalGenerator rejects diagnostic/compatibility settings: "
            + ", ".join(invalid)
        )


__all__ = ["validate_formal_student_config"]
