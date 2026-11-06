from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_8bit_replay_v1 as replay


def test_u8_round_half_up_endpoints_and_error_bound() -> None:
    half_steps = torch.tensor([0.0, 0.5 / 255.0, 1.5 / 255.0, 0.5, 254.5 / 255.0, 1.0], dtype=torch.float64)
    codes = replay.quantize_u8(half_steps)
    assert codes.tolist() == [0, 1, 2, 128, 255, 255]
    decoded = replay.decode_u8(codes)
    assert torch.max(torch.abs(decoded - half_steps)).item() <= 0.5 / 255.0 + 1.0e-15
    assert decoded[0].item() == 0.0
    assert decoded[-1].item() == 1.0


def test_a8_and_i8_keep_their_distinct_quantization_order() -> None:
    amplitude = torch.tensor([0.5], dtype=torch.float64)
    a8_amplitude = replay.decode_u8(replay.quantize_u8(amplitude))
    a8_intensity = a8_amplitude.square()
    i8_intensity = replay.decode_u8(replay.quantize_u8(amplitude.square()))
    assert replay.quantize_u8(amplitude).tolist() == [128]
    assert replay.quantize_u8(amplitude.square()).tolist() == [64]
    assert a8_intensity.item() == (128.0 / 255.0) ** 2
    assert i8_intensity.item() == 64.0 / 255.0
    assert not torch.equal(a8_intensity, i8_intensity)


@pytest.mark.parametrize("values", [
    torch.tensor([float("nan")]),
    torch.tensor([float("inf")]),
    torch.tensor([-1.0]),
    torch.tensor([1.0001]),
])
def test_u8_rejects_nonfinite_or_out_of_range_commands(values: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        replay.quantize_u8(values)


def test_saved_gray_bundle_round_trips_and_rejects_tampering() -> None:
    amplitude = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    amplitude[0, 0, 0, :4] = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    identities = [{"sample_key": "G1_00000", "method": "G1", "cohort_row": 0,
                   "label": 0, "proposal_id": 5, "component_id": 2}]
    bundle = replay._build_encoded_bundle(amplitude, identities, "source-run")
    replay._validate_encoded_bundle(bundle, amplitude, identities, "source-run")
    replay._verify_saved_gray_decoding(amplitude, bundle)
    assert bundle["A8_gray_u8"][0, 0, 0, :4].tolist() == [0, 64, 128, 255]
    assert bundle["I8_gray_u8"][0, 0, 0, :4].tolist() == [0, 16, 64, 255]
    tampered = dict(bundle)
    tampered["A8_gray_u8"] = bundle["A8_gray_u8"].clone()
    tampered["A8_gray_u8"][0, 0, 0, 1] += 1
    with pytest.raises(RuntimeError, match="gray command or decoded input changed"):
        replay._validate_encoded_bundle(tampered, amplitude, identities, "source-run")


def test_population_identity_consumer_rejects_missing_or_duplicate_rows() -> None:
    identities = [
        {"sample_key": "G1_00000", "method": "G1", "cohort_row": 0,
         "label": 0, "proposal_id": 5, "component_id": 2},
        {"sample_key": "G1_00001", "method": "G1", "cohort_row": 1,
         "label": 0, "proposal_id": 6, "component_id": 2},
    ]
    replay._validate_identity_population(identities, identities.copy())
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        replay._validate_identity_population(identities, identities[:1])
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        replay._validate_identity_population(identities, [identities[0], identities[0]])


def test_independent_metric_comparison_rejects_tampered_output_value() -> None:
    reference = torch.ones((4, 4), dtype=torch.float64)
    actual = reference.clone()
    producer_metric = {"direct_cmos_intensity_relative_error_2048": replay._relative(actual, reference)}
    replay._assert_metrics_match(producer_metric, dict(producer_metric))
    actual[0, 0] = 5.0
    recomputed = {"direct_cmos_intensity_relative_error_2048": replay._relative(actual, reference)}
    with pytest.raises(RuntimeError, match="independently recomputed metric"):
        replay._assert_metrics_match(recomputed, producer_metric)


def test_power_table_keeps_the_two_ols_reference_planes_separate() -> None:
    assert "P_before_OLS_incident_W" in replay.POWER_TRACE_FIELDS
    assert "P_after_OLS_reemitted_decoder_entry_W" in replay.POWER_TRACE_FIELDS
    assert "P_encoder_layer_2_after_aperture_W" in replay.POWER_TRACE_FIELDS
    assert "P_generator_entry_W" not in replay.POWER_TRACE_FIELDS
