from __future__ import annotations

import inspect

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_integration_canary as canary
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_reconstruction_pilot as pilot


CONFIG = "configs/research/optical_manifold/lightgen_optical_manifold_identity_smf_reconstruction_pilot_v1.yaml"
REVIEW = "configs/research/optical_manifold/lightgen_optical_manifold_identity_smf_reconstruction_pilot_review_v1.json"


def test_reconstruction_contract_review_and_lazy_cli_are_registered() -> None:
    spec = pilot._load_spec(CONFIG)
    pilot._assert_contract(spec)
    review, _ = pilot._load_review(REVIEW, spec)
    assert review["reviewer_role"] == "project_owner"
    assert review["bounded_diagnostic_training_authorized"] is True
    assert review["training_authorized"] is False

    parser = _parser()
    common = [
        "research", "optical-manifold", "smf-reconstruction-validate",
        "--config", CONFIG, "--review-contract", REVIEW,
        "--dataset-root", "data", "--output", "validation.json",
    ]
    parsed = parser.parse_args(common)
    assert parsed.research_action == "smf-reconstruction-validate"
    for action, extra in (
        (
            "smf-reconstruction-preflight",
            ["--validation-evidence", "validation.json"],
        ),
        (
            "smf-reconstruction-train",
            [
                "--validation-evidence", "validation.json",
                "--preflight-evidence", "preflight.json",
            ],
        ),
    ):
        parsed = parser.parse_args(
            [
                "research", "optical-manifold", action,
                "--config", CONFIG, "--review-contract", REVIEW,
                "--dataset-root", "data", "--output", "result.json", *extra,
            ]
        )
        assert parsed.research_action == action


def test_schedule_is_balanced_seeded_and_excludes_prior_indices() -> None:
    labels = torch.arange(60000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55000, dtype=torch.long)
    excluded = torch.cat(
        [torch.arange(label, 55000, 10)[:100] for label in range(10)]
    )
    schedule = pilot._build_schedule(
        labels, train_indices, excluded, steps=100, seed=pilot.SEED
    )
    repeat = pilot._build_schedule(
        labels, train_indices, excluded, steps=100, seed=pilot.SEED
    )
    assert torch.equal(schedule, repeat)
    assert schedule.shape == (100, 10)
    assert torch.unique(schedule).numel() == schedule.numel()
    assert not torch.isin(schedule.reshape(-1), excluded).any()
    assert torch.equal(
        labels.index_select(0, schedule.reshape(-1)).reshape(schedule.shape),
        torch.arange(10).expand(100, -1),
    )


def test_phase_update_gate_rejects_partial_metadata() -> None:
    assert not canary._four_phase_nonzero({})
    assert not canary._four_phase_nonzero(
        {
            "system.encoder.phase[0]": 1.0,
            "system.encoder.phase[1]": 1.0,
            "system.decoder.phase[0]": 1.0,
        }
    )
    assert canary._four_phase_nonzero(
        {
            "system.encoder.phase[0]": 1.0,
            "system.encoder.phase[1]": 1.0,
            "system.decoder.phase[0]": 1.0,
            "system.decoder.phase[1]": 1.0,
        }
    )


def test_structure_metrics_are_finite_and_label_aware() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    features = torch.randn(20, 8, generator=generator, dtype=torch.float64).to(torch.complex128)
    labels = torch.arange(10).repeat_interleave(2)
    metrics = pilot._structure_metrics(features, labels)
    assert metrics["finite"] is True
    assert metrics["sample_count"] == 20
    assert metrics["feature_dimension_real_imag"] == 16
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in (
        "entropy_effective_rank", "participation_rank", "inter_intra_ratio", "silhouette"
    ))


def test_reconstruction_uses_checkpointed_mse_and_has_no_swd_training_path() -> None:
    source = inspect.getsource(pilot._identity_gradients)
    assert "activation_checkpoint=True" in source
    assert inspect.getsource(pilot._step).count("_mse_gradients") == 1
    assert "swd" not in inspect.getsource(pilot._step).lower()
    assert "load_frozen_evaluator" not in inspect.getsource(pilot)


def test_validation_consumer_binds_review_file_hash() -> None:
    source = inspect.getsource(pilot._load_validation)
    assert "file_sha256(review_path)" in source
    assert "record.input_hashes != expected" in source


def test_preflight_consumer_does_not_require_a_self_referential_hash() -> None:
    source = inspect.getsource(pilot._load_preflight)
    assert 'contract.get("training_hashes") != expected' in source
    assert '"training_preflight": digest' in source


def test_interpolation_keeps_a_batch_dimension_for_formal_decoder(monkeypatch) -> None:
    shapes: list[tuple[int, ...]] = []

    def fake_decode(_model, field, _smf):
        shapes.append(tuple(field.shape))
        return field.real

    monkeypatch.setattr(pilot, "_decode_pre_ols", fake_decode)
    pre = torch.arange(100 * 1 * 2 * 2, dtype=torch.float32).reshape(100, 1, 2, 2).to(torch.complex64)
    camera = pre.real.clone()
    result = pilot._interpolation_metrics(None, None, pre, camera, torch.device("cpu"))
    assert result["finite"] is True
    assert len(shapes) == 50
    assert all(shape == (1, 1, 2, 2) for shape in shapes)


def test_resume_rejects_completed_arm_and_foreign_attempt() -> None:
    source = inspect.getsource(pilot._load_resume)
    assert "target.parent.resolve() != output_dir.resolve()" in source
    assert "not 0 < step < STEPS" in source
    assert "evidence_hashes" in source
