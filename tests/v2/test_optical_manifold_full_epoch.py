from __future__ import annotations

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import optical_manifold_full_epoch as full_epoch


def test_full_epoch_config_review_and_source_checkpoint_are_bound() -> None:
    spec = full_epoch._load_spec(
        "configs/research/lightgen_optical_manifold_identity_full_epoch_v1.yaml"
    )
    review, _ = full_epoch._load_review(
        "configs/research/lightgen_optical_manifold_identity_full_epoch_review_v1.json",
        spec,
    )
    source, checkpoint, training_path, checkpoint_path = full_epoch._load_source(
        spec,
        full_epoch._resolve(
            "configs/research/lightgen_optical_manifold_identity_full_epoch_review_v1.json"
        ),
        spec.payload["source_training_evidence"],
        spec.payload["source_checkpoint"],
        full_epoch._git_state()["commit"],
    )
    assert review["full_epoch_research_training_authorized"] is True
    assert source["evidence"]["status"] == "passed"
    assert checkpoint["step"] == 1000
    assert training_path.name == "training.json"
    assert checkpoint_path.name == "pilot_final_step_001000.pt"


def test_continuation_schedule_is_disjoint_and_covers_the_fixed_train_split() -> None:
    data = full_epoch._load_data("data")
    assert data.continuation_schedule.shape == (4500, 10)
    assert data.manifest["continuation_schedule_sha256"] == full_epoch.CONTINUATION_SCHEDULE_HASH
    all_indices = data.full_schedule.reshape(-1)
    assert all_indices.unique().numel() == 55000
    assert all_indices.sort().values.equal(data.pilot_data.train_indices.sort().values)
    assert not set(data.pilot_data.schedule.reshape(-1).tolist()).intersection(
        set(data.continuation_schedule.reshape(-1).tolist())
    )


def test_full_epoch_uses_cpu_shadow_adam_for_exact_resume() -> None:
    spec = full_epoch._load_spec(
        "configs/research/lightgen_optical_manifold_identity_full_epoch_v1.yaml"
    )
    formal, _ = full_epoch._load_formal(spec)
    model = full_epoch.base.IdentityOpticalManifoldAutoencoder(formal)
    optimizer = full_epoch._make_full_optimizer(model)
    assert spec.payload["training"]["optimizer_update_device"] == "cpu_shadow"
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1.0e-3
    assert len(optimizer.param_groups[0]["params"]) == 2
    assert all(parameter.device.type == "cpu" for parameter in optimizer.param_groups[0]["params"])
    assert all(parameter.device.type == "cpu" for parameter in optimizer._cpu_parameters)


def test_full_epoch_cli_actions_are_registered() -> None:
    parser = _parser()
    common = [
        "research",
        "optical-manifold",
        "full-epoch-validate",
        "--config",
        "config.yaml",
        "--review-contract",
        "review.json",
        "--pilot-training-evidence",
        "training.json",
        "--pilot-checkpoint",
        "pilot.pt",
        "--dataset-root",
        "data",
        "--output",
        "validation.json",
    ]
    args = parser.parse_args(common)
    assert args.research_action == "full-epoch-validate"
    for action, extra in (
        (
            "full-epoch-preflight",
            ["--epoch-validation-evidence", "validation.json"],
        ),
        (
            "full-epoch-train",
            [
                "--epoch-validation-evidence",
                "validation.json",
                "--epoch-preflight-evidence",
                "preflight.json",
            ],
        ),
    ):
        args = parser.parse_args(
            [
                "research",
                "optical-manifold",
                action,
                "--config",
                "config.yaml",
                "--review-contract",
                "review.json",
                "--pilot-training-evidence",
                "training.json",
                "--pilot-checkpoint",
                "pilot.pt",
                "--dataset-root",
                "data",
                "--output",
                "output.json",
                *extra,
            ]
        )
        assert args.research_action == action
