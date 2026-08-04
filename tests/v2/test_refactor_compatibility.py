"""Regression baselines for the v2 lifecycle-boundary refactor."""

from __future__ import annotations

from pathlib import Path

from ldm_project.v2.application import cli
from ldm_project.v2.application.legacy import cli_impl
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]

CONFIG_HASHES = {
    "current/formal_identity_logit_amplitude_finite_crop_identity.yaml": "77db9b7fcfe47cb53f2d089c40454e811a898a329a8f694b8370b8d17d81706d",
    "diagnostic/conditional_class_conditioned_modes_finite_crop_identity.yaml": "387cc7c69074b97dc9596bd178329f398b6083faeaf8fb175653e5839a81697b",
    "diagnostic/conditional_continuous_mode_capacity_finite_crop_identity.yaml": "d1c4833e1d2fd444a3095ac2ca1da060017266795bc84b7da730cfe59e0214e2",
    "diagnostic/conditional_multi_anchor_local_modes_finite_crop_identity.yaml": "8f89f92baf93a46f37f8b77385c753a82a1e07daf942dcc243943e2223c28655",
    "diagnostic/conditional_nonlinear_expanded_manifold.yaml": "dd99e844e29465d1cd50df686c4a7f07418e90e5ee510d9284508afef3e76def",
    "diagnostic/conditional_parity_stabilized_modes_finite_crop_identity.yaml": "63c8e0f0978965aa3bc283096bdfae56aed90c91b1f56930e90881468e1dd09a",
    "diagnostic/conditional_pca_formal_spacing_self_imaging_finite_crop_identity.yaml": "2f6a2b3812b3671f11180aec89beab8d1971f2aa9c8c5191dc1827883a771316",
    "diagnostic/conditional_pca_near_conjugate_relay.yaml": "3b8e8124dad4e9ee638dad4840992b6acdc230f0d4b192414e0b14dc38dc4cc9",
    "diagnostic/conditional_pca_self_imaging_transport_finite_crop_identity_short_spacing.yaml": "bc7af4457d31a699d6f12189ecf9b9de6b8aeb95871e22f77017efe11f99615f",
    "diagnostic/conditional_pca_transport_codesign_finite_crop_identity.yaml": "1e3957b3257f7eec5f53ec89f308f176d1d66f1ac6fd8516fe8bc899f25bce53",
    "diagnostic/conditional_transport_aligned_manifold_v71.yaml": "bcbb953276eec5a281f9f519c756457d570429372dd3ca317c8fddeef513ebc7",
    "diagnostic/conditional_transport_aligned_phase_capacity_v72.yaml": "db15499fa2d5529c7cb90369acaadc0d646683729407128ddfd7275d8b37687f",
    "diagnostic/conditional_transport_aligned_evaluation_qualified_v1.yaml": "c9c662f4ed98d2a33146db0f769219f5aea0c6b273953db13c2621b19206859f",
    "diagnostic/conditional_transport_aligned_evaluation_qualified_v1_1.yaml": "5ab6306be15de834f6c42fa66b440dd72aabf07957b874e93ab4fa8e88c28ace",
    "diagnostic/lightgen_amplitude_fiber_ols_400.yaml": "52c4dca8ed79f06bf9240c87bd02a91e525cb0570571f8f0d63ed973f6858c5d",
    "diagnostic/small_amplitude_finite_crop_identity.yaml": "ac36ddb3a18f5f5bd9c4d406b12dfdce62158d1e29fb7c9d436b368554f90d33",
    "legacy/formal_global_amplitude_finite_crop_identity.yaml": "0048249d7b35848c8b2db4351bc353239d8d59c060e05c888017c099a5136040",
    "legacy/formal_prototype_shape_guard.yaml": "d99c6a433cb56e6c1d11f13d2103deae008436d4e93fbf8975656aed6fdbce7c",
}


def test_active_v2_configuration_hashes_are_stable() -> None:
    config_root = ROOT / "configs" / "v2"
    actual = {
        path.relative_to(config_root).as_posix(): canonical_sha256(
            load_config(path).model_dump(mode="json")
        )
        for path in sorted(config_root.rglob("*.yaml"))
        if path.name != "path_migration_manifest.yaml"
    }
    assert actual == CONFIG_HASHES


def test_legacy_cli_module_delegates_to_the_canonical_parser() -> None:
    cases = (
        ["student", "train", "--config", "identity.yaml"],
        [
            "student",
            "calibrate-prototypes",
            "--config",
            "legacy.yaml",
            "--clean-target-manifest",
            "clean.json",
        ],
        [
            "audit",
            "conditional-pca-transport-admission",
            "--config",
            "diagnostic.yaml",
            "--dataset-root",
            "data",
            "--evaluator-checkpoint",
            "evaluator.pt",
            "--semantic-checkpoint",
            "semantic.pt",
            "--safe-scale-evidence",
            "scale.json",
        ],
    )
    for argv in cases:
        assert vars(cli_impl._parser().parse_args(argv)) == vars(
            cli._parser().parse_args(argv)
        )


def test_moved_diagnostic_import_paths_resolve_to_the_new_implementations() -> None:
    from ldm_project.v2.application.conditional_pca_transport import _metrics
    from ldm_project.v2.application.diagnostic.pca_transport import _metrics as moved

    assert _metrics is moved


def test_current_and_diagnostic_recovery_share_the_same_workflow_implementation() -> (
    None
):
    from ldm_project.v2.application.current import recovery as current
    from ldm_project.v2.application.diagnostic import recovery as diagnostic

    for name in (
        "_conditional_optimizer",
        "controllability_smoke",
        "semantic_recovery_smoke",
    ):
        current_workflow = getattr(current, name)
        diagnostic_workflow = getattr(diagnostic, name)
        assert current_workflow.__module__.endswith("_conditional_recovery_shared")
        assert diagnostic_workflow.__module__ == current_workflow.__module__
        assert diagnostic_workflow.__code__ is current_workflow.__code__
