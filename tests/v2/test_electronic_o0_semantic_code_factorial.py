from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import electronic_o0_semantic_code_factorial as scf


def _base_measurement() -> dict:
    row = {
        "safety": {
            "camera_code_gate": True,
            "diversity_gate": True,
            "coverage_gate": True,
            "saturation_gate": True,
            "label_shift_gate": True,
            "memorization_gate": True,
            "exact_duplicate_gate": True,
            "safety_gate": True,
        },
        "individual_semantic_gate": True,
        "agreement_gate": True,
        "semantic_gate": True,
        "evaluator_finite": True,
        "finite": True,
        "no_update_replay_exact": True,
        "chunk_64_257_exact": True,
    }
    permutation_row = {
        "individual_semantic_gate": True,
        "agreement_gate": True,
        "semantic_gate": True,
        "evaluator_finite": True,
        "finite": True,
    }
    arms = {
        name: {
            "banks": {
                bank: {
                    **row,
                    "safety": dict(row["safety"]),
                }
                for bank in ({"gaussian_1", "gaussian_2"} if name.endswith("G") else {"posterior_a", "posterior_b"})
            },
            "semantic_pass": True,
            "safety_pass": True,
        }
        for name in ("S-G", "S-P", "D-G", "D-P")
    }
    return {
        "optimizer_created": False,
        "parameter_hashes": {"unchanged": True},
        "historical_reproduction": {"all_matches": True},
        "posterior_hash_match": True,
        "chunk_invariance": True,
        "finite": True,
        "arms": arms,
        "controls": {
            "D-P-within-class-permutation": {
                "posterior_a": dict(permutation_row),
                "posterior_b": dict(permutation_row),
            },
            "permutation_semantic_pass": True,
        },
    }


def _logits(prediction: torch.Tensor) -> torch.Tensor:
    logits = torch.zeros((prediction.numel(), 10))
    logits.scatter_(1, prediction[:, None], 1.0)
    return logits


def test_config_and_cli_contract_are_registered() -> None:
    root = Path(__file__).parents[2]
    config = scf._config(root / "configs/research/electronic_o0_semantic_code_factorial_v1.yaml")
    assert config["harness_seed"] == 8811
    assert config["gaussian_bank_seeds"] == [223607, 244949]
    assert config["samples_per_class_per_bank"] == 250
    for action in ("validate", "preflight", "audit", "report", "closeout"):
        args = _parser().parse_args([
            "research", "electronic-o0-semantic-code-factorial", action,
            "--run-manifest", "manifest.yaml",
        ])
        assert args.research_action == action


def test_manifest_only_dispatch_rejects_legacy_invocation() -> None:
    args = _parser().parse_args([
        "research", "electronic-o0-semantic-code-factorial", "validate",
        "--config", "config.yaml",
    ])
    with pytest.raises(ValueError, match="manifest-only"):
        scf.dispatch(args)


def test_registered_schemas_and_hashes_are_fixed() -> None:
    for value in (
        scf.REGISTERED_WAE_CONFIRM_HASH,
        scf.REGISTERED_WAE_TRAINING_HASH,
        scf.REGISTERED_WAE_DECODER_HASH,
        scf.REGISTERED_LATENT_ATTRIBUTION_HASH,
        scf.REGISTERED_H_DIDM_DECODER_HASH,
        scf.REGISTERED_H_DIDM_SCREEN_HASH,
        scf.REGISTERED_H_DIDM_REPORT_HASH,
        scf.REGISTERED_H_DIDM_CLOSEOUT_HASH,
        scf.REGISTERED_PRIMARY_EVALUATOR_HASH,
        scf.REGISTERED_SECONDARY_EVALUATOR_HASH,
        scf.REGISTERED_PRIMARY_QUALIFICATION_HASH,
        scf.REGISTERED_SECONDARY_QUALIFICATION_HASH,
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")
    assert scf.SOURCE_DECODER_SCHEMA == "electronic_o0_wae_decoder.v2"
    assert scf.DIDM_DECODER_SCHEMA == "electronic_o0_direct_image_distribution_decoder.v1"
    assert scf.LATENT_ATTRIBUTION_SCHEMA == "electronic_o0_latent_transport_attribution.v1"


def test_gaussian_and_posterior_banks_are_deterministic_and_disjoint() -> None:
    first_codes, first_labels = scf._gaussian_bank(223607, 250, torch.device("cpu"))
    second_codes, second_labels = scf._gaussian_bank(223607, 250, torch.device("cpu"))
    assert torch.equal(first_codes, second_codes)
    assert torch.equal(first_labels, second_labels)
    posterior_codes = torch.arange(5000 * 8, dtype=torch.float32).reshape(5000, 8)
    posterior_labels = torch.arange(10).repeat_interleave(500)
    banks = scf._balanced_posterior_banks(posterior_codes, posterior_labels, 250)
    assert banks["posterior_a"][0].shape == (2500, 8)
    assert banks["posterior_b"][0].shape == (2500, 8)
    assert torch.equal(banks["posterior_a"][1], banks["posterior_b"][1])
    with pytest.raises(ValueError, match="too few"):
        scf._balanced_posterior_banks(posterior_codes[:1000], posterior_labels[:1000], 250)


def test_permutation_label_shift_zero_code_and_chunk_controls_are_deterministic() -> None:
    codes = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
    labels = torch.arange(10)
    permutation = scf._posterior_permutation(codes, labels, 1)
    assert torch.equal(permutation, codes)
    repeated_labels = torch.arange(10).repeat_interleave(3)
    repeated_codes = torch.arange(30 * 2, dtype=torch.float32).reshape(30, 2)
    shifted = torch.roll(repeated_labels, 3)
    assert torch.equal(shifted[:3], torch.tensor([9, 9, 9]))
    assert torch.equal(shifted[3:], repeated_labels[:-3])
    permuted = scf._posterior_permutation(repeated_codes, repeated_labels, 3)
    assert torch.equal(permuted[:3], torch.roll(repeated_codes[:3], 1, dims=0))
    rows = torch.arange(257 * 3, dtype=torch.float32).reshape(257, 3)
    references = torch.arange(311 * 3, dtype=torch.float32).reshape(311, 3)
    first = scf._nearest_distances(rows, references, chunk_size=64)
    second = scf._nearest_distances(rows, references, chunk_size=257)
    assert torch.equal(first, second)


def test_memorization_context_reuses_validation_thresholds(monkeypatch) -> None:
    train = torch.rand(20, 1, 2, 2)
    train_labels = torch.arange(10).repeat_interleave(2)
    validation = torch.rand(10, 1, 2, 2)
    validation_labels = torch.arange(10)
    calls = []
    original = scf._nearest_distances

    def counted(rows, references, *, chunk_size=64):
        calls.append((rows.shape[0], references.shape[0], chunk_size))
        return original(rows, references, chunk_size=chunk_size)

    monkeypatch.setattr(scf, "_nearest_distances", counted)
    prepared = scf._prepare_memorization_context(
        train, train_labels, validation, validation_labels, torch.device("cpu")
    )
    assert len(calls) == 10
    generated = torch.rand(10, 1, 2, 2)
    generated_labels = torch.arange(10)
    scf._memorization(generated, generated_labels, prepared)
    scf._memorization(generated, generated_labels, prepared)
    assert len(calls) == 30
    assert all(chunk_size == 64 for _, _, chunk_size in calls)


def test_historical_panel_metric_counts_use_exact_integer_reconstruction() -> None:
    panel = {
        "agreement": 2375 / 2500,
        "primary": {"accuracy": 2375 / 2500, "recall": [225 / 250] * 10},
        "secondary": {"accuracy": 2400 / 2500, "recall": [230 / 250] * 10},
    }
    counts = scf._panel_metric_counts(panel, samples_per_class=250)
    assert counts["primary"]["correct_count"] == 2375
    assert counts["primary"]["recall_counts"] == [225] * 10
    assert counts["secondary"]["correct_count"] == 2400
    assert counts["agreement_count"] == 2375


def test_missing_structure_and_nonfinite_measurements_are_invalid_harness() -> None:
    missing = _base_measurement()
    del missing["arms"]["D-P"]["banks"]["posterior_b"]
    assert scf._classify(missing)[0] == "invalid_harness"
    nonfinite = _base_measurement()
    nonfinite["arms"]["D-G"]["banks"]["gaussian_1"]["finite"] = False
    nonfinite["finite"] = False
    assert scf._classify(nonfinite)[0] == "invalid_harness"


def test_semantic_count_gates_use_exact_boundary_counts() -> None:
    labels = torch.arange(10).repeat_interleave(250)
    primary = labels.clone()
    secondary = labels.clone()
    primary[:26] = 1
    secondary[:26] = 1
    row = scf._semantic_row(_logits(primary), _logits(secondary), labels)
    assert row["primary"]["correct_count"] == 2474
    assert row["primary"]["recall_counts"][0] == 224
    assert row["individual_semantic_gate"] is False
    primary[:26] = labels[:26]
    secondary[:26] = labels[:26]
    row = scf._semantic_row(_logits(primary), _logits(secondary), labels)
    assert row["semantic_gate"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("invalid", "invalid_harness"),
        ("safety", "semantic_counterfactual_safety_failed"),
        ("consensus", "generated_evaluator_consensus_failed"),
        ("reachability", "didm_decoder_semantic_reachability_failed"),
        ("pairing", "posterior_instance_pairing_dependency"),
        ("mass", "gaussian_semantic_mass_failed"),
        ("stability", "gaussian_semantic_stability_failed"),
    ),
)
def test_classification_is_fail_closed(mutation: str, expected: str) -> None:
    measurement = _base_measurement()
    if mutation == "invalid":
        measurement["parameter_hashes"]["unchanged"] = False
    elif mutation == "safety":
        measurement["arms"]["S-G"]["banks"]["gaussian_1"]["safety"]["safety_gate"] = False
    elif mutation == "consensus":
        for row in measurement["arms"]["D-P"]["banks"].values():
            row["agreement_gate"] = False
            row["semantic_gate"] = False
    elif mutation == "reachability":
        for row in measurement["arms"]["D-P"]["banks"].values():
            row["individual_semantic_gate"] = False
            row["semantic_gate"] = False
        measurement["arms"]["D-P"]["semantic_pass"] = False
    elif mutation == "pairing":
        measurement["controls"]["permutation_semantic_pass"] = False
    elif mutation == "mass":
        measurement["arms"]["D-G"]["semantic_pass"] = False
        for row in measurement["arms"]["D-G"]["banks"].values():
            row["semantic_gate"] = False
    elif mutation == "stability":
        pass
    classification, _ = scf._classify(measurement)
    assert classification == expected


def test_no_optimizer_or_checkpoint_writer_is_present() -> None:
    source = Path(scf.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "atomic_torch_save" not in source


@pytest.mark.parametrize("action", ("report", "closeout"))
def test_consumers_do_not_initialize_cuda_models_or_evaluators(action, tmp_path, monkeypatch) -> None:
    args = SimpleNamespace(
        research_action=action,
        run_manifest=str(tmp_path / "manifest.yaml"),
        config=str(tmp_path / "config.yaml"),
        dataset_root=str(tmp_path),
        output=str(tmp_path / f"{action}.json"),
        measurement_evidence=str(tmp_path / "audit.json"),
    )
    measurement = {
        "passed": False,
        "measurement_valid": True,
        "classification": "didm_decoder_semantic_reachability_failed",
        "next_action": "close_frozen_decoder_derived_semantic_route",
    }
    monkeypatch.setattr(scf, "prepare_args", lambda namespace, **_: namespace)
    monkeypatch.setattr(scf, "validate_manifest_namespace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scf, "_config", lambda _path: {})
    monkeypatch.setattr(scf, "file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(scf, "_check_consumer", lambda *_args: measurement)
    monkeypatch.setattr(scf, "_stage", lambda *_args, **_kwargs: {"action": "CREATE"})
    monkeypatch.setattr(scf, "_write", lambda *_args, **_kwargs: None)
    for name in ("_require_cuda", "_load_context", "_load_decoder", "_load_secondary", "load_frozen_evaluator"):
        monkeypatch.setattr(scf, name, lambda *_args, **_kwargs: pytest.fail(f"unexpected runtime initialization: {name}"))
    assert scf.dispatch(args) == 0


def test_tampered_upstream_is_rejected_before_evidence_creation(tmp_path, monkeypatch) -> None:
    names = {
        "wae_training": scf.REGISTERED_WAE_TRAINING_HASH,
        "wae_decoder": scf.REGISTERED_WAE_DECODER_HASH,
        "wae_confirm": scf.REGISTERED_WAE_CONFIRM_HASH,
        "latent_attribution": scf.REGISTERED_LATENT_ATTRIBUTION_HASH,
        "h_scf_final_decoder": scf.REGISTERED_H_DIDM_DECODER_HASH,
        "h_didm_screen": "tampered",
        "h_didm_report": scf.REGISTERED_H_DIDM_REPORT_HASH,
        "h_didm_closeout": scf.REGISTERED_H_DIDM_CLOSEOUT_HASH,
        "primary_evaluator": scf.REGISTERED_PRIMARY_EVALUATOR_HASH,
        "secondary_evaluator": scf.REGISTERED_SECONDARY_EVALUATOR_HASH,
        "primary_qualification": scf.REGISTERED_PRIMARY_QUALIFICATION_HASH,
        "secondary_qualification": scf.REGISTERED_SECONDARY_QUALIFICATION_HASH,
    }
    paths = {name: tmp_path / f"{name}.artifact" for name in names}
    for path in paths.values():
        path.write_text("placeholder", encoding="utf-8")
    dataset_root = tmp_path / "mnist"
    dataset_root.mkdir()
    argument_paths = {
        "wae_training_checkpoint": paths["wae_training"],
        "wae_decoder_checkpoint": paths["wae_decoder"],
        "wae_confirm_evidence": paths["wae_confirm"],
        "latent_transport_attribution": paths["latent_attribution"],
        "h_scf_final_decoder_checkpoint": paths["h_scf_final_decoder"],
        "h_didm_screen_evidence": paths["h_didm_screen"],
        "h_didm_report_evidence": paths["h_didm_report"],
        "h_didm_closeout_evidence": paths["h_didm_closeout"],
        "primary_evaluator_checkpoint": paths["primary_evaluator"],
        "secondary_evaluator_checkpoint": paths["secondary_evaluator"],
        "primary_qualification_evidence": paths["primary_qualification"],
        "secondary_qualification_evidence": paths["secondary_qualification"],
    }
    args = SimpleNamespace(
        dataset_root=str(dataset_root),
        config=str(tmp_path / "config.yaml"),
        **{name: str(path) for name, path in argument_paths.items()},
    )

    def fake_file_sha256(path):
        for name, candidate in paths.items():
            if Path(path) == candidate:
                return names[name]
        return "config-hash"

    monkeypatch.setattr(scf, "file_sha256", fake_file_sha256)
    with pytest.raises(ValueError, match="registered immutable input"):
        scf._lineage(args, {})
