"""Contract tests for the independent v2.3 electronic-oracle lifecycle."""

from __future__ import annotations

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research.o0_feasibility import _classification, _selected_records
from ldm_project.v2.application.research.measurement import fixed_panels, panel_hashes
from ldm_project.v2.application.research.o0_models import ProjectionDiscriminator, build_o0_candidate, inference_state_dict
from ldm_project.v2.application.research.o0_objectives import generator_loss, validate_loss_contract
from ldm_project.v2.application.research.o0_spec import O0AggregateDecision, load_o0_spec


def test_o0_cli_is_lazy_and_has_all_lifecycle_actions() -> None:
    for action in ("validate", "preflight", "controls", "probe", "smoke", "confirm", "summarize", "closeout"):
        arguments = ["research", "o0-feasibility", action, "--config", "configs/research/conditional_o0_feasibility_v2_3.yaml"]
        if action in {"preflight", "probe", "smoke", "confirm"}:
            arguments += ["--candidate", "D0", "--prior-evidence", "prior.json"]
        if action == "preflight":
            arguments.remove("--prior-evidence")
            arguments.remove("prior.json")
        if action in {"summarize", "closeout"}:
            arguments += ["--evidence-root", "outputs"]
        parsed = _parser().parse_args(arguments)
        assert parsed.research_program == "o0-feasibility"
        assert parsed.research_action == action


def test_o0_config_and_candidate_parameter_contract() -> None:
    spec = load_o0_spec("configs/research/conditional_o0_feasibility_v2_3.yaml")
    assert spec.seeds == (1234, 4321, 2234)
    assert {candidate.candidate_id for candidate in spec.candidates} == {"D0", "C0", "C1", "C2", "C3"}
    for candidate in spec.candidates:
        model = build_o0_candidate(candidate.candidate_id)
        assert sum(parameter.numel() for parameter in model.parameters()) <= candidate.parameter_budget
    r2 = load_o0_spec("configs/research/conditional_o0_feasibility_v2_3_r2.yaml")
    assert r2.harness_revision == 2 and r2.require_clean_git and r2.stage_endpoint_checkpoint


def test_o0_generators_shapes_finite_and_anchor_contract() -> None:
    labels = torch.tensor([0, 4, 9])
    for candidate in ("D0", "C0", "C1", "C2", "C3"):
        model = build_o0_candidate(candidate)
        code = torch.randn(3, model.code_dim)
        output = model(code, labels)
        assert output.shape == ((3, 1, 28, 28) if candidate == "D0" else (3, 1, 200, 200))
        assert torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
    model = build_o0_candidate("C1")
    zero = torch.zeros(3, 8)
    expected = torch.sigmoid(model.carrier_logits.index_select(0, labels))
    expected = torch.nn.functional.interpolate(expected, size=(200, 200), mode="bilinear", align_corners=False)
    assert torch.allclose(model(zero, labels), expected)
    soft = build_o0_candidate("C2")
    assert soft(zero, labels).shape == expected.shape


def test_o0_training_adversary_is_not_in_inference_checkpoint_and_has_gradients() -> None:
    generator = build_o0_candidate("D0")
    discriminator = ProjectionDiscriminator(code_dim=32, image_size=28)
    labels = torch.arange(10).repeat_interleave(2)
    code = torch.randn(20, 32)
    generated = generator(code, labels)
    score, _, _, _ = discriminator(generated, labels)
    loss, _ = generator_loss(discriminator, generated, labels, code)
    loss.backward()
    assert score.shape == (20,)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in generator.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in discriminator.parameters())
    assert all("discriminator" not in key and "latent_head" not in key for key in inference_state_dict(generator))
    assert validate_loss_contract()["r1_interval"] == 16.0


def test_o0_panels_separate_8d_and_32d_code_contracts() -> None:
    panels8 = fixed_panels(device=torch.device("cpu"), code_dim=8)
    panels32 = fixed_panels(device=torch.device("cpu"), code_dim=32)
    assert all(codes.shape == (170, 8) and labels.shape == (170,) for codes, labels in panels8.values())
    assert all(codes.shape == (170, 32) and labels.shape == (170,) for codes, labels in panels32.values())
    assert panel_hashes(device=torch.device("cpu"), code_dim=8) != panel_hashes(device=torch.device("cpu"), code_dim=32)


def test_o0_aggregate_never_authorizes_c3() -> None:
    decision = O0AggregateDecision("C3", "input_manifold_bottleneck", 2, seed_results=(
        {"seed": 1234, "passed": True}, {"seed": 4321, "passed": True}, {"seed": 2234, "passed": False},
    ))
    assert decision.to_dict()["physical_authorization"] is False


def test_o0_revision_conflicts_are_excluded_and_latest_revision_wins() -> None:
    base = {"config_hash": "c", "dataset": "mnist", "candidate": "D0", "action": "controls", "seed": None}
    records = [
        base | {"evidence_revision": 1, "content_hash": "a", "state": "control_passed"},
        base | {"evidence_revision": 1, "content_hash": "b", "state": "control_failed"},
        base | {"evidence_revision": 2, "content_hash": "c", "state": "control_passed"},
    ]
    selected = _selected_records(records)
    assert len(selected) == 1 and selected[0]["evidence_revision"] == 2
    assert _classification([{ "action": "final", "candidate": "D0", "state": "final_passed" }]) == "d0_image_oracle_reachable"
    assert _classification([{ "action": "probe", "candidate": "D0", "state": "probe_failed" }]) == "training_objective_or_evaluator_inconclusive"
