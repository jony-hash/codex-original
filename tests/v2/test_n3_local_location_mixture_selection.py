from __future__ import annotations

import torch

from scripts.research import execute_n3_local_location_mixture_selection_v1 as consumer
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison


def test_seed_derivation_is_stable_and_substream_specific() -> None:
    first = consumer._seed("N3_TEST_NAMESPACE", 2, "data-vs-model", 0, 0)
    repeat = consumer._seed("N3_TEST_NAMESPACE", 2, "data-vs-model", 0, 0)
    other_k = consumer._seed("N3_TEST_NAMESPACE", 4, "data-vs-model", 0, 0)
    other_role = consumer._seed("N3_TEST_NAMESPACE", 2, "model-vs-model", 0, 0)
    assert first == repeat
    assert first != other_k
    assert first != other_role


def test_synthetic_em_is_deterministic_and_uses_only_shared_covariance() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    first = torch.randn((40, 100), generator=generator, dtype=torch.float64) * 0.03 + 0.20
    second = torch.randn((40, 100), generator=generator, dtype=torch.float64) * 0.03 + 0.70
    q = torch.cat([first, second], dim=0).clamp_min(0.0)
    shared = comparison._fit_shrinkage(q)
    whitened = torch.linalg.solve_triangular(
        shared["chol"], (q - shared["mu"]).transpose(0, 1), upper=False
    ).transpose(0, 1).contiguous()
    context = {
        "shared": shared,
        "whitened": whitened,
        "fit_source_indices": torch.arange(q.shape[0], dtype=torch.int64),
    }
    result = consumer._fit_mixture(
        context,
        fold=0,
        k=2,
        settings={
            "candidate_K": [2, 4, 8],
            "max_iterations": 300,
            "convergence_tolerance": 1.0e-8,
            "monotonicity_tolerance_per_row": 1.0e-10,
            "mass_floor": 1.0e-8,
        },
    )
    assert result["summary"]["converged"] is True
    assert result["summary"]["fit_count"] == 80
    assert len(result["summary"]["component_weights"]) == 2
    assert torch.isfinite(result["component_means"]).all()
    assert torch.allclose(result["weights"].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_selection_uses_smallest_compatible_k_and_fail_closed_states() -> None:
    def candidate(k: int, compatible: bool, operational: bool = True) -> dict[str, object]:
        return {
            "K": k,
            "compatible": compatible,
            "operationally_valid": operational,
        }
    selected = consumer._select_model_order(
        [candidate(2, False), candidate(4, True), candidate(8, True)]
    )
    assert selected["classification"] == consumer.CLASSIFICATION_SELECTED
    assert selected["K_selected"] == 4

    unsupported = consumer._select_model_order(
        [candidate(2, False), candidate(4, False), candidate(8, False)]
    )
    assert unsupported["classification"] == consumer.CLASSIFICATION_NOT_SUPPORTED

    inconclusive = consumer._select_model_order(
        [candidate(2, False, False), candidate(4, False), candidate(8, False)]
    )
    assert inconclusive["classification"] == consumer.CLASSIFICATION_INCONCLUSIVE
