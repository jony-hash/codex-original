"""Stage-local model builders; no stage handler receives unrelated model types."""
from __future__ import annotations

def build_student_model(physics, one_shot_config, *, image_size: int, device):
    """Construct the declared formal or diagnostic student variant.

    A transparent-boundary configuration is intentionally experimental: it
    retains the physical macro path but is not the finite-crop Formal Student
    contract.  Keeping this dispatch here prevents the CLI from sending v68
    diagnostic settings into the Formal validator while leaving all v56/v60
    finite-crop construction unchanged.
    """
    # Resolve at call time so legacy tests/tools can patch the compatibility
    # constructor without importing every stage's model eagerly.
    from ..student import model as student_model

    d2nn = getattr(one_shot_config, "d2nn", None)
    transparent_diagnostic = (
        d2nn is not None and d2nn.exterior_boundary == "transparent_identity"
    )
    model_cls = (
        student_model.FormalOneShotOpticalGenerator
        if physics.physical_forward_contract and not transparent_diagnostic
        else student_model.OneShotOpticalGenerator
    )
    return model_cls(physics, one_shot_config, image_size=image_size).to(device)


__all__ = ["build_student_model"]
