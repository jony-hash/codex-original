"""Stage-local model builders; no stage handler receives unrelated model types."""
from __future__ import annotations

def build_student_model(physics, one_shot_config, *, image_size: int, device):
    """Construct the formal student only for the declared physical contract."""
    # Resolve at call time so legacy tests/tools can patch the compatibility
    # constructor without importing every stage's model eagerly.
    from ..student import model as student_model

    model_cls = (
        student_model.FormalOneShotOpticalGenerator
        if physics.physical_forward_contract
        else student_model.OneShotOpticalGenerator
    )
    return model_cls(physics, one_shot_config, image_size=image_size).to(device)


__all__ = ["build_student_model"]
