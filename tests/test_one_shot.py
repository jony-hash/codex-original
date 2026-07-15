from __future__ import annotations

def test_legacy_one_shot_entrypoints_remain_importable():
    from ldm_project.one_shot import ElectronicGeneratorE
    from ldm_project.student.generator import ElectronicGeneratorE as CanonicalGenerator
    assert ElectronicGeneratorE is CanonicalGenerator


def test_objective_facade_keeps_public_loss_entrypoint():
    from ldm_project.student.objectives import one_shot_loss
    assert callable(one_shot_loss)
