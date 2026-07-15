from __future__ import annotations

def test_legacy_train_cli_entrypoint_remains_importable():
    from ldm_project import train_cli
    assert callable(train_cli.main)


def test_stage_runtime_registry_remains_importable():
    from ldm_project.application.stage_runtimes import build_stage_runtime
    assert callable(build_stage_runtime)
