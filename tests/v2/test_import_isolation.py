from __future__ import annotations

import subprocess
import sys


def _run(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_importing_v2_does_not_initialize_v1() -> None:
    _run(
        """
import sys
import ldm_project.v2
assert not any(name.startswith('ldm_project.v1') for name in sys.modules)
"""
    )


def test_building_v2_parser_does_not_load_legacy_execution_domains() -> None:
    _run(
        """
import sys
from ldm_project.v2.application.cli import _parser
_parser()
forbidden = (
    'ldm_project.v1',
    'ldm_project.v2.teacher',
    'ldm_project.v2.student.capacity',
    'ldm_project.v2.student.direct_template',
    'ldm_project.v2.optics.lightgen',
    'ldm_project.v2.optics.fiber_ols',
)
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(
    name for name in sys.modules if name.startswith(forbidden)
)
"""
    )


def test_identity_command_dependencies_exclude_legacy_execution_domains() -> None:
    _run(
        """
import sys
from ldm_project.v2.application.current.training import conditional_train
assert callable(conditional_train)
assert 'ldm_project.v2.application.current.runtime' in sys.modules
assert 'ldm_project.v2.application.identity_runtime' not in sys.modules
assert 'ldm_project.v2.application.conditional_runtime' not in sys.modules
forbidden = (
    'ldm_project.v1',
    'ldm_project.v2.teacher',
    'ldm_project.v2.student.capacity',
    'ldm_project.v2.student.direct_template',
    'ldm_project.v2.optics.lightgen',
    'ldm_project.v2.optics.fiber_ols',
)
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(
    name for name in sys.modules if name.startswith(forbidden)
)
"""
    )


def test_identity_import_closure_stays_within_the_formal_budget() -> None:
    _run(
        """
import sys
from pathlib import Path
from ldm_project.v2.application.current.training import conditional_train
assert callable(conditional_train)
root = Path.cwd() / 'src' / 'ldm_project' / 'v2'
files = {
    Path(module.__file__).resolve()
    for name, module in sys.modules.items()
    if name.startswith('ldm_project.v2') and getattr(module, '__file__', None)
}
closure_lines = sum(
    len(path.read_text(encoding='utf-8').splitlines())
    for path in files
    if path.suffix == '.py' and root in path.parents
)
assert closure_lines <= 8036, closure_lines
"""
    )


def test_parser_does_not_load_an_execution_runtime_or_torch() -> None:
    _run(
        """
import sys
from ldm_project.v2.application.cli import _parser
_parser()
forbidden = (
    'torch',
    'ldm_project.v1',
    'ldm_project.v2.application.current',
    'ldm_project.v2.application.legacy',
    'ldm_project.v2.application.diagnostic',
)
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(
    name for name in sys.modules if name.startswith(forbidden)
)
"""
    )


def test_teacher_legacy_domain_does_not_install_unrelated_legacy_domains() -> None:
    _run(
        """
import sys
import ldm_project.v2.application.legacy.teacher
forbidden = (
    'ldm_project.v2.student.capacity',
    'ldm_project.v2.student.direct_template',
    'ldm_project.v2.optics.lightgen',
    'ldm_project.v2.optics.fiber_ols',
    'ldm_project.v2.application.diagnostic',
)
assert not any(name.startswith(forbidden) for name in sys.modules), sorted(
    name for name in sys.modules if name.startswith(forbidden)
)
"""
    )
