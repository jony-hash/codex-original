from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_v9_10_label_template_audit.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_v9_10_label_template_audit", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit tool: {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V910AuditToolTests(unittest.TestCase):
    def test_output_root_must_stay_under_outputs(self) -> None:
        tool = _load_tool()
        expected = tool.DEFAULT_OUTPUT_ROOT.resolve()

        self.assertEqual(tool._require_persistent_output_root(tool.DEFAULT_OUTPUT_ROOT), expected)
        with self.assertRaisesRegex(tool.AuditError, "under .*outputs"):
            tool._require_persistent_output_root(Path("/tmp/v9_10_test"))

    def test_config_contract_matches_prototype_baseline(self) -> None:
        contract = _load_tool()._validate_config_contract()

        self.assertEqual(contract["baseline_kind"], "prototype_baseline")
        self.assertFalse(contract["code_conditioning_active"])
        self.assertEqual(contract["generator_e_arch"], "label_template")
        self.assertEqual(contract["generator_command_transform"], "intensity_to_amplitude")

    def test_strict_gate_accepts_only_three_required_checks(self) -> None:
        tool = _load_tool()
        metrics = {
            "strict_readability_acceptance": {
                "passed": False,
                "checks": {
                    "paired_corr": True,
                    "paired_dice": False,
                    "student_high_saturation_ratio": True,
                },
            }
        }

        passed, acceptance = tool._strict_gate(metrics)

        self.assertFalse(passed)
        self.assertFalse(acceptance["checks"]["paired_dice"])
        metrics["strict_readability_acceptance"]["checks"]["legacy_gate"] = True
        with self.assertRaisesRegex(tool.AuditError, "unexpected checks"):
            tool._strict_gate(metrics)

    def test_frozen_report_requires_all_v9_10_groups_and_zero_delta(self) -> None:
        tool = _load_tool()
        report = {
            "passed": True,
            "max_abs_change": 0.0,
            "groups": {
                name: {"max_abs_change": 0.0}
                for name in ("generator_base", "phase", "amplitude")
            },
        }

        self.assertIs(tool._validate_frozen_report({"one_shot_frozen_parameter_check": report}), report)
        report["groups"]["phase"]["max_abs_change"] = 0.1
        with self.assertRaisesRegex(tool.AuditError, "changed"):
            tool._validate_frozen_report({"one_shot_frozen_parameter_check": report})

    def test_summary_path_is_unique_and_does_not_overwrite(self) -> None:
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as tmpdir:
            first = tool._write_summary(Path(tmpdir), {"status": "first"})
            second = tool._write_summary(Path(tmpdir), {"status": "second"})

        self.assertNotEqual(first.name, second.name)
        self.assertEqual(first.parent.name, "reports")


if __name__ == "__main__":
    unittest.main()
