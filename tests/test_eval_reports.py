from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ldm_project.eval_reports import compare_runs, print_summary, summarize_runs


class EvalReportsTests(unittest.TestCase):
    def _make_run_dir(self, base: Path, name: str, meta: dict | None, eval_m: dict | None, sample_m: dict | None) -> Path:
        run_dir = base / name
        run_dir.mkdir()
        if meta is not None:
            (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if eval_m is not None:
            artifacts = run_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "one_shot_distill_eval_metrics.json").write_text(json.dumps(eval_m), encoding="utf-8")
        if sample_m is not None:
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(exist_ok=True)
            (artifacts / "one_shot_sample_metrics.json").write_text(json.dumps(sample_m), encoding="utf-8")
        return run_dir

    # ---- summarize ----

    def test_summarize_eval_run_extracts_all_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "20250101_000000_train_one_shot_distill_eval",
                {
                    "mode": "one_shot_distill_eval",
                    "one_shot_checkpoint": "outputs/runs/fake/checkpoints/one_shot_distill_best.pt",
                    "distill_dataset": "outputs/runs/fake/artifacts/teacher_distill_dataset.pt",
                },
                {
                    "overall": {
                        "count": 20,
                        "paired_mse": 0.02,
                        "paired_corr": 0.81,
                        "paired_dice": 0.76,
                        "student_saturation_ratio": 0.09,
                        "slm_tv": 0.38,
                        "diversity_ratio": 0.53,
                    }
                },
                None,
            )

            summary = summarize_runs([str(base / "20250101_000000_train_one_shot_distill_eval")])
            run_name = "20250101_000000_train_one_shot_distill_eval"
            self.assertIn(run_name, summary)
            e = summary[run_name]
            self.assertEqual(e["stage"], "one_shot_distill_eval")
            self.assertIn("one_shot_distill_best.pt", str(e["checkpoint"]))
            self.assertIn("teacher_distill_dataset.pt", str(e["distill_dataset"]))
            self.assertEqual(e["errors"], [])
            self.assertIsNotNone(e["eval_metrics"])
            self.assertEqual(e["eval_metrics"]["mse"], 0.02)
            self.assertEqual(e["eval_metrics"]["corr"], 0.81)
            self.assertEqual(e["eval_metrics"]["dice"], 0.76)
            self.assertEqual(e["eval_metrics"]["sat"], 0.09)
            self.assertIsNone(e["sample_metrics"])

    def test_summarize_sample_run_extracts_corr_and_dice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "20250101_000000_train_one_shot_sample",
                {"mode": "one_shot_sample"},
                None,
                {
                    "overall": {
                        "count": 20,
                        "dice": 0.62,
                        "corr": 0.59,
                        "hole_proxy": 0.25,
                    }
                },
            )

            summary = summarize_runs([str(base / "20250101_000000_train_one_shot_sample")])
            e = summary["20250101_000000_train_one_shot_sample"]
            self.assertIsNotNone(e["sample_metrics"])
            self.assertIsNone(e["sample_metrics"]["mse"])
            self.assertEqual(e["sample_metrics"]["corr"], 0.59)
            self.assertEqual(e["sample_metrics"]["dice"], 0.62)
            self.assertIsNone(e["eval_metrics"])

    def test_summarize_distill_run_has_best_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "20250101_000000_train_one_shot_distill",
                {
                    "mode": "one_shot_distill",
                    "one_shot_distill_best_loss": 0.1614,
                    "distill_dataset": "outputs/runs/fake/artifacts/teacher_distill_dataset.pt",
                },
                None,
                None,
            )

            summary = summarize_runs([str(base / "20250101_000000_train_one_shot_distill")])
            e = summary["20250101_000000_train_one_shot_distill"]
            self.assertEqual(e["stage"], "one_shot_distill")
            self.assertEqual(e["best_loss"], 0.1614)
            self.assertIsNone(e["eval_metrics"])
            self.assertIsNone(e["sample_metrics"])

    def test_missing_run_meta_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "no_meta_run"
            run_dir.mkdir()

            summary = summarize_runs([str(run_dir)])
            e = summary["no_meta_run"]
            self.assertIn("missing run_meta.json", e["errors"][0])
            self.assertIsNone(e["stage"])

    def test_nonexistent_run_dir_reports_missing_fields(self) -> None:
        summary = summarize_runs(["/nonexistent/path/run"])
        e = summary["run"]
        self.assertIn("missing run_meta.json", e["errors"][0])
        self.assertIsNone(e["eval_metrics"])
        self.assertIsNone(e["sample_metrics"])

    def test_multiple_runs_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(base, "run_a", {"mode": "one_shot_distill_eval"}, None, None)
            self._make_run_dir(base, "run_b", {"mode": "one_shot_sample"}, None, None)

            summary = summarize_runs([str(base / "run_a"), str(base / "run_b")])
            self.assertEqual(len(summary), 2)
            self.assertIn("run_a", summary)
            self.assertIn("run_b", summary)

    def test_print_summary_emits_json(self) -> None:
        import io
        import sys as _sys

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(base, "run_x", {"mode": "one_shot_distill_eval"}, None, None)

            captured = io.StringIO()
            old = _sys.stdout
            _sys.stdout = captured
            try:
                print_summary([str(base / "run_x")])
            finally:
                _sys.stdout = old

            data = json.loads(captured.getvalue())
            self.assertIn("run_x", data)

    # ---- compare ----

    @staticmethod
    def _baseline(metrics: dict | None = None, tolerances: dict | None = None) -> dict:
        return {
            "description": "test baseline",
            "commit": "test",
            "metrics": metrics or {},
            "tolerances": tolerances or {},
        }

    def test_compare_passes_when_all_metrics_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_mse": 0.022, "paired_corr": 0.80, "paired_dice": 0.75, "student_saturation_ratio": 0.10, "slm_tv": 0.40, "diversity_ratio": 0.55}},
                None,
            )
            self._make_run_dir(
                base,
                "sample_run",
                {"mode": "one_shot_sample"},
                None,
                {"overall": {"corr": 0.57, "dice": 0.61}},
            )
            self._make_run_dir(
                base,
                "distill_run",
                {"mode": "one_shot_distill", "one_shot_distill_best_loss": 0.165},
                None,
                None,
            )

            baseline = self._baseline(
                {
                    "eval_mse": 0.020,
                    "eval_corr": 0.807,
                    "eval_dice": 0.756,
                    "eval_sat": 0.091,
                    "eval_slm_tv": 0.382,
                    "eval_diversity_ratio": 0.531,
                    "sample_corr": 0.588,
                    "sample_dice": 0.625,
                    "best_loss": 0.161,
                },
                {
                    "eval_mse": 0.005,
                    "eval_corr": 0.03,
                    "eval_dice": 0.03,
                    "eval_sat": 0.02,
                    "eval_slm_tv": 0.10,
                    "eval_diversity_ratio": 0.15,
                    "sample_corr": 0.05,
                    "sample_dice": 0.05,
                    "best_loss": 0.02,
                },
            )

            result = compare_runs(
                [str(base / "eval_run"), str(base / "sample_run"), str(base / "distill_run")],
                baseline,
            )
            self.assertTrue(result["passed"], f"should pass but got failures: {result['checks']}")
            for check in result["checks"]:
                self.assertTrue(check["passed"], f"metric {check['metric']} should pass: {check}")

    def test_compare_fails_when_corr_drops_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_corr baseline 0.807, tolerance 0.03 → threshold 0.777
            # candidate eval_corr 0.76 < 0.777 → FAIL
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.76, "paired_mse": 0.022, "paired_dice": 0.75, "student_saturation_ratio": 0.10}},
                None,
            )

            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertFalse(result["passed"])
            corr_check = next(c for c in result["checks"] if c["metric"] == "eval_corr")
            self.assertFalse(corr_check["passed"])
            self.assertEqual(corr_check["candidate"], 0.76)

    def test_compare_fails_when_mse_rises_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_mse baseline 0.020, tolerance 0.005 → threshold 0.025
            # candidate eval_mse 0.028 > 0.025 → FAIL
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.80, "paired_mse": 0.028, "paired_dice": 0.75, "student_saturation_ratio": 0.10}},
                None,
            )

            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertFalse(result["passed"])
            mse_check = next(c for c in result["checks"] if c["metric"] == "eval_mse")
            self.assertFalse(mse_check["passed"])
            self.assertEqual(mse_check["candidate"], 0.028)

    def test_compare_fails_when_sat_rises_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_sat baseline 0.091, tolerance 0.02 → threshold 0.111
            # candidate eval_sat 0.115 > 0.111 → FAIL
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.80, "paired_mse": 0.022, "paired_dice": 0.75, "student_saturation_ratio": 0.115}},
                None,
            )

            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertFalse(result["passed"])
            sat_check = next(c for c in result["checks"] if c["metric"] == "eval_sat")
            self.assertFalse(sat_check["passed"])
            self.assertEqual(sat_check["candidate"], 0.115)

    def test_compare_fails_when_slm_tv_rises_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_slm_tv is a total-variation smoothness diagnostic:
            # baseline 0.382, tolerance 0.10 → threshold 0.482
            # candidate 0.50 > 0.482 → FAIL
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"slm_tv": 0.50}},
                None,
            )

            baseline = self._baseline({"eval_slm_tv": 0.382})

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertFalse(result["passed"])
            tv_check = next(c for c in result["checks"] if c["metric"] == "eval_slm_tv")
            self.assertFalse(tv_check["passed"])
            self.assertEqual(tv_check["candidate"], 0.50)
            self.assertEqual(tv_check["direction"], "lower-is-better")

    def test_compare_fails_when_best_loss_rises_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # best_loss baseline 0.161, tolerance 0.02 → threshold 0.181
            # candidate best_loss 0.19 > 0.181 → FAIL
            self._make_run_dir(
                base,
                "distill_run",
                {"mode": "one_shot_distill", "one_shot_distill_best_loss": 0.19},
                None,
                None,
            )

            baseline = self._baseline({"best_loss": 0.161})

            result = compare_runs([str(base / "distill_run")], baseline)
            self.assertFalse(result["passed"])
            bl_check = next(c for c in result["checks"] if c["metric"] == "best_loss")
            self.assertFalse(bl_check["passed"])

    def test_compare_passes_when_corr_at_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_corr baseline 0.807, tolerance 0.03 → threshold 0.777
            # candidate eval_corr 0.777 → exactly at boundary → should pass
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.777, "paired_mse": 0.025, "paired_dice": 0.726, "student_saturation_ratio": 0.111}},
                None,
            )

            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            # All values at exact boundary (baseline ± tolerance) should pass
            self.assertTrue(result["passed"], f"boundary values should pass: {result['checks']}")

    def test_compare_passes_when_slm_tv_at_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # eval_slm_tv baseline 0.382, tolerance 0.10 → threshold 0.482
            # candidate 0.482 → exactly at boundary → should pass
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"slm_tv": 0.482}},
                None,
            )

            baseline = self._baseline({"eval_slm_tv": 0.382})

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertTrue(result["passed"], f"slm_tv boundary should pass: {result['checks']}")
            tv_check = next(c for c in result["checks"] if c["metric"] == "eval_slm_tv")
            self.assertEqual(tv_check["direction"], "lower-is-better")

    def test_compare_missing_required_metric_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Baseline requires eval_slm_tv but run doesn't have it
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.80, "paired_mse": 0.022, "paired_dice": 0.75, "student_saturation_ratio": 0.10}},
                None,
            )

            baseline = self._baseline(
                {
                    "eval_mse": 0.020,
                    "eval_corr": 0.807,
                    "eval_dice": 0.756,
                    "eval_sat": 0.091,
                    "eval_slm_tv": 0.382,  # not present in candidate
                },
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertFalse(result["passed"])
            tv_check = next(c for c in result["checks"] if c["metric"] == "eval_slm_tv")
            self.assertFalse(tv_check["passed"])
            self.assertIsNone(tv_check["candidate"])
            self.assertEqual(tv_check["direction"], "missing")

    def test_compare_skips_baseline_none_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.80, "paired_mse": 0.022}},
                None,
            )

            # eval_dice is null in baseline → should be skipped, not compared
            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": None},
            )

            result = compare_runs([str(base / "eval_run")], baseline)
            self.assertTrue(result["passed"])
            metric_names = {c["metric"] for c in result["checks"]}
            self.assertNotIn("eval_dice", metric_names)

    def test_compare_override_tolerances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # With default tolerance 0.03, corr 0.78 passes (>= 0.807 - 0.03 = 0.777)
            # With override tolerance 0.005, corr 0.78 fails (< 0.807 - 0.005 = 0.802)
            self._make_run_dir(
                base,
                "eval_run",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.78, "paired_mse": 0.022, "paired_dice": 0.75, "student_saturation_ratio": 0.10}},
                None,
            )

            baseline = self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )

            # First verify it passes with default tolerance
            result_default = compare_runs([str(base / "eval_run")], baseline)
            self.assertTrue(result_default["passed"])

            # Then verify it fails with tighter tolerance override
            result_tight = compare_runs([str(base / "eval_run")], baseline, tolerances={"eval_corr": 0.005})
            self.assertFalse(result_tight["passed"])
            corr_check = next(c for c in result_tight["checks"] if c["metric"] == "eval_corr")
            self.assertFalse(corr_check["passed"])

    def test_compare_with_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = self._baseline({"eval_mse": 0.020})
            result = compare_runs([str(Path(tmp) / "nonexistent")], baseline)
            self.assertFalse(result["passed"])
            mse_check = next(c for c in result["checks"] if c["metric"] == "eval_mse")
            self.assertIsNone(mse_check["candidate"])
            self.assertEqual(mse_check["direction"], "missing")

    # ---- CLI ----

    def test_cli_summary_produces_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(base, "run_cli", {"mode": "one_shot_distill_eval"}, None, None)

            proc = subprocess.run(
                [sys.executable, "-m", "ldm_project.eval_reports", "summary", str(base / "run_cli")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
            data = json.loads(proc.stdout)
            self.assertIn("run_cli", data)

    def test_cli_compare_passes_and_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "run_cli",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.80, "paired_mse": 0.022, "paired_dice": 0.75, "student_saturation_ratio": 0.10}},
                None,
            )

            baseline_path = base / "baseline.json"
            baseline_path.write_text(json.dumps(self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-m", "ldm_project.eval_reports", "compare", str(base / "run_cli"), "--baseline", str(baseline_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}")
            data = json.loads(proc.stdout)
            self.assertTrue(data["passed"])

    def test_cli_compare_fails_and_exit_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(
                base,
                "run_cli",
                {"mode": "one_shot_distill_eval"},
                {"overall": {"paired_corr": 0.60, "paired_mse": 0.050, "paired_dice": 0.50, "student_saturation_ratio": 0.30}},
                None,
            )

            baseline_path = base / "baseline.json"
            baseline_path.write_text(json.dumps(self._baseline(
                {"eval_mse": 0.020, "eval_corr": 0.807, "eval_dice": 0.756, "eval_sat": 0.091},
            )), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-m", "ldm_project.eval_reports", "compare", str(base / "run_cli"), "--baseline", str(baseline_path)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            data = json.loads(proc.stdout)
            self.assertFalse(data["passed"])

    def test_cli_compare_missing_baseline_file_exits_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(base, "run_cli", {"mode": "one_shot_distill_eval"}, None, None)

            proc = subprocess.run(
                [sys.executable, "-m", "ldm_project.eval_reports", "compare", str(base / "run_cli"), "--baseline", str(base / "nonexistent.json")],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("cannot read baseline", proc.stdout)


if __name__ == "__main__":
    unittest.main()
