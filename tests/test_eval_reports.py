from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ldm_project.eval_reports import print_summary, summarize_runs


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
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._make_run_dir(base, "run_x", {"mode": "one_shot_distill_eval"}, None, None)

            captured = io.StringIO()
            old = sys.stdout
            sys.stdout = captured
            try:
                print_summary([str(base / "run_x")])
            finally:
                sys.stdout = old

            data = json.loads(captured.getvalue())
            self.assertIn("run_x", data)


if __name__ == "__main__":
    unittest.main()
