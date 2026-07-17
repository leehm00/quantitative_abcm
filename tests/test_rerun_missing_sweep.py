import tempfile
import unittest
from pathlib import Path
import sys

from scripts.rerun_missing_abcm_sweep import completed_run_names, missing_config_paths, run_command


class RerunMissingSweepTests(unittest.TestCase):
    def test_completed_run_names_requires_all_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "run_complete" / "20260705_010101"
            complete.mkdir(parents=True)
            for name in [
                "metrics_alpha.csv",
                "validation_metrics.csv",
                "prediction_accuracy.csv",
                "alpha_long_short.csv",
            ]:
                (complete / name).write_text("x\n")

            incomplete = root / "run_incomplete" / "20260705_010102"
            incomplete.mkdir(parents=True)
            (incomplete / "metrics_alpha.csv").write_text("x\n")

            self.assertEqual(completed_run_names(root), {"run_complete"})

    def test_missing_config_paths_excludes_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = root / "_configs"
            configs.mkdir()
            (configs / "run_complete.yaml").write_text("train:\n  output_dir: x\n")
            (configs / "run_missing.yaml").write_text("train:\n  output_dir: y\n")

            complete = root / "run_complete" / "20260705_010101"
            complete.mkdir(parents=True)
            for name in [
                "metrics_alpha.csv",
                "validation_metrics.csv",
                "prediction_accuracy.csv",
                "alpha_long_short.csv",
            ]:
                (complete / name).write_text("x\n")

            missing = missing_config_paths(root)

            self.assertEqual([path.name for path in missing], ["run_missing.yaml"])

    def test_run_command_failure_includes_stdout_and_stderr(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout marker'); print('stderr marker', file=sys.stderr); sys.exit(7)",
                ],
                Path.cwd(),
            )

        message = str(ctx.exception)
        self.assertIn("exit_status=7", message)
        self.assertIn("stdout marker", message)
        self.assertIn("stderr marker", message)


if __name__ == "__main__":
    unittest.main()
