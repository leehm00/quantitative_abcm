import tempfile
import unittest
from pathlib import Path

from scripts.queue_abcm_sweeps import QueueItem, assign_devices, collect_missing_queue, write_manifest


def _write_complete_run(root: Path, name: str) -> None:
    run_dir = root / name / "20260705_010101"
    run_dir.mkdir(parents=True)
    for file_name in [
        "metrics_alpha.csv",
        "validation_metrics.csv",
        "prediction_accuracy.csv",
        "alpha_long_short.csv",
    ]:
        (run_dir / file_name).write_text("x\n")


class QueueAbcmSweepsTests(unittest.TestCase):
    def test_collect_missing_queue_across_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root_a = base / "sweep_a"
            root_b = base / "sweep_b"
            for root in [root_a, root_b]:
                (root / "_configs").mkdir(parents=True)
            (root_a / "_configs" / "done_a.yaml").write_text("x\n")
            (root_a / "_configs" / "missing_a.yaml").write_text("x\n")
            (root_b / "_configs" / "missing_b.yaml").write_text("x\n")
            _write_complete_run(root_a, "done_a")

            items = collect_missing_queue([root_a, root_b])

            self.assertEqual(
                [(item.sweep_root.name, item.config_name) for item in items],
                [("sweep_a", "missing_a"), ("sweep_b", "missing_b")],
            )

    def test_assign_devices_round_robin(self):
        items = [
            QueueItem(Path("/tmp/sweep"), Path(f"/tmp/sweep/_configs/run_{idx}.yaml"))
            for idx in range(5)
        ]

        assigned = assign_devices(items, ["cuda:0", "cuda:1"])

        self.assertEqual([device for _, device in assigned], ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"])

    def test_write_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            item = QueueItem(Path("/tmp/sweep"), Path("/tmp/sweep/_configs/run_a.yaml"))

            write_manifest(path, [item])

            text = path.read_text()
            self.assertIn("sweep_root,config_name,config_path", text)
            self.assertIn("/tmp/sweep,run_a,/tmp/sweep/_configs/run_a.yaml", text)


if __name__ == "__main__":
    unittest.main()
