from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_EVALUATION_FILES = {
    "metrics_alpha.csv",
    "validation_metrics.csv",
    "prediction_accuracy.csv",
    "alpha_long_short.csv",
}


def completed_run_names(sweep_root: str | Path) -> set[str]:
    root = Path(sweep_root)
    completed: set[str] = set()
    for metrics_path in root.glob("*/20*/metrics_alpha.csv"):
        run_dir = metrics_path.parent
        if all((run_dir / name).exists() for name in REQUIRED_EVALUATION_FILES):
            completed.add(run_dir.parent.name)
    return completed


def missing_config_paths(sweep_root: str | Path, names: set[str] | None = None) -> list[Path]:
    root = Path(sweep_root)
    completed = completed_run_names(root)
    configs = sorted((root / "_configs").glob("*.yaml"))
    missing = [path for path in configs if path.stem not in completed]
    if names is not None:
        missing = [path for path in missing if path.stem in names]
    return missing


def read_names_file(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    names = {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return names


def parse_run_dir(output: str) -> Path:
    match = re.search(r"run_dir=(.+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse run_dir from train output:\n{output[-2000:]}")
    return Path(match.group(1).strip())


def run_command(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed "
            f"exit_status={completed.returncode}\n"
            f"cmd={' '.join(cmd)}\n\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return output


def rerun_config(config_path: Path, device: str | None, sweep_root: Path) -> tuple[str, str, Path | None]:
    name = config_path.stem
    log_dir = sweep_root / "_rerun_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    if name in completed_run_names(sweep_root):
        log_path.write_text(f"skipped completed run {name}\n")
        return name, "skipped", None

    train_cmd = [sys.executable, "scripts/train_abcm1.py", "--config", str(config_path)]
    if device:
        train_cmd.extend(["--device", device])
    train_output = ""
    eval_output = ""
    try:
        train_output = run_command(train_cmd, ROOT)
        run_dir = parse_run_dir(train_output)
        eval_output = run_command(
            [sys.executable, "scripts/evaluate_abcm1.py", "--factors-csv", str(run_dir / "factors.csv")],
            ROOT,
        )
    except Exception as exc:
        log_path.write_text(
            "train_cmd=" + " ".join(train_cmd) + "\n\n" + train_output + "\n" + eval_output + f"\nERROR: {exc}\n"
        )
        return name, "failed", None

    log_path.write_text("train_cmd=" + " ".join(train_cmd) + "\n\n" + train_output + "\n" + eval_output)
    return name, "completed", run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun missing ABCM sweep configs without stopping on one failure.")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--names-file", default=None)
    parser.add_argument("--devices", default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    names = read_names_file(args.names_file)
    configs = missing_config_paths(sweep_root, names=names)
    print(f"missing={len(configs)}")
    for path in configs:
        print(path.stem)
    if args.dry_run or not configs:
        return 0

    devices = [item.strip() for item in (args.devices or "").split(",") if item.strip()]
    parallel = max(1, min(int(args.parallel), len(configs)))
    failures = 0
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_config = {}
        for idx, config_path in enumerate(configs):
            device = devices[idx % len(devices)] if devices else None
            future = executor.submit(rerun_config, config_path, device, sweep_root)
            future_to_config[future] = config_path
        for future in as_completed(future_to_config):
            name, status, run_dir = future.result()
            print(f"{status} {name}" + (f" run_dir={run_dir}" if run_dir else ""))
            if status == "failed":
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
