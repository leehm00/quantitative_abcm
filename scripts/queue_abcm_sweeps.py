from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rerun_missing_abcm_sweep import missing_config_paths, rerun_config, run_command


@dataclass(frozen=True)
class QueueItem:
    sweep_root: Path
    config_path: Path

    @property
    def config_name(self) -> str:
        return self.config_path.stem


@dataclass(frozen=True)
class QueueResult:
    sweep_root: Path
    config_name: str
    status: str
    run_dir: Path | None
    device: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def collect_missing_queue(sweep_roots: list[Path]) -> list[QueueItem]:
    items: list[QueueItem] = []
    for sweep_root in sweep_roots:
        for config_path in missing_config_paths(sweep_root):
            items.append(QueueItem(sweep_root=sweep_root, config_path=config_path))
    return items


def assign_devices(items: list[QueueItem], devices: list[str]) -> list[tuple[QueueItem, str | None]]:
    assigned: list[tuple[QueueItem, str | None]] = []
    for idx, item in enumerate(items):
        device = devices[idx % len(devices)] if devices else None
        assigned.append((item, device))
    return assigned


def write_manifest(path: Path, items: list[QueueItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sweep_root", "config_name", "config_path"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "sweep_root": str(item.sweep_root),
                    "config_name": item.config_name,
                    "config_path": str(item.config_path),
                }
            )


def tmux_session_exists(session: str) -> bool:
    completed = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return completed.returncode == 0


def wait_for_sessions(sessions: list[str], interval_seconds: int) -> None:
    if not sessions:
        return
    interval_seconds = max(60, int(interval_seconds))
    while True:
        active = [session for session in sessions if tmux_session_exists(session)]
        if not active:
            print(f"{utc_now()} all wait sessions are done or absent", flush=True)
            return
        print(f"{utc_now()} waiting sessions={','.join(active)} next_check_seconds={interval_seconds}", flush=True)
        time.sleep(interval_seconds)


def run_item(item: QueueItem, device: str | None) -> QueueResult:
    name, status, run_dir = rerun_config(item.config_path, device, item.sweep_root)
    return QueueResult(
        sweep_root=item.sweep_root,
        config_name=name,
        status=status,
        run_dir=run_dir,
        device=device,
    )


def run_queue(items: list[QueueItem], devices: list[str], parallel: int) -> list[QueueResult]:
    assigned = assign_devices(items, devices)
    if not assigned:
        return []
    parallel = max(1, min(int(parallel), len(assigned)))
    results: list[QueueResult] = []
    print(f"{utc_now()} queue_start items={len(assigned)} parallel={parallel} devices={','.join(devices) or 'config_default'}", flush=True)
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_item = {
            executor.submit(run_item, item, device): (item, device)
            for item, device in assigned
        }
        for idx, future in enumerate(as_completed(future_to_item), start=1):
            result = future.result()
            results.append(result)
            run_dir = f" run_dir={result.run_dir}" if result.run_dir else ""
            print(
                f"{utc_now()} {idx}/{len(assigned)} {result.status} "
                f"root={result.sweep_root.name} config={result.config_name} device={result.device or ''}{run_dir}",
                flush=True,
            )
    return results


def summarize_roots(sweep_roots: list[Path]) -> None:
    for sweep_root in sweep_roots:
        output_csv = sweep_root / "partial_completed_return_summary.csv"
        print(f"{utc_now()} summarize root={sweep_root}", flush=True)
        output = run_command(
            [
                sys.executable,
                "scripts/summarize_abcm_sweep.py",
                "--sweep-root",
                str(sweep_root),
                "--output-csv",
                str(output_csv),
            ],
            ROOT,
        )
        print(output, flush=True)


def compare_roots(sweep_roots: list[Path], output_dir: Path) -> None:
    cmd = [sys.executable, "scripts/compare_abcm_candidates.py", "--output-dir", str(output_dir)]
    for sweep_root in sweep_roots:
        cmd.extend(["--search-root", str(sweep_root)])
    print(f"{utc_now()} compare output_dir={output_dir}", flush=True)
    output = run_command(cmd, ROOT)
    print(output, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue missing ABCM configs across multiple sweep roots.")
    parser.add_argument("--sweep-root", action="append", required=True, help="Sweep root. Repeat for multiple roots.")
    parser.add_argument("--devices", default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--wait-session", action="append", default=[], help="tmux session to wait for before scanning.")
    parser.add_argument("--wait-interval-seconds", type=int, default=3600)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--compare-output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_roots = [Path(path) for path in args.sweep_root]
    devices = parse_csv_list(args.devices)

    wait_for_sessions(args.wait_session, args.wait_interval_seconds)

    items = collect_missing_queue(sweep_roots)
    print(f"{utc_now()} missing_total={len(items)}", flush=True)
    for sweep_root in sweep_roots:
        count = sum(1 for item in items if item.sweep_root == sweep_root)
        print(f"{utc_now()} missing root={sweep_root} count={count}", flush=True)

    if args.manifest_csv:
        write_manifest(Path(args.manifest_csv), items)
        print(f"{utc_now()} manifest_csv={args.manifest_csv}", flush=True)

    if args.dry_run:
        return 0

    results = run_queue(items, devices, args.parallel)
    failures = sum(1 for result in results if result.status == "failed")
    print(f"{utc_now()} queue_done completed={sum(1 for r in results if r.status == 'completed')} skipped={sum(1 for r in results if r.status == 'skipped')} failed={failures}", flush=True)

    if args.summarize:
        summarize_roots(sweep_roots)

    if args.compare_output_dir:
        compare_roots(sweep_roots, Path(args.compare_output_dir))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
