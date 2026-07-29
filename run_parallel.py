"""
Parallel benchmark launcher.

Splits models into groups and runs them concurrently in separate processes.
Each group gets its own log file and saves results independently.
Results can be analyzed together afterwards.

Usage:
    python3 run_parallel.py              # 2 groups (default)
    python3 run_parallel.py --groups 3   # 3 groups
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path


def split_models(config_path: Path, n_groups: int) -> list[list[str]]:
    """Split model IDs into n roughly equal groups."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_ids = [m["id"] for m in config["models"]]

    groups = [[] for _ in range(n_groups)]
    for i, mid in enumerate(model_ids):
        groups[i % n_groups].append(mid)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Parallel benchmark launcher")
    parser.add_argument("--groups", type=int, default=2, help="Number of parallel groups")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    groups = split_models(args.config, args.groups)

    print("=" * 60)
    print(f"PARALLEL BENCHMARK LAUNCHER — {args.groups} groups")
    print("=" * 60)
    for i, group in enumerate(groups):
        print(f"  Group {i+1}: {len(group)} models")
        for mid in group:
            name = mid.split("/")[-1][:40]
            print(f"    - {name}")
    print()

    # Launch each group as a separate process
    processes = []
    log_files = []
    layers_str = " ".join(str(l) for l in args.layers)

    for i, group in enumerate(groups):
        models_str = " ".join(f'"{mid}"' for mid in group)
        log_path = f"benchmark_log_group{i+1}.txt"
        log_files.append(log_path)

        cmd = (
            f"python3 -u -m src.runner "
            f"--config {args.config} "
            f"--layers {layers_str} "
            f"--models {models_str}"
        )

        print(f"Starting Group {i+1}: {log_path}")
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        processes.append((proc, log_fh, log_path))
        time.sleep(2)  # Stagger starts slightly

    print(f"\nAll {args.groups} groups launched.")
    print(f"Monitor with:")
    for i, (_, _, log_path) in enumerate(processes):
        print(f"  tail -f {log_path}   # Group {i+1}")
    print(f"\nOr check progress:")
    print(f'  grep "Judge health" benchmark_log_group*.txt')
    print(f'  grep "checkpoint\\|Results saved" benchmark_log_group*.txt')
    print()

    # Wait for all processes
    print("Waiting for all groups to complete...")
    for i, (proc, log_fh, log_path) in enumerate(processes):
        proc.wait()
        log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        print(f"  Group {i+1}: {status}")

    print()
    print("=" * 60)
    all_ok = all(p.returncode == 0 for p, _, _ in processes)
    if all_ok:
        print("ALL GROUPS COMPLETE — run analysis:")
        print("  python3 -m tests.test_construct_validity --results-dir results/")
        print("  python3 -m tests.test_sensitivity --results-dir results/")
        print("  streamlit run dashboard.py")
    else:
        failed = [i+1 for i, (p, _, _) in enumerate(processes) if p.returncode != 0]
        print(f"GROUPS {failed} FAILED — check their log files for errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
