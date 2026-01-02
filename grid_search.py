from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import time


def parse_powers(values: str) -> list[int]:
    items = []
    for part in values.split(","):
        part = part.strip()
        if not part:
            continue
        if "^" in part:
            base, exp = part.split("^", 1)
            items.append(int(int(base) ** int(exp)))
        else:
            items.append(int(part))
    return items


def parse_floats(values: str) -> list[float]:
    items = []
    for part in values.split(","):
        part = part.strip()
        if not part:
            continue
        if "^" in part:
            base, exp = part.split("^", 1)
            items.append(float(float(base) ** float(exp)))
        else:
            items.append(float(part))
    return items


def run_one(cmd: list[str]) -> dict[str, float | int | str]:
    start = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    stdout = proc.stdout.strip().splitlines()
    stderr = proc.stderr.strip().splitlines()

    last_test_acc = None
    for line in stdout:
        if "test_acc=" in line:
            for token in line.split():
                if token.startswith("test_acc="):
                    last_test_acc = float(token.split("=", 1)[1])

    time_to_90 = None
    for line in stdout[::-1]:
        if line.startswith("time_to_90pct="):
            value = line.split("=", 1)[1].strip()
            if value.startswith("not_reached"):
                time_to_90 = math.inf
            else:
                time_to_90 = float(value.rstrip("s"))
            break

    return {
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "last_test_acc": last_test_acc if last_test_acc is not None else float("nan"),
        "time_to_90_s": time_to_90 if time_to_90 is not None else float("nan"),
        "stdout": "\n".join(stdout[-5:]),
        "stderr": "\n".join(stderr[-5:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for EGGROLL hyperparameters.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--population-powers", type=str, default="2^5,2^6,2^7,2^8,2^9,2^10,2^11,2^12,2^13,2^14")
    parser.add_argument("--population-batch-powers", type=str, default="2^8,2^9,2^10,2^11,2^12")
    parser.add_argument("--group-size-powers", type=str, default="2^5,2^6,2^7")
    parser.add_argument("--sigma-powers", type=str, default="10^1,10^0,10^-1,10^-2")
    parser.add_argument("--es-lr-powers", type=str, default="10^0,10^-1,10^-2")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--output", type=str, default="eggroll_grid.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    populations = parse_powers(args.population_powers)
    pop_batches = parse_powers(args.population_batch_powers)
    group_sizes = parse_powers(args.group_size_powers)
    sigmas = parse_floats(args.sigma_powers)
    es_lrs = parse_floats(args.es_lr_powers)

    rows = []
    run_count = 0
    for pop in populations:
        for group_size in group_sizes:
            if pop % group_size != 0 or group_size % 2 != 0 or pop % 2 != 0:
                continue
            for pop_batch in pop_batches:
                if pop_batch > pop:
                    continue
                if pop_batch % group_size != 0 or pop_batch % 2 != 0:
                    continue
                for sigma in sigmas:
                    for es_lr in es_lrs:
                        cmd = [
                            "uv",
                            "run",
                            "python",
                            "main.py",
                            "--optimizer",
                            "eggroll",
                            "--epochs",
                            str(args.epochs),
                            "--population",
                            str(pop),
                            "--population-batch",
                            str(pop_batch),
                            "--group-size",
                            str(group_size),
                            "--sigma",
                            str(sigma),
                            "--es-lr",
                            str(es_lr),
                        ]
                        if args.dry_run:
                            print(" ".join(shlex.quote(c) for c in cmd))
                            continue
                        print(
                            f"run={run_count + 1} "
                            f"pop={pop} pop_batch={pop_batch} group_size={group_size} "
                            f"sigma={sigma} es_lr={es_lr}"
                        )
                        result = run_one(cmd)
                        print(
                            f"done={run_count + 1} "
                            f"elapsed_s={result['elapsed_s']:.2f} "
                            f"test_acc={result['last_test_acc']:.4f} "
                            f"time_to_90_s={result['time_to_90_s']}"
                        )
                        rows.append(
                            {
                                "population": pop,
                                "population_batch": pop_batch,
                                "group_size": group_size,
                                "sigma": sigma,
                                "es_lr": es_lr,
                                "epochs": args.epochs,
                                **result,
                            }
                        )
                        run_count += 1
                        if args.max_runs and run_count >= args.max_runs:
                            break
                    if args.max_runs and run_count >= args.max_runs:
                        break
                if args.max_runs and run_count >= args.max_runs:
                    break
            if args.max_runs and run_count >= args.max_runs:
                break
        if args.max_runs and run_count >= args.max_runs:
            break

    if args.dry_run:
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
