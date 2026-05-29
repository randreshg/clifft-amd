#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifft_cuda.backends import (
    BackendUnavailable,
    check_cuda,
    check_rocm,
    sample_survivors_cpu,
    sample_survivors_cuda,
    sample_survivors_hip,
)
from clifft_cuda.compiler import compile_for_survivors


def default_circuit() -> Path:
    return ROOT / "circuit_d5_p=0.001.stim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MSC with Clifft compile flow and selectable sampler.")
    parser.add_argument("--circuit", type=Path, default=default_circuit())
    parser.add_argument("--shots", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", choices=["hip", "cuda", "cpu"], default="hip")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print backend diagnostics for the selected backend and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.diagnose:
        diag = check_rocm() if args.backend == "hip" else check_cuda()
        print(json.dumps(diag.__dict__, indent=2, sort_keys=True, default=str))
        return 0

    total_start = time.perf_counter()
    workload = compile_for_survivors(
        args.circuit,
        shots=args.shots,
        seed=args.seed,
        root=REPO_ROOT,
        threads=args.threads,
    )

    try:
        if args.backend == "cuda":
            summary = sample_survivors_cuda(workload)
        elif args.backend == "hip":
            summary = sample_survivors_hip(workload)
        else:
            summary = sample_survivors_cpu(workload)
    except BackendUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary["total_seconds"] = time.perf_counter() - total_start
    summary["shots_per_second_total"] = (
        float(summary["shots"]) / summary["total_seconds"] if summary["total_seconds"] else float("nan")
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
