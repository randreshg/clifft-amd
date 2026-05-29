#!/usr/bin/env python3
"""compare_oracle.py, P3 HIP-vs-CPU statistical comparison for the clifft-amd campaign.

Two jobs, one tool:

  1. Correctness (default). Compare a HIP sampler aggregate JSON against a CPU
     `clifft` oracle aggregate JSON. Each rate is a binomial proportion; the
     script asserts |p_hip - p_cpu| < cross_binomial_tolerance(p_pooled, N, sigma)
     for every metric, using the project's own primitive
     (third_party/clifft/tests/python/conftest.py:19-56):

         binomial_tolerance(p, n, sigma)       = sigma * sqrt(p(1-p)/n)
         cross_binomial_tolerance(p, n, sigma) = sqrt(2) * binomial_tolerance(...)

     The JSON can be supplied as two existing files (--cpu / --hip, matching the
     `run_pair` helper in README.md (Verification)), or generated on the fly by
     pointing at a .stim circuit (--circuit), in which case this script invokes
     the HIP runner twice: once with --cpu-reference and once without.

  2. Determinism (--determinism). Run the HIP sampler twice with the same seed
     and assert the seed-dependent JSON is byte-identical (timing fields are
     excluded, they are expected to vary). This is the P2 gate in script form.

Pure stdlib. No numpy, no third-party deps. Prints a per-metric table and a
final PASS / FAIL line with the numbers. Exit code 0 on PASS, 1 on FAIL,
2 on usage / runtime error.

Examples
--------
  # Compare two pre-generated JSON files (the docs run_pair flow):
  scripts/compare_oracle.py \
      --cpu d5_all_cpu.json \
      --hip d5_all_hip.json \
      --sigma 5

  # Generate both from a circuit, then compare (fixed seed 42):
  scripts/compare_oracle.py \
      --circuit "circuit_d5_p=0.001.stim" --shots 1000000 \
      --postselection all --seed 42 --sigma 5

  # Determinism: same seed twice must be byte-identical:
  scripts/compare_oracle.py --determinism \
      --circuit "circuit_d5_p=0.001.stim" --shots 1000000 \
      --seed 42 --postselection all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Tolerance primitives, kept byte-for-byte faithful to the project's own
# conftest.py so the harness and the pytest suite agree on the band. numpy is
# replaced by math; the arithmetic is identical.
# ---------------------------------------------------------------------------

EPS_DETERMINISTIC = 1e-12


def binomial_tolerance(p: float, n: int, sigma: float = 5.0) -> float:
    """sigma standard errors of a binomial proportion estimate.

    Returns a tiny epsilon for deterministic probabilities (p == 0 or p == 1)
    so exact matches still pass a strict-less-than comparison.
    """
    if n <= 0:
        return EPS_DETERMINISTIC
    if p == 0.0 or p == 1.0:
        return EPS_DETERMINISTIC
    std_err = math.sqrt((p * (1.0 - p)) / n)
    return sigma * std_err


def cross_binomial_tolerance(p: float, n: int, sigma: float = 5.0) -> float:
    """Tolerance for the difference of two independent binomial proportions."""
    return math.sqrt(2.0) * binomial_tolerance(p, n, sigma=sigma)


# ---------------------------------------------------------------------------
# Logging, house style: bracketed, on stderr so stdout
# stays parseable.
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[compare_oracle] {msg}", file=sys.stderr)


def die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[compare_oracle] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Default binary resolution. The task names build-core-hip/run_msc_hip; the docs
# use builds/hip/run_msc_hip. Prefer an explicit --binary / $RUN_MSC_HIP, then
# the task path, then the docs path. All paths are resolved relative to the repo
# root (the parent of this script's directory), never a hardcoded $HOME.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BINARY_CANDIDATES = (
    os.path.join(REPO_ROOT, "build-core-hip", "run_msc_hip"),
    os.path.join(REPO_ROOT, "builds", "hip", "run_msc_hip"),
)


def resolve_binary(explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            die(f"--binary not found: {explicit}")
        return explicit
    env = os.environ.get("RUN_MSC_HIP")
    if env:
        if not os.path.isfile(env):
            die(f"$RUN_MSC_HIP not found: {env}")
        return env
    for cand in DEFAULT_BINARY_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    die(
        "could not locate run_msc_hip. Tried:\n  "
        + "\n  ".join(DEFAULT_BINARY_CANDIDATES)
        + "\nPass --binary PATH or set $RUN_MSC_HIP."
    )


# ---------------------------------------------------------------------------
# Running the sampler. Returns the raw stdout (the JSON document) as text so the
# determinism mode can diff it byte-for-byte before any parsing.
# ---------------------------------------------------------------------------

def run_sampler(
    binary: str,
    circuit: str,
    shots: int,
    seed: int,
    postselection: str,
    cpu_reference: bool,
    block_size: int | None,
    dry_run: bool,
) -> str:
    cmd = [
        binary,
        "--circuit", circuit,
        "--shots", str(shots),
        "--seed", str(seed),
        "--postselection", postselection,
    ]
    if cpu_reference:
        cmd.append("--cpu-reference")
    if block_size is not None:
        cmd += ["--block-size", str(block_size)]

    printable = " ".join(_shell_quote(c) for c in cmd)
    log(f"run: {printable}")
    if dry_run:
        return ""

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # binary not executable, etc.
        die(f"failed to execute {binary}: {exc}", code=2)

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        die(f"{os.path.basename(binary)} exited {proc.returncode}", code=2)
    if not proc.stdout.strip():
        sys.stderr.write(proc.stderr)
        die("sampler produced empty stdout (expected JSON)", code=2)
    return proc.stdout


def _shell_quote(s: str) -> str:
    if s and all(c.isalnum() or c in "-._/=" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json_text(text: str, where: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        die(f"could not parse JSON from {where}: {exc}")


def load_json_file(path: str) -> dict:
    if not os.path.isfile(path):
        die(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return load_json_text(fh.read(), path)


# Fields whose values are seed-independent / count-derived and must agree for
# determinism; timing and throughput vary run-to-run and are excluded.
TIMING_KEYS = {
    "probe_compile_seconds",
    "postselection_compile_seconds",
    "kernel_seconds",
    "sample_seconds",
    "total_seconds",
    "shots_per_second_sampling_only",
    "shots_per_second_total",
}


def stable_subset(doc: dict) -> dict:
    """Drop timing keys; what remains is the seed-dependent count/rate payload."""
    return {k: v for k, v in doc.items() if k not in TIMING_KEYS}


# ---------------------------------------------------------------------------
# Metric extraction. Denominators per README.md (Verification):
#   survival_rate, discard_rate, error_rate_per_total_shot -> N = total shots
#   error_rate_per_survivor, per-observable rates           -> N = passed shots
# ---------------------------------------------------------------------------

def _req(doc: dict, key: str, where: str):
    if key not in doc:
        die(f"missing key '{key}' in {where}")
    return doc[key]


class Metric:
    __slots__ = ("name", "cpu_num", "cpu_den", "hip_num", "hip_den")

    def __init__(self, name, cpu_num, cpu_den, hip_num, hip_den):
        self.name = name
        self.cpu_num = int(cpu_num)
        self.cpu_den = int(cpu_den)
        self.hip_num = int(hip_num)
        self.hip_den = int(hip_den)

    @staticmethod
    def _prop(num: int, den: int) -> float:
        return (num / den) if den > 0 else 0.0

    @property
    def p_cpu(self) -> float:
        return self._prop(self.cpu_num, self.cpu_den)

    @property
    def p_hip(self) -> float:
        return self._prop(self.hip_num, self.hip_den)

    @property
    def p_pooled(self) -> float:
        num = self.cpu_num + self.hip_num
        den = self.cpu_den + self.hip_den
        return self._prop(num, den)

    @property
    def n_min(self) -> int:
        # Conservative: use the smaller per-sampler denominator for the band.
        return min(self.cpu_den, self.hip_den)

    def band(self, sigma: float) -> float:
        return cross_binomial_tolerance(self.p_pooled, self.n_min, sigma=sigma)

    @property
    def delta(self) -> float:
        return abs(self.p_hip - self.p_cpu)

    def passes(self, sigma: float) -> bool:
        return self.delta < self.band(sigma)


def build_metrics(cpu: dict, hip: dict) -> list[Metric]:
    cpu_total = int(_req(cpu, "shots", "cpu json"))
    hip_total = int(_req(hip, "shots", "hip json"))
    cpu_passed = int(_req(cpu, "passed_shots", "cpu json"))
    hip_passed = int(_req(hip, "passed_shots", "hip json"))
    cpu_disc = int(_req(cpu, "discarded_shots", "cpu json"))
    hip_disc = int(_req(hip, "discarded_shots", "hip json"))
    cpu_log = int(_req(cpu, "logical_errors", "cpu json"))
    hip_log = int(_req(hip, "logical_errors", "hip json"))
    cpu_obs = list(_req(cpu, "observable_ones", "cpu json"))
    hip_obs = list(_req(hip, "observable_ones", "hip json"))

    if len(cpu_obs) != len(hip_obs):
        die(
            f"observable_ones length mismatch: cpu={len(cpu_obs)} hip={len(hip_obs)} "
            "(different circuits compiled?)"
        )

    metrics: list[Metric] = [
        Metric("survival_rate", cpu_passed, cpu_total, hip_passed, hip_total),
        Metric("discard_rate", cpu_disc, cpu_total, hip_disc, hip_total),
        Metric("error_rate_per_total_shot", cpu_log, cpu_total, hip_log, hip_total),
        Metric("error_rate_per_survivor", cpu_log, cpu_passed, hip_log, hip_passed),
    ]
    for i, (c, h) in enumerate(zip(cpu_obs, hip_obs)):
        metrics.append(
            Metric(f"observable_ones[{i}]/passed", c, cpu_passed, h, hip_passed)
        )
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metric_table(metrics: list[Metric], sigma: float) -> bool:
    name_w = max(len(m.name) for m in metrics)
    name_w = max(name_w, len("metric"))
    header = (
        f"{'metric':<{name_w}}  {'p_cpu':>12}  {'p_hip':>12}  "
        f"{'|delta|':>12}  {'band(' + str(sigma) + 'σ)':>14}  verdict"
    )
    print(header)
    print("-" * len(header))
    all_pass = True
    for m in metrics:
        ok = m.passes(sigma)
        all_pass = all_pass and ok
        verdict = "PASS" if ok else "FAIL"
        print(
            f"{m.name:<{name_w}}  {m.p_cpu:>12.8f}  {m.p_hip:>12.8f}  "
            f"{m.delta:>12.3e}  {m.band(sigma):>14.3e}  {verdict}"
        )
    return all_pass


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_correctness(args) -> int:
    # Resolve the two JSON documents: either two files, or generated from circuit.
    if args.cpu and args.hip:
        log(f"comparing pre-generated JSON: cpu={args.cpu} hip={args.hip}")
        cpu_doc = load_json_file(args.cpu)
        hip_doc = load_json_file(args.hip)
    elif args.cpu or args.hip:
        die("--cpu and --hip must be given together (or use --circuit to generate both)")
    else:
        if not args.circuit:
            die("provide either --cpu/--hip JSON files, or --circuit to generate them")
        binary = resolve_binary(args.binary)
        log("generating CPU oracle JSON (--cpu-reference) ...")
        cpu_text = run_sampler(
            binary, args.circuit, args.shots, args.seed, args.postselection,
            cpu_reference=True, block_size=args.block_size, dry_run=args.dry_run,
        )
        log("generating HIP JSON ...")
        hip_text = run_sampler(
            binary, args.circuit, args.shots, args.seed, args.postselection,
            cpu_reference=False, block_size=args.block_size, dry_run=args.dry_run,
        )
        if args.dry_run:
            log("--dry-run: commands printed above; no comparison performed")
            return 0
        cpu_doc = load_json_text(cpu_text, "generated cpu run")
        hip_doc = load_json_text(hip_text, "generated hip run")

    # Sanity: the two runs must describe the same experiment configuration.
    for key in ("seed", "shots", "postselection"):
        if key in cpu_doc and key in hip_doc and cpu_doc[key] != hip_doc[key]:
            log(
                f"WARNING: {key} differs between cpu ({cpu_doc[key]}) and "
                f"hip ({hip_doc[key]}) JSON"
            )

    metrics = build_metrics(cpu_doc, hip_doc)

    print()
    print(
        f"clifft-amd P3 correctness: HIP vs CPU oracle  "
        f"(sigma={args.sigma}, cross-binomial band)"
    )
    cpu_shots = cpu_doc.get("shots", "?")
    hip_shots = hip_doc.get("shots", "?")
    print(
        f"  circuit={cpu_doc.get('circuit_path', '?')}  "
        f"seed={cpu_doc.get('seed', '?')}  "
        f"postselection={cpu_doc.get('postselection', '?')}"
    )
    print(
        f"  cpu_shots={cpu_shots}  hip_shots={hip_shots}  "
        f"cpu_backend={cpu_doc.get('backend', '?')}  "
        f"hip_backend={hip_doc.get('backend', '?')}"
    )
    print()
    all_pass = print_metric_table(metrics, args.sigma)
    print()

    n_metrics = len(metrics)
    n_fail = sum(0 if m.passes(args.sigma) else 1 for m in metrics)
    if all_pass:
        print(f"PASS: all {n_metrics} metrics within {args.sigma}σ cross-binomial band")
        return 0
    print(
        f"FAIL: {n_fail}/{n_metrics} metric(s) outside the {args.sigma}σ band "
        "(likely a real kernel bug, not Monte-Carlo noise)"
    )
    return 1


def mode_determinism(args) -> int:
    if not args.circuit:
        die("--determinism requires --circuit (the sampler is run twice)")
    binary = resolve_binary(args.binary)

    log(f"determinism: running {os.path.basename(binary)} twice with seed={args.seed}")
    text_a = run_sampler(
        binary, args.circuit, args.shots, args.seed, args.postselection,
        cpu_reference=False, block_size=args.block_size, dry_run=args.dry_run,
    )
    text_b = run_sampler(
        binary, args.circuit, args.shots, args.seed, args.postselection,
        cpu_reference=False, block_size=args.block_size, dry_run=args.dry_run,
    )
    if args.dry_run:
        log("--dry-run: commands printed above; no runs performed")
        return 0

    doc_a = load_json_text(text_a, "determinism run 1")
    doc_b = load_json_text(text_b, "determinism run 2")
    sub_a = stable_subset(doc_a)
    sub_b = stable_subset(doc_b)

    print()
    print(
        f"clifft-amd P2 determinism: same seed -> identical JSON  "
        f"(circuit={doc_a.get('circuit_path', '?')}, seed={args.seed}, "
        f"postselection={args.postselection})"
    )
    print(
        "  (timing/throughput keys excluded: "
        + ", ".join(sorted(TIMING_KEYS))
        + ")"
    )

    if sub_a == sub_b:
        # Report the load-bearing counts so the PASS is auditable.
        print(
            f"  passed_shots={sub_a.get('passed_shots')}  "
            f"discarded_shots={sub_a.get('discarded_shots')}  "
            f"logical_errors={sub_a.get('logical_errors')}  "
            f"observable_ones={sub_a.get('observable_ones')}"
        )
        print()
        print("PASS: seed-dependent JSON is identical across both runs")
        return 0

    # Enumerate exactly which keys diverged.
    print()
    print("  divergent keys:")
    keys = sorted(set(sub_a) | set(sub_b))
    for k in keys:
        va = sub_a.get(k, "<missing>")
        vb = sub_b.get(k, "<missing>")
        if va != vb:
            print(f"    {k}: run1={va!r}  run2={vb!r}")
    print()
    print("FAIL: seed-dependent JSON differs between runs (nondeterminism)")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_oracle.py",
        description=(
            "P3 HIP-vs-CPU statistical comparison (and P2 determinism check) for "
            "the clifft-amd survivor sampler."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Comparison inputs (file mode).
    p.add_argument("--cpu", help="path to CPU-oracle aggregate JSON")
    p.add_argument("--hip", help="path to HIP aggregate JSON")

    # Generation inputs (circuit mode / determinism mode).
    p.add_argument("--circuit", help="path to a .stim circuit (generate JSON by running the sampler)")
    p.add_argument("--shots", type=int, default=1_000_000, help="shot count (default 1e6)")
    p.add_argument("--seed", type=int, default=42, help="fixed RNG seed (default 42)")
    p.add_argument(
        "--postselection", choices=("all", "none"), default="all",
        help="postselection mode (default all)",
    )
    p.add_argument("--block-size", type=int, default=None, help="optional --block-size passthrough")

    # Binary resolution.
    p.add_argument(
        "--binary", default=None,
        help="path to run_msc_hip (default: build-core-hip/run_msc_hip, "
        "then builds/hip/run_msc_hip, then $RUN_MSC_HIP)",
    )

    # Tolerance / mode.
    p.add_argument("--sigma", type=float, default=5.0, help="cross-binomial sigma (default 5)")
    p.add_argument(
        "--determinism", action="store_true",
        help="run the sampler twice with the same seed and assert identical JSON",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the sampler commands that would run, then exit",
    )
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if args.shots <= 0:
        die("--shots must be positive")
    if args.sigma <= 0:
        die("--sigma must be positive")

    try:
        if args.determinism:
            return mode_determinism(args)
        return mode_correctness(args)
    except KeyboardInterrupt:
        die("interrupted", code=130)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
