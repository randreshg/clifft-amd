from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compiler import CompiledWorkload


class BackendUnavailable(RuntimeError):
    """Raised when a requested sampler backend cannot run in this environment."""


@dataclass(frozen=True)
class CudaDiagnostics:
    nvidia_smi: bool
    libcuda: str | None
    nvcc: str | None
    nvrtc: str | None
    cuda_header: Path | None

    @property
    def driver_available(self) -> bool:
        return self.nvidia_smi and self.libcuda is not None

    @property
    def toolkit_available(self) -> bool:
        return self.nvcc is not None and self.cuda_header is not None

    @property
    def jit_available(self) -> bool:
        return self.nvrtc is not None and self.cuda_header is not None

    @property
    def can_build_native_cuda(self) -> bool:
        return self.driver_available and self.toolkit_available

    @property
    def can_runtime_jit_cuda(self) -> bool:
        return self.driver_available and self.jit_available

    def missing_summary(self) -> str:
        missing: list[str] = []
        if not self.nvidia_smi:
            missing.append("nvidia-smi/GPU driver visibility")
        if self.libcuda is None:
            missing.append("libcuda")
        if self.nvcc is None:
            missing.append("nvcc")
        if self.nvrtc is None:
            missing.append("libnvrtc")
        if self.cuda_header is None:
            missing.append("cuda.h")
        return ", ".join(missing) if missing else "none"


def _command_exists(cmd: str) -> bool:
    exe = shutil.which(cmd)
    if exe is None:
        return False
    try:
        subprocess.run([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        return False
    return True


def _find_cuda_header() -> Path | None:
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDAToolkit_ROOT"),
        os.environ.get("CONDA_PREFIX"),
        str(Path.cwd() / "cuda-env"),
    ]
    candidates = [
        Path("/usr/local/cuda/include/cuda.h"),
        Path("/opt/cuda/include/cuda.h"),
        Path("/usr/include/cuda.h"),
    ]
    for root in roots:
        if not root:
            continue
        base = Path(root)
        candidates.extend(
            [
                base / "include" / "cuda.h",
                base / "targets" / "x86_64-linux" / "include" / "cuda.h",
            ]
        )
    for path in candidates:
        if path.exists() and "linux/cuda.h" not in str(path):
            return path
    return None


def _find_library(name: str) -> str | None:
    found = ctypes.util.find_library(name)
    if found:
        return found
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDAToolkit_ROOT"),
        os.environ.get("CONDA_PREFIX"),
        str(Path.cwd() / "cuda-env"),
    ]
    patterns = [f"lib{name}.so", f"lib{name}.so.*"]
    for root in roots:
        if not root:
            continue
        base = Path(root)
        dirs = [
            base / "lib",
            base / "lib64",
            base / "targets" / "x86_64-linux" / "lib",
        ]
        for directory in dirs:
            for pattern in patterns:
                matches = sorted(directory.glob(pattern))
                if matches:
                    return str(matches[0])
    return None


def _find_nvcc() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDAToolkit_ROOT"),
        os.environ.get("CONDA_PREFIX"),
        str(Path.cwd() / "cuda-env"),
    ]
    for root in roots:
        if not root:
            continue
        nvcc = Path(root) / "bin" / "nvcc"
        if nvcc.exists():
            return str(nvcc)
    return None


def check_cuda() -> CudaDiagnostics:
    return CudaDiagnostics(
        nvidia_smi=_command_exists("nvidia-smi"),
        libcuda=_find_library("cuda"),
        nvcc=_find_nvcc(),
        nvrtc=_find_library("nvrtc"),
        cuda_header=_find_cuda_header(),
    )


def _rate(numerator: int, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def sample_survivors_cpu(workload: CompiledWorkload) -> dict[str, Any]:
    """Run the reference Clifft CPU survivor sampler."""

    program = workload.program
    clifft = workload.clifft

    sample_start = time.perf_counter()
    result = clifft.sample_survivors(program, workload.shots, seed=workload.seed, keep_records=False)
    sample_seconds = time.perf_counter() - sample_start

    total = int(result.total_shots)
    passed = int(result.passed_shots)
    discards = int(result.discards)
    logical_errors = int(result.logical_errors)

    return {
        "backend": "cpu-reference-clifft",
        "clifft_version": clifft.__version__,
        "svm_backend": clifft.svm_backend(),
        "circuit_path": str(workload.circuit_path),
        "shots": total,
        "seed": workload.seed,
        "threads": int(clifft.get_num_threads()),
        "hir_passes": "default_hir_pass_manager()",
        "bytecode_passes": "default_bytecode_pass_manager()",
        "normalize_syndromes": True,
        "postselection": "all detectors",
        "has_postselection": bool(program.has_postselection),
        "peak_rank": int(program.peak_rank),
        "detectors": int(program.num_detectors),
        "observables": int(program.num_observables),
        "measurements": int(program.num_measurements),
        "noise_sites": int(len(program.noise_site_probabilities)),
        "num_instructions": int(program.num_instructions),
        "passed_shots": passed,
        "discarded_shots": discards,
        "logical_errors": logical_errors,
        "observable_ones": [int(x) for x in result.observable_ones.tolist()],
        "discard_rate": _rate(discards, total),
        "survival_rate": _rate(passed, total),
        "error_rate_per_survivor": _rate(logical_errors, passed),
        "error_rate_per_total_shot": _rate(logical_errors, total),
        "probe_compile_seconds": workload.probe_compile_seconds,
        "postselection_compile_seconds": workload.postselection_compile_seconds,
        "sample_seconds": sample_seconds,
        "shots_per_second_sampling_only": _rate(total, sample_seconds),
    }


def sample_survivors_cuda(_: CompiledWorkload) -> dict[str, Any]:
    """Placeholder for the native GPU backend boundary."""

    diag = check_cuda()
    raise BackendUnavailable(
        "CUDA sampler is not buildable/runnable in this workspace. "
        f"Missing: {diag.missing_summary()}. "
        "Install a CUDA toolkit with nvcc and headers, or libnvrtc plus headers, "
        "then build the native clifft-cuda extension."
    )
