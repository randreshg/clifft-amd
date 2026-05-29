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
        "backend": "cpu-reference-clifft-core",
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


# ---------------------------------------------------------------------------
# HIP / ROCm backend (AMD MI300X, gfx942, wavefront 64)
#
# Parallel to the CUDA path above. The native sampler lives in
# src/hip_sampler.hip and is exposed by the run_msc_hip C++ CLI; the Python
# layer mirrors the CUDA boundary so callers can select a backend uniformly.
# Nothing here alters or removes the CUDA path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RocmDiagnostics:
    rocminfo: bool
    rocm_smi: bool
    libamdhip64: str | None
    hipcc: str | None
    hip_header: Path | None
    gcn_arch: str | None

    @property
    def driver_available(self) -> bool:
        return (self.rocminfo or self.rocm_smi) and self.libamdhip64 is not None

    @property
    def toolkit_available(self) -> bool:
        return self.hipcc is not None and self.hip_header is not None

    @property
    def can_build_native_hip(self) -> bool:
        return self.driver_available and self.toolkit_available

    def missing_summary(self) -> str:
        missing: list[str] = []
        if not (self.rocminfo or self.rocm_smi):
            missing.append("rocminfo/rocm-smi (GPU driver visibility)")
        if self.libamdhip64 is None:
            missing.append("libamdhip64")
        if self.hipcc is None:
            missing.append("hipcc")
        if self.hip_header is None:
            missing.append("hip/hip_runtime.h")
        return ", ".join(missing) if missing else "none"


def _rocm_roots() -> list[str | None]:
    return [
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        os.environ.get("HIP_PATH"),
        os.environ.get("HIP_ROOT_DIR"),
        os.environ.get("CONDA_PREFIX"),
        "/opt/rocm",
        str(Path.cwd() / "rocm-env"),
    ]


def _find_rocm_command(cmd: str) -> str | None:
    found = shutil.which(cmd)
    if found:
        return found
    for root in _rocm_roots():
        if not root:
            continue
        exe = Path(root) / "bin" / cmd
        if exe.exists():
            return str(exe)
    return None


def _command_exists_path(path: str | None) -> bool:
    if path is None:
        return False
    try:
        subprocess.run([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        return False
    return True


def _find_hip_header() -> Path | None:
    candidates = [
        Path("/opt/rocm/include/hip/hip_runtime.h"),
        Path("/usr/include/hip/hip_runtime.h"),
    ]
    for root in _rocm_roots():
        if not root:
            continue
        base = Path(root)
        candidates.extend(
            [
                base / "include" / "hip" / "hip_runtime.h",
                base / "hip" / "include" / "hip" / "hip_runtime.h",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_rocm_library(name: str) -> str | None:
    found = ctypes.util.find_library(name)
    if found:
        return found
    patterns = [f"lib{name}.so", f"lib{name}.so.*"]
    for root in _rocm_roots():
        if not root:
            continue
        base = Path(root)
        dirs = [
            base / "lib",
            base / "lib64",
            base / "hip" / "lib",
        ]
        for directory in dirs:
            for pattern in patterns:
                matches = sorted(directory.glob(pattern))
                if matches:
                    return str(matches[0])
    return None


def _detect_gcn_arch() -> str | None:
    """Best-effort probe of the installed AMD GPU arch (e.g. gfx942)."""

    rocminfo = _find_rocm_command("rocminfo")
    if rocminfo is None:
        return None
    try:
        proc = subprocess.run(
            [rocminfo],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "gfx" in stripped and "Name:" in stripped:
            token = stripped.split("Name:", 1)[1].strip()
            if token.startswith("gfx"):
                return token
    return None


def check_rocm() -> RocmDiagnostics:
    rocminfo = _find_rocm_command("rocminfo")
    rocm_smi = _find_rocm_command("rocm-smi")
    return RocmDiagnostics(
        rocminfo=_command_exists_path(rocminfo),
        rocm_smi=_command_exists_path(rocm_smi),
        libamdhip64=_find_rocm_library("amdhip64"),
        hipcc=_find_rocm_command("hipcc"),
        hip_header=_find_hip_header(),
        gcn_arch=_detect_gcn_arch(),
    )


def sample_survivors_hip(_: CompiledWorkload) -> dict[str, Any]:
    """Placeholder for the native HIP/ROCm sampler boundary (MI300X / gfx942)."""

    diag = check_rocm()
    raise BackendUnavailable(
        "HIP sampler is not buildable/runnable in this workspace. "
        f"Missing: {diag.missing_summary()}. "
        "Install ROCm with hipcc and HIP headers plus libamdhip64, "
        "then build the native run_msc_hip target (cmake -DCLIFFT_AMD_ENABLE_HIP=ON)."
    )
