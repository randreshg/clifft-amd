from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "CMakeLists.txt").exists() and (parent / "third_party" / "clifft").exists():
            return parent
    return Path.cwd()


def import_local_clifft(root: Path | None = None) -> Any:
    """Import the local Clifft wheel/site tree used by the benchmark workspace."""

    root = root or _find_repo_root()
    candidates = [
        root / "third_party" / "clifft" / "clifft-site",
        root / "clifft-site",
    ]
    for site in candidates:
        if site.exists():
            site_s = str(site)
            if site_s not in sys.path:
                sys.path.insert(0, site_s)
            break

    import clifft  # type: ignore[import-not-found]

    return clifft


@dataclass(frozen=True)
class CompiledWorkload:
    clifft: Any
    circuit_path: Path
    program: Any
    shots: int
    seed: int
    probe_compile_seconds: float
    postselection_compile_seconds: float


def compile_for_survivors(
    circuit_path: Path,
    *,
    shots: int,
    seed: int,
    root: Path | None = None,
    threads: int | None = None,
) -> CompiledWorkload:
    """Compile a circuit with full Clifft optimization and all-detector postselection."""

    clifft = import_local_clifft(root)
    if threads is not None:
        clifft.set_num_threads(int(threads))

    circuit_text = circuit_path.read_text()

    probe_start = time.perf_counter()
    probe = clifft.compile(
        circuit_text,
        normalize_syndromes=True,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    probe_compile_seconds = time.perf_counter() - probe_start

    compile_start = time.perf_counter()
    program = clifft.compile(
        circuit_text,
        normalize_syndromes=True,
        postselection_mask=[1] * probe.num_detectors,
        hir_passes=clifft.default_hir_pass_manager(),
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    postselection_compile_seconds = time.perf_counter() - compile_start

    return CompiledWorkload(
        clifft=clifft,
        circuit_path=circuit_path,
        program=program,
        shots=shots,
        seed=seed,
        probe_compile_seconds=probe_compile_seconds,
        postselection_compile_seconds=postselection_compile_seconds,
    )
