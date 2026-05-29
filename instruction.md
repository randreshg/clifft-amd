# clifft-cuda Native Sampler Instructions

This project can be built and run from inside the `clifft-cuda` directory. The
native build uses vendored C++ sources under `third_party/`, so it does not need
an external Clifft checkout.

## Requirements

- Linux shell environment.
- CMake 3.24 or newer.
- A CUDA-capable NVIDIA GPU.
- CUDA toolkit available through `CUDA_HOME`.
- The circuit file you want to sample, for example
  `./circuit_d5_p=0.001.stim`.

## Directory Layout

- `third_party/clifft`: vendored Clifft C++ source.
- `third_party/stim`: vendored Stim source used by Clifft.
- `third_party/fast_float`: vendored fast_float source used by Clifft.
- `src/` and `include/`: native CUDA sampler implementation.
- `tools/run_msc_cuda.cpp`: command-line runner.
- `build-core-cuda/`: generated native build output.

## Build

Run from `clifft-cuda`:

```bash
export CUDA_HOME="${CUDA_HOME:-./cuda-env}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
export CC="${CC:-gcc-13}"
export CXX="${CXX:-g++-13}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-g++-13}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cmake -S . \
  -B build-core-cuda \
  -DCLIFFT_CUDA_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
  -DCLIFFT_SOURCE_DIR=./third_party/clifft \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120-real

cmake --build build-core-cuda -j
```

Change `CMAKE_CUDA_ARCHITECTURES` if your GPU is not `sm_120`. The compiler
overrides avoid known nvcc issues with newer default GCC toolchains; if your
system default compiler is supported by nvcc, they can be omitted.

## Check CUDA

```bash
./build-core-cuda/run_msc_cuda --diagnose
```

Expected output lists at least one CUDA device.

## Build for AMD GPUs (ROCm / HIP)

The HIP backend needs ROCm (tested with 7.2.0) and `hipcc` on `PATH`. Build it
with the `CLIFFT_AMD_ENABLE_HIP` option, setting the GPU arch for your card
(`gfx942` for MI300X):

```bash
cmake -S . \
  -B build-core-hip \
  -DCLIFFT_AMD_ENABLE_HIP=ON \
  -DCLIFFT_SOURCE_DIR=./third_party/clifft \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx942

cmake --build build-core-hip -j
```

This builds `build-core-hip/run_msc_hip`, which takes the same flags and emits
the same JSON as `run_msc_cuda`. Check the device with
`./build-core-hip/run_msc_hip --diagnose`; the output lists at least one
`gfx942` device. The CUDA build is independent, so either, both, or neither
backend can be enabled.

## Run Sampling

Example d=5, p=0.001 run:

```bash
./build-core-cuda/run_msc_cuda \
  --circuit ./circuit_d5_p=0.001.stim \
  --shots 10000000 \
  --seed 42 \
  --block-size 256 \
  --postselection all
```

Use `--postselection none` to keep every shot and avoid detector-based discard.

## Output

The runner prints JSON to stdout. Useful fields:

- `peak_rank`: maximum active low-rank state dimension reached by the compiled
  circuit.
- `passed_shots`: shots that survived postselection.
- `discarded_shots`: total shots minus passed shots.
- `logical_errors`: surviving shots with a nonzero logical observable.
- `observable_ones`: per-observable one counts.
- `discard_rate`: `discarded_shots / shots`.
- `survival_rate`: `passed_shots / shots`.
- `error_rate_per_survivor`: `logical_errors / passed_shots`.
- `shots_per_second_sampling_only`: throughput excluding compile/probe time.
- `shots_per_second_total`: throughput including compile/probe time.

## Saving Logs

```bash
mkdir -p runs

./build-core-cuda/run_msc_cuda \
  --circuit ./circuit_d5_p=0.001.stim \
  --shots 10000000 \
  --seed 42 \
  --block-size 256 \
  --postselection all \
  > runs/d5_p0.001_shots10000000_seed42_cuda.stdout.json \
  2> runs/d5_p0.001_shots10000000_seed42_cuda.stderr.log
```
