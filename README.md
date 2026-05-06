# clifft-cuda

`clifft-cuda` runs [Clifft](https://github.com/unitaryfoundation/clifft)-compiled Stim circuits with a native CUDA sampler.

It is implemented with the help of Codex 5.5.

For build/setup instructions, see [instruction.md](instruction.md).

## Performance

for distance-5 magic state cultivation circuits with noise level p=0.1%:

| Tool | processor | power | sampling speed | speed up |
|-----|-----|-----|-----|-----|
|Clifft|Xeon(R) Gold 5218R CPU @ 2.10GHz|125 W|~59,800 shots/s|-|
|Clifft-cuda|NVIDIA RTX PRO 5000|300 W|~1,180,000 shots/s|19.7|

## Run A Circuit

From the `clifft-cuda` directory:

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

Use `--postselection none` to keep every shot and run the full circuit without
detector-based discard.

## Output

Results are written wherever stdout/stderr are redirected. The recommended
location is `runs/`:

- `runs/*.stdout.json`: final aggregate sampling result.
- `runs/*.stderr.log`: progress messages and runtime errors.

The JSON includes fields such as `shots`, `seed`, `peak_rank`, `passed_shots`,
`discarded_shots`, `logical_errors`, `discard_rate`, `survival_rate`,
`kernel_seconds`, and throughput.

## Quick CUDA Check

```bash
./build-core-cuda/run_msc_cuda --diagnose
```
