# Alpamayo six-camera investigation

Saved on 2026-11-15 after stopping the experiments at the user's request.
The best measured throughput is **1.958 FPS** on Chestnut with RX 9060 at 144 W and automatic clocks.
The target of four complete predictions per second remains unmet.

This directory preserves the selected implementation, rejected experiments, shadow-recording helpers, and numerical checks.
The parent directory contains Harald's baseline from commaai/openpilot revision `fe68e1d0595d48ffd0d058404181d873250e4ae4`.
Experimental kernels remain optional. The selected switches are recorded in [best-config.json](best-config.json).
No Alpamayo output is connected to vehicle controls.

## Workload and accuracy

- Six distinct NVIDIA cameras: front wide, front tele, front left/right, and rear left/right.
- Real camera streams causally sampled near 10 Hz; prediction inputs advance every 250 ms.
- Full NVIDIA image budgets: front views 576 × 320, side views 416 × 224, from 1920 × 1080 sources.
- Each input contains 22 history images, with camera counts 3, 5, 3, 3, 3, 5.
- The measured replay contains 32 consecutive snapshots from 4.00 to 11.75 seconds; eight warm up the graphs and 24 are timed.
- Three Euler steps, greedy reasoning, one trajectory candidate, and a maximum of 256 generated tokens.
- Fixed navigation text is `Turn right in 30m`. This replay does not implement changing navigation instructions.

All 32 outputs were compared with the unmodified NVIDIA BF16 CPU implementation using identical images, motion, navigation, noise, and three Euler steps.
Twenty-six generated text sequences matched exactly. Mean trajectory-position difference was **0.108 m**; the maximum was **1.427 m**.
These are numerical comparisons on one overlapping clip. Driving accuracy remains unqualified.
The six differing texts include missing yield-sign wording in some frames.

The selected implementation restores the patch embedding, first two vision layers, and expert projections from the original BF16 checkpoint.
All 63,717,376 restored bytes were checked after upload. Other eligible matrices retain Harald's FP8 quantization.
The 2,211-token reasoning vocabulary is optional and restricts possible text outputs.
Its equivalence to the complete vocabulary requires the complete-head argmax to remain inside that set.

## Where time goes

The timing boundary includes raw RGB file loading, preprocessing, upload, inference, and CPU trajectory decoding.
It excludes camera capture, encoded-video decoding, weight loading, compilation, and graph warmup.
Cold startup still takes minutes.

| Measured stage | Mean time |
| --- | ---: |
| Language prefill of the image/history prompt | 206.7 ms |
| Vision encoder and projector | 153.4 ms |
| Reasoning-token generation | 67.6 ms |
| Trajectory diffusion, three steps | 48.7 ms |
| Input upload | 23.5 ms |
| CPU trajectory decoding | 1.4 ms |
| Total timed inference, including small stage overheads | **502.5 ms** |

These stage times include host submission and device waits.
CPU preprocessing averages another 130.6 ms, file loading 2.2 ms, and buffer preparation 1.4 ms.
Input preparation overlaps the preceding inference, so these values must not be added directly to the inference total.
Actual measured wall time was **12.257 seconds for 24 predictions**, or **510.7 ms per completed prediction**.
The prepared-input queue raises latency: median input-to-output latency was **934 ms**, with **1,273 ms p95**.
Throughput is measured count divided by wall time; it is not inverse queued latency.

See [results/best-run.json](results/best-run.json) for timing rows and [results/official-comparison.json](results/official-comparison.json) for numerical differences.
The fastest run omitted repeated vision-reference graphs. The preceding unchanged vision configuration passed four complete embedding comparisons.
All 32 token, position, and rotation outputs matched the preceding run exactly.
The total throughput gain cannot be assigned solely to the last packing change.

## Changes that helped

These measurements use different scopes. Individual gains must not be added as independent whole-model improvements.

| Change | Evidence and status |
| --- | --- |
| Cached CPU processing and exact image reuse | Reuses processors and unchanged history images; every source image byte is checked before reuse. Twelve new images remain per steady six-camera prediction. Selected. |
| Compact inputs, exact byte pixels, selective transfers | Preserves normalized pixel values and uploads only cache misses. Selected. |
| Batched vision with independent image scales | Reduces launch overhead while preserving each image's quantization. Partial-tile fallbacks retain individual-image rounding. Selected. |
| Packed projector matrices | Isolated four-front projector time fell from 15.61 to 3.11 ms; eight-side time fell from 13.83 to 3.15 ms. All tested values matched. Selected. |
| BF16 restoration and explicit BF16 vision matrices | Improved reference agreement. Explicit matrix boundaries reduced the restored vision stage from about 1,208 to 207 ms in the corresponding validation run. Selected. |
| Online attention and wider key/value reads | A full-prompt attention check improved from 5.31 to 3.85 ms with identical outputs. Selected. |
| Packed prefill values and identity-matrix probability transpose | All 64 attention checks matched. The complete replay retained identical token, position, and rotation outputs. Selected by shape. |
| Exact global reductions and direct packed activation quantization | One actual language layer improved from 8.34 to 7.88 ms, with identical output and all cache writes across changing inputs. Selected. |
| Target-verified draft decoding and token carry | Reuses previous text as proposals, checks each accepted token against the current target, and avoids redundant GPU reads. Selected. Batched arithmetic can differ from serial decoding. |
| Direct small-result USB readback | Passed changing-input and boundary checks; avoids repeated host graph construction for small outputs. Selected for small results. |
| Expert kernels and three Euler steps | Reduced diffusion work. The final comparison uses an official reference with the same three steps. Three steps are not equivalent to the original ten-step default. |
| Verified prequantized bulk loading | Loads prepared weights and checks every uploaded byte. This improves loading, outside steady throughput. |

The early two-camera tests exceeded four FPS with reduced image budgets.
Those measurements do not establish four FPS for this six-camera workload.

## Changes that did not qualify

| Experiment | Result |
| --- | --- |
| Uniform half/quarter image budgets | Rejected after large changes in turning trajectories. More Euler steps did not repair the lost image information. |
| Camera-specific budgets and front-camera crops | Some averages improved, but individual trajectories or signal descriptions regressed. Full budgets retained. |
| Per-token language quantization | About 1.728 FPS; worst reference-position difference increased to 5.105 m. Rejected. |
| FP8 attention | Full-prompt check slowed from 5.46 to 8.25 ms and increased relative error from about 0.2% to 4.7%. Rejected. |
| Forced high clocks | Actual-layer checks were slightly slower than automatic clocks. Automatic clocks restored. |
| Batched CPU image resizing | Exact pixels, but roughly three times slower. Individual resizing retained. |
| Dual compute queues | Fresh full-size checks passed without speed gain. Queue recreation previously hung and required a scoped reset. Excluded from the model runner. |
| Triton attention/GEMMs, split attention, shared tiles, workgroup ordering, unrolling | No useful general gain; some variants changed rounding. Diagnostic failures are retained. Defaults remain disabled. |
| Combined partial-image GEMM | All unit values and 32 model predictions matched. The validation run reached 1.813 FPS with extra reference graphs; comparable speed benefit remains unestablished. |
| AMD vendor GEMMs | Final 60 small changing-input checks passed. Larger matrices changed rounding, so the final exact-output check failed. No model integration. |

The final vendor batch tested five kernels on both language matrix shapes.
The best isolated times were **2.24 → 2.04 ms** and **2.31 → 2.04 ms**, approximately 9% and 12% improvements.
Each candidate changed 728 or 671 rounded output values, respectively.
FP64 checks found both better and worse rounded values; these kernels are not accepted exact replacements.
See [results/vendor-tuned-prefill.json](results/vendor-tuned-prefill.json) and [results/vendor-tuned-small.json](results/vendor-tuned-small.json).

## Reproduce the saved configuration

Use separate checkouts and an isolated Python environment. The repository's tinygrad submodule is not changed by this snapshot.
Obtain the model and dataset under their respective terms and authenticate Hugging Face separately.
Model weights, raw footage, compiled binaries, credentials, and private device addresses are not included.

| Component | Exact revision |
| --- | --- |
| Harald baseline | `fe68e1d0595d48ffd0d058404181d873250e4ae4` |
| commaai/tinygrad | `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e` |
| NVlabs/alpamayo2-edge | `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` |
| nvidia/Alpamayo2-Edge checkpoint | `b92b6aa326bd3e2235db28f248ad5a30b5890e64` |
| nvidia/PhysicalAI-Autonomous-Vehicles dataset | `14ed7a77897edafb113dc86fde8b8173ae7fb028` |

Install the pinned NVIDIA checkout with `uv sync --frozen --extra dataset --no-dev`.
The recorded environment uses Python 3.12, PyTorch 2.8.0, and physical-ai-av 0.2.0.
The measured GPU is gfx1200 with a 5 Gb/s Chestnut USB link and a verified 144 W stock limit.
Confirm the target device, power supply, and USB permissions before running hardware scripts.

Set `EXP` to this directory and `PYTHON` to the pinned NVIDIA environment's interpreter.
Set `MODEL`, `SAMPLES`, `REFERENCE`, `TINYGRAD`, and `FP8_CACHE` to explicit asset directories.

```bash
times=()
samples=()
for ((i=0; i<32; i++)); do
  time_us=$((4000000 + i*250000))
  times+=("$time_us")
  samples+=("$SAMPLES/sample-$time_us")
done
"$PYTHON" "$EXP/export-nvidia.py" --camera-hz 10 --times-us "${times[@]}" --output "$SAMPLES"
"$PYTHON" "$EXP/nvidia-reference.py" --checkpoint "$MODEL" --samples "${samples[@]}" \
  --pixel-scale 1 --steps 3 --output "$REFERENCE"
"$PYTHON" "$EXP/prepare-fp8.py" "$MODEL" "$FP8_CACHE"
"$PYTHON" "$EXP/replay-best.py" --checkpoint "$MODEL" --samples "$SAMPLES" \
  --noise "$REFERENCE/initial_noise.npy" --tinygrad "$TINYGRAD" --fp8-cache "$FP8_CACHE" \
  --cache /tmp/alpamayo-compile-cache.db --output /tmp/alpamayo-best.json
"$PYTHON" "$EXP/compare-nvidia.py" --native /tmp/alpamayo-best.json \
  --reference "$REFERENCE" --output /tmp/alpamayo-quality.json
```

The saved run used CPU affinity `0-7,16-23` on a Ryzen 7950X host.
Use `--print-command` to inspect the selected replay without creating a GPU context.
Use `--verify-vision-warmup` to repeat complete embedding checks; additional reference graphs affect comparison with the fastest saved run.
Initial diffusion noise must match exactly. Equal NumPy and PyTorch seeds do not produce identical noise.

The vocabulary file comes unchanged from NVlabs/alpamayo-recipes revision `622f9d23ff9e4961ba8c6aac71367776cdf6b016`.
Its path is `recipes/alpamayo2edge_edgellm/host/coc_vocab.json`; SHA256 is `a31437be0e5b900d3028ec8fb809bcb86e767ef1e703f727b052675fca9f18b0`.
See [NVIDIA-RECIPES-LICENSE](NVIDIA-RECIPES-LICENSE) for its Apache 2.0 license.

## Archived experiments and shadow helpers

`check-*.py` and `profile-*.py` preserve the numerical and profiling investigations, including rejected paths.
Many research scripts expect named fixtures under `/tmp/opencode`; they are saved investigation tools, not an installed product interface.
Do not run all hardware experiments as an automatic test suite.
Profiling invalidates throughput measurements.

`vendor-gemm.py` uses extracted hipBLASLt 7.0.2 code through the existing USB context.
The package is `hipblaslt_1.0.0.70002-56~24.04_amd64.deb` from AMD's official repository.
Its SHA256 is `82367f5b7ee8615c673faa117e2975991d7e075e5202146541bd210349124144`.
`extract-alpamayo-hipblaslt.py` checks that digest and extracts gfx1200 files without installing packages or executing maintainer scripts.
`inspect-alpamayo-vendor.py` reads AMDGPU metadata from an extracted ELF code object.
The archived selected solution metadata is in `results/vendor-selected.json` and `results/vendor-tuned-selected.json`.
The original CCOB/Zstandard and Clang offload bundle must be unpacked before the ELF inspector can read it.
Vendor code expects the prepared HSACO and metadata under `/tmp/opencode`; these binaries are intentionally external assets.
The descriptor alias preserves every loaded byte and relocation. A scoped argument adapter follows the 176-byte metadata layout.
Vendor and queue checks require explicit `--usb-serial-file` and `--usb-serial` arguments.

`run-shadow.py`, `shadow-capture.py`, and `shadow-worker.py` preserve the earlier two-camera shadow-recording work.
They save predictions and small-model observations for later replay and do not publish control commands.
They are separate from the six-camera indoor benchmark and have not received its latest GPU optimizations.
`DISABLE_CHESTNUT=1` short-circuits modeld's Chestnut selection before any probe, keeping the small model available for shadow sessions.
The environment must be set on the actual model supervisor before a shadow session.

Deployment helpers require explicit `ALPAMAYO_BUNDLE`, `ALPAMAYO_DEPLOY_TARGET`, and hexadecimal `ALPAMAYO_EXPECTED_SOC_SERIAL` values.
`run-shadow.py --serial` requires the exact target serial-file value.
`prepare-bundle.py` requires `ALPAMAYO_ASSET_ROOT` with the pinned checkouts and historical sample assets.
`run-parked.sh` stops openpilot during its parked benchmark and must only run after its fresh parked/disengaged checks pass.
No car deployment occurred while saving this snapshot.

Publication checks cover Python syntax, shell syntax, existing CPU shadow tests, and scoped changes.
The last workspace dependency audit reported 11 existing advisories: nine moderate and two low. None were suppressed.
