# Selected Alpamayo inference

This package specializes the shared `openpilot.selfdrive.alpamayo.model.EdgeModel`.
It implements one fixed selected recipe. The shared default remains BF16 with ten Euler steps.
The initial shadow scheduler target is 500 ms. A 100 ms target belongs to the next phase.
Ten ARM/gfx1200 replay outputs exactly matched the experimental selected runtime. Complete live pipeline timing remains unverified.

## API

Use the isolated Alpamayo interpreter in a fresh worker process.
Import this package before importing tinygrad. The constructor sets the selected compiler environment.

```python
from pathlib import Path
from openpilot.selfdrive.alpamayo.selected import Config, Runtime

config = Config(
  checkpoint=Path('/path/to/checkpoint'),
  fp8_cache=Path('/path/to/fp8-cache'),
  prefill_calibration=Path('/path/to/prefill-scales.json'),
  cache_dir=Path('/path/to/persistent-cache'),
  initial_noise=Path('/path/to/initial_noise.npy'),
)
runtime = Runtime(config)
try:
  prediction = runtime.predict(sample)  # Or predict(sample, explicit_float32_noise).
  metrics = runtime.metrics
  runtime.synchronize()
finally:
  runtime.close()
```

`sample` contains the six canonical fields from the shared README.
Images use `(10,1,3,432,768)` uint8 calibrated RGB, with five camera-1 images followed by five camera-6 images.
All images retain 720 patches. Noise must be finite float32 with shape `(1,64,2)`.
Measured ego history is consumed unchanged. An empty navigation string means unavailable guidance.

`predict` returns all 64 positions, rotations, generated token IDs, reasoning, stage timings, and the supplied noise hash.
`metrics` separately records cache hits, misses, pixel upload bytes, and speculation outcomes.
`reset()` clears sample state, feature caches, decode history, and captured graphs while retaining model weights.
`close()` synchronizes and restores process-local backend overrides. Calls after close are rejected.
`manifest` records the exact numerical recipe and artifact identities; `device` identifies the selected backend.

`selected.ShadowAdapter(config)` implements the shared `runtime.run_prediction(adapter, arrays, metadata)` contract.
This adapter requires `initial_noise`. The shared worker, scheduler, frontend, and replay retain their existing responsibilities.

## Fixed arithmetic

| Component | Selected setting |
| --- | --- |
| Inputs | Two front cameras, 120/30 degrees, history `(-2,-1,-0.5,-0.1,0)` seconds |
| Vision | 144-image content cache; missing images encoded in groups of at most two |
| Vision precision | BF16 patch embedding and first two layers; dynamic per-image FP8 scales thereafter |
| Language | Fixed 112 activation scales, packed FP8, 2048-row prefill, 2304-row KV allocation |
| Reasoning | Reduced CoC vocabulary; six independently target-verified nominal tokens |
| Fallback | Target-verified lookup carry, per-row FP8 draft verification, serial blocks of six, maximum 64 tokens |
| Expert | Three Euler steps; restored BF16 action projections; exact scoped 2x2 PV kernel |
| Compiler | WGP AMD LLVM22; isolated tinygrad `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e` |

Each prediction uploads uncached images in one pixel transfer. The GPU expands the byte patches to BF16 and encodes up to two images per call.
Ten distinct images need 5,529,600 bytes on startup. Four misses need 2,211,840 bytes. A full cache hit sends only metadata.
At a warmed 500 ms cadence, four missing images require two groups; six older images remain cached.
At a warmed 100 ms cadence, two missing images require one group; eight older images remain cached.
Actual reuse follows image content and the capture schedule, including missing or repeated source frames.

Prefill schedules long causal attention blocks first and adds the residual in the second MLP projection, preserving both BF16 rounding steps.
Expert PV applies only inside trajectory execution.
During serial reasoning, 256 threads reduce the 9,216 expansion values to the same FP32 activation scale.
Draft attention retains its original QK, softmax, and PV arithmetic.
Serial decoding reads both KV planes directly after each write. Full blocks keep six tokens' cache and rotary coordinates on the GPU.
No runtime recipe switches or experimental installers are exposed.

## Required assets and shared code

Use the shared installer and its pinned dependency lock. Keep normal openpilot tinygrad unchanged.
NVIDIA code remains at `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e`.
Original checkpoint shards, the existing FP8 `weights.bin`/`index.json`, and fixed calibration remain external files.
`assets/artifacts.json` pins their hashes. Startup verifies complete checkpoint shards and every FP8 upload.
Changing these artifacts requires a separate numerical qualification.

The selected dispatcher uses the 17 HIP source/object pairs and the expert PV object under `assets/gfx1200`.
Startup checks their hashes against `assets/manifest.json`.
HIP/COMGR compilation is unnecessary on ARM. Ordinary tinygrad LLVM kernels still require LLVM22.
The constructor requires existing, hash-checked gfx1200 firmware in the configured persistent cache.
Inference does not fetch firmware or weights.

Shared modules remain `model`, `amd_attention`, `inputs`, `geometry`, and `inference` identity helpers.
`camera.calibrated_image` provides the original portable C sampler using shared projection geometry.
Its caller supplies `cache_dir`; it compiles for the current CPU and does not include a host-specific library.
The selected live worker uses this sampler through the shared frontend.
CPU regression checks require exact agreement with the original projection and calibrated inputs.

Required notices are preserved in [NOTICE.md](NOTICE.md) and `licenses/`.
