# Selected Alpamayo inference

This package specializes the shared `openpilot.selfdrive.alpamayo.model.EdgeModel`.
This branch implements one selected recipe with four-bit weights in the serial language MLP. The shared default remains BF16 with ten Euler steps.
The initial shadow scheduler target is 500 ms. A 100 ms target belongs to the next phase.
Four-bit weights can change reasoning and trajectories relative to the FP8 branch. Live vehicle timing remains unverified.

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
| Language | Fixed 112 activation scales, packed FP8, 2176-row prefill, 2304-row KV allocation |
| Serial language MLP | Signed four-bit FC1 and FC2 weights with one FP16 scale per 32 weights; FP8 activations |
| Reasoning | Reduced CoC vocabulary; the prefill verifies up to 31 draft tokens with 32 independent heads |
| Continuation | Target-verified lookup carry, per-row FP8 draft verification, serial blocks of six, maximum 64 tokens |
| Expert | Three Euler steps; restored BF16 action projections; exact scoped 2x2 PV kernel |
| Compiler | WGP AMD LLVM22; isolated tinygrad `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e` |

Each prediction uploads uncached images in one pixel transfer. The GPU expands the byte patches to BF16 and encodes up to two images per call.
Ten distinct images need 5,529,600 bytes on startup. Four misses need 2,211,840 bytes. A full cache hit sends only metadata.
At a warmed 500 ms cadence, four missing images require two groups; six older images remain cached.
At a warmed 100 ms cadence, two missing images require one group; eight older images remain cached.
Actual reuse follows image content and the capture schedule, including missing or repeated source frames.

Prefill schedules long causal attention blocks first and adds the residual in the second MLP projection, preserving both BF16 rounding steps.
The prefill draft is the latest generated text, or the six-token nominal text before the first prediction. Draft tokens fill the padded rows
after the prompt. Head `i` holds the argmax after draft token `i-1`. The decoder accepts the matching prefix, whose KV rows the prefill wrote,
and continues from the first differing head. Accepted tokens have the same target-verified status as lookup verification.
A 2025-token prompt plus compact navigation text leaves room for all 32 heads; a prompt above 2145 tokens is rejected.
Padding rows follow the prompt, so real prompt rows and their KV are identical to a 2048-row prefill.
Expert attention adds the per-key mask inside the score kernel epilogue with the original BF16 and FP32 rounding.
Expert PV applies only inside trajectory execution.
During serial reasoning, 256 threads reduce the 9,216 expansion values to the same FP32 activation scale.
Startup creates the 56 four-bit matrices from the verified FP8 weights. They use 567 MiB of additional GPU memory; prompt processing still uses FP8 weights.
Draft attention retains its original QK, softmax, and PV arithmetic.
Serial decoding reads both KV planes directly after each write. Full blocks keep six tokens' cache and rotary coordinates on the GPU.
No runtime recipe switches or experimental installers are exposed.

### Graph cache

Captured graphs persist under `<cache>/graphs` after code generation and before command building.
Buffers are stored as owners: model attribute paths, JIT outputs, fresh scratch, or small constants.
Entries are keyed by graph name, every Alpamayo source file, the checkpoint digest, kernel manifest, compiler settings and calibration.
Entries from other keys are deleted at startup. An entry that fails to bind is recaptured normally.

## Required assets and shared code

Use the shared installer and its pinned dependency lock. Keep normal openpilot tinygrad unchanged.
NVIDIA code remains at `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e`.
For a car installation, prepare the BF16 tensors used by this recipe once:

```sh
python -m openpilot.selfdrive.alpamayo.selected.prepare_checkpoint /path/to/original /path/to/selected-checkpoint
```

Set `inference.checkpoint` to the resulting directory. It contains the tokenizer, configuration and a 102 MB BF16 shard.
Preparation verifies the original 5.30 GB of shards. Startup verifies the complete compact shard and index against independently pinned hashes.
The runtime records both the source checkpoint identity and the hashes of the files actually loaded.
The full original checkpoint is also supported. The default BF16 reference still requires the full checkpoint.
The separate FP8 `weights.bin`/`index.json`, fixed calibration, and per-upload verification remain required.
Changing these artifacts requires a separate numerical qualification.

The selected dispatcher uses 21 HIP source/object pairs and the expert PV object under `assets/gfx1200`.
Startup checks their hashes against `assets/manifest.json`.
HIP/COMGR compilation is unnecessary on ARM. Ordinary tinygrad LLVM kernels still require LLVM22.
The compiler pool uses at most one process per CPU available to the worker. Its SQLite cache persists across production launches.
The worker reports import, verification, frontend, GPU setup and weight-loading times during startup. Ready does not mean a driving plan is available.
The constructor requires existing, hash-checked gfx1200 firmware in the configured persistent cache.
Inference does not fetch firmware or weights.

Shared modules remain `model`, `amd_attention`, `inputs`, `geometry`, and `inference` identity helpers.
`camera.calibrated_image` provides the original portable C sampler using shared projection geometry.
Its caller supplies `cache_dir`; it compiles for the current CPU and does not include a host-specific library.
The selected live worker uses this sampler through the shared frontend.
CPU regression checks require exact agreement with the original projection and calibrated inputs.

Required notices are preserved in [NOTICE.md](NOTICE.md) and `licenses/`.
