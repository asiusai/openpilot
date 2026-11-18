# Alpamayo inference and planner evaluation

The small driving model runs by default. Shadow mode saves GPU predictions without changing driving actions.
Planner mode lets modeld select GPU actions at its regular cadence. This mode is under bench evaluation. Vehicle control remains unqualified.

Setting `ALPAMAYO_CONFIG` selects the 500 ms planner unless `ALPAMAYO_SHADOW=1` or `ALPAMAYO_PLANNER=0` is set.
The file must contain a selected-model configuration. With no configuration, the existing openpilot model selection applies.
Planner mode reserves Chestnut for Alpamayo and keeps the small model running for perception and immediate fallback.
The manager starts the GPU worker onroad when Chestnut enumerates at 5 Gbps and the device is healthy. Weights can load alongside modeld startup.
Input admission still requires a healthy local model. The first five completed predictions are startup samples.
Two consecutive timely plans are required before modeld can use a GPU action. Age the plan from camera capture before computing the requested action.
When eligible, Alpamayo supplies the complete model action. Its planned speed must agree with measured vehicle speed; curvature and acceleration must stay within physical limits.
Driver braking or stock emergency braking invalidates GPU selection. Normal openpilot engagement, vehicle-specific controls, radar and Panda remain authoritative downstream.
An absent, expired, invalid, or rejected plan leaves the current small-model action in place. An explicit worker invalidation clears the plan.
An already active planner can use one isolated overrun published less than one 20 Hz model frame (50 ms) late.
It still counts as a missed 500 ms deadline. The plan must be newer and fresh from capture.
Repeated overruns, stale delivery or a missed slot return control to the small model. Two consecutive on-time plans are required to resume.
The scheduler briefly waits for a nearly finished worker, then admits only current input or skips the slot. A stalled worker restarts with fresh history.
When the matching camera and modeld message arrive separately, the scheduler can wait up to 100 ms from the slot opening. The 500 ms deadline stays fixed.
`modelV2.alpamayoQueryMonoTime` identifies the selected plan; zero means the local action was used. `localAction` records the independent small-model action in planner mode.
The logged `alpamayoPlan` contains the GPU path. `modelV2.position` still describes the small model. Radar, longitudinal limits, and Panda safety keep their usual roles.

`PlanSelector.candidate_action(now, v_ego, lat_delay, long_delay)` derives steering and acceleration from Alpamayo alone for live selection and replay.
The small model stays running for camera odometry, perception and immediate fallback. Its action does not vote on an eligible Alpamayo plan.

The default runtime uses original BF16 weights, greedy reasoning, ten Euler steps, and a fixed diffusion-noise file.
It retains all five images per front camera: `(-2, -1, -0.5, -0.1, 0)` seconds.
It retains complete image features, all reasoning tokens, and all 64 future positions and rotations.
The BF16 default encodes images individually. Exact content and position keys permit bounded feature reuse.
The `selected` variant on this branch uses four-bit serial language MLP weights, mixed FP8/BF16 elsewhere, and three Euler steps. See [selected/README.md](selected/README.md).
Both variants retain the full image grids and history. The four-bit and FP8 models can produce different reasoning and trajectories.

## Runtime installation

Keep the normal openpilot environment and tinygrad submodule unchanged.
Run this explicit installer into a new directory:

```sh
python -m openpilot.selfdrive.alpamayo.setup_runtime /path/to/alpamayo-runtime
```

The installer checks out these sources:

| Source | Revision |
| --- | --- |
| comma tinygrad | `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e` |
| NVIDIA Alpamayo 2 Edge | `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` |

NVIDIA dependencies use that revision's committed `uv.lock`, with a frozen install and an existing Python 3.12 interpreter; Python is not downloaded.
Installation seeds the locked, hashed setuptools wheel. Only the reviewed ANTLR 4.9.3 source distribution may build, using those locked build dependencies.
Other dependencies require wheels. `--no-install-project` avoids an unlocked Hatchling build, and `source.pth` adds the verified pinned NVIDIA source.
The additional Linux Python 3.12 PyAV, pycapnp, and zstandard wheels use exact versions and hashes in `extras.txt`.
Inference rejects changed tracked runtime source or different source revisions.
System LLVM, Chestnut access, and the normal openpilot native build remain prerequisites.
Fresh ARM dependency installation has been tested without GPU execution or changes to the normal environment.

Download the `nvidia/Alpamayo2-Edge` checkpoint explicitly at revision `b92b6aa326bd3e2235db28f248ad5a30b5890e64`.
Inference reads local weights and does not download them.
Provide a fixed float32 `.npy` noise array with shape `(1, 64, 2)`.
Use the same file for every comparison. Equal NumPy and PyTorch seeds do not imply equal noise.

## Public API and replay

Use the isolated interpreter. Set `PYTHONPATH` to the isolated tinygrad checkout, openpilot root, and its `opendbc_repo`.
The checkout must contain its initialized normal submodules. The shadow worker sets these three paths explicitly and disables user site packages;
it does not depend on normal-environment editable installs.

```python
from openpilot.selfdrive.alpamayo.inference import Runtime

runtime = Runtime(checkpoint, initial_noise)
prediction = runtime.predict(data)
runtime.close()
```

`data` contains NumPy arrays plus one navigation string:

| Field | Shape and type |
| --- | --- |
| `image_frames` | `(10, 1, 3, 432, 768)`, uint8 calibrated RGB |
| `camera_indices` | `(10,)`, int64; five `1` values, then five `6` values |
| `absolute_timestamps` | `(10,)`, int64 capture timestamps in microseconds |
| `ego_history_xyz` | `(1, 20, 3)`, float32, forward/left/up |
| `ego_history_rot` | `(1, 20, 3, 3)`, float32 |
| `nav_text` | `[text]`; an empty string represents unavailable guidance |

`prediction` contains `xyz`, `rot`, `generated_token_ids`, `reasoning`, and stage timings.
`runtime.run_prediction` validates complete causal inputs and associates outputs with their exact query.
Live inference and replay share the selected variant's `Runtime.predict` implementation or the original default implementation.

Replay accepts a prepared manifest with an ordered `schedule`.
Each row identifies a sample directory, `query_ns`, and its ten `frames`.
Each directory contains the six fields above as `.npy` files; `nav_text.npy` holds a scalar string.
Frame references retain `view`, `capture_ns`, `requested_ns`, and `image_calibration_ns`.
When present, `vehicle_history_audit` remains part of reference identity.

```sh
DEV=USB+AMD:LLVM GMMU=0 TC_OPT=2 CACHEDB=/path/to/compiler-cache.db \
  /path/to/alpamayo-runtime/nvidia/.venv/bin/python -m openpilot.selfdrive.alpamayo.replay \
  --manifest /path/to/prepared/manifest.json --checkpoint /path/to/checkpoint \
  --initial-noise /path/to/noise.npy --output /path/to/new-reference \
  --variant default --period-ms 500 --warmup 20 --runs 20
```

Add `--reference /path/to/reference` to compare another run. `--thresholds` accepts explicit JSON limits.
Use `--start-index N --warmup 0 --runs 1` for one cold prediction at row N.
For warm measurements, start earlier and use `--warmup` before the measured rows. Compare references from the same window.
For a selected prediction generated in full serial blocks, add `--decoder-repeats 8` with `--runs 1`.
Replay times the first six, twelve (if available), and all tokens with the same weights and prompt resident. Each prefix must match the original tokens.
These component trials run after the pipeline measurement. They do not count toward replay throughput.
`startup.initialization_ms` stops when the runtime and weights are ready. The first prediction separately pays for first-use GPU graph setup and execution.
`default` selects the original BF16 reference; `selected` requires the fixed external assets described below.
Unavailable variants are rejected without fallback. `--no-vision-cache` permits a default full-image cache comparison at the same single-image encoding layout.

Outputs contain ordered predictions, actual shard-content hashes, input-file hashes, frame references, and the final `gpu_executed` summary.
Incomplete, reordered, repeated, or mismatched reference sequences fail comparison.
Default limits are ADE 0.25 m, FDE 0.5 m, speed MAE 0.25 m/s, acceleration MAE 0.5 m/s², and heading 2°.
Reasoning and token mismatches are reported separately.
The initial `--period-ms 500` gate requires every measured full pipeline at most 500 ms and elapsed throughput at least 2 Hz.
The selected period must match the manifest's actual query spacing; use `--period-ms 100` for existing 100 ms fixtures.
Accepted periods are 100, 250, and 500 ms. Reports retain the period and actual recorded intervals and separately report the ultimate `goal_100ms` / 10 Hz result.
Prepared replay times sample reads, token/position preprocessing, transfers, inference, and trajectory decoding.
It excludes camera capture, ISP, video decoding, projection, and ego-history construction; it does not measure live camera age.
Reference generation preserves slow outputs and returns failure when the performance gate is unmet.
A qualified PASS requires both reference agreement and speed; generating a reference alone does not qualify a run.
`model_execution_ms` measures the synchronized model call after inputs are ready, like modeld's `modelExecutionTime`.
`full_pipeline_ms` also includes loading and validating the prepared sample. The live slot deadline separately includes waiting for input publication.
A comparison command returns failure if numerical or performance limits fail.

## Live configuration

Configure `ALPAMAYO_CONFIG` in the manager environment before starting openpilot. Keep `ALPAMAYO_SHADOW=1` for capture without selecting GPU actions.
For planner evaluation, unset the shadow flag. Without an available GPU, modeld continues running the local small model.
With Alpamayo disabled, `DISABLE_CHESTNUT=0` restores the existing large driving-model selection.

The private JSON configuration contains:

- `python`: isolated `nvidia/.venv/bin/python` path.
- `tinygrad`: isolated tinygrad checkout path.
- `output`: existing output directory, separate from route recordings.
- `cache`: absolute existing writable persistent cache directory, separate from route recordings and `output`.
- `period_ms`: fixed live slot period; default `500` for the initial 2 Hz shadow car test. Accepted values are `100`, `250`, and `500`.
- `retained_cpu_inputs`: optional boolean, default `false`. The selected 500 ms worker can retain immutable calibrated images and byte patches.
- `geometry`: the qualified source-camera geometry object described below.
- `inference`: `variant` defaults to `default`, with `checkpoint`, `initial_noise`, and optional `cache_vision`. The selected fields are shown below.

Use absolute paths. No inference settings or model paths are added to normal openpilot dependencies.
With retained CPU inputs, all ten images still contribute their ordered bytes to the canonical NPY audit hash.
The selected inference option `native_fresh_patches` independently enables the pinned CPU patch kernel for cache misses. Both options default to `false`.
The worker packs the four new steady-state exposures and copies only missing patch sets to the upload source.

For `native_fresh_patches`, build the AArch64 library before packaging the worker. Run this from the repository root with the pinned Torch 2.8.0 environment:

```sh
python -m openpilot.selfdrive.alpamayo.selected.native_patches.build --output-dir "$STAGED_SELECTED_PACKAGE"
```

`STAGED_SELECTED_PACKAGE` is the staged `openpilot/selfdrive/alpamayo/selected` directory containing `fresh_patches.py`. Include the generated `fresh-aarch64.so` there in the device package.
The source checkout does not contain the library. The builder requires Linux, GCC 13.3.0 (Ubuntu `13.3.0-6ubuntu2~24.04.1`), GNU binutils 2.42, and AArch64 glibc development files.
It selects `aarch64-linux-gnu-gcc` for cross compilation or `gcc` on AArch64. Use `--cc` for another path to the same toolchain.
The build derives the exact filter tables from Torch and checks their hashes. It replaces the output only after matching the qualified library digest.
The runtime requires AArch64 Linux and Torch `2.8.0+cpu` for this option. Missing or mismatched libraries stop startup. With the option off, no library or compiler is required.

The worker sets `XDG_CACHE_HOME` to `cache`; the baseline uses `cache/compiler-cache.db` and the selected backend uses `cache/tinygrad.db` for `CACHEDB`.
Reuse this isolated directory across ignition cycles; it is outside session prediction limits and is never automatically deleted.
Stage the required pinned firmware cache there before hardware use. The installer and CPU tests do not download firmware.
`geometry.cameras` has keys `"1"` and `"6"`.
Each entry supplies `prepared_size`, `content_box`, `focal_pixels`, `principal_point`, and `projection`.
`prepared_size` must be `[768, 432]`; projections are `equidistant` or `pinhole`.
Geometry values describe the prepared coordinate system, with source content bounds mapped from native 1344×760 pixels.
Use the source geometry associated with the calibrated comparison inputs. The code does not guess installed lens geometry.
Projection targets nominal 120° and 30° views, using measured mounting and wide-camera rotations before each capture.
This does not establish measured lens-distortion calibration or support other native camera dimensions.

For selected shadow inference, merge this section into the live configuration, retaining `python`, `tinygrad`, `output`, and the qualified `geometry`:

```json
{
  "period_ms": 500,
  "cache": "/path/to/persistent-cache",
  "inference": {
    "variant": "selected",
    "checkpoint": "/path/to/Alpamayo2-Edge",
    "fp8_cache": "/path/to/fp8-cache",
    "prefill_calibration": "/path/to/prefill-scales.json",
    "initial_noise": "/path/to/initial_noise.npy",
    "device_timeout_ms": 10000
  }
}
```

Required external assets are the pinned checkpoint's `config.json`, weight index, `00000.safetensors`, and `00001.safetensors`;
the fixed FP8 `index.json` plus `weights.bin`; and the fixed JSON containing 112 prefill scales.
Their exact identities are in [selected/assets/artifacts.json](selected/assets/artifacts.json) and are checked at startup.
Supply the same finite float32 `(1,64,2)` noise file used for the reference. Stage the eight hash-checked gfx1200 firmware files in the persistent cache before starting;
the packaged frozen GPU objects are under `selected/assets/gfx1200`. Selected startup does not fetch firmware or weights.
The worker removes `variant` and derives the adapter's `cache_dir` from the top-level `cache`; do not add `cache_dir`, `cache_vision`, or arbitrary recipe fields to selected inference settings.
To run the original baseline instead, use `"variant": "default"` with only `checkpoint`, `initial_noise`, and optional `cache_vision`.

Capture retains bounded native frames and state histories. It requires an exact small-model frame-ID and EOF match.
Current and −100 ms exposures use causal floor timestamps. Invalid state updates block use of older valid state.
Later-received updates replace the same measurement timestamp, including invalidations when a filter stops advancing.
Measured motion supplies vertical motion and tilt. Valid learned vehicle parameters supply planar bicycle-model history.
Motion admission requires the publisher's input, sensor, pose, orientation, and velocity health flags.
Learned parameters require both their aggregate validity and the separately published `sensorValid` flag.
Navigation uses the latest valid causal `navInstruction`; missing guidance remains empty.
App destinations use the authenticated destination RPC and device-owned Mapbox route.
Model guidance uses compact maneuver, direction, distance, and roundabout-exit text within the selected prompt budget.
Full road names and sign text remain in saved navigation metadata. Unsupported oversized prompts are rejected without truncating tokens.
Disabled, replaced, loading, stale, and last-known-position routes are rejected.

The initial car-test cadence is one attempt every 500 ms. Slots align to the monotonic clock once the model worker is ready.
Each slot admits at most one snapshot. Planner mode allows one local-model frame for a previous worker call to finish at the slot boundary.
Skip slots that remain busy or lack causal inputs. Never replay a missed slot. The 500 ms benchmark deadline does not include the grace period.
`250` ms remains an unqualified diagnostic option. `100` ms / 10 Hz is a later phase after the initial car test.
The input camera rate and requested slot rate do not establish achieved output throughput; actual completion times and missed slots determine that rate.
The first five completed outputs are marked startup; graph compilation can overrun multiple slots. Later overruns remain visible in deadline results.
At 500 ms, the worker retains the calibrated current pair from four consecutive slots. A gap restarts that history.
For each front camera it reuses the actual current exposures from four, two, and one slot ago, with their original timestamps and calibration.
Selected inference encodes both current anchor images during each priming slot. Late priming or a lost historical feature restarts the two-second history.
With distinct images, the first complete prediction and each following prediction encode four new images. Priming produces no driving action.
Other cadences select all five exposures from capture timestamps. Feature reuse always depends on actual image content.

The supervisor monitors small-model health, temperatures, memory, and storage while inference runs in its isolated child.
It maintains one input slot in a private `/dev/shm` directory and selects fresh inputs when the worker becomes idle.
The full immutable snapshot is limited to 32 MiB, with at least 8 MiB tmpfs space and 768 MiB system `MemAvailable` left after admission.
Missing tmpfs or inadequate RAM stops shadow inference; input snapshots never fall back to flash storage.
The pending/working slot stays occupied until inference completes. The supervisor removes its RAM directory after worker shutdown or failure.
It terminates only its own worker group after failure or shutdown. Durable predictions and the worker log stay under `output`.
Predictions use atomic, synchronized writes under a boot/session directory, named by query timestamp.
They include complete outputs, input identities, frame references, timing, calibration, ego history, and the matching small-model output.
Every frame records the actual `rpyCalib` and `wideFromDeviceEuler` used for projection alongside its calibration timestamp.
The current ego calibration and all 39 learned-parameter integration samples retain source timestamps, ages, steering ratio, stiffness, angle offset, and roll.
Car parameters and the effective clamped steering ratio/stiffness are also saved. Calibration acts through projection and ego history; it is not added to model text.
The runtime manifest retains model identities and camera geometry.
Live outputs report `live_processing_ms` from the earliest receipt of the selected current camera pair through validated output readiness.
This includes current-frame copying, selection, state/snapshot serialization, RAM handoff delay, preprocessing, transfers, inference, and output validation.
Historical image copies happened on earlier captures; their original receipt/ready times remain in frame references without adding the two-second lookback to this interval.
`worker_processing_ms` measures only worker pickup through validated output readiness. `input_age_ms` retains narrow-query capture-to-output age.
`slot_start_ns`, `slot_deadline_ns`, actual `selection_started_ns`, `output_ready_ns`, selection/completion lateness, and `slot_deadline_met` are saved.
`input_age_at_selection_ms` distinguishes the captured query's age from scheduling delay. `missed_slots` counts skipped slots before the recorded selection,
including worker-rejected inputs acknowledged by the supervisor; overruns during the current inference appear in its deadline result and subsequent missed counts.
All interval endpoints are recorded. These output-readiness metrics exclude durable JSON serialization/fsync; elapsed live throughput must include logging.
Live output no longer calls the narrower worker interval `full_pipeline_ms`. Prepared replay's timing scope remains separate.
The corresponding normal route video and logs remain necessary for video replay.
Completed predictions are never automatically deleted. Storage admission preserves the logger's 10% and 5 GiB thresholds.
Each session limits prediction output to 256 MiB. Output permissions are private, and normal route recording directories are rejected.
Source frame-ID gaps remain in prediction metadata. No uploader or cloud publication is configured.

## Validation and remaining work

Use the BF16 replay as the accuracy reference. Also measure every deadline on advancing camera inputs with the local model running.
The selected two-camera bench implementation still missed 500 ms on a turn. Live timing and vehicle control are unqualified.

The model, input adapter, and AMD kernels derive from comma's Alpamayo prototype, including revision `fe68e1d0595d48ffd0d058404181d873250e4ae4`.
Their Comma.ai copyright and MIT license remain in the repository [LICENSE](../../../LICENSE).
NVIDIA's external package retains its Apache-2.0 license. Checkpoint terms remain with the checkpoint distribution.
