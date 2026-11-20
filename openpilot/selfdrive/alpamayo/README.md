# Alpamayo inference and planner evaluation

The small driving model runs by default. Shadow mode saves GPU predictions without changing driving actions.
Planner mode lets modeld select GPU actions at its regular cadence. This mode is under bench evaluation. Vehicle control remains unqualified.

Setting `ALPAMAYO_CONFIG` selects the planner (period from the configuration) unless `ALPAMAYO_SHADOW=1` or `ALPAMAYO_PLANNER=0` is set.
The file must contain a selected-model configuration. With no configuration, the existing openpilot model selection applies.
Planner mode reserves Chestnut for Alpamayo. The small model keeps running for camera odometry and perception only.
In planner mode the driving action comes only from Alpamayo plans; the small model's action is never used.
The manager starts the GPU worker onroad when Chestnut enumerates at 5 Gbps and the device is healthy. Weights can load alongside modeld startup.
Input admission still requires a healthy local model. The first five completed predictions are startup samples.
A plan is usable when it was ready within 750 ms of its camera query, and it drives while that query is under 1.5 s old.
Two consecutive plans with queries at most 1 s apart are required before the first use. Late, stale, repeated or invalid packets are ignored.
modeld reads plans once per camera frame after its own model run, so it accepts a packet received up to 100 ms after publication.
modeld ages the plan from camera capture before computing the requested action. Its planned speed must agree with measured vehicle speed within 2 m/s;
curvature and acceleration must stay within physical limits. Low speeds, stops and starts use the plan.
`calibration.py` learns a lateral gain per speed bucket, like torqued. Alpamayo plans turned 0.6-0.9 of the measured heading change.
It compares each plan's predicted 1 s heading change with the change `deviceMotion` then measured, from driver-steered samples only.
Plans must turn the same way as the car, so a turn chosen without route guidance does not teach it. The prior is 1; gains stay within 0.8-2.0.
The scaled curvature keeps the same curvature and lateral-acceleration limits. The state is stored in `lateral_calibration` (default `/data/alpamayo-calibration/lateral.json`).
Without a usable plan, modeld holds the last Alpamayo curvature with non-positive acceleration and leaves
`modelV2.alpamayoQueryMonoTime` at zero. selfdrived then raises `alpamayoUnavailable`: no entry, or soft disable when engaged.
Normal openpilot engagement, vehicle-specific controls, radar and Panda safety remain authoritative downstream.
`modelV2.alpamayoQueryMonoTime` identifies the plan that produced the action. `localAction` records the independent small-model action.
The logged `alpamayoPlan` contains the GPU path. `modelV2.position` still describes the small model.

`PlanSelector.candidate_action(now, v_ego, lat_delay, long_delay)` derives steering and acceleration from Alpamayo alone for live selection and replay.

The default runtime uses original BF16 weights, greedy reasoning, ten Euler steps, and a fixed diffusion-noise file.
It retains all five images per front camera: `(-2, -1, -0.5, -0.1, 0)` seconds.
It retains complete image features, all reasoning tokens, and all 64 future positions and rotations.
The BF16 default encodes images individually. Exact content and position keys permit bounded feature reuse.
The `selected` variant on this branch uses four-bit serial language MLP weights, mixed FP8/BF16 elsewhere, and one Euler step. See [selected/README.md](selected/README.md).
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
For planner evaluation, unset the shadow flag. Without Alpamayo plans, selfdrived blocks engagement (`alpamayoUnavailable`).
With Alpamayo disabled, `DISABLE_CHESTNUT=0` restores the existing large driving-model selection.

The private JSON configuration contains:

- `python`: isolated `nvidia/.venv/bin/python` path.
- `tinygrad`: isolated tinygrad checkout path.
- `output`: existing output directory, separate from route recordings.
- `cache`: absolute existing writable persistent cache directory, separate from route recordings and `output`.
- `period_ms`: live slot period; `250` for the car test, `500` for 2 Hz. Accepted values are `0` (free running), `250`, and `500`.
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
Model guidance is only maneuver, direction, roundabout exit and distance, for example `Turn right in 100m`.
An unrecognized maneuver sends no guidance. Full road names and sign text remain in saved navigation metadata.
Disabled, replaced, loading, stale, and last-known-position routes are rejected.

The car-test cadence is a 250 ms slot grid (`period_ms: 250`); 500 ms remains available. Slots align to the monotonic clock.
Every slot keeps its calibrated camera pair in the worker's history ring, because later queries reuse the pairs from
two, four and eight slots ago (-0.5, -1 and -2 s). Each slot is predicted or, when late, only anchored:
- The supervisor selects each slot as soon as it opens and queues it (one input may wait while the worker runs).
- The worker predicts a slot it picks up within 200 ms of its opening (`START_LATENESS_NS`); a later one only encodes its pair.
- A slot superseded before it was queued becomes a backlog anchor, served before the newest slot.
- The supervisor hands the next slot's predicted -0.1 s pair to the worker 160 to 20 ms before the opening. A worker thread
  projects and patches it on the CPU; the main thread queues its GPU encode between predictions.
- While a prediction runs, the same thread projects and patches the current pair of the queued slot. On pickup the
  worker queues that GPU encode at once. A backlogged slot thus skips about 35 ms of CPU work before its GPU work.
- Audit hashes and the prediction record are written on a log thread, which then releases the input mapping.
- A text that reaches 64 tokens without the trajectory marker yields no plan for that slot; the worker keeps its camera history.
- Anchors queue their GPU encode without waiting, and status files are written on the log thread.
- Handing over the predicted current pair before the opening cut median latency by about 25 ms. It lowered throughput
  (replay 3.46 vs 3.58-3.64 Hz): in a backlog both early encodes then serialize their USB uploads after pickup.
Never replay a missed slot. Plans are judged by query-to-ready latency (at most 500 ms), not by the slot deadline.
The first five completed outputs are marked startup. A history gap restarts the two-second history (eight slots at 250 ms).
With distinct images, each prediction encodes four new images: the current pair and the -0.1 s pair.
Other cadences select all five exposures from capture timestamps. Feature reuse always depends on actual image content.
Each history exposure is the latest one that starts at most 10 ms after its nominal offset and not after the query.
Real camera periods jitter by about 0.2 ms, so a strict cutoff alternated the -0.1 s image between 100 and 150 ms old.

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
Each session records up to 2 GiB of predictions. Reaching that limit or the memory or disk reserve stops recording, never inference.
Output permissions are private, and normal route recording directories are rejected.
Source frame-ID gaps remain in prediction metadata. No uploader or cloud publication is configured.

## Validation and remaining work

Use the BF16 replay as the accuracy reference. Also measure every deadline on advancing camera inputs with the local model running.
The selected two-camera bench implementation still missed 500 ms on a turn. Live timing and vehicle control are unqualified.

The model, input adapter, and AMD kernels derive from comma's Alpamayo prototype, including revision `fe68e1d0595d48ffd0d058404181d873250e4ae4`.
Their Comma.ai copyright and MIT license remain in the repository [LICENSE](../../../LICENSE).
NVIDIA's external package retains its Apache-2.0 license. Checkpoint terms remain with the checkpoint distribution.
