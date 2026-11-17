# Alpamayo six-camera investigation

This experimental source archive is preserved on `alpamayo-experiments-20261117`.
It includes rejected candidates and diagnostic scripts, not a supported runtime.
The `alpamayo` branch contains the maintained integration relative to `master`.
Training and distillation work stopped on 2026-11-17. Their source remains archived.
Generated weights, recordings, compiler caches, and local measurement artifacts remain outside this source snapshot.
The measurements below describe their dated workloads. They do not establish 10 Hz device operation.

Initial benchmark saved on 2026-11-15. Recorded-route replay and calibration results added on 2026-11-16.
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

## Replay a recorded route

`replay-route.py` prepares synchronized inputs, runs inference, and renders camera footage with reasoning and future paths.
It requires the normalized export manifest, camera timestamp logs, and native route rlogs.
Run preparation with the openpilot checkout and its `opendbc_repo` on `PYTHONPATH`.
The pinned Python environment supplies NumPy, SciPy, PyAV, Pillow, Cap'n Proto, and Zstandard.
Rendering also requires FFmpeg and the DejaVu Sans font.

Set `EXPORT`, `LOGS`, `OUTPUT`, `SAMPLES`, and `WORK` to explicit directories.
Keep model assets, compilation caches, and source snapshots outside the recording filesystem.
Set `RECORDING_UUID` to the required recording filesystem UUID.

```bash
"$PYTHON" "$EXP/replay-route.py" prepare --export "$EXPORT" --logs "$LOGS" \
  --samples "$SAMPLES" --output "$OUTPUT" --hz 1 --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/replay-route.py" infer --output "$OUTPUT" --checkpoint "$MODEL" \
  --tinygrad "$TINYGRAD" --noise "$REFERENCE/initial_noise.npy" --fp8-cache "$FP8_CACHE" \
  --cache "$WORK/cache.db" --work "$WORK/inference" --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/replay-route.py" render --output "$OUTPUT" --recording-uuid "$RECORDING_UUID"
```

The cadence selects recorded timestamps; offline processing runs as fast as possible.
Every prediction uses measured ego history and an exactly matched logged openpilot frame.
Missing side views are omitted from the prompt when any required history frame exceeds 150 ms age.
The runner retains both front views and supports two to six available cameras without synthetic black inputs.
Each camera layout runs in a separate process to isolate captured GPU graphs and their cache buffers.
Use `--group-prompts --batch-size 64 --collect-garbage` for offline trials with many navigation and image changes.
These options group equal prompt lengths and bound each GPU process while retaining saved predictions and final chronological output.
For long route replays, `--persistent-runtime --group-prompts --collect-garbage` retains one loaded model across all remaining layouts.
It uses single-image vision graphs, eight cached stitching graphs, a shared 3,584-position KV allocation, and collection every 32 predictions.
Prompt lengths, attention masks, image histories, precision, and diffusion steps remain unchanged.
When saved controls exist, the runner verifies generated tokens and trajectory differences before processing new inputs.
The persistent mode ignores process batch limits and avoids speculative warmup of unused graph variants.
Record its control results and measured throughput separately from the saved benchmark configuration.
Interrupted inference resumes from verified journal entries without repeating completed predictions.
Repeated history images share features while retaining every image position in the prompt.
Run `check-vision-cache.py` with the pinned tinygrad checkout on `PYTHONPATH` to check repeated-image graph replay.
The manifest records omitted views, frame references, calibration, and source log checksums.

The default instruction is `Follow the road.` It does not infer a destination from the recorded driving path.
Images retain their complete field of view, receive the recorded orientation correction, and use aspect-preserving padding.
This does not calibrate the camera lenses or side-camera extrinsics to NVIDIA's setup.
The saved FP8/BF16 configuration uses three diffusion steps and the restricted reasoning vocabulary described above.

Outputs include `predictions.jsonl`, `manifest.json`, `comparison.mp4`, previews, and comparison metrics.
The video places Alpamayo and openpilot paths on the large wide-camera view, with five smaller camera views beside it.
Navigation and non-nominal model messages appear below the footage. Display zoom defaults to 1.0.
Captions remove an incomplete leading `<N` marker and display complete `<NOMINAL>` outputs as “No additional reasoning”.
Raw generated text and token IDs remain unchanged. The source of malformed generations remains unresolved.
The overlay uses recorded calibration, camera height, and measured ego motion to align each saved path with subsequent video frames.
The blue path uses every frame-matched logged openpilot prediction, independently of Alpamayo's selected cadence.
The gray path shows integrated recorded motion at each video frame. It is a motion estimate, not independent ground truth.
The wide-camera projection follows openpilot's nominal intrinsics; measured lens distortion remains unavailable.
Run `check-route-geometry.py` to verify axes, camera height, calibration, and motion compensation.
Paths use forward/left/up coordinates; logged openpilot paths are converted from forward/right/down.
Recorded motion is an integrated estimate, not independent ground truth. Future trajectory arrays never enter the model inputs.
Metrics describe agreement with that estimate and do not establish driving safety.
Source recordings remain unchanged.

### Reconstructed navigation

When live navigation is absent, `navigate` creates a separate replay from reviewed turn annotations and integrated travel distance.
This is retrospective route-intent testing. Do not present it as recorded navigation or independent evaluation.
The annotation JSON contains `route`, ordered `turns` with `start_s`, `end_s`, and `direction`, and `arrival_s`.
Times are seconds from the exported capture start. Directions are `left` or `right`.

```bash
"$PYTHON" "$EXP/replay-route.py" navigate --source "$OUTPUT" --turns "$TURNS" \
  --logs "$LOGS" --samples "$NAV_SAMPLES" --output "$NAV_OUTPUT" \
  --trim-end-seconds 50 --recording-uuid "$RECORDING_UUID"
```

Run `infer` and `render` with `NAV_OUTPUT` afterward. Use separate inference work directories for different prepared inputs.
Unchanged input arrays use hardlinks, so their source and destination directories must share a filesystem.
The original replay, logs, footage, and predictions remain available for comparison.

### Camera-input experiments

`camera_profiles.py` compares road-camera crops at 45, 35, and 30 degrees, then adds wide and side-camera adjustments.
The full crop profile targets 120-degree front-wide and cross views, 70-degree rear views, and a 30-degree telephoto view.
`all_map` uses NVIDIA's measured F-theta camera intrinsics with explicitly nominal source lens models.
It preserves each source optical axis; it cannot correct unknown camera mounting angles or positions.
Reject target rays outside the recorded image rather than synthesize unseen content.

Provide source geometry and target calibration JSON explicitly. Preserve their dataset revision, clip identifier, and calibration assumptions.
Source `cameras` use numeric camera-ID keys with `prepared_size`, `content_box`, `focal_pixels`, `principal_point`, and `projection`.
Source projection is `equidistant` or `pinhole`. Target `cameras` use NVIDIA camera names with `width`, `height`, `principal_point`, and `r2th` coefficients.
Polynomial coefficients use ascending powers. Input geometry describes the prepared image, including padding and remote-stream resizing.

```bash
"$PYTHON" "$EXP/camera_profiles.py" extract --source "$NAV_OUTPUT" --output "$RAW_FRAMES"
"$PYTHON" "$EXP/camera_profiles.py" prepare --source "$NAV_OUTPUT" --output "$CROP_OUTPUT" \
  --samples "$CROP_SAMPLES" --profile tele30 --raw-frames "$RAW_FRAMES" \
  --source-calibration "$SOURCE_GEOMETRY" --target-calibration "$NVIDIA_CALIBRATION" \
  --recording-uuid "$RECORDING_UUID"
```

Native extraction preserves original decoded resolution and applies recorded display rotation exactly once.
Use `--raw-frames` to crop before resizing, especially for the telephoto view.
Omitting it explicitly tests crops of the already prepared images, which can lose angular detail.
Use `--ticks` to prepare selected timestamps. Hold navigation, history, model settings, noise, and pixel budgets fixed across candidates.
Choose tuning and held-out timestamps before inference, and report both sets separately.
Model-input cropping does not change the displayed camera footage or the logged openpilot predictions.

### Compare camera configurations

`subset` removes other camera histories without resampling retained pixels or changing navigation and motion inputs.
Both front cameras remain required. Use a new output directory and inference work directory.

```bash
"$PYTHON" "$EXP/camera_profiles.py" subset --source "$CROP_OUTPUT" --output "$TWO_OUTPUT" \
  --samples "$TWO_SAMPLES" --views 1 6 --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/replay-route.py" infer --output "$TWO_OUTPUT" --checkpoint "$MODEL" \
  --tinygrad "$TINYGRAD" --noise "$REFERENCE/initial_noise.npy" --fp8-cache "$FP8_CACHE" \
  --cache "$WORK/cache.db" --work "$WORK/two-inference" --persistent-runtime \
  --group-prompts --collect-garbage --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/route_compare.py" --output "$TWO_OUTPUT" --reference "$CROP_OUTPUT" \
  --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/replay-route.py" render --output "$TWO_OUTPUT" --compare-output "$CROP_OUTPUT" \
  --recording-uuid "$RECORDING_UUID"
```

The combined video shows magenta two-camera and orange surround-camera predictions, with both reasoning messages.
To add front-side views, prepare another subset with `--views 0 1 2 6` and run fresh inference.
Compare all three outputs with `route_compare.py --output FOUR_OUTPUT --reference SURROUND_OUTPUT --other TWO_OUTPUT`.
Render with `replay-route.py render --output FOUR_OUTPUT --compare-output SURROUND_OUTPUT TWO_OUTPUT`.
The four-camera path is green. Each configuration retains its own complete reasoning message and prediction cadence.
Blue openpilot and gray recorded-motion paths retain their independent frame alignment.
The surround run can omit missing views; its six-camera label identifies the configured setup.
`camera-comparison.json` reports every configuration pair, turn windows, full-six-camera cases, and unchanged-input controls.
Controls measure runtime variation when different configurations receive the same available views.
These correlated route samples measure agreement with integrated motion, not closed-loop driving performance.

For calibrated comparisons, subset the separated `calibrated_vehicle_history` output and run fresh inference on the subset.
Retained front images, navigation, and learned motion history must match the surround run exactly.
Pass only these updated outputs to the renderer to replace older Alpamayo overlays.
The legend identifies calibration plus learned history and displays each run's measured Hz and mean trajectory error.
Measure throughput again using the updated inputs. Set each timing result's `profile` to the measured manifest's `camera_input_profile`.
The renderer rejects mismatched profile labels. Small and big openpilot overlays retain their independent recorded timestamps.

For throughput comparisons, select identical chronological timestamps with complete camera histories in every configuration.
Keep the runtime, image budgets, crops, navigation, noise, precision, and solver steps fixed.
Report startup separately from measured execution. Do not compare a warm short replay with a full run containing process restarts.

### Continuous openpilot big-model replay

`replay_openpilot_big.py` runs the installed big-model artifact without starting vehicle publishers or controls.
Use the openpilot checkout's Python environment and its pinned tinygrad version, separately from the Alpamayo environment.
Prepare recorded inputs before reserving the GPU. Keep other GPU benchmarks separate from this replay.

```bash
export PYTHONPATH="$OPENPILOT/tinygrad_repo:$OPENPILOT/opendbc_repo:$OPENPILOT/msgq_repo:$OPENPILOT"
"$OPENPILOT/.venv/bin/python" "$EXP/replay_openpilot_big.py" prepare \
  --source "$FOUR_OUTPUT" --logs "$LOGS" --checkout "$OPENPILOT" \
  --output "$BIG_OUTPUT" --recording-uuid "$RECORDING_UUID"
DEV=USB+AMD:LLVM "$OPENPILOT/.venv/bin/python" "$EXP/replay_openpilot_big.py" infer \
  --output "$BIG_OUTPUT" --recording-uuid "$RECORDING_UUID"
```

The replay checks artifact hashes, source files, imported modules, and the pinned tinygrad revision.
It processes every paired native front frame with continuous recurrent state and recorded vehicle inputs.
At least 20 seconds of recorded pre-roll precede the displayed interval.
Original MP4 frames decode directly to NV12. Standard openpilot camera warps replace the Alpamayo input crops for this model.
An independent CPU warp checks the first compiled GPU warp before predictions are accepted.
The replay follows modeld's recorded-value policy, including invalid message flags, and reports their counts.
Interrupted recurrent runs require a fresh output directory; individual predictions cannot resume without their recurrent state.

Use a new video output directory containing the prepared Alpamayo manifest, predictions, and rendering assets.
Add `--big-model-output "$BIG_OUTPUT"` to both `route_compare.py` and `replay-route.py render`.
The big-model path appears red; the logged small-model path remains blue. Both update at the native camera cadence.
Metrics require an exact matching native capture timestamp and account for the recorded calibration frame.
Openpilot and Alpamayo use different model inputs. Camera-only controlled comparisons apply between the Alpamayo configurations.

### Four predictions per recorded second

Use `prepare --hz 4 --trim-end-seconds 50`, then add navigation and the selected camera profile.
Every prepared timestamp requires a fresh prediction. The renderer does not interpolate model predictions to increase their cadence.
Camera-layout groups run separately, with chronological inputs inside each group unless `--group-prompts` is selected.
Persistent mode uses one process; optional prompt grouping still changes processing order, while the saved route remains chronological.
This is an offline throughput measurement. Camera capture, live layout changes, and streaming deployment are outside its scope.

`infer` writes `command-timing-*.json` into its work directory, including failed and resumed attempts.
The measured wall time includes CPU input processing, model loading, graph compilation, and process restarts.
Prior video decoding, input preparation, and rendering are excluded.
Report new predictions divided by this wall time separately from the requested sampling cadence.
Run `route_timing.py --output "$CROP_OUTPUT" --work "$INFERENCE_WORK" --recording-uuid "$RECORDING_UUID"` to save `timing-summary.json`.
Benchmark reports also preserve startup time, model loading time, measured throughput after warmup, and individual model-stage timings.
Do not add overlapping CPU and GPU stage times or equate reciprocal median latency with measured throughput.

### GPS map in the video

Before rendering, prepare a small map from recorded GPS:

```bash
"$PYTHON" "$EXP/route_map.py" --output "$CROP_OUTPUT" --logs "$LOGS" \
  --tile-cache "$WORK/map-tiles" --recording-uuid "$RECORDING_UUID"
"$PYTHON" "$EXP/replay-route.py" render --output "$CROP_OUTPUT" --recording-uuid "$RECORDING_UUID"
```

The MP4 contains the route overview, progress, and moving GPS marker. Navigation and unabridged model reasoning use separate caption cards.
Recorded GPS aligns by publication timestamps. Gaps longer than one second remain unconnected, and stale markers are hidden.
Preparation downloads and caches the needed OpenStreetMap tiles. The rendered video requires no network access.
`post-caption.txt` contains the map credit and replay context. Include this text in the accompanying publication description.
OSMF's video attribution guidelines permit description credits when the map is not the production's focus.
The video does not show Alpamayo controlling the car or establish real-time operation.

### Short calibration and lens studies

`calibration_trials.py` prepares short, fixed turn windows before inference. It preserves navigation, model settings, and the recorded evaluation paths.

```bash
"$PYTHON" "$EXP/calibration_trials.py" --source "$BASELINE_OUTPUT" --output "$TRIALS" \
  --logs "$LOGS" --raw-frames "$RAW_FRAMES" \
  --source-geometry "$SOURCE_GEOMETRY" --target-geometry "$TARGET_GEOMETRY"
"$PYTHON" "$EXP/replay-route.py" infer --output "$TRIALS" --work "$TRIAL_WORK" \
  --checkpoint "$CHECKPOINT" --tinygrad "$TINYGRAD" --noise "$INITIAL_NOISE" \
  --fp8-cache "$FP8_CACHE" --cache "$CACHE" --persistent-runtime --group-prompts --collect-garbage
"$PYTHON" "$EXP/camera_profiles.py" evaluate --source "$TRIALS" --output "$TRIALS/comparison.json"
```

Use `--study lens-combinations` for the second set of narrower front-camera crops.
That study treats the previous two turn windows as tuning data and reserves another turn window.
These windows belong to one previously inspected drive. They do not provide independent driving validation.

Image calibration uses recorded `rpyCalib` and `wideFromDeviceEuler` from before each historical frame.
Only the two device cameras have this extrinsic calibration. Side-camera mounting angles remain unmeasured.
Source lens projections remain nominal estimates. An identity-resampling control separates interpolation changes from image alignment changes.

The vehicle-history candidates use causal CAN speed and learned steering ratio, offset, stiffness, and roll through openpilot's vehicle model.
They replace planar history while retaining the measured vertical and tilt components.
This is a bicycle-model approximation. Steering-controller torque parameters are not added to the language prompt.
Future vehicle measurements appear only in evaluation. `check-calibration-trials.py` checks geometry, steering signs, and causal history.

### Full-drive calibration follow-up

Use `calibration_trials.py --study full-calibration` with the same preparation arguments to retain every valid baseline timestamp.
It prepares four variants: front-image calibration, NVIDIA target intrinsics, learned vehicle history, and combined calibration/history.
Run inference with the persistent command above. Each sample path is unique; timestamps repeat across variants.
Separate predictions by sample path and `camera_profile` before timestamp-based route comparisons.
Use `route_longitudinal.py --model NAME DIRECTORY` for each separated full-drive variant.
The short-study `camera_profiles.py evaluate` command requires tuning and held-out splits and does not evaluate the full-drive manifest.

The completed follow-up contains 1,001 predictions per variant at four selected timestamps per recorded second.
Each variant retains the baseline's available views: 730 six-camera, 209 four-camera, two five-camera, and 60 two-camera samples.
Navigation, side-camera pixels, model precision, solver steps, and evaluation references remain fixed.

| Input variant | Mean XY trajectory error | Endpoint error | Equal-distance lateral error | Speed MAE |
| --- | ---: | ---: | ---: | ---: |
| Original 120 / 30-degree front geometry | 2.290 m | 6.042 m | 1.260 m | 2.766 km/h |
| Calibrated front images | 1.793 m | 4.703 m | 0.626 m | 2.712 km/h |
| Calibrated fronts with NVIDIA target intrinsics | 1.942 m | 5.183 m | 0.850 m | 2.692 km/h |
| Learned vehicle history | 2.089 m | 5.460 m | 1.213 m | 2.310 km/h |
| Calibrated images and learned history | **1.621 m** | **4.169 m** | **0.595 m** | **2.251 km/h** |

Combined calibration/history improves mean trajectory error by 29.2%, lateral shape error by 52.7%, and speed MAE by 18.6%.
It improves 811 prediction windows and regresses 190. Outside the 32 short-study windows, mean trajectory error changes from 2.249 to 1.566 m.
The earlier reserved left-turn regression remains: 2.809 to 3.303 m. These results come from one previously inspected drive with retrospective navigation.
Slowing results are mixed: unmatched predictions decrease from 69 to 64 windows, but missed recorded slowing increases from 132 to 135.
These overlapping windows are not independent braking events. Recorded motion is an estimate, not independent ground truth.

All 128 overlapping short-study inputs match exactly, as do their generated token sequences.
Of these trajectories, 127 match exactly. One learned-history trajectory differs by 0.018 m mean point distance.
The combined run produced 4,004 predictions in 2,577 seconds, including loading and compilation: 1.554 predictions per second.
Input preparation and video rendering are excluded. This mixed-variant run is not a separate steady-throughput benchmark for each variant.
Earlier rendered videos retain their previous predictions. Render updated comparisons into a separate output directory.

The earlier 110 / 20-degree crop improved short tests but regressed full-drive mean error from 2.290 to 2.726 m.
It remains rejected. Combined calibration/history retains the nominal 120 / 30-degree geometry for further offline validation.
See [the aggregate result record](results/route-calibration-20261116.json) for metrics, checks, and evidence hashes.

For a selected custom lens configuration, use `camera_profiles.py prepare --profile-json "$PROFILE_JSON"`.
The JSON contains a `name` and camera-ID-to-operation `operations` object.
Numeric operations specify nominal horizontal FOV in degrees; `"nvidia"` selects the recorded target-intrinsics mapping.
Unchanged cameras retain their original prepared pixels. Changed cameras should use `--raw-frames` to crop before resizing.

### Speed, slowing, and path shape

```bash
"$PYTHON" "$EXP/route_longitudinal.py" --source "$BASELINE_OUTPUT" --logs "$LOGS" \
  --model alpamayo6 "$BASELINE_OUTPUT" --big-model "$BIG_OUTPUT" --trials "$TRIALS" \
  --output "$REPORT_OUTPUT/longitudinal.json" --recording-uuid "$RECORDING_UUID"
```

The logged small model is included automatically. Repeat `--model NAME DIRECTORY` for other Alpamayo runs.
All models use the same position-derived speed estimator. Native velocity heads and executed braking commands are not compared here.
Recorded CAN speed supplies the speed reference. Future CAN samples never enter model inputs.
Slowing means acceleration below -0.5 m/s² for at least 0.5 seconds after smoothing.
Report brake-pedal observations separately. Overlapping prediction windows are not independent braking events.

Equal-distance path errors compare geometry over the shared traveled distance, with coverage reported explicitly.
They supplement the original time-aligned trajectory error. Faster predictions alone do not establish better or safer driving.

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
