# Alpamayo planner

This branch drives with plans from NVIDIA Alpamayo 2 Edge on an external AMD GPU.
The standard openpilot model keeps running for perception, camera odometry and driver monitoring. It never steers on this branch.

This is research software. It is not a safety-rated driving system. Supervise it at all times.
openpilot engagement rules, vehicle-specific controls and Panda safety stay in force.

## Requirements

| Item | Requirement |
| --- | --- |
| Device | Asius v0 with VamOS |
| GPU | Chestnut with an AMD RX 9060 or RX 9060 XT (`gfx1200`), connected at 5 Gbps USB |
| Storage | 6 GB free on `/data` |
| First start | Parked, with network access |

Other GPUs do not work: the GPU kernels are compiled for `gfx1200` only.

## Install

1. Check out this branch on the device and update the submodules:

   ```sh
   cd /data/openpilot
   git fetch origin alpamayo
   git checkout alpamayo
   git submodule update --init --recursive
   sudo reboot
   ```

2. Keep the car parked with network access. openpilot builds, then `alpamayo_install` runs by itself.
3. Wait until the Asius app no longer shows `Alpamayo install`. The GPU worker then loads the model.

The installer does these steps. Each step checks its result and continues after an interruption.

| Step | Content | Size |
| --- | --- | --- |
| Runtime | Pinned tinygrad and NVIDIA sources with locked Python packages, in their own environment | 1.1 GB |
| Model | The `lat2v` fine-tune from [Hugging Face](https://huggingface.co/karelnagel/alpamayo2-edge-lat2v), each file checked by SHA-256 | 3.1 GB |
| GPU firmware | Eight pinned AMD firmware files from `linux-firmware` | 5 MB |
| Compiled GPU graphs | Graphs and kernels for this code, model and `gfx1200` | 0.1 GB |
| Configuration | `/data/alpamayo/config.json` | - |

Without the compiled graphs the first start compiles for 10 to 20 minutes. Any change to `selected/*.py`, `model.py` or
`amd_attention.py` makes the published graphs unusable; the device then compiles its own.

## Operation

- `launch_env.sh` sets `ALPAMAYO_CONFIG=/data/alpamayo/config.json` and `ALPAMAYO_KEEP_WARM=1`.
- With `ALPAMAYO_KEEP_WARM=1` the GPU worker loads the model while parked and stays loaded between drives.
- The worker predicts on a 100 ms slot grid. Each plan uses five frames per camera at -2, -1, -0.5, -0.1 and 0 s.
- A plan is usable when it was ready within 750 ms of its camera frame. It drives while that frame is under 1.5 s old.
- Two consecutive plans at most 1 s apart are necessary before the first use.
- modeld tracks the plan path with odometry and pure pursuit. It uses the plan's steering and acceleration without gains.
- Without a usable plan, `selfdrived` raises `alpamayoUnavailable`: no entry, or a soft disable when engaged.
- Navigation text is `Continue straight.` without a route. A route maneuver replaces it when it is at most 150 m ahead.
- The Asius app shows the model's reasoning text and the GPU state.

## Settings

Environment variables, set before `launch_openpilot.sh` (for example in `/data/continue.sh`):

| Variable | Effect |
| --- | --- |
| `ALPAMAYO_SHADOW=1` | Compute and record plans only. The standard model drives. |
| `ALPAMAYO_PLANNER=0` | Turn Alpamayo off. The standard model drives. |
| `ALPAMAYO_KEEP_WARM=0` | Load the model only while onroad. The load then takes part of the drive. |
| `ALPAMAYO_CONFIG=/path` | Use your own configuration file. The installer then does nothing. |

To change single settings, write them to `/data/alpamayo/overrides.json`.
The installer applies the file while parked, then the GPU worker restarts. Example:

```json
{"inference": {"power_limit_w": 144}, "navigation_within_m": 100}
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `inference.diffusion_steps` | 2 | Euler steps, 1 to 10. The model is trained for 2. |
| `inference.power_limit_w` | 100 | GPU power limit, 100 to 144 W. |
| `period_ms` | 100 | Slot period: 100, 250, 500, or 0 for free running. |
| `default_navigation` | `Continue straight.` | Navigation text without a near maneuver. |
| `navigation_within_m` | 150 | A maneuver farther ahead is not sent to the model. |
| `output_limit_gb` | 2 | Limit for stored prediction records. Older sessions are deleted at start. |
| `geometry` | Asius v0 lenses | Nominal lens models of the wide and narrow cameras. |

## Files on the device

| Path | Content |
| --- | --- |
| `/data/alpamayo/runtime` | Isolated tinygrad and NVIDIA sources and their Python environment |
| `/data/alpamayo/models/lat2v` | Compact BF16 checkpoint, FP8 weights, activation scales, zero initial noise |
| `/data/alpamayo/cache` | GPU firmware, compiled graphs and kernels |
| `/data/alpamayo/output` | One directory per session: worker log, status and prediction records |
| `/data/alpamayo/status.json` | Installer progress |

To install from the start, stop openpilot and remove `/data/alpamayo`.

## Model

`release.json` pins the model repository, its revision and the SHA-256 of every file.
`selected/assets/artifacts.json` pins the identities that the runtime accepts at startup.
The runtime uses zero initial diffusion noise, greedy reasoning and at most 64 reasoning tokens.
[selected/README.md](selected/README.md) describes the numerical recipe and the GPU kernels.

To run another fine-tune, publish its files in the same layout, pin it in both files, and compile its graphs once on a device.

## Limits

- Tested on one Tesla Model 3 with one device. On straight roads it held the lane. In turns it started to turn too early.
- Night, rain, snow and dense traffic are not tested.
- The plan rate is about 5 per second and a plan is about 0.2 s old when it arrives.
- The GPU can leave the USB bus. The worker restarts up to three times; then openpilot must restart.
- The GNSS receiver can lose its fix near the GPU and its cable.

## Licenses

The model, input adapter and AMD kernels derive from comma's Alpamayo prototype (MIT, see the repository [LICENSE](../../../LICENSE)).
NVIDIA's package has the Apache-2.0 license. The model weights have the OpenMDW-1.1 license.
[selected/NOTICE.md](selected/NOTICE.md) and `selected/licenses/` hold the notices.
