"""Time fresh input loading through decoded output, alternating real replay inputs."""
import argparse
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch


def local_module(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--camera-profile', choices=('openpilot','nvidia'), default='openpilot')
  parser.add_argument('--allow-missing-cameras', action='store_true', help='Omit unavailable side views; require both front histories')
  parser.add_argument('--collect-garbage', action='store_true', help='Collect retired input graphs between offline predictions')
  parser.add_argument('--freeze-warmup-gc', action='store_true', help='Experimental: exclude the warm model heap from subsequent cyclic-GC scans')
  parser.add_argument('--gc-interval', type=int, default=1)
  parser.add_argument('--minimum-cache-capacity', type=int, default=0, help='Keep one KV allocation across camera layouts')
  parser.add_argument('--validation-reference', type=Path, help='Replay saved controls before new predictions; reject changed tokens or paths')
  parser.add_argument('--samples', nargs='+', required=True)
  parser.add_argument('--mmap-inputs', action='store_true', help='Read frozen RGB arrays through copy-on-write file mappings')
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--runs', type=int, default=6)
  parser.add_argument('--device-timeout-ms', type=int, default=300000)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--initial-noise', type=Path, help='Exact float32 [1,64,2] diffusion noise from the reference')
  parser.add_argument('--max-tokens', type=int, default=256, help='Reasoning budget; missing trajectory anchors fail the run')
  parser.add_argument('--expert-kernels', action='store_true')
  parser.add_argument('--bf16-vision-layers', type=int, default=0,
                      help='Accuracy study: restore patch embedding and initial vision layers from the BF16 checkpoint')
  parser.add_argument('--bf16-expert-projections', action='store_true',
                      help='Accuracy study: restore expert input/output projections from the BF16 checkpoint')
  parser.add_argument('--packed-bf16-vision', action='store_true',
                      help='Experimental explicit BF16 matrix kernels for restored vision weights')
  parser.add_argument('--fast-quantize', action='store_true')
  parser.add_argument('--materialize-quant', action='store_true', help='Experimental materialized full-prompt activation reduction')
  parser.add_argument('--quantize-reduction', action='store_true', help='Checked BF16 activation maximum kernels, unchanged FP8 scales')
  parser.add_argument('--grouped-quantize-reduction', action='store_true', help='Experimental independent per-image maximum kernels')
  parser.add_argument('--packed-quantize', action='store_true', help='Experimental direct packed FP8 conversion for language prefill')
  parser.add_argument('--capture-prefill-scales', action='store_true', help='Diagnostic: collect dynamic FP8 activation scales for calibration')
  parser.add_argument('--prefill-calibration', type=Path, help='Experimental fixed per-layer prefill scales; requires separate accuracy validation')
  parser.add_argument('--capture-vision-scales', action='store_true', help='Diagnostic: accumulate dynamic scales for every encoded image on the GPU')
  parser.add_argument('--vision-calibration', type=Path, help='Experimental fixed vision scales; requires separate route accuracy validation')
  parser.add_argument('--trajectory-solver', choices=('euler','midpoint','heun'), default='euler',
                      help='Experimental two-evaluation solver alternatives; require --steps 2 and paired accuracy validation')
  parser.add_argument('--pad-prompt-to', type=int, help='Fixed prompt graph length; requires calibrated prefill scales and block decoding')
  parser.add_argument('--row-quantize', action='store_true', help='Experimental per-token prefill FP8 scales; changes quantization')
  parser.add_argument('--cache-resize', action='store_true')
  parser.add_argument('--reduced-vocab', type=Path)
  parser.add_argument('--decode-block', type=int, default=1)
  parser.add_argument('--draft-previous', action='store_true')
  parser.add_argument('--draft-attention', action='store_true', help='Experimental packed masked attention during draft verification')
  parser.add_argument('--draft-prefix', action='store_true', help='Experimental continuation from verified draft tokens after a mismatch')
  parser.add_argument('--draft-history', type=int, default=0, help='Experimental bounded complete-text drafts after a first-token mismatch')
  parser.add_argument('--history-full-only', action='store_true', help='Restart serial decoding if a historical complete-text draft is rejected')
  parser.add_argument('--draft-tile', type=int, nargs=2, default=(4,4), help='Draft matrix tile, in units of 16 rows and columns')
  parser.add_argument('--lookup-fallback', type=int, default=6, help='Serial token count when no historical continuation matches')
  parser.add_argument('--lookup-draft', action='store_true', help='Experimental target-verified drafts from previous text suffixes')
  parser.add_argument('--phrase-drafts', action='store_true', help='Verify generic phrase proposals only when observed text has no continuation')
  parser.add_argument('--lookup-carry', action='store_true', help='Reuse the current target token already returned by draft verification')
  parser.add_argument('--compact-draft-io', action='store_true', help='Cache token selection graphs and omit redundant GPU conversions')
  parser.add_argument('--direct-readback', action='store_true', help='Read small decoder results through the checked Chestnut BAR path')
  parser.add_argument('--compact-inputs', action='store_true')
  parser.add_argument('--cache-rotary-inputs', action='store_true', help='Keep exactly unchanged rotary arrays on the GPU')
  parser.add_argument('--coarse-timing', action='store_true', help='Measure end-to-end completion without fences between GPU stages')
  parser.add_argument('--byte-pixels', action='store_true', help='Lossless uint8 codes for the checked BF16 pixel palette')
  parser.add_argument('--selective-pixels', action='store_true', help='Upload only images missing from the exact vision cache')
  parser.add_argument('--prefill-kernels', action='store_true')
  parser.add_argument('--vision-cache', action='store_true')
  parser.add_argument('--vision-cache-capacity', type=int, help='Defaults to 48 front-only or 144 six-camera images')
  parser.add_argument('--group-vision', action='store_true')
  parser.add_argument('--vision-group-limit', type=int, default=0, help='Maximum new images per vision graph; zero means unlimited')
  parser.add_argument('--batched-vision', action='store_true')
  parser.add_argument('--image-gemm', action='store_true', help='Batch partial FP8 image tiles with independent bounds and scales')
  parser.add_argument('--packed-projector', action='store_true', help='Experimental packed image projector with independent image scales')
  parser.add_argument('--fused-qkv', action='store_true')
  parser.add_argument('--fused-norm', action='store_true')
  parser.add_argument('--llvm-wgp', action='store_true', help='Experimental LLVM WGP scheduling, unchanged arithmetic')
  parser.add_argument('--online-attention', action='store_true')
  parser.add_argument('--online-expert', action='store_true', help='Experimental masked online attention for diffusion')
  parser.add_argument('--gpu-metrics', action='store_true', help='Read SMU telemetry after each timed prediction')
  parser.add_argument('--pipeline-inputs', action='store_true', help='Overlap one CPU input preparation with current GPU inference')
  parser.add_argument('--late-inputs', action='store_true', help='Prepare the next input during diffusion to reduce queued input age')
  parser.add_argument('--warmup-graphs', action='store_true', help='Capture vision and draft graph variants during startup')
  parser.add_argument('--warm-current-shapes', action='store_true',
                      help='Warm bounded decoder, vision and upload graph shapes using only current input data')
  parser.add_argument('--profile-kernels', action='store_true', help='Diagnostic only: PROFILE=1 GPU timings by pipeline stage')
  parser.add_argument('--profile-host', action='store_true', help='Diagnostic only: profile measured host work; invalidates throughput')
  parser.add_argument('--vision-views', action='store_true')
  parser.add_argument('--verify-vision-cache', action='store_true')
  parser.add_argument('--verify-vision-warmup', action='store_true')
  parser.add_argument('--vision-batch', type=int, default=2)
  parser.add_argument('--pixel-scale', type=float, default=1.0,
                      help='Experimental image pixel budget; values below one require separate quality validation')
  parser.add_argument('--camera-pixel-scales', nargs=6, type=float,
                      help='Experimental scales: front-left, front-wide, front-right, rear-left, rear-right, front-tele')
  parser.add_argument('--direct-byte-inputs', action='store_true', help='Keep exact uint8 patches until the existing GPU palette expansion')
  parser.add_argument('--front-history-indices', help='Experimental comma-separated subset of the five front-history frames, including index 4')
  parser.add_argument('--history-feature-scale', type=float, default=1.0,
                      help='Experimental: compress older vision features while retaining full-resolution image encoding')
  parser.add_argument('--history-pixel-scale', type=float, default=1.0,
                      help='Experimental older-image pixel budget; newest image per camera retains its normal budget')
  parser.add_argument('--native-manifest', type=Path, help='Include native RGB frame loading and calibrated projection in preprocessing')
  parser.add_argument('--native-calibrations', type=Path)
  parser.add_argument('--native-frame-cache', type=Path)
  parser.add_argument('--narrow-crop', type=float, default=1.0,
                       help='Experimental central narrow-camera crop fraction, before the normal resize')
  parser.add_argument('--wide-equidistant-focal', type=float, default=0.0,
                      help='Experimental wide-camera equidistant focal length in source pixels; zero disables projection')
  parser.add_argument('--wide-target-fov', type=float, default=120.0)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(argv)
  if args.warm_current_shapes:
    if not (args.warmup >= 1 and args.lookup_draft and args.compact_inputs and args.selective_pixels and args.vision_cache):
      parser.error('Current-input warmup requires positive warmup, lookup decoding, compact selective inputs and vision caching')
    args.warmup_graphs = True
  native_args = (args.native_manifest, args.native_calibrations, args.native_frame_cache)
  if any(native_args) and not all(native_args):
    parser.error('Native preprocessing requires its manifest, calibration records, and native RGB cache')
  assert args.gc_interval > 0 and args.minimum_cache_capacity >= 0
  return args


def snapshot_sources(output):
  sources = {}
  source_dir = output.parent / (output.stem+'-sources')
  source_dir.mkdir(parents=True, exist_ok=True)
  for path in [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.c'), Path(__file__).parent.parent/'model.py']:
    content = path.read_bytes()
    (source_dir/path.name).write_bytes(content)
    sources[path.name] = hashlib.sha256(content).hexdigest()
  return sources, source_dir


def initialize(args, *, sources, source_dir):
  """Create the existing benchmark components without preparing or replaying inputs.

  Keep local_module calls in installer order so existing import hooks still apply.
  The caller owns source snapshots, startup execution, and subsequent scheduling.
  """
  import benchmark
  from tinygrad import Device
  from tinygrad.helpers import GlobalCounters
  from tinygrad.device import Compiled
  assert args.device_timeout_ms > 0
  Compiled.wait_timeout_ms = args.device_timeout_ms
  torch.set_num_threads(4)
  if args.llvm_wgp:
    import wgp_compiler
    wgp_compiler.install(Device[Device.DEFAULT])
  local_module('benchmark-fast')
  adapter = local_module('fast-inputs').FastInputs(args.checkpoint, pixel_scale=args.pixel_scale,
                                                 cache_resize=args.cache_resize, narrow_crop=args.narrow_crop,
                                                 wide_equidistant_focal=args.wide_equidistant_focal,
                                                   wide_target_fov=args.wide_target_fov, camera_profile=args.camera_profile,
                                                    camera_pixel_scales=args.camera_pixel_scales,
                                                     allow_missing_cameras=args.allow_missing_cameras,
                                                     history_pixel_scale=args.history_pixel_scale,
                                                     history_feature_scale=args.history_feature_scale,
                                                     direct_byte_inputs=args.direct_byte_inputs,
                                                     front_history_indices=None if args.front_history_indices is None else tuple(map(int,args.front_history_indices.split(','))))
  if args.direct_byte_inputs and not (args.compact_inputs and args.byte_pixels):
    raise ValueError('Direct byte inputs require compact inputs and byte-pixel GPU expansion')
  if args.history_feature_scale < 1 and (not args.vision_cache or args.verify_vision_cache or args.verify_vision_warmup):
    raise ValueError('Feature compression requires the vision cache and separate approximation validation')
  assert not args.vision_cache or args.vision_batch == 1
  assert not args.byte_pixels or args.compact_inputs
  assert not args.cache_rotary_inputs or args.compact_inputs
  assert not args.coarse_timing or (args.decode_block and not args.profile_kernels)
  assert not args.selective_pixels or (args.compact_inputs and args.vision_cache)
  assert args.max_tokens >= 64
  settings = SimpleNamespace(max_tokens=args.max_tokens, diffusion_steps=args.steps, vision_batch=args.vision_batch, greedy=True,
                             byte_pixels=args.byte_pixels,cache_rotary_inputs=args.cache_rotary_inputs,coarse_timing=args.coarse_timing)
  started = time.perf_counter()
  model = benchmark.EdgeModel(args.checkpoint, precision='fp8')
  model.minimum_cache_capacity = args.minimum_cache_capacity
  precision_restoration = None
  assert args.bf16_vision_layers >= 0
  if args.bf16_vision_layers or args.bf16_expert_projections:
    precision_restoration = local_module('restore-precision').restore(
      model, args.checkpoint, args.bf16_vision_layers, args.bf16_expert_projections)
  packed_bf16_vision = []
  if args.packed_bf16_vision:
    assert args.bf16_vision_layers > 0
    local_module('prefill-kernels').fix_dispatch_packet()
    packed_bf16_vision = local_module('bf16-linear').install(model)
  if args.fast_quantize or args.materialize_quant:
    local_module('fast-quantize').install(model, materialize=args.materialize_quant)
  assert not args.grouped_quantize_reduction or args.quantize_reduction
  if args.quantize_reduction:
    local_module('prefill-kernels').fix_dispatch_packet()
    local_module('quantize-reduction').install(model, grouped=args.grouped_quantize_reduction, row=args.row_quantize)
  vocab = local_module('reduced-vocab').install(model, args.reduced_vocab) if args.reduced_vocab else None
  if args.expert_kernels:
    print('Packed expert matrices:', local_module('expert-kernels').install(model), flush=True)
  if args.prefill_kernels:
    local_module('prefill-kernels').install(model)
  if args.fused_qkv:
    print('Fused QKV groups:',local_module('fused-qkv').install(model),flush=True)
  if args.online_attention or args.online_expert:
    local_module('prefill-kernels').fix_dispatch_packet()
    local_module('online-attention').install(model, expert=args.online_expert)
  if args.row_quantize:
    assert not args.fused_qkv, 'Per-token QKV scales need a separate fused implementation'
    local_module('row-quantize').install(model)
  if args.fused_norm:
    local_module('prefill-kernels').fix_dispatch_packet()
    local_module('fused-norm').install(model)
  if args.packed_quantize:
    assert args.quantize_reduction and not (args.row_quantize or args.fused_qkv or args.prefill_kernels)
    fixed_scales = json.loads(args.prefill_calibration.read_text())['scales'] if args.prefill_calibration else None
    if fixed_scales is not None:
      assert fixed_scales and all(np.isfinite(v) and v > 0 for v in fixed_scales.values())
    print('Direct packed prefill matrices:',local_module('quantize-packed').install(
      model, fixed_scales=fixed_scales, capture_scales=args.capture_prefill_scales),flush=True)
  assert not (args.capture_prefill_scales and args.prefill_calibration)
  assert not (args.capture_prefill_scales or args.prefill_calibration) or args.packed_quantize
  padding = None
  if args.pad_prompt_to is not None:
    assert args.pad_prompt_to > 0 and args.prefill_calibration and args.decode_block and args.compact_inputs
    padding = local_module('padded-prefill')
    padding.install(model)
  if args.trajectory_solver != 'euler':
    assert args.steps == 2
    local_module('trajectory-solver').install(model,args.trajectory_solver)
  input_class = local_module('compact-inputs').CompactInputs if args.compact_inputs else benchmark.NativeInputs
  assert not args.batched_vision or (args.vision_cache and args.group_vision)
  assert not args.packed_projector or args.batched_vision
  if args.packed_projector:
    local_module('projector-kernels').install(model)
  assert not args.image_gemm or args.batched_vision
  batch_encoder = local_module('batched-vision').install(model,image_gemm=args.image_gemm) if args.batched_vision else None
  vision_provenance = {'config_sha256':hashlib.sha256((Path(args.checkpoint)/'config.json').read_bytes()).hexdigest(),
                       'weight_index_sha256':hashlib.sha256((Path(args.checkpoint)/'model.safetensors.index.json').read_bytes()).hexdigest(),
                       'bf16_vision_layers':args.bf16_vision_layers}
  if args.capture_vision_scales or args.vision_calibration:
    if not (args.batched_vision and args.grouped_quantize_reduction):
      raise ValueError('Vision calibration requires batched vision with independent grouped activation scales')
    if args.capture_vision_scales and args.vision_calibration:
      raise ValueError('Capture dynamic vision scales before applying fixed scales')
    fixed_vision = None
    if args.vision_calibration:
      calibration = json.loads(args.vision_calibration.read_text())
      if calibration['checkpoint_provenance'] != vision_provenance:
        raise ValueError('Vision calibration checkpoint configuration or restored precision differs')
      fixed_vision = calibration['scales']
      expected = {f'vlm.model.visual.encoder.layers.{layer}/{index}'
                  for layer in range(args.bf16_vision_layers, model.config['vlm_config']['vision_config']['num_hidden_layers'])
                  for index in range(4)}
      if set(fixed_vision) != expected or not all(np.isfinite(v) and v > 0 for v in fixed_vision.values()):
        raise ValueError('Incomplete or invalid fixed vision scales')
    local_module('vision-calibration').install(model,fixed_scales=fixed_vision,capture=args.capture_vision_scales)
  cache_capacity = args.vision_cache_capacity if args.vision_cache_capacity is not None else (144 if args.camera_profile == 'nvidia' else 48)
  assert cache_capacity >= (22 if args.camera_profile == 'nvidia' else 10)
  vision_cache = local_module('vision-cache').VisionCache(model, capacity=cache_capacity, verify=args.verify_vision_cache, grouped=args.group_vision,
                                                         views=args.vision_views, batch_encoder=batch_encoder,
                                                         group_limit=args.vision_group_limit) if args.vision_cache else None
  model.synchronize()
  profiler = local_module('profile-kernels').Recorder(Device[Device.DEFAULT], benchmark) if args.profile_kernels else None
  decoder = local_module('draft-decode').DraftDecoder(model, args.decode_block) if args.draft_previous else None
  if args.lookup_draft:
    assert args.draft_previous
    decoder = local_module('lookup-decode').LookupDecoder(model)
    decoder.fallback_block = args.lookup_fallback
    decoder.carry_verified_token = args.lookup_carry
  if args.phrase_drafts:
    assert args.lookup_draft
    phrase_provenance = local_module('phrase-drafts').install(decoder,adapter.tokenizer,adapter.config.traj_ids['future_start'])
  assert not args.lookup_carry or args.lookup_draft
  assert not args.draft_attention or decoder is not None
  assert not args.draft_prefix or decoder is not None
  assert 0 <= args.draft_history <= 64
  assert not args.draft_history or (decoder is not None and not args.lookup_draft)
  assert not args.history_full_only or args.draft_history > 0
  assert not args.compact_draft_io or decoder is not None
  assert not args.direct_readback or decoder is not None
  assert not args.late_inputs or (args.pipeline_inputs and decoder is not None)
  if decoder:
    decoder.fast_attention = args.draft_attention
    decoder.keep_verified_prefix = args.draft_prefix
    decoder.history_limit = args.draft_history
    decoder.history_full_only = args.history_full_only
    decoder.tile = tuple(args.draft_tile)
    decoder.compact_io = args.compact_draft_io
    if args.direct_readback:
      decoder.readback = local_module('usb-readback').download
      decoder.direct_readback = True
  startup_active = [False]
  startup_events = {}
  startup = None
  if args.warm_current_shapes:
    def startup_record(name, **details):
      event = startup_events.setdefault(name, {'count':0})
      event['count'] += 1
      event['last'] = details
    startup = local_module('agent-pipeline-warmup').StartupWarmup(lambda:startup_active[0],startup_record,graph_budget=32)
    startup.patch_decoder(type(decoder))
    startup.patch_inputs(input_class)
    startup.patch_vision(type(vision_cache))
  runner = decoder.run if decoder else local_module('block-decode').BlockDecoder(model, args.decode_block).run
  report = {'model_load_ms': (time.perf_counter()-started)*1000, 'arch': Device[Device.DEFAULT].arch,
              'prefill_calibration_sha256': hashlib.sha256(args.prefill_calibration.read_bytes()).hexdigest() if args.prefill_calibration else None,
               'pad_prompt_to': args.pad_prompt_to,
               'trajectory_solver': args.trajectory_solver,
               'capture_prefill_scales': args.capture_prefill_scales,
               'capture_vision_scales': args.capture_vision_scales,
               'vision_calibration_sha256': hashlib.sha256(args.vision_calibration.read_bytes()).hexdigest() if args.vision_calibration else None,
               'precision_restoration': precision_restoration,
               'packed_bf16_vision': packed_bf16_vision,
               'packed_quantize': args.packed_quantize,
              'camera_profile': args.camera_profile, 'vision_cache_capacity': cache_capacity, 'mmap_inputs': args.mmap_inputs,
              'device_timeout_ms': args.device_timeout_ms,
             'source_sha256': sources, 'source_snapshot': str(source_dir),
              'runtime_environment': {key:os.environ.get(key) for key in ('DEV','GMMU','AM_POWER_LIMIT','AM_PERF_LEVEL','TC_OPT',
                                           'ALPAMAYO_ONLINE_WAVES','ALPAMAYO_ONLINE_BLOCK_KEYS','ALPAMAYO_ONLINE_STAGE_KV','ALPAMAYO_ONLINE_WAVE_SYNC','ALPAMAYO_HIP_WGP','ALPAMAYO_ONLINE_NATIVE_EXP','ALPAMAYO_ONLINE_REGISTER_PROB',
                                            'ALPAMAYO_ONLINE_PACKED_K','ALPAMAYO_ONLINE_KEY_PADDING','ALPAMAYO_ONLINE_PROB_PADDING','ALPAMAYO_ONLINE_LOCAL_FENCES','ALPAMAYO_ONLINE_DPP','ALPAMAYO_ONLINE_KV_LOAD',
                                            'ALPAMAYO_ONLINE_VALUE_LAYOUT','ALPAMAYO_ONLINE_VALUE_MIN_ROWS','ALPAMAYO_ONLINE_PROBABILITY_MMA',
                                        'ALPAMAYO_EXPERT_WAVES')},
             'settings': vars(settings), 'expert_kernels': args.expert_kernels, 'compact_inputs': args.compact_inputs,
              'selective_pixels': args.selective_pixels,
               'pipeline_inputs': args.pipeline_inputs,
               'late_inputs': args.late_inputs,
               'warmup_graphs': args.warmup_graphs,
               'warm_current_shapes': args.warm_current_shapes,
               'first_measured_input_after_startup': args.warm_current_shapes or args.freeze_warmup_gc,
               'current_shape_startup_events': startup_events,
             'prefill_kernels': args.prefill_kernels,
               'fast_quantize': args.fast_quantize,
                 'materialize_quant': args.materialize_quant,
                 'quantize_reduction': args.quantize_reduction,
                 'grouped_quantize_reduction': args.grouped_quantize_reduction,
                'row_quantize': args.row_quantize,
              'cache_resize': args.cache_resize,
             'reduced_vocab': vocab,
              'decode_block': args.decode_block,
               'draft_previous': args.draft_previous,
               'draft_attention': args.draft_attention,
                 'draft_prefix': args.draft_prefix,
                  'draft_history': args.draft_history,
                  'history_full_only': args.history_full_only,
                'draft_tile': args.draft_tile,
                'lookup_fallback': args.lookup_fallback,
                 'lookup_draft': args.lookup_draft,
                  'lookup_carry': args.lookup_carry,
                  'phrase_drafts': phrase_provenance if args.phrase_drafts else None,
                 'compact_draft_io': args.compact_draft_io,
                 'direct_readback': args.direct_readback,
             'vision_cache': args.vision_cache,
                'group_vision': args.group_vision,
                'vision_group_limit': args.vision_group_limit,
                  'batched_vision': args.batched_vision,
                  'image_gemm': args.image_gemm,
                 'packed_projector': args.packed_projector,
                 'fused_qkv': args.fused_qkv,
                  'fused_norm': args.fused_norm,
                  'llvm_wgp': args.llvm_wgp,
               'online_attention': args.online_attention,
               'online_expert': args.online_expert,
                'profiling': args.profile_kernels or args.profile_host,
              'vision_views': args.vision_views,
             'verify_vision_cache': args.verify_vision_cache,
             'verify_vision_warmup': args.verify_vision_warmup,
                'pixel_scale': args.pixel_scale, 'camera_pixel_scales': args.camera_pixel_scales, 'pixel_bounds': adapter.pixel_bounds,
                 'history_pixel_scale': args.history_pixel_scale,
                 'history_feature_scale': args.history_feature_scale,
                 'direct_byte_inputs': args.direct_byte_inputs,
                 'front_history_indices': args.front_history_indices,
               'allow_missing_cameras': args.allow_missing_cameras,
                'collect_garbage': args.collect_garbage,
                'gc_interval': args.gc_interval, 'minimum_cache_capacity': args.minimum_cache_capacity,
              'narrow_crop': args.narrow_crop,
              'wide_equidistant_focal': args.wide_equidistant_focal,
              'wide_target_fov': args.wide_target_fov,
             'timing_boundary': 'RGB sample file load through decoded trajectory; camera capture and video decoding excluded',
             'warmup': [], 'runs': []}
  if args.initial_noise:
    settings.initial_noise = np.load(args.initial_noise, allow_pickle=False)
    assert settings.initial_noise.shape == (1,64,2) and settings.initial_noise.dtype == np.float32
    assert np.isfinite(settings.initial_noise).all()
    # Keep the binary tensor out of the JSON settings while retaining provenance.
    report['settings'] = {key:value for key,value in vars(settings).items() if key != 'initial_noise'}
    report['initial_noise_sha256'] = hashlib.sha256(settings.initial_noise.tobytes()).hexdigest()
  return SimpleNamespace(adapter=adapter, model=model, settings=settings, input_class=input_class,
                         config=adapter.config, tokenizer=adapter.tokenizer, vision_cache=vision_cache,
                         decoder=decoder, runner=runner, pad_inputs=padding.pad_inputs if padding else None,
                         startup=startup, startup_active=startup_active, startup_events=startup_events,
                         profiler=profiler, report=report, provenance=copy.deepcopy(report),
                         vision_provenance=vision_provenance, benchmark=benchmark, Device=Device,
                         GlobalCounters=GlobalCounters, inputs=None)


def main():
  process_started = time.perf_counter()
  args = parse_args()
  native_args = (args.native_manifest, args.native_calibrations, args.native_frame_cache)
  sources, source_dir = snapshot_sources(args.output)
  components = initialize(args, sources=sources, source_dir=source_dir)
  adapter, model, settings = components.adapter, components.model, components.settings
  input_class, vision_cache = components.input_class, components.vision_cache
  decoder, runner, profiler = components.decoder, components.runner, components.profiler
  startup_active, report = components.startup_active, components.report
  vision_provenance = components.vision_provenance
  benchmark, Device, GlobalCounters = components.benchmark, components.Device, components.GlobalCounters
  inputs = None
  validation = json.loads(args.validation_reference.read_text()) if args.validation_reference else {}
  validation_count = len(validation)
  paths = list(validation) + [args.samples[i % len(args.samples)] for i in range(args.warmup+args.runs)]
  report['validation'] = []
  native_frames = local_module('stream-inputs').NativeFrames(*native_args) if args.native_manifest else None
  report['native_preprocessing'] = ({'manifest':str(args.native_manifest), 'calibrations':str(args.native_calibrations),
                                    'raw_cache':str(args.native_frame_cache),
                                    'manifest_sha256':hashlib.sha256(args.native_manifest.read_bytes()).hexdigest(),
                                    'calibrations_sha256':hashlib.sha256(args.native_calibrations.read_bytes()).hexdigest()}
                                   if native_frames else None)
  if native_frames:
    report['timing_boundary'] = 'Native RGB frame load, calibration, model preprocessing, transfers, inference, and decoded trajectory; camera ISP and CAN history construction excluded'
  progress_path = args.output.with_suffix('.jsonl')
  progress_path.write_text('')
  def prepare_input(path):
    start = time.perf_counter()
    native_timing = {}
    if Path(path).is_dir():
      data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
              else torch.from_numpy(np.load(p, allow_pickle=False, mmap_mode='c' if args.mmap_inputs else None))
              for p in Path(path).glob('*.npy') if not (native_frames and p.stem == 'image_frames')}
      if native_frames:
        pixels, native_timing = native_frames.prepare(path)
        data['image_frames'] = torch.from_numpy(pixels)
    else:
      data = benchmark.load_sample(path)
    loaded = time.perf_counter()
    config, tokenizer, prepared = adapter.prepare(data)
    packed = adapter.native_inputs(prepared)
    if args.pad_prompt_to is not None:
      prepared,packed=local_module('padded-prefill').pad_inputs(prepared,packed,args.pad_prompt_to)
    # Pure CPU work can overlap inference; GPU buffers and cache state stay on the main thread.
    arrays = input_class.arrays(settings, prepared, packed) if args.pipeline_inputs and args.compact_inputs else None
    keys = vision_cache.input_keys(prepared) if args.pipeline_inputs and vision_cache else None
    return data, config, tokenizer, prepared, packed, arrays, keys, start, loaded, time.perf_counter(), native_timing
  # A separate tokenizer avoids concurrent Rust tokenizer mutation during decode.
  decode_tokenizer = copy.deepcopy(adapter.tokenizer) if args.pipeline_inputs else None
  pool = ThreadPoolExecutor(max_workers=1) if args.pipeline_inputs else None
  pending = pool.submit(prepare_input, paths[0]) if pool else None
  measured_begin = None
  captured_scales = {}
  if args.profile_host:
    import cProfile
    host_profile = cProfile.Profile()
  for index, path in enumerate(paths):
    iteration = index-validation_count
    startup_active[0] = iteration < args.warmup
    if args.freeze_warmup_gc and iteration == args.warmup:
      import gc
      freeze_started = time.perf_counter()
      gc.collect()
      gc.freeze()
      frozen_objects = gc.get_freeze_count()
      report['warmup_gc_freeze'] = {'seconds':time.perf_counter()-freeze_started, 'frozen_objects':frozen_objects,
                                  'limit':'One process-lifetime freeze; validate retained memory across prompt and camera-layout changes'}
    iteration_started = time.perf_counter()
    if args.collect_garbage and index % args.gc_interval == 0:
      # Selective upload graphs capture their input owner. Collect retired
      # cycles before the next GPU allocation, including prompt shape changes.
      import gc
      gc.collect()
    collected = time.perf_counter()
    if profiler:
      profiler.reset()
    if iteration == args.warmup:
      measured_begin = time.perf_counter()
      report['startup_before_measured_input_s'] = measured_begin-process_started
    if args.profile_host and iteration >= args.warmup:
      host_profile.enable()
    data, config, tokenizer, prepared, packed, arrays, keys, start, loaded, cpu_ready, native_timing = pending.result() if pending else prepare_input(path)
    if pool:
      tokenizer = decode_tokenizer
      next_path = paths[index+1] if index+1 < len(paths) else None
      # Start the first measured input after all startup work, including GC.
      # Subsequent inputs retain the same single-item overlap and timestamps.
      if (args.warm_current_shapes or args.freeze_warmup_gc) and iteration+1 == args.warmup:
        next_path = None
      if args.late_inputs:
        pending = None
        def queue_next(path=next_path):
          nonlocal pending
          assert pending is None
          if path is not None:
            pending = pool.submit(prepare_input, path)
        decoder.before_diffusion = queue_next
      else:
        pending = pool.submit(prepare_input, next_path) if next_path is not None else None
    selected = time.perf_counter()
    if vision_cache:
      vision_cache.warmup_graphs = args.warmup_graphs and iteration < args.warmup
      vision_cache.verify = args.verify_vision_cache or (args.verify_vision_warmup and iteration < max(1,args.warmup-4))
      vision_cache.set_layout(prepared)
      if keys is None:
        vision_cache.prepare(prepared)
      else:
        vision_cache.keys = keys
    if decoder:
      decoder.warmup_graphs = args.warmup_graphs and iteration < args.warmup
    preprocessed = time.perf_counter()
    if not (args.compact_inputs and inputs is not None and inputs.update(settings, config, prepared, packed, arrays=arrays)):
      fresh = input_class(settings, config, prepared, packed, **({'arrays':arrays} if args.compact_inputs else {}))
      if (inputs is not None and inputs.batches == fresh.batches and inputs.spans == fresh.spans
          and getattr(inputs, 'layout_key', None) == getattr(fresh, 'layout_key', None)
          and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(inputs.cpu, fresh.cpu, strict=True))):
        for old, new in zip(inputs.cpu, fresh.cpu, strict=True):
          old._buffer().as_memoryview(allow_zero_copy=True)[:] = new._buffer().as_memoryview(allow_zero_copy=True)
      else:
        inputs = fresh
      del fresh
    if args.selective_pixels:
      inputs.selective_cache = vision_cache
      if args.warmup_graphs and iteration < args.warmup and len(inputs.batches) in (10, 22):
        inputs.warmup_selections()
    staged = time.perf_counter()
    prediction = runner(model, settings, config, tokenizer, prepared, packed, seed=0, inputs=inputs)
    done = time.perf_counter()
    if index == 0:
      report['startup_through_first_prediction_s'] = done-process_started
    if args.profile_host and iteration >= args.warmup:
      host_profile.disable()
    entry = {'sample': path, 'load_ms': (loaded-start)*1000,
               'preprocess_ms': ((cpu_ready-loaded)+(preprocessed-selected))*1000,
               'prepared_queue_ms': (selected-cpu_ready)*1000,
                 'cpu_buffers_ms': (staged-preprocessed)*1000, 'pipeline_ms': (done-start)*1000,
                  'service_ms': (done-iteration_started)*1000, 'prediction': prediction}
    entry['gc_ms'] = (collected-iteration_started)*1000
    entry.update(native_timing)
    entry['allocated_bytes_by_device'] = dict(GlobalCounters.mem_used_per_device)
    if vision_cache:
      entry['vision_reference_checks'] = vision_cache.reference_checks
    entry['input_latest_timestamp_us'] = int(data['absolute_timestamps'].max().item())
    if vision_cache:
      entry.update(vision_hits=vision_cache.hits, vision_misses=vision_cache.misses)
    if args.selective_pixels:
      entry['uploaded_pixel_bytes'] = inputs.uploaded_pixel_bytes
    if decoder and decoder.statistics:
      entry['draft'] = decoder.statistics[-1]
    if args.gpu_metrics:
      entry['gpu_metrics'] = local_module('gpu-metrics').read(Device[Device.DEFAULT])
    if profiler:
      entry['device_profile'] = profiler.report()
    if args.capture_prefill_scales:
      from tinygrad import Tensor
      keys = sorted(model.prefill_activation_scales)
      values = Tensor.stack(*(model.prefill_activation_scales[k].reshape(()) for k in keys)).numpy().tolist()
      if len(keys) != 4*model.config['vlm_config']['text_config']['num_hidden_layers']:
        raise ValueError('Incomplete language activation-scale capture')
      if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('Invalid activation scale')
      entry['prefill_activation_scales'] = dict(zip(keys, values, strict=True))
      for key, value in entry['prefill_activation_scales'].items():
        captured_scales[key] = max(captured_scales.get(key, 0.), value)
    if iteration < 0:
      expected = validation[path]['prediction']
      difference = np.abs(np.asarray(prediction['xyz'])-np.asarray(expected['xyz']))
      check = {'sample': path, 'max_xyz_difference_m': float(difference.max()),
               'mean_xyz_difference_m': float(difference.mean()),
               'tokens_equal': prediction['generated_token_ids'] == expected['generated_token_ids']}
      report['validation'].append(check)
      args.output.with_suffix('.validation.json').write_text(json.dumps(report['validation'], indent=2)+'\n')
      if not check['tokens_equal'] or difference.max() > 0.15 or difference.mean() > 0.025:
        raise ValueError(f'Persistent replay changed a saved control: {check}')
      print('Validated saved prediction: '+json.dumps(check), flush=True)
      continue
    report['warmup' if iteration < args.warmup else 'runs'].append(entry)
    with progress_path.open('a') as stream:
      stream.write(json.dumps({'iteration':iteration,'warmup':iteration < args.warmup, **entry})+'\n')
    print(f'Iteration {iteration}: {entry["pipeline_ms"]:.2f} ms total; {prediction["reasoning"]}', flush=True)
  measured_elapsed = time.perf_counter()-measured_begin
  if pool:
    pool.shutdown(wait=True)
  totals = [r['pipeline_ms'] for r in report['runs']]
  timestamps = [r['input_latest_timestamp_us'] for r in report['warmup']+report['runs']]
  report['summary'] = {'median_ms': float(np.median(totals)), 'p95_ms': float(np.percentile(totals,95)),
                       'max_ms': max(totals), 'throughput_fps': len(totals)/measured_elapsed,
                        'measured_wall_seconds': measured_elapsed,
                        'inverse_mean_latency_fps': 1000/float(np.mean(totals)),
                        'predictions_over_250ms': sum(t > 250 for t in totals), 'measured_predictions': len(totals),
                        'all_inputs_strictly_advance': all(b > a for a,b in zip(timestamps,timestamps[1:])),
                             'profiling_enabled': args.profile_kernels or args.profile_host or args.capture_prefill_scales or args.capture_vision_scales}
  services = [r['service_ms'] for r in report['runs']]
  report['summary'].update(service_median_ms=float(np.median(services)), service_p95_ms=float(np.percentile(services,95)),
                           input_to_output_over_100ms=sum(t > 100 for t in totals),
                           input_to_output_over_200ms=sum(t > 200 for t in totals),
                           service_over_100ms=sum(t > 100 for t in services), service_over_200ms=sum(t > 200 for t in services),
                           measured_inputs_strictly_advance=all(b['input_latest_timestamp_us'] > a['input_latest_timestamp_us']
                                                               for a,b in zip(report['runs'],report['runs'][1:])))
  if args.capture_prefill_scales:
    args.output.with_suffix('.prefill-scales.json').write_text(json.dumps({
      'schema': 1, 'method': 'Maximum observed dynamic tensor-wide FP8 scale per prefill layer/input',
      'scales': captured_scales, 'samples': list(dict.fromkeys(paths)),
      'checkpoint': args.checkpoint, 'source_sha256': sources,
       'limit': 'Calibration changes numerical behavior. Validate on separate route windows before adoption.'}, indent=2)+'\n')
  if args.capture_vision_scales:
    from tinygrad import Tensor
    keys = sorted(model.vision_activation_scales)
    expected_count = 4*(model.config['vlm_config']['vision_config']['num_hidden_layers']-args.bf16_vision_layers)
    if len(keys) != expected_count:
      raise ValueError(f'Incomplete vision calibration: {len(keys)} scales instead of {expected_count}')
    values = Tensor.cat(*(model.vision_activation_scales[k] for k in keys)).numpy().tolist()
    if not all(np.isfinite(v) and v > 0 for v in values):
      raise ValueError('Invalid accumulated vision scale')
    args.output.with_suffix('.vision-scales.json').write_text(json.dumps({
      'schema':1, 'method':'Maximum dynamic per-image scale accumulated during every encoded image, including graph replays',
      'scales':dict(zip(keys,values,strict=True)), 'checkpoint_provenance':vision_provenance,
      'pixel_bounds':adapter.pixel_bounds, 'samples':list(dict.fromkeys(paths)), 'source_sha256':sources,
      'limit':'Experimental numerical change. Camera and route holdout validation is required; file hashes do not verify weight contents.'
    },indent=2)+'\n')
  report['process_elapsed_s'] = time.perf_counter()-process_started
  if args.profile_host:
    host_profile.dump_stats(str(args.output.with_suffix('.pstats')))
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(f'Full pipeline median {np.median(totals):.2f} ms; p95 {np.percentile(totals,95):.2f} ms', flush=True)


if __name__ == '__main__':
  main()
