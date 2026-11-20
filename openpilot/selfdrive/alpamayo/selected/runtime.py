"""Selected inference API. The shared runtime owns live/replay scheduling and logging."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import TypedDict
from .frozen import ASSETS, TINYGRAD_REVISION, manifest, verify_assets


@dataclass(frozen=True)
class Config:
  checkpoint: Path
  fp8_cache: Path
  prefill_calibration: Path
  cache_dir: Path
  initial_noise: Path | None = None
  device_timeout_ms: int = 10000
  native_fresh_patches: bool = False

  def __post_init__(self):
    for name in ('checkpoint','fp8_cache','prefill_calibration','cache_dir','initial_noise'):
      if (value := getattr(self,name)) is not None:
        object.__setattr__(self,name,Path(value).resolve())
    if self.device_timeout_ms <= 0:
      raise ValueError('Device timeout must be positive')
    if type(self.native_fresh_patches) is not bool:
      raise ValueError('native_fresh_patches must be boolean')


@dataclass
class Metrics:
  stages_ms: dict = field(default_factory=dict)
  runtime_ms: float = 0
  vision_hits: int = 0
  vision_misses: int = 0
  uploaded_pixel_bytes: int = 0
  speculation: dict = field(default_factory=dict)


class Prediction(TypedDict):
  xyz: list
  rot: list
  generated_token_ids: list[int]
  generated_tokens: int
  reasoning: str
  timings_ms: dict[str,float]
  runtime_ms: float
  prompt_tokens: int
  prefill_tokens: int
  noise_bytes_sha256: str


def environment(cache_dir):
  values = json.loads((ASSETS/'environment.json').read_text())
  # The compilation pool uses spawn. Avoid importing 32 compiler interpreters on four available Dragon cores.
  cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
  values['PARALLEL'] = str(min(int(values['PARALLEL']), cores))
  values.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(cache_dir),
                CACHEDB=str(Path(cache_dir)/'tinygrad.db'))
  if 'tinygrad' in sys.modules and any(os.environ.get(k) != v for k,v in values.items()):
    raise RuntimeError('Configure selected inference in a fresh dedicated process before importing tinygrad')
  for key in list(os.environ):
    if key.startswith('ALPAMAYO_'):
      del os.environ[key]
  os.environ.update(values)
  return values


def require_firmware(cache):
  from tinygrad.runtime.autogen.am import fw
  names = ['psp_14_0_2_sos.bin','smu_14_0_2.bin','sdma_7_0_0.bin',
           *[f'gc_12_0_0_{kind}.bin' for kind in ('pfp','me','mec','imu','rlc')]]
  # Derive the required names from the pinned backend, not from a network request.
  for name in names:
    if name not in fw.hashes:
      raise ValueError(f'Pinned firmware inventory changed: {name}')
    url = f'https://gitlab.com/kernel-firmware/linux-firmware/-/raw/0a6871b19abf5d6e024b5d208b101ae53e7fa0de/amdgpu/{name}'
    path = Path(cache)/'tinygrad/downloads/fw'/hashlib.md5(url.encode()).hexdigest()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != fw.hashes[name]:
      raise ValueError(f'Stage exact gfx1200 firmware before startup: {name}')


def artifact_identities(config):
  from openpilot.selfdrive.alpamayo.inference import sha
  expected = json.loads((ASSETS/'artifacts.json').read_text())
  paths = {'checkpoint_config_sha256':config.checkpoint/'config.json',
           'checkpoint_weight_index_sha256':config.checkpoint/'model.safetensors.index.json',
           'fp8_index_sha256':config.fp8_cache/'index.json','prefill_calibration_sha256':config.prefill_calibration}
  identities = {key:sha(path) for key,path in paths.items()}
  index_key = 'checkpoint_weight_index_sha256'
  compact = identities[index_key] == expected['compact_checkpoint']['index_sha256']
  if any(identities[key] != expected[key] for key in paths if key != index_key or not compact):
    raise ValueError('Selected checkpoint, FP8 index, or fixed calibration differs from the pinned recipe')
  index = json.loads(paths[index_key].read_text())['weight_map']
  shards = {name:sha(config.checkpoint/name) for name in sorted(set(index.values()))}
  if shards != (expected['compact_checkpoint']['shards'] if compact else expected['checkpoint_shards']):
    raise ValueError('BF16 checkpoint shard bytes changed')
  # Both pinned representations derive from the same checkpoint. Record exactly which files this process verified.
  identities.update(loaded_checkpoint_index_sha256=identities[index_key], loaded_checkpoint_shards=shards,
                    checkpoint_representation='selected-bf16-subset' if compact else 'original')
  original = expected['checkpoint_shards']
  identities.update(checkpoint_weight_index_sha256=expected[index_key], checkpoint_shards=original,
                    checkpoint_weights_sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest())
  if config.initial_noise is not None:
    identities['initial_noise_sha256'] = sha(config.initial_noise)
  return expected,identities


class Runtime:
  """Persistent selected backend; pass explicit float32 noise to predict for replay."""
  def __init__(self, config):
    self.config = config if isinstance(config,Config) else Config(**config)
    self.closed,self.backend = False,None
    self.lock = threading.Lock()
    self.stage_sync = True
    self.prime_done = 0
    self.metrics = Metrics()
    self.startup_ms = {}
    previous = time.monotonic_ns()
    def finished(stage):
      nonlocal previous
      now = time.monotonic_ns(); self.startup_ms[stage] = (now-previous)/1e6; previous = now
      print(f'Alpamayo startup: {stage} {self.startup_ms[stage]:.0f} ms', file=sys.stderr, flush=True)
    environment(self.config.cache_dir)
    from openpilot.selfdrive.alpamayo.inference import verify_runtime, NVIDIA_REVISION
    verify_runtime()
    import numpy as np
    import torch
    from tinygrad import Device
    from .backend import Backend
    from .buffers import VisionBuffers
    from .inputs import SelectedInputs
    from .model import SelectedModel
    from .prefill import SpeculativePrefill
    from .decode import Decoder
    from .vision import VisionCache
    from .policy import CHECKPOINT_DRAFT_IDS, DRAFT_TEXT, validate_draft
    finished('runtime_imports')
    verify_assets()
    artifacts,identities = artifact_identities(self.config)
    require_firmware(self.config.cache_dir)
    finished('asset_verification')
    torch.set_num_threads(2)
    scales = json.loads(self.config.prefill_calibration.read_text())['scales']
    self.noise = None if self.config.initial_noise is None else np.load(self.config.initial_noise,allow_pickle=False)
    if self.noise is not None:
      self.validate_noise(self.noise)
      self.noise.setflags(write=False)
    # The tokenizer and input adapter load on the CPU while the weights stream over USB.
    adapter = {}
    def load_adapter():
      try:
        adapter['value'] = SelectedInputs(self.config.checkpoint, native_fresh_patches=self.config.native_fresh_patches)
      except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
        adapter['error'] = exc
    loader = threading.Thread(target=load_adapter, name='alpamayo-adapter', daemon=True)
    try:
      self.backend = Backend(self.config.device_timeout_ms)
      finished('gpu_backend')
      loader.start()  # after GPU init, which is CPU-bound; the weight upload below mostly waits on I/O
      self.model = SelectedModel(self.config.checkpoint,self.config.fp8_cache,scales)
      finished('weights_and_packing')
      print(f'Alpamayo startup: weight stages {self.model.load_ms}', file=sys.stderr, flush=True)
      loader.join()
      if 'error' in adapter:
        raise adapter['error']
      self.adapter = adapter['value']
      self.anchor = self.adapter.config.traj_ids['future_start']
      self.draft = validate_draft(self.adapter.tokenizer.encode(DRAFT_TEXT,add_special_tokens=False),self.anchor)
      if self.draft != CHECKPOINT_DRAFT_IDS:
        raise ValueError('Checkpoint nominal draft token IDs changed')
      finished('input_adapter_join')
      if self.model.weight_digest != artifacts['fp8_weights_sha256']:
        raise ValueError('Selected FP8 weight bytes changed')
      from . import graph_cache
      from tinygrad.codegen import to_program_config
      from tinygrad.helpers import PROFILE
      graph_cache.configure(self.config.cache_dir/'graphs',{'model':self.model},
                            # Profiled graphs carry timestamp commands; they never mix with production graphs.
                            (TINYGRAD_REVISION,self.model.weight_digest,manifest(),[(v.key,v.value) for v in to_program_config],
                             bool(PROFILE.value),
                             hashlib.sha256(self.config.prefill_calibration.read_bytes()).hexdigest()))
      self.prefill = SpeculativePrefill(self.model)
      self.decoder = Decoder(self.model,self.draft,self.anchor)
      self.vision = VisionCache(self.model)
      self.inputs,self.vision_buffers = None,VisionBuffers()
      self.settings = SimpleNamespace(max_tokens=64,vision_batch=1)
      self.model.synchronize()
      finished('inference_setup')
      self.device = Device.DEFAULT
      self.manifest = {'variant':'selected','native_fresh_patches':self.config.native_fresh_patches,
                       'recipe':'fp8-group2-pad2112-draft32-euler3-w4-serial-mlp','precision':'mixed-w4-fp8-bf16',
                       'serial_mlp':'W4 group32 FP16 scales, dynamic E4M3 activations, BF16 output',
                       'diffusion_steps':3,'solver':'euler','greedy':True,'max_tokens':64,'prefill_rows':2112,
                       'cache_rows':2304,'vision_group_limit':2,'vision_cache_capacity':144,'images':10,'patches_per_image':720,
                       'history_offsets_s':[-2,-1,-.5,-.1,0],'front_fov_degrees':[120,30],'bf16_vision_layers':2,
                       'bf16_expert_projections':True,'vision_scales':'dynamic-per-image','language_scales':'fixed-112',
                       'reduced_vocabulary':True,'prefill_draft_heads':32,'lookup_carry':True,'serial_block':6,
                        'device':self.device,'arch':'gfx1200','cpu_threads':2,'tinygrad_revision':TINYGRAD_REVISION,
                       'nvidia_revision':NVIDIA_REVISION,'identities':identities,
                       'fp8_weights_sha256':self.model.weight_digest,'kernel_manifest':manifest(),
                        'prefill_calibration_sha256':hashlib.sha256(self.config.prefill_calibration.read_bytes()).hexdigest()}
    except BaseException:
      if self.backend is not None:
        self.backend.close()
      self.closed = True
      raise

  @staticmethod
  def validate_noise(noise):
    import numpy as np
    if not isinstance(noise,np.ndarray) or noise.shape != (1,64,2) or noise.dtype != np.float32 or not np.isfinite(noise).all():
      raise ValueError('Require finite float32 diffusion noise [1,64,2]')

  def anchor_inputs(self, anchors):
    """CPU-only patching and content keys for prime(); safe on another thread."""
    from openpilot.selfdrive.alpamayo.segments import PatchRows
    patches,grids = self.adapter.anchor_patches(anchors)
    pixels,layout = (patches if type(patches) is PatchRows else patches.numpy()),grids.numpy()
    return pixels,layout,grids,self.vision.patch_keys(pixels,layout)

  def prime(self, anchors, deadline_ns, wait=True, inputs=None):
    """Encode only the two real current exposures. A cold graph may outlive its slot.

    wait=False returns once the GPU work is queued; later GPU work is ordered after it.
    `inputs` from anchor_inputs() skips the CPU patching here.
    """
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      return self._prime(anchors, deadline_ns, wait, inputs)

  def _prime(self, anchors, deadline_ns, wait=True, inputs=None):
    import numpy as np
    from .transport import submitted, wait_for
    from openpilot.selfdrive.alpamayo.segments import PatchRows
    if time.monotonic_ns() >= deadline_ns:
      return False
    # CPU patching overlaps queued GPU work. Before the shared upload buffer is rewritten, wait only for the previous prime.
    pixels,layout,grids,keys = self.anchor_inputs(anchors) if inputs is None else inputs
    missing = self.vision.missing_indices(keys)
    encoded = {}
    if missing:
      batches = self.vision_buffers.batches(layout,1)
      positions = self.adapter.vision_positions(grids)
      raw = np.concatenate([pixels.rows(*batches[i][:2]) if type(pixels) is PatchRows else pixels[batches[i][0]:batches[i][1]] for i in missing])
      if time.monotonic_ns() >= deadline_ns:
        return False
      positions = self.vision_buffers.position_buffer(layout,positions)
      wait_for(self.prime_done)  # a queued earlier prime may still read the persistent pixel upload buffer
      expanded = self.vision_buffers.expand_captured(raw)
      self.prime_done = submitted()
      encoded = self.vision.encode_missing(keys,expanded,positions,batches,missing)
      if wait:
        self.model.synchronize()
    if wait and time.monotonic_ns() >= deadline_ns:
      return False
    self.vision.retain(keys,encoded)
    return True

  def predict(self, sample, noise=None, *, retained=()):
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      noise = self.noise if noise is None else noise
      self.validate_noise(noise)
      return self._predict(sample,noise,retained)

  def _predict(self, sample, noise, retained):
    import torch
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace
    from .buffers import CompactInputs
    from .policy import padded_inputs
    from .model import PREFILL_ROWS
    from .transport import download
    started = time.monotonic_ns()
    metrics = Metrics()
    @contextmanager
    def timed(name):
      # Each synchronize is a USB round trip (~1 ms) and stops host work from overlapping the GPU. Without stage_sync,
      # stage times measure host dispatch; the trajectory download still waits for the GPU.
      if self.stage_sync:
        self.model.synchronize()
      begin = time.monotonic_ns()
      yield
      if self.stage_sync:
        self.model.synchronize()
      metrics.stages_ms[name] = (time.monotonic_ns()-begin)/1e6
    with timed('preprocess'):
      prepared,packed = self.adapter.prepare(sample)
      prepared,packed = padded_inputs(prepared,packed,PREFILL_ROWS,self.decoder.prefill_draft())
      self.vision.prepare(prepared)
      if any(self.vision.keys[i] not in self.vision.cache for i in retained):
        from openpilot.selfdrive.alpamayo.frontend import Unavailable
        raise Unavailable('Lost a retained front-camera feature; restarting two-second history')
      missing = self.vision.missing_indices(self.vision.keys)
    with timed('cpu_buffers'):
      if self.inputs is None or not self.inputs.update(self.settings,self.adapter.config,prepared,packed):
        self.inputs = CompactInputs(self.settings,self.adapter.config,prepared,packed,vision=self.vision_buffers)
    with timed('input_upload'):
      pixels,vpos,ids,cos,sin,dc,ds,ac,ass,actions = self.inputs.upload(noise,missing)
    with timed('vision'):
      embedding = self.vision.encode(pixels,vpos,ids,self.inputs.batches,self.inputs.spans,missing)
    prompt = prepared['actual_prompt_length']
    with timed('prefill'):
      logits = self.prefill(embedding,cos,sin,64,prompt)
    with timed('reasoning'):
      generated = self.decoder.generate(self.prefill,logits,dc,ds,prompt,prepared['prefill_draft'])
    with timed('diffusion'):
      actions = self.model.trajectory(actions,ac,ass,prompt+len(generated),3,len(generated))
      action_array = download(actions)
    with timed('trajectory_decode'):
      action_config = dict(self.adapter.config.expert_config.action_space_cfg)
      action_config.pop('_target_')
      action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
      with torch.no_grad():
        xyz,rot = action_space.action_to_traj(torch.from_numpy(action_array.copy()),prepared['ego_history_xyz'][:,0],prepared['ego_history_rot'][:,0])
      if xyz.shape != (1,64,3) or rot.shape != (1,64,3,3) or not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
        raise RuntimeError('Invalid decoded trajectory')
    prediction = {'xyz':xyz.numpy().tolist(),'rot':rot.numpy().tolist(),'generated_token_ids':generated,
                  'generated_tokens':len(generated),'reasoning':self.adapter.tokenizer.decode(generated),
                  'prompt_tokens':prompt,'prefill_tokens':PREFILL_ROWS,'timings_ms':dict(metrics.stages_ms),
                  'noise_bytes_sha256':hashlib.sha256(noise.tobytes()).hexdigest()}
    metrics.runtime_ms = (time.monotonic_ns()-started)/1e6
    metrics.vision_hits,metrics.vision_misses = self.vision.hits,self.vision.misses
    metrics.uploaded_pixel_bytes = self.inputs.uploaded_pixel_bytes
    metrics.speculation = dict(self.decoder.statistics)
    self.metrics = metrics
    prediction['runtime_ms'] = metrics.runtime_ms
    return prediction

  def warmup(self):
    """Compile and capture every steady-state graph shape before live slots start.

    Without this, the first turn (new verifier width), a repriming slot (different upload count) or an odd image
    group compiles during driving and blocks the planner for tens of seconds. Synthetic inputs only shape the graphs;
    all sample state is cleared afterwards, while compiled programs and captured graphs are kept.
    """
    import numpy as np
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      started = time.monotonic_ns()
      rng = np.random.default_rng(0)
      images = rng.integers(0, 256, (10,1,3,432,768), dtype=np.uint8)
      sample = {'image_frames':images,'camera_indices':np.array([1]*5+[6]*5,dtype=np.int64),
                'absolute_timestamps':np.tile(np.array([-2_000_000,-1_000_000,-500_000,-100_000,0],dtype=np.int64)+10_000_000,2),
                'ego_history_xyz':np.zeros((1,20,3),dtype=np.float32),
                'ego_history_rot':np.tile(np.eye(3,dtype=np.float32),(1,20,1,1)),'nav_text':['']}
      from .policy import NoTrajectoryAnchor
      def run(fresh):
        # Replace `fresh` current images per call: the live cadence uploads 4; repriming and gaps use other counts.
        for i in (4,9,3,8,2,7,1,6,0,5)[:fresh]:
          images[i] = rng.integers(0, 256, images.shape[1:], dtype=np.uint8)
        try:
          self._predict(sample, self.noise, ())
        except NoTrajectoryAnchor:  # Random pixels need not produce a complete text.
          pass
      phases = {}
      def phase(name):
        nonlocal started_phase
        now = time.monotonic_ns(); phases[name] = round((now-started_phase)/1e6); started_phase = now
      started_phase = started
      run(10)
      phase('first_prediction')
      # The 500 ms rolling worker primes two anchors per slot and then uploads four new images per prediction.
      # With the -0.1 s pair encoded early, a matching slot uploads only its two current images. When only one camera's
      # early frame matches, three images are new; odd counts also use the single-image encoder.
      # Free-running inference re-encodes a history image when no held frame falls in its window: 5 or 6 images.
      for fresh in (4,4,4,2,2,2,3,3,3,1,1,1,6,6,6,5,5,5):
        run(fresh)
      for i in range(3):
        anchors = [(rng.integers(0, 256, (1,3,432,768), dtype=np.uint8), {'view':camera,'capture_ns':(i+1)*1_000_000_000})
                   for camera in (1,6)]
        if not self._prime(anchors, time.monotonic_ns()+600_000_000_000):
          raise RuntimeError('Warmup priming failed')
      phase('rolling_shapes')
      # Navigation text lengthens the prompt; the prefill graph takes the length as a variable.
      sample['nav_text'] = ['At the roundabout, take exit 2 in 150m']
      run(0)
      sample['nav_text'] = ['']
      phase('navigation_prompts')
      decoder = self.decoder
      generate,first_token = decoder.generate,decoder.first_token
      seen = []
      def first(logits):
        seen.append(first_token(logits))
        return seen[-1]
      # Force the lookup path, then propose drafts of each verifier width from the observed first token.
      decoder.generate = lambda prefill,logits,dc,ds,prompt,placed,max_tokens=64: decoder.generate_lookup(logits,dc,ds,prompt,max_tokens)
      decoder.first_token = first
      filler = self.draft[0]
      try:
        decoder.history = [[filler]*20]
        run(0)
        for width in (16,32):
          for _ in range(3):
            decoder.history = [[seen[-1]]+[filler]*(width-2)+[self.anchor]]  # A proposal of exactly `width` tokens.
            run(0)
          phase(f'verifier_{width}')
      finally:
        decoder.generate,decoder.first_token = generate,first_token
        self.model.synchronize()
        decoder.history.clear()
        decoder.statistics = {}
        self.vision.cache.clear()
        self.vision.keys = ()
        self.vision.hits = self.vision.misses = 0
        self.adapter.reset()
        self.prefill.last_targets = self.prefill.last_prompt = self.prefill.last_logits = None
        self.metrics = Metrics()
      self.startup_ms['warmup'] = (time.monotonic_ns()-started)/1e6
      print(f"Alpamayo startup: warmup {self.startup_ms['warmup']:.0f} ms {phases}", file=sys.stderr, flush=True)

  def synchronize(self):
    if not self.closed:
      self.model.synchronize()

  def gpu_metrics(self):
    """Chestnut temperatures, power, clock and fan from the SMU metrics table (as modeld reads them)."""
    import ctypes
    from tinygrad import Device
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      smu = Device['AMD'].iface.dev_impl.smu
      metrics_t = smu.smu_mod.SmuMetricsExternal_t
      smu._send_msg(smu.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, smu.smu_mod.TABLE_SMU_METRICS, timeout=100)
      m = metrics_t.from_buffer(bytearray(smu.adev.vram.view(smu.driver_table_paddr, ctypes.sizeof(metrics_t))[:])).SmuMetrics
      return {'hotspot_c': m.AvgTemperature[smu.smu_mod.TEMP_HOTSPOT], 'memory_c': m.AvgTemperature[smu.smu_mod.TEMP_MEM],
              'power_w': m.AverageSocketPower, 'gfx_mhz': m.AverageGfxclkFrequencyPostDs, 'fan_rpm': m.AvgFanRpm}

  def reset(self):
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      self.synchronize()
      self.model.reset()
      self.prefill.reset()
      self.decoder.reset()
      self.vision.reset()
      self.adapter.reset()
      self.inputs = None
      self.vision_buffers.positions.clear()
      self.metrics = Metrics()

  def close(self):
    with self.lock:
      if self.closed:
        return
      try:
        self.model.synchronize()
      finally:
        self.inputs = self.vision = self.decoder = self.prefill = self.model = None
        self.vision_buffers = None
        try:
          self.adapter.close()
        finally:
          self.closed = True
          self.backend.close()


class ShadowAdapter:
  """Adapter for the shared runtime.run_prediction boundary."""
  def __init__(self, config):
    config = config if isinstance(config,Config) else Config(**config)
    if config.initial_noise is None:
      raise ValueError('The shared shadow adapter requires an initial_noise file')
    self.runtime = Runtime(config)
    self.runtime.stage_sync = False  # Live slots: no per-stage USB round trips; stage times record host dispatch.
    self.manifest = self.runtime.manifest
    self.startup_ms = self.runtime.startup_ms

  def predict(self, arrays, metadata):
    retained = tuple(i for i,ref in enumerate(metadata['frames']) if 'anchor_slot_ns' in ref)
    prediction = self.runtime.predict(arrays,retained=retained)
    prediction['selected_metrics'] = asdict(self.runtime.metrics)
    return {'query_ns':metadata['query_ns'],'prediction':prediction}

  def prime(self, anchors, deadline_ns, wait=True, inputs=None):
    return self.runtime.prime(anchors,deadline_ns,wait,inputs)

  def anchor_inputs(self, anchors):
    return self.runtime.anchor_inputs(anchors)

  def encode_ahead(self, anchors, inputs=None):
    """Queue the current pair's vision encoding while the CPU prepares the rest of the input.

    `inputs` from anchor_inputs() of these exact images skips the CPU patching.
    """
    self.runtime.prime(anchors,time.monotonic_ns()+10**12,wait=False,inputs=inputs)

  def warmup(self):
    self.runtime.warmup()

  def pin_threads(self, cpus):
    # The camera-patch pool thread already exists; threads created later inherit the caller's affinity.
    self.runtime.adapter.worker.submit(os.sched_setaffinity, 0, cpus).result()

  def synchronize(self):
    self.runtime.synchronize()

  def gpu_metrics(self):
    return self.runtime.gpu_metrics()

  def close(self):
    self.runtime.close()
