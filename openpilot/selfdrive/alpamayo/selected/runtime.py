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

  def __post_init__(self):
    for name in ('checkpoint','fp8_cache','prefill_calibration','cache_dir','initial_noise'):
      if (value := getattr(self,name)) is not None:
        object.__setattr__(self,name,Path(value).resolve())
    if self.device_timeout_ms <= 0:
      raise ValueError('Device timeout must be positive')


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
  if any(identities[key] != expected[key] for key in paths):
    raise ValueError('Selected checkpoint, FP8 index, or fixed calibration differs from the pinned recipe')
  index = json.loads(paths['checkpoint_weight_index_sha256'].read_text())['weight_map']
  shards = {name:sha(config.checkpoint/name) for name in sorted(set(index.values()))}
  if shards != expected['checkpoint_shards']:
    raise ValueError('Original BF16 checkpoint shard bytes changed')
  identities.update(checkpoint_shards=shards,checkpoint_weights_sha256=hashlib.sha256(json.dumps(shards,sort_keys=True).encode()).hexdigest())
  if config.initial_noise is not None:
    identities['initial_noise_sha256'] = sha(config.initial_noise)
  return expected,identities


class Runtime:
  """Persistent selected backend; pass explicit float32 noise to predict for replay."""
  def __init__(self, config):
    self.config = config if isinstance(config,Config) else Config(**config)
    self.closed,self.backend = False,None
    self.lock = threading.Lock()
    self.metrics = Metrics()
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
    verify_assets()
    artifacts,identities = artifact_identities(self.config)
    require_firmware(self.config.cache_dir)
    torch.set_num_threads(2)
    scales = json.loads(self.config.prefill_calibration.read_text())['scales']
    self.noise = None if self.config.initial_noise is None else np.load(self.config.initial_noise,allow_pickle=False)
    if self.noise is not None:
      self.validate_noise(self.noise)
      self.noise.setflags(write=False)
    self.adapter = SelectedInputs(self.config.checkpoint)
    self.anchor = self.adapter.config.traj_ids['future_start']
    self.draft = validate_draft(self.adapter.tokenizer.encode(DRAFT_TEXT,add_special_tokens=False),self.anchor)
    if self.draft != CHECKPOINT_DRAFT_IDS:
      raise ValueError('Checkpoint nominal draft token IDs changed')
    try:
      self.backend = Backend(self.config.device_timeout_ms)
      self.model = SelectedModel(self.config.checkpoint,self.config.fp8_cache,scales,self.backend.pv_program())
      if self.model.weight_digest != artifacts['fp8_weights_sha256']:
        raise ValueError('Selected FP8 weight bytes changed')
      self.prefill = SpeculativePrefill(self.model)
      self.decoder = Decoder(self.model,self.draft,self.anchor)
      self.vision = VisionCache(self.model)
      self.inputs,self.vision_buffers = None,VisionBuffers()
      self.settings = SimpleNamespace(max_tokens=64,vision_batch=1)
      self.model.synchronize()
      self.device = Device.DEFAULT
      self.manifest = {'variant':'selected','recipe':'fp8-group2-pad2048-nominal6-euler3-w4-serial-mlp','precision':'mixed-w4-fp8-bf16',
                       'serial_mlp':'W4 group32 FP16 scales, dynamic E4M3 activations, BF16 output',
                       'diffusion_steps':3,'solver':'euler','greedy':True,'max_tokens':64,'prefill_rows':2048,
                       'cache_rows':2304,'vision_group_limit':2,'vision_cache_capacity':144,'images':10,'patches_per_image':720,
                       'history_offsets_s':[-2,-1,-.5,-.1,0],'front_fov_degrees':[120,30],'bf16_vision_layers':2,
                       'bf16_expert_projections':True,'vision_scales':'dynamic-per-image','language_scales':'fixed-112',
                       'reduced_vocabulary':True,'nominal_target_verification':6,'lookup_carry':True,'serial_block':6,
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

  def prime(self, anchors, deadline_ns):
    """Encode only the two real current exposures. A cold graph may outlive its slot."""
    import numpy as np
    from tinygrad import Tensor
    with self.lock:
      if self.closed:
        raise RuntimeError('Runtime is closed')
      if time.monotonic_ns() >= deadline_ns:
        return False
      patches,grids = self.adapter.anchor_patches(anchors)
      pixels,layout = patches.numpy(),grids.numpy()
      keys = self.vision.patch_keys(pixels,layout)
      missing = self.vision.missing_indices(keys)
      encoded = {}
      if missing:
        batches = self.vision_buffers.batches(layout,1)
        positions = self.adapter.vision_positions(grids)
        raw = np.concatenate([pixels[batches[i][0]:batches[i][1]] for i in missing])
        if time.monotonic_ns() >= deadline_ns:
          return False
        positions = self.vision_buffers.position_buffer(layout,positions)
        expanded = self.vision_buffers.expand(Tensor(raw,device='CPU'))
        encoded = self.vision.encode_missing(keys,expanded,positions,batches,missing)
        self.model.synchronize()
      if time.monotonic_ns() >= deadline_ns:
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
    from .transport import download
    started = time.monotonic_ns()
    metrics = Metrics()
    @contextmanager
    def timed(name):
      self.model.synchronize()
      begin = time.monotonic_ns()
      yield
      self.model.synchronize()
      metrics.stages_ms[name] = (time.monotonic_ns()-begin)/1e6
    with timed('preprocess'):
      prepared,packed = self.adapter.prepare(sample)
      prepared,packed = padded_inputs(prepared,packed,2048,self.draft,self.anchor)
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
      generated = self.decoder.generate(self.prefill,logits,dc,ds,prompt)
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
                  'prompt_tokens':prompt,'prefill_tokens':2048,'timings_ms':dict(metrics.stages_ms),
                  'noise_bytes_sha256':hashlib.sha256(noise.tobytes()).hexdigest()}
    metrics.runtime_ms = (time.monotonic_ns()-started)/1e6
    metrics.vision_hits,metrics.vision_misses = self.vision.hits,self.vision.misses
    metrics.uploaded_pixel_bytes = self.inputs.uploaded_pixel_bytes
    metrics.speculation = dict(self.decoder.statistics)
    self.metrics = metrics
    prediction['runtime_ms'] = metrics.runtime_ms
    return prediction

  def synchronize(self):
    if not self.closed:
      self.model.synchronize()

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
          self.backend.close()
        finally:
          self.closed = True


class ShadowAdapter:
  """Adapter for the shared runtime.run_prediction boundary."""
  def __init__(self, config):
    config = config if isinstance(config,Config) else Config(**config)
    if config.initial_noise is None:
      raise ValueError('The shared shadow adapter requires an initial_noise file')
    self.runtime = Runtime(config)
    self.manifest = self.runtime.manifest

  def predict(self, arrays, metadata):
    retained = tuple(i for i,ref in enumerate(metadata['frames']) if 'anchor_slot_ns' in ref)
    prediction = self.runtime.predict(arrays,retained=retained)
    prediction['selected_metrics'] = asdict(self.runtime.metrics)
    return {'query_ns':metadata['query_ns'],'prediction':prediction}

  def prime(self, anchors, deadline_ns):
    return self.runtime.prime(anchors,deadline_ns)

  def synchronize(self):
    self.runtime.synchronize()

  def close(self):
    self.runtime.close()
