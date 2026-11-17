"""Optional exact byte-packing hook. Resize, history, text, and cache guards stay original."""
from collections import Counter
import ctypes
import hashlib
import inspect
from pathlib import Path
import time
from types import MethodType

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BINARY=WORK/'agent-live-history-fast-artifacts/pack.so'
SOURCE_SHA='1bc366c0856b6ebab526e31d8f31d5e6d09392144309baeef84a52d3386f6f0c'
BINARY_SHA='82542f35dd2af506e0b19c4de951fe28ca8a09170780ff384a8c4888c6c3a843'
PINS={'fast-inputs.py':'d4a38845ef4505bf6d3f73c5011a329e8451ab5bf54cd30927a7142679d72e3d',
  'image_processing_cosmos3_edge.py':'3b97a86df032303c4502b151bfe68c6d369f1e8b4155e194f0c4cd04516244cb'}

def sha(path):
  with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Handle:
  def __init__(self,adapter,binary=BINARY):
    import torch
    import numpy as np
    start=time.perf_counter_ns()
    self.image=adapter.processor.image_processor
    if 'patchify' in vars(self.image):raise ValueError('Image processor already has a patchify override')
    assert type(adapter).__name__=='FastInputs' and type(adapter).__mro__[1:]==(object,),'Expected actual live base FastInputs'
    paths=[Path(inspect.getfile(type(adapter).__init__)),Path(inspect.getfile(type(self.image).patchify))]
    assert {p.name for p in paths}==PINS.keys()
    assert all(sha(p)==PINS[p.name] for p in paths),'Unexpected original adapter/processor source'
    assert torch.__version__=='2.8.0+cu128' and torch.backends.cpu.get_cpu_capability() in ('AVX2','AVX512')
    assert sha(Path(__file__).with_name('agent-live-history-fast-pack.c'))==SOURCE_SHA
    assert sha(binary)==BINARY_SHA,'Unexpected packing library bytes'
    self.library=ctypes.CDLL(str(binary));pack=self.library.pack_320_576
    pack.argtypes=[ctypes.c_void_p,ctypes.c_void_p];pack.restype=None
    self.original=self.image.patchify;self.counts=Counter();self.closed=False
    self.provenance=dict(source_sha256=SOURCE_SHA,binary_sha256=BINARY_SHA,
      original_source_sha256={str(p):sha(p) for p in paths},input_shape=[1,3,320,576],layout=[16,2,1],
      no_arithmetic=True,retained_pixel_bytes=0,extra_threads=0,precomputed_pixels=False)
    owner=self
    def patchify(this,images,patch_size,merge_size,temporal_patch_size):
      assert this is owner.image and not owner.closed
      if (type(images) is torch.Tensor and images.device.type=='cpu' and images.dtype==torch.uint8 and
          tuple(images.shape)==(1,3,320,576) and images.is_contiguous() and
          (patch_size,merge_size,temporal_patch_size)==(16,2,1)):
        output=np.empty((1,720,768),dtype=np.uint8)
        pack(output.ctypes.data,images.data_ptr())
        owner.counts['packed_images']+=1;owner.counts['output_bytes']+=output.nbytes
        return torch.from_numpy(output),20,36
      owner.counts['fallback_images']+=1
      return owner.original(images,patch_size,merge_size,temporal_patch_size)
    self.installed=MethodType(patchify,self.image);self.image.patchify=self.installed
    self.install_ms=(time.perf_counter_ns()-start)/1e6

  def state(self):
    return dict(counts=dict(self.counts),closed=self.closed,install_ms=self.install_ms,
      retained_pixel_bytes=0,source_and_binary=self.provenance)

  def close(self):
    if self.closed:return
    assert self.image.patchify is self.installed,'Packing hook owner changed'
    del self.image.patchify
    assert self.image.patchify==self.original
    self.closed=True

def install(adapter,binary=BINARY):return Handle(adapter,binary)
