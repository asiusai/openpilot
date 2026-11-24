import ctypes
import os
import struct
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tinygrad import Context, Device
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.renderer import TinyELF
from tinygrad.runtime.autogen import mesa, msm_drm
from tinygrad.runtime.ops_qcom import MSMIface, QCOMComputeQueue, QCOMProgramData, _qcom_pvtmem_sizes, pkt4_hdr, pkt7_hdr
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.uop.ops import KernelInfo, UOp


def test_private_memory_register_units():
  assert _qcom_pvtmem_sizes(0, 4096, 2) == (0, 0, 0, 4096)
  assert _qcom_pvtmem_sizes(1, 4096, 2) == (1, 512, 1024, 4 << 20)
  assert _qcom_pvtmem_sizes(513, 4096, 2) == (2, 1024, 2048, 8 << 20)


@pytest.mark.parametrize('arch', ['a630', 'a635,GPU_ID=635,CHIP_ID=0x306030500'])
def test_program_metadata_compiles_from_target_without_gpu(arch):
  with Context(DEV=f'MSM+QCOM:IR3:{arch}:copy=0'):
    buf, i = UOp.param(0, dtypes.float32, 64), UOp.range(64, 0)
    ast = buf[i].store(buf[i] + 1).end(i).sink(arg=KernelInfo('target_metadata'))
    program = to_program(ast, Device.renderer_for('QCOM'))
    data = QCOMProgramData('QCOM', TinyELF.from_program(program))
    assert data.NIR and data.image_size > 0
    assert data.dev_info.fibers_per_sp > 0 and data.dev_info.num_sp_cores > 0
    assert data.dev_info.a6xx.has_lpac == arch.startswith('a635')
    assert Device.target_for('QCOM').interface == 'MSM'
    assert Device.target_for('QCOM').get('copy', 1) == 0


def test_a635_registers_follow_compiler_metadata():
  queue = object.__new__(QCOMComputeQueue)
  queue.devs, queue.blob, queue.patches = ('QCOM',), bytearray(), []
  data = SimpleNamespace(dev_info=SimpleNamespace(a6xx=SimpleNamespace(supports_double_threadsize=False, has_lpac=True)),
                         NIR=True, wgsz=1, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                         pvtmem_size_per_item=0, pvtmem_size_total=0, pvtmem_per_wave=False, stack_size=4096,
                         hw_stack_offset=0, image_size=0x3280, constlen=28, double_threadsize=False, early_preamble=False,
                         mergedregs=False, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc, max_threads=1024, kernargs_alloc_size=2048)
  program = SimpleNamespace(arg=SimpleNamespace(launch_dims=lambda _: ((2, 3, 3), (2, 3, 4))))
  buffer = UOp.alloc((4096,), dtypes.uint8, device='QCOM')
  with patch('tinygrad.runtime.ops_qcom.qcom_build_program', return_value=(data, buffer)), patch.object(queue, 'kernargs', return_value=buffer):
    queue.exec(None, program)
  words = list(struct.unpack(f'{len(queue.blob) // 4}I', queue.blob))
  const = words.index(pkt4_hdr(mesa.REG_A6XX_SP_REG_PROG_ID_0, 5))
  execute = words.index(pkt7_hdr(mesa.CP_EXEC_CS, 4))
  assert words[const + 5] == mesa.A6XX_SP_CS_CONST_CONFIG_ENABLED | 7
  assert words[execute + 1:execute + 5] == [0, 2, 3, 3]
  assert pkt4_hdr(mesa.REG_A6XX_HLSQ_CS_CTRL_REG1, 1) in words
  assert pkt4_hdr(mesa.REG_A6XX_SP_PS_WAVE_CNTL, 1) in words
  assert pkt4_hdr(mesa.REG_A6XX_SP_CS_WIE_CNTL_0, 2) in words


def make_iface():
  iface = object.__new__(MSMIface)
  iface.fd, iface.bind_queue, iface.imports = object(), 2, {}
  iface.va_allocator = TLSFAllocator(1 << 20, base=0x100000, block_size=4096)
  return iface


def test_shared_camera_buffer_lifetime():
  iface = make_iface()
  pixels = (ctypes.c_uint8 * 4096)()
  file = tempfile.TemporaryFile()
  fd = file.fileno()
  os.ftruncate(fd, 4096)
  try:
    with patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=SimpleNamespace(handle=19)), \
         patch.object(iface, '_bind', return_value=0x100000) as bind, \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_VM_BIND') as unmap, \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close, \
         patch('tinygrad.runtime.ops_qcom.FileIOInterface.munmap') as cpu_unmap:
      first = iface.map(ctypes.addressof(pixels), 4096, fd)
      second = iface.map(ctypes.addressof(pixels), 4096, fd)
      assert first.buf == second.buf == 0x100000
      assert first.meta is second.meta
      bind.assert_called_once_with(19, 4096)
      pixels[0] = 123
      assert first.host[0] == second.host[0] == 123
      # Match the real allocation bookkeeping supplied by _bind.
      iface.va_allocator.alloc(4096, 4096)
      iface.free(first)
      close.assert_not_called()
      iface.free(second)
      close.assert_called_once_with(iface.fd, handle=19)
      assert unmap.call_args.kwargs['op'].op == msm_drm.MSM_VM_BIND_OP_UNMAP
      cpu_unmap.assert_not_called()
      assert not iface.imports
  finally:
    file.close()


def test_camera_import_rejects_missing_fd_and_oversized_view():
  iface = make_iface()
  with pytest.raises(ValueError, match='DMA-BUF fd'):
    iface.map(0, 4096)
  file = tempfile.TemporaryFile()
  fd = file.fileno()
  try:
    os.ftruncate(fd, 4096)
    with pytest.raises(ValueError, match='mapping size'):
      iface.map(0, 4097, fd)
  finally:
    file.close()


def test_failed_camera_import_releases_handle():
  iface = make_iface()
  file = tempfile.TemporaryFile()
  fd = file.fileno()
  os.ftruncate(fd, 4096)
  try:
    with patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=SimpleNamespace(handle=19)), \
         patch.object(iface, '_bind', side_effect=OSError('bind failed')), \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close:
      with pytest.raises(OSError, match='bind failed'):
        iface.map(0, 4096, fd)
      close.assert_called_once_with(iface.fd, handle=19)
      assert not iface.imports
  finally:
    file.close()
