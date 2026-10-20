import ctypes, errno, mmap, os, unittest
from types import SimpleNamespace
from unittest.mock import patch

import tinygrad.runtime.autogen as autogen
from tinygrad.helpers import mv_address
from tinygrad.runtime.autogen import msm_drm
from tinygrad.runtime.support.system import FileIOInterface


def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number


class RecordingMSMFile(FileIOInterface):
  def __init__(self, name=b"msm"):
    self.name, self.memory = name, bytearray([0xaa] * mmap.PAGESIZE)
    self.cpu_addr = mv_address(memoryview(self.memory))
    self.requests, self.mmaps, self.unmaps, self.closed_handles, self.gem_flags = [], [], [], [], []
    self.binds, self.submissions, self.waits, self.new_queues, self.closed_queues = [], [], [], [], []
    self.bind_errno = self.wait_errno = None
    self.chip_id, self.gpu_id, self.import_handle = 0xac06030500, 0, 19

  def __del__(self): pass

  def ioctl(self, request, arg):
    self.requests.append(request)
    if request == ioctl_number(msm_drm.DRM_IOCTL_VERSION):
      ctypes.memmove(arg.name, self.name, min(len(self.name), arg.name_len))
      arg.name_len = len(self.name)
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_NEW):
      self.gem_flags.append(arg.flags)
      arg.handle = 17
    elif request == ioctl_number(msm_drm.DRM_IOCTL_PRIME_FD_TO_HANDLE): arg.handle = self.import_handle
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM):
      arg.value = {msm_drm.MSM_PARAM_GPU_ID: self.gpu_id, msm_drm.MSM_PARAM_CHIP_ID: self.chip_id,
                   msm_drm.MSM_PARAM_VA_START: 0x1234_0000, msm_drm.MSM_PARAM_VA_SIZE: 0x1000_0000}[arg.param]
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_INFO): arg.value = 0x8000
    elif request == ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE): self.closed_handles.append(arg.handle)
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_VM_BIND):
      if self.bind_errno is not None: raise OSError(self.bind_errno, "bind failed")
      self.binds.append((arg.op.op, arg.op.handle, arg.op.iova, arg.op.range))
      arg.fence_fd, write_fd = os.pipe()
      os.write(write_fd, b"x")
      os.close(write_fd)
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT):
      cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * arg.nr_cmds).from_address(arg.cmds)
      self.submissions.append((arg.flags, arg.queueid, [(cmd.type, cmd.size, cmd.iova) for cmd in cmds]))
      arg.fence = 42
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE):
      self.waits.append((arg.fence, arg.timeout.tv_sec, arg.timeout.tv_nsec, arg.queueid))
      if self.wait_errno is not None: raise OSError(self.wait_errno, "wait failed")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW):
      self.new_queues.append((arg.flags, arg.prio))
      arg.id = 2 if arg.flags == msm_drm.MSM_SUBMITQUEUE_VM_BIND else 3
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE): self.closed_queues.append(arg.value)
    return 0

  def mmap(self, start, size, prot, flags, offset):
    self.mmaps.append((start, size, prot, flags, offset))
    return self.cpu_addr

  def munmap(self, addr, size):
    self.unmaps.append((addr, size))
    return 0


def make_iface(fd):
  from tinygrad.runtime.ops_qcom import MSMIface
  from tinygrad.runtime.support.memory import TLSFAllocator

  iface = object.__new__(MSMIface)
  iface.dev, iface.fd = SimpleNamespace(last_cmd=0, timeline_value=1, timeline_signal=SimpleNamespace(wait=lambda _: None)), fd
  iface.dev.msm_fence = SimpleNamespace(host=SimpleNamespace(view=lambda **_: [iface.dev.last_cmd]))
  iface.queue_id, iface.vm_bind_queue_id = 3, 2
  iface.submit_flags = msm_drm.MSM_PIPE_3D0 | msm_drm.MSM_SUBMIT_SUDO
  iface.va_allocator = TLSFAllocator(0x1000_0000, base=0x1234_0000, block_size=mmap.PAGESIZE)
  iface.allocations = {}
  return iface


class TestMSMDRMUAPI(unittest.TestCase):
  def test_autogen_targets_aarch64(self):
    with patch.object(autogen, "load") as load: autogen.__getattr__("msm_drm")
    self.assertIn("--target=aarch64-linux-gnu", load.call_args.kwargs["args"])

  def test_critical_layouts_and_ioctls(self):
    self.assertEqual(ctypes.sizeof(msm_drm.struct_drm_msm_gem_submit), 72)
    self.assertEqual(ctypes.sizeof(msm_drm.struct_drm_msm_vm_bind), 88)
    self.assertEqual(ctypes.sizeof(msm_drm.struct_drm_msm_wait_fence), 32)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE), 0x40206447)


class TestMSMIface(unittest.TestCase):
  def test_init_selects_msm_and_configures_a635(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    foreign, fd = RecordingMSMFile(b"vgem"), RecordingMSMFile()
    with patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", side_effect=[foreign, fd]):
      iface = MSMIface(SimpleNamespace(), 0)

    self.assertIs(iface.fd, fd)
    self.assertEqual((iface.gpu_id, iface.mesa_gpu_id), ((6, 3, 5), 0))
    self.assertEqual(iface.submit_flags, msm_drm.MSM_PIPE_3D0)
    self.assertEqual(fd.new_queues, [(msm_drm.MSM_SUBMITQUEUE_VM_BIND, 0), (0, 0)])
    self.assertLess(fd.requests.index(ioctl_number(msm_drm.DRM_IOCTL_MSM_SET_PARAM)),
                    fd.requests.index(ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW)))

  def test_init_allows_explicit_sudo_submits(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    fd = RecordingMSMFile()
    with patch("tinygrad.runtime.ops_qcom.getenv", return_value=1), \
         patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", return_value=fd):
      iface = MSMIface(SimpleNamespace(), 0)

    self.assertEqual(iface.submit_flags, msm_drm.MSM_PIPE_3D0 | msm_drm.MSM_SUBMIT_SUDO)

  def test_init_rejects_unvalidated_gpu(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    fd = RecordingMSMFile()
    fd.chip_id = 0x06050000
    with patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", return_value=fd), \
         self.assertRaisesRegex(RuntimeError, "Adreno 630/635"):
      MSMIface(SimpleNamespace(), 0)
    self.assertEqual(fd.new_queues, [])

  def test_alloc_and_free_lifecycle(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)

    buf = iface.alloc(17, fill_zeroes=True)

    self.assertEqual((buf.buf, buf.host.addr, len(buf.host)), (0x1234_0000, fd.cpu_addr, 17))
    self.assertNotEqual(buf.buf, buf.host.addr)
    self.assertEqual(fd.memory[:17], bytes(17))
    self.assertEqual(fd.gem_flags, [msm_drm.MSM_BO_CACHED_COHERENT])
    self.assertEqual(fd.mmaps, [(0, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0x8000)])
    self.assertEqual(fd.binds, [(msm_drm.MSM_VM_BIND_OP_MAP, 17, 0x1234_0000, mmap.PAGESIZE)])

    iface.free(buf)
    self.assertEqual(fd.binds[-1], (msm_drm.MSM_VM_BIND_OP_UNMAP, 0, 0x1234_0000, mmap.PAGESIZE))
    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

  def test_failed_bind_releases_allocation(self):
    fd = RecordingMSMFile()
    fd.bind_errno = errno.EIO
    iface = make_iface(fd)

    with self.assertRaisesRegex(OSError, "bind failed"): iface.alloc(17)

    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

  def test_dma_buf_requires_fd_and_valid_size(self):
    iface = make_iface(RecordingMSMFile())
    with self.assertRaisesRegex(ValueError, "DMA-BUF fd"): iface.map(0x1000, 16)
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=15)), \
         self.assertRaisesRegex(ValueError, "exceeds DMA-BUF size"):
      iface.map(0x1000, 16, 9)

  def test_dma_buf_import_is_bound_once(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=mmap.PAGESIZE)):
      first = iface.map(fd.cpu_addr, 17, 9)
      second = iface.map(fd.cpu_addr + 0x40, 17, 9)

    self.assertEqual((first.buf, second.buf), (0x1234_0000, 0x1234_0000))
    self.assertEqual((first.host.addr, second.host.addr), (fd.cpu_addr, fd.cpu_addr + 0x40))
    self.assertIs(first.meta, second.meta)
    self.assertEqual(first.meta.refcount, 2)
    self.assertEqual(fd.binds, [(msm_drm.MSM_VM_BIND_OP_MAP, 19, 0x1234_0000, mmap.PAGESIZE)])

    iface.free(first)
    self.assertEqual(fd.closed_handles, [])
    iface.free(second)
    self.assertEqual(fd.closed_handles, [19])
    self.assertEqual(fd.binds[-1], (msm_drm.MSM_VM_BIND_OP_UNMAP, 0, 0x1234_0000, mmap.PAGESIZE))

  def test_wait_fence_uses_absolute_deadline(self):
    from tinygrad.runtime.ops_qcom import MSM_WAIT_SLICE_NS

    fd = RecordingMSMFile()
    fd.wait_errno = errno.ETIMEDOUT
    iface = make_iface(fd)
    iface.dev.last_cmd = 41
    with patch("tinygrad.runtime.ops_qcom.time.monotonic_ns", return_value=5_000_000_123): iface.sleep(0)

    deadline = 5_000_000_123 + MSM_WAIT_SLICE_NS
    self.assertEqual(fd.waits, [(41, deadline // 1_000_000_000, deadline % 1_000_000_000, 3)])

  def test_wait_fence_reports_driver_errors(self):
    fd = RecordingMSMFile()
    fd.wait_errno = errno.EIO
    iface = make_iface(fd)
    iface.dev.last_cmd = 41
    with self.assertRaisesRegex(RuntimeError, "MSM fence wait failed"): iface.sleep(0)

  def test_device_fini_closes_queues(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    iface.device_fini()
    self.assertEqual(fd.closed_queues, [3, 2])


if __name__ == "__main__":
  unittest.main()
