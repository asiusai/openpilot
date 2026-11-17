"""Strict pinned-tinygrad identity checks. Import never opens a device.

DEV configures transport/renderer. Device.DEFAULT names the backend (AMD).
Only require_gpu_device() without an explicit object can open a device; parent only.
"""


def configuration(spec):
  from tinygrad.helpers import Target
  if not isinstance(spec, str) or not spec or ';' in spec:
    raise ValueError('Require one explicit USB+AMD:LLVM target')
  target = Target.parse(spec)
  if (target.device != 'AMD' or target.renderer != 'LLVM' or target.interface != 'USB'
      or target.indices != '' or target.arch not in ('', 'gfx1200')):
    raise ValueError(f'Require default USB AMD device, LLVM renderer, gfx1200 hardware; requested {target!r}')
  return target


def require_gpu_device(device=None):
  """Validate actual native classes and hardware-derived arch, not only DEV text."""
  from tinygrad import Device
  from tinygrad.helpers import DEV
  requested = configuration(repr(DEV))
  canonical = Device.canonicalize(Device.DEFAULT)
  if canonical != 'AMD':
    raise ValueError(f'Require canonical default AMD device; got {canonical!r}')
  if device is None: device = Device[canonical]  # Executed only by parent GPU entry points.
  from tinygrad.runtime.ops_amd import AMDDevice, USBIface
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  if type(device) is not AMDDevice or device.device != canonical or device.device_id != 0:
    raise ValueError('Require the actual default AMDDevice, index0')
  if (device.is_usb is not True or type(device.iface) is not USBIface or device.iface.dev is not device):
    raise ValueError('Require the actual native USBIface; PCI, KFD, proxy, and mock interfaces are rejected')
  if device.arch != 'gfx1200' or device.target != (12, 0, 0) or device.iface.props.get('gfx_target_version') != 120000:
    raise ValueError(f'Require hardware-derived gfx1200/(12,0,0)/120000 identity; got {device.arch!r}, {device.target!r}')
  renderer = device.renderer
  expected = requested.replacedefault(arch=device.arch)
  if type(renderer) is not AMDLLVMRenderer or renderer.target != expected:
    raise ValueError(f'Require actual AMDLLVMRenderer with target {expected!r}; got {type(renderer).__name__}')
  return dict(configured_target=repr(requested), canonical_default=canonical, actual_device=device.device,
    device_index=device.device_id, hardware_arch=device.arch, hardware_target=list(device.target),
    hardware_gfx_target_version=device.iface.props['gfx_target_version'],
    interface_class=type(device.iface).__module__+'.'+type(device.iface).__name__,
    renderer_class=type(renderer).__module__+'.'+type(renderer).__name__, renderer_target=repr(renderer.target))
