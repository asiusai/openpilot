#!/usr/bin/env python3
"""CPU-only canonicalization and strict identity regressions. No AMD constructor runs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run():
  if os.environ.get('DEV', 'CPU') != 'CPU': raise ValueError('Run identity regressions with DEV=CPU')
  from tinygrad import Device
  from tinygrad.helpers import Context, DEV, Target
  from tinygrad.runtime.ops_amd import AMDDevice, USBIface, PCIIface, KFDIface
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  guard = load('agent-activation-sparsity-device')
  opened_before = set(Device._opened_devices)
  accepted, rejected = [], []
  # Unconstructed real runtime/interface instances carry only inspected fields.
  # The real renderer constructor initializes its CPU LLVM compiler, not a GPU.
  def identity():
    obj = object.__new__(AMDDevice)
    obj.device, obj.device_id, obj.arch, obj.target, obj.is_usb = 'AMD', 0, 'gfx1200', (12, 0, 0), True
    obj.iface = object.__new__(USBIface)
    obj.iface.dev, obj.iface.props = obj, {'gfx_target_version': 120000}
    target = DEV.target('AMD', arch=obj.arch)
    obj.renderers, obj.cached_renderer = [AMDLLVMRenderer], {(AMDLLVMRenderer, target): AMDLLVMRenderer(target)}
    return obj
  def rejects(label, fn):
    try: fn()
    except ValueError: rejected.append(label)
    else: raise AssertionError('Invalid identity accepted: '+label)
  with Context(DEV='CPU', ALLOW_DEVICE_USAGE=0):
    for spec in ('USB+AMD:LLVM', 'USB+AMD:LLVM:gfx1200', 'USB+amd:llvm'):
      with Context(DEV=spec):
        assert Device.DEFAULT == 'AMD', 'Pinned Device.DEFAULT strips transport and renderer'
        assert DEV.target('AMD').interface == 'USB' and DEV.target('AMD').renderer == 'LLVM'
        actual = guard.require_gpu_device(identity())
        assert actual['hardware_arch'] == 'gfx1200' and actual['canonical_default'] == 'AMD'
        accepted.append(actual)
    for spec in ('CPU', 'AMD:LLVM', 'PCI+AMD:LLVM', 'MOCKUSB+AMD:LLVM', 'USB+AMD:HIP',
                 'USB+NV:LLVM', 'USB+AMD:LLVM:gfx1201', 'USB:1+AMD:LLVM', 'USB+AMD:LLVM;CPU', ''):
      rejects('configuration:'+spec, lambda spec=spec: guard.configuration(spec))
    with Context(DEV='USB+AMD:LLVM'):
      # Every rejection uses the actual pinned classes and selector properties.
      for name, value in (('arch', 'gfx1201'), ('target', (12, 0, 1)), ('device', 'AMD:1'),
                           ('device_id', 1), ('is_usb', False)):
        obj = identity();setattr(obj, name, value)
        rejects('actual:'+name, lambda obj=obj: guard.require_gpu_device(obj))
      obj = identity();obj.iface.props['gfx_target_version'] = 120001
      rejects('actual:hardware-properties', lambda: guard.require_gpu_device(obj))
      for iface_class in (PCIIface, KFDIface, type('MOCKUSBIface', (USBIface,), {})):
        obj = identity();obj.iface = object.__new__(iface_class)
        obj.iface.dev, obj.iface.props = obj, {'gfx_target_version': 120000}
        rejects('actual:'+iface_class.__name__, lambda obj=obj: guard.require_gpu_device(obj))
      obj = identity();obj.iface.dev = object()
      rejects('actual:interface-owner', lambda: guard.require_gpu_device(obj))
      obj = identity()
      rejects('actual:proxy-object', lambda: guard.require_gpu_device(SimpleNamespace(**obj.__dict__)))
      for field, value in (('arch', 'gfx1201'), ('interface', 'PCI'), ('renderer', 'HIP')):
        obj = identity();renderer = obj.renderer
        fields = dict(device='AMD', renderer='LLVM', arch='gfx1200', interface='USB')
        fields[field] = value;renderer.target = Target(**fields)
        rejects('actual:renderer-target-'+field, lambda obj=obj: guard.require_gpu_device(obj))
      obj = identity();target = DEV.target('AMD', arch='gfx1200')
      obj.cached_renderer[(AMDLLVMRenderer, target)] = object.__new__(HIPRenderer)
      rejects('actual:HIP-renderer', lambda: guard.require_gpu_device(obj))
      # Runner runtime validation must precede model initialization and run again
      # afterwards. Exercise that exact helper without pipeline/device execution.
      runner = load('agent-activation-sparsity-run')
      obj = identity();events = []
      original_validate = guard.require_gpu_device
      def initialize():
        import wgp_compiler
        events.append('factory');wgp_compiler.install(obj)
        return SimpleNamespace()
      def validate(): events.append('gate');return original_validate(obj)
      with patch.object(guard, 'require_gpu_device', validate):
        result = {}
        runner.initialize_checked(initialize, guard, result)
        assert events == ['gate', 'factory', 'gate'] and result['gpu_device_identity']['hardware_arch'] == 'gfx1200'
      accepted.append({'runner_pre_and_post_factory': True, 'actual_wgp_compiler_installer_compatible': True})
      events.clear()
      with patch.object(guard, 'require_gpu_device', side_effect=ValueError('invalid actual identity')):
        rejects('runner:reject-before-factory', lambda: runner.initialize_checked(initialize, guard, {}))
      assert events == [], 'Wrong hardware entered the original model factory'
      expected = original_validate(obj)
      with patch.object(guard, 'require_gpu_device', side_effect=[expected, dict(expected, hardware_arch='gfx1201')]):
        rejects('runner:identity-change-during-factory', lambda: runner.initialize_checked(initialize, guard, {}))
      assert events == ['factory']
  assert Device._opened_devices == opened_before, 'CPU identity checks opened a device'
  return dict(cpu_only=True, complete=True, all_pass=True, accepted=accepted, rejected=rejected,
    accepted_count=len(accepted), rejected_count=len(rejected), new_devices_opened=[],
    evidence='Pinned Target.parse, DEV, Device.DEFAULT and real unconstructed AMDDevice/USBIface plus AMDLLVMRenderer',
    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('agent-activation-sparsity*.py'))})


def main():
  parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args();report = run()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({key: report[key] for key in ('cpu_only', 'all_pass', 'accepted_count', 'rejected_count', 'new_devices_opened')}))


if __name__ == '__main__': main()
