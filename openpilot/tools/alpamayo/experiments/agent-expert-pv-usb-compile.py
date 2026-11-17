"""CPU-only full USB target, saved ELF/ABI, corruption, and WAW validation."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', CCACHE='0', TC_OPT='2', BEAM='0', DEBUG='0', PROFILE='0')
  spec = importlib.util.spec_from_file_location('pv_usb_common', Path(__file__).with_name('agent-expert-pv-usb-common.py'))
  U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
  B = U.old()
  provenance, capture = U.verify_capture()
  from tinygrad import Device
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program_context
  assert {v.key: str(v.value) for v in to_program_context} == capture['runtime']['compiler_context']
  E, C, K = (U.load(n) for n in ('agent-expert-gemm', 'agent-gemm-compile', 'agent-expert-pv-kernel'))
  native = json.loads(B.CPU.read_text())
  report = dict(provenance, complete=False, gpu_executed=False, target=U.TARGET, compiler_cases={},
    model_predictions=0, source_sha256={}, saved_capture_sha256=U.CAPTURE_SHA,
    native_target='AMD:LLVM:gfx1200', native_cpu_report_sha256=U.sha(B.CPU.read_bytes()))
  out = args.output.parent/'agent-expert-pv-usb-compiler'; out.mkdir(exist_ok=True)
  controls, renderers = {}, U.renderers()
  for name, shape in K.CASES.items():
    row = report['compiler_cases'][name] = {}
    for mode, renderer in renderers.items():
      row[mode] = {}
      for label in U.LABELS:
        ast, program = U.compile_program(renderer, shape, label)
        data = B.describe(program, ast)
        assert data['target'] == U.TARGET
        original = native['compiler_cases'][name][mode][label]
        differences = [k for k in original if original[k] != data[k]]
        assert set(differences) == {'program_key', 'target', 'program_info'}, differences
        row[mode][label] = data
        controls[name, mode, label] = ast, program
        stem = out/f'agent-expert-pv-usb-{name}-{mode}-{label}'
        stem.with_suffix('.ll').write_text(program.src[2].arg)
        stem.with_suffix('.hsaco').write_bytes(program.src[3].arg)
        stem.with_suffix('.s').write_text(C.disassemble(program.src[3].arg))
        B.write(stem.with_suffix('.elf.json'), U.elf(program.src[3].arg))
        if name == 'expert':
          params = [UOp.param(slot, x.dtype, x.numel(), device='AMD')
                    for slot, x in zip((2, 0, 1), program.kernel_params, strict=True)]
          proof = E.prove_cpu_queue([program.call(*params)]*40, device='AMD')
          assert proof['ordered_execution_barriers'] == 39 and proof['output_input_slot'] == 2
          report.setdefault('serial_40_call_proofs', {})[mode+'/'+label] = proof
        print(json.dumps({'case': name, 'mode': mode, 'label': label, 'target': data['target'], 'binary': data['binary_sha256']}), flush=True)
    for label in U.LABELS:
      a, b = (controls[name, mode, label][1] for mode in U.MODES)
      assert C.disassemble(a.src[3].arg) == C.disassemble(b.src[3].arg)
      assert len(a.src[3].arg) == len(b.src[3].arg)
      assert sum(x != y for x, y in zip(a.src[3].arg, b.src[3].arg, strict=True)) == 2
  ast, program = controls['expert', 'worker-cu', 'original']
  recorded = U.captured_program(ast, program, capture)
  assert U.elf(recorded.src[3].arg) == U.elf(U.STEM.with_suffix('.hsaco').read_bytes())
  report.update(all_actual_capture_fields_exact=True, saved_ast_text_exact=True, saved_source_exact=True,
    saved_binary_exact=True, saved_elf_abi_exact=True, saved_program_key_exact=True,
    same_instructions_both_modes=True, actual_capture_calls={'pv': 84, 'qk': 84, 'softmax': 84})
  expected = capture['actual_reference']
  rejected = []
  def reject(name, fn):
    try: fn()
    except AssertionError: rejected.append(name)
    else: raise AssertionError('Invalid binding accepted: '+name)
  for field in expected:
    changed = copy.deepcopy(expected); changed[field] = None
    reject(field, lambda changed=changed: U.validate_description(B.describe(recorded, ast), changed))
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  for target in ('AMD:LLVM:gfx1200', 'PCI+AMD:LLVM:gfx1200', 'USB+AMD:LLVM:gfx1201', 'USB:1+AMD:LLVM:gfx1200'):
    r = AMDLLVMRenderer(Target.parse(target))
    reject(target, lambda r=r: U.validate_renderer(r, 'worker-cu'))
  reject('wrong compiler mode', lambda: U.validate_renderer(renderers['parent-wgp'], 'worker-cu'))
  from types import SimpleNamespace
  reject('renderer subclass/stand-in', lambda: U.validate_renderer(SimpleNamespace(target=Target.parse(U.TARGET)), 'worker-cu'))
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(recorded.src[3].arg)
  offset = next(s.header.sh_offset for s in sections if s.name == '.text')
  corrupt = bytearray(recorded.src[3].arg); corrupt[offset] ^= 1
  corrupt = bytes(corrupt)
  reject('ELF byte corruption', lambda: U.validate_description(B.describe(recorded.replace(
    src=(*recorded.src[:3], recorded.src[3].replace(arg=corrupt))), ast), expected))
  assert not Device._opened_devices
  report.update(negative_tests_passed=True, rejected_corruptions=rejected, complete=True)
  report['source_sha256'] = {str(p): U.sha(p.read_bytes()) for p in Path(__file__).parent.glob('agent-expert-pv-usb-*.py')}
  B.write(args.output, report)


if __name__ == '__main__': main()
