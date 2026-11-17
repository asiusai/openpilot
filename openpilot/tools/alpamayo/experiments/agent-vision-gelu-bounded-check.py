"""Independent CPU execution, gfx1200 compile, and immutable-evidence preflight."""
import argparse
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def module():
  spec = importlib.util.spec_from_file_location('bounded_run', HERE/'agent-vision-gelu-bounded-run.py')
  result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
  return result


def oracle(a, b, fmt, block=4096):
  import numpy as np
  def finite(values):
    if fmt == 'f32': return np.isfinite(values.view(np.float32))
    if fmt == 'bf16': return np.isfinite((values.astype(np.uint32) << 16).view(np.float32))
    return ~np.isin(values, [127,255])  # E4M3FN has two NaN codes and no infinity code.
  mismatch, bad_a, bad_b = a != b, ~finite(a), ~finite(b)
  return np.array([int(mismatch[i:i+block].any()) | (int(bad_a[i:i+block].any()) << 1) | (int(bad_b[i:i+block].any()) << 2)
                   for i in range(0,len(a),block)], np.uint32)


def host(output):
  import numpy as np
  from tinygrad import Tensor, TinyJit, Device, dtypes
  S = module(); V, M = S.V, S.M
  assert os.environ['DEV'] == 'CPU'
  original = json.loads(S.OLD_PROTOCOL.read_text())
  records = []
  def tensor(bits, dt): return M.tensor_buffer('CPU', bits.shape, dt, bits.tobytes())
  def execute(a, b, fmt):
    flags = tensor(np.full((len(a)+V.BLOCK-1)//V.BLOCK, 0xffffffff, np.uint32), dtypes.uint32)
    result = Tensor.custom_kernel(flags, tensor(a,V.uint_dtype(fmt)), tensor(b,V.uint_dtype(fmt)), fxn=partial(V.compare,fmt=fmt))[0].realize()
    actual = result.numpy()
    np.testing.assert_array_equal(actual, oracle(a,b,fmt))
    assert not (actual & np.uint32(0xfffffff8)).any()
    return actual
  def execute_pattern(a, word, fmt):
    flags = tensor(np.full((len(a)+V.BLOCK-1)//V.BLOCK,0xffffffff,np.uint32),dtypes.uint32)
    control = tensor(np.array([word,0],np.uint32),dtypes.uint32)
    result = Tensor.custom_kernel(flags,tensor(a,V.uint_dtype(fmt)),control,fxn=partial(V.pattern,fmt=fmt))[0].realize()
    np.testing.assert_array_equal(result.numpy(),oracle(a,np.full_like(a,word),fmt))
  for row in original['plans']['original']['layout'][1:]:
    name, n = row['name'], int(np.prod(row['shape']))
    fmt, dt = V.FORMATS[name], V.uint_dtype(V.FORMATS[name])
    npdt = {1:np.uint8,2:np.uint16,4:np.uint32}[dt.itemsize]
    native_dt = {'dtypes.bf16':dtypes.bfloat16,'dtypes.f32':dtypes.float32,'dtypes.fp8e4m3':dtypes.fp8e4m3}[row['dtype']]
    native = M.tensor_buffer('CPU',row['shape'],native_dt,np.zeros(n,npdt).tobytes())
    alias = V.raw_view(native,fmt)
    control = tensor(np.array([V.ONE[fmt],0],np.uint32),dtypes.uint32)
    alias_fill = TinyJit(lambda target,ctl:Tensor.custom_kernel(target,ctl,fxn=V.fill)[0].realize())
    for state,word in enumerate((V.ONE[fmt],V.ONE[fmt],V.ONE[fmt],V.POISON[fmt],0,1,V.MASK[fmt],V.ONE[fmt]^1,V.ONE[fmt])):
      M.write_buffer(control,np.array([word,0],np.uint32).tobytes())
      actual = alias_fill(alias,control)
      assert actual.uop.buffer.base is native.uop.buffer.base and actual.uop.buffer.offset == native.uop.buffer.offset
      native_bytes = bytes(native.uop.buffer.host.view(size=native.nbytes(),fmt='B')[:])
      np.testing.assert_array_equal(np.frombuffer(native_bytes,npdt),np.full(n,word,npdt))
    rng = np.random.default_rng(n+dt.itemsize)
    a = rng.integers(0, 1 << (8*dt.itemsize), n, dtype=npdt)
    if fmt == 'f32': a[~np.isfinite(a.view(np.float32))] = V.ONE[fmt]
    elif fmt == 'bf16': a[~np.isfinite((a.astype(np.uint32)<<16).view(np.float32))] = V.ONE[fmt]
    else: a[np.isin(a,[127,255])] = V.ONE[fmt]
    cases = []
    execute(a,a.copy(),fmt); cases.append('equal-random-finite')
    for index in V.positions(n):
      b = a.copy(); b[index] ^= npdt(1)
      execute(a,b,fmt); cases.append(f'one-bit-{index}')
    b = a.copy()
    for start in range(0,n,V.BLOCK): b[min(start+V.BLOCK-1,n-1)] ^= npdt(1)
    execute(a,b,fmt); cases.append('one-mutation-in-every-block-including-tail')
    execute(a,a^npdt(1),fmt); cases.append('every-element-mutated')
    zero = np.zeros(n,npdt)
    execute(zero,np.full(n,1 << (dt.itemsize*8-1),npdt),fmt); cases.append('signed-zero-bits')
    execute(np.ones(n,npdt),np.ones(n,npdt),fmt); cases.append('subnormal-finite')
    for word in set((V.MASK[fmt],V.POISON[fmt],V.MASK[fmt] | (1 << (dt.itemsize*8-1)))):
      bad = np.full(n,word,npdt)
      for tag, aa, bb in (('equal',bad,bad),('left',bad,a),('right',a,bad)):
        execute(aa,bb,fmt); cases.append(f'{tag}-nonfinite-{word}')
    out = tensor(np.zeros(n,npdt),dt)
    control = tensor(np.array([V.POISON[fmt],n-1],np.uint32),dtypes.uint32)
    out = Tensor.custom_kernel(out,control,fxn=V.fill)[0].realize()
    expected = np.full(n,V.POISON[fmt],npdt)
    np.testing.assert_array_equal(out.numpy(),expected)
    M.write_buffer(control,np.array([V.ONE[fmt],n-1],np.uint32).tobytes())
    out = Tensor.custom_kernel(out,control,fxn=V.poke)[0].realize()
    expected[-1] = V.ONE[fmt]
    np.testing.assert_array_equal(out.numpy(),expected)
    source = tensor(a,dt)
    out = Tensor.custom_kernel(out,source,fxn=V.copy)[0].realize()
    np.testing.assert_array_equal(out.numpy(),a)
    # The actual CPU TinyJit executes mutable inputs, poisoned flags, and all tail writes.
    flags = tensor(np.full((n+V.BLOCK-1)//V.BLOCK,0xffffffff,np.uint32),dtypes.uint32)
    left, right = tensor(a,dt), tensor(a.copy(),dt)
    poison = tensor(np.array([0xffffffff,0],np.uint32),dtypes.uint32)
    def compare_jit(f,x,y,p):
      f = Tensor.custom_kernel(f,p,fxn=V.fill)[0].realize()
      return Tensor.custom_kernel(f,x,y,fxn=partial(V.compare,fmt=fmt))[0].realize()
    jit = TinyJit(compare_jit)
    for _ in range(3): np.testing.assert_array_equal(jit(flags,left,right,poison).numpy(),oracle(a,a,fmt))
    for state in range(6):
      b = a.copy(); b[V.positions(n)[state % len(V.positions(n))]] ^= npdt(1)
      if state == 5: b = a.copy()
      M.write_buffer(right,b.tobytes())
      np.testing.assert_array_equal(jit(flags,left,right,poison).numpy(),oracle(a,b,fmt))
    # Independently execute the complete-pattern validator, including mutable controls.
    for word in (V.ONE[fmt],V.POISON[fmt],V.MASK[fmt],0,1,1 << (8*dt.itemsize-1)):
      pattern = np.full(n,word,npdt)
      execute_pattern(pattern,word,fmt)
      for index in V.positions(n):
        changed = pattern.copy(); changed[index] ^= npdt(1)
        execute_pattern(changed,word,fmt)
    pattern = np.full(n,V.ONE[fmt],npdt)
    source = tensor(pattern,dt)
    mutable = tensor(np.array([V.ONE[fmt],0],np.uint32),dtypes.uint32)
    def pattern_jit(f,x,c,p):
      f = Tensor.custom_kernel(f,p,fxn=V.fill)[0].realize()
      return Tensor.custom_kernel(f,x,c,fxn=partial(V.pattern,fmt=fmt))[0].realize()
    pj = TinyJit(pattern_jit)
    for _ in range(3): np.testing.assert_array_equal(pj(flags,source,mutable,poison).numpy(),oracle(pattern,pattern,fmt))
    for state,word in enumerate((V.ONE[fmt]^1,V.POISON[fmt],0,1,V.MASK[fmt],V.ONE[fmt])):
      changed = pattern.copy()
      if state != 5: changed[V.positions(n)[state % len(V.positions(n))]] = npdt(word)
      M.write_buffer(source,changed.tobytes())
      M.write_buffer(mutable,np.array([word,0],np.uint32).tobytes())
      np.testing.assert_array_equal(pj(flags,source,mutable,poison).numpy(),oracle(changed,np.full(n,word,npdt),fmt))
    records.append(dict(output=name,shape=row['shape'],dtype=fmt,elements=n,cases=cases,
      all_flags_match_independent_oracle=True,fill_poke_copy_full_bytes_exact=True,actual_cpu_jit_mutations_and_restoration=True,
      complete_pattern_matches_independent_oracle=True,actual_cpu_pattern_jit_input_and_control_mutations=True,
      raw_alias_mutates_original_native_dtype_storage=True,raw_alias_actual_cpu_jit_six_changes_and_restoration=True))
  # Independent finite classification across every BF16/FP8 encoding and float32 exponent families.
  for fmt, values in (
      ('bf16',np.arange(65536,dtype=np.uint16)), ('fp8',np.arange(256,dtype=np.uint8)),
      ('f32',np.array([(sign << 31)|(exp << 23)|mantissa for sign in range(2) for exp in range(256)
                      for mantissa in (0,1,0x7fffff)],np.uint32))):
    execute(values,values.copy(),fmt)
    for bit in range(values.dtype.itemsize*8):
      a = np.full(4097,V.ONE[fmt],values.dtype); b = a.copy(); b[-1] ^= values.dtype.type(1 << bit)
      execute(a,b,fmt)
  # Extra small and partial domains exercise all mask transitions, including a one-element output.
  for fmt in ('bf16','fp8','f32'):
    npdt = {'bf16':np.uint16,'fp8':np.uint8,'f32':np.uint32}[fmt]
    for n in (1,31,32,4095,4096,4097):
      a = np.full(n,V.ONE[fmt],npdt); b=a.copy(); b[-1]^=npdt(1)
      execute(a,b,fmt)
  assert all(name in ('CPU','NPY') for name in Device._opened_devices)
  S.M.save(output,dict(completed=True,gpu_executed=False,opened_devices=sorted(Device._opened_devices),
    records=records,every_bf16_fp8_encoding=True,float32_exponent_and_sign_family=True,every_storage_bit_in_tail=True,
    unsigned_integer_checks_only=True,model_numeric_kernels_executed=False))
  print(json.dumps(dict(completed=True,gpu_executed=False,actual_output_shapes=len(records))),flush=True)


def helper_jit_checks(S,programs,layout,previous):
  from tinygrad.engine import jit as jitmod
  from tinygrad import tensor as tensormod
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  dispatch = S.V.Dispatch(programs,S.M)
  records,seen = [],[]
  def compile_linear(linear,**kwargs):
    if linear.src:
      assert len(linear.src) == 1 and linear.src[0].body.op is Ops.PROGRAM
      hcq2._build_queues(previous.queue_context(linear.src))
    return linear
  def run_linear(linear,var_vals=None,input_uops=(),**kwargs):
    if input_uops:
      assert all(u.op is Ops.BUFFER for u in input_uops)
      seen.append(bytes(input_uops[-1].buffer.host.view(size=1,fmt='B')[:]))
  with previous.patched(jitmod,compile_linear=compile_linear,link_linear=lambda value:value,run_linear=run_linear), \
       previous.patched(tensormod,run_linear=lambda *args,**kwargs:None):
    for name,(_,specs) in S.V.specifications(layout).items():
      values = tuple(previous.mutable_tensor(shape,dt) for shape,dt in specs)
      dispatch(name,*values)
      assert dispatch.jits[name].captured is not None
      assert dispatch.identities[name] == S.M.program_record(programs[name])
      for value in range(1,7):
        values[-1].uop.buffer.host.view(size=1,fmt='B')[:] = bytes([value])
        dispatch(name,*values)
      assert len(set(seen[-6:])) == 6
      records.append(dict(program=name,actual_jit_frontend=True,numeric_execution_mocked=True,
                          one_compiled_program=True,exact_parameter_mapping=True,changed_buffer_replays=6))
  return records


def prepare(directory):
  S = module(); M,V = S.M,S.V
  original,old,evidence_files = S.evidence()
  env = M.environment(original,cpu=True)
  env['CACHEDB'] = str(HERE/'agent-vision-gelu-bounded-cache.db')
  os.environ.clear(); os.environ.update(env)
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  renderer = AMDLLVMRenderer(Target('AMD','LLVM','gfx1200',interface='USB'))
  renderer.compiler = M.load('wgp_compiler').WGPCompiler('gfx1200')
  plans = {name:M.build_plan(name,renderer) for name in ('original','candidate')}
  S.numeric_audit(plans,original,old)
  helpers,helper_records = V.compile_all(plans['original']['layout'],renderer,M)
  fixtures = []
  for state in range(7):
    data = M.fixture(state)
    row = dict(state=M.STATES[state],sha256=M.sha(data.tobytes()),bytes=data.nbytes)
    if state < 5: assert row['sha256'] == old['single_controls'][state]['input_sha256']
    fixtures.append(row)
  assert M.fixture(7).tobytes() == M.fixture(0).tobytes()
  # Reuse immutable CPU queue/mutable-input checks without modifying their files or reports.
  previous_check = S.local('agent-vision-gelu-micro-check')
  queue = {name:previous_check.queue_checks(M,plan) for name,plan in plans.items()}
  jit = {name:previous_check.jit_checks(M,plan) for name,plan in plans.items()}
  helper_jit = helper_jit_checks(S,helpers,plans['original']['layout'],previous_check)
  assert not Device._opened_devices
  directory.mkdir(parents=True,exist_ok=True)
  protocol_path = directory/'agent-vision-gelu-bounded-protocol.json'
  assert not protocol_path.exists(), 'Do not overwrite a frozen retry protocol'
  for name,program in helpers.items():
    (directory/(name+'.source')).write_text(program.src[2].arg)
    (directory/(name+'.hsaco')).write_bytes(program.src[3].arg)
  child = directory/'agent-vision-gelu-bounded-host.json'
  host_env = dict(env,DEV='CPU',ALLOW_DEVICE_USAGE='1')
  result = subprocess.run([sys.executable,str(Path(__file__)),'--host','--output',str(child)],env=host_env,
                          capture_output=True,text=True,timeout=240)
  (directory/'agent-vision-gelu-bounded-host.log').write_text(result.stdout+result.stderr)
  assert result.returncode == 0, result.stdout+result.stderr
  host_report = json.loads(child.read_text())
  assert host_report['completed'] and not host_report['gpu_executed']
  sources = dict(original['source_sha256'],**evidence_files)
  for path in (Path(__file__),HERE/'agent-vision-gelu-bounded-run.py',HERE/'agent-vision-gelu-bounded-validate.py',child):
    sources[str(path)] = M.digest(path)
  protocol = dict(schema=1,shape=list(M.SHAPE),operations=40,states=list(M.STATES),timeout_seconds=300,
    source_sha256=sources,old_protocol_sha256=S.OLD_PROTOCOL_SHA,old_report_sha256=S.OLD_REPORT_SHA,
    numeric_plans=original['plans'],helper_programs=helper_records,fixtures=fixtures,
    diagnosis=S.diagnosis(original),native_gelu_equality_proven=False,model_hook_installed=False,adoption_allowed=False,
    evidence_reuse_condition='Exact source, active numeric binary, launch, queue-map, and input-fixture identity. Prior result remains incomplete.',
    prior_controls_substitute_for_retry_checks=False)
  S.verify(protocol)
  M.save(protocol_path,protocol)
  M.save(directory/'agent-vision-gelu-bounded-cpu.json',dict(completed=True,gpu_executed=False,native_gelu_equality_proven=False,
    protocol_sha256=M.digest(protocol_path),old_evidence_unchanged=True,numeric_sources_and_binaries_unchanged=True,
    helper_compilation=helper_records,queue_proofs=queue,mutable_jit_frontend_mocks=jit,helper_jit_frontend_mocks=helper_jit,host_execution=host_report,
    diagnosis=protocol['diagnosis'],parent_gpu_not_run=True))
  print(json.dumps(dict(gpu_executed=False,completed=True,helper_programs=len(helpers),protocol=str(protocol_path),
                       protocol_sha256=M.digest(protocol_path),diagnosis=protocol['diagnosis'])),flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--freeze',type=Path)
  parser.add_argument('--host',action='store_true')
  parser.add_argument('--output',type=Path)
  args = parser.parse_args()
  if args.host:
    assert args.output and not args.freeze
    host(args.output)
  else:
    assert args.freeze and not args.output
    prepare(args.freeze)
