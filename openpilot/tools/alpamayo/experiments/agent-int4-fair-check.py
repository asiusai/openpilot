"""Independent controls, actual compiler/HCQ proofs, and parent-only GPU runner."""
from dataclasses import replace
from types import SimpleNamespace
import math
import re
import statistics
import subprocess
import time
import numpy as np


def placeholders(items):
  from tinygrad.uop.ops import UOp
  return [UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(items)]


def compiler_record(f, program):
  binary = program.src[3].arg
  text = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  names = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
           'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode')
  resource = {k: int(v) for k, v in re.findall(r'\.('+'|'.join(names)+r'):\s*(\d+)', text)}
  assert resource['wavefront_size'] == 32 and resource['workgroup_processor_mode'] == 1
  return dict(name=program.src[0].arg.name, binary_sha256=f.sha(binary), binary_bytes=len(binary),
    source_sha256=f.sha(program.src[2].arg.encode()), global_size=program.arg.global_size,
    local_size=program.arg.local_size, workgroups=math.prod(program.arg.global_size), resources=resource,
    output_argument_indices=list(program.arg.outs), input_argument_indices=list(program.arg.ins))


def original_operation_capture(f, x, weight, xs, ws, renderer, matrix=None):
  """Capture the unchanged original complete operation before rebinding scratch."""
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.engine.realize import capturing, get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  qp, amd, _, _ = f.modules();recorder = SimpleNamespace(calls=[])
  recorder.add_linear = lambda linear: recorder.calls.extend(linear.src)
  assert not capturing
  capturing.append(recorder)
  try:
    q, scale = qp.quantize(x, xs);assert scale is xs
    out = Tensor.custom_kernel(Tensor.empty(x.shape[0], weight.shape[0], device=x.device, dtype=dtypes.bfloat16),
                               q, weight, xs, ws, fxn=amd._gemm if matrix is None else matrix)[0].realize()
  finally: assert capturing.pop() is recorder
  assert len(recorder.calls) == 2, 'Original complete FP8 operation no longer contains exactly quantize+pack and GEMM'
  roles = {value.uop.storage_base: role for role, value in [('BF16-input', x), ('resident-packed-weight', weight),
    ('fixed-activation-scalar', xs), ('fixed-weight-scalar', ws), ('packed-activation', q), ('BF16-output', out)]}
  assert len(roles) == 6
  result = []
  for phase, call in enumerate(recorder.calls):
    call = call.without_after;program = to_program(call.body, renderer);actual = program.call(*call.src[1:])
    outs, ins = get_call_outs_ins(actual);arguments = []
    for arg in actual.src[1:]:
      base, lane, offset = unwrap_lane(arg);assert lane is None
      arguments.append(dict(role=roles[base], byte_offset=offset, bytes=arg.max_numel()*arg.dtype.itemsize, dtype=str(arg.dtype)))
    wanted_out = {'packed-activation'} if phase == 0 else {'BF16-output'}
    wanted_in = {'BF16-input', 'fixed-activation-scalar'} if phase == 0 else {
      'packed-activation', 'resident-packed-weight', 'fixed-activation-scalar', 'fixed-weight-scalar'}
    assert {arguments[i]['role'] for i in outs} == wanted_out and {arguments[i]['role'] for i in ins} == wanted_in
    result.append(dict(compiler_record(f, program), actual_argument_roles=arguments))
  return dict(original_complete_kernel_calls=2, captured_public_quantize_and_original_packed_gemm=True, phases=result)


def compiled(f, shape):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import UOp
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'));renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  _, amd, kernel, _ = f.modules();m, n, k = shape
  qargs = placeholders([(((m+15)//16*16, k), dtypes.fp8e4m3), ((m, k), dtypes.bfloat16), ((), dtypes.float32)])
  original, _, roles = f.quant_template(m, k)
  direct = to_program(original, renderer);bound = to_program(f.quant_program(*qargs), renderer)
  assert direct.src[2].arg == bound.src[2].arg and direct.src[3].arg == bound.src[3].arg, 'Output binding changed original quantizer code'
  gargs = placeholders([((m, n), dtypes.bfloat16), (((m+15)//16*16, k), dtypes.fp8e4m3),
                       ((n, k), dtypes.fp8e4m3), ((), dtypes.float32), ((1,), dtypes.float32)])
  gemm = to_program(f.bind_renderer(amd._gemm, renderer)(*gargs), renderer)
  from tinygrad import Tensor
  actual_inputs = [Tensor.empty(*shape, dtype=dtype, device='CPU').realize() for shape, dtype in [
    ((m, k), dtypes.bfloat16), ((n, k), dtypes.fp8e4m3), ((), dtypes.float32), ((1,), dtypes.float32)]]
  original_complete = original_operation_capture(f, *actual_inputs, renderer, matrix=f.bind_renderer(amd._gemm, renderer))
  for captured, prg in zip(original_complete['phases'], (bound, gemm), strict=True):
    assert captured['binary_sha256'] == f.sha(prg.src[3].arg) and captured['source_sha256'] == f.sha(prg.src[2].arg.encode())
  iqargs = placeholders([(((m+15)//16, k//32, 32, 8), dtypes.uint8), ((m,), dtypes.float32), ((m, k), dtypes.bfloat16)])
  igargs = placeholders([((m, n), dtypes.bfloat16), (((m+15)//16, k//32, 32, 8), dtypes.uint8),
                        (((n+15)//16, k//32, 32, 8), dtypes.uint8), ((m,), dtypes.float32), ((n,), dtypes.float32)])
  iq = kernel.quantize_program(*iqargs, rows=m, width=k, wgp=True)
  ig = kernel.gemm_program(*igargs, rows=m, columns=n, width=k, wgp=True)
  # Use real compiled PROGRAMs and actual argument shapes with explicit PARAMs.
  def call(program, args, slots):
    return program.call(*(UOp.param(slot, arg.dtype, arg.numel(), device='CPU') for arg, slot in zip(args, slots, strict=True)))
  return dict(shape_mnk=shape, original_quantizer_captured_calls=1, original_quantizer_argument_roles=roles,
              original_complete_operation_capture=original_complete,
              original_quantizer_binary_identical_after_output_rebind=True,
              programs={name: compiler_record(f, p) for name, p in [('fp8-quant', bound), ('fp8-gemm', gemm), ('a4-quant', iq), ('a4-gemm', ig)]}), {
    'fp8': [call(bound, qargs, (4, 0, 2)), call(gemm, gargs, (5, 4, 1, 2, 3))]*f.CALLS,
    'a4w4': [call(iq, iqargs, (3, 4, 0)), call(ig, igargs, (5, 3, 1, 4, 2))]*f.CALLS}


def finalizer_proof(f, calls, variant, device):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp, Ops
  owner = SimpleNamespace(variant=variant, proofs=[])
  ctx = hcq2.BatchCtx([(c.without_after, device, 'COMPUTE:0') for c in calls], False)
  old = hcq2._build_queues
  with f.observe(owner): graph = hcq2._finalize_batch(ctx)
  assert hcq2._build_queues is old
  owner.jit = SimpleNamespace(captured=SimpleNamespace(_linear=UOp(Ops.LINEAR, src=(graph,))))
  return f.inspect(owner), owner, ctx


def rejects(label, fn, records):
  try: fn()
  except AssertionError as exc: records.append(dict(case=label, rejected=True, reason=str(exc)));return
  raise AssertionError('Invalid proof accepted: '+label)


def cpu_jit(f):
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.uop.ops import UOp, KernelInfo
  # The matrix stand-in has output argument 1. Its arithmetic is only a CPU
  # graph/control fixture; actual AMD kernels are separately compiled above.
  def matrix(q, out, w, xs, ws):
    r = UOp.range(out.numel(), 0)
    value = sum(v.flatten().index(0).load().float() for v in (q, w, xs, ws))
    return out.flatten().index(r).store(value.cast(out.dtype)).end(r).sink(arg=KernelInfo(name='agent_int4_fair_cpu_matrix'))
  m, n, k = 17, 48, 32;f.quant_template(m, k)
  x = Tensor.full((m, k), 2, dtype=dtypes.bfloat16, device='CPU').realize()
  w = Tensor.full((n, k), 2, dtype=dtypes.fp8e4m3, device='CPU').realize()
  xs = Tensor.ones((), dtype=dtypes.float32, device='CPU').realize()
  ws = Tensor.ones(1, dtype=dtypes.float32, device='CPU').realize()
  q = Tensor.empty(32, k, dtype=dtypes.fp8e4m3, device='CPU').realize()
  out = Tensor.empty(m, n, dtype=dtypes.bfloat16, device='CPU').realize()
  def run(x, w, xs, ws, q, out):
    for _ in range(f.CALLS):
      q = Tensor.custom_kernel(q, Tensor(x.uop.after(out.uop)), xs, fxn=f.quant_program)[0]
      out = Tensor.custom_kernel(q, out, w, xs, ws, fxn=matrix)[1]
    return out.realize()
  jit = TinyJit(run);args = (x, w, xs, ws, q, out)
  for _ in range(3): jit(*args)
  np.testing.assert_array_equal(jit(*args).float().numpy(), 6)
  f.copyin(x, f.bf(np.full((m, k), 4, np.float32)));f.copyin(out, np.full(out.nbytes(), 0xA5, np.uint8))
  np.testing.assert_array_equal(jit(*args).float().numpy(), 8)
  f.copyin(xs, np.array(2, np.float32));f.copyin(q, np.full(q.nbytes(), 0xA5, np.uint8))
  np.testing.assert_array_equal(jit(*args).float().numpy(), 7)
  f.copyin(ws, np.array([3], np.float32));f.copyin(out, np.full(out.nbytes(), 0xA5, np.uint8))
  np.testing.assert_array_equal(jit(*args).float().numpy(), 9)
  proof, _, _ = finalizer_proof(f, jit.captured._linear.src, 'fp8', 'CPU')
  assert proof['kernel_arguments'][1]['output_argument_indices'] == [1]
  return dict(mutation_and_poison_exact=True, activation_and_both_scalars_mutated=True,
              actual_CPU_TinyJit=True, output_argument_one_supported=True, proof=proof)


def cpu_checks(f):
  from tinygrad import Tensor, dtypes
  from tinygrad.dtype import float_to_fp8
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp, Ops
  qp, amd, kernel, oldcheck = f.modules();records = []
  table = f.decode_fp8(np.arange(127, dtype=np.uint8));mid = (table[:-1]+table[1:])*np.float32(.5)
  values = np.concatenate([table, mid, np.nextafter(mid, np.float32(0)), np.nextafter(mid, np.float32(np.inf))]);values = np.r_[values, -values]
  np.testing.assert_array_equal(f.encode_fp8(values), np.array([float_to_fp8(float(x), dtypes.fp8e4m3) for x in values], np.uint8))
  records.append(dict(case='independent-FP8-all-codepoints-midpoints-neighbors-signed-subnormals', elements=len(values)))
  for label, a, w in oldcheck.integer_cases():
    np.testing.assert_array_equal(f.unpack4(kernel.pack_unsigned_cpu(a), len(a)), a)
    np.testing.assert_array_equal(f.unpack4(kernel.pack_weights_cpu(w), len(w), True), w)
    integer = a.astype(np.int64)@w.astype(np.int64).T
    assert np.max(np.abs(integer)) < 2**24
    records.append(dict(case='independent-INT4-layout-'+label, integer_elements=int(integer.size)))
  bits = oldcheck.activation_bits(17, 32);q, s = f.quant4(bits)
  oq, os = oldcheck.quantize_cpu(bits);np.testing.assert_array_equal(q, oq);np.testing.assert_array_equal(s.view(np.uint32), os.view(np.uint32))
  np.testing.assert_array_equal(f.unpack4(kernel.pack_unsigned_cpu(q), len(q)), q)
  records.append(dict(case='A4-zero-ties-outlier-maxfinite-subnormal', exact=True))
  for m in (17, 64):
    raw = np.random.default_rng(m).integers(0, 255, (m, 64), dtype=np.uint8)
    np.testing.assert_array_equal(amd.pack(Tensor(raw, device='CPU')).numpy(), f.fp8_layout(raw))
    np.testing.assert_array_equal(f.fp8_layout(f.fp8_layout(raw), unpack=True, rows=m), raw)
    records.append(dict(case='independent-FP8-layout', rows=m))
  result = dict(controls=records, actual_CPU_capture=cpu_jit(f), compiler=[], negative_proofs=[])
  for shape in ((17, 48, 32), f.SHAPE):
    record, variants = compiled(f, shape);record['serial_proofs'] = {}
    for variant, calls in variants.items():
      proof, owner, ctx = finalizer_proof(f, calls, variant, 'CPU');record['serial_proofs'][variant] = dict(proof,
        offline_only=True, description='Actual AMD PROGRAMs finalized with CPU-labelled symbolic storage; no GPU linking or execution')
      bad = list(calls);c = bad[2];outs = c.body.arg.outs;src = list(c.src);arg = src[1+outs[0]]
      src[1+outs[0]] = UOp.param(999, arg.dtype, arg.numel(), device='CPU');bad[2] = c.replace(src=tuple(src))
      rejects(f'independent-scratch-{shape}-{variant}', lambda: finalizer_proof(f, bad, variant, 'CPU'), result['negative_proofs'])
      queues = hcq2._build_queues(ctx)
      broken = {k: [c for c in v if not (c.op is Ops.CALL and c.body.op is Ops.CUSTOM_FUNCTION and c.body.arg.name == 'exec_barrier')]
                for k, v in queues.items()}
      rejects(f'absent-barriers-{shape}-{variant}', lambda: f.queue_proof(ctx.batch, broken, variant), result['negative_proofs'])
      graph = owner.jit.captured._linear.src[0];aux = graph.arg.aux;badmeta = list(aux.kernels);entry = list(badmeta[0]);entry[5] = ();badmeta[0] = tuple(entry)
      corrupt = graph.replace(arg=replace(graph.arg, aux=replace(aux, kernels=tuple(badmeta))))
      holder = SimpleNamespace(proofs=owner.proofs, jit=SimpleNamespace(captured=SimpleNamespace(_linear=UOp(Ops.LINEAR, src=(corrupt,)))))
      rejects(f'empty-HCQ-argument-map-{shape}-{variant}', lambda: f.inspect(holder), result['negative_proofs'])
      # Corrupt the actual packed-input binding, not a hard-coded output slot.
      bad = list(calls);c = bad[1];src = list(c.src);packed_slot = 4 if variant == 'fp8' else 3
      j, = [j for j in c.body.arg.ins if hcq2.unwrap_lane(c.src[1+j])[0].arg.slot == packed_slot]
      arg = src[1+j];src[1+j] = UOp.param(998, arg.dtype, arg.numel(), device='CPU');bad[1] = c.replace(src=tuple(src))
      rejects(f'absent-packed-RAW-{shape}-{variant}', lambda: finalizer_proof(f, bad, variant, 'CPU'), result['negative_proofs'])
      if variant == 'a4w4':
        bad = list(calls);c = bad[0];body = c.body.replace(arg=replace(c.body.arg, outs=(0,)))
        bad[0] = c.replace(src=(body, *c.src[1:]))
        rejects(f'absent-row-scale-write-{shape}', lambda: finalizer_proof(f, bad, variant, 'CPU'), result['negative_proofs'])
    result['compiler'].append(record)
  return result


def upload(array, dtype=None):
  from tinygrad import Tensor
  data = np.array(array, copy=True)
  # A zero-dimensional NumPy input becomes a CONST on this pinned runtime.
  # Scale kernels require a mutable buffer, including the scalar scale.
  value = Tensor(data.reshape(1)).reshape(()) if data.ndim == 0 else Tensor(data)
  return (value if dtype is None else value.bitcast(dtype)).realize()


def sample_axes(m, n):
  return np.unique([0, 1, 2, 3, 7, 15, 16, m//2, m-1]), np.unique([0, 1, 7, 15, 16, n//2, n-1])


def fp8_numerical_samples(f, codes, weights, scale, weight_scale, result):
  """Independent FP64 dot plus conservative FP32 reduction/BF16 rounding bound.

  Full-bit implementation parity is checked separately. This bound neither
  guesses WMMA summation order nor falsely demands ideal FP64 equals FP32.
  """
  r, c = sample_axes(len(codes), len(weights));a = f.decode_fp8(codes[r]).astype(np.float64);w = f.decode_fp8(weights[c]).astype(np.float64)
  dot = a@w.T;abssum = np.abs(a)@np.abs(w).T
  product = np.float32(scale*weight_scale);ideal = dot*np.float64(product)
  u = 2.**-24;k = codes.shape[1];gamma = (k*u)/(1-k*u)
  accum_error = gamma*abssum*abs(float(product)) + np.abs(ideal)*u
  tolerance = accum_error + (np.abs(ideal)+accum_error)*2.**-8 + 2.**-126
  observed = f.unbf(result[np.ix_(r, c)]).astype(np.float64);error = np.abs(observed-ideal)
  assert np.isfinite(observed).all() and np.all(error <= tolerance), (float(error.max()), float(tolerance.max()))
  return dict(elements=int(dot.size), oracle='FP64 dot; conservative FP32 K-addition bound plus BF16 rounding',
              max_absolute_error=float(error.max()), max_allowed_bound=float(tolerance.max()))


def gpu_case(f, weights, bits, actual=False):
  from tinygrad import Tensor, Device, dtypes
  qp, amd, kernel, oldcheck = f.modules();m, k = bits.shape;n = len(weights['fp8'])
  x = upload(bits, dtypes.bfloat16);wf = upload(weights['fp8'], dtypes.fp8e4m3)
  xs = upload(np.array(weights['activation_scale'], np.float32));wfs = upload(weights['fp8_scale'])
  wi = upload(weights['w4']);wis = upload(weights['w4_scale'])
  inputs = {'fp8': (x, wf, xs, wfs), 'a4w4': (x, wi, wis)}
  batches = {v: f.Batch(v, (m, n, k)) for v in inputs}
  offline, _ = compiled(f, (m, n, k))
  original_capture = original_operation_capture(f, x, wf, xs, wfs, Device[x.device].renderer)
  for capture, name in zip(original_capture['phases'], ('fp8-quant', 'fp8-gemm'), strict=True):
    assert capture['binary_sha256'] == offline['programs'][name]['binary_sha256']
    assert capture['source_sha256'] == offline['programs'][name]['source_sha256']
  for v, batch in batches.items():
    for _ in range(3): batch(*inputs[v])
    proof = f.inspect(batch)
    for phase, name in enumerate(('fp8-quant', 'fp8-gemm') if v == 'fp8' else ('a4-quant', 'a4-gemm')):
      actual_program = proof['kernel_arguments'][phase];expected_program = offline['programs'][name]
      for field in ('name', 'source_sha256', 'binary_sha256'):
        assert actual_program[field] == expected_program[field], ('Actual captured program differs from CPU-compiled original', v, phase, field)
  original = bits.copy();states = ['original', 'activation-outlier', 'scales', 'restored'];records = []
  previous = None
  fp8_plain_weights = f.fp8_layout(weights['fp8'], unpack=True)
  # Four states exercise the same captured 40-call graph. Immutable weight
  # payloads stay resident. Scale mutation uses temporary device scalar/vector
  # contents and is restored before any timed block.
  for state in states:
    current = original.copy();ascalar = weights['activation_scale'];fscale = weights['fp8_scale'].copy();iscale = weights['w4_scale'].copy()
    if state == 'activation-outlier':
      current[-1, -1] = f.bf(np.float32(128));current[1, 7] = f.bf(np.float32(.125));current[0] = 0
    if state == 'scales':
      ascalar = np.float32(ascalar*1.25);fscale *= np.float32(.75);iscale *= np.float32(1.25)
    f.copyin(x, current);f.copyin(xs, np.array(ascalar, np.float32));f.copyin(wfs, fscale);f.copyin(wis, iscale)
    result = {};quant = {};scales = {}
    for variant, batch in batches.items():
      f.poison(batch);result[variant] = f.read(batch(*inputs[variant]), np.uint16)
      quant[variant] = f.read(batch.q, np.uint8)
      if variant == 'a4w4': scales[variant] = f.read(batch.scales, np.float32)
    direct_q, direct_scale = qp.quantize(x, xs);assert direct_scale is xs
    np.testing.assert_array_equal(quant['fp8'], f.read(direct_q, np.uint8))
    direct_out = Tensor.custom_kernel(Tensor.empty(m, n, dtype=dtypes.bfloat16), direct_q, wf, xs, wfs, fxn=amd._gemm)[0].realize()
    np.testing.assert_array_equal(result['fp8'], f.read(direct_out, np.uint16))
    # Hardware reciprocal is isolated from either candidate. Every subsequent
    # FP8 conversion and packed address is checked by independent CPU code.
    inverse = np.float32(f.read(xs.reciprocal().realize(), np.float32))
    fq = f.encode_fp8(np.multiply(f.unbf(current), inverse, dtype=np.float32))
    np.testing.assert_array_equal(quant['fp8'], f.fp8_layout(fq))
    aq, ascales = f.quant4(current)
    np.testing.assert_array_equal(f.unpack4(quant['a4w4'], m), aq)
    np.testing.assert_array_equal(scales['a4w4'].view(np.uint32), ascales.view(np.uint32))
    # Independent FP8 integer WMMA oracle is used only for INT4 correctness,
    # never as the timed selected-model FP8 policy.
    af = upload(f.fp8_layout(f.encode_fp8(aq)), dtypes.fp8e4m3)
    bw = upload(f.fp8_layout(f.encode_fp8(weights['w4_codes'])), dtypes.fp8e4m3)
    sx = upload(ascales)
    raw_ref = oldcheck.gpu_reference(af, bw, sx, wis, raw=True).realize()
    raw = f.read(raw_ref, np.float32)
    r, c = sample_axes(m, n) if actual else (np.arange(m), np.arange(n))
    integer = aq[r].astype(np.int64)@weights['w4_codes'][c].astype(np.int64).T
    np.testing.assert_array_equal(raw[np.ix_(r, c)], integer)
    assert np.max(np.abs(raw)) <= 120*k and np.isfinite(raw).all()
    expected = f.bf(np.multiply(raw, np.multiply(ascales[:, None], iscale[None, :], dtype=np.float32), dtype=np.float32))
    np.testing.assert_array_equal(result['a4w4'], expected)
    fpcheck = fp8_numerical_samples(f, fq, fp8_plain_weights, ascalar, fscale[0], result['fp8'])
    if not actual:
      # Power-of-two scalars and integer-representable FP8 operands yield exact
      # sums in FP32, independently checking every small FP8 BF16 result.
      a = f.decode_fp8(fq).astype(np.float64);w = f.decode_fp8(fp8_plain_weights).astype(np.float64)
      direct = f.bf((a@w.T).astype(np.float32)*np.float32(ascalar*fscale[0]))
      np.testing.assert_array_equal(result['fp8'], direct)
    digests = {v: f.sha(o.tobytes()) for v, o in result.items()}
    if state == 'original': previous = digests
    elif state == 'restored': assert digests == previous
    else: assert all(digests[v] != previous[v] for v in result), 'Mutation failed to change a complete output'
    records.append(dict(state=state, fp8_packed_bytes_exact=True, fp8_full_BF16_equal_original=True,
      fp8_independent_numeric=fpcheck, isolated_reciprocal_fp32_bits=int(np.asarray(inverse).view(np.uint32)),
      a4_codes_and_row_scales_exact=True, int4_full_BF16_equal_independent_integer_oracle=True,
      int4_CPU_int64_spots=int(integer.size), poison_overwritten=True, output_sha256=digests))
  return dict(shape_mnk=[m, n, k], mutation_checks=records,
              original_complete_operation_capture=original_capture,
              actual_captured_programs_equal_offline_originals=True,
              serial_proofs={v: f.inspect(b) for v, b in batches.items()}), batches, inputs


def gpu_checks(f, weights, bits, report, path):
  from tinygrad import Device
  from tinygrad.helpers import GlobalCounters, PROFILE, DEBUG, BEAM
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  assert not PROFILE and DEBUG < 2 and not BEAM, 'Timing forbids profiling, debug waits, and search'
  device = Device[Device.DEFAULT];assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
  old_compiler = device.renderer.compiler
  wgp_path = f.FROZEN/'wgp_compiler.py'
  assert f.sha(wgp_path.read_bytes()) == 'ccd7427848f13a8032e8af3c305fc8b2db3150242b3ff67a63f25acd52da8890'
  result = dict(complete=False, numerical_formats_compared_for_equality=False, selected_FP8_policy='fixed tensor activation scalar and fixed tensor weight scalar',
                compiler_mode='WGP for both; matches selected --llvm-wgp pipeline', timing=[])
  report['gpu'] = result;f.save(path, report)
  try:
    f.load(wgp_path).install(device)
    _, _, kernel, _ = f.modules();rng = np.random.default_rng(9181)
    m, n, k = 17, 48, 64
    codes = rng.integers(-7, 8, (n, k), dtype=np.int8)
    control = dict(fp8=f.fp8_layout(f.encode_fp8(codes)), fp8_scale=np.array([.25], np.float32),
      w4=kernel.pack_weights_cpu(codes), w4_codes=codes, w4_scale=np.full(n, .25, np.float32), activation_scale=np.float32(.125))
    cbits = f.bf((rng.integers(0, 31, (m, k)).astype(np.float32)/2)*np.float32(.125));cbits[0] = 0
    result['small_control'], _, _ = gpu_case(f, control, cbits)
    f.save(path, report);print('fair small control passed; actual layer-14 complete-operation gate', flush=True)
    result['actual'], batches, inputs = gpu_case(f, weights, bits, actual=True)
    f.save(path, report);print('fair actual gate passed; starting three alternating complete-operation blocks', flush=True)
    # Gates finish before timing. Uploads, poisoning, readbacks, compilation,
    # and all independent oracles are excluded from both timed boundaries.
    for block in range(3):
      order = ('fp8', 'a4w4') if block%2 == 0 else ('a4w4', 'fp8')
      for variant in order:
        batch = batches[variant];batch(*inputs[variant]);device.synchronize()
        before = GlobalCounters.kernel_count;start = time.perf_counter()
        batch(*inputs[variant]);device.synchronize()
        seconds = time.perf_counter()-start;count = GlobalCounters.kernel_count-before
        assert count == 2*f.CALLS, ('Incomplete operation boundary', variant, count)
        result['timing'].append(dict(block=block, order=list(order), variant=variant, seconds=seconds,
          complete_operations=f.CALLS, actual_kernel_calls=count, ms_per_complete_operation=seconds*1000/f.CALLS))
        f.save(path, report)
    fp = statistics.median(t['ms_per_complete_operation'] for t in result['timing'] if t['variant'] == 'fp8')
    a4 = statistics.median(t['ms_per_complete_operation'] for t in result['timing'] if t['variant'] == 'a4w4')
    # Post-timing outputs and resident weight payloads are checked again.
    for variant, batch in batches.items():
      expected = result['actual']['mutation_checks'][-1]['output_sha256'][variant]
      assert f.sha(f.read(batch.out, np.uint16).tobytes()) == expected
    assert f.sha(f.read(inputs['fp8'][1], np.uint8).tobytes()) == f.sha(weights['fp8'].tobytes())
    assert f.sha(f.read(inputs['a4w4'][1], np.uint8).tobytes()) == f.sha(weights['w4'].tobytes())
    np.testing.assert_array_equal(f.read(inputs['fp8'][2], np.float32).reshape(-1), np.array([weights['activation_scale']], np.float32))
    np.testing.assert_array_equal(f.read(inputs['fp8'][3], np.float32), weights['fp8_scale'])
    np.testing.assert_array_equal(f.read(inputs['a4w4'][2], np.float32), weights['w4_scale'])
    result.update(complete=True, fp8_complete_median_ms=fp, a4w4_complete_median_ms=a4, a4w4_over_fp8_complete_ratio=a4/fp,
      boundary='one replay of 40 serialized complete BF16-to-BF16 operations plus completion fence; amortized host-inclusive',
      per_launch_USB_timing=False, immutable_resident_weights_verified_after_timing=True,
      conclusion='Complete microbenchmark only. Parent decides whether separate model accuracy testing is worthwhile; no model-quality claim.')
    return result
  finally:
    device.renderer.compiler = old_compiler
