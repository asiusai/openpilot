"""CPU study and parent-only GPU checks for packed FP8 QK / BF16 PV.

Quality is measured against independent FP64 attention and never called exact.
Implementation gates compare conversion bytes and quantized-input BF16 outputs.
Timing includes the combined Q/K conversion and value packing, at 40 calls.
"""
import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


candidate = load('agent-attention-fp8')
helpers = load('agent-attention-check')


def provenance():
  paths = [Path(__file__), Path(candidate.__file__), Path(helpers.__file__),
    Path(candidate.baseline_module().__file__)]
  from tinygrad.runtime.support import compiler_amd
  paths.append(Path(compiler_amd.__file__))
  return dict(files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    tinygrad_revision=subprocess.check_output(['git', '-C', str(paths[-1].parents[3]), 'rev-parse', 'HEAD'], text=True).strip(),
    target='gfx1200', compiler_policy='Unmodified pinned HIPCompiler default options; no fast-math override.')


def bf16_bits_cpu(x):
  bits = np.asarray(x, dtype=np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def rounding_regression_cpu():
  """Independent arithmetic witnesses, not a software model of WMMA hardware."""
  fmaf = ctypes.CDLL('libm.so.6').fmaf
  fmaf.argtypes, fmaf.restype = [ctypes.c_float]*3, ctypes.c_float
  raw, scale, maximum = np.float32(-15.8331298828125), np.float32(72**-.5), np.float32(4.402106761932373)
  fused = np.float32(fmaf(float(raw), float(scale), -float(maximum)))
  separate = np.float32(np.float32(raw*scale)-maximum)
  probability = np.exp2(np.array([fused, separate], np.float32)*np.float32(1.4426950408889634))
  np.testing.assert_array_equal(bf16_bits_cpu(probability), np.array([0x3af8, 0x3af9], np.uint16))
  # Both operands below are exactly representable in E4M3 and BF16.
  # Different fused reduction group sizes are valid mathematical witnesses.
  # They are NOT a claim about the undocumented WMMA internal reduction tree.
  a = np.array([2.]*16+[2.**-9]*16, np.float32)
  b = np.array([4.]*16+[2.**-9]*16, np.float32)
  for values in (a, b):
    tensor = torch.from_numpy(values)
    assert torch.equal(tensor.to(torch.float8_e4m3fn).float(), tensor)
    np.testing.assert_array_equal(bf16_bits_cpu(values), tensor.to(torch.bfloat16).view(torch.uint16).numpy())
  def grouped_dot(left, right, group):
    result = np.zeros(right.shape[:-1], np.float32)
    for start in range(0, left.shape[-1], group):
      products = left[start:start+group].astype(np.float64)*right[..., start:start+group].astype(np.float64)
      result = (result.astype(np.float64)+products.sum(-1)).astype(np.float32)
    return result
  two, four = grouped_dot(a, b, 2), grouped_dot(a, b, 4)
  assert two == np.float32(128) and four == np.float32(128.00006103515625)
  # With normal FP32 intermediates, exact powers of two commute with a FIXED
  # reduction tree. Merely moving these scales cannot repair another tree.
  for group in (1, 2, 4, 8, 16):
    for exponent in (-10, 0, 10):
      lhs = grouped_dot(a, b, group)*np.float32(2.**(2*exponent))
      rhs = grouped_dot(a*np.float32(2.**exponent), b*np.float32(2.**exponent), group)
      np.testing.assert_array_equal(np.asarray(lhs).view(np.uint32), np.asarray(rhs).view(np.uint32))
  # Recreate the existing random fixture, preserving every preceding RNG draw.
  rng = np.random.default_rng(906080)
  for case in CONTROLS+TARGETS:
    q, k, _, qs, ks = inputs(case, rng)
    if case[0] != 'prefill2027':
      continue
    a = fp8_cpu(q[0, 8, 1753], qs[0, 8]).numpy()
    b = fp8_cpu(k[0, 4], ks[0, 4]).numpy()
    break
  serial, tile = grouped_dot(a, b, 1), grouped_dot(a, b, 16)
  assert serial[52] == np.float32(31.8414306640625) and tile[52] == np.float32(31.841432571411133)
  def online_probability(dots):
    scores, maximum, output = dots*np.float32(128**-.5), np.float32(-np.inf), []
    for start in range(0, 1760, 32):
      values = scores[start:start+32].copy()
      values[start+np.arange(len(values)) > 1753] = -np.inf
      maximum = np.maximum(maximum, values.max())
      output.extend(np.exp2(np.float32(np.float32(values-maximum)*np.float32(1.4426950408889634))))
    return np.asarray(output, np.float32)
  serial_p, tile_p = online_probability(serial), online_probability(tile)
  serial_bits, tile_bits = int(serial_p[744].view(np.uint32)), int(tile_p[744].view(np.uint32))
  # Host libm and GPU v_exp_f32 need not agree. A one-ULP perturbation reaches
  # a ties-to-even midpoint here, so GPU intermediates must identify the cause.
  assert abs((serial_bits & 65535)-32768) <= 2 and (tile_bits & 65535) < 32768
  midpoint = np.array([0x3e157fff, 0x3e158000, 0x3e158001], np.uint32).view(np.float32)
  np.testing.assert_array_equal(bf16_bits_cpu(midpoint), np.array([0x3e15, 0x3e16, 0x3e16], np.uint16))
  return dict(score_fma_witness=dict(fused_delta=float(fused), separate_delta=float(separate),
      probability_bf16_bits=[hex(int(x)) for x in bf16_bits_cpu(probability)]),
    reduction_witness=dict(dot2_grouped=float(two), dot4_grouped=float(four),
      inputs_exact_e4m3_and_bf16=True, hardware_tree_not_assumed=True),
    power2_commutation_checks=15,
    original_fixture=dict(seed=906080, case='prefill2027', batch=0, head=8, query=1753,
      dequantized_query_and_key_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
      key52_dot_serial=float(serial[52]), key52_dot_tile16=float(tile[52]),
      key744_cpu_probability_bits=[hex(serial_bits), hex(tile_bits)],
      key744_cpu_bf16_bits=[hex(int(bf16_bits_cpu(x)[744])) for x in (serial_p, tile_p)],
      gpu_exp2_not_emulated=True), strict_output_gate_unchanged=True, gpu_executed=False)


def rounding_compile_checks(args):
  """Inspect the actual pinned gfx1200 code without initializing any GPU."""
  from tinygrad import Device
  assert Device.DEFAULT.startswith('CPU'), 'Compile-only checks require DEV=CPU'
  reports = []
  if args.artifacts:
    args.artifacts.mkdir(parents=True, exist_ok=True)
  for rows in (2027, 2112):
    for variant in ('bf16', 'fp8'):
      for trace in (False, True):
        if trace:
          source, binary, _, layout, size = candidate.trace_compiled(16, 8, rows, rows, 128, True, True,
            variant, 8, 1753)
        elif variant == 'fp8':
          source, binary = candidate.compiled(16, 8, rows, rows, 128, True, True)
        else:
          source, binary = candidate.baseline_module().compiled(16, 8, rows, rows, True, waves=8, width=128,
            block_keys=32, stage_kv=True, key_padding=4, kv_load=1, value_layout=True, probability_mma=True)
        asm = helpers.disassemble(binary)
        scale_hex = hex(int(np.array(128**-.5, np.float32).view(np.uint32)))
        if variant == 'bf16':
          scale_fma = [line.strip() for line in asm.splitlines() if re.search(r'v_(?:dual_)?fma[c]?_f32', line) and scale_hex in line]
        else:
          scale_register = re.findall(r's_mul_f32\s+(s\d+),\s+s\d+,\s+'+scale_hex, asm)
          assert len(scale_register) == 1, scale_register
          scale_fma = [line.strip() for line in asm.splitlines() if re.search(r'v_(?:dual_)?fma[c]?_f32', line)
            and re.search(r'\b'+scale_register[0]+r'\b', line)]
        # Do not "fix" causal attention by importing noncausal contraction.
        assert not scale_fma, (rows, variant, trace, scale_fma)
        assert 'v_exp_f32' in asm and 'v_div_scale_f32' in asm and 'v_fmac_f32' in asm
        counts = Counter(re.findall(r'\b(v_[a-z0-9_]+)', asm))
        fma_count = sum(value for instruction, value in counts.items()
          if re.fullmatch(r'v_(?:dual_)?fma[c]?_f32(?:_e32|_e64)?', instruction))
        # Count both VOPD slots as operations, including v_dual_fmac_f32.
        # 64 divisions use five refinement FMAs each. The online loop uses
        # 24 exp2 range-handling FMAs and eight denominator FMAs.
        assert fma_count == 64*5+24+8 and counts['v_exp_f32_e32'] == 24
        if not trace:
          assert counts['v_wmma_f32_16x16x16_fp8_fp8'] == (16 if variant == 'fp8' else 0)
          assert counts['v_wmma_f32_16x16x16_bf16'] == (18 if variant == 'fp8' else 34)
        metadata = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
        resources = {key: int(value) for key, value in re.findall(
          r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):\s*(\d+)', metadata)}
        row = dict(rows=rows, variant=variant, diagnostic_trace=trace,
          score_scale_minus_maximum_fma_count=len(scale_fma), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
          binary_sha256=hashlib.sha256(binary).hexdigest(), assembly_sha256=hashlib.sha256(asm.encode()).hexdigest(),
          total_fma_count=fma_count, exp2_instruction_count=counts['v_exp_f32_e32'], resources=resources)
        if trace:
          row.update(trace_bytes=size*4, trace_layout=layout)
        if args.artifacts:
          stem = args.artifacts/f"rounding-{rows}-{variant}{'-trace' if trace else ''}"
          stem.with_suffix('.cpp').write_text(source)
          stem.with_suffix('.asm').write_text(asm)
          stem.with_suffix('.metadata').write_text(metadata)
        reports.append(row)
        print(json.dumps(row), flush=True)
  return reports


def trace_graph_checks_cpu():
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.uop.ops import Ops
  assert Device.DEFAULT.startswith('CPU')
  reports = []
  # Independent lane ownership check for the original failure location.
  selected = [(wave, lane, i) for wave in range(8) for lane in range(32) for i in range(8)
    if (1753//128*8+wave)*16+(lane//16)*8+i == 1753]
  assert len(selected) == 16 and len({lane % 16 for _, lane, _ in selected}) == 16
  assert sum(lane % 16 == 0 for _, lane, _ in selected) == 1
  for variant in ('bf16', 'fp8'):
    q = Tensor.empty(1, 16, 2027, 128, dtype=dtypes.bfloat16)
    k, v = [Tensor.empty(1, 8, 2027, 128, dtype=dtypes.bfloat16) for _ in range(2)]
    qs, ks = Tensor.empty(1, 16), Tensor.empty(1, 8)
    out, trace = candidate.attention_trace(q, k, v, qs, ks,
      causal=True, variant=variant, target_head=8, target_row=1753)
    assert out.dtype == dtypes.bfloat16 and trace.dtype == dtypes.float32
    assert trace.shape == (28864,)
    linear, _ = out.linear_with_vars(trace)
    sources = [next(x.arg for x in call.src[0].src if x.op == Ops.SOURCE) for call in linear.src
      if call.op == Ops.CALL and call.src[0].op == Ops.PROGRAM]
    assert sum('void agent_attention_'+variant+'_trace(' in text for text in sources) == 1
    assert sum('void agent_attention_fp8_quantize(' in text for text in sources) == int(variant == 'fp8')
    assert all('void agent_attention_fp8_qk(' not in text for text in sources)
    reports.append(dict(variant=variant, trace_bytes=trace.numel()*4, hip_executed=False,
      conversion_included=variant == 'fp8', selected_query_stores_owned_once=True))
  return reports


def fp8_cpu(x, scale):
  normalized = (x/scale).clamp(-448, 448)
  return normalized.to(torch.float8_e4m3fn).float()*scale


def metrics(actual, expected):
  error = np.asarray(actual, dtype=np.float64)-np.asarray(expected, dtype=np.float64)
  rmse = float(np.sqrt(np.mean(error**2)))
  rms = float(np.sqrt(np.mean(np.asarray(expected, dtype=np.float64)**2)))
  return dict(rmse=rmse, relative_rmse=rmse/max(rms, 1e-30), max_abs=float(np.abs(error).max()),
    max_head_rmse=float(np.sqrt(np.mean(error**2, axis=(-2, -1))).max()))


def reference(q, k, v, causal):
  repeat = q.shape[1]//k.shape[1]
  return torch.nn.functional.scaled_dot_product_attention(q.double(), k.double().repeat_interleave(repeat, 1),
    v.double().repeat_interleave(repeat, 1), is_causal=causal).numpy()


def old_or_hybrid_oracle(q, k, v, causal, mode, qs, ks):
  """Independent CPU online recurrence with each path's documented rounding."""
  if mode == 'old-fp8':
    q = fp8_cpu(q, (q.abs().amax(-1, keepdim=True)/448).clamp(min=1e-12))
    k = fp8_cpu(k, (k.abs().amax(-1, keepdim=True)/448).clamp(min=1e-12))
    v = fp8_cpu(v, (v.abs().amax((-2, -1), keepdim=True)/448).clamp(min=1e-12))
  elif mode == 'hybrid-fp8-qk':
    q, k = fp8_cpu(q, qs[:, :, None, None]), fp8_cpu(k, ks[:, :, None, None])
  else:
    assert mode == 'bf16'
  repeat = q.shape[1]//k.shape[1]
  k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
  rows, keys = q.shape[2], k.shape[2]
  maximum = torch.full(q.shape[:-1]+(1,), -torch.inf)
  denominator = torch.zeros_like(maximum)
  accumulator = torch.zeros_like(q)
  for start in range(0, keys, 32):
    scores = q@k[:, :, start:start+32].transpose(-1, -2)*q.shape[-1]**-.5
    if causal:
      scores.masked_fill_(torch.arange(start, min(start+32, keys))[None, :] > torch.arange(rows)[:, None], -torch.inf)
    next_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
    probabilities = (scores-next_max).exp()
    rescale = (maximum-next_max).exp()
    denominator = denominator*rescale+probabilities.sum(-1, keepdim=True)
    rounded = ((probabilities*448).to(torch.float8_e4m3fn).float()/448 if mode == 'old-fp8'
               else probabilities.to(torch.bfloat16).float())
    accumulator = accumulator*rescale+rounded@v[:, :, start:start+32]
    maximum = next_max
  return (accumulator/denominator).to(torch.bfloat16).float().numpy()


CONTROLS = [
  ('tail-causal', 2, 4, 2, 17, 33, 72, True, 'normal'),
  ('tail-noncausal', 2, 4, 2, 17, 33, 72, False, 'normal'),
  ('gqa-head-ranges', 1, 8, 2, 65, 97, 128, True, 'head-ranges'),
  ('outliers', 1, 4, 2, 129, 129, 128, True, 'outliers'),
  ('zeros', 1, 2, 1, 33, 33, 72, False, 'zeros'),
  ('small-logits', 1, 4, 2, 65, 97, 128, True, 'small'),
]
TARGETS = [
  ('vision720', 2, 16, 16, 720, 720, 72, False, 'normal'),
  ('prefill2027', 1, 16, 8, 2027, 2027, 128, True, 'normal'),
  ('prefill2112', 1, 16, 8, 2112, 2112, 128, True, 'normal'),
]


def inputs(case, rng):
  _, batch, heads, kv_heads, rows, keys, width, _, kind = case
  q_amp = np.ones((batch, heads), dtype=np.float32)
  k_amp = np.ones((batch, kv_heads), dtype=np.float32)
  if kind == 'head-ranges':
    q_amp *= np.exp2(np.arange(heads, dtype=np.float32)-heads//2)
    k_amp *= np.exp2(np.arange(kv_heads, dtype=np.float32)-kv_heads//2)
  if kind == 'small':
    q_amp *= .125
    k_amp *= .125
  def data(h, length, amp):
    return (rng.standard_normal((batch, h, length, width)).astype(np.float32)*amp[:, :, None, None])
  q, k, v = data(heads, rows, q_amp), data(kv_heads, keys, k_amp), data(kv_heads, keys, np.ones_like(k_amp))
  if kind == 'outliers':
    # Deliberately exceed the fixed range. Report clipping and quality loss.
    q[:, :, ::17, ::11] *= 32
    k[:, :, ::13, ::7] *= 32
  if kind == 'zeros':
    q.fill(0)
    k.fill(0)
  # Predetermined fixture ranges, not an amax of future test tokens.
  # Range is 14 standard deviations; it stays fixed across graph mutations.
  qs, ks = q_amp/32, k_amp/32
  q, k, v = [torch.from_numpy(x).to(torch.bfloat16).float() for x in (q, k, v)]
  return q, k, v, torch.from_numpy(qs), torch.from_numpy(ks)


def clipping(q, k, qs, ks):
  return {name: dict(count=int((x.abs() > s[:, :, None, None]*448).sum()), elements=x.numel())
    for name, x, s in (('q', q, qs), ('k', k, ks))}


def conversion_cpu():
  bits = np.arange(65536, dtype=np.uint16)
  bits = bits[(bits & 0x7f80) != 0x7f80]
  values = (bits.astype(np.uint32) << 16).view(np.float32)
  positive = np.arange(127, dtype=np.uint8)
  exponent, fraction = positive.astype(np.int32) >> 3, positive.astype(np.int32) & 7
  table = np.where(exponent == 0, fraction*2.**-9, (1+fraction/8)*np.exp2(exponent-7)).astype(np.float32)
  reports = []
  for scale in (.03125, 1., 32.):
    with np.errstate(over='ignore'):
      normalized = np.clip(values/np.float32(scale), -448, 448)
    absolute = np.abs(normalized)
    high = np.minimum(np.searchsorted(table, absolute), 126)
    low = np.maximum(high-1, 0)
    lower_error, upper_error = absolute-table[low], table[high]-absolute
    choose_high = (upper_error < lower_error) | ((upper_error == lower_error) & ((high & 1) == 0))
    expected = np.where(choose_high, high, low).astype(np.uint8) | (np.signbit(normalized).astype(np.uint8) << 7)
    actual = torch.from_numpy(normalized).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    np.testing.assert_array_equal(actual, expected)
    # Four adjacent byte pairs form the two vector words used by quant_eight.
    words = sum(actual.reshape(-1, 8)[:, i::4].astype(np.uint32) << (8*i) for i in range(4))
    reconstructed = np.stack([(words >> (8*i)) & 255 for i in range(4)], axis=-1).reshape(-1).astype(np.uint8)
    np.testing.assert_array_equal(reconstructed, expected)
    reports.append(dict(scale=scale, finite_bf16_values_exact=len(values)))
  return reports


def pack_mapping_cpu():
  reports = []
  for case in CONTROLS+TARGETS:
    name, batch, heads, kv_heads, rows, keys, width, _, _ = case
    totals = []
    for head_count, length in ((batch*heads, rows), (batch*kv_heads, keys)):
      stride = length*width
      blocks = (stride+2047)//2048
      group = np.arange(head_count*blocks, dtype=np.int64)[:, None]
      index = ((group % blocks)*256+np.arange(256))*8
      offsets = (group//blocks)*stride+index
      offsets = offsets[index < stride]
      np.testing.assert_array_equal(offsets, np.arange(0, head_count*stride, 8))
      totals.append(len(offsets)*8)
    reports.append(dict(case=name, covered_qk_elements=totals, one_store_per_element=True))
  return reports


def graph_checks_cpu():
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.uop.ops import Ops
  assert Device.DEFAULT.startswith('CPU')
  reports = []
  for rows, width, heads, kh in ((17, 72, 4, 2), (2027, 128, 16, 8)):
    q = Tensor.empty(1, heads, rows, width, dtype=dtypes.bfloat16)
    k, v = [Tensor.empty(1, kh, rows, width, dtype=dtypes.bfloat16) for _ in range(2)]
    qs, ks = Tensor.empty(1, heads), Tensor.empty(1, kh)
    output = candidate.attention(q, k, v, qs, ks, True)
    assert output.dtype == dtypes.bfloat16
    linear, _ = output.linear_with_vars()
    names = []
    for call in linear.src:
      if call.op == Ops.CALL and call.src[0].op == Ops.PROGRAM:
        source = next(x.arg for x in call.src[0].src if x.op == Ops.SOURCE)
        names.append(source[source.index('void agent_attention'):].split('(')[0])
    assert names.count('void agent_attention_fp8_quantize') == names.count('void agent_attention_fp8_qk') == 1
    reports.append(dict(rows=rows, width=width, custom_calls=names, total_calls=len(linear.src), executed=False))

  class Model:
    def attention(self, q, k, v, mask=None, causal=False):
      return q+1
    def text_layer(self, x, cos, sin, prefix):
      return self.attention(x, cos, sin, causal=True)
  model = Model()
  captured = candidate.install_capture(model)
  prefix = 'vlm.model.language_model.layers.0'
  jit = TinyJit(lambda q, k, v: model.text_layer(q, k, v, prefix).realize())
  for factor in (1., 2., 3., 7., 4.):
    tensors = [Tensor(np.broadcast_to((np.arange(heads, dtype=np.float32)+1)[None, :, None, None]*factor,
      (1, heads, 128, 128)).copy()).cast(dtypes.bfloat16).realize() for heads in (16, 8, 8)]
    jit(*tensors)
  for label, heads in (('q', 16), ('k', 8)):
    np.testing.assert_array_equal(captured[prefix][label].numpy(), (np.arange(heads, dtype=np.float32)+1)*7)
  calibrated = candidate.captured_scales(captured)
  for label, heads in (('q', 16), ('k', 8)):
    candidate.validate_scales(calibrated['scales'][prefix][label], heads)
  return dict(graphs=reports, cpu_calibration_replays=5, persistent_head_maxima_exact=True)


def cost_budget():
  results = []
  for case in TARGETS:
    name, batch, heads, kh, rows, keys, width, causal, _ = case
    groups = (rows+127)//128
    ends = [min(keys, (group+1)*128) if causal else keys for group in range(groups)]
    q_elements, k_elements = batch*heads*rows*width, batch*kh*keys*width
    repeated_k = batch*heads*sum(ends)*width
    qk_flops = 2*batch*heads*128*sum((end+31)//32*32 for end in ends)*((width+15)//16*16)
    quant_traffic = 3*(q_elements+k_elements)
    saved_qk_reads = q_elements+repeated_k
    results.append(dict(case=name, combined_quant_read_write_bytes=quant_traffic,
      fp8_temporary_bytes=q_elements+k_elements, avoided_repeated_conversion_elements=repeated_k-k_elements,
      qk_matrix_gflop=qk_flops/1e9, qk_peak_ms_bf16=qk_flops/86e12*1000,
      qk_peak_ms_fp8=qk_flops/172e12*1000, quant_streaming_ms_at_288GBs=quant_traffic/288e9*1000,
      net_streaming_bytes_saved=saved_qk_reads-quant_traffic,
      net_streaming_ms_saved_at_288GBs=(saved_qk_reads-quant_traffic)/288e9*1000))
  return dict(shapes=results, caveat='Arithmetic peaks and a streaming traffic model, not measured latency. '
    'Caches, overlap, occupancy, conversion instructions, and launch costs prevent adding these terms as a speedup.')


def compile_checks(args):
  from tinygrad import Device
  assert Device.DEFAULT.startswith('CPU')
  old = load('fp8-attention')
  result = []
  for case in TARGETS:
    name, _, heads, kh, rows, keys, width, causal, _ = case
    for variant in ('selected-bf16', 'old-fp8', 'hybrid-fp8-qk'):
      if variant == 'selected-bf16':
        source, binary = candidate.baseline_module().compiled(heads, kh, rows, keys, causal, waves=8, width=width,
          block_keys=32, stage_kv=True, key_padding=4, kv_load=1, value_layout=rows >= 1024, probability_mma=True)
      elif variant == 'old-fp8':
        source, binary = old.compiled(heads, kh, rows, keys, width, causal, 8)
      else:
        source, binary = candidate.compiled(heads, kh, rows, keys, width, causal, rows >= 1024)
      asm = helpers.disassemble(binary)
      metadata = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
      resources = {key: int(value) for key, value in re.findall(
        r'\.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):\s*(\d+)', metadata)}
      counts = Counter(re.findall(r'^\s*((?:[sv]_|ds_|global_|buffer_)[a-z0-9_]+)', asm, re.M))
      row = dict(shape=name, variant=variant, resources=resources, instructions=dict(counts))
      if variant == 'hybrid-fp8-qk':
        assert counts['v_wmma_f32_16x16x16_fp8_fp8'] == 2*((width+15)//16)
        assert counts['v_wmma_f32_16x16x16_bf16'] == 2*((width+15)//16)+2
        assert counts['global_load_u8'] == counts['ds_load_u8'] == 0
        assert counts['v_cvt_pk_fp8_f32'] == 0
        assert resources['private_segment_fixed_size'] == resources['vgpr_spill_count'] == 0
      if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        stem = args.artifacts/f'agent-attention-fp8-{name}-{variant}'
        stem.with_suffix('.cpp').write_text(source)
        stem.with_suffix('.asm').write_text(asm)
        stem.with_suffix('.metadata').write_text(metadata)
      result.append(row)
      print(json.dumps({k: v for k, v in row.items() if k != 'instructions'}), flush=True)
  for case in CONTROLS+TARGETS:
    name, batch, heads, kh, rows, keys, width, causal, _ = case
    candidate.compiled(heads, kh, rows, keys, width, causal, rows >= 1024)
    source, binary, groups = candidate.quant_compiled((batch, heads, rows, width), (batch, kh, keys, width))
    asm = helpers.disassemble(binary)
    assert 'v_cvt_pk_fp8_f32' in asm and 'global_store_b64' in asm and 'global_load_b128' in asm
    if args.artifacts:
      stem = args.artifacts/f'agent-attention-fp8-{name}-quantize'
      stem.with_suffix('.cpp').write_text(source)
      stem.with_suffix('.asm').write_text(asm)
  candidate.probe_compiled(8160, 8)
  return result


def cpu_study(args):
  rows = []
  rng = np.random.default_rng(906080)
  for case in CONTROLS+TARGETS:
    q, k, v, qs, ks = inputs(case, rng)
    truth = reference(q, k, v, case[7])
    row = dict(case=case[0], shape=list(q.shape), clipping=clipping(q, k, qs, ks), errors={})
    for mode in ('bf16', 'old-fp8', 'hybrid-fp8-qk'):
      actual = old_or_hybrid_oracle(q, k, v, case[7], mode, qs, ks)
      row['errors'][mode] = metrics(actual, truth)
    rows.append(row)
    print(json.dumps(row), flush=True)
  return rows


def gpu_conversion():
  from tinygrad import Tensor, dtypes
  bits = np.arange(65536, dtype=np.uint16)
  bits = bits[(bits & 0x7f80) != 0x7f80]
  x = Tensor(bits.reshape(8160, 8)).bitcast(dtypes.bfloat16).realize()
  values = torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32))
  results = []
  for scale in (.03125, 1., 32.):
    expected = (values/scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    actual = candidate.conversion_probe(x, Tensor([scale], dtype=dtypes.float32).realize()).numpy().reshape(-1)
    np.testing.assert_array_equal(actual, expected)
    results.append(dict(scale=scale, finite_bf16_values_exact=len(bits)))
  return results


def gpu_rounding_diagnostic(args, case, q, k, v, qs, ks, actual_bits, expected_bits):
  """Parent-only, one query, no timing. Original strict failure still propagates."""
  from tinygrad import Tensor, dtypes
  mismatch = np.argwhere(actual_bits != expected_bits)
  batch, head, query, _ = map(int, mismatch[0])
  assert batch == 0, 'The bounded trace currently selects batch zero only'
  _, _, heads, kv_heads, rows, keys, width, causal, _ = case
  kv_head = head//(heads//kv_heads)
  qd, kd = fp8_cpu(q, qs[:, :, None, None]), fp8_cpu(k, ks[:, :, None, None])
  def tensor(x):
    return Tensor(x.numpy()).cast(dtypes.bfloat16).realize()
  scales = [Tensor(x.numpy(), dtype=dtypes.float32).realize() for x in (qs, ks)]
  layout, size = candidate.trace_layout(keys, width)
  used_tiles = (min(keys, (query//128+1)*128)+31)//32 if causal else (keys+31)//32
  traces, output_checks = {}, {}
  for variant, a, b, untapped in [('bf16', qd, kd, expected_bits), ('fp8', q, k, actual_bits)]:
    out, trace = candidate.attention_trace(tensor(a), tensor(b), tensor(v), *scales,
      causal=causal, variant=variant, target_head=head, target_row=query)
    out.realize(trace)
    bits, packed = out.bitcast(dtypes.uint16).numpy(), trace.numpy()
    assert packed.shape == (size,)
    output_checks[variant] = dict(matches_untapped=bool(np.array_equal(bits, untapped)),
      mismatched_bf16=int(np.count_nonzero(bits != untapped)))
    traces[variant] = {name: packed[item['offset']:item['offset']+item['size']].reshape(item['shape'])[:used_tiles].copy()
      for name, item in layout.items()}
    for name, values in traces[variant].items():
      assert not np.isnan(values).any(), (variant, name, 'unwritten trace slot')
      if name != 'scaled':
        assert np.isfinite(values).all(), (variant, name)
  scale_product = np.float32(qs[batch, head].item()*ks[batch, kv_head].item())
  # Preserve native raw values separately. Compare in the BF16 input units.
  normalized_fp8_partial = np.float32(traces['fp8']['partial']*scale_product)
  comparisons = {}
  for name in layout:
    a = normalized_fp8_partial if name == 'partial' else traces['fp8'][name]
    b = traces['bf16'][name]
    different = a.view(np.uint32) != b.view(np.uint32)
    first = np.argwhere(different)
    comparisons[name] = dict(mismatched_fp32=int(different.sum()), elements=a.size,
      first=[dict(index=index.tolist(), bf16=float(b[tuple(index)]), fp8=float(a[tuple(index)]),
        bf16_bits=hex(int(b[tuple(index)].view(np.uint32))), fp8_bits=hex(int(a[tuple(index)].view(np.uint32))))
        for index in first[:8]])
  probability_bf16 = {variant: bf16_bits_cpu(value['probability']) for variant, value in traces.items()}
  comparisons['probability_bf16'] = dict(mismatched=int(np.count_nonzero(probability_bf16['bf16'] != probability_bf16['fp8'])))
  # Independent, correctly rounded FP64 dot products for the quantized inputs.
  qrow, khead = qd[batch, head, query].numpy(), kd[batch, kv_head].numpy()
  depth = (width+15)//16
  exact_partial = np.zeros((used_tiles, depth, 32), np.float64)
  for d in range(depth):
    dots = (qrow[:min((d+1)*16, width)].astype(np.float64)[None, :]
      * khead[:, :min((d+1)*16, width)].astype(np.float64)).sum(-1)
    padded = np.pad(dots, (0, max(0, used_tiles*32-len(dots))))[:used_tiles*32]
    exact_partial[:, d] = padded.reshape(used_tiles, 32)
  independent = {}
  for variant, observed in [('bf16', traces['bf16']['partial']), ('fp8', normalized_fp8_partial)]:
    error = observed.astype(np.float64)-exact_partial
    independent[variant] = dict(partial_fp32_differs_from_correctly_rounded_fp64=int(np.count_nonzero(
      observed.view(np.uint32) != exact_partial.astype(np.float32).view(np.uint32))),
      partial_max_abs_error=float(np.abs(error).max()), partial_rmse=float(np.sqrt(np.mean(error**2))))
  dots = (qrow.astype(np.float64)[None, :]*khead.astype(np.float64)).sum(-1)*width**-.5
  if causal:
    dots[query+1:] = -np.inf
  probability = np.exp(dots-dots.max())
  truth = (probability/probability.sum())@v[batch, kv_head].double().numpy()
  for variant, bits in [('bf16', expected_bits), ('fp8', actual_bits)]:
    row = (bits[batch, head, query].astype(np.uint32) << 16).view(np.float32)
    independent[variant]['final_row_vs_quantized_input_fp64'] = dict(
      rmse=float(np.sqrt(np.mean((row-truth)**2))), max_abs=float(np.abs(row-truth).max()))
  classification = ('tap_changed_output' if not all(x['matches_untapped'] for x in output_checks.values()) else
    'qk_accumulation' if comparisons['partial']['mismatched_fp32'] else
    'score_scale_or_mask' if comparisons['scaled']['mismatched_fp32'] else
    'softmax_probability_or_state' if comparisons['probability']['mismatched_fp32'] or comparisons['state']['mismatched_fp32'] else
    'pv_accumulation_or_final_division')
  diagnostic = dict(case=case[0], batch=batch, head=head, kv_head=kv_head, query=query,
    used_key_tiles=used_tiles, allocated_trace_bytes_per_variant=size*4, native_fp8_qk_scale_product=float(scale_product),
    tapped_outputs=output_checks, earliest_difference=classification, intermediate_comparisons=comparisons,
    independent_quantized_input_reference=independent,
    original_input_quantization_quality_not_tested_here=True, strict_failure_retained=True,
    diagnostic_not_a_timing=True)
  if args.artifacts:
    args.artifacts.mkdir(parents=True, exist_ok=True)
    path = args.artifacts/f'rounding-{case[0]}-head{head}-row{query}.npz'
    np.savez(path, **{f'{variant}_{name}': value for variant, trace in traces.items() for name, value in trace.items()},
      dequantized_query=qrow, dequantized_key=khead, value=v[batch, kv_head].numpy(),
      fp8_output_bits=actual_bits[batch, head, query], bf16_output_bits=expected_bits[batch, head, query],
      exact_partial_fp64=exact_partial)
    diagnostic.update(artifact=str(path), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  return diagnostic


def gpu_checks(args):
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled
  Compiled.wait_timeout_ms = 10000
  helpers.module('prefill-kernels').fix_dispatch_packet()
  report = dict(numerical_candidate=True, quality_validated=False, conversion=gpu_conversion(), correctness=[], timings=[],
    provenance=provenance(), microcheck=args.microcheck, timed=not args.microcheck, gpu_executed=True)
  rng = np.random.default_rng(906080)
  def tensor(x):
    return Tensor(x.numpy()).cast(dtypes.bfloat16).realize()
  for case in CONTROLS+TARGETS:
    name, _, _, _, rows, _, _, causal, _ = case
    q, k, v, qs, ks = inputs(case, rng)
    # Keep all preceding RNG draws, including skipped cases. This reproduces
    # the original prefill2027 failure rather than creating a new random test.
    if args.microcheck and case not in CONTROLS and name != 'prefill2027':
      continue
    tq, tk, tv = map(tensor, (q, k, v))
    sq, sk = [Tensor(x.numpy(), dtype=dtypes.float32).realize() for x in (qs, ks)]
    baseline = candidate.baseline_module().attention
    jit = TinyJit(lambda a, b, c: candidate.attention(a, b, c, sq, sk, causal).realize())
    for _ in range(3):
      jit(tq, tk, tv)
    for step in range(2 if case in CONTROLS else 1):
      before = jit(tq, tk, tv).float().numpy() if step else None
      if step:
        split = min(q.shape[2], k.shape[2])//2
        # Static scales must not introduce a future-token amax dependency.
        k[:, :, split:] *= -7
        v[:, :, split:] += 19
        k, v = k.to(torch.bfloat16).float(), v.to(torch.bfloat16).float()
        tk.assign(tensor(k)).realize()
        tv.assign(tensor(v)).realize()
      qd, kd = fp8_cpu(q, qs[:, :, None, None]), fp8_cpu(k, ks[:, :, None, None])
      assert torch.equal(qd.to(torch.bfloat16).float(), qd) and torch.equal(kd.to(torch.bfloat16).float(), kd)
      packed = candidate.quantized_inputs(tq, tk, sq, sk)
      for got, raw, scale in zip(packed, (q, k), (qs, ks)):
        expected = (raw/scale[:, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        np.testing.assert_array_equal(got.numpy(), expected)
      quantized_reference = baseline(tensor(qd), tensor(kd), tv, causal).realize()
      full_baseline = None if args.microcheck else baseline(tq, tk, tv, causal).realize()
      actual_tensor = jit(tq, tk, tv)
      actual = actual_tensor.float().numpy()
      bits = actual_tensor.bitcast(dtypes.uint16).numpy()
      expected_bits = quantized_reference.bitcast(dtypes.uint16).numpy()
      row = dict(case=name, mutation=step, shape=list(q.shape), clipping=clipping(q, k, qs, ks),
        implementation_matches_quantized_input_reference=bool(np.array_equal(bits, expected_bits)),
        implementation_mismatched_bf16=int(np.count_nonzero(bits != expected_bits)),
        input_sha256={label: hashlib.sha256(x.numpy().tobytes()).hexdigest() for label, x in
          [('q', q), ('k', k), ('v', v), ('q_scales', qs), ('k_scales', ks)]})
      if not args.microcheck:
        truth = reference(q, k, v, causal)
        row.update(candidate_error=metrics(actual, truth), selected_bf16_error=metrics(full_baseline.float().numpy(), truth))
      else:
        row['original_input_quality_not_retested'] = True
      different = np.argwhere(bits != expected_bits)
      if len(different):
        row['first_mismatches'] = [dict(index=index.tolist(), actual=int(bits[tuple(index)]),
          expected=int(expected_bits[tuple(index)])) for index in different[:16]]
      report['correctness'].append(row)
      # Retain the original failure even if the optional diagnostic fails.
      args.output.write_text(json.dumps(report, indent=2)+'\n')
      if len(different):
        if args.trace_on_mismatch:
          row['rounding_diagnostic'] = gpu_rounding_diagnostic(args, case, q, k, v, qs, ks, bits, expected_bits)
      args.output.write_text(json.dumps(report, indent=2)+'\n')
      print(json.dumps(row), flush=True)
      assert np.isfinite(actual).all()
      # This verifies the FP8 implementation only. Original-input error above
      # remains nonzero and needs route validation; no accuracy bound is waived.
      np.testing.assert_array_equal(bits, expected_bits, err_msg=str(row))
      if step and causal:
        np.testing.assert_array_equal(actual[:, :, :split], before[:, :, :split])
    if case not in TARGETS or args.microcheck:
      continue
    runners = []
    for label, fn in [('selected-bf16', lambda a, b, c: baseline(a, b, c, causal)),
        ('hybrid-fp8-qk', lambda a, b, c: candidate.attention(a, b, c, sq, sk, causal))]:
      def chain(a, b, c, fn=fn):
        for _ in range(4):
          a = fn(a, b, c).realize()
        return a
      runner = TinyJit(chain)
      for _ in range(3):
        runner(tq, tk, tv)
      runners.append((label, runner))
    samples = {label: [] for label, _ in runners}
    # Five alternating blocks * two graph replays * four calls = 40 calls.
    for block in range(5):
      for label, runner in (runners if block % 2 == 0 else runners[::-1]):
        Device[Device.DEFAULT].synchronize()
        started = time.perf_counter()
        for _ in range(2):
          runner(tq, tk, tv)
        Device[Device.DEFAULT].synchronize()
        samples[label].append((time.perf_counter()-started)*1000/8)
    timing = dict(case=name, timed_calls_per_implementation=40, includes_combined_qk_quantization=True,
      includes_value_packing=rows >= 1024, ms={label: dict(median=float(np.median(x)), blocks=x) for label, x in samples.items()})
    report['timings'].append(timing)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(timing), flush=True)
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--cpu-only', action='store_true')
  parser.add_argument('--artifacts', type=Path)
  parser.add_argument('--rounding-only', action='store_true', help='Bounded independent CPU arithmetic and gfx1200 compilation checks')
  parser.add_argument('--microcheck', action='store_true', help='Parent GPU controls and original prefill2027 fixture; no timing or quality study')
  parser.add_argument('--trace-on-mismatch', action='store_true', help='Parent GPU diagnostic taps before the unchanged strict gate fails')
  args = parser.parse_args()
  if args.rounding_only and not args.cpu_only:
    parser.error('--rounding-only requires --cpu-only')
  if args.cpu_only and (args.microcheck or args.trace_on_mismatch):
    parser.error('--microcheck and --trace-on-mismatch require parent GPU mode')
  torch.set_num_threads(4)
  helpers.configure_baseline()
  if args.rounding_only:
    report = dict(provenance=provenance(), rounding_regression=rounding_regression_cpu(),
      compilation=rounding_compile_checks(args), trace_graphs=trace_graph_checks_cpu(),
      gpu_executed=False, strict_gate_unchanged=True,
      implementation_fix_validated=False, quality_validated=False)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return
  report = dict(cpu_conversion=conversion_cpu(), cpu_pack_mapping=pack_mapping_cpu(), cost_budget=cost_budget(),
    numerical_candidate=True, quality_validated=False)
  if args.cpu_only:
    report.update(compilation=compile_checks(args), cpu_graphs=graph_checks_cpu(), cpu_error_study=cpu_study(args), gpu_executed=False)
  else:
    report.update(gpu_checks(args), gpu_executed=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  main()
