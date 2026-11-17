"""CPU-only source, actual-seed forward, independent gradient, mapping, and guard checks. No fitting."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', TC='0', ALLOW_TF32='0', TRAINING='0', BEAM='0', JITBEAM='0', PARALLEL='2',
                    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
  spec = importlib.util.spec_from_file_location('agent_distill_vision_check_common', HERE/'agent-distill-vision-common.py')
  C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
  C.storage(args.output); C.ROOT.mkdir(exist_ok=True)
  checked_sources = C.pins()
  os.environ['CACHEDB'] = str(C.ROOT/'cpu-cache.db')
  (C.ROOT/'cpu-tmp').mkdir(exist_ok=True); os.environ['TMPDIR'] = str(C.ROOT/'cpu-tmp')
  def hardware_guard(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)):
      path = os.fsdecode(values[0])
      if path.startswith(('/dev/dri/', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb/')): raise RuntimeError('CPU check tried hardware '+path)
  sys.addaudithook(hardware_guard)
  import numpy as np
  B = C.graph()
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.helpers import Context
  arrays, entries = C.seeds(); assert len(entries) == 16
  training = C.capture(C.TRAIN, C.TRAIN_SHA, 'train'); validation = C.capture(C.VALIDATION, C.VALIDATION_SHA, 'validation')
  for rows in (training, validation):
    assert [r['image_index'] for r in rows] == list(range(10))
    for r in rows:
      assert r['image_index'] == r['batch']*2+r['member']
      assert r['x'].shape == r['y'].shape == (1, 720, 1152)
  # Three patches per image retain real width, all heads, all parameters, and cross-image isolation.
  x = np.concatenate([training[i]['x'][:, :3] for i in (0, 1)], axis=0).copy()
  target = np.concatenate([training[i]['y'][:, :3] for i in (0, 1)], axis=0).copy()
  block = B.Block(arrays)
  # The full graph remains lazy on CPU. Its materialization belongs to the parent qualification.
  with Context(TRAINING=1):
    full_input = Tensor.empty(1, 720, 1152, device='CPU').is_param_(False)
    full_target = Tensor.empty(1, 720, 1152, device='CPU').is_param_(False)
    full_output = block(full_input); assert full_output.shape == (1, 720, 1152)
    full_grads = B.mse(full_output, full_target).gradient(*block.weights.values())
    assert [g.shape for g in full_grads] == list(B.shapes().values())
    del full_input, full_target, full_output, full_grads
  # Import the original implementation without invoking its complete model loader.
  sys.path.insert(0, str(C.NATIVE.parent))
  native_module = C.module('agent_distill_vision_original_model', C.NATIVE)
  native = native_module.EdgeModel.__new__(native_module.EdgeModel)
  native.weights = {B.PREFIX+k: v for k, v in block.weights.items()}
  native.weight_scales = {}; native.dtype = dtypes.float32; native.cuda_kernels = native.fp8_wmma = False
  with Context(TRAINING=1):
    tx = Tensor(x.copy(), device='CPU').is_param_(False)
    ty = Tensor(target.copy(), device='CPU').is_param_(False)
    actual = block(tx)
    loss = B.mse(actual, ty); gradients = loss.gradient(*block.weights.values()); Tensor.realize(loss, actual, *gradients)
    grads = {name: value.numpy() for name, value in zip(block.weights, gradients, strict=True)}
    reference = native.vision_layer(tx, 12)
    a, b = actual.numpy(), reference.numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=2e-6)
    assert tx.grad is None and ty.grad is None and not tx.is_param and not ty.is_param
  # An independent Torch CPU equation checks every FP32 gradient, including both LayerNorms and all biases.
  import torch
  import torch.nn.functional as F
  torch.set_num_threads(4); torch.set_default_device('cpu')
  weights = {name: torch.tensor(value.copy(), dtype=torch.float32, requires_grad=True) for name, value in arrays.items()}
  def oracle(value):
    def linear(v, name): return F.linear(v, weights[name+'.weight'], weights[name+'.bias'])
    def norm(v, name): return F.layer_norm(v, (1152,), weights[name+'.weight'], weights[name+'.bias'], 1e-6)
    h = norm(value, 'layer_norm1')
    q, k, v = (linear(h, 'self_attn.'+n+'_proj').reshape(*h.shape[:2], 16, 72).transpose(1, 2) for n in ('q', 'k', 'v'))
    h = (q.matmul(k.transpose(-2, -1))*72**-.5).softmax(-1).matmul(v).transpose(1, 2).reshape(value.shape)
    value = value + linear(h, 'self_attn.out_proj')
    return value + linear(F.gelu(linear(norm(value, 'layer_norm2'), 'mlp.fc1'), approximate='tanh'), 'mlp.fc2')
  tout = oracle(torch.tensor(x)); tloss = ((tout-torch.tensor(target))**2).mean(); tloss.backward()
  torch_output = tout.detach().numpy(); np.testing.assert_allclose(a, torch_output, rtol=2e-4, atol=5e-5)
  comparisons = {}
  for name, value in weights.items():
    expected = value.grad.numpy(); got = grads[name]
    assert got.shape == arrays[name].shape and np.isfinite(got).all()
    maximum = float(np.abs(got-expected).max())
    relative_l2 = float(np.linalg.norm((got-expected).ravel())/max(np.linalg.norm(expected.ravel()), 1e-7))
    np.testing.assert_allclose(got, expected, rtol=3e-3, atol=3e-5, err_msg=name)
    assert relative_l2 < 3e-3 or maximum < 2e-7, (name, relative_l2, maximum)
    comparisons[name] = dict(max_abs=maximum, relative_l2=relative_l2, gradient_l2=float(np.linalg.norm(got.ravel())))
  # Image2 cannot change image1: attention must never join the captured batch dimension.
  altered = x.copy(); altered[1] *= -2
  isolated = block(Tensor(altered, device='CPU')).numpy()
  np.testing.assert_array_equal(a[0], isolated[0])
  changed = float(np.max(np.abs(a[1]-isolated[1]))); assert changed > 1e-3
  rejected = []
  def reject(name, fn):
    try: fn()
    except (AssertionError, ValueError): rejected.append(name)
    else: raise AssertionError('Accepted negative control '+name)
  reject('target_broadcast', lambda: B.mse(Tensor.zeros(1, 3, 1152), Tensor.zeros(1, 1, 1152)))
  reject('wrong_width', lambda: block(Tensor.zeros(1, 3, 72)))
  reject('wrong_seed_shape', lambda: B.Block(dict(arrays, **{'mlp.fc1.bias': np.zeros(4303, np.float32)})))
  reject('wrong_capture_hash', lambda: C.capture(C.TRAIN, '0'*64, 'train'))
  reject('cross_split_capture', lambda: C.capture(C.TRAIN, C.TRAIN_SHA, 'validation'))
  # Read-only header mutations must reject before target numerical files are opened.
  original_read = C.read
  def changed_header(kind):
    header = original_read(C.TRAIN)
    if kind == 'unfinished': header['complete'] = False
    else: header['rows'][0]['route_id'] = '00000090--1cf3f7d926'
    C.read = lambda path: header
    try: C.capture(C.TRAIN, C.TRAIN_SHA, 'train')
    finally: C.read = original_read
  reject('unfinished_capture', lambda: changed_header('unfinished'))
  reject('sealed_route', lambda: changed_header('sealed'))
  L = C.module('agent_distill_vision_check_ledger', C.LEDGER)
  fake_availability = lambda: dict(selected_node=dict(gpu_id=27211, pci='0000:0c:00.0'), kfd_proc_root_read=True,
                                    kfd_processes=[], observed_selected_kfd_owners=[], vram_free_bytes=24*1024**3)
  ledger = L.Ledger(C.LIMITS, fake_availability)
  iface = SimpleNamespace(gpu_id=27211, dev_sysfs_path='/sys/bus/pci/devices/0000:0c:00.0')
  reject('unrelated_gpu', lambda: ledger.validate(SimpleNamespace(gpu_id=1, dev_sysfs_path=iface.dev_sysfs_path)))
  bounded = ledger.alloc(lambda _, size: SimpleNamespace(meta=SimpleNamespace(handle=1, size=size)))
  reject('oversize_allocation', lambda: bounded(iface, 256*1024**2+1))
  bounded(iface, 16); ledger.release(1); assert ledger.report['requested_bytes'] == 16 and ledger.report['live_bytes'] == 0
  ledger.report['requested_bytes'] = 256*1024**3
  reject('cumulative_allocation', lambda: bounded(iface, 16))
  # Recovery requires a clean journal. This creates only small synthetic CPU state files.
  S = C.module('agent_distill_vision_check_state', C.STATE)
  import tempfile
  with tempfile.TemporaryDirectory(prefix='cpu-state-', dir=C.ROOT) as directory:
    root = Path(directory); state = {k+'/p': np.zeros(2, np.float32) for k in ('weight', 'moment1', 'moment2')}
    state.update(beta1_power=np.ones(1, np.float32), beta2_power=np.ones(1, np.float32), learning_rate=np.array([3.5e-6], np.float32))
    S.save(root, state, dict(step=0), dict(test=True), {'p': (2,)})
    S.load(root, dict(test=True), {'p': (2,)})
    S.intent(root, 0)
    reject('interrupted_checkpoint', lambda: S.load(root, dict(test=True), {'p': (2,)}))
  old_argv = sys.argv[:]
  try:
    sys.argv = [str(HERE/'agent-distill-vision-train.py'), '--help']
    try: runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as exc: assert exc.code == 0
  finally: sys.argv = old_argv
  C.no_devices(); assert all(d.startswith(('CPU', 'NPY', 'PYTHON')) for d in Device._opened_devices), Device._opened_devices
  report = dict(complete=True, gpu_executed=False, optimizer_updates=0, fitting_executed=False,
                fixture_shape=list(x.shape), real_width=1152, real_mlp=4304, attention_heads=16, attention_head_dim=72,
                parameter_count=sum(v.size for v in arrays.values()), native_forward_max_abs=float(np.max(np.abs(a-b))),
                torch_forward_max_abs=float(np.max(np.abs(a-torch_output))), mse=float(loss.item()), torch_mse=float(tloss.detach()),
                gradients=comparisons, image_isolation_exact=True, other_image_changed_max_abs=changed,
                rejected=rejected, training_report_sha256=C.TRAIN_SHA, validation_report_sha256=C.VALIDATION_SHA,
                mapping_checked_images=20, independent_anchors=dict(train=1, validation=1),
                full720_lazy_forward_gradient_shapes=True, full720_gpu_graph_qualified=False, quality_qualified=False,
                opened_devices=sorted(Device._opened_devices), sources=checked_sources)
  assert C.pins() == checked_sources, 'Source changed during CPU checks'
  C.write(args.output, report)
  print(json.dumps({k: v for k, v in report.items() if k not in ('gradients', 'sources')}, indent=2))


if __name__ == '__main__': main()
