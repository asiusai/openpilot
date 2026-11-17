"""CPU preflight by default. Explicit parent execution qualifies one update, then fits 25 updates."""
import argparse
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_vision_common_train', HERE/'agent-distill-vision-common.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)


def prepare():
  C.no_devices(); C.storage(C.ROOT); C.ROOT.mkdir(exist_ok=True)
  train = C.capture(C.TRAIN, C.TRAIN_SHA, 'train'); validation = C.capture(C.VALIDATION, C.VALIDATION_SHA, 'validation')
  arrays, entries = C.seeds(); B = C.graph()
  p = dict(schema=1, tinygrad_revision=C.REV, limits=C.LIMITS, hyperparameters=C.HYPER,
           sources=C.pins(), seed=entries, parameter_count=sum(a.size for a in arrays.values()),
           teacher_layers=[12, 13, 14, 15], seed_layer=12, geometry=dict(patches=720, width=1152, mlp=4304, heads=16, head_dim=72, layernorm_eps=1e-6),
           train_report=dict(path=str(C.TRAIN), sha256=C.TRAIN_SHA), validation_report=dict(path=str(C.VALIDATION), sha256=C.VALIDATION_SHA),
           sample_mapping={split: [{k: v for k, v in r.items() if k not in ('x', 'y')} for r in rows]
                           for split, rows in [('train', train), ('validation', validation)]},
           independent_anchors=dict(train=1, validation=1), optimizer_updates=0, gpu_executed=False)
  assert len(B.shapes()) == 16
  path = C.ROOT/'plan.json'; C.write(path, p); C.no_devices()
  print(json.dumps(dict(plan=str(path), sha256=C.sha(path), parameter_count=p['parameter_count'], gpu_executed=False), indent=2))


def environment(root):
  env = os.environ.copy()
  for key in list(env):
    if key.startswith(('AM_', 'AMD_', 'HCQ_', 'ALPAMAYO_')) or key in ('CPU_CC', 'CPU_LLVM'): env.pop(key, None)
  env.update(DEV=C.TARGET, TC='0', ALLOW_TF32='0', TRAINING='0', PARALLEL='2', BEAM='0', JITBEAM='0',
             PROFILE='0', PMC='', SQTT='', VIZ='0', LRU='0', MAX_BUFFER_SIZE=str(C.LIMITS['max_single_buffer']),
             ALLOW_DEVICE_USAGE='1', OPTIM_DTYPE='float32', FUSE_OPTIM='0', CONST_LR='0', NO_MEMORY_PLANNER='1',
             PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(C.TINY), CACHEDB=str(root/'compiler-cache.db'), TMPDIR=str(root/'tmp'),
             OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  return env


def availability(args):
  assert args.availability_snapshot and args.reviewed_availability_sha
  assert args.availability_snapshot.resolve().is_relative_to(C.ROOT)
  assert C.sha(args.availability_snapshot) == args.reviewed_availability_sha
  a = C.read(args.availability_snapshot)
  assert 0 <= time.time_ns()-a['captured_unix_ns'] <= 60_000_000_000, 'Need fresh parent-reviewed availability'
  K = C.module('agent_distill_vision_kfd', C.KFD); K.require_no_observed_conflict(a)
  assert a['vram_free_bytes'] >= C.LIMITS['live_allocation_limit']+C.LIMITS['free_reserve']
  return K


def qualification(args):
  assert args.qualification and args.qualification_sha
  path = args.qualification.resolve(); assert path.is_relative_to(C.ROOT) and C.sha(path) == args.qualification_sha
  q = C.read(path); wall = C.read(path.parent/'wall.json')
  assert wall['returncode'] == 0 and not wall['timed_out']
  assert q['complete'] and q['mode'] == 'qualify' and q['plan_sha256'] == args.plan_sha
  assert q['gpu_executed'] and q['real_optimizer_updates'] == q['committed_step'] == 1 and q['patches'] == 720
  assert q['projected_pilot']['cumulative_bytes'] < C.LIMITS['cumulative_allocation_limit']
  assert q['projected_pilot']['seconds'] < 900
  return q


def child(args, p, train, validation):
  import numpy as np
  root = args.output_dir.resolve(); C.no_devices()
  assert (root/'parent-invocation.json').is_file()
  K = availability(args)
  fresh = K.passive_availability(); K.require_no_observed_conflict(fresh)
  assert fresh['vram_free_bytes'] >= C.LIMITS['live_allocation_limit']+C.LIMITS['free_reserve']
  C.write(root/'availability-before-open.json', fresh)
  def hardware_guard(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)):
      path = os.fsdecode(values[0])
      if path.startswith(('/dev/dri/', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb/')):
        assert path in ('/dev/kfd', '/dev/dri/renderD129'), path
  sys.addaudithook(hardware_guard)
  sys.path.insert(0, str(C.TINY))
  from tinygrad import Tensor, Device
  from tinygrad.helpers import Context, DEV, MAX_BUFFER_SIZE, PROFILE, USE_TC, ALLOW_TF32, NO_MEMORY_PLANNER, LRU
  from tinygrad.runtime.ops_amd import KFDIface, PMC, SQTT
  from tinygrad.nn.optim import AdamW
  assert not Device._opened_devices and repr(DEV) == C.TARGET and DEV.target('AMD').indices == '1'
  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value == C.LIMITS['max_single_buffer']
  assert USE_TC.value == ALLOW_TF32.value == 0 and NO_MEMORY_PLANNER.value == 1 and LRU.value == 0
  L = C.module('agent_distill_vision_ledger', C.LEDGER); ledger = L.Ledger(C.LIMITS, K.passive_availability)
  KFDIface.alloc = ledger.alloc(KFDIface.alloc); KFDIface.free = ledger.free(KFDIface.free); KFDIface.unmap = ledger.unmap(KFDIface.unmap)
  S = C.module('agent_distill_vision_state', C.STATE)
  report = dict(complete=False, gpu_executed=False, real_optimizer_updates=0, committed_step=0,
                mode=args.mode, plan_sha256=args.plan_sha, patches=720, allocation_budget=ledger.report,
                quality_qualified=False, throughput_qualified=False, driving_model_modified=False,
                independent_anchors=dict(train=1, validation=1), teacher_reports=[p['train_report'], p['validation_report']])
  B = C.graph(); shapes = B.shapes(); started = time.monotonic()
  context = dict(plan_sha256=args.plan_sha, hyperparameters=C.HYPER, mode=args.mode)
  try:
    ledger.phase('device-and-seed'); dev = Device['AMD']
    identity = dict(requested_kfd_index=1, logical_device_id=dev.device_id, filtered_kfd_nodes=list(KFDIface.gpus),
                    interface=type(dev.iface).__name__, gpu_id=dev.iface.gpu_id, pci=Path(dev.iface.dev_sysfs_path).resolve().name,
                    render_minor=dev.iface.props['drm_render_minor'], arch=dev.arch, renderer_target=repr(dev.renderer.target),
                    is_usb=dev.is_usb, native_pci=dev.is_am(), can_recover=dev.can_recover, allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(identity, K.selected_node(K.passive_topology())); report['actual_device'] = identity
    arrays, _ = C.seeds(); block = B.Block(arrays, 'AMD'); del arrays
    opt = AdamW(list(block.weights.values()), lr=3.5e-6, b1=.9, b2=.999, eps=1e-8, weight_decay=.01, fused=False)
    initial_bias = block.weights['mlp.fc2.bias'].numpy().copy()

    def objective(row):
      data = {k: Tensor(row[k], device='AMD').is_param_(False).realize() for k in ('x', 'y')}
      assert data['x'].shape == data['y'].shape == (1, 720, 1152)
      return B.mse(block(data['x']), data['y']), data

    def evaluate(rows, step):
      ledger.phase('validation-'+str(step)); losses = []
      with Context(TRAINING=0):
        for row in rows:
          loss, data = objective(row); value = float(loss.item()); assert np.isfinite(value); losses.append(value)
          assert all(v.grad is None and not v.is_param for v in data.values())
      dev.synchronize(); report['gpu_executed'] = True
      return dict(step=step, mean=float(np.mean(losses)), per_image=losses, images=[r['id'] for r in rows])

    def best(metadata):
      temporary = root/'best-next.npz'
      with temporary.open('xb') as stream:
        np.savez(stream, **{name: value.numpy() for name, value in block.weights.items()}); stream.flush(); os.fsync(stream.fileno())
      os.replace(temporary, root/'best.npz')
      S.atomic_json(root/'best.json', dict(file='best.npz', sha256=C.sha(root/'best.npz'),
                                         selected_step=metadata['best_step'], validation_mean=metadata['best_validation']))

    forward_begin = time.monotonic(); before_forward = ledger.report['requested_bytes']
    if args.mode == 'qualify':
      item = evaluate(train[:1], 0)
      metadata = dict(step=0, qualification_train_mse=item['mean'])
    elif args.resume_from:
      old = args.resume_from.resolve(); restored, metadata = S.load(old, context, shapes)
      assert 0 <= metadata['step'] < 25 and metadata['best_step'] == 0
      previous = C.read(old/'best.json')
      assert previous['file'] == 'best.npz' and C.sha(old/'best.npz') == previous['sha256']
      assert previous['selected_step'] == 0 and previous['validation_mean'] == metadata['best_validation']
      S.restore(block.weights, opt, restored); del restored
      shutil.copyfile(old/'best.npz', root/'best.npz'); S.atomic_json(root/'best.json', previous)
    else:
      item = evaluate(validation, 0)
      metadata = dict(step=0, best_validation=item['mean'], best_step=0, validation=[item]); best(metadata)
    forward_seconds = time.monotonic()-forward_begin
    forward_bytes = ledger.report['requested_bytes']-before_forward
    ledger.phase('checkpoint-initial'); S.save(root, S.snapshot(block.weights, opt), metadata, context, shapes,
                                             initial_recovery=bool(args.resume_from))
    setup_seconds = time.monotonic()-started
    starting_step = metadata['step']; report['starting_checkpoint_step'] = starting_step
    final_step = 1 if args.mode == 'qualify' else 25
    for step in range(metadata['step']+1, final_step+1):
      update_begin = time.monotonic(); before_update = ledger.report['requested_bytes']; ledger.phase('update-'+str(step))
      S.intent(root, step-1); row = train[(step-1) % 10]
      with Context(TRAINING=1):
        opt.zero_grad(); loss, data = objective(row); grads = loss.gradient(*block.weights.values()); Tensor.realize(*grads)
        value = float(loss.item()); assert np.isfinite(value)
        for parameter, gradient in zip(block.weights.values(), grads, strict=True):
          assert bool(gradient.isfinite().all().item()); parameter.grad = gradient
        assert float(block.weights['mlp.fc2.weight'].grad.square().sum().item()) > 0, 'No differentiable student gradient'
        assert all(v.grad is None and not v.is_param for v in data.values())
        opt.step(); dev.synchronize()
      report['gpu_executed'] = True; report['real_optimizer_updates'] += 1
      if step == 1:
        updated_bias = block.weights['mlp.fc2.bias'].numpy()
        assert np.any(updated_bias != initial_bias), 'AdamW did not change the student'
        report['first_update_bias_max_delta'] = float(np.max(np.abs(updated_bias-initial_bias)))
      del grads, gradient, parameter, loss, data; opt.zero_grad(); gc.collect()
      metadata['step'] = step
      if args.mode == 'pilot' and step == 25:
        item = evaluate(validation, step); metadata['validation'].append(item)
        if item['mean'] < metadata['best_validation']:
          metadata['best_validation'] = item['mean']; metadata['best_step'] = step; best(metadata)
      ledger.phase('checkpoint-'+str(step)); C.storage(root)
      S.save(root, S.snapshot(block.weights, opt), metadata, context, shapes)
      report.update(committed_step=step, last_training_image=row['id'], last_training_mse=value)
      if args.mode == 'qualify':
        update_seconds = time.monotonic()-update_begin; update_bytes = ledger.report['requested_bytes']-before_update
        # This includes a compiled first update and a complete recovery checkpoint.
        # Twenty validation forwards and two best snapshots receive extra readback allowance.
        report['measurements'] = dict(setup_seconds=setup_seconds, forward_seconds=forward_seconds,
                                       forward_bytes=forward_bytes, update_checkpoint_seconds=update_seconds,
                                       update_checkpoint_bytes=update_bytes)
        report['projected_pilot'] = dict(cumulative_bytes=ledger.report['requested_bytes'] + 25*update_bytes + 20*forward_bytes + 4*p['parameter_count']*4,
                                         seconds=setup_seconds + 25*update_seconds + 20*forward_seconds + 30,
                                         basis='Cold measured update/checkpoint, 20 single-image forwards, plus readback and 30-second margin')
      else: report.update(best_step=metadata['best_step'], validation=metadata['validation'])
      S.atomic_json(root/'report.json', report)
    assert report['real_optimizer_updates']+starting_step == final_step
    report.update(complete=True, selection='qualification has no selection; pilot compares validation MSE at steps 0 and 25 only')
  finally:
    report['seconds'] = time.monotonic()-started
    report['allocator_guards_retained_through_finalizers'] = True
    S.atomic_json(root/'report.json', report)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--prepare', action='store_true')
  parser.add_argument('--plan', type=Path, default=C.ROOT/'plan.json'); parser.add_argument('--plan-sha')
  parser.add_argument('--mode', choices=('qualify', 'pilot'), default='qualify')
  parser.add_argument('--execute-gpu', action='store_true'); parser.add_argument('--child', action='store_true')
  parser.add_argument('--snapshot-availability', action='store_true')
  parser.add_argument('--availability-snapshot', type=Path); parser.add_argument('--reviewed-availability-sha')
  parser.add_argument('--qualification', type=Path); parser.add_argument('--qualification-sha')
  parser.add_argument('--output-dir', type=Path); parser.add_argument('--resume-from', type=Path)
  args = parser.parse_args()
  if sys.flags.optimize: raise RuntimeError('Assertions are required; do not use python -O')
  if not args.child: os.environ.update(DEV='CPU', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  if args.prepare:
    assert not args.execute_gpu and not args.child; return prepare()
  assert args.plan_sha; C.no_devices(); p = C.verify_plan(args.plan, args.plan_sha)
  if args.snapshot_availability:
    assert not args.execute_gpu and not args.child and args.availability_snapshot
    C.storage(args.availability_snapshot)
    K = C.module('agent_distill_vision_snapshot', C.KFD); C.write(args.availability_snapshot, K.passive_availability()); C.no_devices()
    print(json.dumps(dict(snapshot=str(args.availability_snapshot), sha256=C.sha(args.availability_snapshot), gpu_executed=False))); return
  train = C.capture(C.TRAIN, C.TRAIN_SHA, 'train'); validation = C.capture(C.VALIDATION, C.VALIDATION_SHA, 'validation')
  if not args.execute_gpu:
    assert not args.child
    C.no_devices(); print(json.dumps(dict(preflight_passed=True, gpu_executed=False, optimizer_updates=0,
      train_images=len(train), validation_images=len(validation), independent_anchors=dict(train=1, validation=1)))); return
  assert args.output_dir; root = args.output_dir.resolve(); C.storage(root)
  assert root.name.startswith(('qualify-', 'pilot-'))
  if args.resume_from:
    assert args.mode == 'pilot' and args.resume_from.resolve().is_relative_to(C.ROOT) and args.resume_from.resolve() != root
  if args.mode == 'pilot': qualification(args)
  availability(args)
  if args.child:
    assert os.environ.get('AGENT_DISTILL_VISION_PARENT_PID') == str(os.getppid())
    return child(args, p, train, validation)
  assert not root.exists()
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root.mkdir(); (root/'tmp').mkdir()
    command = [str(C.PY), str(Path(__file__).resolve()), *sys.argv[1:], '--child']
    timeout = 300 if args.mode == 'qualify' else 900
    C.write(root/'parent-invocation.json', dict(command=command, timeout_seconds=timeout, plan_sha256=args.plan_sha))
    env = environment(root); env['AGENT_DISTILL_VISION_PARENT_PID'] = str(os.getpid())
    started = time.monotonic(); timed_out = False
    with (root/'run.log').open('w') as stream:
      process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
      try: code = process.wait(timeout=timeout)
      except subprocess.TimeoutExpired:
        timed_out = True; os.killpg(process.pid, signal.SIGTERM)
        try: code = process.wait(timeout=5)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); code = process.wait()
    C.write(root/'wall.json', dict(seconds=time.monotonic()-started, returncode=code, timed_out=timed_out))
    assert code == 0 and not timed_out and C.read(root/'report.json')['complete']


if __name__ == '__main__': main()
