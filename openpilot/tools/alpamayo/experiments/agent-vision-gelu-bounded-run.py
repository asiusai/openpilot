"""Bounded retry: immutable numeric programs, complete integer flags, explicit stage/IO evidence."""
import argparse
from contextlib import contextmanager
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD_PROTOCOL = HERE/'agent-vision-gelu-micro-frozen/agent-vision-gelu-micro-protocol.json'
OLD = WORK/'agent-vision-gelu-micro-parent'
OLD_PROTOCOL_SHA = '2d48dd3976dd317ab93cd306a9b95ebec995a46b093bb1caec34ff15507658e3'
OLD_REPORT_SHA = '4a5f00c92f4185f6b2edc41306ee6cceab1e6d876c13256a05dbff5b759d7fca'


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


M = local('agent-vision-gelu-micro-run')
V = local('agent-vision-gelu-bounded-validate')


def evidence():
  assert M.digest(OLD_PROTOCOL) == OLD_PROTOCOL_SHA and M.digest(OLD/'report.json') == OLD_REPORT_SHA
  protocol, old = json.loads(OLD_PROTOCOL.read_text()), json.loads((OLD/'report.json').read_text())
  M.verify(protocol)
  assert old['gpu_executed'] and not old['completed'] and not old['native_gelu_equality_proven']
  assert old['failure'] == 'Hard bounded parent diagnostic timeout' and 'timings' not in old and 'batch_controls' not in old
  assert [row['state'] for row in old['single_controls']] == list(M.STATES[:5])
  assert all(row['exact'] and row['all_finite'] and row['actual_changed_input_tinyjit_replay'] for row in old['single_controls'])
  assert old['active_programs'] == {name:plan['records'] for name, plan in protocol['plans'].items()}
  files = [OLD_PROTOCOL, *sorted(OLD.glob('*.*'))]
  for name, records in old['active_programs'].items():
    proof = old['single_queue_proof'][name]
    assert proof['active_program_keys'] == [row['program_key'] for row in records]
    assert proof['mappings'] == [row['mapping'] for row in records]
    for i, row in enumerate(records):
      for suffix, field in (('source', 'source_sha256'), ('hsaco', 'binary_sha256')):
        path = OLD/'agent-vision-gelu-micro-active'/f'{name}-{i}.{suffix}'
        assert M.digest(path) == row[field]
        files.append(path)
  return protocol, old, {str(p):M.digest(p) for p in files}


def diagnosis(protocol):
  layout = protocol['plans']['original']['layout']
  output = sum(row['nbytes'] for row in layout[1:])
  incoming = layout[0]['nbytes']
  flags = sum(((row['nbytes']//({'dtypes.bf16':2, 'dtypes.f32':4, 'dtypes.fp8e4m3':1}[row['dtype']])+V.BLOCK-1)//V.BLOCK)*4
              for row in layout[1:])
  return dict(old_input_bytes=incoming, old_seven_output_bytes=output,
    old_explicit_upload_per_control=incoming+2*output, old_explicit_download_per_control=2*output,
    five_completed_controls_minimum_explicit_transfer_bytes=5*(incoming+4*output),
    old_nineteen_controls_explicit_transfer_bytes=19*(incoming+4*output),
    transfer_path='USBMMIOInterface writes use PCIe bulk OUT with 1 MiB streaming chunks; reads use 64 KiB BAR chunks.',
    no_stage_timing_or_stack_in_old_evidence=True,
    bottleneck_proven=False, interpretation='Large synchronous transfer volume is identified; its share of 295 seconds is unmeasured.',
    retry_fixture_resident_upload_bytes=7*incoming, retry_all_seven_flags_bytes_per_comparison=flags,
    retry_full_output_host_readback_bytes=0, old_status_preserved=True)


def verify(protocol):
  original, old, files = evidence()
  assert protocol['schema'] == 1 and protocol['timeout_seconds'] == 300 and protocol['operations'] == 40
  assert protocol['shape'] == list(M.SHAPE) and protocol['states'] == list(M.STATES)
  assert protocol['old_protocol_sha256'] == OLD_PROTOCOL_SHA and protocol['old_report_sha256'] == OLD_REPORT_SHA
  for path, expected in protocol['source_sha256'].items(): assert M.digest(path) == expected, f'Changed source/evidence: {path}'
  assert all(protocol['source_sha256'][path] == value for path, value in files.items())
  assert protocol['numeric_plans'] == original['plans']
  return original, old


class Progress:
  def __init__(self, report, output):
    self.report, self.output, self.started, self.io = report, output, time.monotonic(), None
    self.report['stages'] = []
    self.report['started_monotonic'] = self.started

  def save(self):
    self.report['elapsed_seconds'] = time.monotonic()-self.started
    if stage:=self.report.get('active_stage'): stage['elapsed_seconds'] = self.report['elapsed_seconds']-stage['started_elapsed_seconds']
    if self.io: self.report['io'] = dict(self.io.counters)
    M.save(self.output, self.report)

  @contextmanager
  def stage(self, name):
    start = time.monotonic()
    before = dict(self.io.counters) if self.io else {}
    self.report['active_stage'] = dict(name=name, started_elapsed_seconds=start-self.started, counters_before=before)
    self.save()
    try: yield
    except BaseException:
      self.report['active_stage']['failed'] = True
      self.save()
      raise
    else:
      now = dict(self.io.counters) if self.io else {}
      self.report['stages'].append(dict(name=name, seconds=time.monotonic()-start,
        counters_delta={key:value-before.get(key,0) for key,value in now.items()}))
      self.report['active_stage'] = None
      self.save()


class IO:
  def __init__(self, device, artifacts):
    self.device, self.artifacts = device, artifacts
    artifacts.mkdir(exist_ok=True)
    self.counters = dict(upload_bytes=0, download_bytes=0, upload_calls=0, download_calls=0,
                         gpu_copy_bytes=0, gpu_fill_bytes=0, allocated_bytes=0)

  def allocate(self, shape, dtype, raw=None):
    value = M.tensor_buffer(self.device.device, shape, dtype, raw)
    self.counters['allocated_bytes'] += value.nbytes()
    if raw is not None:
      self.counters['upload_bytes'] += len(raw); self.counters['upload_calls'] += 1
    return value

  def write(self, tensor, raw):
    M.write_buffer(tensor, raw)
    self.counters['upload_bytes'] += len(raw); self.counters['upload_calls'] += 1

  def read(self, tensor):
    raw = M.read_buffer(tensor, self.device)
    self.counters['download_bytes'] += len(raw); self.counters['download_calls'] += 1
    return raw


def numeric_audit(plans, original, old, runners=()):
  for name, plan in plans.items():
    assert plan['layout'] == original['plans'][name]['layout']
    assert plan['records'] == original['plans'][name]['records'] == old['active_programs'][name]
    for (program, mapping), frozen in zip(plan['steps'], plan['records'], strict=True):
      assert M.program_record(program) == {key:value for key,value in frozen.items() if key not in ('mapping','ast_key')}
      assert mapping == frozen['mapping']
  return {label:runner.identity() for label, runner in runners}


def checks(runners, inputs, refs, initial, resident, verifier, progress, report, phase, states=range(8)):
  import numpy as np
  device, io = verifier.device, verifier.io
  labels = M.LABELS[1:]
  current, saved = dict(zip(labels, refs, strict=True)), dict(zip(labels, initial, strict=True))
  records = []
  for state in states:
    prefix = f'{phase}-{M.STATES[state]}'
    with progress.stage(prefix+'-input-copy-check'):
      verifier.copy(inputs[0], resident[0 if state == 7 else state], 'bf16')
      verifier.compare('activation', inputs[0], resident[0 if state == 7 else state], prefix+'-input')
    for label in ('original', 'candidate'):
      with progress.stage(prefix+'-'+label+'-poison'):
        verifier.poison(inputs[1:]); device.synchronize()
      with progress.stage(prefix+'-'+label+'-execute'):
        before = runners[label].jit.cnt
        runners[label](inputs); device.synchronize()
        assert before >= 2 and runners[label].jit.cnt == before+1
      if label == 'original':
        with progress.stage(prefix+'-save-original-on-device'):
          for name, dest, source in zip(labels, refs, inputs[1:], strict=True): verifier.copy(dest, source, V.FORMATS[name])
          device.synchronize()
      else:
        with progress.stage(prefix+'-compare-all-seven'):
          for name, reference, result in zip(labels, refs, inputs[1:], strict=True): verifier.compare(name, reference, result, prefix)
    with progress.stage(prefix+'-mutation-restoration-and-scales'):
      scale, maximum = io.read(current['scale']), io.read(current['maximum'])
      assert np.isfinite(np.frombuffer(scale, np.float32)).all() and np.isfinite(np.frombuffer(maximum, np.float32)).all()
      if phase == 'single' and state == 0:
        for name, dest, source in zip(labels, initial, refs, strict=True): verifier.copy(dest, source, V.FORMATS[name])
        device.synchronize()
        report['initial_per_image_scale_bytes_hex'] = scale.hex()
      elif state in (0, 7):
        for name in labels: verifier.compare(name, saved[name], current[name], prefix+'-restoration')
      else:
        verifier.compare('activation', saved['activation'], current['activation'], prefix+'-changed', changed=True)
        initial_scale = bytes.fromhex(report['initial_per_image_scale_bytes_hex'])
        assert scale != initial_scale, 'Mutation must change dynamic image scale'
        if state in (1, 2, 3, 5):
          same_image = 1 if state in (1, 3, 5) else 0
          assert scale[same_image*4:same_image*4+4] == initial_scale[same_image*4:same_image*4+4]
        if state == 3: assert np.frombuffer(scale, np.float32)[0] == np.float32(1e-12)
    records.append(dict(state=M.STATES[state], exact_all_seven=True, all_finite=True,
      actual_changed_input_tinyjit_replay=True, all_output_poison=True, input_copy_checked_every_element=True,
      output_hashes_used=False, input_fixture_sha256=report['fixtures'][0 if state == 7 else state]['sha256'],
      per_image_scale=np.frombuffer(scale, np.float32).tolist(), per_image_maximum=np.frombuffer(maximum, np.float32).tolist()))
    report[phase+'_controls'] = records
    report['validation'] = verifier.records
    progress.save()


def gpu(protocol, output):
  import numpy as np
  from tinygrad import Device, dtypes
  from tinygrad.device import Compiled
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  report = dict(completed=False, gpu_executed=False, native_gelu_equality_proven=False, adoption_allowed=False,
    model_hook_installed=False, diagnostic_only=True, prior_report_sha256=OLD_REPORT_SHA,
    prior_result='Incomplete timeout; five single controls passed. No saved control substitutes for any retry check.',
    source_sha256=protocol['source_sha256'], diagnosis=protocol['diagnosis'],
    explicit_io_counter_scope='Completed operand and validation payloads; excludes command uploads, queue fences, and runtime setup traffic.')
  progress = Progress(report, output); progress.save()
  try:
    with progress.stage('verify-source-and-immutable-evidence'): original, old = verify(protocol)
    with progress.stage('device-and-compiler-setup'):
      Compiled.wait_timeout_ms = 10000
      M.load('prefill-kernels').fix_dispatch_packet()
      device = Device[Device.DEFAULT]
      report['gpu_executed'] = True
      assert device.is_usb and device.arch == 'gfx1200' and isinstance(device.renderer, AMDLLVMRenderer)
      M.load('wgp_compiler').install(device)
    with progress.stage('numeric-and-validator-compile-identity'):
      plans = {name:M.build_plan(name, device.renderer) for name in ('original', 'candidate')}
      numeric_audit(plans, original, old)
      layout = plans['original']['layout']
      programs, helper_records = V.compile_all(layout, device.renderer, M)
      assert helper_records == protocol['helper_programs']
      report['active_numeric_programs'] = {name:plan['records'] for name,plan in plans.items()}
      report['active_helper_programs'] = helper_records
      artifacts = output.parent/'agent-vision-gelu-bounded-active'
      artifacts.mkdir(exist_ok=True)
      for name, program in [(f'{key}-{i}',p) for key,plan in plans.items() for i,(p,_) in enumerate(plan['steps'])]+list(programs.items()):
        (artifacts/(name+'.source')).write_text(program.src[2].arg)
        (artifacts/(name+'.hsaco')).write_bytes(program.src[3].arg)
    io = IO(device, output.parent/'agent-vision-gelu-bounded-flags')
    progress.io = io
    with progress.stage('allocate-shared-input-output-and-device-references'):
      inputs = tuple(io.allocate(row['shape'], dt) for row,dt in zip(layout, plans['original']['dtypes'], strict=True))
      refs, initial = [tuple(io.allocate(row['shape'], dt) for row,dt in zip(layout[1:],plans['original']['dtypes'][1:],strict=True)) for _ in range(2)]
      verifier = V.Verifier(io, layout, programs, M)
      report['validation'] = verifier.records
    resident = []
    report['fixtures'] = []
    for state in range(7):
      with progress.stage(f'fixture-generate-upload-{M.STATES[state]}'):
        bits = M.fixture(state)
        digest = M.sha(bits.tobytes())
        assert digest == protocol['fixtures'][state]['sha256']
        resident.append(io.allocate(M.SHAPE, dtypes.bfloat16, bits.tobytes()))
        report['fixtures'].append(dict(state=M.STATES[state], sha256=digest, bytes=bits.nbytes))
    verifier.selftest(inputs[1:], refs, progress.stage)
    report['validation'] = verifier.records
    report['native_validator_controls_passed'] = True
    with progress.stage('initialize-mutable-numeric-input'):
      verifier.copy(inputs[0], resident[0], 'bf16'); device.synchronize()
    single = {name:M.Runner(plan,1) for name,plan in plans.items()}
    for name, runner in single.items():
      with progress.stage('capture-single-'+name): runner(inputs); runner(inputs); device.synchronize()
    report['single_queue_proof'] = numeric_audit(plans, original, old, single.items())
    checks(single, inputs, refs, initial, resident, verifier, progress, report, 'single')
    batch = {name:M.Runner(plan,40) for name,plan in plans.items()}
    for name, runner in batch.items():
      with progress.stage('capture-forty-'+name): runner(inputs); device.synchronize()
    report['batch_queue_proof'] = numeric_audit(plans, original, old, batch.items())
    checks(batch, inputs, refs, initial, resident, verifier, progress, report, 'batch')
    with progress.stage('pre-timing-source-binary-and-queue-audit'):
      verify(protocol); numeric_audit(plans, original, old, batch.items()); verifier.dispatch.audit()
    pairs = []
    before = dict(io.counters)
    for repeat in range(3):
      order = ('original','candidate') if repeat%2 == 0 else ('candidate','original')
      row = dict(order=list(order), batch_ms={})
      with progress.stage(f'timing-pair-{repeat}'):
        for name in order:
          device.synchronize()
          start = time.perf_counter()
          batch[name](inputs)
          device.synchronize()
          row['batch_ms'][name] = (time.perf_counter()-start)*1000
      pairs.append(row)
    assert io.counters == before, 'Operand/validator work entered timed batches'
    checks(batch, inputs, refs, initial, resident, verifier, progress, report, 'post_timing', states=(0,6,7))
    with progress.stage('post-timing-source-binary-and-queue-audit'):
      verify(protocol)
      report['post_numeric_queue_proof'] = numeric_audit(plans, original, old, batch.items())
      assert V.compile_all(layout, device.renderer, M)[1] == helper_records
      report['native_helper_capture_identities'] = verifier.dispatch.audit()
    medians = {name:float(np.median([row['batch_ms'][name] for row in pairs]))/40 for name in batch}
    report.update(completed=True, native_gelu_equality_proven=True,
      native_equality_scope='This frozen finite-input diagnostic only; model adoption remains disabled.',
      timings=dict(pairs=pairs, median_ms_per_boundary=medians, improvement_ms=medians['original']-medians['candidate'],
        operations_per_batch=40, completion_fences_per_batch=1, source_binary_postchecks=True,
         throughput_or_route_claim=False, numeric_kernels_per_batch=dict(original=280,candidate=240)))
    progress.save()
    print(json.dumps(dict(completed=True,timings=report['timings'],io=report['io'])),flush=True)
  except BaseException as exc:
    report.update(completed=False,native_gelu_equality_proven=False,failure=f'{type(exc).__name__}: {exc}')
    report.pop('timings',None); progress.save()
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol',required=True,type=Path)
  parser.add_argument('--output',required=True,type=Path)
  parser.add_argument('--parent-gpu',action='store_true')
  parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
  args = parser.parse_args()
  assert args.parent_gpu
  protocol = json.loads(args.protocol.read_text())
  original,_ = verify(protocol)
  if args.child:
    assert os.environ.get('AGENT_VISION_GELU_BOUNDED_CHILD') == '1'
    for part in original['environment']['PYTHONPATH'].split(os.pathsep): sys.path.insert(0,part)
    with (args.output.parent/'agent-vision-gelu-bounded-stack.txt').open('w') as stack:
      faulthandler.enable(file=stack)
      faulthandler.dump_traceback_later(270,file=stack)
      try: gpu(protocol,args.output)
      finally: faulthandler.cancel_dump_traceback_later()
    return
  args.output.parent.mkdir(parents=True,exist_ok=True)
  assert not args.output.exists(), 'Use a fresh retry output path'
  M.save(args.output,dict(completed=False,native_gelu_equality_proven=False,gpu_executed=False))
  env = M.environment(original)
  env.update(CACHEDB=str(HERE/'agent-vision-gelu-bounded-cache.db'),AGENT_VISION_GELU_BOUNDED_CHILD='1')
  command = [original['python'],str(Path(__file__).resolve()),'--protocol',str(args.protocol.resolve()),
             '--output',str(args.output.resolve()),'--parent-gpu','--child']
  process = subprocess.Popen(command,env=env,start_new_session=True)
  try: code = process.wait(timeout=295)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=2)
    report = json.loads(args.output.read_text())
    report.update(completed=False,native_gelu_equality_proven=False,failure='Hard bounded retry timeout after 295 seconds')
    if start:=report.get('started_monotonic'):
      report['elapsed_seconds'] = time.monotonic()-start
      if stage:=report.get('active_stage'): stage.update(elapsed_seconds=report['elapsed_seconds']-stage['started_elapsed_seconds'],timed_out=True)
    report.pop('timings',None); M.save(args.output,report)
    raise SystemExit(124)
  raise SystemExit(code)


if __name__ == '__main__': main()
