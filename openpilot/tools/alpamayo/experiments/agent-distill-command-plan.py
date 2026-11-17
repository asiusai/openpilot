"""CPU-only source freezing and FP8-cache admission for nine bounded internal captures."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_command_common_plan', HERE/'agent-distill-command-common.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, default=c.ROOT/'capture-v1')
  args = parser.parse_args(); root = args.output_dir.resolve()
  assert root.is_relative_to(c.ROOT) and not root.exists()
  input_manifest = c.ROOT/'inputs-v1/agent-distill-command-inputs.json'
  inputs = json.loads(input_manifest.read_text())
  assert inputs['admissible_for_internal_layer_capture'] and inputs['protocol'] == c.protocol()
  for path, digest in inputs['source_sha256'].items(): assert c.sha(path) == digest
  root.mkdir(parents=True, exist_ok=False)
  historical = c.WORK/'current-profile-dynamic-2048/report.current-profile-plan.json'
  old = json.loads(historical.read_text())
  frozen = c.WORK/'current-profile-dynamic-2048/report-sources'
  old_core = c.WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo'
  runtime = root/'runtime'; runtime.mkdir(); (runtime/'experiments').mkdir()
  copied = {}
  for path, digest in old['source_sha256'].items():
    p = Path(path)
    if p.parent == HERE.parent:
      source, target = old_core/p.name, runtime/p.name
    elif p.parent == HERE and (frozen/p.name).is_file():
      source, target = frozen/p.name, runtime/'experiments'/p.name
    else: continue
    assert c.sha(source) == digest, source
    target.write_bytes(source.read_bytes()); copied[str(target)] = digest
  assert c.sha(runtime/'model.py') == c.sha(old_core/'model.py')
  command = old['current_control_invocation']['command'][:]
  assert '--vision-calibration' not in command
  # Replace the entire protected replay list. These are the only new sensor examples.
  a, b = command.index('--samples')+1, command.index('--initial-noise')
  command[a:b] = [row['sample'] for row in inputs['samples']]
  for flag in ('--native-manifest', '--native-calibrations', '--native-frame-cache'):
    i = command.index(flag); del command[i:i+2]
  command[1] = str(runtime/'experiments/benchmark-pipeline.py')
  for flag, value in (('--warmup', '0'), ('--runs', '9'), ('--output', str(root/'parent-placeholder.json'))):
    command[command.index(flag)+1] = value
  artifacts = {}
  for flag in ('--prefill-calibration', '--initial-noise', '--reduced-vocab'):
    p = Path(command[command.index(flag)+1]); digest = c.sha(p)
    if str(p) in old['artifact_sha256']: assert digest == old['artifact_sha256'][str(p)]
    target = root/('agent-distill-command-'+p.name)
    target.write_bytes(p.read_bytes()); copied[str(target)] = digest
    command[command.index(flag)+1] = str(target)
    artifacts[flag] = dict(path=str(target), sha256=digest, original_path=str(p))
  assert c.PROTECTED not in json.dumps(command)
  assert command[command.index('--steps')+1] == '3' and command[command.index('--pad-prompt-to')+1] == '2048'
  environment = old['current_control_invocation']['environment'].copy()
  environment.update(ALPAMAYO_FP8_CACHE=str(c.CACHE), PYTHONPATH=str(c.BASE/'tinygrad')+os.pathsep+str(runtime),
    PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
  environment['CACHEDB'] = '<parent-owned fresh cache copied from immutable seed>'
  invocation = root/'agent-distill-command-invocation.json'
  # The parent capture helper replaces CACHEDB before invoking this launcher.
  recorded_environment = {k: v for k, v in environment.items() if k != 'CACHEDB'}
  c.write(invocation, dict(command=command, environment=recorded_environment))
  copied[str(invocation)] = c.sha(invocation)
  # Verify the known on-disk cache copy completely, before a device may initialize.
  index_path, weights_path = c.CACHE/'index.json', c.CACHE/'weights.bin'
  index = json.loads(index_path.read_text())
  assert c.sha(index_path) == 'c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e'
  assert index['format'] == 1 and weights_path.stat().st_size == index['nbytes'] == 2929010944
  assert c.sha(weights_path) == index['sha256'] == 'dd6b716a2ade3646eff71d874bef676926422d9aca39358fd3f8b4fe75755052'
  end, identities = 0, set()
  for row in sorted(index['records'], key=lambda r: r['offset']):
    identity = row['name'], row['kind']; assert identity not in identities; identities.add(identity)
    width = {'bf16': 2, 'scale': 4, 'fp8': 1}[row['kind']]
    elements = 1
    for n in row['shape']: elements *= n
    assert elements*width == row['nbytes'] and row['offset'] >= end
    end = row['offset']+row['nbytes']; assert end <= index['nbytes']
  # Independent CPU conversion/layout check for the same original FC1 admitted
  # by benchmark-fast's mandatory target-architecture check. Keep both gates.
  torch, _ = c.cpu_imports()
  import numpy as np
  from safetensors import safe_open
  master_index = json.loads((c.BASE/'model/model.safetensors.index.json').read_text())['weight_map']
  name = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
  with safe_open(c.BASE/'model'/master_index[name], framework='pt', device='cpu') as shard: master = shard.get_tensor(name)
  scale = (master.float().abs().max()/448).clamp(min=1e-12)
  codes = (master.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
  rows, width = codes.shape
  packed = codes.reshape(rows//16,16,width//16,4,4).transpose(0,2,3,1,4).copy().reshape(rows,width)
  checks = {}
  with weights_path.open('rb') as stream:
    for kind, data in [('scale', scale.numpy().tobytes()), ('fp8', packed.tobytes())]:
      record = next(r for r in index['records'] if r['name'] == name and r['kind'] == kind)
      stream.seek(record['offset']); actual = stream.read(record['nbytes'])
      checks[kind] = dict(equal=actual == data, source_sha256=hashlib.sha256(data).hexdigest(), cache_sha256=hashlib.sha256(actual).hexdigest())
      assert actual == data, f'CPU original-master cache {kind} differs'
  # Explicit original BF16 master shard provenance. No FP8 values become masters.
  shards = {str(c.BASE/'model'/name): c.sha(c.BASE/'model'/name) for name in sorted(set(master_index.values()))}
  seed_source = c.WORK/'agent-expert-combined-cache-artifacts/gpu-cache.db'
  seed = root/'agent-distill-command-compiler-cache-seed.db'
  source_db = sqlite3.connect(seed_source.as_uri()+'?mode=ro', uri=True)
  destination_db = sqlite3.connect(seed)
  try: source_db.backup(destination_db)
  finally: source_db.close(); destination_db.close()
  tinygrad = c.BASE/'tinygrad'
  revision = subprocess.check_output(['git', '-C', str(tinygrad), 'rev-parse', 'HEAD'], text=True).strip()
  assert revision == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  tiny_sources = {str(p): c.sha(p) for p in sorted((tinygrad/'tinygrad').rglob('*.py'))}
  owned = {str(p): c.sha(p) for p in HERE.glob('agent-distill-command-*.py')}
  plan = dict(schema=1, protocol='experimenter-command-layer-reconstruction-v1', gpu_executed=False,
    diagnostic_only=True, parent_timeout_seconds=300, expected_examples=9, max_model_prefills=9,
    capture_layers=[12,15], runtime_root=str(runtime), invocation_path=str(invocation), environment=recorded_environment,
    inputs_manifest=dict(path=str(input_manifest), sha256=c.sha(input_manifest)),
    historical_recipe=dict(path=str(historical), sha256=c.sha(historical),
      use='Frozen source identities and runtime settings only; all historical sensor/navigation/trajectory replay inputs removed.'),
    runtime_files_sha256={str(Path(p).relative_to(runtime)): digest for p, digest in copied.items() if Path(p).is_relative_to(runtime)},
    source_sha256={**copied, **owned, **tiny_sources}, tinygrad_revision=revision,
    checkpoint_shards_sha256=shards,
    fp8_cache=dict(path=str(c.CACHE), index_sha256=c.sha(index_path), weights_sha256=index['sha256'],
      nbytes=index['nbytes'], cpu_original_bf16_master_conversion_and_packing=checks,
      parent_gate='Unmodified benchmark-fast target GPU conversion/packing parity plus complete bounded upload readback/hash verification.'),
    compiler_cache_seed=dict(path=str(seed), sha256=c.sha(seed), bytes=seed.stat().st_size,
      provenance='SQLite read-only consistent backup of existing local compiler cache; parent uses a fresh owned copy.'),
    target_semantics=inputs['protocol']['targets'],
    capture_boundary='One original selected vision encode and original language layers0..15 per conditioned example; stop before LM output/reasoning/expert/action decoding.',
    scheduling='Three samples per evidenced drive; exact image features may be cached across commands. No previous reasoning, trajectory, or target is reused.',
    numerical_policy='No target-error thresholds or example/command selection. Reject incomplete/nonfinite/wrong-shape captures.',
    sealed_policy='Train/validation manifest excludes all sealed-test target paths. Sealed-test target manifest remains separate.',
    throughput_claim=False)
  c.assert_no_devices(); c.write(root/'agent-distill-command-capture-plan.json', plan)
  print(json.dumps(dict(plan=str(root/'agent-distill-command-capture-plan.json'), cache_verified_bytes=index['nbytes'],
    cpu_conversion_checks={k: v['equal'] for k,v in checks.items()}, compiler_cache_seed_bytes=seed.stat().st_size, gpu_executed=False), indent=2))


if __name__ == '__main__': main()
