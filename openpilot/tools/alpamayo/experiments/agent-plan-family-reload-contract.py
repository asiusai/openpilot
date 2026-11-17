"""Seal the actual independently proved serial family. No runtime plan execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
BASE=WORK/'agent-plan-family-protocol1.json'
BASE_SHA='32d8c29b5c5b45c49a382b127af4342c305d83b867c15f7fc7597c909f709047'
CAPTURE=WORK/'agent-plan-family-capture1/report.json'
CAPTURE_SHA='3666a1c88d517fc945f6f63f4207da000ba53d84c042af88d10a5c2cfa31e10d'
OFFLINE=WORK/'agent-plan-family-capture1-offline.json'
OFFLINE_SHA='c5e05efc9565e1160fb4da3c53d3ad6920eab93dbb1c224f5b0b0942e15e6ba6'
SOURCES=tuple('agent-plan-family-reload-'+n+'.py' for n in ('contract','loader','diagnostic','run','check','parent'))
LIMITS=dict(child_seconds=300,outer_timeout_seconds=0,start_next_count_before_seconds=225,
            live_tensor_bytes=7*1024**3,work_reserve_bytes=5*1024**3)

def local(path):
  spec=importlib.util.spec_from_file_location('family_reload_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def family(): return local(EXP/'agent-plan-family-contract.py')

def evidence():
  if sha(BASE)!=BASE_SHA or sha(CAPTURE)!=CAPTURE_SHA or sha(OFFLINE)!=OFFLINE_SHA: raise ValueError('Frozen family capture/proof/protocol changed')
  f=family();f.source_gate(json.loads(BASE.read_text()))
  capture=json.loads(CAPTURE.read_text());offline=json.loads(OFFLINE.read_text());result=capture['plan_family']
  if (not result['capture_completed'] or not all(result['restored'].values()) or result['counts']!=[2,3,4,5,6] or
      result['current_inputs']!=1 or result['future_inputs']!=0 or result['count1_reexecuted']): raise ValueError('Family capture scope differs')
  if (not offline['all_requested_scratch_proofs_passed'] or offline['devices'] or offline['gpu_executed'] or
      offline['protocol_sha256']!=BASE_SHA or offline['capture_report_sha256']!=CAPTURE_SHA): raise ValueError('Offline proof incomplete')
  files={str(p):sha(p) for p in (BASE,CAPTURE,OFFLINE,Path('/tmp/opencode/run-alpamayo-speed-trial.py'))}
  members={}
  for count in range(2,7):
    row=result['plans'][str(count)];proved=offline['plans'][str(count)]
    if row!=proved['capture'] or proved['contract']!=row['contract'] or not proved['scratch_initialization_proved']:
      raise ValueError('Per-count proof linkage differs')
    for path,digest in ((row['audit'],row['audit_sha256']),(row['sidecar'],row['sidecar_sha256']),
                        (proved['proof'],proved['proof_sha256'])):
      p=Path(path)
      if not p.resolve().is_relative_to(WORK) or p.is_symlink() or sha(p)!=digest: raise ValueError('Actual per-count artifact changed')
      files[str(p)]=digest
    doc=json.loads(Path(row['audit']).read_text())
    for record in doc['programs'].values():
      root=Path(row['audit']).parent/'programs';p=root/record['file']
      if p.is_symlink() or sha(p)!=record['json_sha256']: raise ValueError('Original PROGRAM JSON changed')
      files[str(p)]=record['json_sha256']
      for digest,size in record['binaries'].items():
        p=root/(digest+'.elf')
        if p.is_symlink() or p.stat().st_size!=size or sha(p)!=digest: raise ValueError('Original numerical ELF changed')
        files[str(p)]=digest
    members[str(count)]=dict(capture=row,proof=proved['proof'],proof_sha256=proved['proof_sha256'])
  return files,members

def gate(protocol):
  if protocol['schema']!=1 or protocol['base']!={'path':str(BASE),'sha256':BASE_SHA}: raise ValueError('Reload base changed')
  if protocol['sources']!={name:sha(EXP/name) for name in SOURCES}: raise ValueError('Reload source changed')
  if protocol['limits']!=LIMITS or protocol['counts'] not in ([2,3,4,5,6],[6]): raise ValueError('Budget or counts changed')
  if protocol['target']!='USB+AMD:LLVM:gfx1200': raise ValueError('Full target changed')
  files,members=evidence()
  if protocol['artifacts']!=files or protocol['members']!=members: raise ValueError('Actual plan/proof/ELF allowlist changed')
  if len(protocol['cpu_evidence'])!=1: raise ValueError('One final CPU evidence file is required')
  for path,digest in protocol['cpu_evidence'].items():
    if sha(path)!=digest: raise ValueError('CPU evidence changed')
    checked=json.loads(Path(path).read_text())
    if (not checked['passed'] or checked['gpu_executed'] or checked['sources']!=protocol['sources'] or
        checked['production_mutation_cases']!=30 or not checked['all_five_proofs_recomputed_exact']):
      raise ValueError('Full-family production CPU evidence does not match these sources')
  return protocol

def read_member(count,protocol,reprove=False):
  if type(count) is not int or count not in protocol['counts']: raise ValueError('Count is outside this sealed protocol')
  entry=protocol['members'][str(count)];summary=entry['capture'];path=Path(summary['audit']);sidepath=Path(summary['sidecar'])
  if sha(CAPTURE)!=CAPTURE_SHA or sha(OFFLINE)!=OFFLINE_SHA: raise ValueError('Authoritative member evidence changed')
  original=json.loads(CAPTURE.read_text())['plan_family']['plans'][str(count)]
  proved=json.loads(OFFLINE.read_text())['plans'][str(count)]
  if entry!=dict(capture=original,proof=proved['proof'],proof_sha256=proved['proof_sha256']):
    raise ValueError('Only the original per-count artifacts are allowed')
  for p in (path,sidepath,Path(entry['proof'])):
    if sha(p)!=protocol['artifacts'][str(p)]: raise ValueError('Member evidence changed')
  doc=json.loads(path.read_text());side=json.loads(sidepath.read_text());proof=json.loads(Path(entry['proof']).read_text())
  f=family();member=f.validate_document(doc,count,f.signatures()[str(count)],side['output_views'])
  if member!=summary['contract'] or not proof['proved'] or proof['calls']!=len(doc['calls']): raise ValueError('Member proof contract differs')
  if proof['scratch_bytes']!=member['scratch_bytes'] or set(proof['full_output_writes'])!={'return/logits','return/token'}:
    raise ValueError('Scratch or complete output proof differs')
  codec=local(EXP/'agent-plan-bind-codec.py')
  programs={key:codec.load_offline(path.parent/'programs',row) for key,row in doc['programs'].items()}
  if reprove and local(EXP/'agent-plan-reload-proof.py').prove(doc,programs)!=proof:
    raise ValueError('Recomputed instruction-level proof differs from sealed evidence')
  return doc,programs,proof,side['output_views']
