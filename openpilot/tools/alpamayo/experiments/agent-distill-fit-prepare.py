"""Pin one blocked pilot and verify only the six actual train/validation targets. No model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_fit_common_prepare',HERE/'agent-distill-fit-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def main():
  C.no_devices();C.ROOT.mkdir(parents=True,exist_ok=True)
  assert subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(C.ROOT)],text=True).strip()=='/mnt/nvme-500'
  path=C.ROOT/'agent-distill-fit-protocol.json';assert not path.exists()
  report_path=C.WORK/'agent-distill-command-gpu/report.json';report=C.read(report_path)
  assert report['complete'] and report['gpu_executed'] and report['prefills']==9 and len(report['completed_examples'])==9
  assert not report['driving_outputs_executed'] and not report['training_executed'] and report['launcher_hooks_restored']
  rows=C.records();by_id={r['id']:r for r in report['completed_examples']}
  summaries=[]
  for row in rows:
    assert row==by_id[row['id']]
    values=C.load_target(row,rows)
    summaries.append(dict(id=row['id'],split=row['split'],actual_prompt_tokens=row['actual_prompt_tokens'],
      finite_fp32=True,source_dtype='uint16 original BF16 bits',interpretation='uint32(bits)<<16, reinterpret float32',
      shapes={k:list(v.shape) for k,v in values.items()},source_sha256=row['capture_sha256']))
  sealed=Path(report['sealed_manifest'])
  assert C.sha(sealed)=='7caddaa9f006dc1eabb658c8f2abbc9dafcef78cdaf108a0b29a192a590e899d'
  # Only the held-out manifest checksum is read. No held-out NPZ is opened.
  oracle=C.read(C.WORK/'agent-distill-tinygrad-artifacts/oracle.json')
  native_config=C.read(C.BASE/'model/config.json')['vlm_config']['text_config']
  prefix='vlm.model.language_model.layers.12.'
  weights={name.removeprefix(prefix):dict(bf16_sha256=digest,shape=oracle['arrays']['w:'+name.removeprefix(prefix)]['shape'])
    for name,digest in oracle['student_bf16_master_sha256'].items()}
  assert len(weights)==8 and sum(__import__('math').prod(w['shape']) for w in weights.values())==50335744
  numeric=C.WORK/'agent-distill-kfd1-import-v2-artifacts/parent-gfx1100-kfd1/report.json'
  evidence=C.read(numeric)
  assert evidence['complete'] and evidence['numerical']['all_passed'] and evidence['optimizer_updates']==0
  assert evidence['actual_device']['pci']=='0000:0c:00.0'
  tiny=C.BASE/'tinygrad'
  assert subprocess.check_output(['git','-C',str(tiny),'rev-parse','HEAD'],text=True).strip()=='89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  sources=[*HERE.glob('agent-distill-fit-*.py'),C.GRAPH,C.KFD,C.INPUTS,C.MANIFEST,report_path,numeric,
    C.WORK/'agent-distill-tinygrad-artifacts/oracle.json',C.BASE/'model/config.json',C.BASE/'model/model.safetensors.index.json']
  sources+=list((tiny/'tinygrad').rglob('*.py'))
  p=dict(schema=1,hyperparameters=C.HYPER,**C.BOUNDS,protected_routes=[C.PROTECTED],
    target='KFD:1+AMD:LLVM:gfx1100',selected_pci='0000:0c:00.0',selected_gpu_id=27211,
    training_manifest_sha256=C.MANIFEST_SHA,input_manifest_sha256=C.INPUTS_SHA,graph_sha256=C.GRAPH_SHA,
    source_sha256={str(p.resolve()):C.sha(p) for p in sources},master_weights=weights,block_config=native_config,
    teacher_contract=C.read(C.INPUTS)['protocol']['teacher_contract'],
    recorded_navigation=False,conditioning='All three predeclared experimental commands at one anchor from each evidenced drive.',
    teacher_frozen=True,sealed_targets_opened=False,
    fixed_language_calibration='Preexisting teacher scales used 32 original-route calibration windows. This pilot adds no fitting/stopping/calibration on that route.',
    quality_gate='Whole model and original 5% behavior gates remain unchanged; no route application or navigation-quality qualification.',
    admission_status='BLOCKED: full L2048 zero-update GPU qualification and AdamW resource/time evidence unavailable.',
    required_admission=dict(full_length='Successful parent-reviewed L2048 forward/backward, original frozen graph, selected KFD1, optimizer_updates=0.',
      optimizer='Full-parameter FP32 AdamW states allocated; dummy optimizer update on isolated synthetic buffers, zero actual-data optimizer updates; resource/time upper bounds.',
      availability='Fresh parent-reviewed complete KFD ownership snapshot, no selected owner, >=21GiB observed free, no reset/displacement.',
      bound='25*train_update_with_checkpoint_seconds_upper + 6*validation_example_seconds_upper + startup_seconds_upper <=870 seconds.'),
    accounting_correction='Official22 images minus selected10 images equals12 omitted surround images, from four camera views.',
    allowed_outputs='Private new 500GB artifact directory only; original BF16 masters, selected model, protected route, and public repositories unchanged.')
  # Check original BF16 payloads without running a student or optimizer.
  trainer=C.module('agent_distill_fit_train_prepare',HERE/'agent-distill-fit-train.py')
  loaded=trainer.seed_arrays(p);assert len(loaded)==8
  C.write(path,p)
  C.write(C.ROOT/'agent-distill-fit-data-check.json',dict(complete=True,actual_targets_checked=6,sealed_target_arrays_opened=0,
    student_forward_executed=False,optimizer_updates=0,gpu_executed=False,targets=summaries,
    original_bf16_seed_payloads_verified=8,protocol_sha256=C.sha(path)))
  template=dict(schema=1,pilot_protocol_sha256=C.sha(path),parent_reviewed=False,no_reset_or_owner_displacement=False,no_memory_planner=None,
    full_length=dict(path=None,sha256=None,fields=dict(length=None,all_passed=None,target=None,pci=None,graph_sha256=None,no_memory_planner=None)),
    optimizer_resource_evidence=dict(path=None,sha256=None),availability_review=dict(path=None,sha256=None))
  C.write(C.ROOT/'agent-distill-fit-admission-template.json',template)
  C.write(C.ROOT/'agent-distill-fit-commands.json',dict(executed=False,protocol_sha256=C.sha(path),
    preflight=[str(C.BASE/'upstream/.venv/bin/python'),str(HERE/'agent-distill-fit-train.py'),'--protocol-sha',C.sha(path)],
    gated_parent_pilot=[str(C.BASE/'upstream/.venv/bin/python'),str(HERE/'agent-distill-fit-train.py'),'--protocol-sha',C.sha(path),
      '--admission',str(C.ROOT/'agent-distill-fit-admission.json'),'--reviewed-admission-sha','<parent-reviewed receipt SHA256>','--execute-training']))
  C.no_devices();print(json.dumps(dict(protocol=str(path),protocol_sha256=C.sha(path),actual_targets_checked=6,sealed_targets_opened=0,
    gpu_executed=False,optimizer_updates=0,blocked=True),indent=2))

if __name__=='__main__':main()
