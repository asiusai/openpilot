"""Actual-report comparison mutations, snapshot corruption, and CPU loader admission."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compare_check_common',HERE/'agent-expert-live-compare-common.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V);N=V.N


def admission():
  check=N.load(V.S.OLD/'agent-expert-live-check.py','compare_frozen_actual_loader_check');check.N=N
  result=check.admission()
  assert result['factory']['actual_factory_load_executed'] and result['actual_runpy_import_and_pickle']
  assert result['factory']['real_device_open_attempts']==[] and result['actual_storage_bindings']==N.storage()
  return result


def controls(directory):
  assert not directory.exists();directory.mkdir()
  recovery=V.recover_current();report=V.read(V.CURRENT)
  validator=V.load(V.FROZEN/'agent-expert-live-storage-common.py','compare_real_mutation_validator')
  base=V.read(V.BASE);rejected=[]
  def qualify(value):
    value=copy.deepcopy(value);value['agent_expert_live_storage']['passed']=True
    return V.projected_qualify(validator,validator.qualify,value,base,V.BASE_SHA,'current',V.CURRENT)
  def reject(label,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,FileNotFoundError):rejected.append(label)
    else:raise AssertionError('Invalid comparison accepted: '+label)
  qualify(report)
  for name in ('model','config','input','owner','source_sha','prediction','unknown_factory_field','unknown_provenance_field',
               'nonfinite_timing','negative_timing','boolean_timing','missing_timing','wrong_snapshot_path','non_string_snapshot',
               'kv_prefix','sixth_target','constructor_cleanup','pv_binary','storage_path'):
    value=copy.deepcopy(report);factory=value['consumer_provenance']['initializer']['factory']
    if name=='model':value['consumer_provenance']['callables'][0]['source_sha256']='0'*64
    elif name=='config':value['initialized']['config_sha256']='0'*64
    elif name=='input':value['rows'][0]['cpu_buffers'][0]['sha256']='0'*64
    elif name=='owner':value['rows'][0]['consumer_protocol']['owned_input_objects']=2
    elif name=='source_sha':value['source_sha256']['model.py']='0'*64
    elif name=='prediction':value['rows'][0]['prediction']['generated_token_ids'][0]+=1
    elif name=='unknown_factory_field':factory['unreviewed_field']=1
    elif name=='unknown_provenance_field':value['consumer_provenance']['unreviewed_field']=1
    elif name=='nonfinite_timing':factory['model_load_ms']=float('nan')
    elif name=='negative_timing':factory['model_load_ms']=-1
    elif name=='boolean_timing':factory['model_load_ms']=True
    elif name=='missing_timing':del factory['model_load_ms']
    elif name=='wrong_snapshot_path':factory['source_snapshot']=V.read(N.CONSUMER)['consumer_provenance']['initializer']['factory']['source_snapshot']
    elif name=='non_string_snapshot':factory['source_snapshot']={'claimed_identity':value['source_sha256']}
    elif name=='kv_prefix':value['rows'][0]['defined_kv_prefix_bits'][27]['sha256']='0'*64
    elif name=='sixth_target':value['rows'][0]['speculation']['targets'][5]+=1
    elif name=='constructor_cleanup':value['agent_expert_live_storage']['constructor_restored']=False
    elif name=='pv_binary':value['agent_expert_live_storage']['pv']['captures'][0]['pv_full_description']['binary_sha256']='0'*64
    else:value['agent_expert_live_storage']['storage_events'][1]['actual_environment']['ALPAMAYO_FP8_CACHE']='/tmp/foreign'
    reject(name,lambda:qualify(value))
  for elapsed in (0,1.5,1e8):
    value=copy.deepcopy(report);value['consumer_provenance']['initializer']['factory']['model_load_ms']=elapsed;qualify(value)
  source=Path(report['consumer_provenance']['initializer']['factory']['source_snapshot'])
  for name in ('missing_file','tampered_file','extra_file','renamed_file','symlink_file','extra_directory'):
    output=directory/name/'report.json';folder=output.parent/'report.benchmark-sources'
    shutil.copytree(source,folder)
    value=copy.deepcopy(report);value['consumer_provenance']['initializer']['factory']['source_snapshot']=str(folder)
    V.project(value,output)
    file=folder/'model.py'
    if name=='missing_file':file.unlink()
    elif name=='tampered_file':file.write_bytes(file.read_bytes()+b'\n# CPU negative\n')
    elif name=='extra_file':(folder/'foreign.py').write_text('CPU negative\n')
    elif name=='renamed_file':file.rename(folder/'different.py')
    elif name=='symlink_file':file.unlink();file.symlink_to(source/'model.py')
    else:(folder/'extra').mkdir()
    reject(name,lambda:V.project(value,output))
  # Positive control: unchanged bytes at the declared new per-run snapshot location.
  output=directory/'valid-relocation/report.json';folder=output.parent/'report.benchmark-sources'
  shutil.copytree(source,folder);value=copy.deepcopy(report)
  value['consumer_provenance']['initializer']['factory']['source_snapshot']=str(folder)
  assert V.project(value,output)==V.project(report,V.CURRENT)
  overlay=V.runtime_module()[2];assert json.loads(json.dumps(overlay))==overlay
  return dict(complete=True,gpu_executed=False,current_admission=recovery,rejected=rejected,
    actual_report_mutations=True,actual_snapshot_files=True,valid_timing_changes_passed=True,
    valid_byte_identical_snapshot_relocation_passed=True,runtime_overlay=overlay,json_overlay_roundtrip_exact=True)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--admit',action='store_true')
  args=p.parse_args();assert not args.output.exists();V.configure()
  result=admission() if args.admit else controls(V.ARTIFACTS/('controls-'+args.output.stem))
  if not args.admit:result['factory_preflight']=admission()
  result['source_sha256']={str(p):V.sha(p) for p in HERE.glob('agent-expert-live-compare-*.py')}
  V.write(args.output,result,new=True);print(dict(complete=True,gpu_executed=False,negative_controls=len(result.get('rejected',[]))))


if __name__=='__main__':main()
