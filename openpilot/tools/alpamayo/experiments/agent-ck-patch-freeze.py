"""Freeze the narrow transport repair and retain every original protocol gate."""
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import sys

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-patch-work'
TARGET=ROOT/'prepared-final'
OLD=WORK/'agent-ck-source-work/prepared-final/protocol.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  assert not (TARGET/'protocol.json').exists()
  sys.path.insert(0,str(EXP));runner=importlib.import_module('agent-ck-patch-run')
  old=json.loads(OLD.read_text());catalog=json.loads((TARGET/'catalog.json').read_text());proof=json.loads((TARGET/'cpu-proof.json').read_text())
  assert proof['strict_runtime_guard_exercised'] and proof['device_copy_count']==120 and not proof['gpu_executed']
  source=runner.replacement();(TARGET/'qualified-transport-run.py').write_text(source)
  (TARGET/'qualified-transport.diff').write_text(''.join(difflib.unified_diff(runner.OLD_RUN.read_text().splitlines(True),source.splitlines(True),
    fromfile=str(runner.OLD_RUN),tofile=str(TARGET/'qualified-transport-run.py'))))
  failures={WORK/'agent-ck-source-parent/report.json':'31f309581d0aa29b5c8ccc1e323f53e0593468df86039a66b260b4bcdefee25b',
    WORK/'agent-ck-source-parent/run.log':'d424281138dc36dd37af02705d095fbb3a3c83399788ebb605f31ede1583c490',
    WORK/'agent-ck-source-work/parent-result.json':'ad7188d154bcc793a6f615f8db3ddebba2200fc39902854d217b957a0af83db2'}
  assert all(sha(p)==h for p,h in failures.items())
  prior=json.loads((WORK/'agent-ck-source-parent/report.json').read_text())
  assert len(prior['full_actual_shape_checks'])==7 and prior['causal_gqa_mutations_passed']
  data={k:v for k,v in prior.items() if k in ('failure','elapsed_seconds','io','phases','full_actual_shape_checks','causal_gqa_mutations_passed')}
  data.update(classification='Harness host-mapping assertion after all seven full numerical checks; no numerical failure shown.',
    repair='Mapped writes use Buffer.get_storage().host. Device-only returned dwords use physical Buffer.view and staged Buffer.copy_from, then synchronize.',
    reference_policy='Run all seven fixtures again. Prior results are retained evidence, never used as fresh replay references.',
    copy_transport='One four-byte AMD SDMA transfer template; eight exact USB-aware CPU submit-program source/AST/ELF/ABI associations.',
    deadline_estimate={'previous_elapsed_seconds':prior['elapsed_seconds'],'previous_payload_bytes':sum(prior['io'][k] for k in ('host_write_bytes','host_read_bytes')),
      'remaining_conservative_payload_bytes':old['budget']['conservative_total_payload_bytes']-prior['io']['host_write_bytes']-prior['io']['host_read_bytes'],
      'assumed_remaining_payload_bytes_per_second':1350000,'additional_host_dispatch_allowance_seconds':15,
      'estimated_total_seconds':prior['elapsed_seconds']+(old['budget']['conservative_total_payload_bytes']-prior['io']['host_write_bytes']-prior['io']['host_read_bytes'])/1350000+15,
      'wrapper_seconds':300,'internal_seconds':290,'estimate_only':True})
  (TARGET/'diagnosis.json').write_text(json.dumps(data,indent=2)+'\n')
  extra=[EXP/(name+'.py') for name in ('agent-ck-patch-transport','agent-ck-patch-identity','agent-ck-patch-guard','agent-ck-patch-cpu',
    'agent-ck-patch-run','agent-ck-patch-freeze')]
  # The new transport consumes these additional pinned implementation and register-definition sources.
  tiny=Path(old['tinygrad'])/'tinygrad'
  extra += [tiny/'runtime/support/amd.py',tiny/'runtime/ops_cpu.py',tiny/'renderer/cstyle.py',tiny/'runtime/support/compiler_cpu.py',
    tiny/'runtime/support/compiler_llvm.py',tiny/'runtime/autogen/am/regs.py',tiny/'runtime/autogen/am/__init__.py']
  extra += [p for p in (tiny/'runtime/autogen/am').glob('*') if p.is_file() and ('sdma_6_0_0' in p.name or 'nbif_6_3_1' in p.name or p.name=='navi_offsets.py')]
  extra += [tiny.parent/'LICENSE',OLD,*failures,*[p for p in TARGET.rglob('*') if p.is_file()]]
  assert all(p.is_file() for p in extra)
  plan={**old,**catalog,'patch_output_root':str(ROOT)}
  plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p) for p in extra}}
  plan['patch_repair']={'prior_report':str(WORK/'agent-ck-source-parent/report.json'),'prior_report_sha256':failures[WORK/'agent-ck-source-parent/report.json'],
    'prior_protocol':str(OLD),'prior_protocol_sha256':sha(OLD),'original_numerical_run_sha256':runner.OLD_RUN_SHA,
    'all_seven_fixtures_rerun':True,'changed_nested_function':'write','added_preflight':'compile and authorize copy host programs before fixtures',
    'added_transport_accounting_gate':'120 successful dword copies, 60 exact destinations mutated and restored, 480 staging payload bytes',
    'gpu_binaries_unchanged':True,'original_limits_unchanged':True,'deadline_estimate':data['deadline_estimate']}
  (TARGET/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'protocol':str(TARGET/'protocol.json'),'sha256':sha(TARGET/'protocol.json'),'files_bound':len(plan['source_sha256']),
    'deadline_estimate':data['deadline_estimate']},indent=2))


if __name__=='__main__': main()
