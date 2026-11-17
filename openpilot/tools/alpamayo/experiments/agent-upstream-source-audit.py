"""Read-only source provenance for the November 16 upstream tinygrad audit."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import subprocess

PIN = '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
UPSTREAM = '5db89dca1e08012bcc6da699fe86898e24ee8027'
PATHS = (
  'tinygrad/renderer/tc.py', 'tinygrad/renderer/llvmir.py', 'tinygrad/runtime/autogen/llvm.py',
  'tinygrad/runtime/autogen/amd/rdna4/ins.py', 'tinygrad/runtime/support/compiler_llvm.py',
  'tinygrad/runtime/support/compiler_amd.py', 'tinygrad/codegen/opt/search.py',
  'tinygrad/codegen/opt/heuristic.py', 'tinygrad/codegen/opt/postrange.py', 'tinygrad/codegen/late/linearizer.py',
  'tinygrad/codegen/late/regalloc.py', 'tinygrad/codegen/simplify.py', 'tinygrad/uop/ops.py',
  'tinygrad/runtime/support/hcq2.py', 'tinygrad/runtime/support/usb.py', 'tinygrad/runtime/ops_amd.py',
  'tinygrad/runtime/support/am/ip.py', 'tinygrad/runtime/support/compiler_cpu.py', 'tinygrad/helpers.py',
  'tinygrad/llm/kernels/amd.py', 'tinygrad/llm/kernels/amd_gguf.py',
  'tinygrad/renderer/amd/elf.py', 'tinygrad/renderer/amd/sqtt.py', 'tinygrad/viz/cli.py',
  'extra/mmapeak/mmapeak.py', 'extra/optimization/test_beam_search.py', 'extra/optimization/extract_dataset.py',
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--repository', type=Path, default=Path('/tmp/opencode/agent-upstream-tinygrad-20261116.git'))
  parser.add_argument('--output', type=Path, default=Path(__file__).with_name('agent-upstream-source-report.json'))
  args = parser.parse_args()
  def git(*argv): return subprocess.check_output(['git', '--git-dir', str(args.repository), *argv])
  def text(*argv): return git(*argv).decode().strip()
  assert text('rev-parse', 'HEAD') == UPSTREAM
  base = text('merge-base', PIN, UPSTREAM)
  records = []
  for path in PATHS:
    revisions = {}
    for name, rev in (('pinned', PIN), ('upstream', UPSTREAM)):
      exists = subprocess.run(['git', '--git-dir', str(args.repository), 'cat-file', '-e', f'{rev}:{path}'],
                              capture_output=True).returncode == 0
      if exists:
        data = git('show', f'{rev}:{path}')
        revisions[name] = {'git_blob': text('rev-parse', f'{rev}:{path}'), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
      else: revisions[name] = None
    records.append({'path': path, **revisions, 'identical': revisions['pinned'] == revisions['upstream'],
                    'diffstat': text('diff', '--numstat', PIN, UPSTREAM, '--', path)})
  from tinygrad import Device
  from tinygrad.runtime.autogen import llvm
  version = [ctypes.c_uint32() for _ in range(3)]
  llvm.LLVMGetVersion(*[ctypes.byref(x) for x in version])
  assert not Device._opened_devices, Device._opened_devices
  report = {'observed_date': '2026-11-16', 'pinned': PIN, 'upstream': UPSTREAM, 'merge_base': base,
            'pinned_only_commits': text('log', '--format=%H %s', f'{base}..{PIN}').splitlines(),
            'upstream_commits_since_merge_base': int(text('rev-list', '--count', f'{base}..{UPSTREAM}')),
            'upstream_metadata': text('show', '-s', '--format=%H%n%cI%n%s', UPSTREAM).splitlines(),
            'llvm_version': [x.value for x in version], 'libtcc': ctypes.util.find_library('tcc'),
            'opened_devices': list(Device._opened_devices), 'files': records,
            'search_and_peak_tools_changed': text('diff', '--name-only', PIN, UPSTREAM, '--', 'extra/optimization', 'extra/mmapeak').splitlines(),
            'commits': text('log', '--format=%H %cI %s', f'{PIN}..{UPSTREAM}', '--',
                           'tinygrad/codegen', 'tinygrad/renderer', 'tinygrad/llm/kernels/amd_gguf.py',
                           'tinygrad/runtime/support/usb.py', 'tinygrad/runtime/support/compiler_cpu.py',
                           'tinygrad/runtime/support/hcq2.py', 'tinygrad/runtime/support/am/ip.py').splitlines()}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'output': str(args.output), 'files': len(records), 'unchanged_files': sum(r['identical'] for r in records),
                    'upstream': UPSTREAM, 'llvm_version': report['llvm_version'], 'libtcc': report['libtcc']}))


if __name__ == '__main__': main()
