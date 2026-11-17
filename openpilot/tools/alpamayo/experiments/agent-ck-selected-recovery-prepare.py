"""Prepare private caches, or freeze a small direct protocol after CPU checks."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sqlite3

s=importlib.util.spec_from_file_location('ck_selected_recovery',Path(__file__).with_name('agent-ck-selected-recovery-common.py'))
R=importlib.util.module_from_spec(s);s.loader.exec_module(R)

def stage():
  root=R.ROOT;src=R.S.ROOT/'artifacts';dst=root/'artifacts';assert not dst.exists()
  for rel in ('tmp','cache/huggingface','cache/torch','hooks/agent-exact-live-artifacts'):(dst/rel).mkdir(parents=True,exist_ok=True)
  copies={}
  for rel in ('hooks/agent-exact-combined-install.py','hooks/agent-exact-live-artifacts/12460b8ec1eae4609835e65cab03ec855bff660cc872123cf1494f652cee8dd7.so'):
    before=R.S.sha(src/rel);shutil.copyfile(src/rel,dst/rel);assert R.S.sha(src/rel)==R.S.sha(dst/rel)==before;copies[str(src/rel)]=before
  frozen=root/'cache-source';frozen.mkdir()
  for suffix in ('','-wal','-shm'):
    path=src/('compiler-cache.db'+suffix)
    if path.exists():
      before=R.S.sha(path);shutil.copyfile(path,frozen/path.name);assert R.S.sha(path)==R.S.sha(frozen/path.name)==before;copies[str(path)]=before
  with sqlite3.connect(str(frozen/'compiler-cache.db')) as source,sqlite3.connect(str(dst/'compiler-cache.db')) as target:source.backup(target)
  R.S.write(root/'cache-copy.json',{'source_bytes_unchanged':copies,'old_sqlite_never_opened':True,'new_cache':str(dst/'compiler-cache.db')})

def freeze():
  target=R.ROOT/'protocol.json';assert not target.exists()
  files=[*sorted(R.EXP.glob('agent-ck-selected-recovery-*.py')),R.ROOT/'receipt-v1.json',R.ROOT/'cpu-check.json',R.ROOT/'cache-copy.json',
    R.ROOT/'quality-order.diff']
  R.S.write(target,dict(schema=1,original_protocol_sha256=R.ORIGINAL_PROTOCOL,
    failed_report_sha256=R.FAILED_SHA,receipt_sha256=R.S.sha(R.ROOT/'receipt-v1.json'),
    files={str(p):R.S.sha(p) for p in files},cpu_check=str(R.ROOT/'cpu-check.json'),timeout_seconds=600,
    automatic_launch=False,default_adoption=False,numerical_code_changed=False))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--mode',choices=('stage','freeze'),required=True);a=p.parse_args()
  {'stage':stage,'freeze':freeze}[a.mode]()
