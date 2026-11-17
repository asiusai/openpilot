"""Generate a separate CPU checker correcting only its cached-getenv negative control."""
import ast
import hashlib
from pathlib import Path

ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-kfd1-import-v2-artifacts')

def main():
  source=ROOT/'frozen/agent-distill-kfd1-import-v2-check.py'
  target=ROOT/'frozen/agent-distill-kfd1-import-v2-cpu-check.py'
  assert not target.exists()
  text=source.read_text();assert text.count('helpers.getenv(key,False)')==2
  # The pinned helper is @functools.cache. Its original wrapped implementation
  # tests parsing without reading the previous environment result from that cache.
  corrected=text.replace('helpers.getenv(key,False)','helpers.getenv.__wrapped__(key,False)')
  ast.parse(corrected,filename=str(target));target.write_text(corrected);target.chmod(0o444)
  print(hashlib.sha256(target.read_bytes()).hexdigest())

if __name__=='__main__':main()
