"""Import-only repair: reuse immutable KFD1 identity and numerical code; use a new artifact root."""
import hashlib
import importlib.util
from pathlib import Path
import sys

PRIOR=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-kfd1-artifacts')
PRIOR_PROTOCOL_SHA='32340abb876ffb037eaec1f13b7f29f593355152b32e1cfe03776e92bce785b9'
PRIOR_COMMON_SHA='392613932109d8a6dc0a65ff6230f7592ad572882dd6464bc9eb960c75e53a2e'
path=PRIOR/'frozen/agent-distill-kfd1-common.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()==PRIOR_COMMON_SHA
name='agent-distill-kfd1-import-prior-common'
spec=importlib.util.spec_from_file_location(name,path);prior=importlib.util.module_from_spec(spec)
sys.modules[name]=prior;spec.loader.exec_module(prior)
for key in ('EXP','WORK','OLD','NUMERIC','TARGET','SELECTED_INDEX','SELECTED_PCI','SELECTED_NODE','SELECTED_GPU_ID',
  'SELECTED_RENDER_MINOR','LIMIT','OLD_PROTOCOL_SHA','NUMERIC_COMMON_SHA','NUMERIC_CHECK_SHA','import_numeric','B',
  'file_sha','read','write','no_gpu_fds','passive_topology','TINY','PY','TOLERANCES','selected_node',
  'validate_opened_identity','passive_availability','require_no_observed_conflict'):
  globals()[key]=getattr(prior,key)
ROOT=WORK/'agent-distill-kfd1-import-artifacts'
