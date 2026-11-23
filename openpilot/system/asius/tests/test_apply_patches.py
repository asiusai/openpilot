import subprocess
from pathlib import Path

import pytest

from openpilot.system.asius.apply_patches import apply_tinygrad_patches, patched_msgq_source


@pytest.fixture(params=["source_archive", "release", "submodule"])
def source_tree(tmp_path, request):
  root = Path(__file__).resolve().parents[4]
  relative = "msgq/visionipc/visionbuf.cc"
  source = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root / "msgq_repo")
  target = tmp_path / "msgq_repo" / relative
  target.parent.mkdir(parents=True)
  target.write_bytes(source)
  if request.param != "source_archive":
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
  if request.param == "submodule":
    subprocess.run(["git", "init", "-q", str(tmp_path / "msgq_repo")], check=True)
  return tmp_path, target


def test_apply_and_repeat(source_tree):
  root, target = source_tree
  original = target.read_bytes()
  output = patched_msgq_source(root)
  patched = output.read_bytes()
  assert target.read_bytes() == original
  assert patched != original
  assert b"DMA_HEAP_IOCTL_ALLOC" in patched
  modified = output.stat().st_mtime_ns
  patched_msgq_source(root)
  assert target.read_bytes() == original
  assert output.read_bytes() == patched
  assert output.stat().st_mtime_ns == modified


def test_preserves_unrelated_edits(source_tree):
  root, target = source_tree
  target.write_bytes(target.read_bytes() + b"\n// Local experiment\n")
  original = target.read_bytes()
  output = patched_msgq_source(root)
  assert target.read_bytes() == original
  assert output.read_bytes().endswith(b"// Local experiment\n")


def test_conflict_stops_without_modifying_source(source_tree):
  root, target = source_tree
  changed = target.read_bytes().replace(b"this->addr = malloc_with_fd", b"this->addr = different_allocator")
  target.write_bytes(changed)
  with pytest.raises(RuntimeError, match="reconcile the patch with msgq_repo"):
    patched_msgq_source(root)
  assert target.read_bytes() == changed


def test_missing_source_stops(tmp_path):
  with pytest.raises(RuntimeError, match="reconcile the patch with msgq_repo"):
    patched_msgq_source(tmp_path)


@pytest.fixture(params=["source_archive", "release", "submodule"])
def tinygrad_tree(tmp_path, request):
  root = Path(__file__).resolve().parents[4]
  files = ['tinygrad/runtime/ops_qcom.py', 'tinygrad/runtime/support/compiler_mesa.py', 'tinygrad/codegen/opt/heuristic.py']
  for relative in files:
    source = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=root / 'tinygrad_repo')
    target = tmp_path / 'tinygrad_repo' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
  if request.param != 'source_archive':
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
  if request.param == 'submodule':
    subprocess.run(['git', 'init', '-q', str(tmp_path / 'tinygrad_repo')], check=True)
  return tmp_path


def test_tinygrad_patch_survives_source_and_release_packaging(tinygrad_tree):
  source = tinygrad_tree / 'tinygrad_repo/tinygrad/runtime/ops_qcom.py'
  original = source.read_bytes()
  apply_tinygrad_patches(tinygrad_tree)
  patched = source.read_bytes()
  assert patched != original
  assert b'DRM_IOCTL_PRIME_FD_TO_HANDLE' in patched
  modified = source.stat().st_mtime_ns
  apply_tinygrad_patches(tinygrad_tree)
  assert source.read_bytes() == patched
  assert source.stat().st_mtime_ns == modified


def test_tinygrad_patch_conflict_preserves_all_sources(tinygrad_tree):
  source = tinygrad_tree / 'tinygrad_repo/tinygrad/runtime/ops_qcom.py'
  source.write_bytes(source.read_bytes().replace(b'class MSMIface:', b'class LocalMSMIface:'))
  before = {path: path.read_bytes() for path in tinygrad_tree.rglob('*.py')}
  with pytest.raises(RuntimeError, match='reconcile the patch with tinygrad_repo'):
    apply_tinygrad_patches(tinygrad_tree)
  assert before == {path: path.read_bytes() for path in before}
