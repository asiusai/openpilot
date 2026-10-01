import subprocess
from pathlib import Path

import pytest

from openpilot.system.asius.apply_patches import patched_msgq_source


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
