import shutil
import subprocess
import tempfile
from pathlib import Path


def apply_tinygrad_patches(root: Path) -> None:
  """Apply the pinned Dragon runtime patch, rejecting unexpected source changes."""
  source = root / 'tinygrad_repo'
  patch = Path(__file__).resolve().parent / 'patches/tinygrad-msm.patch'
  command = ['git', 'apply', '--unsafe-paths', f'--directory={source.resolve()}', '--whitespace=error-all']
  # Releases contain the already patched sources, without a nested Git repository.
  for reverse in (True, False):
    check = subprocess.run([*command, '--check', *(['--reverse'] if reverse else []), str(patch)],
                           cwd=root, capture_output=True, text=True)
    if check.returncode == 0:
      if not reverse:
        subprocess.run([*command, str(patch)], cwd=root, check=True)
      return
  raise RuntimeError(f'Cannot apply {patch.name}; reconcile the patch with tinygrad_repo before building.\n{check.stderr}')


def patched_msgq_source(root: Path) -> Path:
  """Prepare a build-only source copy, leaving the upstream checkout untouched."""
  source = root / 'msgq_repo/msgq/visionipc/visionbuf.cc'
  patch = Path(__file__).resolve().parent / 'patches/msgq-dma-heap.patch'
  target = root / '.cache/msgq/visionbuf.cc'
  target.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=target.parent) as directory:
    staged = Path(directory) / source.name
    if not source.is_file():
      raise RuntimeError('Missing visionbuf.cc; reconcile the patch with msgq_repo before building.')
    shutil.copyfile(source, staged)
    result = subprocess.run(['git', 'apply', '--unsafe-paths', '-p4', f'--directory={directory}', str(patch)],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
      raise RuntimeError(f'Cannot apply {patch.name}; reconcile the patch with msgq_repo before building.\n{result.stderr}')
    content = staged.read_bytes()
    if not target.exists() or target.read_bytes() != content:
      staged.replace(target)
  return target


if __name__ == '__main__':
  root = Path(__file__).resolve().parents[3]
  apply_tinygrad_patches(root)
  patched_msgq_source(root)
