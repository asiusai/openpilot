import subprocess
from pathlib import Path


def apply_msgq_patch(root: Path) -> None:
  """Apply our buffer change to upstream msgq, including flattened releases."""
  patch = Path(__file__).resolve().parent / "patches/msgq-dma-heap.patch"
  command = ["git", "apply", str(patch)]
  # Check in reverse first so repeated builds preserve the source's mtime.
  if subprocess.run(command + ["--reverse", "--check"], cwd=root, capture_output=True).returncode == 0:
    return
  check = subprocess.run(command + ["--check"], cwd=root, capture_output=True, text=True)
  if check.returncode:
    raise RuntimeError(f"Cannot apply {patch.name}; reconcile the patch with msgq_repo before building.\n{check.stderr}")
  subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
  apply_msgq_patch(Path(__file__).resolve().parents[3])
