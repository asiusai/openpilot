#!/usr/bin/env python3
"""Package the native Asius build without publishing a release branch."""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

MODEL_NAMES = (
  "driving_tinygrad.pkl", "dmonitoring_model_tinygrad.pkl",
  "driving_warp_1344x760_tinygrad.pkl", "driving_warp_1928x1208_tinygrad.pkl",
  "dm_warp_1344x760_tinygrad.pkl", "dm_warp_1928x1208_tinygrad.pkl",
  "big_driving_tinygrad.pkl", "big_driving_warp_1344x760_tinygrad.pkl", "big_driving_warp_1928x1208_tinygrad.pkl",
)
BINARIES = (
  "openpilot/system/camerad/camerad", "openpilot/system/loggerd/loggerd",
  "openpilot/system/loggerd/encoderd", "openpilot/selfdrive/pandad/pandad",
  "openpilot/common/libparams_c.so", "msgq_repo/msgq/visionipc/visionipc_pyx.so",
)


def git(root: Path, *args: str) -> str:
  return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha256(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(root: Path) -> None:
  if (root / ".gitmodules").exists():
    raise ValueError("Release must contain the submodule files, without Git submodules")
  for name in MODEL_NAMES:
    path = root / "openpilot/selfdrive/modeld/models" / name
    if not path.is_file() or path.stat().st_size < 1024:
      raise ValueError(f"Missing compiled model or unresolved LFS pointer: {name}")
  for name in BINARIES:
    with (root / name).open("rb") as stream:
      header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
      raise ValueError(f"Expected an AArch64 ELF binary: {name}")


def package(source: Path, root: Path, output: Path) -> dict:
  if root == source or not (root / ".git").is_dir() or not (root / "prebuilt").is_file():
    raise ValueError("Package a separate prebuilt checkout with its own Git directory")
  validate(root)
  commit = git(source, "rev-parse", "HEAD")
  model_dir = "openpilot/selfdrive/modeld/models"
  for name in MODEL_NAMES:
    if name.startswith("big_") and sha256(source / model_dir / name) != sha256(root / model_dir / name):
      raise ValueError(f"Chestnut model differs from the source checkout: {name}")

  version = re.search(r'COMMA_VERSION\s+"([^"]+)"', (root / "openpilot/common/version.h").read_text())[1]
  metadata = {
    "channel": "master",
    "openpilot": {
      "version": version, "release_notes": "", "git_commit": commit,
      "git_origin": "https://github.com/asiusai/openpilot.git",
      "git_commit_date": git(source, "show", "-s", "--format=%ci", "HEAD"),
      "build_style": "prebuilt",
    },
  }
  (root / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
  git(root, "remote", "add", "origin", metadata["openpilot"]["git_origin"])
  git(root, "add", "--force", ".")
  # Do not race automatic repacking against the explicit cleanup below.
  git(root, "-c", "maintenance.auto=false", "-c", "gc.auto=0",
      "-c", "user.name=Asius CI", "-c", "user.email=ci@asius.ai",
      "commit", "--amend", "--no-gpg-sign", "-m", f"Asius prebuilt from {commit}")
  git(root, "reflog", "expire", "--expire=now", "--all")
  git(root, "gc", "--prune=now")

  # Keep LFS functional, but store each large model only once in the archive.
  for row in git(root, "lfs", "ls-files", "-l").splitlines():
    oid, _, name = row.split(" ", 2)
    model = root / name
    obj = root / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    if sha256(model) != oid or sha256(obj) != oid:
      raise ValueError(f"Invalid LFS model: {name}")
    obj.unlink()
    os.link(model, obj)
  git(root, "lfs", "fsck")
  if git(root, "status", "--porcelain") or git(root, "submodule", "status"):
    raise ValueError("Prebuilt checkout is dirty or contains submodules")

  archive = output / "openpilot-v0-chestnut.tar.xz"
  with archive.open("wb") as destination:
    tar = subprocess.Popen(["tar", "--sort=name", "-cf", "-", "-C", str(root), "."], stdout=subprocess.PIPE)
    assert tar.stdout is not None
    compressed = subprocess.run(["xz", "-1", "-T2", "--block-size=16MiB"], stdin=tar.stdout, stdout=destination)
    tar.stdout.close()
    if tar.wait() or compressed.returncode:
      raise RuntimeError("Could not create the prebuilt archive")

  report = {
    "source_commit": commit,
    "artifact_commit": git(root, "rev-parse", "HEAD"),
    "target": "asius-v0",
    "chestnut": True,
    "validation": "Native build and model compiler checks; full device and Chestnut tests still required",
    "vamos_build": Path("/BUILD").read_text().strip(),
    "python": platform.python_version(),
    "uv_lock_sha256": sha256(source / "uv.lock"),
    "installed_bytes": int(subprocess.check_output(["du", "-s", "-B1", str(root)], text=True).split()[0]),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": sha256(archive),
    "models": {name: {"bytes": (root / model_dir / name).stat().st_size,
                      "sha256": sha256(root / model_dir / name)} for name in MODEL_NAMES},
  }
  (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
  (output / "SHA256SUMS").write_text(f"{report['archive_sha256']}  {archive.name}\n")
  if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(summary, "a") as stream:
      stream.write("\n".join([
        f"Source: `{commit}`", "",
        f"- Installed: {report['installed_bytes'] / 2**20:.1f} MiB",
        f"- Download: {report['archive_bytes'] / 2**20:.1f} MiB",
        "- Chestnut model and both camera transforms included.",
        f"- {report['validation']}.", "",
      ]))
  return report


if __name__ == "__main__":
  print(json.dumps(package(*(Path(arg).resolve() for arg in sys.argv[1:])), indent=2))
