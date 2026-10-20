import tempfile
import unittest
from pathlib import Path

from package_prebuilt import BINARIES, MODEL_NAMES, validate


class TestPrebuiltValidation(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    for name in MODEL_NAMES:
      path = self.root / "openpilot/selfdrive/modeld/models" / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(b"\0" * 1024)
    for name in BINARIES:
      path = self.root / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (183).to_bytes(2, "little"))

  def test_requires_chestnut_warps(self):
    (self.root / "openpilot/selfdrive/modeld/models/big_driving_warp_1344x760_tinygrad.pkl").unlink()
    with self.assertRaisesRegex(ValueError, "Missing compiled model"):
      validate(self.root)

  def test_rejects_lfs_pointer(self):
    (self.root / "openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl").write_text(
      "version https://git-lfs.github.com/spec/v1\n")
    with self.assertRaisesRegex(ValueError, "unresolved LFS pointer"):
      validate(self.root)

  def test_rejects_host_binary(self):
    path = self.root / BINARIES[0]
    path.write_bytes(path.read_bytes()[:18] + (62).to_bytes(2, "little"))
    with self.assertRaisesRegex(ValueError, "AArch64"):
      validate(self.root)

  def test_requires_flattened_submodules(self):
    (self.root / ".gitmodules").touch()
    with self.assertRaisesRegex(ValueError, "submodule"):
      validate(self.root)

  def test_accepts_complete_arm64_build(self):
    validate(self.root)
