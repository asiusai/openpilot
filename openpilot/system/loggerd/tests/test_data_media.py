import base64
import hashlib
import json
import shutil
import subprocess

import pytest

from openpilot.system.loggerd.data_media import derive_cenc_key, package_cenc_mp4, read_boxes
from openpilot.system.loggerd.tests.media_fixture import make_video, decrypt_recording


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def frame_hash(path, key: bytes | None = None) -> list[str]:
  command = [FFMPEG, "-hide_banner", "-loglevel", "error"]
  if key is not None:
    path = decrypt_recording(path, key)
  command += ["-i", str(path), "-map", "0:v:0", "-f", "framehash", "-hash", "sha256", "-"]
  output = subprocess.check_output(command, text=True)
  return [line.rsplit(",", 1)[-1].strip() for line in output.splitlines() if line and not line.startswith("#")]


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="FFmpeg is required")
def test_cenc_fmp4_round_trip(tmp_path) -> None:
  clear = tmp_path / "clear.mp4"
  encrypted = tmp_path / "encrypted.mp4"
  frame = bytes([16]) * (64 * 64) + bytes([128]) * (64 * 64 // 2)
  subprocess.run(
    [
      FFMPEG,
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x64", "-r", "20", "-i", "pipe:0",
      "-c:v", "libx264", "-preset", "ultrafast", "-g", "20", "-bf", "0",
      "-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_sidx", str(clear),
    ],
    input=frame * 40,
    check=True,
  )
  key = bytes(range(32))
  kid = bytes.fromhex("ffeeddccbbaa99887766554433221100")
  manifest = package_cenc_mp4(clear, encrypted, key, kid)

  assert manifest["alg"] == "CENC-A256CTR"
  assert manifest["codec"].startswith("avc1.")
  assert manifest["kid"] == base64.urlsafe_b64encode(kid).decode().rstrip("=")
  assert manifest["encryptedLength"] == encrypted.stat().st_size
  assert manifest["fragments"][0]["kind"] == "init"
  assert all(fragment["kind"] == "media" for fragment in manifest["fragments"][1:])
  assert all(isinstance(fragment["timeMs"], int) for fragment in manifest["fragments"][1:])
  assert all(isinstance(fragment["durationMs"], int) for fragment in manifest["fragments"][1:])

  payload = encrypted.read_bytes()
  cursor = 0
  for fragment in manifest["fragments"]:
    assert fragment["offset"] == cursor
    chunk = payload[cursor:cursor + fragment["length"]]
    assert base64.urlsafe_b64encode(hashlib.sha256(chunk).digest()).decode().rstrip("=") == fragment["sha256"]
    cursor += fragment["length"]
  assert cursor == len(payload)

  probe = subprocess.check_output(
    [FFPROBE, "-v", "error", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "json", str(decrypt_recording(encrypted, key))],
    text=True,
  )
  assert json.loads(probe)["streams"][0]["nb_read_packets"] == "40"
  assert frame_hash(clear) == frame_hash(encrypted, key)


def test_cenc_key_is_scoped_to_owner_epoch_and_path() -> None:
  folder_key = bytes(range(32))
  first = derive_cenc_key(folder_key, "owner", "epoch", "routes/route--0/fcamera.mp4")
  assert first == derive_cenc_key(folder_key, "owner", "epoch", "routes/route--0/fcamera.mp4")
  assert first != derive_cenc_key(folder_key, "owner", "epoch", "routes/route--1/fcamera.mp4")
  assert first != derive_cenc_key(folder_key, "owner", "next", "routes/route--0/fcamera.mp4")


def test_cenc_upload_recovers_complete_frames_from_interrupted_recording(tmp_path):
  source = tmp_path / 'source.mp4'
  make_video(source, 3)
  original = source.read_bytes()
  boxes = read_boxes(original, 0, len(original))
  moof = [box for box in boxes if box.type == b'moof'][-1]
  mdat = [box for box in boxes if box.type == b'mdat'][-1]
  reference = tmp_path / 'reference.mp4'
  reference.write_bytes(original[:moof.start])
  truncated = original[:mdat.end - 1]
  source.write_bytes(truncated)
  encrypted = tmp_path / 'encrypted.mp4'
  key = bytes(range(32))
  manifest = package_cenc_mp4(source, encrypted, key, bytes(reversed(key[:16])))
  assert manifest['plaintextLength'] == len(truncated)
  assert manifest['discardedBytes'] == len(truncated) - moof.start
  assert len(frame_hash(encrypted, key)) == 40
  assert frame_hash(encrypted, key) == frame_hash(reference)
  assert source.read_bytes() == truncated


def test_embedded_index_keeps_video_seekable_and_hashes_each_range(tmp_path):
  from openpilot.system.asius.pqc import RootIdentity
  from openpilot.system.loggerd.data_api import public_identity, canonical_json
  from openpilot.system.loggerd.data_media import embed_index, INDEX_BYTES
  clear, encrypted = tmp_path/'clear.mp4', tmp_path/'encrypted.mp4'
  make_video(clear, 3)
  key = bytes(range(32))
  manifest = package_cenc_mp4(clear, encrypted, key, bytes(reversed(key[:16])))
  private = RootIdentity.generate()
  signed = embed_index(str(encrypted), manifest, public_identity(private), 'routes/00000001--1234567890/0/qcamera.mp4','key',private)
  data = encrypted.read_bytes()
  assert len(data) == signed['encryptedLength']
  assert signed['fragments'][0]['offset'] == INDEX_BYTES
  private.public_key().verify(base64.urlsafe_b64decode(signed['signature']+'=='), canonical_json({k:v for k,v in signed.items() if k!='signature'}).encode())
  for fragment in signed['fragments']:
    chunk=data[fragment['offset']:fragment['offset']+fragment['length']]
    assert base64.urlsafe_b64encode(hashlib.sha256(chunk).digest()).decode().rstrip('=') == fragment['sha256']
  assert frame_hash(clear)==frame_hash(encrypted,key)
