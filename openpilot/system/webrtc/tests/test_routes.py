import asyncio
import json
import subprocess
import sys
import threading

import pytest

from openpilot.system.webrtc.routes import MAX_RANGE_BYTES, RecordingUnavailableError, RouteFiles, RouteSession, mp4_manifest
from openpilot.system.loggerd.data_media import make_box, read_boxes
from openpilot.system.loggerd.tests.media_fixture import make_video


def test_ranges_reject_traversal_symlinks_and_out_of_bounds(tmp_path):
  root = tmp_path / 'routes'
  segment = root / '00000001--abc--0'
  segment.mkdir(parents=True)
  source = segment / 'qcamera.mp4'
  source.write_bytes(b'0123456789')
  files = RouteFiles(root, tmp_path)
  assert files.read('00000001--abc--0/qcamera.mp4', 2, 4) == b'2345'
  for path in ('../secret', '00000001--abc--0/../../secret', '/etc/passwd', '00000001--abc--0/other.mp4'):
    with pytest.raises(ValueError):
      files.path(path)
  for offset, length in ((-1, 1), (0, 0), (10, 1), (0, MAX_RANGE_BYTES + 1), (True, 1)):
    with pytest.raises(ValueError):
      files.read('00000001--abc--0/qcamera.mp4', offset, length)
  source.unlink()
  source.symlink_to(tmp_path / 'private-key')
  with pytest.raises(ValueError):
    files.path('00000001--abc--0/qcamera.mp4')


def test_inventory_groups_segments_and_excludes_active_recordings(tmp_path):
  for segment in ('00000001--abc--10', '00000001--abc--2', '00000002--def--0', 'boot'):
    path = tmp_path / segment
    path.mkdir()
    (path / 'qcamera.mp4').write_bytes(b'video')
  (tmp_path / '00000002--def--0/qcamera.mp4.lock').touch()
  routes = RouteFiles(tmp_path, tmp_path).list_routes()
  assert len(routes) == 1
  assert [file['segment'] for file in routes[0]['files']] == [2, 10]


def test_fragment_manifest_covers_seekable_mp4(tmp_path):
  source = tmp_path / 'clip.mp4'
  make_video(source, 2)
  manifest = mp4_manifest(source)
  assert manifest['codec'].startswith('avc1.')
  assert manifest['fragments'][0]['kind'] == 'init'
  assert [fragment['timeMs'] for fragment in manifest['fragments'][1:]] == [0, 1000]
  cursor = 0
  for fragment in manifest['fragments']:
    assert fragment['offset'] == cursor
    cursor += fragment['length']
  assert cursor == source.stat().st_size


def test_removed_peer_cannot_read_over_existing_connection():
  session = RouteSession.__new__(RouteSession)
  session.authorized = lambda: False
  sent = []

  async def send(message):
    sent.append(json.loads(message))

  session.send = send
  asyncio.run(session.respond(1, {'op': 'read', 'path': '00000001--abc--0/qcamera.mp4', 'offset': 0, 'length': 1}))
  assert sent == [{'id': 1, 'error': 'device access has been removed'}]


def test_send_waits_for_native_drain_without_reading_buffered_amount():
  async def run():
    class Channel:
      def set_buffered_amount_low_threshold(self, threshold):
        assert threshold == 0

      def on_buffered_amount_low(self, callback):
        self.drained = callback

      def is_open(self):
        return True

      def send(self, data):
        assert data == b'fragment'
        asyncio.get_running_loop().call_later(0.01, self.drained)
        return False

    channel = Channel()
    session = RouteSession.__new__(RouteSession)
    session.stream = type('Stream', (), {'get_messaging_channel': lambda _: channel})()
    session.channel_ready = False
    session.closed = False
    session.send_lock = asyncio.Lock()
    session.drained = asyncio.Event()
    await asyncio.wait_for(session.send(b'fragment'), timeout=1)
    assert session.drained.is_set()

  asyncio.run(run())


@pytest.mark.parametrize('name,codec,format_name', [('qcamera.ts', 'libx264', 'mpegts'), ('fcamera.hevc', 'libx265', 'hevc')])
def test_original_remux_and_browser_conversion_preserve_source(tmp_path, name, codec, format_name):
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  source = segment / name
  make_video(source, 2, codec, format_name)
  original = source.read_bytes()
  cache = tmp_path / 'cache'
  cache.mkdir()
  files = RouteFiles(tmp_path, cache)
  relative = f'00000001--abc--0/{name}'
  for h264 in (False, True):
    manifest = files.manifest(relative, h264)
    assert manifest['codec'].startswith('avc1.' if h264 or codec == 'libx264' else 'hvc1.')
    assert len(manifest['fragments']) >= 3
    init = manifest['fragments'][0]
    assert files.read(relative, 0, init['length'], True, h264)[4:8] == b'ftyp'
  assert source.read_bytes() == original


def test_recording_date_ignores_changes_to_segment_directory(tmp_path):
  import os
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  file = segment / 'qcamera.mp4'
  file.write_bytes(b'video')
  os.utime(file, (1700000000, 1700000000))
  files = RouteFiles(tmp_path, tmp_path)
  before = files.list_routes()[0]['createdAt']
  (segment / '.temporary-upload').write_bytes(b'temporary')
  assert files.list_routes()[0]['createdAt'] == before == 1700000000000


def test_truncated_tail_plays_only_complete_fragments(tmp_path):
  source = tmp_path / 'clip.mp4'
  make_video(source, 3)
  original = source.read_bytes()
  boxes = read_boxes(original, 0, len(original))
  last_mdat = [box for box in boxes if box.type == b'mdat'][-1]
  last_moof = [box for box in boxes if box.type == b'moof'][-1]
  for cut in (last_moof.start + 3, last_moof.end, last_mdat.payload_start + 1, last_mdat.end - 1):
    source.write_bytes(original[:cut])
    manifest = mp4_manifest(source)
    assert manifest['recovered']
    assert len(manifest['fragments']) == 3
    end = manifest['fragments'][-1]
    assert end['offset'] + end['length'] == last_moof.start
    assert end['timeMs'] + end['durationMs'] == 2000
    assert source.read_bytes() == original[:cut]


@pytest.mark.parametrize('content', [b'', make_box(b'ftyp', b'isom0000'), make_box(b'moov', make_box(b'trak', b''))])
def test_empty_or_missing_metadata_returns_async_error_without_hanging(tmp_path, content):
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  (segment / 'qcamera.mp4').write_bytes(content)
  session = RouteSession.__new__(RouteSession)
  session.authorized = lambda: True
  session.files = RouteFiles(tmp_path, tmp_path)
  sent = []

  async def send(message):
    sent.append(json.loads(message))

  session.send = send

  async def run():
    await asyncio.wait_for(session.respond(1, {'op': 'manifest', 'path': '00000001--abc--0/qcamera.mp4'}), timeout=1)

  asyncio.run(run())
  assert sent == [{'id': 1, 'error': 'This segment has no usable video frames', 'code': 'recording_unavailable'}]


def test_conversion_prepares_only_requested_section(tmp_path):
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  source = segment / 'fcamera.mp4'
  make_video(source, 14)
  original = source.read_bytes()
  cache = tmp_path / 'cache'
  cache.mkdir()
  files = RouteFiles(tmp_path, cache)
  relative = f'{segment.name}/fcamera.mp4'
  for start, expected in [(0, 6000), (6, 6000), (12, 2000)]:
    manifest = files.manifest(relative, True, start)
    media = manifest['fragments'][1:]
    assert media[-1]['timeMs'] + media[-1]['durationMs'] == expected
    for fragment in manifest['fragments']:
      assert len(files.read(relative, fragment['offset'], fragment['length'], True, True, start)) == fragment['length']
  with pytest.raises(RecordingUnavailableError):
    files.manifest(relative, True, 18)
  for start in (-6, 1, 60, True):
    with pytest.raises(ValueError, match='invalid video conversion range'):
      files.manifest(relative, True, start)
  assert source.read_bytes() == original


def test_cancelled_playback_kills_conversion_and_releases_next_request(tmp_path, monkeypatch):
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  (segment / 'fcamera.mp4').write_bytes(b'unused by fake encoder')
  session = RouteSession.__new__(RouteSession)
  session.authorized = lambda: True
  session.files = RouteFiles(tmp_path, tmp_path)
  started = threading.Event()
  processes = []
  popen = subprocess.Popen

  def slow_encoder(_command, **kwargs):
    process = popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
    processes.append(process)
    started.set()
    return process

  monkeypatch.setattr('openpilot.system.webrtc.routes.subprocess.Popen', slow_encoder)

  async def run():
    task = asyncio.create_task(session.respond(1, {'op': 'manifest', 'path': f'{segment.name}/fcamera.mp4', 'h264': True}))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert await asyncio.to_thread(processes[0].wait, 2) != 0
    # Cancellation must release the preparation lock for the next camera.
    data = await asyncio.wait_for(asyncio.to_thread(session.files.read, f'{segment.name}/fcamera.mp4', 0, 6), 2)
    assert data == b'unused'

  asyncio.run(run())
