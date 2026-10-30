import json
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from openpilot.system.loggerd.data_media import package_cenc_mp4
from openpilot.system.loggerd.tests.media_fixture import make_video


@pytest.mark.parametrize('audio_delay_us', [-100000, 150000])
def test_capture_timestamps_and_audio_share_one_clock(tmp_path, audio_delay_us):
  fixture = Path(__file__).with_name('video_writer_fixture')
  assert fixture.is_file(), 'Build openpilot/system/loggerd/tests/video_writer_fixture with scons first'
  source, output = tmp_path / 'source.mp4', tmp_path / 'output.mp4'
  make_video(source, 2)
  subprocess.run([str(fixture), str(source), str(output), str(audio_delay_us)], check=True)
  result = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_packets', '-show_streams', '-show_format', '-of', 'json', str(output),
  ]))
  packets = result['packets']
  video = [p for p in packets if p['codec_type'] == 'video']
  audio = [p for p in packets if p['codec_type'] == 'audio']
  assert len(video) == 40
  assert abs(float(video[0]['pts_time']) - float(audio[0]['pts_time'])) < 0.05
  assert float(video[0]['pts_time']) < 0.05
  assert float(audio[-1]['pts_time']) < 2.25
  assert all(float(b['pts_time']) - float(a['pts_time']) == pytest.approx(1024 / 48000, abs=0.00001)
             for a, b in zip(audio, audio[1:], strict=False))
  assert float(result['format']['duration']) == pytest.approx(2.2, abs=0.05)
  assert float(video[20]['pts_time']) - float(video[19]['pts_time']) == pytest.approx(0.25, abs=0.001)
  assert result['format']['tags']['ai.asius.startOffsetMs'] == '46000'
  manifest = package_cenc_mp4(output, tmp_path / 'encrypted.mp4', bytes(range(16)), bytes(range(16)))
  assert manifest['startOffsetMs'] == 46000
  assert sum(f.get('durationMs', 0) for f in manifest['fragments']) == pytest.approx(2200, abs=30)


def test_summary_reader_does_not_block_logger_start(tmp_path, monkeypatch):
  from openpilot.common.prefix import OpenpilotPrefix
  monkeypatch.setenv('LOG_ROOT', str(tmp_path))
  binary = Path(__file__).parents[1] / 'loggerd'
  with OpenpilotPrefix(), (tmp_path / '.route-counter.lock').open('a') as reader:
    fcntl.flock(reader, fcntl.LOCK_SH)
    process = subprocess.Popen([str(binary)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
    try:
      deadline = time.monotonic() + 3
      while not list(tmp_path.glob('*--0/qlog.zst')) and time.monotonic() < deadline:
        assert process.poll() is None
        time.sleep(0.01)
      assert list(tmp_path.glob('*--0/qlog.zst')), 'Summary reader blocked logger startup'
      fcntl.flock(reader, fcntl.LOCK_UN)
      # Logger retains a shared lock, so recovery cannot rename active files.
      with pytest.raises(BlockingIOError):
        fcntl.flock(reader, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
      process.send_signal(signal.SIGINT)
      process.wait(timeout=5)
    fcntl.flock(reader, fcntl.LOCK_EX | fcntl.LOCK_NB)
