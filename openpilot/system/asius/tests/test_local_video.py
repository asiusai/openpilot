from unittest.mock import Mock

import pytest

from openpilot.system.asius import local_video
from openpilot.system.webrtc import helpers


def test_local_stream_checks_lease_and_accepts_no_control_services(monkeypatch):
  wifi = Mock(return_value={'address': '192.168.43.1'})
  post = Mock(return_value={'sdp': 'answer', 'type': 'answer'})
  monkeypatch.setattr(local_video, 'wifi', wifi)
  monkeypatch.setattr(helpers, 'wait_for_webrtcd', Mock())
  monkeypatch.setattr(helpers, 'post_stream_request', post)
  assert local_video.methods('authorized-app')['startLocalStream']('lease', 'offer', 'road')['type'] == 'answer'
  wifi.assert_called_once_with('authorized-app', 'check', 'lease')
  body = post.call_args.args[0]
  assert body.cameras == ['road']
  assert body.bind_address == '192.168.43.1'
  assert body.bridge_services_in == []
  assert body.bridge_services_out == []
  assert body.in_car and body.enabled


@pytest.mark.parametrize('sdp,camera', [('offer', 'unknown'), ('x' * 65537, 'road'), (123, 'driver')])
def test_invalid_stream_never_opens_wifi(monkeypatch, sdp, camera):
  wifi = Mock()
  monkeypatch.setattr(local_video, 'wifi', wifi)
  with pytest.raises(ValueError):
    local_video.start_stream('app', 'lease', sdp, camera)
  wifi.assert_not_called()


def test_invalid_lease_cannot_start_camera(monkeypatch):
  monkeypatch.setattr(local_video, 'wifi', Mock(side_effect=ValueError('Expired lease')))
  post = Mock()
  monkeypatch.setattr(helpers, 'post_stream_request', post)
  with pytest.raises(ValueError, match='Expired lease'):
    local_video.start_stream('other-app', 'lease', 'offer', 'driver')
  post.assert_not_called()
