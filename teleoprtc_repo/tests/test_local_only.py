from unittest.mock import Mock

import pytest

from teleoprtc.builder import WebRTCAnswerBuilder


@pytest.mark.parametrize("local_only", [False, True])
def test_local_answer_disables_external_ice_servers(monkeypatch, local_only):
  connection = Mock()
  monkeypatch.setattr("teleoprtc.stream.PeerConnection", connection)
  WebRTCAnswerBuilder("offer", bind_address="192.168.43.1", local_only=local_only).stream()
  config = connection.call_args.args[0]
  assert config.bind_address == "192.168.43.1"
  assert len(config.ice_servers) == (0 if local_only else 1)
