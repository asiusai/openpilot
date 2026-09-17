import json
import subprocess
import unittest
from unittest.mock import patch

from openpilot.system.app.clock import ClockChallenges


class TestClockChallenges(unittest.TestCase):
  def test_fresh_peer_bound_one_use_challenge(self):
    clock = ClockChallenges()
    challenge = clock.challenge('owner')['challenge']
    result = subprocess.CompletedProcess([], 0, json.dumps({'accepted': True, 'correction': 'forward'}))
    with patch('openpilot.system.app.clock.subprocess.run', return_value=result) as run:
      with self.assertRaises(ValueError):
        clock.sync('other-peer', challenge, 1800000000000)
      self.assertTrue(clock.sync('owner', challenge, 1800000000000)['accepted'])
      with self.assertRaises(ValueError):
        clock.sync('owner', challenge, 1800000000000)
    run.assert_called_once()
    self.assertIn('/usr/bin/vamos-clock', run.call_args.args[0])

  def test_expiry_is_monotonic_not_wall_clock_based(self):
    clock = ClockChallenges()
    with patch('openpilot.system.app.clock.time.monotonic', return_value=10):
      challenge = clock.challenge('owner')['challenge']
    with patch('openpilot.system.app.clock.time.monotonic', return_value=16), patch('openpilot.system.app.clock.subprocess.run') as run:
      with self.assertRaisesRegex(ValueError, 'expired'):
        clock.sync('owner', challenge, 1800000000000)
    run.assert_not_called()

  def test_invalid_samples_and_replaced_challenges(self):
    clock = ClockChallenges()
    old = clock.challenge('owner')['challenge']
    current = clock.challenge('owner')['challenge']
    with patch('openpilot.system.app.clock.subprocess.run') as run:
      with self.assertRaises(ValueError):
        clock.sync('owner', old, 1800000000000)
      for value in (True, 'yesterday', float('nan'), float('inf')):
        with self.assertRaises(ValueError):
          clock.sync('owner', current, value)
    run.assert_not_called()
