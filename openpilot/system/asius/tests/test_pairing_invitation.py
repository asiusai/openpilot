from concurrent.futures import ThreadPoolExecutor

from openpilot.system.asius import pairing_invitation as invitation


def test_invitation_uses_monotonic_expiry_and_consumes_once(tmp_path):
  token = invitation.issue('device', 30, tmp_path, 100)
  assert invitation.consume(token, 'device', tmp_path, 129)
  assert not invitation.consume(token, 'device', tmp_path, 129)
  token = invitation.issue('device', 30, tmp_path, 200)
  assert not invitation.consume(token, 'device', tmp_path, 230)
  assert not invitation.consume('../status', 'device', tmp_path, 200)


def test_concurrent_authorization_cannot_reuse_one_invitation(tmp_path):
  token = invitation.issue('device', 30, tmp_path, 100)
  with ThreadPoolExecutor(max_workers=16) as workers:
    results = list(workers.map(lambda _: invitation.consume(token, 'device', tmp_path, 101), range(32)))
  assert sum(results) == 1


def test_concurrent_issuers_share_the_pending_limit(tmp_path):
  from concurrent.futures import ThreadPoolExecutor
  from openpilot.system.asius.pairing_invitation import issue

  def attempt(_):
    try:
      return issue('recipient', 60, tmp_path, now=10.)
    except ValueError:
      return None

  with ThreadPoolExecutor(max_workers=16) as workers:
    issued = list(workers.map(attempt, range(80)))
  assert len([token for token in issued if token]) == 32
  assert len(list(tmp_path.glob('*.json'))) == 32
