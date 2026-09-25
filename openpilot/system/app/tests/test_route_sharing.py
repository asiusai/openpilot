import unittest
import base64
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from openpilot.system.loggerd.data_api import canonical_json

from openpilot.system.app.route_sharing import publication_request, authorize_publication


class TestRouteSharing(unittest.TestCase):
  def test_only_selected_route_files_can_be_published(self):
    entry = {"path": "routes/route-one--0/qlog.zst", "checksumSha256": "a" * 43, "key": "b" * 43}
    self.assertEqual(publication_request("route-one", True, [entry]), {"enabled": True, "files": [entry]})
    for path in ("routes/route-two--0/qlog.zst", "routes/route-one--0/../access.json", "access.json"):
      with self.assertRaises(ValueError):
        publication_request("route-one", True, [entry | {"path": path}])

  def test_authorization_is_bound_to_exact_route_and_body_without_network(self):
    private = Ed25519PrivateKey.generate()
    result = authorize_publication(private, "route-one", False, [], 1234567890)
    claims = json.loads(base64.urlsafe_b64decode(result["authorization"].split()[1] + "=="))
    signature = base64.urlsafe_b64decode(claims.pop("signature") + "==")
    private.public_key().verify(signature, canonical_json(claims).encode())
    self.assertEqual(claims["method"], "PUT")
    self.assertEqual(claims["path"], result["path"])
    self.assertEqual(claims["timestamp"], 1234567890)
    self.assertTrue(result["path"].endswith("/routes/route-one/publication"))
    self.assertEqual(claims["bodyHash"], base64.urlsafe_b64encode(hashlib.sha256(result["body"].encode()).digest()).decode().rstrip("="))
    with self.assertRaises(ValueError):
      authorize_publication(private, "../access", False, [], 123)

  def test_disabling_discards_keys_and_enabling_requires_files(self):
    self.assertEqual(publication_request("route-one", False, []), {"enabled": False, "files": []})
    with self.assertRaises(ValueError):
      publication_request("route-one", True, [])
    with self.assertRaises(ValueError):
      publication_request("../route-one", True, [])


if __name__ == "__main__":
  unittest.main()
