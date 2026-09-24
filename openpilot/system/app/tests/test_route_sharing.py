import unittest

from openpilot.system.app.route_sharing import publication_request


class TestRouteSharing(unittest.TestCase):
  def test_only_selected_route_files_can_be_published(self):
    entry = {"path": "routes/route-one--0/qlog.zst", "checksumSha256": "a" * 43, "key": "b" * 43}
    self.assertEqual(publication_request("route-one", True, [entry]), {"enabled": True, "files": [entry]})
    for path in ("routes/route-two--0/qlog.zst", "routes/route-one--0/../access.json", "access.json"):
      with self.assertRaises(ValueError):
        publication_request("route-one", True, [entry | {"path": path}])

  def test_disabling_discards_keys_and_enabling_requires_files(self):
    self.assertEqual(publication_request("route-one", False, []), {"enabled": False, "files": []})
    with self.assertRaises(ValueError):
      publication_request("route-one", True, [])
    with self.assertRaises(ValueError):
      publication_request("../route-one", True, [])


if __name__ == "__main__":
  unittest.main()
