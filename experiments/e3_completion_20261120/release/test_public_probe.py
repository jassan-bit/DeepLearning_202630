"""Guard against false public acceptance/publication and proxy attribution."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import URLError

spec = importlib.util.spec_from_file_location("public_probe", Path(__file__).with_name("public_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PublicEvidenceGuards(unittest.TestCase):
    def test_connect_proxy_denial_is_not_a_render_origin_403(self):
        class Opener:
            def open(self, *args, **kwargs):
                raise URLError("Tunnel connection failed: 403 Forbidden")
        with patch.object(probe, "build_opener", return_value=Opener()):
            record = probe.request(probe.RENDER + "/healthz", Path("unused"), "health")
        self.assertIsNone(record["origin_http_status"])
        self.assertFalse(record["upstream_response_received"])

    def test_reserved_or_blocked_doi_is_not_publication(self):
        rows = {name: {"origin_http_status": None, "upstream_response_received": False}
                for name in ("zenodo_public_api", "doi_resolver")}
        self.assertFalse(probe.doi_verdict(rows)["publication_verified"])

    def test_matching_public_metadata_and_resolver_are_both_required(self):
        api = {"origin_http_status": 200, "json": {"id": 24356217,
               "doi": probe.DOI, "metadata": {"title": "Selected source audit"}}}
        rows = {"zenodo_public_api": api, "doi_resolver": {"origin_http_status": 200,
                "effective_url": "https://zenodo.org/uploads/24356217"}}
        self.assertFalse(probe.doi_verdict(rows)["publication_verified"])
        rows["doi_resolver"]["effective_url"] = "https://zenodo.org/records/24356217"
        verdict = probe.doi_verdict(rows)
        self.assertTrue(verdict["publication_verified"])
        self.assertFalse(verdict["archive_completeness_verified"])


if __name__ == "__main__":
    unittest.main()
