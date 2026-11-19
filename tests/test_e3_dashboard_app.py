"""Application wiring checks in the framework-free runtime, without fitting models."""
from pathlib import Path
import json
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DashboardApplicationTests(unittest.TestCase):
    def test_health_exposes_only_valid_source_sha_and_render_precedence(self):
        program = r'''
import json, os
from unittest.mock import patch
import app
client = app.server.test_client()
cases = [({}, None, None),
         ({"APP_SOURCE_COMMIT":"b"*40}, "b"*40, "APP_SOURCE_COMMIT"),
         ({"APP_SOURCE_COMMIT":"b"*40,"RENDER_GIT_COMMIT":"A"*40}, "a"*40, "RENDER_GIT_COMMIT"),
         ({"RENDER_GIT_COMMIT":"invalid_private_value"}, None, None),
         ({"RENDER_GIT_COMMIT":"invalid_private_value","APP_SOURCE_COMMIT":"b"*40}, "b"*40, "APP_SOURCE_COMMIT")]
for env, expected, field in cases:
    with patch.dict(os.environ, env, clear=True):
        response = client.get("/healthz")
    assert response.status_code == 200
    health = response.get_json()
    assert health["status"] == "ok" and health["catalogue_systems"] == 17
    assert health["source_commit"] == expected and health["source_commit_source"] == field
    assert "invalid_private_value" not in response.get_data(as_text=True)
print(json.dumps({"health_cases":len(cases)}))
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(json.loads(result.stdout.strip())["health_cases"], 5)

    def test_production_lock_excludes_training_and_browser_frameworks(self):
        lines = (ROOT / "requirements-dashboard.lock.txt").read_text().splitlines()
        packages = {line.split("==", 1)[0].lower().replace("_", "-")
                    for line in lines if line and not line.startswith("#")}
        self.assertIn("dash", packages)
        self.assertIn("onnxruntime", packages)
        self.assertTrue({"torch", "tensorflow", "jax", "playwright", "pytest"}.isdisjoint(packages))
        self.assertTrue(all("==" in line for line in lines if line and not line.startswith("#")))

    def test_app_imports_no_training_code_and_serves_real_dash_endpoints(self):
        program = r'''
import base64, json, sys
import app, dash
forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
             or name == "training" or name.startswith("training.")
             or name.startswith("src.e3_") or name == "src.original_model"]
assert not forbidden, forbidden
client = app.server.test_client()
paths = ["/", "/eda", "/modelos", "/_dash-layout", "/_dash-dependencies"]
statuses = {path: client.get(path).status_code for path in paths}
assert all(status == 200 for status in statuses.values()), statuses
layout = client.get("/_dash-layout").get_json()
assert isinstance(layout, dict)
dependencies = client.get("/_dash-dependencies").get_json()
assert isinstance(dependencies, list) and dependencies
assert app.app.use_pages is True
oversized = client.post("/_dash-update-component", data=b" " * (512 * 1024 + 1),
                        content_type="application/json")
assert oversized.status_code == 413
assert oversized.get_json() == {"error": "La solicitud supera el tamaño permitido."}
payload = {
    "output": "prediction-result.children",
    "outputs": {"id": "prediction-result", "property": "children"},
    "inputs": [{"id": "predict-button", "property": "n_clicks", "value": 1}],
    "state": [
        {"id": "live-model", "property": "value", "value": "MLP16"},
        {"id": "live-source", "property": "value", "value": "upload"},
        {"id": "live-example", "property": "value", "value": None},
        {"id": "live-upload", "property": "contents", "value": "data:text/csv;base64," +
         base64.b64encode(b"date,close\n2026-01-01,abc\n").decode()},
        {"id": "live-origin", "property": "date", "value": "2026-01-02"},
    ],
    "changedPropIds": ["predict-button.n_clicks"],
}
invalid_csv = client.post("/_dash-update-component", json=payload)
assert invalid_csv.status_code == 200, invalid_csv.get_data(as_text=True)
error_render = json.dumps(invalid_csv.get_json(), ensure_ascii=False)
assert "columnas" in error_render.lower() and "Traceback" not in error_render
registered_paths = {page["path"] for page in dash.page_registry.values()}
assert {"/", "/eda", "/modelos"}.issubset(registered_paths), registered_paths
print(json.dumps({"statuses": statuses, "callbacks": len(dependencies), "training_imports": forbidden,
                  "registered_paths": sorted(registered_paths)}))
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:] + result.stdout[-2000:])
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["training_imports"], [])
        self.assertGreaterEqual(report["callbacks"], 3)


if __name__ == "__main__":
    unittest.main()
