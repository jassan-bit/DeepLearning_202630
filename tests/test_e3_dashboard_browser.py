"""Real browser acceptance checks; opt in with E3_DASHBOARD_TEST_URL.

The server must already be running in the runtime environment. This client uses
Playwright only as a development dependency, never in the production service.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import unittest
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("E3_DASHBOARD_TEST_URL", "").rstrip("/")
GRAPH_IDS = ("eda-series", "eda-hist", "eda-feature", "eda-partitions-figure", "eda-acf", "eda-pacf")


@unittest.skipUnless(BASE_URL, "Set E3_DASHBOARD_TEST_URL to run real browser checks")
class DashboardBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        launch = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if os.environ.get("E3_CHROMIUM_PATH"):
            launch["executable_path"] = os.environ["E3_CHROMIUM_PATH"]
        cls.browser = cls.pw.chromium.launch(**launch)
        cls.context = cls.browser.new_context(viewport={"width": 1440, "height": 1000},
                                              locale="es-CO", timezone_id="America/Bogota")
        cls.page = cls.context.new_page()
        cls.page.set_default_timeout(20000)
        cls.page_errors = []
        cls.console_errors = []
        cls.server_errors = []
        cls.failed_requests = []
        cls.checks = []
        cls.screenshots = []
        cls._result = None
        cls.page.on("pageerror", lambda error: cls.page_errors.append(str(error)))
        cls.page.on("console", lambda message: cls.console_errors.append(message.text)
                    if message.type == "error" else None)
        cls.page.on("response", lambda response: cls.server_errors.append(
            {"status": response.status, "url": response.url}) if
            response.status >= 500 and urlparse(response.url).netloc == urlparse(BASE_URL).netloc else None)
        cls.page.on("requestfailed", lambda request: cls.failed_requests.append(
            {"url": request.url, "failure": request.failure}))
        cls.screenshot_dir = Path(os.environ.get(
            "E3_DASHBOARD_SCREENSHOT_DIR", ROOT / "book_e3/_static/e3_dashboard_20261117"))
        cls.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        type(self)._result = self._outcome.result

    def verify(self, name, condition, detail=None):
        self.checks.append({"check": name, "passed": bool(condition), "detail": detail})
        self.assertTrue(condition, f"{name}: {detail}")

    def goto(self, path):
        response = self.page.goto(BASE_URL + path, wait_until="networkidle")
        self.assertIsNotNone(response)
        self.assertEqual(response.status, 200)
        self.page.locator("main h1").wait_for()

    def screenshot(self, name):
        path = self.screenshot_dir / name
        self.page.screenshot(path=str(path), full_page=True, animations="disabled")
        self.screenshots.append({"name": name, "bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def wait_eda_n(self, expected):
        self.page.wait_for_function(
            """([ids,n]) => ids.every(id => {
              const gd=document.querySelector('#'+id+' .js-plotly-plot');
              return gd && gd.layout && gd.layout.meta && gd.layout.meta.filtered_n===n;
            })""", arg=[list(GRAPH_IDS), expected])
        text = self.page.locator("#eda-n").inner_text()
        self.verify(f"EDA shared mask n={expected}",
                    self.page.locator("#eda-n strong").inner_text().replace(".", "").replace(",", "") == str(expected), text)

    def choose(self, component_id, search):
        field = self.page.locator(f"#{component_id} input").first
        field.fill(search)
        field.press("Enter")

    def predict(self):
        def is_result(response):
            if not response.url.endswith("/_dash-update-component") or response.request.method != "POST":
                return False
            return response.request.post_data_json.get("output") == "prediction-result.children"
        with self.page.expect_response(is_result) as pending:
            self.page.locator("#predict-button").click()
        self.assertEqual(pending.value.status, 200)

    def test_01_context_and_documentation(self):
        self.goto("/")
        hrefs = self.page.locator("nav a").evaluate_all("nodes=>nodes.map(n=>n.getAttribute('href'))")
        self.verify("Three-page navigation", {"/", "/eda", "/modelos"}.issubset(set(hrefs)), hrefs)
        body = self.page.locator("main").inner_text()
        self.verify("Context includes real cohort and task", "608" in body and "σ" in body, body[:250])
        for path in ("/model-card", "/data-statement"):
            response = self.page.request.get(BASE_URL + path)
            self.verify("Accessible document " + path, response.status == 200 and len(response.text()) > 200)
        health = self.page.request.get(BASE_URL + "/healthz")
        self.verify("Health includes complete catalogue", health.status == 200 and health.json().get("catalogue_systems") == 17)
        self.screenshot("contexto_desktop.png")

    def test_02_global_eda_filters(self):
        self.goto("/eda")
        self.wait_eda_n(1316)
        self.screenshot("eda_desktop.png")
        dates = self.page.locator("#eda-dates input")
        dates.nth(0).fill("01/01/2025")
        dates.nth(0).press("Enter")
        self.page.locator("#eda-variable").click()
        self.wait_eda_n(608)
        self.page.locator("#eda-partitions .Select-clear-zone").click()
        self.wait_eda_n(0)
        self.choose("eda-partitions", "Prueba")
        self.wait_eda_n(608)
        supports = self.page.evaluate("""() => {
          const g=id=>document.querySelector('#'+id+' .js-plotly-plot');
          return {series:g('eda-series').data.reduce((n,t)=>n+t.y.filter(v=>v!==null).length,0),
            histogram:g('eda-hist').data[0].x.length,
            feature:g('eda-feature').data[0].y.filter(v=>v!==null).length,
            partitions:g('eda-partitions-figure').data.reduce((n,t)=>n+t.y.length,0)};
        }""")
        self.verify("EDA trace supports follow the same 608-row mask", all(n == 608 for n in supports.values()), supports)
        self.choose("eda-variable", "volumen")
        self.page.wait_for_function("""() => {
          const gd=document.querySelector('#eda-feature .js-plotly-plot');
          return gd && gd.layout.yaxis.title.text==='BTC';
        }""")
        self.wait_eda_n(608)
        upper = self.page.locator("#eda-range [role=slider]").nth(1)
        upper.click()
        upper.press("Home")
        self.wait_eda_n(0)
        self.verify("Empty EDA explains missing diagnostics", "vacía" in self.page.locator("#eda-insight").inner_text().lower())
        empty_counts = self.page.evaluate("""ids=>ids.map(id=>
            document.querySelector('#'+id+' .js-plotly-plot').data.length)""", list(GRAPH_IDS))
        self.verify("Empty EDA clears all stale traces", all(n == 0 for n in empty_counts), empty_counts)

    def test_03_models_catalogue_and_diagnostics(self):
        self.goto("/modelos")
        self.page.locator("#model-select").wait_for()
        body = self.page.locator("main").inner_text()
        self.verify("Model page represents all 17 systems", "17" in body and "CNN" in body)
        cells = self.page.locator('#leaders-table td[data-dash-column="model"]')
        self.page.wait_for_function("document.querySelectorAll('#leaders-table td[data-dash-column=\"model\"]').length===17")
        self.verify("Leaderboard contains all 17 rows", cells.count() == 17)
        first_before = cells.first.inner_text()
        self.page.locator('#leaders-table th[data-dash-column="RMSE_mean"] .column-header--sort').click()
        self.page.wait_for_function("""before => document.querySelector(
            '#leaders-table td[data-dash-column="model"]').innerText!==before""", arg=first_before)
        self.verify("Leaderboard sort changes the visible order", cells.first.inner_text() != first_before)
        filter_box = self.page.locator('#leaders-table th[data-dash-column="model"] input[type=text]')
        filter_box.fill("ARIMA")
        filter_box.press("Enter")
        self.page.wait_for_function("document.querySelectorAll('#leaders-table td[data-dash-column=\"model\"]').length===1")
        self.verify("Leaderboard filtering preserves the requested row", cells.first.inner_text() == "ARIMA")
        filter_box.fill("")
        filter_box.press("Enter")
        self.page.wait_for_function("document.querySelectorAll('#leaders-table td[data-dash-column=\"model\"]').length===17")
        self.choose("model-select", "ARIMA")
        self.page.wait_for_function("() => document.querySelector('#model-summary').innerText.includes('ARIMA')")
        graphs = self.page.locator("main .js-plotly-plot").count()
        self.verify("Model diagnostics render", graphs >= 3, graphs)
        self.page.locator("#model-card-open").click()
        self.page.locator("#model-card-modal").wait_for(state="visible")
        self.verify("Model card opens inside the application", "1210" in self.page.locator("#model-card-modal").inner_text())
        self.page.locator("#model-card-modal .btn-close").click()
        self.page.locator("#model-card-modal").wait_for(state="hidden")
        self.page.get_by_role("button", name="Eficiencia y costes comparables", exact=True).click()
        self.page.locator("#pareto-panel .js-plotly-plot").wait_for()
        pareto_count = self.page.locator("#pareto-panel .js-plotly-plot").evaluate("gd=>gd.data[0].x.length")
        self.verify("Pareto uses the four measured CPU exports", pareto_count == 4, pareto_count)
        self.screenshot("modelos_desktop.png")

    def test_04_explicit_prediction_and_input_error(self):
        self.goto("/modelos")
        self.page.locator("#predict-button").wait_for()
        self.choose("live-model", "MLP16")
        self.predict()
        self.page.wait_for_function("""() => {
          const el=document.querySelector('#prediction-result');
          return el && el.innerText.trim().length>20 && !el.innerText.includes('Calculando');
        }""")
        text = self.page.locator("#prediction-result").inner_text()
        self.verify("Live CPU prediction returns a result", "σ" in text and not any(
            bad in text.lower() for bad in ("traceback", "no se pudo", "no disponible")), text[:500])
        self.verify("Reference interval has an explicit nominal level", "INTERVALO NOMINAL" in text and "90%" in text)
        self.choose("live-model", "CNN-LSTM")
        self.predict()
        self.page.wait_for_function("""() => document.querySelector('#prediction-result').innerText
            .includes('CNN-LSTM-atencion-aditiva E3')""")
        e3_text = self.page.locator("#prediction-result").inner_text()
        self.verify("E3 missing calibrated interval is explicit", "Sin intervalo calibrado" in e3_text)
        shown = float(self.page.locator("#prediction-result .forecast-value strong").inner_text())
        examples = json.loads((ROOT / "data/e3_dashboard_v1/examples.json").read_bytes())
        trajectories = json.loads((ROOT / "data/e3_dashboard_v1/predictions.json").read_bytes())
        index = trajectories["dates"].index(examples[0]["target_date"])
        frozen = trajectories["models"]["CNN-LSTM-atencion-aditiva E3"]["predictions"][index]
        self.verify("Displayed E3 forecast matches its frozen representative", math.isfinite(shown) and abs(shown-frozen) < 1e-6,
                    {"shown_six_decimals": shown, "frozen": frozen})
        # Actual malformed file upload, not a direct call to the parsing helper.
        upload_option = self.page.locator("#live-source input[value='upload']")
        if upload_option.count():
            upload_option.check()
        else:
            self.page.get_by_label("Mi archivo CSV", exact=True).check()
        self.page.locator("#live-upload input[type=file]").set_input_files(
            {"name": "valid.csv", "mimeType": "text/csv", "buffer": examples[0]["csv_text"].encode()})
        self.page.wait_for_function("() => document.querySelector('#upload-description').innerText.includes('valid.csv')")
        self.predict()
        self.page.locator("#prediction-result .forecast-value strong").wait_for()
        uploaded = float(self.page.locator("#prediction-result .forecast-value strong").inner_text())
        self.verify("Valid uploaded CSV preserves the same raw-window forecast", abs(uploaded-frozen) < 1e-6,
                    {"uploaded_six_decimals": uploaded, "frozen": frozen})
        self.page.locator("#live-upload input[type=file]").set_input_files(
            {"name": "invalid.csv", "mimeType": "text/csv", "buffer": b"date,close\n2026-01-01,abc\n"})
        self.page.wait_for_function("() => document.querySelector('#upload-description').innerText.includes('invalid.csv')")
        self.predict()
        self.page.wait_for_function("""() => {
          const el=document.querySelector('#prediction-result');
          return el && el.querySelector('[role="alert"]') && !el.querySelector('.forecast-value');
        }""")
        message = self.page.locator("#prediction-result").inner_text()
        self.verify("Invalid uploaded CSV is explained without a trace", "columnas" in message.lower() and
                    "traceback" not in message.lower() and "/workspace" not in message, message)

    def test_05_responsive_tablet_and_keyboard(self):
        self.page.set_viewport_size({"width": 800, "height": 1050})
        for path, name in (("/", "contexto_tablet.png"), ("/eda", "eda_tablet.png"),
                           ("/modelos", "modelos_tablet.png")):
            self.goto(path)
            if path == "/eda":
                self.wait_eda_n(1316)
            if path == "/modelos":
                self.page.locator("#model-select").wait_for()
            dimensions = self.page.evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth})")
            self.verify("Tablet has no document overflow: " + path,
                        dimensions["scroll"] <= dimensions["width"] + 2, dimensions)
            self.screenshot(name)
        self.goto("/")
        self.page.keyboard.press("Tab")
        self.verify("Keyboard skip link is reachable", "Ir al contenido" in self.page.evaluate(
            "document.activeElement.textContent"))
        self.page.set_viewport_size({"width": 1440, "height": 1000})

    def test_06_no_browser_or_server_exceptions(self):
        self.verify("No uncaught browser exception", not self.page_errors, self.page_errors)
        self.verify("No browser console error", not self.console_errors, self.console_errors)
        self.verify("No internal HTTP 5xx", not self.server_errors, self.server_errors)

    @classmethod
    def tearDownClass(cls):
        report = {
            "schema_version": 1,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "local_automated_browser_acceptance_not_Render_deployment_or_human_reproduction",
            "server_url": BASE_URL,
            "python": platform.python_version(), "chromium": cls.browser.version,
            "viewports": [{"width": 1440, "height": 1000}, {"width": 800, "height": 1050}],
            "checks": cls.checks,
            "uncaught_browser_errors": cls.page_errors,
            "browser_console_errors": cls.console_errors,
            "internal_HTTP_5xx": cls.server_errors,
            "failed_requests": cls.failed_requests,
            "screenshots": cls.screenshots,
            "test_cases": cls._result.testsRun if cls._result else 0,
            "failures": len(cls._result.failures) if cls._result else None,
            "errors": len(cls._result.errors) if cls._result else None,
            "render_deployed": False,
            "R1_R2_R3_certified": False,
        }
        report["status"] = "passed" if report["failures"] == report["errors"] == 0 and not cls.page_errors and not cls.console_errors and not cls.server_errors else "failed"
        source_files = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py")),
                        ROOT / "requirements-dashboard.lock.txt"]
        report["source_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_files if path.is_file()}
        destination = Path(os.environ.get("E3_DASHBOARD_BROWSER_REPORT",
            ROOT / "experiments/e3_dashboard_20261117/browser_verification.json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        cls.browser.close()
        cls.pw.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
