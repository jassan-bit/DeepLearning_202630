"""Execute the shipped CPU-only predictor through its real CSV contract."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, "/app")
import app
from e3_dashboard.inference import available_models, predict_csv


root = Path("/app")
expected = {"MLP16", "LSTM32", "GRU32", "CNN-LSTM-atencion-aditiva E3", "TSMixer"}
capabilities = available_models()
assert {row["id"] for row in capabilities} == expected
client = app.server.test_client()
statuses = {path: client.get(path).status_code for path in ("/", "/eda", "/modelos", "/healthz")}
assert set(statuses.values()) == {200}, statuses
health = client.get("/healthz").get_json()
assert health["catalogue_systems"] == 17
examples = json.loads((root / "data/e3_dashboard_v1/examples.json").read_bytes())
predictions = json.loads((root / "data/e3_dashboard_v1/predictions.json").read_bytes())
example = examples[0]
index = predictions["dates"].index(example["target_date"])
results = []
for model_id in sorted(expected):
    result = predict_csv(model_id, example["csv_text"], example["target_date"])
    frozen = predictions["models"][model_id]["predictions"][index]
    error = abs(result["prediction"]-frozen)
    assert error < 1e-6, (model_id, error)
    assert result["interval"] is not None and result["nominal_coverage"] == .9
    assert len(result["explanation"]["attributions"]) == 7
    assert math.isfinite(result["prediction"])
    results.append({"model_id": model_id, "prediction_sigma7": result["prediction"],
                    "absolute_error_vs_frozen": error, "interval": result["interval"],
                    "calibration_profile": result.get("calibration_profile"),
                    "explanation_days": len(result["explanation"]["attributions"])})
forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
             or name == "training" or name.startswith("training.") or name.startswith("src.e3_")]
assert not forbidden, forbidden
sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
           ("app.py", "pages/modelos.py", "pages/contexto.py", "e3_dashboard/inference.py",
            "artifacts/e3_serving_calibration_v2/manifest.json",
            "artifacts/e3_serving_extensions_v2/manifest.json")}
print(json.dumps({"status": "passed_local_container_only", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "python": platform.python_version(), "source_hashes": sources,
                  "HTTP_statuses": statuses, "health": health,
                  "forbidden_imports": forbidden, "example_target_date": example["target_date"],
                  "results": results, "new_training_runs": 0, "new_calibration_fits": 0,
                  "public_Render_acceptance": False}, indent=2))
