"""Text-only audit contracts; no training packages, arrays or checkpoints."""
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import e3_audit_innovation as audit


@pytest.fixture(scope="module")
def admitted():
    snapshot, proof, payloads, docs = audit.load_inputs(audit.DEFAULT_AUDIT / "inputs")
    return snapshot, proof, payloads, docs


@pytest.fixture(scope="module")
def artifacts():
    return audit.build_artifacts()


def _rewrite_csv(payload, change):
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    change(rows)
    return audit._csv_bytes(rows)


def test_authoritative_run_is_reproduced_without_fitting_or_selection_changes(artifacts):
    metrics = json.loads(artifacts["metrics.json"])
    assert metrics["results_commit"] == audit.COMMIT
    assert metrics["text_files_verified"] == 24
    assert len(metrics["scientific_source_hashes_verified"]) == 8
    assert metrics["best_epochs"] == {"additive_positive_head": 38, "innovation_mean_only": 39, "innovation_density_mean": 39}
    assert metrics["decision"]["best_fixed_control"] == "additive_positive_head"
    assert metrics["decision"]["continuation_gate_passed"] is False
    assert metrics["decision"]["selection_metric_changed"] is False
    assert metrics["operations"] == {"NPZ_opened": False, "checkpoints_opened": False,
        "new_fits": 0, "new_market_inferences": 0, "calibration_or_evaluation_accessed": False,
        "statistical_tests_or_bootstrap": False}
    assert metrics["regime_definition"]["counts"] == {"high": 18, "normal": 98}
    assert metrics["regime_definition"]["observable_at_inference"] is False
    assert metrics["budget"]["global_charged_seconds"] == pytest.approx(556.2126593619992)
    assert len(list(csv.DictReader(io.StringIO(artifacts["per_date_errors.csv"].decode())))) == 696
    assert len(list(csv.DictReader(io.StringIO(artifacts["error_cases.csv"].decode())))) == 60
    assert len(list(csv.DictReader(io.StringIO(artifacts["metrics_by_regime.csv"].decode())))) == 18
    ET.fromstring(artifacts["regime_errors.svg"])


def test_pinned_input_corruption_fails_before_calculating_metrics(tmp_path):
    shutil.copytree(audit.DEFAULT_AUDIT / "inputs", tmp_path / "inputs")
    shutil.copy(audit.DEFAULT_AUDIT / "source_provenance.json", tmp_path / "source_provenance.json")
    with (tmp_path / "inputs/innovation_density_mean/validation_predictions.csv").open("ab") as stream:
        stream.write(b"unexpected")
    with pytest.raises(ValueError, match="SHA-256/size"):
        audit.load_inputs(tmp_path / "inputs")


def test_snapshot_anchor_prevents_rewriting_the_input_inventory(tmp_path):
    shutil.copytree(audit.DEFAULT_AUDIT / "inputs", tmp_path / "inputs")
    shutil.copy(audit.DEFAULT_AUDIT / "source_provenance.json", tmp_path / "source_provenance.json")
    snapshot = tmp_path / "inputs/snapshot_manifest.json"
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Snapshot manifest SHA"):
        audit.load_inputs(tmp_path / "inputs")


@pytest.mark.parametrize("field", ["run_id", "claim_id", "charge", "epoch", "calibration", "best_epoch"])
def test_inconsistent_root_metadata_cannot_be_reconciled_silently(admitted, field):
    snapshot, _proof, payloads, original = admitted
    docs = deepcopy(original)
    if field in ("run_id", "claim_id"):
        docs["summary"][field] = "another_identity"
    elif field == "charge":
        docs["summary"]["charged_seconds"] = 0
    elif field == "epoch":
        docs["state"]["models"][audit.CANDIDATE]["epoch"] = 0
    elif field == "calibration":
        docs["summary"]["calibration_scored"] = True
    else:
        docs["summary"]["models"][audit.CANDIDATE]["best_epoch"] = 1
    with pytest.raises(ValueError):
        audit.validate_metadata(docs, snapshot, payloads)


def test_audit_rejects_changed_local_scientific_source(admitted, tmp_path):
    snapshot, _proof, payloads, docs = admitted
    paths = list(docs["manifest"]["source_hashes"]) + ["experiments/e3_innovacion_v2_20261116/plan.json"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, path)
    with (tmp_path / "src/e3_innovation_model.py").open("ab") as stream:
        stream.write(b"# mutated source\n")
    with pytest.raises(ValueError, match="Scientific source changed"):
        audit.validate_metadata(docs, snapshot, payloads, source_root=tmp_path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "target", "floor"])
def test_every_model_must_have_identical_unique_dates_targets_and_floors(admitted, change):
    _snapshot, _proof, original, docs = admitted
    payloads = dict(original)

    def mutate(rows):
        if change == "missing":
            rows.pop()
        elif change == "duplicate":
            rows[-1]["date"] = rows[0]["date"]
        elif change == "target":
            rows[0]["target_sigma7"] = str(float(rows[0]["target_sigma7"]) + .001)
        else:
            rows[0]["known_floor"] = str(float(rows[0]["known_floor"]) - .001)

    name = audit.CANDIDATE + "/validation_predictions.csv"
    payloads[name] = _rewrite_csv(payloads[name], mutate)
    with pytest.raises(ValueError, match="alignment|Targets/floors"):
        audit.align_predictions(payloads, docs["data_admission"]["validation_dates"])


def test_regime_includes_the_exact_fit_q75_boundary():
    threshold = .026577276251655846
    assert audit.regime(threshold, threshold) == "high"
    assert audit.regime(math.nextafter(threshold, -math.inf), threshold) == "normal"
    assert audit.regime(math.nextafter(threshold, math.inf), threshold) == "high"


def test_sse_decomposition_and_population_bias_variance_have_correct_units():
    rows = [{"prediction_sigma7": p, "target_sigma7": y, "known_floor": .1, "NLL_U": None}
            for p, y in [(1., 2.), (4., 3.), (6., 4.)]]
    all_metrics = audit.aggregate(rows, 3.)
    normal = audit.aggregate(rows[:1], 3.)
    high = audit.aggregate(rows[1:], 3.)
    assert all_metrics["SSE"] == 6.
    assert all_metrics["MSE"] == 2.
    assert all_metrics["SSE"] == normal["SSE"] + high["SSE"]
    assert all_metrics["MSE"] == pytest.approx(all_metrics["bias_squared"] + all_metrics["residual_variance_ddof0"])
    assert all_metrics["residual_variance_ddof0"] == pytest.approx(14 / 9)
    assert all_metrics["underestimation_count"] == 1
    assert all_metrics["mean_predicted_excess"] - all_metrics["mean_realized_excess"] == pytest.approx(all_metrics["bias"])


def test_case_selection_never_drops_dates_or_redefines_the_main_comparison(artifacts):
    records = list(csv.DictReader(io.StringIO(artifacts["per_date_errors.csv"].decode())))
    cases = list(csv.DictReader(io.StringIO(artifacts["error_cases.csv"].decode())))
    expected = sorted([r for r in records if r["model"] == audit.CASE_REFERENCE],
                      key=lambda r: (-float(r["absolute_error"]), r["date"]))[:10]
    assert {r["date"] for r in cases} == {r["date"] for r in expected}
    assert all(r["selection_reference"] == "additive_positive_head" for r in cases)
    assert all(r["selection_status"].startswith("posthoc") for r in cases)
    for name in audit.MODELS:
        assert len([r for r in records if r["model"] == name]) == 116
        assert len([r for r in cases if r["model"] == name]) == 10


def test_check_is_deterministic_accepts_only_tiny_ulp_drift_and_rejects_changed_metadata(artifacts, tmp_path):
    for name, value in artifacts.items():
        (tmp_path / name).write_bytes(value)
    audit.check_artifacts(tmp_path, artifacts)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    old = metrics["models"][audit.CANDIDATE]["all"]["RMSE"]
    metrics["models"][audit.CANDIDATE]["all"]["RMSE"] = math.nextafter(old, math.inf)
    metrics_path.write_text(json.dumps(metrics))
    audit.check_artifacts(tmp_path, artifacts)
    metrics["models"][audit.CANDIDATE]["all"]["RMSE"] = old + .000001
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="numerical mismatch"):
        audit.check_artifacts(tmp_path, artifacts)
    metrics["models"][audit.CANDIDATE]["all"]["RMSE"] = old
    metrics["decision"]["continuation_gate_passed"] = True
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="metadata changed"):
        audit.check_artifacts(tmp_path, artifacts)
    # A second render is byte-identical; no timestamp or randomized SVG IDs.
    assert audit.build_artifacts()["regime_errors.svg"] == artifacts["regime_errors.svg"]


def test_published_downloads_cannot_drift_even_with_self_consistent_new_hashes(artifacts, tmp_path):
    master, exports = tmp_path / "master", tmp_path / "exports"
    master.mkdir(); exports.mkdir()
    inventory = {}
    for name, payload in artifacts.items():
        (master / name).write_bytes(payload)
        (exports / name).write_bytes(payload)
        inventory[name] = {"bytes": len(payload), "sha256": audit.sha(payload),
                           "source": "experiments/e3_cierre_v2_20261116/audit/" + name}
    manifest = {"version": "e3-v2-closure-downloads-v1", "results_commit": audit.COMMIT,
                "files": inventory}
    (exports / "manifest.json").write_text(json.dumps(manifest))
    audit.check_book_exports(exports, master)
    changed = artifacts["metrics_by_regime.csv"] + b"altered\n"
    (exports / "metrics_by_regime.csv").write_bytes(changed)
    manifest["files"]["metrics_by_regime.csv"].update(bytes=len(changed), sha256=audit.sha(changed))
    (exports / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from audited master"):
        audit.check_book_exports(exports, master)
