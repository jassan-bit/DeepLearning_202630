"""New post hoc comparison of the original saved calibration/test trajectories.

No model is loaded and no CPU/GPU predictions are mixed. Calibration remains
calibration, not independent HPO validation. Repeated seeds condition on two
fixed periods; no dates are paired or asserted independent.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "book_e2/_static/e2_scientific_closure_20261116/period_comparison"
BUNDLE = ROOT / "book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json"
BUNDLE_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
VERSION = "e2-original-period-comparison-v1"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer", "MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SEEDS = tuple(range(271801, 271807))


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def sources():
    paths = {
        "bundle": BUNDLE,
        "freeze": ROOT / "book_e2/_static/e2_benchmark_final_20261113/final_freeze.json",
        "A2": ROOT / "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv",
        "original_analysis": ROOT / "book_e2/_static/e2_benchmark_final_20261113/benchmark_analysis.json",
        "phase_resources": ROOT / "book_e2/_static/e2_efficiency_closure_20261113/phase_resources.csv",
        "exporter": ROOT / "scripts/export_e2_f2a_cached_data.py",
        "original_engine_in_checkout": ROOT / "scripts/e2_reduced_final.py",
    }
    require(sha(BUNDLE) == BUNDLE_SHA, "Original exported trajectory bytes changed")
    require(sha(paths["original_engine_in_checkout"]) == "033e4bf7db0d4f69624063830d116a1b5cf159a401c310a79979aa46307dad19",
            "Reviewed prediction policy differs from the original scientific engine")
    return {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
            for name, path in paths.items()}


def register_plan(directory):
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "New exploratory specification after benchmark, earlier descriptive intervals and CPU diagnostics; before this new inferential calculation",
        "script_sha256": sha(__file__), "sources": sources(),
        "models": list(NEURAL), "seeds": list(SEEDS), "calibration_dates": 53, "test_dates": 608,
        "effect": "per-seed RMSE_test608 minus RMSE_calibration53 of the same original frozen predictor; then mean across six seeds",
        "H0": "mean seed difference=0", "H1": "mean seed difference!=0",
        "primary": "two-sided Student t of six differences,df5; marginal95percent t CI; separate Holm13 family at0.05",
        "sensitivity": "two-sided64exact sign assignments of six original differences; abs(mean) statistic, no MonteCarlo+1; separate Holm13 family at0.05",
        "assumptions": ["Independent optimization repetitions conditional on the fixed calibration/test calendars.",
                        "Adequate small-sample normal approximation of six differences for t; not normality of daily residuals.",
                        "Sign-flip sensitivity additionally requires symmetry under the null; enumeration does not verify symmetry.",
                        "Shared calendars, overlapping target supports and temporal dependence are not treated as independent observations."],
        "zero_seed_variance_policy": "t,CI andp not estimable; never fabricate zero uncertainty or p0",
        "classical_policy": "three single trajectories described by period risk/difference only; no seed t/IC or invented repetitions",
        "decisions": "Reject only with valid t p_Holm13<=0.05; sign-flips remain sensitivity, not a replacement selected by smaller p",
        "limits": ["Calibration53 was used for pilot and interval calibration; it is not independent HPO validation.",
                   "A period difference may reflect distribution change, period difficulty or estimation, not identified overfitting.",
                   "Non-rejection does not demonstrate consistency, equivalence or universal absence of overfitting.",
                   "No new model fits, inference, CPU/GPU mixing, new holdout or original preregistration is claimed.",
                   "The original export's predictor/calibrator/checkpoint verification is inherited from its pinned bundle; raw market inputs are not reacquired here."],
        "new_fits": 0, "new_model_inference": False, "A2_or_RQ1_changed": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__},
    }
    write_json(directory / "analysis_plan.json", plan)
    return {"plan": str(directory / "analysis_plan.json"), "sha256": sha(directory / "analysis_plan.json")}


def admit(payload, freeze):
    test_dates = [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)]
    cal_dates = [(date(2024, 11, 8)+timedelta(days=i)).isoformat() for i in range(53)]
    require(payload["dates"] == test_dates and payload["mask"] == [True]*608, "Wrong test calendar/mask")
    truth = np.asarray(payload["y_true"], float)
    require(truth.shape == (608,) and np.isfinite(truth).all(), "Nonfinite or missing original test targets")
    expected = {(m, s) for m in NEURAL for s in SEEDS} | {(m, "deterministic") for m in CLASSICAL}
    frozen = {(r["model"], r["seed"]): r["checkpoint_sha256"] for r in freeze["models"]}
    require(set(frozen) == expected and len(freeze["models"]) == 81, "Original81-predictor freeze incomplete")
    indexed, common_calibration = {}, None
    for row in payload["predictors"]:
        key = row["model"], row["seed"]
        require(key in expected and key not in indexed, "Unknown/duplicate original trajectory")
        require(row["checkpoint_sha256"] == frozen[key], "Calibration/test checkpoint does not match original freeze")
        require(row["calibration_dates"] == cal_dates, "Wrong calibration dates")
        for name, size in (("prediction_original", 608), ("calibration_predictions", 53), ("calibration_targets", 53)):
            values = np.asarray(row[name], float)
            require(values.shape == (size,) and np.isfinite(values).all(), "Invalid original period trajectory")
        if common_calibration is None: common_calibration = row["calibration_targets"]
        require(row["calibration_targets"] == common_calibration, "Calibration target vector differs between predictors")
        indexed[key] = row
    require(set(indexed) == expected, "All81 original predictors required")
    provenance = payload["provenance"]
    require(value_sha(payload["y_true"]) == provenance["targets_sha256"]
            and value_sha(common_calibration) == provenance["calibration_targets_sha256"], "Original target hashes changed")
    require(provenance["all_81_original_seed_RMSE_verified"] is True, "Original exporter verification absent")
    return indexed


def t_period(differences):
    d = np.asarray(differences, float)
    require(d.shape == (6,) and np.isfinite(d).all(), "Six finite paired seed differences required")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if np.ptp(d) == 0 or sd == 0:
        return {"difference_mean": mean, "SD_seed_differences": sd, "t": None, "df": 5,
                "p_raw": None, "CI95_lower": None, "CI95_upper": None, "status": "not_estimable_zero_seed_variance"}
    se = sd/np.sqrt(6); statistic = mean/se; radius = float(stats.t.ppf(.975, 5))*se
    return {"difference_mean": mean, "SD_seed_differences": sd, "t": float(statistic), "df": 5,
            "p_raw": float(2*stats.t.sf(abs(statistic), 5)), "CI95_lower": mean-radius,
            "CI95_upper": mean+radius, "status": "estimable_conditional_small_sample_t"}


def sign_flip(differences):
    d = np.asarray(differences, float)
    require(d.shape == (6,) and np.isfinite(d).all(), "Six finite differences required")
    values = np.asarray(list(itertools.product((-1, 1), repeat=6)), float) @ d/6
    observed = abs(float(d.mean()))
    tolerance = 16*np.finfo(float).eps*max(float(np.max(np.abs(d))), np.finfo(float).tiny)
    # Include numerically equal boundary ties, not only bit-identical sums.
    return {"mean_difference": float(d.mean()), "p_raw": float(np.count_nonzero(np.abs(values) >= observed-tolerance)/64),
            "assignments": 64, "scope": "conditional seed sensitivity; symmetry under the null assumed, not established"}


def holm13(rows):
    require(len(rows) == 13, "This period family contains13 neural models")
    ordered = sorted((1. if row["p_raw"] is None else row["p_raw"], i) for i, row in enumerate(rows))
    previous = 0.
    for rank, (value, i) in enumerate(ordered):
        previous = max(previous, min(1., (13-rank)*value))
        rows[i]["p_Holm13"] = previous if rows[i]["p_raw"] is not None else None
        rows[i]["reject_FWER05"] = rows[i]["p_raw"] is not None and previous <= .05
    return rows


def period_risks(row, y):
    calibration = float(np.sqrt(np.mean((np.asarray(row["calibration_predictions"])-np.asarray(row["calibration_targets"]))**2)))
    test = float(np.sqrt(np.mean((np.asarray(row["prediction_original"])-y)**2)))
    return calibration, test


def execute(directory):
    started = datetime.now(timezone.utc).isoformat()
    plan_path = directory / "analysis_plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["version"] == VERSION and plan["script_sha256"] == sha(__file__), "Registered code changed")
    require(plan["sources"] == sources(), "Registered source bytes changed")
    require(not (directory / "execution_receipt.json").exists(), "Preserve completed calculations")
    payload = json.loads(BUNDLE.read_text())
    freeze = json.loads((ROOT / plan["sources"]["freeze"]["path"]).read_text())
    indexed = admit(payload, freeze)
    y = np.asarray(payload["y_true"])
    seed_rows, primary, sensitivity, classical = [], [], [], []
    for model in (*NEURAL, *CLASSICAL):
        members = SEEDS if model in NEURAL else ("deterministic",)
        calibration_risks, test_risks = [], []
        for seed in members:
            row = indexed[model, seed]
            calibration, test = period_risks(row, y)
            calibration_risks.append(calibration); test_risks.append(test)
            seed_rows.append({"model": model, "seed": seed, "checkpoint_sha256": row["checkpoint_sha256"],
                "RMSE_calibration53": calibration, "RMSE_test608": test,
                "difference_test_minus_calibration": test-calibration,
                "ratio_test_over_calibration": test/calibration if calibration > 0 else None,
                "calibration_dates": 53, "test_dates": 608, "source": "original_saved_predictor_bundle"})
        if model in NEURAL:
            differences = np.asarray(test_risks)-np.asarray(calibration_risks)
            primary.append({"model": model, "mean_RMSE_calibration53": float(np.mean(calibration_risks)),
                "mean_RMSE_test608": float(np.mean(test_risks)), **t_period(differences),
                "n_seed_pairs": 6, "direction": "positive means higher test RMSE; no cause attributed"})
            sensitivity.append({"model": model, **sign_flip(differences)})
        else:
            classical.append({"model": model, "RMSE_calibration53": calibration_risks[0],
                "RMSE_test608": test_risks[0], "difference_test_minus_calibration": test_risks[0]-calibration_risks[0],
                "n_predictor_trajectories": 1, "p_raw": None, "CI_seed": None,
                "status": "descriptive_only_no_stochastic_seed_replicates"})
    holm13(primary); holm13(sensitivity)
    phases = list(csv.DictReader((ROOT / plan["sources"]["phase_resources"]["path"]).open()))
    final = [row for row in phases if row["stage"] == "final"]
    backend = {key: sorted({row[key] for row in final}) for key in ("device_name", "torch", "precision", "amp", "tf32", "cpu_threads")}
    phase_pairs = {(row["model"], int(row["seed"])) for row in final}
    require(phase_pairs == {(model, seed) for model in NEURAL for seed in SEEDS},
            "Recorded original neural runtime metadata incomplete")
    backend.update({"metadata_scope": "recorded original final-fit phase metadata; no new hardware observation",
                    "neural_predictors_with_original_phase_metadata": len(phase_pairs),
                    "prediction_policy_from_engine": "eval mode, float32 normalized tensors, batch1, same frozen scalers; original engine source recorded byhash",
                    "checkpoint_identity_verified_for81_bundle_rows": True,
                    "exporter_verified_original_calibration_same_predictor_as_final": True,
                    "exporter_verification_inherited_from_pinned_bundle_not_reexecuted_against_missing_raw_market": True})
    write_csv(directory / "period_differences_t_Holm13.csv", primary)
    write_csv(directory / "period_sign_flip_sensitivity_Holm13.csv", sensitivity)
    write_csv(directory / "period_risks_81.csv", seed_rows)
    write_csv(directory / "classical_period_differences_descriptive.csv", classical)
    write_json(directory / "source_admission.json", backend)
    summary = {"version": VERSION, "neural_period_tests": 13, "classical_descriptive_rows": 3,
        "t_rejections_Holm13": sum(row["reject_FWER05"] for row in primary),
        "t_rejected_models": [row["model"] for row in primary if row["reject_FWER05"]],
        "t_not_estimable": sum(row["p_raw"] is None for row in primary),
        "sign_flip_rejections_Holm13": sum(row["reject_FWER05"] for row in sensitivity),
        "sign_flip_minimum_two_sided_resolution_no_zero_ties": 2/64,
        "family_size": 13, "alpha_family": .05,
        "scope": "original saved calibration/test risks, same predictor; initialization uncertainty on fixed periods",
        "calibration_is_not_independent_HPO_validation": True, "equivalence_established": False,
        "overfitting_cause_identified": False, "CPU_GPU_mixed": False,
        "new_model_fits": 0, "new_model_inference": False, "A2_RQ1_modified": False,
        "limits": plan["limits"], "assumptions": plan["assumptions"]}
    write_json(directory / "summary.json", summary)
    require(plan["sources"] == sources(), "Source bytes changed during calculation")
    write_json(directory / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(plan_path),
        "script_sha256": sha(__file__), "input_sources": sources(), "new_fits": 0, "new_inference": False})
    (directory / "README.md").write_text("""# Comparación de los periodos originales de calibración y prueba

Análisis exploratorio registrado después del benchmark y antes de este nuevo
cálculo. Usa exclusivamente las trayectorias originales guardadas, emparejadas
por modelo, semilla y checkpoint: RMSE_test608 − RMSE_calibración53.

La familia principal aplica t bilateral a seis diferencias por red (5 gl), IC95 %
marginal y Holm13. Se condiciona a ambos calendarios y a la aproximación de las
diferencias entre inicializaciones. No se emparejan fechas ni se tratan como iid.
La media de RMSE individuales no se sustituye por la RMSE de un ensemble.

La sensibilidad enumera 64 asignaciones de signos bajo simetría, con familia Holm13
separada. Su resolución bilateral mínima 2/64=0,03125 no permite rechazo tras Holm13
a 0,05; no se escoge el procedimiento con p menor. Los tres clásicos R=1 permanecen
descriptivos: no se inventan seis semillas ni IC de inicialización para ellos.

La calibración ya se usó para piloto e intervalos y NO se renombra validación HPO
independiente. Una diferencia de periodos no identifica sobreajuste; no rechazo
no demuestra consistencia ni equivalencia. No se ejecuta el antiguo plan714 ni
se mezclan riesgos CPU/GPU. No hay nuevos ajustes, predicciones o cambios en A2/RQ1.

Las identidades y verificaciones del exportador original se heredan del JSON
fijado por SHA; sus comprobaciones sobre los brutos originales no se repiten aquí.
La política eval/FP32/batch1 y los metadatos originales disponibles se documentan
en source_admission.json, sin fingir una observación nueva del hardware.

Repetición desde el repositorio con NumPy/SciPy y una carpeta nueva:

```sh
python scripts/e2_original_period_comparison.py --output /tmp/e2-period-repeat --register-plan
python scripts/e2_original_period_comparison.py --output /tmp/e2-period-repeat --execute
```
""", encoding="utf-8")
    write_json(directory / "manifest.json", {"version": VERSION, "files": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(directory.iterdir()) if path.is_file()}, "new_fits": 0})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = register_plan(args.output) if args.register_plan else execute(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
