"""Package completed F2a evidence and figures; never repeat statistical tests.

Original result CSVs and archive members retain their exact bytes. The public
directories contain derived evidence only; cached market data and model weights
are neither read nor copied by this packager.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-f2a-public-evidence-v1"
INPUT_COMMIT = "2121b048d3b379c2c28dfe366cba35a1c08a3617"
INPUT_SHA256 = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
INPUT_PATH = "runs/f2a-65ae74b547a6073b2c286dea-posthoc/artifacts/e2_f2a_input.json"
INPUT_MANIFEST_PATH = "runs/f2a-65ae74b547a6073b2c286dea-posthoc/manifest.json"
ORIGINAL_BENCHMARK_COMMIT = "265ad10738423fac7e0d185f725de09c5f627ec6"
ORIGINAL_ANALYSIS_SHA256 = "73206e1aa5bf004b7e0f884ba3685c0bdfbed209425c6397a68fa280c04ff637"
ANALYSIS_CODE_SHA256 = "466d0d53a6540f2f83f367a50e637b1a539a6beab010c410b4f6290d461c9861"
STATS_CSV = ("tests.csv", "system_metrics.csv", "quarterly_metrics.csv", "coverage_intervals.csv",
             "calibration_final_comparison.csv", "original_rmse_checks.csv", "rolling90_metrics.csv")
TRAINING_FILES = ("training_curves.csv", "auxiliary_curves.csv", "fit_overfit_summary.csv", "source_evidence.json")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def deterministic_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, raw)
    return stream.getvalue()


def summarize_tests(audit):
    rows = audit["tests"]
    names = list(dict.fromkeys(row["test"] for row in rows))
    by_test = []
    for name in names:
        selected = [row for row in rows if row["test"] == name]
        rejected = [row for row in selected if row["reject"]]
        over = sum(row.get("coverage90", .9) > .9 for row in rejected) if name == "Coverage90_HAC" else 0
        under = sum(row.get("coverage90", .9) < .9 for row in rejected) if name == "Coverage90_HAC" else 0
        direction = ("overcoverage only" if over and not under else "coverage differs from 0.90") if name == "Coverage90_HAC" else {
            "Mincer_Zarnowitz": "joint intercept/slope departure; coefficients are descriptive",
            "Calendar_quarter_loss_stability": "no adjusted rejection; does not establish stability",
            "Jarque_Bera": "departure from normality; nominal under serial dependence",
        }.get(name, "residual dependence or conditional heteroskedasticity; nominal assumptions apply")
        by_test.append({"test": name, "procedures": len(selected),
            "estimable": sum(row["status"] == "estimable" for row in selected),
            "reject_Holm826": len(rejected), "do_not_reject_Holm826": len(selected)-len(rejected),
            "overcoverage_rejections": over, "undercoverage_rejections": under,
            "raw_p_numeric_underflow_zeros": sum(row["p_raw"] == 0 for row in selected),
            "direction_and_limit": direction})
    by_model = []
    for metric in audit["system_metrics"]:
        name = metric["model"]
        selected = [row for row in rows if row["model"] == name]
        mz = [row for row in selected if row["test"] == "Mincer_Zarnowitz"]
        coverage = [row for row in selected if row["test"] == "Coverage90_HAC"]
        quarter = next(row for row in selected if row["test"] == "Calendar_quarter_loss_stability")
        by_model.append({"model": name, "R": metric["R"], "procedures": len(selected),
            "reject_Holm826": sum(row["reject"] for row in selected),
            "Mincer_Zarnowitz_rejections": sum(row["reject"] for row in mz),
            "MZ_intercept_mean_descriptive": sum(row["intercept"] for row in mz)/len(mz),
            "MZ_slope_mean_descriptive": sum(row["slope"] for row in mz)/len(mz),
            "MZ_direction": "joint null intercept=0/slope=1; coefficients are not separate adjusted tests",
            "coverage90_mean": metric["coverage90"],
            "coverage90_min": min(row["coverage90"] for row in coverage),
            "coverage90_max": max(row["coverage90"] for row in coverage),
            "overcoverage_rejections": sum(row["reject"] and row["coverage90"] > .9 for row in coverage),
            "undercoverage_rejections": sum(row["reject"] and row["coverage90"] < .9 for row in coverage),
            "quarter_stability_statistic": quarter["statistic"],
            "quarter_stability_p_raw": quarter["p_raw"],
            "quarter_stability_p_Holm826": quarter["p_holm"],
            "quarter_stability_reject": quarter["reject"],
            "quarter_direction": "seven calendar quarters; 2026Q3 partial; non-rejection is not stability"})
    if (len(by_test), sum(row["procedures"] for row in by_test), sum(row["reject_Holm826"] for row in by_test)) != (11, 826, 707):
        raise ValueError("Actual diagnostic counts differ from the reviewed 11/826/707 evidence")
    if len(by_model) != 16 or sum(row["procedures"] for row in by_model) != 826:
        raise ValueError("Actual model diagnostic summary does not cover all 16 systems")
    return by_test, by_model


def prepare_package(evidence_root, plan_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", plan_commit):
        raise ValueError("The complete immutable pre-data post hoc plan commit is required")
    evidence_root = Path(evidence_root)
    stats_root, training_root = evidence_root / "statistical_results", evidence_root / "training_evidence"
    raw_input = (evidence_root / "e2_f2a_input.json").read_bytes()
    if sha(raw_input) != INPUT_SHA256:
        raise ValueError("The completed immutable derived input identity changed")
    inputs = json.loads(raw_input)
    input_manifest_raw = (evidence_root / "input_manifest.json").read_bytes()
    input_manifest = json.loads(input_manifest_raw)
    if (len(input_manifest["artifacts"]) != 1 or input_manifest["artifacts"][0]["sha256"] != INPUT_SHA256
            or input_manifest["artifacts"][0]["remote_path"] != INPUT_PATH
            or input_manifest["artifacts"][0]["bytes"] != len(raw_input)):
        raise ValueError("The immutable derived input manifest changed")
    original_root = ROOT / "experiments/e2_benchmark_final_20261113"
    original_raw = (original_root / "benchmark_analysis.json").read_bytes()
    if sha(original_raw) != ORIGINAL_ANALYSIS_SHA256 or inputs["original_analysis"] != json.loads(original_raw):
        raise ValueError("The exported original analysis differs from the frozen benchmark")
    analysis_code = ROOT / "scripts/e2_f2a_posthoc_audit.py"
    if sha(analysis_code.read_bytes()) != ANALYSIS_CODE_SHA256:
        raise ValueError("The actual executed statistical analysis code changed")
    stats_files = {name: (stats_root / name).read_bytes() for name in ("audit.json", *STATS_CSV)}
    training_files = {name: (training_root / name).read_bytes() for name in TRAINING_FILES}
    audit = json.loads(stats_files["audit.json"])
    training = json.loads(training_files["source_evidence.json"])
    if (audit["input_sha256"] != INPUT_SHA256 or audit["family"]["size"] != 826
            or audit["fits_performed"] != 0 or audit["model_inference_calls"] != 0
            or audit["original_analysis_modified"] or audit["original_twenty_comparisons_repeated"]):
        raise ValueError("The completed audit does not match the read-only scope")
    for name, receipt in training["outputs"].items():
        if name not in training_files or (len(training_files[name]), sha(training_files[name])) != (receipt["bytes"], receipt["sha256"]):
            raise ValueError("A collected original training evidence file changed")
    if (training["training_performed"] or training["model_inference_performed"]
            or training["market_payload_loaded"] or training["weight_bytes_downloaded"]
            or training["final_fit_count"] != 78 or training["selected_hpo_fit_count"] != 39):
        raise ValueError("Original training metadata no longer matches its reviewed scope")
    by_test, by_model = summarize_tests(audit)
    heavy = {"tests", "system_metrics", "quarterly_metrics", "rolling90_metrics", "coverage_intervals",
             "calibration_final_comparison", "original_rmse_checks", "input_provenance"}
    summary = {key: value for key, value in audit.items() if key not in heavy}
    rejected_coverage = [r for r in audit["tests"] if r["test"] == "Coverage90_HAC" and r["reject"]]
    gaps = [r["final_minus_calibration_RMSE"] for r in audit["calibration_final_comparison"]]
    summary.update(tests=826, rejected_Holm=707, estimable=826, non_estimable=0,
        diagnostics_by_test=by_test,
        original_primary_comparisons_unchanged=20,
        coverage_rejections={"count": len(rejected_coverage), "direction": "overcoverage only",
            "coverage_min": min(r["coverage90"] for r in rejected_coverage),
            "coverage_max": max(r["coverage90"] for r in rejected_coverage)},
        calibration_final_descriptive={"count": len(gaps), "increases": sum(g > 0 for g in gaps),
            "decreases": sum(g < 0 for g in gaps), "delta_g": None,
            "absence_of_overfitting_established": False},
        quarterly_rejections=0, quarterly_non_rejection_establishes_stability=False,
        numeric_underflow={"Jarque_Bera_raw_p_zeros": sum(r["test"] == "Jarque_Bera" and r["p_raw"] == 0 for r in audit["tests"]),
            "display_note": "29 raw Jarque–Bera p-values underflow to zero; describe as p < 1e-300; preserve original zeros in raw files"})
    files = {name: stats_files[name] for name in STATS_CSV if name != "rolling90_metrics.csv"}
    files.update({"audit_summary.json": json_bytes(summary),
        "diagnostics_by_test.csv": csv_bytes(by_test), "diagnostics_by_model.csv": csv_bytes(by_model),
        "fit_overfit_summary.csv": training_files["fit_overfit_summary.csv"],
        "statistical_results.zip": deterministic_zip(stats_files),
        "training_evidence.zip": deterministic_zip(training_files)})
    from plot_e2_f2a_public_evidence import generate_figures
    with tempfile.TemporaryDirectory(prefix="e2-f2a-public-figures-") as temporary:
        figure_root = Path(temporary)
        generate_figures(evidence_root / "e2_f2a_input.json", stats_root, training_root, figure_root)
        for path in sorted(figure_root.iterdir()):
            if path.is_file():
                files[path.name] = path.read_bytes()
    benchmark_sources = json.loads((original_root / "source_evidence.json").read_bytes())
    original_primary = inputs["original_analysis"]["primary_contrasts"]
    if len(original_primary) != 20 or not all(row["p_Holm"] == 1 for row in original_primary):
        raise ValueError("The frozen original Holm20 comparisons changed")
    code_files = ("scripts/build_e2_f2a_public_evidence.py", "scripts/plot_e2_f2a_public_evidence.py",
                  "scripts/e2_f2a_posthoc_audit.py", "scripts/collect_e2_f2a_training_evidence.py")
    source = {"version": VERSION, "repository": "jassan-bit/DeepLearning_202630",
        "input": {"immutable_commit": INPUT_COMMIT, "path": INPUT_PATH,
            "url": "https://github.com/jassan-bit/DeepLearning_202630/blob/"+INPUT_COMMIT+"/"+INPUT_PATH,
            "sha256": INPUT_SHA256, "bytes": len(raw_input), "manifest_path": INPUT_MANIFEST_PATH,
            "manifest_url": "https://github.com/jassan-bit/DeepLearning_202630/blob/"+INPUT_COMMIT+"/"+INPUT_MANIFEST_PATH,
            "manifest_sha256": sha(input_manifest_raw)},
        "post_hoc_plan": {"immutable_commit": plan_commit, "published_before_derived_input_access": True,
            "path": "experiments/e2_f2a_audit_20261113/plan.json",
            "sha256": sha((ROOT / "experiments/e2_f2a_audit_20261113/plan.json").read_bytes())},
        "original_benchmark": {"immutable_commit": ORIGINAL_BENCHMARK_COMMIT,
            "analysis_file_sha256": ORIGINAL_ANALYSIS_SHA256,
            "analysis_canonical_sha256": sha(canonical(inputs["original_analysis"])),
            "exported_original_analysis_exact_object_equality_verified": True,
            "primary_comparisons": 20, "all_original_primary_p_Holm_equal_one": True,
            "delta": original_primary[0]["delta"], "primary_comparisons_repeated": False,
            "original_files_modified": False,
            "control_state_id": benchmark_sources["control_state_id"],
            "control_head_artifact_sha256": benchmark_sources["source_control_head_artifact_sha256"],
            "terminal_budget_event_file_sha256": sha((original_root / "terminal_budget_event.json").read_bytes()),
            "control_pointer_modified": False, "budget_extended": False},
        "statistical_analysis": {"code_sha256": ANALYSIS_CODE_SHA256,
            "input_sha256": audit["input_sha256"], "Holm_family_size": 826,
            "rejected": 707, "raw_files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in stats_files.items()}},
        "training_evidence": {"source_evidence_sha256": sha(training_files["source_evidence.json"]),
            "original_metadata_results_commit": training["immutable_results_commit"],
            "final_fit_count": 78, "selected_hpo_fit_count": 39,
            "selected_hpo_full_curves": 29, "selected_hpo_missing_curves": 10,
            "supervised_epoch_rows": 6440, "auxiliary_epoch_rows": 3250,
            "raw_files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in training_files.items()},
            "raw_collected_source_metadata_copied_to_website": False},
        "execution_scope": {"scientific_fits": 0, "model_inference_calls": 0,
            "calibrator_refits": 0, "raw_OHLCV_copied": False, "weights_copied": False,
            "wall_time_measurement": "CPU wall time was not instrumented; no elapsed-time claim"},
        "CPU_runtime": {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")}},
        "source_code_sha256": {name: sha((ROOT / name).read_bytes()) for name in code_files},
        "focused_validation": {"statistical_tests": 14, "training_metadata_tests": 18, "cached_export_tests": 11,
            "status": "all passed before packaging"},
        "interpretation_limits": audit["scope_limits"]}
    files["source_evidence.json"] = json_bytes(source)
    manifest = {"version": VERSION, "scope": "new post hoc artifact directories only",
        "manifest_excludes_itself": True, "original_primary_artifact_directory_modified": False,
        "mirrors_byte_identical": True, "input_copied_to_website": False,
        "files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(files.items())}}
    files["manifest.json"] = json_bytes(manifest)
    return files


def write_mirrors(files):
    destinations = (ROOT / "experiments/e2_f2a_audit_20261113/results",
                    ROOT / "book_e2/_static/e2_f2a_audit_20261113")
    for destination in destinations:
        if destination.exists() and any(path.name not in files for path in destination.iterdir()):
            raise ValueError("An unexpected existing public artifact must be reconciled explicitly")
        for name, raw in files.items():
            path = destination / name
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise ValueError("Public artifact directories must not use symlinks")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("An existing public artifact differs: " + name)
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            path = destination / name
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(raw)
    for name, raw in files.items():
        if any((destination / name).read_bytes() != raw for destination in destinations):
            raise ValueError("The public evidence mirrors differ")
    return destinations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--plan-commit", required=True)
    args = parser.parse_args(argv)
    files = prepare_package(args.evidence_root, args.plan_commit)
    destinations = write_mirrors(files)
    print(json.dumps({"files_per_mirror": len(files), "bytes_per_mirror": sum(map(len, files.values())),
        "destinations": [str(path) for path in destinations], "mirrors_byte_identical": True}))


if __name__ == "__main__":
    main()
