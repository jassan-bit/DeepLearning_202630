"""Verify and describe saved V3 development outputs; no model imports or fitting.

Prepared with declared AI assistance. The post hoc diagnostics in this file do
not change the predeclared selection/continuation rule, estimate prospective
power, compute significance tests, or certify R1/R3. NumPy is used for arithmetic;
Matplotlib is optional and renders a static scientific figure only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

RUN = "20261120T031424176339Z"
SOURCE = "8f65eac34aaf49293924ae8328c235cedbdd2866"
RESULTS = "8eaa581bc34b615a8507c9e384a3e4be29084720"
NEURAL = ("additive_positive_head", "atom_crps", "atom_mse_only", "fixed_probs", "atom_plugin")
ANALYTICAL = ("floor_only", "floor_plus_mean_excess")
RECIPES = NEURAL + ANALYTICAL
SEEDS = ("20261118", "20261119", "20261120")
LABELS = {"additive_positive_head": "Aditiva", "atom_crps": "V3 completa (CRPS)",
          "atom_mse_only": "Sin CRPS", "fixed_probs": "Masas fijas (1236 p.)",
          "atom_plugin": "Segundo momento", "floor_only": "Sólo piso",
          "floor_plus_mean_excess": "Piso + exceso de ajuste"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), "Cannot write an empty diagnostic table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected, what):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-14), what)


def verify_archive(root):
    archive = root / "archive"
    provenance = read_json(root / "source_objects.json")
    require(provenance["results_commit"] == RESULTS, "Results commit identity differs")
    verified = []
    for name, entry in provenance["files"].items():
        raw = (archive / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(blob == entry["git_blob_sha"], "Git blob transport mismatch: " + name)
        verified.append({"path": name, "bytes": len(raw), "sha256": sha(raw), "git_blob_sha": blob})
    manifest = read_json(archive / "snapshot_manifest.json")
    require(manifest["source_commit"] == SOURCE and manifest["run_id"] == RUN
            and manifest["sequence"] == 93, "Snapshot identity differs")
    for entry in verified:
        if entry["path"].startswith("payload/"):
            name = entry["path"][len("payload/"):]
            expected = manifest["files"][name]
            require(expected["bytes"] == entry["bytes"] and expected["sha256"] == entry["sha256"],
                    "Snapshot payload mismatch: " + name)
    payload = archive / "payload"
    plan, receipt = read_json(payload / "plan.json"), read_json(payload / "pilot_receipt.json")
    state, admission = read_json(payload / "state.json"), read_json(payload / "admission.json")
    selection, fit_complete = read_json(payload / "closed_selection.json"), read_json(payload / "all_fits_complete.json")
    verdict, published = read_json(payload / "development_verdict.json"), read_json(payload / "publication_receipt.json")
    digest = sha(canonical(plan))
    require(plan["master_run_id"] == RUN and plan["identity"]["source_commit"] == SOURCE,
            "Pilot plan identity differs")
    require(receipt["plan_sha256"] == digest == fit_complete["plan_sha256"] == selection["plan_sha256"]
            == published["plan_sha256"], "Sealed plan digest differs")
    require(plan["admission_sha256"] == sha(canonical(admission)) == published["admission_sha256"],
            "Admission digest differs")
    require(state["pilot_receipt_sha256"] == sha((payload / "pilot_receipt.json").read_bytes()),
            "Final receipt digest differs")
    require(fit_complete["closed_selection_sha256"] == sha(canonical(selection)), "Closed selection digest differs")
    for key, filename in (("hardware_profile_sha256", "hardware_profile.json"),
                          ("launcher_lock_sha256", "launcher_lock.json"),
                          ("launcher_identity_sha256", "launcher_identity.json")):
        require(plan[key] == sha((payload / filename).read_bytes()), "Launcher binding differs: " + filename)
    require(plan["neural_variants"] == list(NEURAL) and plan["analytical_controls"] == list(ANALYTICAL),
            "Predeclared recipes changed")
    require(plan["total_required_fits"] == receipt["required_fits"] == receipt["completed_fits"]
            == state["completed_fits"] == 30, "Thirty completed fits required")
    require(fit_complete["HPO_fits"] == fit_complete["final_fits"] == 15, "HPO/final fit split differs")
    require(receipt["status"] == state["status"] == "completed_development_only", "Pilot is not complete")
    require(receipt["market_calibration_arrays_opened"] == receipt["market_test_arrays_opened"]
            == receipt["additional_datasets_used"] == 0, "Pilot data scope changed")
    require(admission["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
            and not admission["forbidden_arrays_loaded"], "Training-only admission differs")
    require([plan[k] for k in ("fit_rows", "selection_rows", "purged_rows", "assessment_rows")]
            == [532, 54, 7, 55], "Development split differs")
    require((date.fromisoformat(admission["assessment_dates"][0])
             - date.fromisoformat(admission["selection_dates"][-1])).days >= 8,
            "Selection/assessment raw supports overlap")
    for name, fingerprint in receipt["artifact_sha256"].items():
        require(sha((payload / name).read_bytes()) == fingerprint, "Pilot artifact differs: " + name)
    resources = receipt["fit_resource_receipts"]
    require(len(resources) == 30 and all(r["status"] == "completed" and r["optimizer_steps"] > 0
            and r["best_validation_mse_scaled"] is not None and not r["untrained_checkpoint_accepted"]
            for r in resources.values()), "Incomplete or untrained phase admitted")
    require(all(r["budget_seconds"] == 120 and r["deadline_includes_restart_downtime"]
                for r in resources.values()), "Immutable phase budget differs")
    require(not any(receipt[k] for k in ("p_values_computed", "R1_accredited", "R3_accredited",
                                        "confirmatory_claim_allowed")), "Pilot claim scope changed")
    require(all(receipt[k] == verdict[k] for k in verdict), "Published verdict differs from final receipt")
    return payload, plan, receipt, admission, selection, verified, len(manifest["files"])


def metric(prediction, target):
    error = prediction - target
    return {"RMSE": float(np.sqrt(np.mean(error**2))), "MAE": float(np.mean(np.abs(error))),
            "bias_prediction_minus_target": float(np.mean(error))}


def correlation(left, right):
    if float(np.std(left)) == 0. or float(np.std(right)) == 0.:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze(root, output, figures=True):
    payload, plan, receipt, admission, selection, inventory, full_inventory_count = verify_archive(root)
    raw = read_csv(payload / "assessment_predictions.csv")
    final = read_csv(payload / "final_assessment_scores.csv")
    hpo = read_csv(payload / "HPO_scores.csv")
    analytical = read_csv(payload / "analytical_controls.csv")
    archived_summary = read_csv(payload / "recipe_summary.csv")
    require(len(raw) == 935 and len(final) == len(hpo) == 15 and len(analytical) == 2
            and len(archived_summary) == 5, "Saved assessment geometry differs")
    dates = admission["assessment_dates"]
    require(len(dates) == 55 and dates == sorted(set(dates)), "Assessment calendar differs")
    require(dates == [(date.fromisoformat(dates[0]) + timedelta(days=i)).isoformat() for i in range(55)],
            "Assessment calendar is not consecutive")
    trajectories = {}
    target = None
    for name in RECIPES:
        expected_seeds = SEEDS if name in NEURAL else ("analytic",)
        for seed in expected_seeds:
            rows = sorted((r for r in raw if r["name"] == name and r["seed"] == seed), key=lambda r: r["date"])
            require([r["date"] for r in rows] == dates, "Misaligned trajectory: " + name + "/" + seed)
            y = np.array([float(r["target"]) for r in rows])
            p = np.array([float(r["prediction"]) for r in rows])
            require(np.isfinite(y).all() and np.isfinite(p).all(), "Nonfinite saved assessment")
            target = y if target is None else target
            require(np.array_equal(y, target), "Targets differ between recipes/seeds")
            trajectories[name, seed] = p
    require(len(trajectories) == 17 and set(r["name"] for r in raw) == set(RECIPES), "Extra/missing recipe")
    floor = trajectories["floor_only", "analytic"]
    require(all(np.all(p + 1e-12 >= floor) for p in trajectories.values()), "Saved forecast violates common floor")
    d_fit = admission["scaler"]["d_fit"]
    require(np.allclose(trajectories["floor_plus_mean_excess", "analytic"] - floor, d_fit,
                        rtol=1e-10, atol=1e-14), "Recorded fit-only excess offset differs")
    for name in NEURAL:
        trials = [r for r in hpo if r["name"] == name]
        require(len(trials) == 3 and {int(r["seed"]) for r in trials} == {plan["hpo_seed"]}
                and sorted(float(r["learning_rate"]) for r in trials) == plan["hpo_learning_rates"],
                "Incomplete declared HPO: " + name)
        best = min(trials, key=lambda r: (float(r["selection_RMSE"]), float(r["learning_rate"])))
        close(selection["selected_learning_rates"][name], best["learning_rate"], "Wrong frozen HPO choice")
    per_seed, residual, monthly, acf = [], [], [], []
    for (name, seed), prediction in trajectories.items():
        score = metric(prediction, target)
        error = prediction - target
        row = {"name": name, "seed": seed, "dates": 55, **score}
        per_seed.append(row)
        source_row = next(r for r in (final if name in NEURAL else analytical)
                          if r["name"] == name and (name not in NEURAL or r["seed"] == seed))
        close(score["RMSE"], source_row["RMSE"], "Recomputed RMSE differs")
        close(score["MAE"], source_row["MAE"], "Recomputed MAE differs")
        residual.append({**row, "median_error": float(np.median(error)),
                         "forecast_below_target_fraction": float(np.mean(error < 0)),
                         "absolute_error_q95_empirical": float(np.quantile(np.abs(error), .95)),
                         "residual_lag1_correlation": correlation(error[:-1], error[1:]),
                         "residual_lag7_correlation": correlation(error[:-7], error[7:])})
        for lag in range(1, 8):
            acf.append({"name": name, "seed": seed, "lag": lag, "pairs": 55-lag,
                        "residual_correlation": correlation(error[:-lag], error[lag:])})
        for month in sorted(set(day[:7] for day in dates)):
            indices = np.array([i for i, day in enumerate(dates) if day.startswith(month)])
            monthly.append({"name": name, "seed": seed, "month": month, "dates": len(indices),
                            "target_mean": float(target[indices].mean()),
                            **metric(prediction[indices], target[indices])})
    summary = []
    for name in RECIPES:
        scores = [r for r in per_seed if r["name"] == name]
        row = {"name": name, "kind": "neural" if name in NEURAL else "analytical",
               "trajectories": len(scores), "dates_per_trajectory": 55,
               "mean_individual_RMSE": float(np.mean([r["RMSE"] for r in scores])),
               "SD_individual_RMSE": float(np.std([r["RMSE"] for r in scores], ddof=1)) if len(scores)>1 else "",
               "mean_individual_MAE": float(np.mean([r["MAE"] for r in scores])),
               "mean_bias_prediction_minus_target": float(np.mean([r["bias_prediction_minus_target"] for r in scores])),
               "source_mean_CRPS_U_not_recomputed": ""}
        if name in NEURAL:
            archived = next(r for r in archived_summary if r["name"] == name)
            for actual, key in ((row["mean_individual_RMSE"], "mean_RMSE"),
                                (row["SD_individual_RMSE"], "SD_RMSE"),
                                (row["mean_individual_MAE"], "mean_MAE")):
                close(actual, archived[key], "Archived summary differs")
            if archived["mean_CRPS_U"]:
                crps = [float(r["CRPS_U"]) for r in final if r["name"] == name]
                close(np.mean(crps), archived["mean_CRPS_U"], "Archived CRPS mean differs")
                row["source_mean_CRPS_U_not_recomputed"] = float(archived["mean_CRPS_U"])
        summary.append(row)
    ranked = sorted(summary, key=lambda r: r["mean_individual_RMSE"])
    for row in summary:
        row["RMSE_rank_of_7"] = next(i+1 for i, r in enumerate(ranked) if r["name"] == row["name"])
    candidate = next(r for r in summary if r["name"] == "atom_crps")
    comparisons = []
    for name in RECIPES:
        if name == "atom_crps":
            continue
        control = next(r for r in summary if r["name"] == name)
        gain = 1. - candidate["mean_individual_RMSE"] / control["mean_individual_RMSE"]
        paired = []
        for seed in SEEDS:
            a = next(r["RMSE"] for r in per_seed if r["name"] == "atom_crps" and r["seed"] == seed)
            b = next(r["RMSE"] for r in per_seed if r["name"] == name and r["seed"] == (seed if name in NEURAL else "analytic"))
            paired.append(a-b)
        comparisons.append({"candidate": "atom_crps", "control": name,
                            "candidate_mean_individual_RMSE": candidate["mean_individual_RMSE"],
                            "control_mean_individual_RMSE": control["mean_individual_RMSE"],
                            "RMSE_difference_candidate_minus_control": candidate["mean_individual_RMSE"]-control["mean_individual_RMSE"],
                            "relative_RMSE_gain_candidate": gain,
                            "candidate_lower_RMSE_paired_seed_count": sum(x<0 for x in paired),
                            "paired_seeds": 3,
                            "minimum_paired_seed_RMSE_difference": min(paired),
                            "maximum_paired_seed_RMSE_difference": max(paired)})
    best_all = min((r for r in summary if r["name"] != "atom_crps"), key=lambda r:r["mean_individual_RMSE"])
    best_neural = min((r for r in summary if r["name"] in NEURAL and r["name"] != "atom_crps"),
                      key=lambda r:r["mean_individual_RMSE"])
    gain_all = 1.-candidate["mean_individual_RMSE"]/best_all["mean_individual_RMSE"]
    gain_neural = 1.-candidate["mean_individual_RMSE"]/best_neural["mean_individual_RMSE"]
    close(gain_all, receipt["relative_gain_vs_best_all"], "Original gain differs")
    close(gain_neural, receipt["relative_gain_vs_best_neural"], "Original neural gain differs")
    continued = gain_all >= .02 and gain_neural >= .02
    require(continued == receipt["continuation_heuristic_passed"] is False, "Continuation verdict changed")
    resources = list(receipt["fit_resource_receipts"].values())
    readout = {"status": "verified_saved_development_outputs", "run_id": RUN,
               "source_commit": SOURCE, "results_commit": RESULTS, "snapshot_sequence": 93,
               "archived_source_files_verified": len(inventory), "full_snapshot_payload_files": full_inventory_count,
               "checkpoint_files_downloaded_or_reexecuted": 0,
               "completed_neural_fits_in_original_run": 30, "HPO_fits": 15, "final_fits": 15,
               "saved_trajectories": 17, "saved_prediction_rows": len(raw), "distinct_target_dates": 55,
               "assessment_start": dates[0], "assessment_end": dates[-1],
               "assessment_previously_exposed": True, "selection_rows": 54, "purged_rows": 7,
               "all_selected_rates_at_upper_grid_boundary": all(float(r)==.003 for r in selection["selected_learning_rates"].values()),
               "all_saved_forecasts_respect_observable_floor": True,
               "primary_estimand": "mean of three individual-seed RMSEs; not ensemble RMSE",
               "candidate_RMSE_rank_of_7": candidate["RMSE_rank_of_7"],
               "best_control": best_all["name"], "relative_gain_vs_best_control": gain_all,
               "candidate_RMSE_limit_for_original_heuristic": .98*best_all["mean_individual_RMSE"],
               "continuation_heuristic_passed": continued,
               "original_engine_elapsed_this_invocation_seconds": receipt["elapsed_this_invocation_seconds"],
               "sum_original_phase_elapsed_wall_seconds": sum(r["elapsed_wall_including_setup_io_and_restart_seconds"] for r in resources),
               "maximum_original_phase_deadline_overrun_seconds": max(r["deadline_overrun_seconds"] for r in resources),
               "new_real_data_fits": 0, "new_model_inference": 0, "new_statistical_resamples": 0,
               "p_values_computed": False, "prospective_power_certified": False,
               "R1_accredited": False, "R3_accredited": False,
               "AI_assistance_declared": True,
               "limits": ["Only an archived subset and its metadata hashes were checked; checkpoints and all93snapshots were not replayed.",
                          "55 overlapping dates and three seeds are not165independent observations.",
                          "Residual and monthly diagnostics were specified after the result; no confirmatory inference.",
                          "No saved fit predictions: cannot close fit/validation risk gap or full adequacy TablesA/B.",
                          "No per-observation atomic laws or probability intervals: CRPS_U source means not independently recomputed; cannot establish probabilistic calibration.",
                          "The fixed-probability control has1236parameters versus1262forlearnedmasses; comparison is not capacity matched.",
                          "No retuning or promotion of atom_mse_only follows from this report; any revised design needs a newversion and prospectiveprotocol."]}
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("metrics_recomputed.csv", summary), ("per_seed_metrics_recomputed.csv", per_seed),
                           ("ablation_comparisons.csv", comparisons), ("residual_diagnostics.csv", residual),
                           ("monthly_diagnostics.csv", monthly), ("residual_lag_correlations.csv", acf)):
        write_csv(output / filename, rows)
    adequacy = [
        {"requirement": "Original continuation heuristic", "status": "evaluated_and_not_met",
         "available_evidence": "All30fits and all7recipes; recomputed mean individual RMSE; frozen2percent rule",
         "remaining_limit": "A development failure is not a novelty refutation or confirmatory rejection"},
        {"requirement": "Common observable support floor", "status": "verified_on_saved_assessment_forecasts",
         "available_evidence": "Every saved forecast respects the recorded per-date floor on55dates",
         "remaining_limit": "Finite sample check does not extend the formal bound or certify predictive usefulness"},
        {"requirement": "Trivial comparator", "status": "available_development_comparison",
         "available_evidence": "Bothfloorcontrols retained; fullV3 hashigherRMSE thanfloor_plus_mean_excess",
         "remaining_limit": "Point ordering is not a significance, equivalence or noninferiority test"},
        {"requirement": "Residual and temporal diagnosis", "status": "post_hoc_descriptive_only",
         "available_evidence": "Bias, absolute errors, lag1to7correlations and allmonthly24/31date slices",
         "remaining_limit": "NoGaussianity, independence or temporal stability established; no chosenfavorablemonth"},
        {"requirement": "Full adequacy TablesA/B and overfit gap", "status": "not_closed_with_saved_assessment_outputs",
         "available_evidence": "Selection scores and assessment predictions exist",
         "remaining_limit": "Fit predictions and complete comparable training risks absent; no new inference performed"},
        {"requirement": "Probabilistic calibration and independent CRPS verification", "status": "not_closed",
         "available_evidence": "Original per-seed and mean CRPS_U scores retained and cross-checked arithmetically",
         "remaining_limit": "Per-observation atomic laws, predictive intervals and raw CRPS terms are not inthissubset"},
        {"requirement": "Ablation and capacity", "status": "development_evidence_available",
         "available_evidence": "Loss, masses and readout controls present; allseedresults retained",
         "remaining_limit": "Fixedmasses1236parameters differsfrom1262; no causal guarantee or confirmatory family"},
        {"requirement": "Prospective power and confirmatory R3", "status": "cannot_be_reconstructed_retroactively",
         "available_evidence": "Threefinalseeds and historically exposed55dates may inform a future design",
         "remaining_limit": "No formal delta, prospective jointpower or threeappropriatefamilyadjustedtests here"},
        {"requirement": "Independent integral reproduction", "status": "hash_and_arithmetic_subset_verified",
         "available_evidence": "19archivedfiles checked against sourceobjects/snapshot;17savedtrajectories recomputed",
         "remaining_limit": "No checkpoint replay, no fresh fit, no full259file or93snapshot restoration"},
    ]
    write_csv(output / "adequacy_scope_matrix.csv", adequacy)
    write_json(output / "readout.json", readout)
    write_json(output / "archive_verification.json", {"results_commit": RESULTS, "files_verified": inventory,
               "scope": "curated19file subset; no checkpoint replay; original snapshot contains additionalfiles"})
    if figures:
        make_figure(output, summary, per_seed, acf)
    print(json.dumps({"status": readout["status"], "source_files_verified": len(inventory),
                      "recomputed_trajectories": len(per_seed), "saved_rows": len(raw),
                      "continuation_heuristic_passed": continued, "new_real_data_fits": 0}))
    return readout


def make_figure(output, summary, per_seed, acf):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e3-v3-saved-output-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e3-v3-saved-output-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "e3-v3-saved-development-v1"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13., 5.1), gridspec_kw={"width_ratios": [1.45, 1.]})
    ordered = sorted(summary, key=lambda r:r["mean_individual_RMSE"])
    for index, row in enumerate(ordered):
        name = row["name"]
        values = [r["RMSE"] for r in per_seed if r["name"] == name]
        color = "#155DAA" if name == "atom_crps" else "#59636E"
        if name in NEURAL:
            left.scatter(values, np.full(len(values), index), marker="o", s=28, color=color, alpha=.55)
            left.scatter(row["mean_individual_RMSE"], index, marker="|", s=260, color=color, linewidth=2.3)
        else:
            left.scatter(values, [index], marker="D", s=37, facecolors="white", edgecolors=color, linewidth=1.5)
    left.set_yticks(range(len(ordered)), [LABELS[r["name"]] for r in ordered])
    left.invert_yaxis()
    left.set_xlabel("RMSE individual de σ₇ (unidades originales)")
    left.set_title("A. Desarrollo: 55 fechas; tres semillas por red", loc="left", fontsize=11)
    left.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    left.grid(axis="x", alpha=.2)
    for name, color in (("atom_crps", "#155DAA"), ("atom_mse_only", "#BA5E15"), ("additive_positive_head", "#6E747A")):
        points = []
        for lag in range(1, 8):
            points.append(np.mean([r["residual_correlation"] for r in acf if r["name"] == name and r["lag"] == lag]))
        right.plot(range(1, 8), points, marker="o", ms=4, color=color, label=LABELS[name])
    right.axhline(0, color="#8A8A8A", linewidth=.8)
    right.set_xticks(range(1, 8))
    right.set_ylim(-1, 1)
    right.set_xlabel("Desfase diario")
    right.set_ylabel("Correlación descriptiva de residuos")
    right.set_title("B. Dependencia temporal observada", loc="left", fontsize=11)
    right.grid(alpha=.2)
    right.legend(frameon=False, loc="lower left", fontsize=9)
    fig.text(.01, .015, "Puntos: semillas; barra: media de RMSE individuales; rombo: control analítico. Panel B: media de correlaciones entre semillas.\n"
             "Datos de desarrollo previamente expuestos. Sin intervalos, valores p, pruebas confirmatorias o acreditación R1/R3.", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / ("pilot_rmse_and_dependence." + suffix), dpi=170, bbox_inches="tight",
                    metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Read-only archive and independent arithmetic checks; no figures or writes")
    args = parser.parse_args()
    if args.verify_only:
        from verify_saved_pilot import verify
        _, _, _, _, _, original_files, _ = verify_archive(args.root)
        result = verify(args.root)
        print(json.dumps({**result, "archive_files_verified": len(original_files), "read_only": True}))
    else:
        analyze(args.root, args.output or args.root / "generated", figures=not args.no_figures)
