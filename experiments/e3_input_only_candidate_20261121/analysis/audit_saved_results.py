#!/usr/bin/env python3
"""Recompute the exposed V5 pilot from saved CSVs; stdlib, no model imports.

No training, forward pass, new prediction, resampling, or significance test occurs.
Default paths are the immutable local evidence delivered for this task.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics as st
import zipfile

EXPECTED_ZIP_SHA256 = "beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c"
SOURCE_COMMIT = "c4ca86f1431afeaa958ff0b544b453ab17a4d7b8"
RESULT_COMMIT = "05c1a637f13356dd1df6b8c5aa8f4676dda61c12"
DEFAULT_ZIP = Path(__file__).resolve().parent.parent / "inputs" / "public_results.zip"
DEFAULT_OUT = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def mean(xs):
    return st.mean(list(xs))


def sd(xs):
    xs = list(xs)
    return st.stdev(xs) if len(xs) > 1 else None


def quantile(xs, q):
    """Linear interpolation, same convention as numpy.quantile(method='linear')."""
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return xs[lo] + (i - lo) * (xs[hi] - xs[lo])


def corr(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy) if xx * yy else None


def metrics(rows):
    errors = [r["prediction"] - r["target"] for r in rows]
    return {"n_dates": len(rows), "RMSE": math.sqrt(mean(e * e for e in errors)),
            "MAE": mean(abs(e) for e in errors), "mean_error": mean(errors)}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def audit(zip_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    blob = zip_path.read_bytes()
    assert sha256(blob) == EXPECTED_ZIP_SHA256, "ZIP differs from pinned evidence"
    z = zipfile.ZipFile(io.BytesIO(blob))
    assert len(z.namelist()) == len(set(z.namelist())), "Duplicate ZIP member"
    J = lambda p: json.loads(z.read(p))
    C = lambda p: list(csv.DictReader(io.StringIO(z.read(p).decode())))
    plan, protocol, admission, result = [J(p) for p in ("plan.json", "protocol.json", "admission.json", "pilot_result.json")]
    prepared, anchor, identity = [J(p) for p in ("prepared.json", "PUBLIC_SOURCE_ANCHOR.json", "source_identity.json")]
    seeds = plan["seeds"]
    neural = plan["recipes"]
    analytic = plan["analytical_controls"]
    recipes = neural + analytic
    checks = []

    def check(name, expected, actual, method="exact"):
        ok = math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-15) if method == "float" else expected == actual
        checks.append({"check": name, "expected": expected, "actual": actual, "method": method, "passed": ok})
        return ok

    check("source_anchor_commit", SOURCE_COMMIT, anchor["source_commit"])
    check("source_identity_anchor", anchor, identity["public_source_anchor"])
    check("plan_raw_sha256", prepared["plan_file_sha256"], sha256(z.read("plan.json")))
    plan_hash = canonical_hash(plan)
    for name, obj in [("prepared", prepared), ("pilot_result", result), ("assessment_started", J("assessment_started.json")), ("remote_restore", J("remote_restore.json"))]:
        check(name + "_canonical_plan_sha256", obj["plan_sha256"], plan_hash)
    check("protocol_raw_sha256", plan["protocol_sha256"], sha256(z.read("protocol.json")))
    check("admission_canonical_sha256", plan["admission_sha256"], canonical_hash(admission))
    for name, rec in result["files"].items():
        check(name + "_sha256", rec["sha256"], sha256(z.read(name)))
        check(name + "_bytes", rec["bytes"], len(z.read(name)))
    source_protocol = next(x for x in identity["files"] if x["path"] == "protocol.json")
    check("source_identity_protocol_sha256", source_protocol["sha256"], sha256(z.read("protocol.json")))
    check("source_identity_manifest_anchor_consistency", anchor["manifest_sha256"], identity["manifest_sha256"])

    raw_predictions = C("predictions.csv")
    rows = []
    by_trajectory = defaultdict(list)
    by_date = defaultdict(list)
    seen = set()
    for r in raw_predictions:
        row = {"recipe": r["recipe"], "seed": int(r["seed"]) if r["seed"] else None,
               "date": r["date"], "target": float(r["target"]), "prediction": float(r["prediction"])}
        assert row["recipe"] in recipes
        assert math.isfinite(row["target"]) and math.isfinite(row["prediction"])
        key = row["recipe"], row["seed"], row["date"]
        assert key not in seen, "duplicate prediction"
        seen.add(key)
        rows.append(row)
        by_trajectory[key[:2]].append(row)
        by_date[row["date"]].append(row)
    dates = sorted(by_date)
    check("assessment_dates_exact", admission["assessment_dates"], dates)
    check("assessment_prediction_rows", 1760, len(rows))
    check("neural_assessment_prediction_rows", 1650, sum(r["recipe"] in neural for r in rows))
    check("analytical_assessment_prediction_rows", 110, sum(r["recipe"] in analytic for r in rows))
    check("trajectory_count", 32, len(by_trajectory))
    for key, rs in by_trajectory.items():
        rs.sort(key=lambda r: r["date"])
        check(f"complete_dates/{key}", dates, [r["date"] for r in rs])
    for date, rs in by_date.items():
        assert len({r["target"] for r in rs}) == 1, f"target mismatch {date}"
        assert len(rs) == 32
    expected_keys = {(r, s) for r in neural for s in seeds} | {(r, None) for r in analytic}
    check("planned_trajectory_identities", sorted((r, str(s)) for r, s in expected_keys), sorted((r, str(s)) for r, s in by_trajectory))

    recalculated = [{"recipe": key[0], "seed": key[1], **metrics(rs)} for key, rs in by_trajectory.items()]
    M = {(r["recipe"], r["seed"]): r for r in recalculated}
    for r in C("metrics.csv"):
        key = r["recipe"], int(r["seed"]) if r["seed"] else None
        for field in ("RMSE", "MAE"):
            check(f"metrics/{key}/{field}", float(r[field]), M[key][field], "float")
    summaries = []
    for recipe in recipes:
        ms = [r for r in recalculated if r["recipe"] == recipe]
        summaries.append({"recipe": recipe, "trajectories": len(ms), "dates_each": 55,
                          "mean_RMSE": mean(r["RMSE"] for r in ms), "sample_SD_RMSE": sd(r["RMSE"] for r in ms),
                          "min_RMSE": min(r["RMSE"] for r in ms), "max_RMSE": max(r["RMSE"] for r in ms),
                          "mean_MAE": mean(r["MAE"] for r in ms), "sample_SD_MAE": sd(r["MAE"] for r in ms),
                          "pooled_RMSE_descriptive_only": metrics([r for r in rows if r["recipe"] == recipe])["RMSE"]})
    S = {r["recipe"]: r for r in summaries}
    for rank, row in enumerate(sorted(summaries, key=lambda r: r["mean_RMSE"]), 1):
        row["rank_mean_RMSE"] = rank
    for r in C("recipe_summary.csv"):
        for field in ("mean_RMSE", "mean_MAE", "sample_SD_RMSE"):
            if r[field]:
                check(f"recipe_summary/{r['recipe']}/{field}", float(r[field]), S[r["recipe"]][field], "float")

    pair_rows, pair_summaries = [], []
    for candidate, control in itertools.permutations(recipes, 2):
        paired_seeds = seeds if candidate in neural or control in neural else [None]
        ps = []
        for seed in paired_seeds:
            a = M[candidate, seed if candidate in neural else None]
            b = M[control, seed if control in neural else None]
            p = {"candidate": candidate, "control": control, "seed": seed, "n_dates": 55,
                 "candidate_RMSE": a["RMSE"], "control_RMSE": b["RMSE"],
                 "RMSE_difference_candidate_minus_control": a["RMSE"] - b["RMSE"],
                 "relative_RMSE_gain": 1 - a["RMSE"] / b["RMSE"],
                 "candidate_MAE": a["MAE"], "control_MAE": b["MAE"],
                 "MAE_difference_candidate_minus_control": a["MAE"] - b["MAE"],
                 "both_repeated_neural_fits": candidate in neural and control in neural,
                 "analytical_trajectory_reused": candidate in analytic or control in analytic}
            ps.append(p)
            pair_rows.append(p)
        ds = [p["RMSE_difference_candidate_minus_control"] for p in ps]
        candidate_mean, control_mean = S[candidate]["mean_RMSE"], S[control]["mean_RMSE"]
        pair_summaries.append({"candidate": candidate, "control": control, "paired_rows": len(ps),
                               "independent_time_panels": 1, "shared_dates": 55,
                               "mean_RMSE_difference": mean(ds), "sample_SD_seed_RMSE_difference": sd(ds),
                               "min_RMSE_difference": min(ds), "max_RMSE_difference": max(ds),
                               "relative_mean_RMSE_gain": 1 - candidate_mean / control_mean,
                               "seed_wins": sum(d < 0 for d in ds) if candidate in neural and control in neural else None,
                               "seed_ties": sum(d == 0 for d in ds) if candidate in neural and control in neural else None,
                               "candidate_trajectories_better_than_analytical": sum(d < 0 for d in ds) if control in analytic and candidate in neural else None,
                               "mean_MAE_difference": S[candidate]["mean_MAE"] - S[control]["mean_MAE"],
                               "mean_MAE_ratio": S[candidate]["mean_MAE"] / S[control]["mean_MAE"],
                               "no_inferential_p_value_or_CI": True})
    P = {(p["candidate"], p["control"], p["seed"]): p for p in pair_rows}
    for r in C("paired_differences.csv"):
        seed = int(r["seed"]) if r["seed"] else None
        p = P[r["candidate"], r["control"], seed]
        for field in ("candidate_RMSE", "control_RMSE", "RMSE_difference_candidate_minus_control", "relative_RMSE_gain", "candidate_MAE", "control_MAE"):
            check(f"saved_pair/{r['control']}/{seed}/{field}", float(r[field]), p[field], "float")

    # Curves certify the chosen epoch numerically; selection predictions are absent.
    resources = C("resources.csv")
    R = {r["job"]: r for r in resources}
    inventories = {r["job"]["job_id"]: r for r in J("checkpoint_inventory.json")}
    job_audit = []
    gaps = []
    for planned in plan["jobs"]:
        job_id = planned["job_id"]
        job = J(f"jobs/{job_id}/job.json")
        curve = J(f"jobs/{job_id}/curve.json")
        attempts = J(f"jobs/{job_id}/attempts.json")
        best = min(curve, key=lambda r: (r["selection_mse_scaled"], r["epoch"]))
        selected_rmse = math.sqrt(best["selection_mse_scaled"]) * admission["y_scale"]
        check(job_id + "/status", "completed_validated_fit", job["status"])
        check(job_id + "/plan", plan_hash, job["plan_sha256"])
        check(job_id + "/identity", planned, job["job"])
        check(job_id + "/epochs", len(curve), job["epochs"])
        check(job_id + "/curve_epochs", list(range(1, job["epochs"] + 1)), [r["epoch"] for r in curve])
        check(job_id + "/best_epoch", best["epoch"], job["best_epoch"])
        check(job_id + "/selection_RMSE", selected_rmse, job["selection_RMSE"], "float")
        check(job_id + "/optimizer_steps", job["epochs"] * math.ceil(admission["fit_rows"] / plan["training"]["batch_size"]), job["optimizer_steps_committed"])
        check(job_id + "/checkpoint_inventory_hash_consistency", job["checkpoint_sha256"], inventories[job_id]["checkpoint_sha256"])
        check(job_id + "/checkpoint_payload_absent", False, inventories[job_id]["checkpoint_payload_exported"])
        for field in ("epochs", "best_epoch", "optimizer_steps_committed", "parameters", "interrupted_epochs"):
            check(job_id + "/resource_" + field, job[field], int(R[job_id][field]))
        for field in ("active_seconds_charged", "checkpoint_io_seconds"):
            check(job_id + "/resource_" + field, job[field], float(R[job_id][field]), "float")
        check(job_id + "/stop_reason", job["stop_reason"], R[job_id]["stop_reason"])
        key = planned["recipe"], planned["seed"]
        assessment_rmse = M[key]["RMSE"]
        gap = {"recipe": key[0], "seed": key[1], "best_epoch": best["epoch"],
               "selection_RMSE_from_curve": selected_rmse, "assessment_RMSE_from_predictions": assessment_rmse,
               "assessment_minus_selection_RMSE": assessment_rmse - selected_rmse,
               "relative_gap": assessment_rmse / selected_rmse - 1,
               "selection_n_dates": 54, "assessment_n_dates": 55, "assessment_previously_exposed": True}
        gaps.append(gap)
        job_audit.append({"job": job_id, "status": job["status"], "epochs": job["epochs"], "best_epoch": best["epoch"],
                          "selection_RMSE_from_curve": selected_rmse, "attempts": len(attempts),
                          "interrupted_epochs": job["interrupted_epochs"], "stop_reason": job["stop_reason"],
                          "optimizer_steps_committed": job["optimizer_steps_committed"],
                          "active_seconds_charged": job["active_seconds_charged"], "checkpoint_io_seconds": job["checkpoint_io_seconds"],
                          "unobserved_remote_compute_possible": job["unobserved_remote_compute_possible"],
                          "checkpoint_bytes_verified": False})
    G = {(r["recipe"], r["seed"]): r for r in gaps}
    for r in C("selection_assessment_gaps.csv"):
        key = r["recipe"], int(r["seed"])
        mapping = {"selection_RMSE": "selection_RMSE_from_curve", "assessment_RMSE": "assessment_RMSE_from_predictions", "assessment_minus_selection_RMSE": "assessment_minus_selection_RMSE", "relative_gap": "relative_gap"}
        for old, new in mapping.items():
            check(f"saved_gap/{key}/{old}", float(r[old]), G[key][new], "float")
    selection_summary = []
    for recipe in neural:
        gs = [r for r in gaps if r["recipe"] == recipe]
        js = [r for r in job_audit if r["job"].endswith("/" + recipe)]
        selection_summary.append({"recipe": recipe, "mean_selection_RMSE": mean(r["selection_RMSE_from_curve"] for r in gs),
                                  "mean_assessment_RMSE": S[recipe]["mean_RMSE"], "mean_gap": mean(r["assessment_minus_selection_RMSE"] for r in gs),
                                  "sample_SD_selection_RMSE": sd(r["selection_RMSE_from_curve"] for r in gs),
                                  "mean_selected_epoch": mean(r["best_epoch"] for r in gs),
                                  "min_selected_epoch": min(r["best_epoch"] for r in gs), "max_selected_epoch": max(r["best_epoch"] for r in gs),
                                  "mean_epochs_run": mean(r["epochs"] for r in js)})
    for rank, row in enumerate(sorted(selection_summary, key=lambda r: r["mean_selection_RMSE"]), 1):
        row["rank_selection"] = rank
    for rank, row in enumerate(sorted(selection_summary, key=lambda r: r["mean_assessment_RMSE"]), 1):
        row["rank_assessment_neural"] = rank
    selection_pair = []
    for control in neural:
        if control == "input_only_mse":
            continue
        for seed in seeds:
            a, b = G["input_only_mse", seed], G[control, seed]
            selection_pair.append({"candidate": "input_only_mse", "control": control, "seed": seed,
                                   "selection_RMSE_difference": a["selection_RMSE_from_curve"] - b["selection_RMSE_from_curve"],
                                   "assessment_RMSE_difference": a["assessment_RMSE_from_predictions"] - b["assessment_RMSE_from_predictions"],
                                   "same_sign_difference": (a["selection_RMSE_from_curve"] < b["selection_RMSE_from_curve"]) == (a["assessment_RMSE_from_predictions"] < b["assessment_RMSE_from_predictions"])})

    # Post-hoc descriptive subsets; never treated as independent experiments.
    target_by_date = {date: by_date[date][0]["target"] for date in dates}
    qcuts = [quantile(list(target_by_date.values()), q) for q in (.25, .5, .75)]
    date_regimes = []
    blocks = defaultdict(list)
    for i, date in enumerate(dates):
        q = 1 + sum(target_by_date[date] > cut for cut in qcuts)
        half = "first_27" if i < len(dates) // 2 else "last_28"
        block = f"block_{i // 7 + 1:02d}"
        date_regimes.append({"date": date, "target_realized_sigma7": target_by_date[date],
                             "month": date[:7], "chronological_half": half,
                             "consecutive_7day_block": block, "retrospective_target_quartile": f"Q{q}"})
        for kind, group in (("month", date[:7]), ("half", half), ("seven_day_block", block), ("realized_target_quartile", f"Q{q}")):
            blocks[kind, group].append(date)
    subset_metrics, subset_summaries, subset_pairs = [], [], []
    for (kind, group), selected_dates in blocks.items():
        date_set = set(selected_dates)
        local = {}
        for key, rs in by_trajectory.items():
            m = metrics([r for r in rs if r["date"] in date_set])
            local[key] = m
            subset_metrics.append({"subset_kind": kind, "subset": group, "recipe": key[0], "seed": key[1], **m})
        sm = {}
        for recipe in recipes:
            values = [m for key, m in local.items() if key[0] == recipe]
            sm[recipe] = {"subset_kind": kind, "subset": group, "recipe": recipe, "n_dates": len(selected_dates),
                          "first_date": min(selected_dates), "last_date": max(selected_dates),
                          "mean_individual_seed_RMSE": mean(m["RMSE"] for m in values),
                          "sample_SD_seed_RMSE": sd(m["RMSE"] for m in values), "mean_MAE": mean(m["MAE"] for m in values)}
        for rank, recipe in enumerate(sorted(recipes, key=lambda r: sm[r]["mean_individual_seed_RMSE"]), 1):
            sm[recipe]["rank_RMSE"] = rank
        subset_summaries.extend(sm.values())
        for control in recipes:
            if control == "input_only_mse":
                continue
            a, b = sm["input_only_mse"], sm[control]
            subset_pairs.append({"subset_kind": kind, "subset": group, "candidate": "input_only_mse", "control": control,
                                 "n_dates": len(selected_dates), "first_date": min(selected_dates), "last_date": max(selected_dates),
                                 "candidate_mean_RMSE": a["mean_individual_seed_RMSE"], "control_mean_RMSE": b["mean_individual_seed_RMSE"],
                                 "relative_mean_RMSE_gain": 1 - a["mean_individual_seed_RMSE"] / b["mean_individual_seed_RMSE"],
                                 "mean_MAE_difference": a["mean_MAE"] - b["mean_MAE"],
                                 "seed_wins": sum(local["input_only_mse", s]["RMSE"] < local[control, s]["RMSE"] for s in seeds) if control in neural else None,
                                 "post_hoc_descriptive_only": True})

    exclusions = []
    for excluded_kind, exclusion in [("one_date", {date: [date] for date in dates}),
                                      ("seven_day_block", {g: ds for (k, g), ds in blocks.items() if k == "seven_day_block"})]:
        for name, remove_dates in exclusion.items():
            date_set = set(remove_dates)
            mm = {key: metrics([r for r in rs if r["date"] not in date_set]) for key, rs in by_trajectory.items()}
            cm = mean(mm["input_only_mse", s]["RMSE"] for s in seeds)
            for control in recipes:
                if control == "input_only_mse":
                    continue
                bm = mean(mm[control, s]["RMSE"] for s in seeds) if control in neural else mm[control, None]["RMSE"]
                exclusions.append({"excluded_kind": excluded_kind, "excluded": name, "excluded_dates": len(date_set),
                                   "remaining_dates": len(dates) - len(date_set), "candidate": "input_only_mse", "control": control,
                                   "candidate_mean_RMSE": cm, "control_mean_RMSE": bm, "relative_mean_RMSE_gain": 1 - cm / bm,
                                   "post_hoc_descriptive_only": True})

    indexed = {(r["recipe"], r["seed"], r["date"]): r for r in rows}
    date_diffs, dependence = [], []
    for control in recipes:
        if control == "input_only_mse":
            continue
        dms, das = [], []
        for date in dates:
            errs_a = [indexed["input_only_mse", s, date]["prediction"] - target_by_date[date] for s in seeds]
            errs_b = [(indexed[control, s if control in neural else None, date]["prediction"] - target_by_date[date]) for s in seeds]
            dm = mean(e * e for e in errs_a) - mean(e * e for e in errs_b)
            da = mean(abs(e) for e in errs_a) - mean(abs(e) for e in errs_b)
            dms.append(dm)
            das.append(da)
            date_diffs.append({"candidate": "input_only_mse", "control": control, "date": date,
                               "target": target_by_date[date], "mean_over_seeds_squared_error_difference": dm,
                               "mean_over_seeds_absolute_error_difference": da})
        dependence.append({"series": f"input_only_minus_{control}_seed_mean_squared_error", "n_dates": 55,
                           "lag1_Pearson_correlation_descriptive": corr(dms[:-1], dms[1:]),
                           "dates_with_negative_mean_loss_difference": sum(x < 0 for x in dms),
                           "not_effective_sample_size_or_inference": True})
    ys = [target_by_date[d] for d in dates]
    dependence.append({"series": "realized_target_sigma7", "n_dates": 55, "lag1_Pearson_correlation_descriptive": corr(ys[:-1], ys[1:]), "not_effective_sample_size_or_inference": True})

    # The floor-only point forecast permits a check of published means' support.
    support = []
    for key, rs in by_trajectory.items():
        margins = [r["prediction"] - indexed["floor_only", None, r["date"]]["prediction"] for r in rs]
        support.append({"recipe": key[0], "seed": key[1], "min_prediction_minus_floor_control": min(margins),
                        "point_means_below_floor_tolerance1e-12": sum(x < -1e-12 for x in margins),
                        "point_means_within1e-12_of_floor": sum(abs(x) <= 1e-12 for x in margins),
                        "atomic_distribution_or_calibration_checked": False})
    check("empty_failures_log", [], J("failures.json"))
    fit_status, restore, session = [J(p) for p in ("fit_status.json", "remote_restore.json", "github_session.json")]
    check("fit_status_completed_fits", len(job_audit), fit_status["completed_neural_fits"])
    check("pilot_completed_fits", len(job_audit), result["completed_neural_fits"])
    check("closed_restored_session", False, restore["last_saved_session_open"])
    check("closed_current_session", False, session["session_open"])
    check("no_unobserved_compute_reported_fit_status", False, fit_status["unobserved_remote_compute_possible"])
    check("no_unobserved_compute_reported_restore", False, restore["unobserved_remote_compute_possible"])
    check("no_unobserved_compute_reported_result", False, result["unobserved_remote_compute_possible"])
    check("zero_interrupted_epochs", 0, sum(j["interrupted_epochs"] for j in job_audit))
    check("all_attempt_counts_one", [1] * 30, [j["attempts"] for j in job_audit])
    best = min((S[r] for r in recipes if r != "seoi_mse"), key=lambda r: r["mean_RMSE"])["recipe"]
    full_pair = next(p for p in pair_summaries if p["candidate"] == "seoi_mse" and p["control"] == best)
    check("fullV5_best_control", result["decision"]["best_all_control_by_mean_RMSE"], best)
    check("fullV5_gain", result["decision"]["relative_mean_RMSE_gain_vs_best_all"], full_pair["relative_mean_RMSE_gain"], "float")
    check("fullV5_seed_wins", result["decision"]["seed_wins_vs_best_neural"], full_pair["seed_wins"])
    check("fullV5_MAE", result["decision"]["candidate_mean_MAE"], S["seoi_mse"]["mean_MAE"], "float")
    check("fullV5_max_allowed_MAE", result["decision"]["maximum_allowed_MAE"], 1.05 * S[best]["mean_MAE"], "float")
    check("fullV5_original_decision", "ABANDON_CURRENT_SEOI_RECIPE", result["decision"]["decision"])

    # Prior readouts are context only; all numbers above came from CSV / curves.
    context_checks = []
    for context_path in (zip_path.parent / "RESULT_READOUT.json", zip_path.parent / "INPUT_ONLY_COMPARISON.json"):
        if not context_path.exists():
            continue
        obj = json.loads(context_path.read_text())
        if "means_reconstructed_from_saved_predictions" in obj:
            for recipe, values in obj["means_reconstructed_from_saved_predictions"].items():
                for metric in ("RMSE", "MAE"):
                    context_checks.append({"context": context_path.name, "recipe": recipe, "metric": metric,
                                           "saved": values[metric], "recomputed": S[recipe]["mean_" + metric],
                                           "agrees": math.isclose(values[metric], S[recipe]["mean_" + metric], abs_tol=1e-15, rel_tol=1e-12)})
        if "comparison" in obj:
            for r in obj["comparison"]:
                p = next(p for p in pair_summaries if p["candidate"] == "input_only_mse" and p["control"] == r["control"])
                context_checks.append({"context": context_path.name, "control": r["control"], "metric": "gain_percent",
                                       "saved": r["input_only_gain_percent"], "recomputed": 100 * p["relative_mean_RMSE_gain"],
                                       "agrees": math.isclose(r["input_only_gain_percent"], 100 * p["relative_mean_RMSE_gain"], abs_tol=1e-12, rel_tol=1e-12)})
    context_budget = {"nominal_max_active_seconds": 30 * plan["budget"]["active_seconds_per_fit"],
                      "recorded_active_seconds_total": sum(j["active_seconds_charged"] for j in job_audit),
                      "recorded_checkpoint_IO_seconds_total": sum(j["checkpoint_io_seconds"] for j in job_audit),
                      "committed_optimizer_steps": sum(j["optimizer_steps_committed"] for j in job_audit),
                      "epochs_total": sum(j["epochs"] for j in job_audit),
                      "maximum_recorded_active_seconds_one_fit": max(j["active_seconds_charged"] for j in job_audit),
                      "stop_reasons": dict(Counter(j["stop_reason"] for j in job_audit)),
                      "interrupted_epochs": sum(j["interrupted_epochs"] for j in job_audit),
                      "logged_failures": len(J("failures.json")),
                      "scope": "recorded fit activity and checkpoint IO; no VM/setup/transport/total physical cost inference",
                      "closed_session_restored": not restore["last_saved_session_open"],
                      "restore_commit": restore["commit"], "restored_at_utc_as_recorded": restore["restored_at_utc"],
                      "current_session": session, "restore_receipt": restore}
    recovery_per_recipe = []
    for recipe in neural:
        js = [j for j in job_audit if j["job"].endswith("/" + recipe)]
        recovery_per_recipe.append({"recipe": recipe, "fits": len(js), "epochs": sum(j["epochs"] for j in js),
                                   "optimizer_steps": sum(j["optimizer_steps_committed"] for j in js),
                                   "active_seconds": sum(j["active_seconds_charged"] for j in js),
                                   "checkpoint_IO_seconds": sum(j["checkpoint_io_seconds"] for j in js),
                                   "interrupted_epochs": sum(j["interrupted_epochs"] for j in js)})
    exclusions_global = [x for x in exclusions if x["control"] == "global_mse"]
    loo_summary = {kind: {"min_gain": min(x["relative_mean_RMSE_gain"] for x in exclusions_global if x["excluded_kind"] == kind),
                           "max_gain": max(x["relative_mean_RMSE_gain"] for x in exclusions_global if x["excluded_kind"] == kind)}
                   for kind in ("one_date", "seven_day_block")}
    primary_pair = next(p for p in pair_summaries if p["candidate"] == "input_only_mse" and p["control"] == "global_mse")
    summary = {"status": "PASS" if all(c["passed"] for c in checks) and all(c["agrees"] for c in context_checks) else "DISCREPANCIES",
               "result_commit_context": RESULT_COMMIT, "source_commit": SOURCE_COMMIT,
               "ZIP_path_at_audit": str(zip_path), "ZIP_sha256": sha256(blob), "ZIP_members": len(z.namelist()), "checks_count": len(checks),
               "failed_checks": [c for c in checks if not c["passed"]],
               "context_discrepancies": [c for c in context_checks if not c["agrees"]],
               "predictions_rows": len(rows), "unique_dates": len(dates), "neural_seeds": seeds,
               "mean_of_per_seed_metrics": sorted(summaries, key=lambda r: r["rank_mean_RMSE"]),
               "input_only_vs_global": primary_pair,
               "selection_assessment": sorted(selection_summary, key=lambda r: r["rank_selection"]),
               "input_only_SD_ratio_to_global": S["input_only_mse"]["sample_SD_RMSE"] / S["global_mse"]["sample_SD_RMSE"],
               "descriptive_target_quartile_cutpoints": qcuts,
               "leave_out_sensitivity_vs_global": loo_summary,
               "budget_and_recovery": context_budget,
               "decision_scope": "Retrospective candidate signal only; full V5 decision unchanged; no automatic promotion; no R3 accreditation",
               "fullV5_original_decision": result["decision"],
               "new_fits": 0, "new_model_forwards": 0, "new_market_predictions": 0,
               "statistical_inference": "None; no p-values, independent-observation CIs, bootstrap, power claim or preregistration",
               "limitations": ["55 shared temporally dependent dates, five seeds, one market panel; not 275 independent observations",
                               "Candidate chosen after examining exposed development results; winner's selection bias remains",
                               "54 selection targets differ from 55 assessment targets; difference of errors does not establish generalization stability",
                               "Selection prediction CSVs absent; selection RMSE reconstructed from stored curves and recorded target scale only",
                               "Only point predictions exported: no atom positions/masses, CRPS, PIT, coverage, or calibration assessment possible",
                               "Known-window scale and raw input arrays absent; target-volatility quartiles are retrospective realized outcomes, not causal regimes",
                               "Full source files, NPZ and checkpoint payloads absent from ZIP; only available protocol/source metadata and checkpoint hash consistency checked",
                               "Results commit identifier supplied by readout context; ZIP SHA pinned independently; no remote Git ref verification in this subaudit",
                               "Recorded activity excludes setup, transport and all physical accounting; restore receipts are declarations, not a full external ledger",
                               "Historical 796/V3/V4 statements preserved from protocol, not independently re-audited because their original result files are outside this ZIP"]}
    outputs = {"metrics_recomputed.csv": recalculated, "recipe_summary_recomputed.csv": sorted(summaries, key=lambda r: r["rank_mean_RMSE"]),
               "all_pairwise_seed_differences.csv": pair_rows, "all_pairwise_summary.csv": pair_summaries,
               "selection_assessment_recomputed.csv": gaps, "selection_assessment_summary.csv": sorted(selection_summary, key=lambda r: r["rank_selection"]),
               "selection_assessment_input_only_pairs.csv": selection_pair, "jobs_audited.csv": job_audit,
               "resources_by_recipe.csv": recovery_per_recipe, "date_regimes_posthoc.csv": date_regimes,
               "subset_metrics_per_seed.csv": subset_metrics, "subset_summary.csv": subset_summaries,
               "subset_input_only_comparisons.csv": subset_pairs, "leave_out_sensitivity.csv": exclusions,
               "date_loss_differences.csv": date_diffs, "descriptive_dependence.csv": dependence,
               "point_support_audit.csv": support, "integrity_checks.csv": checks, "context_readout_checks.csv": context_checks}
    for name, output_rows in outputs.items():
        write_csv(out / name, output_rows)
    write_json(out / "AUDIT_SUMMARY.json", summary)
    write_json(out / "HASH_AND_RECOVERY_AUDIT.json", {"ZIP_sha256": sha256(blob), "checks": checks,
                                                       "budget_and_recovery": context_budget,
                                                       "independent_payload_verification_limitations": summary["limitations"][6:9]})
    report = render_report(summary, pair_summaries, subset_pairs, dependence, support)
    (out / "EMPIRICAL_AUDIT.md").write_text(report, encoding="utf-8")
    artifacts = {p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name != "ARTIFACT_HASHES.json"}
    write_json(out / "ARTIFACT_HASHES.json", {"input_ZIP_sha256": sha256(blob), "artifacts": artifacts})
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failed_checks": len(summary["failed_checks"]),
                      "new_fits": 0, "new_model_forwards": 0, "output": str(out)}, ensure_ascii=False))
    assert summary["status"] == "PASS", "Audit discrepancies; inspect AUDIT_SUMMARY.json"


def render_report(s, pairs, subsets, dependence, support):
    primary = s["input_only_vs_global"]
    rank_rows = "\n".join(f"| {r['rank_mean_RMSE']} | {r['recipe']} | {r['mean_RMSE']:.15g} | {r['sample_SD_RMSE'] if r['sample_SD_RMSE'] is not None else '—'} | {r['mean_MAE']:.15g} |" for r in s["mean_of_per_seed_metrics"])
    candidate_pairs = [p for p in pairs if p["candidate"] == "input_only_mse"]
    pair_rows = "\n".join(f"| {p['control']} | {p['mean_RMSE_difference']:.9g} | {100*p['relative_mean_RMSE_gain']:.6f}% | {p['seed_wins'] if p['seed_wins'] is not None else 'no aplica'} | {p['mean_MAE_difference']:.9g} |" for p in candidate_pairs)
    selected = [p for p in subsets if p["control"] == "global_mse" and p["subset_kind"] != "seven_day_block"]
    subset_rows = "\n".join(f"| {r['subset_kind']} | {r['subset']} | {r['n_dates']} | {100*r['relative_mean_RMSE_gain']:.6f}% | {r['seed_wins']} |" for r in selected)
    selection_rows = "\n".join(f"| {r['recipe']} | {r['mean_selection_RMSE']:.12g} | {r['rank_selection']} | {r['mean_assessment_RMSE']:.12g} | {r['rank_assessment_neural']} | {r['min_selected_epoch']}–{r['max_selected_epoch']} |" for r in s["selection_assessment"])
    b = s["budget_and_recovery"]
    acf_target = next(r["lag1_Pearson_correlation_descriptive"] for r in dependence if r["series"] == "realized_target_sigma7")
    loo = s["leave_out_sensitivity_vs_global"]
    below = sum(r["point_means_below_floor_tolerance1e-12"] for r in support)
    return f"""# Auditoría empírica retrospectiva de input_only_mse

El CSV confirma una señal exploratoria favorable para estudiar `input_only_mse`: RMSE medio {s['mean_of_per_seed_metrics'][0]['mean_RMSE']:.17g}, mejora {100*primary['relative_mean_RMSE_gain']:.9f}% respecto de `global_mse`, cuatro victorias de cinco semillas. Esto no es una confirmación independiente ni una autorización para promover una ablación. La decisión original de V5 completo sigue siendo **ABANDON_CURRENT_SEOI_RECIPE**. La suficiencia de la contribución y el dictamen global CONTINUAR/DESCARTAR requieren la auditoría técnica y la guía; este informe no los sustituye.

## Fuentes, integridad y alcance ejecutado

- ZIP local: `{s['ZIP_path_at_audit']}`; SHA-256 `{s['ZIP_sha256']}`; {s['ZIP_members']} miembros.
- Código original declarado: `{SOURCE_COMMIT}`. Resultados declarados: `{RESULT_COMMIT}`, `runs/e3_seoi_v5/run-01`. La vinculación remota del commit debe comprobarse en el expediente principal; este script comprueba el ZIP fijado y su consistencia interna.
- {s['checks_count']} comprobaciones internas: **{s['status']}**; los siete archivos del manifiesto de resultados coinciden en bytes y hash. SHA de protocolo, plan serializado y plan/admisión canónicos verificados. Las métricas se recalculan desde `predictions.csv`; los JSON previos se usan solo para contrastar después los resultados.
- 30 ajustes históricos del piloto, seis recetas × cinco semillas. 1.760 predicciones guardadas: 1.650 neurales y 110 analíticas, sobre las mismas 55 fechas (2024-09-07 a 2024-10-31). Sin filas duplicadas, faltantes, valores no finitos ni objetivos discordantes. Los identificadores enteros de semilla no son fechas de calendario.
- Ejecutado en esta auditoría: lectura, aritmética, validación de recibos y escritura de tablas. **0 entrenamientos, 0 forwards, 0 predicciones nuevas, 0 bootstrap y 0 pruebas de significancia**. El script usa solo la biblioteca estándar.

`protocol.json` conserva `execution_status=prepared_no_training_executed` y `new_real_data_fits=0`, como propuesta fuente anterior a la ejecución. `prepared.json` y `fit_status.assessment_started_by_run=false` son recibos de fases distintas. La evidencia de cierre es `pilot_result`, los 30 `job.json`, las curvas y los CSV; esos campos históricos no se reinterpretan como estado final. El protocolo denomina el control `floor_plus_fit_mean_excess` y el plan/CSV `floor_plus_mean_excess`; se conserva la identidad efectiva del CSV y se registra la diferencia de nombre.

## Métricas y comparaciones emparejadas

La métrica primaria conservada es la media de RMSE individuales por semilla. No es la raíz del MSE combinado ni el error de un ensamble. MAE se promedia con el mismo criterio. Para los controles analíticos hay una sola trayectoria, sin SD entre semillas.

| Rango | Receta | RMSE medio | SD muestral RMSE entre semillas | MAE medio |
|---:|---|---:|---:|---:|
{rank_rows}

Cada diferencia compara las mismas fechas y, para dos recetas neurales, la misma semilla. Ganancia positiva favorece `input_only_mse`; diferencias negativas de error también la favorecen.

| Control | Δ RMSE medio | Ganancia RMSE | Victorias neurales /5 | Δ MAE medio |
|---|---:|---:|---:|---:|
{pair_rows}

`metrics_recomputed.csv` contiene RMSE y MAE de cada trayectoria. `all_pairwise_seed_differences.csv` y `all_pairwise_summary.csv` cubren las 56 comparaciones dirigidas entre las ocho recetas/controles, sin excluir rivales. Cuando hay un control analítico se reutiliza su única trayectoria: las cinco diferencias no convierten ese control en cinco entrenamientos.

La SD de RMSE de `input_only_mse` es {s['input_only_SD_ratio_to_global']:.6f} veces la de `global_mse`. Esa variabilidad expresa inicialización/optimización condicionada al único panel de fechas; no estima por sí sola la incertidumbre temporal. La media de diferencias frente a global es {primary['mean_RMSE_difference']:.12g}; su SD entre semillas es {primary['sample_SD_seed_RMSE_difference']:.12g}. La ventaja no es uniforme: una semilla pierde frente a global y una frente a aditiva.

## Selección frente a evaluación expuesta

La selección comprende 54 fechas de 2024-07-08 a 2024-08-30; assessment comprende otras 55 fechas, ya expuestas antes del piloto. Se excluyeron siete fechas entre fit y selección, y siete entre selección y assessment (14 en total). Los 532 registros de fit y las purgas constan en admisión; sus arrays no están en el ZIP. El RMSE de selección se recupera de `sqrt(min(selection_mse_scaled))*y_scale` con desempate por época más temprana. Coincide con cada `job.json`; no se reconstruyen predicciones de selección ausentes.

| Receta | RMSE medio selección | Rango selección | RMSE assessment | Rango assessment neural | Épocas seleccionadas |
|---|---:|---:|---:|---:|---:|
{selection_rows}

**`input_only_mse` pasa del último puesto de selección al primero de assessment.** Todas las recetas tienen menor error en assessment, un cambio de dificultad o régimen compatible con las fechas distintas; no demuestra que la selección generalice mejor. Las curvas y las diferencias se documentan en `selection_assessment_recomputed.csv`, `selection_assessment_summary.csv` y `selection_assessment_input_only_pairs.csv`. La candidata se eligió después de examinar el desarrollo; este análisis posterior no es un preregistro ni elimina el sesgo de selección.

## Sensibilidad descriptiva a fechas y volatilidad realizada

Particiones posteriores al resultado: meses; primeras 27/últimas 28 fechas; ocho bloques consecutivos de siete días (último de seis); cuartiles del objetivo realizado `sigma7`. Los puntos de corte de los cuartiles (interpolación lineal) son {s['descriptive_target_quartile_cutpoints']}. Se usan las mismas fechas de cada partición para todas las recetas y se promedian RMSE individuales, preservando la definición de la métrica. `date_regimes_posthoc.csv` permite reconstruir cada subconjunto.

| Partición | Subconjunto | Fechas | Ganancia de input_only frente a global | Victorias /5 |
|---|---|---:|---:|---:|
{subset_rows}

La ventaja es más pequeña en la segunda mitad y se invierte en el cuartil de menor volatilidad realizada. Las particiones contienen pocas fechas y no son resultados independientes. El cuartil utiliza el objetivo futuro realizado: **es retrospectivo y no define una regla operativa causal**. La escala de la ventana conocida no está en las predicciones; no se inventan regímenes ex ante. Todos los bloques y controles están en `subset_*`.

Al quitar una sola fecha, la ganancia media RMSE frente a global oscila entre {100*loo['one_date']['min_gain']:.6f}% y {100*loo['one_date']['max_gain']:.6f}%; al quitar un bloque consecutivo, entre {100*loo['seven_day_block']['min_gain']:.6f}% y {100*loo['seven_day_block']['max_gain']:.6f}%. Se recalcula la métrica tras la exclusión: no es una nueva selección ni una validación cruzada. Véase `leave_out_sensitivity.csv`.

Los 275 pares semilla-fecha comparten objetivos y ventanas temporales. La autocorrelación Pearson de lag 1 del objetivo es {acf_target:.6f}; los diferenciales diarios promediados sobre semillas también están en `descriptive_dependence.csv`. Son descripciones, sin tamaño efectivo inferido. No se tratan 275 observaciones como independientes, no se calcula potencia confirmatoria con cinco semillas y un panel corto, ni se añaden p-valores o intervalos convencionales que ignoren selección y dependencia. `date_loss_differences.csv` conserva la contribución diaria de cada control.

## Fallos, presupuesto y recuperación

Los 30 trabajos presentan `completed_validated_fit`, una entrada de intento por trabajo y {b['interrupted_epochs']} épocas interrumpidas; `failures.json` está vacío. Los motivos de parada fueron {b['stop_reasons']}. Se verifican el número de épocas, época seleccionada, pasos Adam y concordancia del inventario de hashes de checkpoint con cada trabajo. Los pesos no están exportados: no se afirma haber verificado sus bytes ni reejecutado los modelos.

- Épocas totales: {b['epochs_total']}; pasos de optimizador comprometidos: {b['committed_optimizer_steps']}.
- Actividad registrada de ajuste: {b['recorded_active_seconds_total']:.9f} s; IO de checkpoint registrado: {b['recorded_checkpoint_IO_seconds_total']:.9f} s. Máximo por ajuste: {b['maximum_recorded_active_seconds_one_fit']:.9f} s, frente a 120 s nominales; techo nominal de 30 ajustes: {b['nominal_max_active_seconds']:.0f} s. No equivale al coste total de VM ni incluye instalación, preparación, transporte y respaldo.
- `remote_restore.json` declara restauración `{b['restore_commit']}` en `{b['restored_at_utc_as_recorded']}`, con sesión previa cerrada, identidad/semilla conservadas, protocolo/curvas/pesos sin modificación y sin cómputo remoto no observado reportado. `github_session.json` termina cerrado en `assessment_finished`. `fit_status` y resultado dicen `budget_comparability_verified=true`; la contabilidad disponible es consistente con esa declaración.
- El propio recibo advierte que una pérdida abrupta puede ocultar cómputo no subido. `unuploaded_progress_may_be_lost=true` es una posibilidad general; `last_saved_session_open=false` indica que esta recuperación no documenta rollback de una sesión abierta. No se infiere trabajo físico desconocido ni se convierte el respaldo en un libro externo completo. Tampoco se reconstruye el número de forwards por duplicados o por filas únicas.

`resources_by_recipe.csv`, `jobs_audited.csv` y `HASH_AND_RECOVERY_AUDIT.json` contienen el detalle. Los tiempos cortos son los registrados en CPU, no una estimación transportable para el futuro protocolo R3.

## Soporte, evidencia histórica y límites de lo verificable

La predicción `floor_only` permite contrastar el piso publicado en cada fecha. Hay {below} medias puntuales por debajo del piso con tolerancia 1e-12 (`point_support_audit.csv`). Esto no verifica la distribución atómica ni su calibración: no se exportaron masas, posiciones, CRPS, PIT, cobertura o intervalos. Una pérdida MSE por sí sola tampoco acreditaría calibración.

Las declaraciones históricas preservadas en `protocol.prior_evidence` son 796 trabajos aditivos, V3 con 30 ajustes, perjuicio 2,7043119888% y NO_GO, y V4 con 20 ajustes y ABANDON_CURRENT_V4_RECIPE. Sus resultados originales no están en este ZIP; esta auditoría no los reetiqueta ni afirma verificarlos de nuevo. El piloto V5 conserva su candidato original `seoi_mse`, mejora negativa frente a `input_only_mse`, 0/5 victorias y MAE por encima del máximo; las tres condiciones de continuación originales fracasan. No se aplican esas reglas retrospectivamente a la ablación ganadora.

La coincidencia numérica con `RESULT_READOUT.json` e `INPUT_ONLY_COMPARISON.json` es completa a tolerancia de coma flotante. No aparecen discrepancias numéricas materiales. La diferencia `0.0020515608242855483`/`0.0020515608242855487` en promedios obedece al orden/suma de coma flotante y está por debajo de 1e-18. Identificadores de commit, protocolo y hashes son trazabilidad del piloto, **no preregistro de la selección posterior de input_only**.

Reproducción: `python analysis/audit_saved_results.py --zip /ruta/public_results.zip --out analysis`. El script se niega a aceptar un ZIP con SHA distinto y termina con error ante discordancias comprobables. `ARTIFACT_HASHES.json` fija los archivos generados. El dictamen global debe conservar estos límites; una decisión de continuar solo abriría preparación de una candidata independiente y confirmación prospectiva, con R3 todavía no acreditada.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", "--out", dest="out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    audit(args.zip, args.out)
