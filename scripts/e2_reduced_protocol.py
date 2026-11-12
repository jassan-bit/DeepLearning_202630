"""Reduced E2 selection domains and criteria, without fitting or test access.

Six common *slots* retain deterministic indices of each recipe's previously
registered eighteen-point space. Numeric LR/WD/axes remain recipe-specific;
this is equal declared effort, not a claim of six identical architectures or
optimizer tuples. Current TiDE parent rows are provenance only until its
architecture is adjudicated. Historical results never become new weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = "experiments/e2_reduced_protocol_20261112"
PROTOCOL_PATH = DIRECTORY + "/protocol.json"
CONFIGURATIONS_PATH = DIRECTORY + "/configurations.json"
REGISTRATION_PATH = DIRECTORY + "/registration.json"
GIT_RECEIPT_PATH = DIRECTORY + "/git_registration.json"
PROTOCOL_ID = "e2-reduced-delivery-v1"
VERSION = "e2-reduced-protocol-v1"
PARENT = "52ea623fd76ceb1685cc44891316b48be3552f65"
DELTA = .00025772842664981106
HPO_SEED = 271201
FOLDS = ("2024-01-01", "2024-07-01", "2024-10-01")
PARENT_INDICES = (5, 7, 8, 9, 10, 15)
SLOT_IDS = tuple("C%02d" % (i + 1) for i in range(6))
TOP10 = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
         "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer")
NEURAL = TOP10 + ("MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SYSTEMS = NEURAL + CLASSICAL
RESTORED = {"LSTM-autoencoder con predictor": ("SAE-D7-PUBLISHED-v1", .3),
            "TimesNet": ("TN-D7-PUBLISHED-v1", .1),
            "TimeMixer": ("TM-D7-PUBLISHED-v1", .1),
            "PatchTST": ("PT-D7-PUBLISHED-v1", .3)}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _iso(value):
    _require(type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value),
             "Use an explicit ISO UTC calendar date")
    return date.fromisoformat(value)


def calendar_rows():
    """Three internal folds; their scores select one recipe for later E1 dates."""
    rows = []
    for text in FOLDS:
        origin = _iso(text)
        rows.append({"fold_origin": text,
            "inner_train_label_end": (origin - timedelta(days=98)).isoformat(),
            "inner_validation_start": (origin - timedelta(days=90)).isoformat(),
            "inner_validation_end": (origin - timedelta(days=8)).isoformat(),
            "embargo_days": 7, "validation_calendar_days": 83,
            "role": "internal_HPO_validation_not_final_test"})
    return rows


def planning_six_domains(registry=None):
    """Return 13 six-row domains without looking at results, costs or prices.

    ``registry`` may supply the full 18-row mappings for offline verification.
    With no argument, the exact existing published registry is imported lazily.
    Parent trial dictionaries stay byte-equivalent as canonical JSON, including
    their historical two-seed declaration; only ``execution_seed_ids`` determines
    the single admitted new HPO seed. No model is constructed here.
    """
    if registry is None:
        try:
            from .e2_published_registry import registered_trials
        except ImportError:
            from e2_published_registry import registered_trials
        registry = {name: registered_trials(name) for name in NEURAL}
    _require(isinstance(registry, Mapping) and set(registry) == set(NEURAL),
             "Provide the thirteen mandatory neural parent domains")
    result = {}
    for model in NEURAL:
        parents = list(registry[model])
        _require(len(parents) == 18 and len({p.get("trial_id") for p in parents}) == 18,
                 "Every parent domain requires eighteen unique registered points")
        result[model] = []
        for configuration_id, index in zip(SLOT_IDS, PARENT_INDICES):
            trial = deepcopy(parents[index])
            _require(isinstance(trial, dict) and isinstance(trial.get("trial_id"), str)
                     and trial["trial_id"].endswith(":hpo-%03d" % index)
                     and isinstance(trial.get("recipe_id"), str)
                     and trial.get("optimization_seeds") == [271201, 271202],
                     "Parent row must retain its exact stable identity and original seeds")
            if model in RESTORED:
                recipe, dropout = RESTORED[model]
                _require(trial["recipe_id"] == recipe and trial.get("dropout") == dropout,
                         "Keep the restored published recipe identity and fixed dropout")
            row = {"model_id": model, "configuration_id": configuration_id,
                   "parent_trial_index": index, "parent_trial_id": trial["trial_id"],
                   "trial": trial, "optimization_seed": HPO_SEED,
                   "execution_seed_ids": [HPO_SEED],
                   "architecture_status": "blocked_pending_architecture_adjudication" if model == "TiDE"
                       else "published_or_verified_fixed_body",
                   "execution_allowed": model != "TiDE",
                   "parent_seed_list_is_historical_not_execution_schedule": True}
            row["configuration_sha256"] = digest(row)
            result[model].append(row)
    validate_domains(result)
    return result


def validate_domains(domains):
    """Verify the frozen six-slot record; this does not certify author code."""
    if isinstance(domains, Mapping) and "models" in domains:
        domains = domains["models"]
    _require(isinstance(domains, Mapping) and set(domains) == set(NEURAL),
             "The reduced domain must name exactly thirteen neural systems")
    for model in NEURAL:
        rows = domains[model]
        _require(isinstance(rows, list) and len(rows) == 6,
                 "Every model requires six registered configuration slots")
        for row, slot, index in zip(rows, SLOT_IDS, PARENT_INDICES):
            _require(isinstance(row, Mapping), "Configuration rows must be mappings")
            plain = {k: v for k, v in row.items() if k != "configuration_sha256"}
            _require(row.get("configuration_sha256") == digest(plain),
                     "The reduced configuration hash changed")
            trial = row.get("trial", {})
            _require(row.get("model_id") == model and row.get("configuration_id") == slot
                     and row.get("parent_trial_index") == index
                     and row.get("parent_trial_id") == trial.get("trial_id")
                     and isinstance(trial.get("trial_id"), str)
                     and trial["trial_id"].endswith(":hpo-%03d" % index),
                     "Keep all six predetermined parent indices and slot identities")
            _require(row.get("optimization_seed") == HPO_SEED
                     and row.get("execution_seed_ids") == [HPO_SEED]
                     and trial.get("optimization_seeds") == [271201, 271202],
                     "The reduced execution admits one seed without rewriting parent history")
            blocked = model == "TiDE"
            _require(row.get("execution_allowed") is (not blocked)
                     and row.get("architecture_status") == ("blocked_pending_architecture_adjudication"
                         if blocked else "published_or_verified_fixed_body"),
                     "TiDE cannot be activated by relabelling its archived NAS parent")
            if model in RESTORED:
                recipe, dropout = RESTORED[model]
                _require(trial.get("recipe_id") == recipe and trial.get("dropout") == dropout,
                         "A restored body cannot inherit a retired recipe or altered dropout")
        _require({row["trial"].get("batch_size") for row in rows} == {16, 32, 64},
                 "The common slots must preserve all three registered batch sizes")
        decays = [row["trial"].get("weight_decay") for row in rows]
        _require(any(type(value) in (int, float) and value == 0 for value in decays)
                 and any(type(value) in (int, float) and value > 0 for value in decays),
                 "The common slots must preserve both zero and positive weight decay")
    return deepcopy(dict(domains))


def build_reduced_protocol(configurations):
    """Core metadata template for a new prospective registration, not a launch."""
    configurations = validate_domains(configurations)
    return {
        "protocol_id": PROTOCOL_ID, "schema_version": 1,
        "implementation_version": VERSION, "parent_source_commit": PARENT,
        "delivery_date_local": "2026-11-12", "display_timezone": "America/Bogota",
        "scope": "Prospective reduced continuation; prior data and results are known and disclosed",
        "systems": list(SYSTEMS), "neural_systems": list(NEURAL),
        "contract": {"input_shape": [7, 5], "horizon_days": 1,
            "target": "population sigma7 of daily Close log returns; ddof0; unannualized",
            "primary_metric": "mean_of_individual_seed_RMSE_original_units", "delta": DELTA,
            "target_scaling": "training mean only; invert before scoring",
            "embargo_days": 7},
        "hpo": {"configuration_ids": list(SLOT_IDS), "parent_trial_indices": list(PARENT_INDICES),
            "configuration_rule": "same six parent indices chosen from hyperparameter metadata only: batch16/32/64 and zero/positive WD coverage, then max-min normalized logLR span, sum of spans, and lexicographic tie; no outcome-based choice",
            "identical_numeric_optimizer_tuples_across_models": False,
            "seed_ids": [HPO_SEED], "fold_origins": list(FOLDS),
            "folds": calendar_rows(), "logical_fit_upper_bound_before_reuse": 234,
            "SAE_additional_auxiliary_optimizer_stages": 36,
            "optimizer_stage_upper_bound_before_reuse": 270,
            "selection_scope": "one_global_recipe_per_model_for_post_HPO_final_cohort",
            "selection_rule": "lowest original-unit RMSE on concatenated common validation predictions: sqrt(sum(n_fold*RMSE_fold**2)/sum(n_fold)); exact tie -> configuration_id",
            "validation_count_per_fold": 83, "concatenated_validation_count": 249,
            "mean_of_fold_RMSEs_is_selection_metric": False,
            "epoch_rule": "ceil(median(three winner strict-first-best epoch counts))",
            "SAE_phase_epoch_rule": "separate three-fold medians for ae1, ae2 and supervised; never discard auxiliary stages",
            "max_epochs": 200, "patience_epochs": 20,
            "failed_attempt_policy": "retain with reason, no replacement; fatal/systematic failure blocks selection",
            "earlier_outer_forecasts_using_future_global_recipe": False,
            "configuration_domains_sha256": digest(configurations)},
        "evaluation": {"start": "2025-01-01", "end": "2026-08-31",
            "calendar_days": 608, "role": "inherited_known_historical_E1_partition",
            "training_label_end": "2024-10-31",
            "training_to_calibration_bridge": ["2024-11-01", "2024-11-07"],
            "calibration_start": "2024-11-08", "calibration_end": "2024-12-30",
            "calibration_calendar_days": 53, "causal_bridge_after_calibration": ["2024-12-31"],
            "bridge_role": "causal observations only; no refit or recalibration",
            "calibration_dependence": "overlapping sigma7 targets; no iid/exchangeability or exact coverage guarantee",
            "previous_exposure_disclosed": True, "blindness_recovered": False,
            "new_100_day_cohort_required_for_this_delivery": False,
            "freeze_weights_scalers_calibrators_and_analysis_before_current_E2_scoring": True,
            "selection_or_retuning_from_final_scores_allowed": False},
        "replications": {"operative_R": 6, "seed_ids": list(range(271801, 271807)),
            "power_and_MDE_decision": "pending_development_pilot_before_final_scoring",
            "power_80_percent_claimed": False, "automatic_R_increase": False},
        "architecture": {"published_or_fixed_bodies": 12, "TiDE_execution_allowed": False,
            "TiDE_gate": "accept a separately verified published recipe before execution, otherwise the applicable NAS remains required",
            "six_TiDE_parent_rows_are_provenance_only": True},
        "budget": {"authorized_total_hours": 64.0, "budget_reset": False,
            "budget_extension_authorized": False,
            "remaining_seconds": None, "state": "current_ledger_reconciliation_required"},
        "execution": {"market_training_started": False, "final_market_values_read": False,
            "private_colab_modified": False, "GPU_verified_in_private_Colab": False,
            "ready": False, "scope": "registration and criteria only; actual execution gates remain separate"},
    }


def validate_reduced_protocol(protocol, configurations):
    expected = build_reduced_protocol(configurations)
    for key in ("protocol_id", "schema_version", "implementation_version", "parent_source_commit",
                "systems", "neural_systems", "contract", "hpo", "evaluation", "replications", "architecture"):
        _require(protocol.get(key) == expected[key], "Reduced protocol invariant changed: " + key)
    budget = protocol.get("budget", {})
    _require(budget.get("authorized_total_hours") == 64 and budget.get("budget_reset") is False
             and budget.get("budget_extension_authorized") is False,
             "The reduced design does not grant more than the authorized cumulative64 hours")
    return True


def select_reduced_hpo(ledger, model, configured_domain, *, final_origin="2025-01-01"):
    """Apply global three-fold criteria after all eighteen run slots are closed.

    A run row binds model_id, configuration_id, fold_origin, seed,
    parent_trial_id, configuration_sha256, status and run_receipt_sha256.
    Every row records the verified common validation_count83. Successful rows
    additionally need rmse_original and best_epoch. SAE rows
    also require phase_best_epochs for ae1, ae2 and supervised. This pure
    function checks metadata and selection arithmetic, not actual artifact bytes.
    """
    _require(model in NEURAL and model != "TiDE", "TiDE remains blocked pending architecture adjudication")
    _require(_iso(final_origin) >= _iso("2025-01-01"),
             "A global three-fold recipe cannot forecast an earlier outer period")
    if isinstance(configured_domain, Mapping):
        domains = configured_domain.get("models", configured_domain)
        configured_domain = validate_domains(domains)[model]
    domain = list(configured_domain)
    _require(len(domain) == 6 and tuple(row.get("configuration_id") for row in domain) == SLOT_IDS,
             "Supply the six ordered, hash-bound configuration rows for this model")
    by_slot = {}
    for row, index in zip(domain, PARENT_INDICES):
        plain = {k: v for k, v in row.items() if k != "configuration_sha256"}
        trial = row.get("trial", {})
        _require(row.get("model_id") == model and row.get("parent_trial_index") == index
                 and row.get("configuration_sha256") == digest(plain)
                 and row.get("optimization_seed") == HPO_SEED and row.get("execution_allowed") is True
                 and row.get("execution_seed_ids") == [HPO_SEED]
                 and row.get("architecture_status") == "published_or_verified_fixed_body"
                 and row.get("parent_trial_id") == trial.get("trial_id")
                 and isinstance(trial.get("trial_id"), str)
                 and trial["trial_id"].endswith(":hpo-%03d" % index)
                 and trial.get("optimization_seeds") == [271201, 271202],
                 "The supplied domain changed its model, parent position, seed or hash")
        if model in RESTORED:
            recipe, dropout = RESTORED[model]
            _require(trial.get("recipe_id") == recipe and trial.get("dropout") == dropout,
                     "A restored body cannot select a retired recipe or changed dropout")
        by_slot[row["configuration_id"]] = row
    rows = list(ledger)
    _require(len(rows) == 18, "All six configurations by three folds require eighteen explicit run rows")
    seen, successes, failures = set(), {}, {}
    for row in rows:
        _require(isinstance(row, Mapping), "Run ledger rows must be mappings")
        slot, fold = row.get("configuration_id"), row.get("fold_origin")
        _require(slot in by_slot and fold in FOLDS, "Foreign configuration or fold in reduced HPO ledger")
        key = (slot, fold)
        _require(key not in seen, "Duplicate reduced HPO configuration/fold")
        seen.add(key)
        configured = by_slot[slot]
        _require(row.get("model_id") == model and row.get("seed") == HPO_SEED
                 and row.get("parent_trial_id") == configured["parent_trial_id"]
                 and row.get("configuration_sha256") == configured["configuration_sha256"],
                 "Run identity differs from the registered reduced configuration")
        _require(type(row.get("validation_count")) is int and row["validation_count"] == 83,
                 "Each fold must retain its verified common83 validation dates")
        _require(isinstance(row.get("run_receipt_sha256"), str)
                 and SHA.fullmatch(row["run_receipt_sha256"]), "Every attempt needs a bound run receipt")
        use = row.get("resource_use", {})
        _require(isinstance(use, Mapping), "Run resource use must be a mapping")
        flags = [source.get(key, False) for source in (row, use) for key in ("fatal", "systematic_failure")]
        _require(all(type(flag) is bool for flag in flags), "Failure flags must be explicit booleans")
        _require(not any(flags), "Fatal or systematic failure blocks reduced selection")
        status = row.get("status")
        if status == "failed":
            _require(isinstance(row.get("reason"), str) and bool(row["reason"].strip()),
                     "Failed attempts remain in the ledger with a reason")
            failures[key] = row["reason"]
        else:
            score, epoch = row.get("rmse_original"), row.get("best_epoch")
            _require(status == "completed" and type(score) in (int, float)
                     and math.isfinite(score) and score >= 0,
                     "Completed attempts need finite nonnegative original-unit RMSE")
            _require(type(epoch) is int and 1 <= epoch <= 200, "Best epoch must lie within1..200")
            phases = row.get("phase_best_epochs", {"supervised": epoch})
            required_phases = {"ae1", "ae2", "supervised"} if model == "LSTM-autoencoder con predictor" else {"supervised"}
            _require(isinstance(phases, Mapping) and set(phases) == required_phases
                     and all(type(value) is int and 1 <= value <= 200 for value in phases.values())
                     and phases["supervised"] == epoch,
                     "Preserve the strict-best epoch of every admitted optimizer phase")
            successes[key] = (float(score), epoch, dict(phases), row["validation_count"])
    expected = {(slot, fold) for slot in SLOT_IDS for fold in FOLDS}
    _require(seen == expected, "The reduced ledger is incomplete")
    candidates = []
    for slot in SLOT_IDS:
        failed = [fold for fold in FOLDS if (slot, fold) in failures]
        if failed:
            candidates.append({"configuration_id": slot, "status": "failed",
                "failed_folds": failed, "replacement_performed": False})
            continue
        pairs = [successes[(slot, fold)] for fold in FOLDS]
        count = sum(p[3] for p in pairs)
        pooled_mse = sum(p[3] * p[0]**2 for p in pairs) / count
        _require(math.isfinite(pooled_mse), "Concatenated validation score overflow")
        candidates.append({"configuration_id": slot, "status": "completed",
            "pooled_cv_RMSE_original": math.sqrt(pooled_mse),
            "mean_fold_RMSE_original_diagnostic": statistics.mean(p[0] for p in pairs),
            "validation_count": count,
            "fixed_epochs": int(math.ceil(statistics.median(p[1] for p in pairs))),
            "fixed_phase_epochs": {phase: int(math.ceil(statistics.median(p[2][phase] for p in pairs)))
                for phase in pairs[0][2]},
            "fold_RMSE_original": {fold: successes[(slot, fold)][0] for fold in FOLDS}})
    complete = [row for row in candidates if row["status"] == "completed"]
    _require(bool(complete), "No complete three-fold configuration remains")
    winner = min(complete, key=lambda row: (row["pooled_cv_RMSE_original"], row["configuration_id"]))
    return {"version": VERSION, "model_id": model, "winner_configuration_id": winner["configuration_id"],
        "winner_parent_trial_id": by_slot[winner["configuration_id"]]["parent_trial_id"],
        "pooled_cv_RMSE": winner["pooled_cv_RMSE_original"],
        "winner_pooled_cv_RMSE_original": winner["pooled_cv_RMSE_original"],
        "winner_mean_fold_RMSE_original_diagnostic": winner["mean_fold_RMSE_original_diagnostic"],
        "validation_count": winner["validation_count"],
        "fixed_epochs": winner["fixed_epochs"], "fixed_phase_epochs": winner["fixed_phase_epochs"],
        "candidate_ledger": candidates,
        "all_eighteen_slots_accounted_for": True, "configuration_count": 6, "fold_count": 3,
        "selection_scope": "one_global_recipe_for_post_HPO_final_cohort",
        "earliest_final_origin": "2025-01-01", "criteria_only": True,
        "run_artifact_bytes_verified": False, "new_training_performed": False}


def _path(root, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Use a repository-relative registered path")
    root = Path(root).resolve()
    path = root / relative
    _require(path.resolve().is_relative_to(root)
             and not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)),
             "Registered paths may not escape the root or follow symlinks")
    return path


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reduced_registration(root=ROOT, *, require_git=False):
    """Check frozen bytes and optionally actual immutable Git registration time."""
    root = Path(root)
    protocol = json.loads(_path(root, PROTOCOL_PATH).read_text())
    configured = json.loads(_path(root, CONFIGURATIONS_PATH).read_text())
    domains = configured.get("models", configured) if isinstance(configured, Mapping) else configured
    validate_reduced_protocol(protocol, domains)
    registration = json.loads(_path(root, REGISTRATION_PATH).read_text())
    _require(registration.get("protocol_id") == PROTOCOL_ID
             and registration.get("parent_source_commit") == PARENT
             and registration.get("protocol_sha256") == file_digest(_path(root, PROTOCOL_PATH)),
             "Reduced registration identity or protocol hash changed")
    sources = registration.get("files_sha256", {})
    required = {PROTOCOL_PATH, CONFIGURATIONS_PATH, "scripts/e2_reduced_protocol.py"}
    _require(isinstance(sources, Mapping) and required.issubset(sources),
             "Bind the reduced protocol, configurations and criteria implementation")
    for relative, expected in sources.items():
        _require(isinstance(expected, str) and SHA.fullmatch(expected)
                 and file_digest(_path(root, relative)) == expected,
                 "Reduced registered source bytes changed: " + str(relative))
    proof = {"verified": False, "reason": "actual immutable Git registration not requested"}
    if require_git:
        receipt = json.loads(_path(root, GIT_RECEIPT_PATH).read_text())
        commit = receipt.get("registration_commit")
        _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
                 "Bind a complete immutable registration commit")
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
        stamp = git("show", "-s", "--format=%cI", commit).decode().strip()
        actual = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        claimed = datetime.fromisoformat(str(receipt.get("committed_at_utc", "")).replace("Z", "+00:00"))
        _require(actual.tzinfo is not None and claimed.tzinfo is not None and actual == claimed,
                 "Git registration time must equal the actual immutable commit time")
        for relative in (PROTOCOL_PATH, REGISTRATION_PATH):
            _require(git("show", commit + ":" + relative) == _path(root, relative).read_bytes(),
                     "Immutable registration commit contains different bytes: " + relative)
        headers = git("cat-file", "-p", commit).decode().split("\n\n", 1)[0].splitlines()
        _require("parent " + PARENT in headers, "Reduced registration has a different source parent")
        _require(receipt.get("protocol_sha256") == registration["protocol_sha256"]
                 and receipt.get("registration_receipt_sha256") == file_digest(_path(root, REGISTRATION_PATH)),
                 "Git receipt refers to another reduced registration")
        proof = {"verified": True, "registration_commit": commit,
                 "committed_at_utc": actual.astimezone(timezone.utc).isoformat()}
    return {"protocol_id": PROTOCOL_ID, "protocol_sha256": registration["protocol_sha256"],
        "source_bytes_verified": len(sources), "git_registration": proof,
        "planned_hpo_logical_fits_before_reuse": 234, "market_training_started": False,
        "private_colab_GPU_verified": False, "execution_ready": False,
        "TiDE_architecture_gate_closed": False, "budget_extended": False}
