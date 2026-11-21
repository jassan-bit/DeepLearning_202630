"""Versioned, stdlib-only metadata preflight. Never runs scientific workloads.

Technical permits are conditional attestations for a separate reviewed runner;
receipt provenance and scientific conclusions remain external responsibilities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import date, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def unique_strings(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(x, str) and x for x in value)
            and len(value) == len(set(value)))


def dates(value):
    need(unique_strings(value), "nonempty unique ISO target dates required")
    parsed = [date.fromisoformat(x) for x in value]
    need(value == [x.isoformat() for x in parsed] and parsed == sorted(parsed),
         "canonical increasing target dates required")
    return parsed


def seeds(value, minimum):
    need(isinstance(value, list) and len(value) >= minimum
         and all(integer(x, 0) for x in value) and len(set(value)) == len(value),
         "distinct integer seeds required; bool/float labels are invalid")
    return value


def instant(value):
    x = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(x.utcoffset() is not None, "timezone required")
    return x


def inherited(repo_root):
    """Import exactly pinned existing helpers, without re-running its audit."""
    pins = json.loads((HERE / "DEPENDENCIES.json").read_text())
    resolved = {}
    root = Path(repo_root).resolve()
    for name, binding in pins["imports"].items():
        p = root / binding["path"]
        current = root
        for part in Path(binding["path"]).parts:
            current = current / part
            need(not current.is_symlink(), "symlink legacy import")
        need(not Path(binding["path"]).is_absolute() and p.resolve().is_relative_to(root), "unsafe legacy import")
        need(p.is_file() and p.stat().st_size == binding["bytes"]
             and hashlib.sha256(p.read_bytes()).hexdigest() == binding["sha256"],
             "pinned legacy dependency differs: " + name)
        resolved[name] = p
    spec = importlib.util.spec_from_file_location("r3_pinned_power_helpers", resolved["module"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads(resolved["template"].read_text())


def positive_semidefinite(matrix, size, tolerance):
    """Numerical LDL check; tolerance is relative to largest diagonal scale."""
    need(isinstance(matrix, list) and len(matrix) == size
         and all(isinstance(row, list) and len(row) == size
                 and all(number(x) for x in row) for row in matrix), "covariance dimensions/values")
    need(number(tolerance) and 0 <= tolerance <= 1e-8, "invalid numerical tolerance")
    scale = max([abs(matrix[i][i]) for i in range(size)] + [1e-300])
    eps = tolerance * scale
    for i in range(size):
        need(matrix[i][i] >= -eps, "negative covariance diagonal")
        for j in range(size):
            need(abs(matrix[i][j] - matrix[j][i]) <= eps, "asymmetric covariance")
    work = [row[:] for row in matrix]
    for k in range(size):
        pivot = work[k][k]
        need(pivot >= -eps, "indefinite covariance")
        if pivot <= eps:
            need(all(abs(work[k][j]) <= eps for j in range(k + 1, size)),
                 "nonzero covariance with zero conditional variance")
            continue
        for i in range(k + 1, size):
            for j in range(i, size):
                work[i][j] -= work[k][i] * work[k][j] / pivot
                work[j][i] = work[i][j]


def material_cost(design):
    """Count real recipe jobs; never infer times from abandoned recipes."""
    family, candidate = design.get("family"), design.get("candidate")
    if not isinstance(family, dict) or not isinstance(candidate, dict):
        return {"current_panel_rival_cells": None, "current_procedure_components": None,
                "planning_fit_jobs": None, "confirmation_fit_jobs": None,
                "serial_fit_cap_seconds": None, "estimated_elapsed_seconds": None,
                "reason": "candidate/family/resources unset; no measured runtime extrapolation"}
    systems = [{"id": candidate["version"], "kind": "stochastic"}] + family["rivals"]
    p = len(design.get("panels", []))
    stochastic = sum(x["kind"] == "stochastic" for x in systems)
    deterministic = sum(x["kind"] == "deterministic" for x in systems)
    sampling = design.get("sampling", {})
    pilot = sampling.get("planning_seed_ids")
    final = sampling.get("final_seed_ids")
    return {"current_panel_rival_cells": p * len(family["rivals"]),
            "current_procedure_components": p * len(family["rivals"]) * len(design.get("methods", [])),
            "planning_fit_jobs": p * (stochastic * len(pilot) + deterministic) if isinstance(pilot, list) else None,
            "confirmation_fit_jobs": p * (stochastic * len(final) + deterministic) if isinstance(final, list) else None,
            "stochastic_systems": stochastic, "deterministic_systems": deterministic,
            "HPO_fit_jobs": design.get("resources", {}).get("additional_HPO_fit_jobs"),
            "serial_fit_cap_seconds": None, "estimated_elapsed_seconds": None,
            "time_claim": "caps require explicit inputs; elapsed time needs measured schedule evidence"}


def assess(config, repo_root, evidence_root, phase="confirmation"):
    need(phase in ("planning", "confirmation"), "unknown phase")
    old, base = inherited(repo_root)
    design = config.get("design", {})
    need(isinstance(design, dict), "design object required")
    target = {k: base[k] for k in ("estimand", "units", "seed_uncertainty", "calendar_uncertainty")}
    target.update(design.get("target", {}))
    design_hash = sha(design)
    checks, errors, cache = {}, {}, {}
    candidate = design.get("candidate") or {}
    family = design.get("family") or {}
    panels, methods = design.get("panels", []), design.get("methods", [])
    rivals = family.get("rivals", [])
    rival_ids = [x.get("id") for x in rivals]
    panel_ids = [x.get("id") for x in panels]
    method_ids = [x.get("id") for x in methods]
    candidate_hash = (candidate.get("artifact") or {}).get("sha256")
    family_hash = sha(family)
    bindings = config.get("evidence", {})

    def check(name, callback):
        try:
            callback()
            checks[name] = True
        except (ValueError, TypeError, KeyError, IndexError, OSError, AttributeError) as exc:
            checks[name] = False
            errors[name] = str(exc)

    def bound(binding):
        return old.bound_json(evidence_root, binding)

    def receipt(name, scoped=True):
        if name not in cache:
            cache[name] = bound(bindings.get(name))
        value = cache[name]
        need(value.get("candidate_version") == candidate.get("version")
             and value.get("candidate_sha256") == candidate_hash and bool(candidate_hash),
             name + ": candidate identity mismatch")
        if scoped:
            need(value.get("design_sha256") == design_hash
                 and value.get("family_sha256") == family_hash
                 and all(value.get(k) == v for k, v in target.items()),
                 name + ": design/family/estimand/units mismatch")
        return value

    def check_candidate():
        need(isinstance(candidate.get("version"), str) and candidate["version"], "candidate unset")
        artifact = old.safe_file(evidence_root, candidate["artifact"]["path"])
        need(old.digest(artifact) == candidate_hash, "candidate bytes differ")
        for name in ("r1", "r2"):
            x = receipt(name, False)
            need(x.get("decision") == "accredited_for_this_candidate"
                 and x.get("reviewer") and x.get("external_evidence_uri")
                 and instant(x["reviewed_at"]), name + ": own authentic assessment missing")
        x = receipt("candidate_decision", False)
        need(x.get("planning_allowed") is True and x.get("recipe_abandoned") is False,
             "candidate is abandoned or lacks a continuation decision")

    def check_family():
        need(unique_strings(rival_ids) and candidate.get("version") not in rival_ids,
             "unique explicit rivals required")
        top, strong = family.get("top10", []), family.get("strong_baselines", [])
        need(unique_strings(top) and len(top) == 10 and unique_strings(strong)
             and set(top + strong + family.get("other_declared_rivals", [])) == set(rival_ids),
             "Top10/strong-baseline roster coverage required; no inherited 19/21")
        for rival in rivals:
            need(rival.get("kind") in ("stochastic", "deterministic"), "system kind missing")
            bound(rival["recipe"])
        need(len(method_ids) == 3 and unique_strings(method_ids), "three distinct procedures required")
        need(any(x.get("global_or_multiple") is True for x in methods), "one global/multiple procedure required")
        definitions, actual_components, global_components = [], set(), set()
        expected_pairs = {p + "|" + r for p in panel_ids for r in rival_ids}
        for method in methods:
            definition = bound(method["definition"])
            implementation = old.safe_file(evidence_root, method["implementation"]["path"])
            need(old.digest(implementation) == method["implementation"]["sha256"], "method implementation hash mismatch")
            need(definition.get("method_id") == method["id"]
                 and definition.get("estimand") == target["estimand"]
                 and definition.get("sampling_population") == [target["seed_uncertainty"], target["calendar_uncertainty"]]
                 and definition.get("null") and definition.get("direction") == "candidate_better"
                 and definition.get("assumptions"), "method definition not aligned")
            mapping = bound(method["contrast_mapping"])
            need(mapping.get("rival_ids") == rival_ids and mapping.get("method_id") == method["id"]
                 and mapping.get("translation") in ("direct_contrast", "justified_global_to_contrast")
                 and mapping.get("derivation"), "global p-value duplication is not a mapping")
            mapped = mapping.get("component_keys_by_panel_rival", {})
            need(set(mapped) == expected_pairs and all(unique_strings(keys) for keys in mapped.values()),
                 "each panel/rival needs an explicit mapping to actual raw hypotheses")
            flat = [key for keys in mapped.values() for key in keys]
            if mapping["translation"] == "direct_contrast":
                need(len(flat) == len(set(flat)), "a direct contrast cannot repeat a global raw p-value")
            else:
                need(method.get("global_or_multiple") is True
                     and mapping.get("directional_exclusion_construction")
                     and mapping.get("global_p_repeated_as_contrast") is False,
                     "global-to-contrast mapping needs a validated exclusion construction")
                global_components.update(flat)
            actual_components.update(flat)
            definitions.append(method["definition"]["sha256"])
        need(len(set(definitions)) == 3, "three aliases of one definition are not three methods")
        expected = {p + "|" + r + "|" + m for p in panel_ids for r in rival_ids for m in method_ids}
        groups = family.get("multiplicity_groups", [])
        need(unique_strings([g.get("id") for g in groups]), "unique multiplicity group IDs required")
        need(all(g["id"] != "complete_decision_rule" for g in groups), "reserved calibration scope ID")
        components = [key for group in groups for key in group["components"]]
        for group in groups:
            need(group.get("adjustment") in ("Holm", "intrinsic_global", "predeclared_validated_joint")
                 and number(group.get("alpha")) and 0 < group["alpha"] < 1
                 and group.get("justification"), "multiplicity adjustment/level/justification missing")
            need((group["adjustment"] == "intrinsic_global" and set(group["components"]).issubset(global_components))
                 or group["alpha"] <= .05, "a level above .05 requires an explicitly mapped intrinsic global role")
        need(bool(expected) and len(components) == len(actual_components) and set(components) == actual_components,
             "exact multiplicity coverage required, without omitted/duplicated components")
        rule = bound(family["decision_rule"])
        rule_code = old.safe_file(evidence_root, rule["implementation"]["path"])
        need(old.digest(rule_code) == rule["implementation"]["sha256"], "decision rule code hash differs")
        need(rule.get("method_ids") == method_ids and rule.get("component_keys") == sorted(actual_components)
             and rule.get("contrast_method_keys") == sorted(expected)
             and rule.get("requires_all_three") is True and rule.get("includes_direction_margin_effect_CI_adequacy_robustness") is True,
             "complete reviewed decision rule required")
        control = rule["complete_rule_error_control"]
        need(control.get("scope") == "complete_decision_rule"
             and control.get("event") == "any_false_rejection_complete_declared_family"
             and number(control.get("alpha")) and 0 < control["alpha"] <= .05
             and control.get("justification"), "explicit primary complete-rule error-control level required")

    def check_panels():
        need(len(panels) == 4 and unique_strings(panel_ids), "main plus three additional panels required")
        x = receipt("admission")
        need(x.get("admitted_panel_ids") == panel_ids and len(x.get("panels", [])) == 4, "0/4 proposals are not admitted")
        admissions = {a["id"]: a for a in x["panels"]}
        for index, panel in enumerate(panels):
            a = admissions[panel["id"]]
            need(panel.get("task") == "forecast_unannualized_sigma7"
                 and type(panel.get("forecast_horizon")) is int and panel["forecast_horizon"] == 1
                 and type(panel.get("target_close_support")) is int and panel["target_close_support"] == 8,
                 "same task, horizon1, eight-close target support required")
            need(a.get("license") and a.get("source_uri") and a.get("exposure_declaration_uri")
                 and a.get("calendar") and a.get("volume_units") and a.get("dependence_review"),
                 "access/exposure/license/calendar/dependence evidence missing")
            if index:
                need(unique_strings(a.get("difference_axes")) and len(a["difference_axes"]) >= 2,
                     "each additional dataset needs two justified diversity axes")
            planning, confirmation = dates(panel["planning_dates"]), dates(panel["confirmation_dates"])
            need(set(planning).isdisjoint(confirmation), "planning and final targets overlap")
            p_support = a["planning_support"]
            f_support = a["confirmation_support"]
            need(date.fromisoformat(p_support[0]) <= planning[0] <= planning[-1] == date.fromisoformat(p_support[1])
                 and date.fromisoformat(f_support[0]) <= confirmation[0] <= confirmation[-1] == date.fromisoformat(f_support[1])
                 and date.fromisoformat(p_support[1]) < date.fromisoformat(f_support[0]),
                 "raw close supports overlap or do not cover declared target dates")
            for part in (("planning",) if phase == "planning" else ("planning", "confirmation")):
                data = old.safe_file(evidence_root, a[part + "_data"]["path"])
                need(old.digest(data) == a[part + "_data"]["sha256"] == panel["data_sha256"][part],
                     "admitted dataset hash differs from bytes or frozen design")
                need(a[part + "_partition_used_in_other_phase"] is False, "partition was reused")

    def check_target_and_sampling():
        need(target == {k: base[k] for k in target}, "MSE/pooled RMSE/units cannot silently replace mean individual RMSE")
        sampling = design["sampling"]
        seeds(sampling.get("planning_seed_ids"), 5)
        need(sampling.get("seed_x_date_IID") is False, "crossed seed/date rows are not IID")
        for panel in panels:
            margin = panel["practical_margin"]
            alt = panel["planning_alternative"]
            need(number(margin) and margin >= 0 and number(alt) and alt > margin
                 and panel.get("margin_units") == target["units"]
                 and panel.get("margin_rationale"), "delta and alternative must be distinct and justified in metric units")
        need(design.get("finite_calendar_bias_statement"), "document population/finite-T target and R/T bias limitation")

    def check_calibration():
        x = receipt("calibration")
        need(x.get("decision_rule_sha256") == family["decision_rule"]["sha256"]
             and x.get("decision_rule_implementation_sha256") == bound(family["decision_rule"])["implementation"]["sha256"]
             and x.get("method_definition_sha256s") == [m["definition"]["sha256"] for m in methods]
             and x.get("method_implementation_sha256s") == [m["implementation"]["sha256"] for m in methods],
             "calibration for different code/rule")
        r_values = [len(seeds(design["sampling"]["planning_seed_ids"], 5))]
        t_values = [len(p["planning_dates"]) for p in panels]
        if phase == "confirmation":
            r_values.append(design["sampling"]["final_R"])
            t_values += [len(p["confirmation_dates"]) for p in panels]
        scope = x["validated_R_T_scope"]
        need(all(integer(r) and scope["R_range"][0] <= r <= scope["R_range"][1] for r in r_values)
             and all(integer(t) and scope["T_range"][0] <= t <= scope["T_range"][1] for t in t_values)
             and scope.get("population_null_and_finite_calendar_target")
             and scope.get("restrictions_and_extrapolation_review"), "calibration does not cover actual joint R/T/target scope")
        protocol = bound(x["frozen_validation_protocol"])
        need(instant(protocol["registered_at"]) < instant(x["first_validation_result_at"])
             and protocol.get("external_registration_uri") and protocol.get("acceptance_criteria"),
             "calibration criteria were not externally frozen before validation")
        need(protocol.get("decision_rule_sha256") == family["decision_rule"]["sha256"]
             and protocol.get("family_sha256") == family_hash,
             "frozen calibration criteria refer to another rule/family")
        required = {"complete_null", "partial_null", "seed_only", "calendar_only", "shared_shocks",
                    "heavy_tails", "degeneracy", "heterogeneous_dependence_population_equal"}
        rows = x.get("strong_control_rows", [])
        need(len({(r["scenario"], r["criterion_id"]) for r in rows}) == len(rows), "duplicate calibration scenario/criterion")
        eta = protocol["MC_error_probability"]
        need(number(eta) and 0 < eta < 1, "frozen calibration MC error probability missing")
        rule_control = bound(family["decision_rule"])["complete_rule_error_control"]
        need(number(rule_control.get("alpha")) and 0 < rule_control["alpha"] <= .05,
             "primary complete-rule error-control level missing")
        groups = {g["id"]: g for g in family["multiplicity_groups"]}
        covered = {"complete_decision_rule": set(), **{key: set() for key in groups}}
        for row in rows:
            criterion = protocol["acceptance_criteria"][row["criterion_id"]]
            if criterion.get("scope") == "complete_decision_rule":
                level = rule_control["alpha"]
                event = rule_control["event"]
                scope_key = "complete_decision_rule"
                need(criterion.get("decision_rule_sha256") == family["decision_rule"]["sha256"],
                     "complete-rule criterion hash mismatch")
            elif criterion.get("scope") == "multiplicity_group":
                group_id = criterion["group_id"]
                level = groups[group_id]["alpha"]
                event = "any_false_rejection_in_multiplicity_group"
                scope_key = group_id
                need(row.get("group_id") == group_id, "calibration row belongs to another multiplicity group")
            else:
                raise ValueError("explicit complete-rule or multiplicity-group calibration scope required")
            need(number(criterion.get("maximum_error_rate"))
                 and criterion.get("nominal_alpha") == level
                 and 0 < criterion["maximum_error_rate"] <= level
                 and row.get("scope") == criterion["scope"],
                 "calibration ceiling/nominal alpha differs from exact declared rule or group level")
            n, errors = row.get("independent_replicates"), row.get("false_rejection_replicates")
            need(integer(n) and integer(errors, 0) and errors <= n
                 and row.get("event") == event,
                 "familywise validation event/count missing")
            upper = min(1., errors/n + math.sqrt(math.log(len(rows)/eta)/(2*n)))
            need(number(row.get("simultaneous_MC_upper"))
                 and math.isclose(row["simultaneous_MC_upper"], upper, rel_tol=1e-10, abs_tol=1e-12)
                 and 0 <= upper <= criterion["maximum_error_rate"] <= 1,
                 "strong size/FWER criterion failed or MC uncertainty missing")
            covered[scope_key].add(row["scenario"])
        need(all(required.issubset(scenarios) for scenarios in covered.values()),
             "strong calibration must cover complete rule and each exact multiplicity group separately")
        need(x.get("reviewer") and x.get("external_review_uri") and x.get("finite_calendar_bias_scope"),
             "calibration scope and finite-calendar bias review required")

    def check_freeze():
        name = "planning_registration" if phase == "planning" else "confirmation_registration"
        x = receipt(name)
        record = bound(x["archived_design"])
        need(sha(record) == design_hash and x.get("external_registration_uri")
             and x.get("immutable_revision") and x.get("custodian"), "local DRAFT/hash is not a real preregistration")
        registered = instant(x["registered_at"])
        for panel in receipt("admission")["panels"]:
            access = instant(panel[phase + "_first_outcome_access_at"])
            need(registered < access, "registration occurred after first outcome access")
        need(instant(x["reviewed_at"]) >= registered and x.get("external_review_uri"),
             "external order-of-access review missing")

    jobs = []

    def check_resources_and_persistence():
        resource = design["resources"]
        need(integer(resource.get("max_fit_attempts")) and integer(resource.get("max_wall_seconds"))
             and resource.get("hardware") and resource.get("software_environment_sha256"), "resource ceilings/cohort missing")
        sample = design["sampling"]["planning_seed_ids" if phase == "planning" else "final_seed_ids"]
        seeds(sample, 5 if phase == "planning" else 10)
        systems = [{"id": candidate["version"], "kind": "stochastic"}] + rivals
        serial_cap = 0
        for panel in panel_ids:
            for system in systems:
                ids = sample if system["kind"] == "stochastic" else [None]
                cap = resource["per_fit_cap_seconds"][system["id"]]
                need(integer(cap), "explicit per-recipe fit ceiling required; not an elapsed-time estimate")
                for seed in ids:
                    job = {"phase": phase, "panel": panel, "system": system["id"], "seed": seed,
                           "design_sha256": design_hash, "max_fit_seconds": cap}
                    job["job_id"] = sha(job)
                    jobs.append(job)
                    serial_cap += cap
        need(integer(resource.get("additional_HPO_fit_jobs"), 0), "explicit HPO accounting required")
        need(resource["additional_HPO_fit_jobs"] == 0,
             "this fixed-recipe runner plan excludes HPO; finish selection and freeze recipes first")
        need(len(jobs) <= resource["max_fit_attempts"], "fit count exceeds authorized ceiling")
        material["serial_fit_cap_seconds"] = serial_cap
        auth = config.get("campaign_authorization")
        a = bound(auth)
        need(a.get("design_sha256") == design_hash and a.get("phase") == phase
             and a.get("authorizer") and a.get("external_authorization_uri")
             and integer(a.get("max_fit_attempts")) and a["max_fit_attempts"] >= resource["max_fit_attempts"]
             and integer(a.get("max_wall_seconds")) and a["max_wall_seconds"] >= resource["max_wall_seconds"],
             "explicit budget authorization for exact design/phase missing")
        persistence = design["persistence"]
        need(persistence.get("verified_checkpoint_required") is True
             and persistence.get("skip_verified_complete") is True
             and persistence.get("overwrite_existing") is False
             and persistence.get("storage_uri") and persistence.get("atomic_write") is True,
             "durable, atomic, no-overwrite persistence contract required")
        runner = bound(bindings.get("runner_review"))
        runner_code = old.safe_file(evidence_root, runner["runner_code"]["path"])
        need(old.digest(runner_code) == runner["runner_code"]["sha256"], "reviewed runner code hash differs")
        need(runner.get("design_sha256") == design_hash and runner.get("reviewer")
             and runner.get("external_evidence_uri")
             and runner.get("enforces_wall_and_attempt_caps") is True
             and runner.get("checkpoint_integrity_and_resume_tested") is True,
             "reviewed separate runner is required; preflight is not a trainer")

    def check_covariance_and_design():
        x = receipt("paired_covariance")
        cube = bound(x["crossed_covariance_payload"])
        expected = [p + "|" + r for p in panel_ids for r in rival_ids]
        need(cube.get("matrix_order") == expected and cube.get("variance_units") == target["units"] + "^2",
             "covariance order/units mismatch")
        need(cube.get("planning_seed_ids") == design["sampling"]["planning_seed_ids"]
             and cube.get("dates_by_panel") == {p["id"]: p["planning_dates"] for p in panels}
             and cube.get("date_seed_cells_are_IID") is False, "covariance sampling axes differ")
        seeds(cube["planning_seed_ids"], 5)
        observations = bound(cube["paired_observations"])
        expected_axes = {(p["id"], r, seed, day) for p in panels for r in rival_ids
                         for seed in cube["planning_seed_ids"] for day in p["planning_dates"]}
        rows = observations["rows"]
        axes = [(row["panel"], row["rival"], row["seed"], row["date"]) for row in rows]
        need(len(axes) == len(expected_axes) and set(axes) == expected_axes
             and all(integer(row["seed"], 0) and number(row["candidate_squared_error"])
                     and number(row["rival_squared_error"]) and row["candidate_squared_error"] >= 0
                     and row["rival_squared_error"] >= 0 for row in rows),
             "paired source cube is missing/duplicated/nonfinite; do not impute cells")
        indexed = {(row["panel"], row["rival"], row["seed"], row["date"]): row for row in rows}
        by_seed = {}
        for panel in panels:
            for seed in cube["planning_seed_ids"]:
                for day in panel["planning_dates"]:
                    candidate_errors = {indexed[panel["id"], r, seed, day]["candidate_squared_error"] for r in rival_ids}
                    need(len(candidate_errors) == 1, "candidate trajectory differs across matched rivals")
            for rival in rivals:
                key = panel["id"] + "|" + rival["id"]
                by_seed[key] = []
                for seed in cube["planning_seed_ids"]:
                    paired = [indexed[panel["id"], rival["id"], seed, day] for day in panel["planning_dates"]]
                    by_seed[key].append(math.sqrt(sum(a["rival_squared_error"] for a in paired)/len(paired))
                                        - math.sqrt(sum(a["candidate_squared_error"] for a in paired)/len(paired)))
                if rival["kind"] == "deterministic":
                    for day in panel["planning_dates"]:
                        need(len({indexed[panel["id"], rival["id"], seed, day]["rival_squared_error"]
                                  for seed in cube["planning_seed_ids"]}) == 1,
                             "deterministic control has fake independent seed trajectories")
        for key in ("seed_covariance", "calendar_LRV", "seed_calendar_interaction_LRV"):
            positive_semidefinite(cube[key], len(expected), cube["numerical_relative_tolerance"])
        for i, left in enumerate(expected):
            for j, right in enumerate(expected):
                a, b = by_seed[left], by_seed[right]
                ma, mb = sum(a)/len(a), sum(b)/len(b)
                estimate = sum((u-ma)*(v-mb) for u, v in zip(a,b))/(len(a)-1)
                need(math.isclose(cube["seed_covariance"][i][j], estimate, rel_tol=1e-8, abs_tol=1e-20),
                     "seed covariance differs from paired mean-individual-RMSE source cube")
        need(cube.get("temporal_LRV_derivation") and cube.get("cross_rival_and_cross_panel_dependence")
             and cube.get("finite_T_estimand_bias_assessment"), "unidentified covariance components/finite-T scope")
        sampling = design["sampling"]
        seeds(sampling["final_seed_ids"], 10)
        need(integer(sampling.get("final_R"), 10) and sampling["final_R"] == len(sampling["final_seed_ids"]),
             "R must match exact independent final seeds")
        need(set(sampling["planning_seed_ids"]).isdisjoint(sampling["final_seed_ids"]),
             "pilot and final initializations must be separate")
        need(isinstance(sampling.get("final_T"), dict)
             and sampling["final_T"] == {p["id"]: len(p["confirmation_dates"]) for p in panels}
             and all(integer(v) for v in sampling["final_T"].values()), "T must match admitted calendar per panel")

    power_summary = {}

    def check_power():
        x = receipt("power")
        need(x.get("stage") == "prospective_planning_simulation"
             and x.get("uses_observed_confirmation_results") is False
             and x.get("decision_rule_sha256") == family["decision_rule"]["sha256"]
             and x.get("decision_rule_implementation_sha256") == bound(family["decision_rule"])["implementation"]["sha256"]
             and x.get("selected_R") == design["sampling"]["final_R"]
             and x.get("selected_T") == design["sampling"]["final_T"], "wrong power stage/rule/R/T")
        need(x.get("independent_replicates_review_uri") and x.get("DGP_scope_and_finite_calendar_bias_review_uri"),
             "independent simulation and finite-calendar target review required")
        payload = bound(x["replicate_decisions"])
        rows = payload["replicates"]
        need(isinstance(rows, list) and bool(rows)
             and all(integer(row.get("replicate"), 0) for row in rows)
             and len({row["replicate"] for row in rows}) == len(rows), "unique independent replicate IDs required")
        keys = [p + "|" + r for p in panel_ids for r in rival_ids]
        counts = {key: 0 for key in keys}
        joint = 0
        for row in rows:
            need(set(row["contrasts"]) == set(keys), "missing/extra power contrast")
            passed = []
            for key in keys:
                value = row["contrasts"][key]
                need(set(value["procedure_rejects"]) == set(method_ids)
                     and all(type(v) is bool for v in value["procedure_rejects"].values())
                     and type(value["remaining_complete_rule_pass"]) is bool,
                     "all three adjusted procedure decisions plus remaining rule required")
                success = all(value["procedure_rejects"].values()) and value["remaining_complete_rule_pass"]
                counts[key] += success
                passed.append(success)
            joint += all(passed)
        eta = x["MC_error_probability"]
        need(number(eta) and 0 < eta < 1, "MC error probability required")
        n, m = len(rows), len(keys)
        lower = {key: max(0., count / n - math.sqrt(math.log(m / eta) / (2 * n)))
                 for key, count in counts.items()}
        joint_point = joint / n
        radius = math.sqrt(math.log(2 / eta) / (2 * n))
        power_summary.update(independent_saved_replicates=n, per_contrast_successes=counts,
                             simultaneous_Hoeffding_MC_lower=lower, joint_power=joint_point,
                             joint_Hoeffding_MC_interval=[max(0., joint_point-radius), min(1., joint_point+radius)],
                             MC_interval_scope="conditional on externally reviewed DGP/replicate independence",
                             MC_error_probability_each_report=eta,
                             aggregate_contrast_and_joint_error_bound=min(1., 2*eta),
                             joint90_required=False)
        need(all(v >= .9 for v in lower.values()), "full-rule per-contrast power lower bound below .90")

    material = material_cost(design)
    for name, fn in (("candidate_own_R1_R2_and_continuation", check_candidate),
                     ("exact_family_three_methods_and_mapping", check_family),
                     ("four_admitted_disjoint_panels", check_panels),
                     ("estimand_margin_and_seed_sampling", check_target_and_sampling),
                     ("strong_calibration_for_exact_rule", check_calibration),
                     ("real_registration_before_outcome_access", check_freeze),
                     ("authorized_resources_and_persistent_runner", check_resources_and_persistence)):
        check(name, fn)
    if phase == "confirmation":
        check("complete_covariance_and_joint_R_T_design", check_covariance_and_design)
        check("complete_rule_power_and_joint_report", check_power)
    ready = all(checks.values())
    return {"schema_version": 1, "kind": "conditional_technical_preflight_not_R3_accreditation",
            "controller_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "phase": phase, "design_sha256": design_hash,
            "protocol_status": config.get("kind"), "checks": checks,
            "blockers": [name for name, passed in checks.items() if not passed], "errors": errors,
            "technical_preflight_ready": ready, "execution_permit_conditionally_ready": ready,
            "permit_condition": "External reviewers and authorizer must authenticate bound receipts; use only the reviewed runner within exact ceilings.",
            "scientific_accreditation": "NOT_ASSESSED_BY_SOFTWARE", "R3_accredited": False,
            "receipt_authenticity_verified_by_program": False,
            "power_certified_by_this_program": False,
            "material_cost": material, "saved_power_summary": power_summary,
            "execution_manifest": jobs if ready else [],
            "new_model_fits": 0, "new_model_predictions": 0, "new_MC_replicates": 0,
            "new_model_forward_passes": 0, "new_market_data_downloads": 0,
            "new_market_p_values": 0, "new_statistical_resamples": 0,
            "limitations": ["Hash consistency cannot establish authenticity, scientific adequacy, or actual independence.",
                            "Population RMSE and finite-calendar expected sample RMSE can differ; extra seeds do not remove finite-T bias.",
                            "No R/T is selected here; saved full-rule decisions are read only."]}


def checkpoint_state(job, receipt, root):
    """Only completed artifacts with matching immutable job identity may be skipped."""
    need(receipt.get("job_id") == job["job_id"] and receipt.get("design_sha256") == job["design_sha256"]
         and receipt.get("status") == "complete", "checkpoint job identity/status mismatch")
    module, _ = inherited(root["repo_root"])
    need(bool(receipt.get("artifacts")), "completed checkpoint has no artifacts")
    for binding in receipt["artifacts"]:
        path = module.safe_file(root["artifact_root"], binding["path"])
        need(module.digest(path) == binding["sha256"], "checkpoint artifact corrupt")
    return "SKIP_VERIFIED_COMPLETE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("planning", "confirmation"), default="confirmation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists() and not args.output.is_symlink(), "refuse existing output")
    result = assess(json.loads(args.contract.read_text()), args.repo_root, args.evidence_root, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"technical_preflight_ready": result["technical_preflight_ready"],
                      "blockers": result["blockers"], "output": str(args.output)}))
    return 0 if result["technical_preflight_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
