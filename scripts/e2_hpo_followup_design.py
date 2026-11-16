#!/usr/bin/env python3
"""Summarize recorded E2 HPO; prepare an unexecuted follow-up and input forms.

Standard library only. No model imports, inference, fitting, resampling, web calls,
or changes to historical evidence. Freeze inputs before the charged producer run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone


VERSION = "e2-hpo-followup-design-v1"
PARAMETERS = ("learning_rate", "weight_decay", "batch_size", "dropout", "warmup_fraction")
STATIC = Path("book_e2/_static")
OLD = STATIC / "e2_corrections_20261115/hpo"
SCI = STATIC / "e2_scientific_closure_20261116"
INPUTS = {
    "configurations": OLD / "configurations_78.csv",
    "opportunities": OLD / "opportunities_234.csv",
    "parameter_coverage": OLD / "parameter_coverage_65.csv",
    "charges": SCI / "catalogue_charge_accounting/catalogue_charge_opportunities_234.csv",
    "charge_summary": SCI / "catalogue_charge_accounting/summary.json",
    "charge_by_model": SCI / "catalogue_charge_accounting/catalogue_charge_by_model_13.csv",
    "phase_summary": SCI / "hpo_cost_reconciliation/summary.json",
    "cpu_costs": SCI / "cpu_fixed_recipe_reproduction/result_v1/neural_model_costs_13.csv",
    "architecture_inputs": STATIC / "e2_flops_final_audit_20261114/frozen_architecture_inputs.json",
    "native_requirements": SCI / "preregistration/native_export_recovery_requirements.json",
    "scopus_manifest": STATIC / "literature_scopus/source_manifest.json",
    "wos_manifest": STATIC / "literature_wos/source_manifest.json",
    "received_groups": Path("references/literature_e2/prisma_received_v1/received_groups_9559.csv"),
    "retrieval_attempts": Path("references/literature_e2/prisma_received_v1/retrieval_attempts.csv"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def truth(value) -> bool:
    return str(value).lower() == "true"


def values(row: dict) -> tuple[float, ...]:
    return tuple(float(row[p] or 0) for p in PARAMETERS)


def numerical_rank(matrix: list[list[float]], tolerance=1e-10) -> int:
    """Rank of the normalized parameter design, not a fitted response model."""
    a = [row[:] for row in matrix]
    rank = 0
    for col in range(len(a[0])):
        pivot = max(range(rank, len(a)), key=lambda r: abs(a[r][col]), default=None)
        if pivot is None or abs(a[pivot][col]) <= tolerance:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for row in range(len(a)):
            if row != rank:
                scale = a[row][col]
                a[row] = [x - scale * y for x, y in zip(a[row], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def freeze(repo: Path, out: Path) -> None:
    target = out / "hpo/frozen_source_inputs.json"
    if target.exists():
        raise SystemExit("Freeze already exists; preserve it or use a new follow-up directory.")
    paths = {**INPUTS, "producer_source": Path(__file__).resolve().relative_to(repo),
             "analysis_plan": (out / "hpo/analysis_plan.json").relative_to(repo)}
    recorded = {k: {"path": str(p), "sha256": sha(repo / p), "bytes": (repo / p).stat().st_size}
                for k, p in paths.items()}
    write_json(target, {
        "version": VERSION, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "inputs": recorded, "prior_results_known": True,
        "scope": "Posterior descriptive reconciliation and unexecuted study design; not original preregistration",
    })
    print(json.dumps({"frozen": str(target), "sha256": sha(target)}))


def check_freeze(repo: Path, out: Path) -> dict:
    record = json.loads((out / "hpo/frozen_source_inputs.json").read_text())
    for name, item in record["inputs"].items():
        if sha(repo / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    return record


def summarize_hpo(repo: Path, out: Path, frozen: dict) -> dict:
    configurations = read_csv(repo / INPUTS["configurations"])
    opportunities = read_csv(repo / INPUTS["opportunities"])
    coverage = read_csv(repo / INPUTS["parameter_coverage"])
    models = list(dict.fromkeys(row["model"] for row in configurations))
    assert len(models) == 13 and len(configurations) == 78 and len(opportunities) == 234
    pair_rows, model_rows, marginals = [], [], []
    for model in models:
        rows = [r for r in configurations if r["model"] == model]
        assert len(rows) == 6
        one_factor_pairs = 0
        for a, b in itertools.combinations(rows, 2):
            changed = [p for p, av, bv in zip(PARAMETERS, values(a), values(b)) if av != bv]
            one_factor_pairs += len(changed) == 1
            pair_rows.append({"model": model, "configuration_a": a["configuration_id"],
                              "configuration_b": b["configuration_id"], "changed_parameter_count": len(changed),
                              "changed_parameters": json.dumps(changed), "one_factor_pair": len(changed) == 1})
        cols = []
        for j, p in enumerate(PARAMETERS):
            col = [values(r)[j] for r in rows]
            if p == "learning_rate":
                col = [math.log10(x) for x in col]
            lo, hi = min(col), max(col)
            cols.append([(x - lo) / (hi - lo) if hi > lo else 0 for x in col])
        matrix = [[1.0] + [c[i] for c in cols] for i in range(6)]
        rank = numerical_rank(matrix)
        eligible = [r for r in rows if truth(r["eligible_for_selection"])]
        model_rows.append({
            "model": model, "registered_configurations": 6, "eligible_complete_configurations": len(eligible),
            "one_parameter_only_configuration_pairs": one_factor_pairs,
            "normalized_design_rank_including_intercept": rank,
            "n_configurations_minus_design_rank": 6 - rank,
            "independent_hpo_seeds": 1,
            "response_model_fitted": False, "independent_importance_identified": False,
            "selection_variance_fully_quantified": False,
        })
        for p in PARAMETERS:
            levels = sorted(set(float(r[p] or 0) for r in rows))
            for level in levels:
                group = [r for r in eligible if float(r[p] or 0) == level]
                scores = [float(r["pooled_cv_RMSE"]) for r in group]
                marginals.append({
                    "model": model, "parameter": p, "value": level,
                    "registered_configuration_count": sum(float(r[p] or 0) == level for r in rows),
                    "eligible_configuration_count": len(group),
                    "pooled_cv_RMSE_mean_descriptive": statistics.mean(scores) if scores else None,
                    "pooled_cv_RMSE_min_descriptive": min(scores) if scores else None,
                    "pooled_cv_RMSE_max_descriptive": max(scores) if scores else None,
                    "interpretation": "Joint-tuple association; other parameters covary; no isolated importance or p-value",
                })
    write_csv(out / "hpo/configuration_pair_identifiability_195.csv", pair_rows)
    write_csv(out / "hpo/model_design_identifiability_13.csv", model_rows)
    write_csv(out / "hpo/marginal_associations_descriptive.csv", marginals)
    write_csv(out / "hpo/parameter_coverage_recorded_65.csv", coverage)
    charge_summary = json.loads((repo / INPUTS["charge_summary"]).read_text())
    phase_summary = json.loads((repo / INPUTS["phase_summary"]).read_text())
    write_csv(out / "hpo/catalogue_costs_by_model_13.csv", read_csv(repo / INPUTS["charge_by_model"]))
    summary = {
        "version": VERSION, "source_commit": frozen["source_commit"],
        "models": len(models), "configurations": len(configurations), "planned_fold_opportunities": len(opportunities),
        "completed_fold_outcomes": sum(r["status"] == "completed" for r in opportunities),
        "recorded_winners": sum(truth(r["recorded_winner"]) for r in configurations),
        "one_factor_only_comparison_pairs": sum(r["one_parameter_only_configuration_pairs"] for r in model_rows),
        "marginal_rows": len(marginals), "new_fits": 0, "model_forward_passes": 0,
        "response_regressions": 0, "statistical_tests": 0, "resampling_draws": 0,
        "isolated_hyperparameter_importance_established": False,
        "selection_variance_established": False, "NAS_required_for_current_defined_bodies": False,
        "observed_catalogue_cost_accounting": charge_summary,
        "phase_breakdown_recorded_jobs": phase_summary["jobs_with_phase_receipts_union"],
        "completed_jobs_without_phase_breakdown": phase_summary["completed_jobs_without_phase_durations"],
        "remaining": ["Minimum family-specific HPO breadth not fully explored",
                      "No identified independent importance from the six covarying tuples",
                      "One HPO seed does not measure optimization-induced selection variance",
                      "Classical resource timings and full historical selection cost remain incomparable/unknown"],
    }
    write_json(out / "hpo/descriptive_summary.json", summary)
    make_plot(out / "hpo/recorded_configuration_scores.svg", models, configurations)
    design_followup(repo, out, models, coverage, frozen)
    return summary


def make_plot(path: Path, models: list[str], configurations: list[dict]) -> None:
    w, h = 1140, 1230
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" fill="#172333">',
           '<text x="22" y="30" font-size="20">RMSE de validación: seis tuplas conjuntas registradas por red</text>',
           '<text x="22" y="53" font-size="13">Asociación descriptiva: todos los hiperparámetros pueden variar juntos. Sin IC ni importancia aislada.</text>']
    for i, model in enumerate(models):
        x, y = 25 + (i % 3) * 375, 87 + (i // 3) * 220
        rows = [r for r in configurations if r["model"] == model]
        scores = [float(r["pooled_cv_RMSE"]) for r in rows if truth(r["eligible_for_selection"])]
        hi = max(scores) * 1.08
        svg.append(f'<text x="{x}" y="{y}" font-size="14">{html.escape(model)}</text>')
        for j, r in enumerate(rows):
            bx = x + j * 52
            if truth(r["eligible_for_selection"]):
                value = float(r["pooled_cv_RMSE"])
                bh = 135 * value / hi
                color = "#116a8d" if truth(r["recorded_winner"]) else "#9db9c8"
                svg.append(f'<circle cx="{bx+17}" cy="{y+160-bh:.3f}" r="5" fill="{color}"/>')
                svg.append(f'<text x="{bx}" y="{y+147-bh:.3f}" font-size="10">{value:.4f}</text>')
            else:
                svg.append(f'<text x="{bx}" y="{y+145}" font-size="10">incompleta</text>')
            svg.append(f'<text x="{bx}" y="{y+178}" font-size="11">{r["configuration_id"]}</text>')
    svg.append('<text x="25" y="1210" font-size="12">Azul: ganador registrado; gris: otra configuración completa. Escala vertical propia por panel.</text></g></svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def design_followup(repo: Path, out: Path, models: list[str], coverage: list[dict], frozen: dict) -> None:
    domain, obligations = [], []
    for model in models:
        cov = {r["parameter"]: json.loads(r["observed_registered_values"]) for r in coverage if r["model"] == model}
        lo, hi = min(cov["learning_rate"]), max(cov["learning_rate"])
        lrs = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / 3) for i in range(4)]
        wds = [0.0, max(cov["weight_decay"])]
        assert wds[1] > 0
        for i, (li, wi, bi) in enumerate(itertools.product(range(4), range(2), range(2))):
            a, b = (li >> 1) & 1, li & 1
            drop = cov["dropout"][(a ^ wi) * (len(cov["dropout"]) - 1)]
            warm = cov["warmup_fraction"][(b ^ bi) * (len(cov["warmup_fraction"]) - 1)]
            domain.append({"model": model, "configuration_id": f"F{i+1:02d}",
                           "learning_rate": lrs[li], "weight_decay": wds[wi], "batch_size": [16,64][bi],
                           "dropout": drop, "warmup_fraction": warm,
                           "body": "same previously defined published body; no NAS",
                           "status": "registered_candidate_not_executed"})
        obligations.append({
            "model": model, "training_domain_unique_configurations": 16,
            "dropout_varied_if_existing_recipe_allows": len(cov["dropout"]) > 1,
            "warmup_varied_if_existing_recipe_allows": len(cov["warmup_fraction"]) > 1,
            "context_days_fixed_by_inherited_contract": 7,
            "depth_width_kernels_heads_normalization": "Not expanded by this producer; source-supported variants require explicit adjudication before any new fit",
            "full_11_2_minimum_closed_by_this_design": False,
        })
    assert len(domain) == 208
    write_csv(out / "hpo/followup_training_domain_208_not_executed.csv", domain)
    write_csv(out / "hpo/family_specific_obligations_13.csv", obligations)
    cpu = read_csv(repo / INPUTS["cpu_costs"])
    cpu78 = sum(float(r["charged_worker_wall_seconds_sum"]) for r in cpu)
    one_each = sum(float(r["charged_worker_wall_seconds_mean"]) for r in cpu)
    plan = {
        "version": "e2-proposed-training-factor-followup-v1", "status": "registered_design_no_jobs_started",
        "source_freeze_at_utc": frozen["frozen_at_utc"], "prior_E2_results_known": True,
        "independent_prospective_confirmation_restored": False,
        "purpose": "Improve training-factor coverage, importance identifiability and selection-variance accounting in a separate posterior study",
        "scope_limits": "This training-factor design alone does not cover all structural minimum dimensions of guide11.2",
        "inputs": "Same655 pretest training examples and3 internal chronological folds; never select from cal53 or test608",
        "published_bodies_and_NAS": "Bodies unchanged; chapter14 NAS not applicable; unsupported structural variations are not silently introduced",
        "training_domain": "4logLR x2WD x2batch=16 unique tuples per network; permitted dropout/warmup vary as balanced auxiliary factors",
        "auxiliary_factor_aliasing": "Dropout/warmup columns are orthogonal to base main factors but alias some interactions; estimate only declared main effects unless a richer design is registered",
        "HPO_seeds_provisional": [271211,271212,271213],
        "seed_count_is_not_power_adequacy": True,
        "planned_fold_jobs_before_family_domain_amendment": 13 * 16 * 3 * 3,
        "training_policy": {"max_epochs":200,"patience":20,"optimizer":"unchanged AdamW/clipping/scheduler recipe", "fixed_epoch_scheduler_horizon":200},
        "selection_rule": "Same pooled249-date RMSE; deterministic configuration-ID tie break; all failures retain assigned opportunities",
        "importance_analysis_before_results": "Per-model factorial main contrasts for logLR, WD, batch; declared main-effects auxiliary estimates only. Do not call six-tuple historical associations independent importance. No confirmatory significance claims from this posterior follow-up.",
        "selection_variance_before_results": "Repeat complete selection across prespecified HPO seeds; report winner frequencies and selected validation risk dispersion. Temporal uncertainty separately requires paired complete candidate-origin predictions and block resampling under a separately frozen executable analysis plan.",
        "classical_fairness": "Before main jobs, register same-machine comparable opportunity/time rule and justified domains for ARIMA/SeasonalNaive; Persistence is fixed. Record all workers and timings. Historical unequal timing is not overwritten.",
        "future_timing_pilot_proposal_not_authorized_or_executed": {
            "requested_group_cap_seconds":450,
            "representatives": [
                {"model":"MLP16","worker_cap_seconds":60},
                {"model":"iTransformer","worker_cap_seconds":90},
                {"model":"TimesNet","worker_cap_seconds":270}
            ],
            "controller_allowance_seconds":30,"sum_caps_seconds":450,
            "candidate_choice": "F07 in each predefined training domain; same tuple level positions, no choice from observed fitness",
            "fold": "First internal fold: train253, validation83, origin2024-01-01; other two folds remain in main design",
            "seed":271211,
            "training_policy": "Complete original HPO policy: max200epochs, patience20, scheduler horizon200, no fixed small epoch/batch substitute. Per-worker cap can censor completion and is recorded as such, without automatic retry.",
            "purpose": "Measure real complete selection-fit wall where admitted; keep timeout cost censored. No partial fit is treated as a complete HPO trial or evidence of coverage/selection variance.",
            "requires": "Parent assignment of non-overlapping leases within remaining global budget; pinned runtime/sources; no active training from this script",
            "after_pilot": "Estimate per-job cost by observed complete fits and workload strata; any timed-out representative only gives a lower bound. Freeze family-domain amendment and a conservative admitted cap sum before asking for concrete extra budget; do not extrapolate three representatives as exact costs for all13bodies.",
            "projection_formula": "For each model m, sum setup_m + epochs_m * (ceil(n_train_fold/batch)*train_step_m + validation_epoch_m + checkpoint_m) over16configs*3folds*3seeds; add SAE auxiliary phases and classical jobs. Pilot and per-model measurements must identify the coefficients. If unknown or censored, use declared synchronized job cap, never substitute0.",
        },
        "observed_cost_context_not_a_new_HPO_forecast": {
            "78_fixed_recipe_CPU_final_fits_worker_seconds":cpu78,
            "one_fit_each_network_sum_recorded_means_seconds":one_each,
            "remaining_global_worker_seconds_at_task_assignment":4646,
            "proportional_fixed_recipe_scenario_1872_jobs_seconds":one_each*16*3*3,
            "scenario_warning": "Arithmetical scenario ONLY: the stored fits use655examples and fixed selected epochs, unlike HPO folds/200epoch early-stopping. Hardware/epochs/setup differ; not an estimate, quota request, lower bound or completion promise.",
            "historical_hpo_costs": "233observed mixed-hardware/mixed-clock worker charges;1860unknown allocation separate; not homogeneous CPU calibration",
        },
        "execution_gate": "No main HPO before source-supported family adjudications, pilot/time estimate, budget allocation and exact runtime/analysis freeze",
    }
    write_json(out / "hpo/followup_registration_not_executed.json", plan)


def summarize_literature(repo: Path, out: Path) -> dict:
    """Inventory bibliographic attachments locally; do not export their full contents."""
    known_scopus = json.loads((repo / INPUTS["scopus_manifest"]).read_text())
    known_wos = json.loads((repo / INPUTS["wos_manifest"]).read_text())
    known_hashes = {known_scopus["source_file"]["sha256"]: "original_Scopus7500_already_received"}
    known_hashes.update({r["sha256"]: "original_WoS_lot_already_received_no_native_range" for r in known_wos["sources"]})
    for name in ("/workspace/outputs/e2-scopus-search-20261114/full_export_audit.json",):
        p = Path(name)
        if p.exists():
            a = json.loads(p.read_text()); known_hashes[a["sha256"]] = "new_directed_Scopus1104_not_original7874"
    candidates = []
    root = Path("/workspace/attachments")
    if root.exists():
        for p in sorted(root.rglob("*")):
            if not p.is_file() or not (p.name.startswith("export_") or p.name.startswith("savedrecs")):
                continue
            digest = sha(p)
            count, eid_count, kind = None, None, "unclassified_native_candidate"
            if p.suffix.lower() == ".csv":
                with p.open(encoding="utf-8-sig",newline="") as f:
                    reader=csv.DictReader(f); count=0; eids=set()
                    for row in reader:
                        count+=1
                        if row.get("EID"):eids.add(row["EID"])
                    eid_count=len(eids)
                if count == 9:kind="nine_architecture_citation_export_not_original_missing374"
                elif count == 1104:kind="new_directed_Scopus1104_not_original7874"
                elif count == 10:kind="partial10_new_query_already_superseded_by1104"
            else:
                with p.open(encoding="utf-8-sig",errors="replace") as f:
                    count=sum(line.rstrip("\r\n")=="ER" for line in f)
                kind="WoS_native_lot_received; query_history_and_result_ranges_not_embedded"
            candidates.append({"filename":p.name,"private_local_path":str(p),"bytes":p.stat().st_size,
                               "sha256":digest,"record_count":count,"unique_EIDs_if_CSV":eid_count,
                               "classification":known_hashes.get(digest,kind),"native_original_coverage_proven":False})
    write_csv(out/"literature/local_native_candidate_inventory.csv",candidates)
    groups=read_csv(repo/INPUTS["received_groups"])
    absent=[r for r in groups if r["final_disposition"]=="report_not_retrieved"]
    assert len(absent)==174
    public_queue=[]
    for r in absent:
        title=r["title"]; lower=title.lower()
        score=2*any(w in lower for w in ("volatil","variance","dispersion"))+any(w in lower for w in ("bitcoin","crypto","ethereum"))
        public_queue.append({"group_id":r["group_id"],"title":title,"DOIs":r["DOIs"],"arxiv_ids":r["arxiv_ids"],
                             "title_only_priority_score":score,"priority_does_not_adjudicate_eligibility":True,
                             "retrieval_status":"not_retrieved_in_existing_snapshot; not_retried_by_this_producer"})
    public_queue.sort(key=lambda r:(-r["title_only_priority_score"],r["group_id"]))
    write_csv(out/"literature/public_recovery_priority_174.csv",public_queue)
    requirement=json.loads((repo/INPUTS["native_requirements"]).read_text())
    summary={"version":VERSION,"network_calls":0,"queries_executed":0,"new_reports_retrieved":0,
             "native_candidate_files_in_attachments":len(candidates),"native_original_missing374_identifiers_recovered":False,
             "new_complete_original7874_export_found":any(r["record_count"]==7874 for r in candidates),
             "original_query_history_and_ranges_recovered":False,"full_report_recovery_queue_count":len(absent),
             "requirements_preserved":requirement,
             "search_scope":"All export_*.csv and savedrecs* native candidates in/workspace/attachments; received repo manifests and known1104receipt. Existing duplicate worktrees are not new sources; no claim to have inspected every PDF or image.",
             "market_data_needed":False,"GitHub_access_needed":False,"credentials_requested":False}
    write_json(out/"literature/local_recovery_status.json",summary)
    template={"schema":"native-original-query-recovery-form-v1","do_not_fill_unknowns_by_guessing":True,
              "queries":[{"database":"Scopus","query_id":"original_result7874","exact_native_query":None,
                          "filters":None,"execution_datetime_with_timezone_if_known":None,"total_at_original_execution":7874,
                          "sort_order":None,"history_file_or_screenshot":None,
                          "export_batches":[{"filename":None,"first_result_inclusive":None,"last_result_inclusive":None,"sha256":None}],
                          "requested":"Full original7874export or missing374withEID/DOI and native batch ranges; original7500alreadyheld"},
                         {"database":"WoS","query_id":"original_Q1_and_Q2_separately","exact_native_query":None,
                          "filters":None,"execution_datetime_with_timezone_if_known":None,"total_at_original_execution":None,
                          "sort_order":None,"history_file_or_screenshot":None,
                          "export_batches":[{"filename":None,"query_assignment":None,"first_result_inclusive":None,"last_result_inclusive":None,"sha256":None}],
                          "requested":"Native history and totals/ranges mapping received302/500/500lots; repeat for any additional original query"}]}
    write_json(out/"literature/native_export_recovery_form.json",template)
    (out/"literature/README.md").write_text(LITERATURE_README,encoding="utf-8")
    return summary


LITERATURE_README="""# Recuperación bibliográfica pendiente

Este expediente vuelve a comprobar los archivos locales; no ejecuta búsquedas web ni recupera artículos. La cola de174informes es una priorización por título, no un dictamen de elegibilidad.

## Entrega mínima del historial original

1. Scopus: exportar el resultado original de7874registros, o los374restantes conEID/DOI. Ya se posee el CSV7500. Adjuntar el historial/captura con cadena exacta, campos consultados, filtros, fecha, total y orden de resultados; por lote, archivo y rango inicial/final. Si la consulta original ya no se puede recuperar, declarar ese hecho y conservar el límite; una ejecución actual es una actualización separada.
2. WoS: historial/capturas con Q1 y Q2 separados, bases/colecciones, cadena, filtros, fecha, total y orden. Vincular cada lote302/500/500 recibido con consulta y rango. No inferir los rangos del nombre savedrecs.
3. Completar `native_export_recovery_form.json` con datos acreditados. Los desconocidos permanecen null. Los CSV nuevos1104, el parcial10 y las nueve fuentes arquitectónicas corresponden a consultas/cortes distintos y no prueban por sí solos recepción de los374originales.

Exportación recomendada: CSV Scopus con EID, DOI, título, autores, año, fuente y resumen; WoS texto completo conUT/DOI y registro completo. Guardar originales de forma privada, sin subir resúmenes íntegros oPDF comerciales a una web pública. Puede aportarse ZIP de los lotes más historial/capturas; no hacen falta credenciales.

## Artículos no recuperados

`public_recovery_priority_174.csv` permite empezar por los títulos más próximos a volatilidad cripto. Intentar primero DOI/editor, repositorio de autor y versión abierta. Registrar URL, fecha, versión, SHA y resultado; verificar identidad y después elegibilidad, sin convertir inaccesibilidad en exclusión científica. Solicitar al usuario sólo los informes pertinentes que sigan inaccesibles y pueda compartir legítimamente. No es obligatorio subir174PDF antes de continuar. Un informe legítimamente no recuperado puede permanecer como tal enPRISMA con su límite explícito.

Ya están disponibles los20meses Binance y acceso de lectura al repositorio; no se piden otra vez datos de mercado, permisos GitHub ni contraseñas. Este paquete no certifica exhaustividad original.
"""


HPO_README="""# HPO: resumen posterior y diseño sin ejecutar

`descriptive_summary.json` concilia recibos existentes. Las asociaciones marginales y la figura muestran las seis tuplas registradas; no son importancia independiente, una selección nueva ni evidencia causal. `configuration_pair_identifiability_195.csv` cuenta qué comparaciones sólo cambian un parámetro; el rango de la matriz de diseño describe sus columnas, sin ajustar RMSE.

Los cargos observados233y la reserva1860s desconocida permanecen separados. Los180desgloses de fases están contenidos en cargos; no se suman otra vez. No se reconstruyen costes perdidos ni se equiparan relojes/hardware. El coste histórico completo y recursos clásicos comparables siguen pendientes.

El dominio de entrenamiento propuesto tiene16tuplas únicas por red, LRlog4×WD2×batch2. Dropout/warmup sólo varían donde las recetas existentes ya lo permiten, en columnas balanceadas con alias de interacciones declarado. No se cambian cuerpos ni D7; NAS§14no aplica. Esto no cubre por sí solo profundidad/ancho/cabezas/kernels/normalización: las alternativas publicadas requieren adjudicación previa y una enmienda del dominio antes de nuevos ajustes. Lo que siga fijo y no explorado conserva ese estado.

`followup_registration_not_executed.json` registra un estudio posterior con3semillas HPO provisionales,1872trabajosantes de completar dimensiones familiares. No acredita potencia ni cronología prospectiva del estudio original. Ningún trabajo se lanza desde este productor. El piloto de coste propuesto admite hasta450s: MLP16/60s, iTransformer/90s, TimesNet/270s y30sdelcontrolador; una tupla y el primer fold por representante. Usa max200épocas/patience20yconserva los timeouts, sin reemplazar el coste completo por pocos pasos. Sólo se ejecutaría con asignación explícita del controlador, no como presupuesto nuevo implícito. La proyección proporcional desde recetas finales CPU es sólo escenario aritmético, no estimación HPO ni solicitud de horas. El piloto medido y la fórmula por pasos/fases deben sustentar cualquier ampliación concreta; un representante censurado no prueba que su coste completo quepa.

## Reproducir el descriptivo

Desde la raíz del repositorio: `python scripts/e2_hpo_followup_design.py --out book_e2/_static/e2_closure_followup_20261116`. Antes de la primera ejecución se congela el plan con `--freeze-plan`; el productor rechaza cualquier fuente cambiada. Ejecutar mediante el ledger continuado y su reserva asignada, sin reiniciar cargos. No requiereTorch, red, datos de mercado, inferencia, entrenamiento ni bootstrap. Los recibos históricos permanecen intactos.
"""


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,default=STATIC/"e2_closure_followup_20261116")
    parser.add_argument("--freeze-plan",action="store_true")
    args=parser.parse_args();repo=args.repo.resolve();out=args.out
    out=(repo/out).resolve() if not out.is_absolute() else out.resolve()
    if repo not in out.parents:raise SystemExit("Output must remain inside repository")
    if args.freeze_plan:freeze(repo,out);return
    frozen=check_freeze(repo,out)
    hpo=summarize_hpo(repo,out,frozen)
    lit=summarize_literature(repo,out)
    (out/"hpo/README.md").write_text(HPO_README,encoding="utf-8")
    for sub in ("hpo","literature"):
        directory=out/sub
        files=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha(p)}
               for p in sorted(directory.rglob("*")) if p.is_file() and p.name!="manifest.json"]
        write_json(directory/"manifest.json",{"version":VERSION,"files":files,"producer_sha256":sha(Path(__file__)),
                   "frozen_input_manifest_sha256":sha(out/"hpo/frozen_source_inputs.json")})
    print(json.dumps({"version":VERSION,"status":"completed_descriptive_no_fits", "models":hpo["models"],
                      "new_fits":0,"network_calls":0,"native_missing374_recovered":False,
                      "artifacts":str(out)},ensure_ascii=False))


if __name__=="__main__":main()
