"""Reconcile recorded E3 designs and existing job records, without model imports.

AI assistance declared. All inputs are frozen text/CSV records in sources.zip.
No market arrays/checkpoints are loaded and no fit or forecast is performed.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
MAIN = "29f9a26f1fd05484c6581ea6538ad7cb5a4781cb"
PILOT = "8eaa581bc34b615a8507c9e384a3e4be29084720"
AUDIT = "16a4504f6fb9208c7d6f89b65017194fa20470da"

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode()

def source(name):
    with zipfile.ZipFile(ROOT / "sources.zip") as z:
        return z.read(name)

def read_csv(name):
    return list(csv.DictReader(io.StringIO(source(name).decode())))

def read_json(name):
    return json.loads(source(name))

def save_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def designs():
    # One entry per documented functional neural variant/control/proposal.
    # Loss changes define recipes; seeds, learning rates and phases do not.
    values = [
      ("N01","full","","neural","market_fitted",1211,"penalty+saturation+content","MSE","v1_full","Preservar: peor que M00 en v1.","legacy_inventory.csv"),
      ("N02","without_penalty","N01","neural","synthetic_only",1210,"remove_penalty","MSE","not_market_fitted","Sólo intervención funcional artificial; no RMSE de mercado.","initial_design_log.md"),
      ("N03","without_saturation","N01","neural","synthetic_only",1211,"remove_saturation","MSE","not_market_fitted","Sólo intervención funcional artificial; no RMSE de mercado.","initial_design_log.md"),
      ("N04","without_content","N01","neural","synthetic_only",1199,"remove_query","MSE","not_market_fitted","Contenido convencional; no tercera innovación.","initial_design_log.md"),
      ("N05","M00","N01","neural","market_fitted",1210,"remove_penalty_and_saturation","MSE","v1_M00","Referencia conjunta; no aísla dos componentes.","v1_design_log.md"),
      ("N06","additive_positive_head","N05","neural","market_fitted",1210,"observable_floor_plus_positive_excess","normalized_MSE","v2_additive","Conservar como referencia; repeticiones no crean diseños.","legacy_inventory.csv"),
      ("N07","innovation_mean_only","N05","neural","market_fitted",1210,"half_t5_mean_quadrature128","normalized_MSE","v2_half_t_mean","Control sin NLL; conservar resultado.","legacy_inventory.csv"),
      ("N08","innovation_density_mean","N07","neural","market_fitted",1210,"same_half_t_mean","normalized_MSE+0.1_NLL_U","v2_half_t_joint","No supera heurística exploratoria.","legacy_inventory.csv"),
      ("N09","moment_joint","N05","neural","market_fitted",1236,"moment_envelope_three_heads","normalized_MSE+0.1_m_score+0.1_q_score","moments_joint","3,635 % peor que aditiva congelada; no promover.","legacy_inventory.csv"),
      ("N10","moment_mean_only","N09","neural","market_fitted",1236,"same_moment_mean_ancestors","normalized_MSE","moments_mean_only","No afirmar identificación de momentos sin sus scores.","legacy_inventory.csv"),
      ("N11","direct_moment_aux","N09","neural","market_fitted",1236,"direct_mean_same_auxiliary_heads","normalized_MSE+0.1_m_score+0.1_q_score","moments_direct","Control de vínculo entre media y momentos.","legacy_inventory.csv"),
      ("N12","atom_crps","N05","neural","market_fitted",1262,"learned_three_atom_law_exact_pushforward_mean","normalized_MSE+0.1_CRPS_U","atomic_full","No supera regla: RMSE 2,7043 % peor que N13.","v3_plan.json"),
      ("N13","atom_mse_only","N12","neural","market_fitted",1262,"same_atomic_law_remove_CRPS","normalized_MSE","atomic_no_CRPS","Mejor control en desarrollo; no promoción automática.","v3_plan.json"),
      ("N14","fixed_probs","N12","neural","market_fitted",1236,"uniform_fixed_masses","normalized_MSE+0.1_CRPS_U","atomic_fixed_masses","Capacidad menor declarada; no parámetros inactivos.","v3_plan.json"),
      ("N15","atom_plugin","N12","neural","market_fitted",1262,"second_moment_plugin_same_law","normalized_MSE+0.1_CRPS_U","atomic_plugin","Aísla agregación, no superioridad frente a aditiva.","v3_plan.json"),
      ("C01","floor_only","","analytical_control","evaluated",0,"sqrt_A","none","analytic","Sin ajuste neuronal.","v1_design_log.md"),
      ("C02","floor_plus_mean_excess","C01","analytical_control","evaluated",1,"sqrt_A+fit_only_mean_excess","fit_only_empirical_mean","analytic_fit_statistic","Un parámetro estadístico; no optimizador neuronal.","legacy_inventory.csv"),
      ("C03","frozen_M00_projected","N05","analytical_control","evaluated",1210,"max_saved_M00_floor","none_new","saved_prediction_transform","Reutiliza pesos; 1210 son heredados, cero parámetros nuevos.","v1_design_log.md"),
      ("C04","frozen_full_projected","N01","analytical_control","evaluated",1211,"max_saved_full_floor","none_new","saved_prediction_transform","Diagnóstico posterior omitido del subtotal antiguo de tres controles.","v1_design_log.md"),
      ("P01","faithful_inspired_detached_aux","N11","neural_proposal","proposed_not_implemented",1236,"separate_aux_gradients_from_mean_context","point+auxiliary_scores_proposed","not_market_fitted","Comparador conceptual: sin garantía Faithful ni ejecución acreditada.","moment_design_spec.json"),
      ("P02","moment_Jensen_plugin","N09","neural_proposal","proposed_not_implemented",1210,"sqrt_A_plus_c_m","not_frozen","not_market_fitted","No confundir con N15: una cabeza de momento no es la ley atómica.","moment_design_spec.json"),
      ("P03","fixed_q_9m2_learned_rho","N09","neural_proposal","proposed_not_implemented",1223,"fixed_q_to_m_squared_ratio","not_frozen","not_market_fitted","Capacidad propuesta, no instanciada o evaluada en mercado.","moment_design_spec.json"),
    ]
    columns = ["design_id","canonical_name","parent_id","category","evidence_state","parameters","mechanism_change","training_loss","implementation_context","decision","source_record"]
    return [dict(zip(columns, row)) for row in values]

def aliases():
    rows = [("full","N01","v1/full; M11","La completa inicial y su piloto son la misma estructura."),
            ("without_penalty","N02","M01","Penalización ausente, saturación presente."),
            ("without_saturation","N03","M10","Penalización presente, saturación ausente."),
            ("M00","N05","base_M00; frozen_M00","No unir con su salida proyectada C03."),
            ("additive_positive_head","N06","CNN-LSTM-atencion-aditiva E3; frozen_v2_additive","La cabeza aditiva es la misma; inicialización y selección se registran aparte."),
            ("floor_plus_mean_excess","C02","floor_plus_fit_mean; floor_plus_mean_excess","Misma regla con media sólo de ajuste; se recalcula por cohorte."),
            ("atom_plugin","N15","secondmoment_plugin","Nombre de preparación; versión publicada atom_plugin."),
            ("faithful_mean_only_reference","UNRESOLVED_REFERENCE","su propia media sola","Bosquejo sin especificación suficiente para decidir si equivale a N06 o es otra parametrización; no se fuerza un nuevo diseño.")]
    return [dict(zip(["canonical_name","design_id","aliases","deduplication_rule"], row)) for row in rows]

def jobs():
    rows=[]
    mapping={d["canonical_name"]:d["design_id"] for d in designs()}
    legacy=read_csv("legacy_inventory.csv")
    for r in legacy:
        if int(r["new_market_neural_fits"] or 0)!=1:
            continue
        rows.append({"job_id":"legacy:"+r["record_id"],"scope":"legacy_development_or_replicates","source_record":"legacy_inventory.csv","dataset":"D00","model":r["recipe"],"design_id":mapping[r["recipe"]],"seed":r["seed_or_seeds"],"stage":r["stage"],"job_type":"neural_fit","resource_phase_rows":1,"parameters":r["parameters"],"optimizer_steps":"","elapsed_seconds":"","unclosed_crash_compute_unknown":"unknown_not_reconstructed","status":"completed"})
    resources=read_csv("audit_resources.csv")
    grouped={}
    for r in resources:
        key=tuple(r[k] for k in ("dataset","model","seed","kind"))
        grouped.setdefault(key,[]).append(r)
    for (dataset,model,seed,kind), values in sorted(grouped.items()):
        typ="ARIMA_fit" if model=="ARIMA" else "no_optimizer_control" if kind=="deterministic" else "neural_fit"
        rows.append({"job_id":"audit:"+"/".join((dataset,model,seed,kind)),"scope":"extended_audit_796","source_record":"audit_resources.csv","dataset":dataset,"model":model,"design_id":"N06" if model=="CNN-LSTM-atencion-aditiva E3" else mapping.get(model,"BENCHMARK_BANK"),"seed":seed,"stage":kind,"job_type":typ,"resource_phase_rows":len(values),"parameters":max(int(v["parameters"] or 0) for v in values),"optimizer_steps":sum(int(v["optimizer_steps"] or 0) for v in values),"elapsed_seconds":sum(float(v["seconds"] or 0) for v in values),"unclosed_crash_compute_unknown":any(v["unclosed_crash_compute_unknown"].lower()=="true" for v in values),"status":"completed"})
    receipt=read_json("v3_pilot_receipt.json")
    plan=read_json("v3_plan.json")
    assert receipt["plan_sha256"]==hashlib.sha256(canonical(plan)).hexdigest()
    for path,r in sorted(receipt["fit_resource_receipts"].items()):
        _,stage,model,trial,_=path.split("/")
        seed=trial.split("_")[0].removeprefix("seed")
        rows.append({"job_id":"v3:"+path,"scope":"v3_controlled_development_30","source_record":"v3_pilot_receipt.json","dataset":"D00","model":model,"design_id":mapping[model],"seed":seed,"stage":stage,"job_type":"neural_fit","resource_phase_rows":1,"parameters":plan["expected_parameters"][model],"optimizer_steps":r["optimizer_steps"],"elapsed_seconds":r["elapsed_wall_including_setup_io_and_restart_seconds"],"unclosed_crash_compute_unknown":r["active_compute_lost_in_an_unclosed_crash_segment_is_unknown"],"status":r["status"]})
    return rows

def selections():
    rows=[]
    for r in read_csv("audit_hpo_metrics.csv"):
        rows.append({"selection_id":"audit:"+r["dataset"]+":"+r["model"]+":"+r["hpo_trial"],"scope":"extended_audit_796","dataset":r["dataset"],"model":r["model"],"learning_rate":r["learning_rate"],"hpo_seed":r["seed"],"selected":r["selected_by_validation"],"validation_RMSE":r["validation_RMSE"],"selection_data":"recorded_dataset_validation","initialization_policy":"native_constructor_no_v2_fit_bias_matching_for_E3","source_record":"audit_hpo_metrics.csv"})
    chosen=read_json("v3_closed_selection.json")["selected_learning_rates"]
    for i,r in enumerate(read_csv("v3_HPO_scores.csv")):
        rows.append({"selection_id":"v3:"+r["name"]+":"+r["learning_rate"],"scope":"v3_controlled_development_30","dataset":"D00","model":r["name"],"learning_rate":r["learning_rate"],"hpo_seed":r["seed"],"selected":float(r["learning_rate"])==chosen[r["name"]],"validation_RMSE":r["selection_RMSE"],"selection_data":"selection54_exposed_development","initialization_policy":"native_constructor_no_v2_fit_bias_matching_for_additive","source_record":"v3_HPO_scores.csv"})
    return rows

def main():
    manifest=read_json("source_manifest.json")
    for row in manifest["sources"]:
        assert hashlib.sha256(source(row["archive_name"])).hexdigest()==row["sha256"],row["archive_name"]
    d, a, j, s=designs(),aliases(),jobs(),selections()
    for row in d:
        row['parameter_scope']=('trainable_total' if row['category']=='neural' else 'proposed_not_instantiated' if row['category']=='neural_proposal' else 'inherited_checkpoint_zero_new' if row['design_id'] in ('C03','C04') else 'analytical_or_fit_mean_statistic')
        row['completed_neural_fits_in_named_sources']=sum(r['design_id']==row['design_id'] and r['job_type']=='neural_fit' for r in j)
    assert len(d)==22 and len({r['design_id'] for r in d})==22
    assert sum(r['category']=='neural' for r in d)==15
    assert sum(r['evidence_state']=='market_fitted' for r in d)==12
    assert len(j)==844 and len({r['job_id'] for r in j})==844
    assert sum(r['job_type']=='neural_fit' for r in j)==824
    assert sum(r['job_type']=='ARIMA_fit' for r in j)==4
    assert sum(r['job_type']=='no_optimizer_control' for r in j)==16
    assert len(s)==183 and len({r['selection_id'] for r in s})==183
    assert sum(str(r['selected']).lower()=='true' for r in s)==61
    resources=read_csv('audit_resources.csv')
    source_unknown=sum(r['unclosed_crash_compute_unknown'].lower()=='true' for r in resources)
    source_overrun=max(float(r['deadline_overrun_seconds'] or 0) for r in resources)
    result={"kind":"retrospective_source_bounded_E3_design_reconciliation","recorded_date_local":"2026-11-19","timezone":"America/Bogota","base_main_commit":MAIN,"pilot_results_commit":PILOT,"completed_audit_commit":AUDIT,"preregistration_claimed":False,"AI_assistance_declared":True,"Nd_model":{"implemented_neural_functional_variants":15,"market_fitted_distinct_neural_variants":12,"synthetic_only_neural_variants":3,"evaluated_analytical_output_controls":4,"specified_unimplemented_neural_proposals":3,"counted_documented_design_entries":22,"incompletely_specified_mean_reference_sketches":1,"documented_entry_interval_if_that_sketch_is_distinct":[22,23],"informal_or_unarchived_designs_exhaustively_observed":False,"global_all_thoughts_count_certified":False},"Nd_selection":{"observed_HPO_trial_jobs":183,"extended_audit_HPO":168,"V3_HPO":15,"E3_additive_audit_HPO":12,"V3_all_five_recipe_HPO":15,"learning_rate_values":[0.0003,0.001,0.003],"bias_initialization_contexts_not_merged":["legacy_v2_fit_mean_excess_bias_matching","audit_and_V3_native_constructor"],"validation_contexts_not_merged":["legacy_116_reused_development","audit_per_dataset_validation","V3_selection54_after_new_purge"],"single_all_selection_opportunities_count_certified":False,"reason":"HPO trial jobs, selection criteria, numerical precision choices and retrospective diagnostics have different units; unrecorded informal choices cannot be reconstructed."},"attempts":{"completed_job_records_in_declared_scopes":844,"legacy_completed_neural_fits":18,"extended_audit_unique_jobs":796,"extended_audit_resource_phase_rows":900,"extended_audit_neural_fits":776,"extended_audit_ARIMA_fits":4,"extended_audit_no_optimizer_control_jobs":16,"V3_completed_neural_fits":30,"all_declared_scopes_completed_neural_fits":824,"candidate_additive_unique_jobs_in_extended_audit":76,"audit_unclosed_compute_unknown_phase_rows":source_unknown,"audit_max_deadline_overrun_seconds_including_restart_downtime":source_overrun,"physical_invocation_or_restart_count_certified":False},"scientific_claims":{"new_model_fits":0,"new_market_predictions":0,"R1_accredited":False,"R3_accredited":False,"selection_bias_numerically_corrected":False,"three_seeds_or_many_jobs_prove_power":False},"remaining_uncertainties":["One Faithful mean-only reference sketch lacks a frozen full specification sufficient for deduplication.","Informal or unarchived proposals, failed physical invocations and all human/AI effort have no exhaustive immutable event log.","Three historical audit phase records retain unknown compute in unclosed crash segments; restart downtime must not be reported as GPU compute.","Counting designs is descriptive and does not recover independence, blind confirmation or a numerical winner-selection bias correction."]}
    (ROOT/'design_inventory.csv').write_bytes(csv_bytes(d))
    (ROOT/'design_aliases.csv').write_bytes(csv_bytes(a))
    (ROOT/'selection_opportunities.csv').write_bytes(csv_bytes(s))
    (ROOT/'attempt_ledger.csv.gz').write_bytes(gzip.compress(csv_bytes(j),mtime=0))
    protocol=read_json('audit_protocol.json')
    bank=[]
    for model in protocol['family']:
        if model in ('CNN-LSTM-atencion-aditiva E3','floor_only','floor_plus_mean_excess'):
            continue
        source_rows=[r for r in resources if r['model']==model]
        bank.append({'bank_id':'B%02d'%(len(bank)+1),'model':model,'category':'classical' if model in ('Persistence','SeasonalNaive','ARIMA') else 'neural_benchmark','parameters_max_in_saved_rows':max(int(r['parameters'] or 0) for r in source_rows),'candidate_Nd_model_entry':False,'scope':'existing_comparator_bank_not_new_E3_candidate_design'})
    assert len(bank)==16
    (ROOT/'benchmark_bank.csv').write_bytes(csv_bytes(bank))
    save_json('design_inventory.json',{'designs':d,'aliases':a})
    save_json('reconciliation.json',result)
    print(json.dumps({"design_entries":len(d),"implemented_neural":15,"market_neural_recipes":12,"HPO_trial_jobs":len(s),"job_records":len(j),"neural_fit_records":824,"new_market_fits":0}))

if __name__=='__main__':
    main()
