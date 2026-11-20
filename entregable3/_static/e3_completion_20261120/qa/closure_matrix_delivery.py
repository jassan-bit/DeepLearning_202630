"""Evidence-based18-item delivery matrix; no scientific go or publishing claim."""
from pathlib import Path
import csv
import json

ROOT=Path('/workspace/e3_completion_delivery/qa')
initial=json.loads((ROOT/'closure_matrix_initial.json').read_text())
updates={
1:('documented_awaiting_final_integration_QA','qa/module_manifest_review.json; qa/closure_matrix_delivery.csv','Integrar enlaces/estado del libro; ejecutar build y verificar publicación real.','root'),
2:('object_separation_verified','qa/baseline_preservation.json; qa/zero_shot_output_review.json','Preservar la distinción: aditiva796, V3piloto30, diagnóstico posterior51trayectorias/20.400filas; sin herencia científica.','root editorial'),
3:('permitted_checkpoint_and_saved_prediction_work_executed','qa/V3_binary_provenance_review.json; qa/zero_shot_output_review.json','Análisis/diagnóstico terminados sin fits; conservar resultado adverso y contadores separados de paridad/benchmark.','root editorial'),
4:('retrospective_limits_and_new_decision_order_verified','qa/zero_shot_remote_registration_metadata.json; qa/zero_shot_output_review.json; qa/protocol_power_review.json','Las fechas antiguas siguen expuestas. Sólo el nuevo diagnóstico se registró antes de sus nuevos forwards; no es prerregistro confirmatorio.','sin aprobación ficticia'),
5:('nine_old_neural_exports_offline_verified_publication_approval_blocked','qa/nine_export_provenance_review.json; qa/module_manifest_review.json','9 ONNX y 5.949 filas de paridad, error máximo 3,1492e-7 sigma7; pesos, escaladores y q conservados. Reexportación CI no pasó SHA estricto; fallback local conserva bytes. Subida rechazada por revisión automática y detenida. No implica 14 modelos habilitados.','autor: aprobación específica del paquete local antes de publicación; integración de producto posterior si procede'),
6:('V3_experimental_profile_and_levelB_executed','reproduction/payload/artifacts/atom_crps_seed20261118/; reproduction/container_replay/receipt.json','ONNX V3 de menor semilla final,55fechas,error7.83e-9; sin calibrador ni promoción ni aceptación pública.','autor/evaluador decide producto, no automático'),
7:('common_forward_benchmark_and_final_metadata_bindings_verified','qa/common_cost_review.json; costs/measured/receipt.json; costs/measured/costs.csv','15 procesos ORT CPU seriales, 1 CPU lógico y 1 hilo, batch1, 10 calentamientos y 100 mediciones. Metadatos finales revalidados; intento previo conservado. Falta energía/FLOPs/HTTP y no se atribuye memoria incremental.','root editorial; recursos remotos pendientes'),
8:('public_readiness_observed_acceptance_not_executed','release/github_actions_public_report.json; release/public_probe_execution_receipt.json','Origen Render pasó503→200 con17sistemas pero sin source_commit. Cliente V2 completo no arrancó por identidad no disponible. ManualDeploy del corte compatible y repetir prueba.','autor Render: SHA40 y despliegue manual si procede'),
9:('new_diagnostic_decision_registered_confirmation_no_go_preserved','protocol_power/decision_zero_shot.json; qa/zero_shot_output_review.json','Commit6aca875 anterior a forwards; pregunta limitada ejecutada. No autoriza nueva campaña ni cambia regla2% fallida o lambda.','nueva justificación del autor sólo si se reconsidera confirmación'),
10:('concrete_DRAFT_prepared_not_confirmatory_protocol','protocol_power/protocol.DRAFT.json; qa/protocol_power_review.json','20sistemas/19rivales/57componentes; roles, márgenes, F2, adecuación y budget detallados. Datos no expuestos, deltas y motor siguen sin certificar.','autor y revisión metodológica para campos finales'),
11:('conditional_planning_executed_final_power_unresolved','protocol_power/results/planning_receipt.json; qa/protocol_power_review.json','486escenarios analíticos;3semillas<5,55fechas,3varianzas principales disponibles/16faltantes;R y calendario finales nulos. NoMC/potenciaobservada.','datos/varianzas suficientes y decisión antes de dimensionar'),
12:('diagnostic_registration_verified_final_protocol_tag_absent','qa/zero_shot_remote_registration_metadata.json; protocol_power/protocol.DRAFT.json','Registro diagnóstico6aca875 real, tag final confirmatorio null. Un eventual tagDRAFT no cumple prerregistro deR3.','resolver puertas científicas antes del tag final'),
13:('CPU_preparation_adapter_prepared_fail_closed','protocol_power/preflight.py; protocol_power/colab_cpu_preparation.py; protocol_power/verification.json','Sin ruta de entrenamiento/inferencia. Root debe fijar destino real y lock de commit/manifest en celda externa; preparación permanece BLOCKED.','root source lock; autor ejecución CPU si desea recibo Colab'),
14:('scientific_R3_not_accredited_adverse_results_preserved','qa/zero_shot_output_review.json; protocol_power/protocol.DRAFT.json','V3no supera mejorcontrol enD01/D02/D03; noPV/confirmación. Novedad acotada deV3 no prueba utilidad niR3.','evaluador, sin forzar resultado favorable'),
15:('reproduction_levels_separated_A_and_B_only','reproduction/container_replay/receipt.json; reproduction/payload/fifteen_replay/receipt.json','A previo:11CSV. B nuevo:15pesos/825preds y contenedor limpio1peso/55preds. C entrenamiento integral no ejecutado.','autor delimita alcance de reproducción'),
16:('automatic_container_B_executed_human_independent_full_replay_absent','reproduction/container_replay/receipt.json; reproduction/payload/MANIFEST.sha256.json','Contenedor nuevo sin red verificó122archivos. No persona ajena ni repetición integral de796fits; no firma ni extensión de excepción IA/R1.','persona externa si requisito docente sigue vigente'),
17:('full_chain_metadata_and_hydration_material_prepared_not_full_archive','release/index/metadata_validation_receipt.json; release/index/index_manifest.json; release/hydrate_evidence.py; release/rights_by_source.csv','Cadena1..193 verificada por metadatos;8582archivosaudit4.268GB+259V3/2.359MB indexados. Cero bytespayload hidratados en esa verificación; delta193sólo35archivos.','autor confirmar derechos/licencia y elegir suplemento o archivo integral'),
18:('public_origin_DOI_not_registered_at_current_check','release/github_actions_public_report.json; release/zenodo_fields.DRAFT.json; release/AUTHOR_ACTIONS.md','2026-11-20T14:34Z: ZenodoAPI/record y DOIresolver devuelven404;API dice identificador persistente no registrado. Reserva previa informada no se marca publicado. Campos y acciones concretas preparados.','autor ratifica derechos y publica depósito; verificarURLpública/archivos después'),
}
rows=[]
for item in initial['items']:
 n=item['task_item'];status,evidence,pending,owner=updates[n]
 rows.append({'task_item':n,'requirement':item['requirement'],'guide_reference':item['guide_reference'],
 'status_at_module_delivery':status,'executed_or_prepared_evidence':evidence,
 'remaining_or_scope':pending,'owner':owner,'forbidden_inference':item['forbidden_inference']})
result={'schema_version':1,'status':'delivery_evidence_matrix_pending_final_staging_build_and_publication_QA',
 'AI_assistance_declared':True,'base_commit':initial['base_commit'],
 'diagnostic_registration_commit':'6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe',
 'new_QA_model_fits':0,'new_QA_model_predictions':0,'items':rows,
 'note':'Technical delivery status is not a score, institutionalR1/R2/R3approval, service activation or DOIpublication. Integration/publication checks are separate final evidence.'}
(ROOT/'closure_matrix_delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'closure_matrix_delivery.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('18-item delivery matrix written; no book or evidence artifacts mutated.')
