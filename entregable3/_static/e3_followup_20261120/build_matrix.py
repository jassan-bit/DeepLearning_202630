"""Generate the evidence map; no training, inference, network access or deployment."""
import csv
import json
from pathlib import Path

BASE = '7db8040619f602d7c52ae0c8bf54f7b6f5b5110e'

def main():
    here = Path(__file__).resolve().parent
    rows = []
    def row(key, requirement, version, evidence, check, status, missing, owner):
        rows.append(dict(id=key, requirement=requirement, model_version=version,
                         evidence=evidence, executed_check=check, status=status,
                         missing_concrete=missing, owner=owner))
    row('01','Punto de partida y cambios posteriores',BASE,'initial_audit.json',
        'GET conectado de branches/main; coincidencia exacta con el punto solicitado',
        'VERIFICADO','Sin cambios posteriores al iniciar el encargo','auditor IA')
    row('02','Preservación de trabajos y fuentes','aditiva master20261119T031104515520Z; V3 piloto20261120T031424176339Z',
        '../e3_science_followup_20261120/execution_receipt.json; preservation.json',
        'Hashes y recibos congelados; recálculo descriptivo desde CSV guardados',
        'VERIFICADO','No significa reproducción de los 796 entrenamientos ni de los 30 ajustes','auditor IA')
    row('03','Decisión científica y resultados adversos','V3 atom_crps original',
        '../e3_science_followup_20261120/decision.json',
        'Piloto +2.7043119888% de RMSE frente a atom_mse_only; mejor control superado en 0/3 dominios',
        'NO_GO_CONFIRMACION','No hay justificación suficiente para confirmar la receta actual; no se promueve la ablación','investigador IA; autor si reconsidera')
    row('04','Alcance de R1 y defensa R2','V3, sin herencia a aditiva',
        '../e3_science_followup_20261120/SCIENTIFIC_DECISION.md; ../e3_r1_substantive_20261120',
        'Dictamen técnico favorable acotado de R1 y declaración de IA conservados',
        'DOCUMENTADO_NO_APROBACION_INSTITUCIONAL','Defensa observada del autor para R2; aceptación docente de R1','autor y docente')
    row('05','Transferencia y confirmación','15 pesos BTC V3; D01/D02/D03 expuestos',
        '../e3_science_followup_20261120/execution_receipt.json',
        '51 trayectorias y 20400 filas cotejadas; calendarios anteriores al ajuste y solapamiento identificados',
        'RETROSPECTIVO_NO_CONFIRMATORIO','No puede recuperarse cegamiento ni disponibilidad histórica mediante documentación posterior','autor si plantea nueva investigación')
    row('06','Potencia y familia temporal F2','V3 vs19 rivales;57 componentes',
        '../e3_science_followup_20261120/requirements_matrix.csv; ../e3_completion_20261120/protocol_power',
        '486 escenarios analíticos revisados;3 semillas piloto<5;16 varianzas de rivales faltantes',
        'DESARROLLO_NO_POTENCIA_FINAL','Justificación científica, datos, métodos, márgenes, R y calendario finales; conjunta reportada y >=.90 por contraste','autor y revisión metodológica')
    row('07','Protocolo, registro y Colab','candidata V3 sin campaña nueva',
        '../e3_science_followup_20261120/REOPENING_CHECKLIST.md; ../e3_completion_20261120/protocol_power/preflight.py',
        'Preparación conserva puertas bloqueadas; ningún tag final ni entrenamiento nuevo',
        'BLOQUEADO_POR_NO_GO','No crear preregistro retrospectivo ni ejecutar campaña hasta resolver todas las puertas','autor sólo si cambia la decisión con evidencia')
    row('08','Exportaciones históricas y perfil experimental','14 ONNX históricos; V3 aparte',
        '../e3_completion_publication_20261120/E3_ONNX_PARITY_REPORT.json; ../e3_product_followup_20261120',
        'Nueve ONNX públicos ya aceptados;5949 replays remotos anteriores; integración nueva sin cambiar pesos o cuantiles',
        'EXPORTACIONES_PUBLICADAS','La exportación no demuestra utilidad predictiva; V3 carece de calibrador propio','ingeniería')
    row('09','Perfil de producto ampliado','perfil14 opt-in; servicio por defecto5',
        '../e3_product_followup_20261120',
        '14 API y navegador; cliente histórico5;5949 replays de adaptadores; contexto base más overlay explícito',
        'LOCAL_APROBADO_PUBLICO_PENDIENTE','CI de imagen exacta final aprobado antes de Manual Deploy; después aceptación pública del commit servido; no atribuir pruebas locales a Render','propietario Render')
    row('10','Aceptación pública y SHA efectivo','Render actual; SHA desconocido',
        'public_probe_execution.json; public_probe_20261120T1746Z.json',
        'Probe real de origen:HTTP200, salud sin SHA; el cliente existente conserva su guarda de identidad',
        'ACEPTACION_NO_INICIADA_IDENTIDAD_AUSENTE','Desplegar commit compatible, comprobar RENDER_GIT_COMMIT y ejecutar cliente con huellas/capturas','propietario Render')
    row('11','Costes y clásicos','perfil14 histórico; Persistence/SeasonalNaive/ARIMA',
        '../e3_product_followup_20261120/contract_review.json; ../e3_completion_20261120/costs/measured',
        'Historia y estimando examinados; coste del nuevo servicio en sus recibos; forwards previos separados',
        'ALCANCE_DE_MEDICION_EXPLICITO','Recursos/coste/latencia del Render real no se infieren del contenedor local; clásicos necesitan historia/estado','ingeniería y propietario Render')
    row('12','Reproducción A:tablas','11 CSV históricos',
        '../e3_reproduction_followup_20261120/existing_levels_audit.json',
        'Recibo público anterior descargado y verificado; no se repite la reconstrucción ya terminada',
        'RECIBO_A_VERIFICADO','No reproduce entrenamiento ni todas las tablas de cualquier campaña','tercero según alcance')
    row('13','Reproducción B:predicciones','ONNX V3 público aceptado;55 fechas',
        '../e3_reproduction_followup_20261120',
        'Descarga con hashes, contenedor nuevo sin red y comparación por campo; sin reexportar',
        'REPRODUCCION_B_AUTOMATICA_APROBADA','No es una persona independiente, ni confirmación, ni entrenamiento C','tercero si requisito humano sigue vigente')
    row('14','Reproducción C y acta independiente','796 trabajos/30 ajustes conservados',
        '../e3_reproduction_followup_20261120/REPRODUCTION_FORM_DRAFT.md',
        'Formulario sin firma y comandos preparados; ningún entrenamiento repetido',
        'NO_EJECUTADO','Tercero real y alcance acordado; la excepción IA de R1 no cubre automáticamente este requisito','autor, tercero y docente')
    row('15','Archivo y derechos','suplemento documental mínimo;193 deltas indexados',
        '../e3_archive_followup_20261120/README.md; ../e3_archive_followup_20261120/rights_by_component.csv',
        'Archivo candidato, manifiesto, cita, localizadores y verificador preparados; no se afirman4.27GB hidratados',
        'CANDIDATO_NO_DEPOSITADO','Ratificación de derechos, licencia propia y alcance del depósito; no inventar permisos de terceros','propietario')
    row('16','DOI público y bytes depositados','10.5281/zenodo.24356217',
        'public_probe_execution.json; ../e3_archive_followup_20261120/verify_zenodo_deposit.py',
        '2026-11-20T17:46Z:API,registro,resolvedor404; API indica identificador persistente no registrado',
        'RESERVA_PREVIA_PUBLICACION_NO_VERIFICADA','Publicar sólo tras derechos/licencia/autorización; luego verificar resolución y hashes reales del depósito','propietario Zenodo')
    row('17','Dictamen R3','V3 frente a familia pertinente',
        '../e3_science_followup_20261120/decision.json',
        'NO-GO científico y ausencia de diseño confirmatorio suficiente conservados',
        'NO_ACREDITADA','La finalización técnica o el cierre documental no prueban superioridad','docente')
    out={'schema_version':1,'base_commit':BASE,'AI_assistance_declared':True,
         'new_training_runs':0,'R3_accredited':False,'rows':rows}
    (here/'requirements_matrix.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (here/'requirements_matrix.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'new_training_runs':0,'R3_accredited':False}))

if __name__=='__main__':main()
