"""Build the E3 A/B evidence page and deployment table from measured reports.

This joins existing evidence. It never fits, selects from test, exports a model,
or reruns the nondeterministic CPU benchmark. --check verifies saved outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

DATA = Path('experiments/e3_bloques_ab_20261115')
STATIC = Path('book_e3/_static/e3_bloques_ab_20261115')
PAGE = Path('book_e3/sections/08_bloques_a_b.md')
REFERENCES = ('MLP16', 'LSTM32', 'GRU32')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value, digits=8):
    return 'No disponible' if value is None else f'{value:.{digits}f}'


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                         if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8')


def generate(root):
    root = Path(root)
    audit = read(root/DATA/'audit_summary.json')
    selection = read(root/DATA/'selection_report.json')
    selected = read(root/DATA/'master_configurations.json')
    if audit['block_A_closed'] or audit['results_changed'] or len(selected) != 16:
        raise ValueError('Reassess the page when Block A closes or benchmark results change')
    expected_audit = {'systems': 16, 'predictors': 81,
                      'downloaded_metadata_hashes_verified': 167,
                      'scaler_calibrator_predictor_bindings_verified': 81,
                      'frozen_file_hashes_reconciled': 245,
                      'release_weight_asset_metadata_verified': 79}
    expected_selection = {'model_count': 16, 'neural_recipes_verified': 13,
                          'all_neural_pooled_and_arithmetic_winners_agree': True,
                          'validation_scored_systems': 15,
                          'available_validation_leader': 'TimeMixer',
                          'global_validation_winner_16_systems': None,
                          'available_validation_leader_mcs_retained': False,
                          'historical_test_descriptive_leader': 'LSTM-autoencoder con predictor',
                          'historical_test_leader_mcs_retained': True}
    if (any(audit.get(k) != v for k, v in expected_audit.items())
            or any(selection.get(k) != v for k, v in expected_selection.items())):
        raise ValueError('Reassess the page: source counts or selection conclusions changed')
    reports = {}
    for model in REFERENCES:
        directory = root/'artifacts/e3_handoff'/model
        report = read(directory/'roundtrip_report.json')
        metadata = read(directory/'metadata.json')
        actual = digest(directory/'model.torchscript.pt')
        if (actual != report['export_sha256'] or actual != metadata['inference_artifact_sha256']
                or report['model'] != model or metadata['model'] != model):
            raise ValueError('Inference artifact identity or hash mismatch')
        rt = report['roundtrip']
        if (not rt['within_tolerance'] or rt['n'] != 53 or report['seed'] != 271801
                or report['cohort']['role'] != 'historical_calibration'
                or report['cohort']['A2_test_608_reproduced'] or report['retrained']):
            raise ValueError('Unexpected roundtrip scope')
        row = next(x for x in selected if x['model'] == model)
        frozen = next(x for x in row['frozen_predictors'] if x['seed'] == report['seed'])
        if frozen['checkpoint_sha256'] != report['checkpoint_sha256']:
            raise ValueError('Export does not bind to the selected frozen checkpoint')
        reports[model] = report
    deployment = []
    for row in selected:
        row = dict(row)
        report = reports.get(row['model'])
        profile = report['CPU_profile'] if report else {}
        rt = report['roundtrip'] if report else {}
        row.update({
            'export_status': 'verified_on_53_calibration_dates' if report else 'pending',
            'inference_artifact': f'artifacts/e3_handoff/{row["model"]}/model.torchscript.pt' if report else None,
            'cpu_latency_median_ms': profile.get('latency_median_ms'),
            'cpu_latency_p95_ms': profile.get('latency_p95_ms'),
            'cpu_process_peak_RSS_MiB': profile.get('process_peak_RSS_MiB'),
            'cpu_memory_scope': profile.get('memory_scope'),
            'cpu_latency_scope': profile.get('latency_scope'),
            'exported_calibration_RMSE': rt.get('RMSE_export_original'),
            'calibration_RMSE_difference': rt.get('RMSE_export_minus_memory'),
            'roundtrip_max_absolute_error': rt.get('maximum_absolute_prediction_error_original'),
            'roundtrip_tolerance': rt.get('absolute_tolerance_original_sigma7'),
            'exported_A2_test_RMSE': None,
            'selected_checkpoint_optimizer_state_present': report.get('selected_checkpoint_has_optimizer_state') if report else None,
        })
        deployment.append(row)
    result = {
        'version': 'e3-blocks-ab-evidence-v1', 'source_commit': audit['source_commit'],
        'systems': 16, 'predictors': 81, 'metadata_byte_hashes_verified': 167,
        'selection_recipes_verified': 13, 'exported_references': list(REFERENCES),
        'roundtrip_dates': 53, 'roundtrip_cohort': 'historical_calibration',
        'roundtrip_tolerance_original_sigma7': 1e-6,
        'exported_A2_test_cohort_verified': False,
        'new_training_performed': False, 'historical_A2_changed': False,
        'block_A_closed': False, 'block_B_closed': False,
        'E3_benchmark_freeze_issued': False,
        'source_reports_sha256': {str(DATA/name): digest(root/DATA/name) for name in
                                  ['audit_summary.json', 'selection_report.json', 'master_configurations.json']},
        'export_report_sha256': {model: digest(root/'artifacts/e3_handoff'/model/'roundtrip_report.json') for model in REFERENCES},
    }
    page = '''# Bloques A y B con evidencias

**Avance al 15 de noviembre de 2026.** Se ejecutaron la auditoría de artefactos, la revisión de configuraciones y tres exportaciones CPU del benchmark heredado. **Los bloques A y B permanecen abiertos:** los controles científicos pendientes de A y la exportación completa de B todavía no están cerrados. No se reentrenó ni se cambió la tabla A2 de E2.

(e3-auditoria-artefactos)=
## Bloque A y alcance de la auditoría

El inventario concilia **16 sistemas y 81 predictores**: 78 redes y tres comparadores deterministas. Se comprobaron **167 archivos de metadatos**, incluidos 81 escaladores y 81 calibradores, sus hashes y sus vínculos con cada checkpoint. Los 245 hashes del congelamiento histórico concuerdan con el manifiesto del respaldo.

Los 79 assets de pesos existen en el inventario de releases de GitHub. Esto comprueba sus localizadores, nombres y tamaños; no equivale a haber descargado y verificado sus 79 contenidos. La auditoría de metadatos comprobó los bytes de dos predictores JSON. La exportación posterior descargó y verificó además los tres checkpoints neuronales usados abajo. El ZIP de metadatos tiene 218.267 bytes y se incluye para repetir la auditoría sin red.

La cronología guardada contiene 655 objetivos de entrenamiento hasta el 31/10/2024, 53 de calibración del 08/11 al 30/12/2024 y 608 de evaluación del 01/01/2025 al 31/08/2026. Sus IDs no se solapan. Esta comprobación de IDs y metadatos no certifica el contenido de todos los tensores ni cómo se ajustó cada transformación.

### Controles que impiden cerrar A

- Comparación de duplicados y vecinos sobre los tensores finales efectivos.
- Auditoría completa de transformaciones y atributos de cada pipeline real.
- Validación adversarial sobre cohortes reales, con interpretación de su alcance.
- Control entrenado con etiquetas aleatorias en el pipeline vigente.
- Prueba formal comparable de la brecha train–validación con el margen δg heredado.

La [tabla de trazabilidad](06_trazabilidad_correcciones.md) conserva las columnas de la guía y distingue los hallazgos internos de observaciones docentes no recibidas. Si un control demuestra un error, se corrigen y recalculan todos sus productos dependientes y se publica el ranking antes/después. En esta revisión no se ha demostrado un fallo que cambie métricas: el impacto observado en A2 y ranking es **ninguno**.

### Determinismo y congelamiento

Se recuperó la política histórica de Python, PyTorch/CUDA, FP32 y configuración de determinismo de E2. El núcleo no fija una semilla global de NumPy; los análisis usan generadores registrados. Esta distinción se mantiene explícita, sin atribuir una llamada global inexistente ni garantizar igualdad universal entre plataformas.

La A2 histórica conserva SHA-256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. Es un ancla de lectura del benchmark heredado, **no un nuevo cierre del Bloque A ni un congelamiento definitivo de E3**. No se reajustan rivales para favorecer al modelo original.

## Bloque B y selección por validación

Se recuperaron trece recibos de selección originales y se verificaron sus seis configuraciones, tres folds temporales, parámetros y enlace al ajuste final. E2 eligió por RMSE agrupada en validación; la media aritmética de las tres RMSE elige la misma configuración en **13/13 redes**. No se usó la métrica de prueba ni la mejor semilla para elegir recetas.

La desviación que aparece abajo describe tres folds temporales correlacionados. No es dispersión de semillas independientes; por ello no se inventa un IC de validación. La tabla maestra descargable conserva por separado la media, DE e IC95 % de prueba entre seis semillas, sus límites condicionales a una cronología, parámetros, FLOPs core y costes existentes.

| Modelo | Configuración | RMSE media de folds | DE descriptiva de folds |
|---|---|---:|---:|
'''
    for row in selected:
        config = row['configuration_id'].replace('|', '/')
        page += f'| {row["model"]} | `{config}` | {numeric(row["validation_mean_fold_rmse"])} | {numeric(row["validation_fold_sd_descriptive"])} |\n'
    page += '''
### Líderes y conjunto de finalistas

TimeMixer C01 lidera los quince sistemas con score de validación recuperado: RMSE agrupada 0,00414459. Persistence tiene receta fija y no publica score CV en su recibo; no se certifica un líder global de validación de los dieciséis.

El líder descriptivo de la A2 de prueba histórica es el LSTM-autoencoder y sí pertenece al MCS90 de E2. TimeMixer no pertenece a ese conjunto. Se reportan ambos hechos sin convertir al ganador de prueba en criterio para cambiar configuraciones o semillas. El MCS sigue condicionado a la cronología conocida y no constituye una evaluación independiente de E3.

(e3-exportacion-referencias)=
## Exportación real de tres referencias

MLP16, LSTM32 y GRU32 se exportaron a **TorchScript** desde sus checkpoints guardados. Se usó la semilla 271801 por ser el menor ID registrado, no por su RMSE. Se conservaron el escalador y el calibrador originales; no se reentrenó ni recalibró.

La ida y vuelta usa las mismas **53 fechas reales de calibración** y una tolerancia absoluta de **10⁻⁶ en unidades de σ₇**. La tabla muestra la RMSE del artefacto y su diferencia frente al modelo cargado en memoria. No sustituye la verificación sobre las 608 fechas de prueba A2, que permanece pendiente.

| Referencia | RMSE calibración exportada | ΔRMSE exportado − memoria | Error absoluto máximo | Mediana CPU (ms) | P95 CPU (ms) |
|---|---:|---:|---:|---:|---:|
'''
    for model, report in reports.items():
        rt, cpu = report['roundtrip'], report['CPU_profile']
        page += f'| {model} | {numeric(rt["RMSE_export_original"])} | {numeric(rt["RMSE_export_minus_memory"])} | {numeric(rt["maximum_absolute_prediction_error_original"])} | {numeric(cpu["latency_median_ms"],6)} | {numeric(cpu["latency_p95_ms"],6)} |\n'
    page += '''
Las mediciones usan CPU, un hilo, lote de una muestra, diez calentamientos y cien llamadas. La latencia corresponde al forward TorchScript con entradas preprocesadas y excluye lectura y preprocesamiento. El pico RSS de cada proceso incluye Python, PyTorch, checkpoint y cohorte; no es memoria incremental del modelo ni una medición del dashboard en Render.

| Referencia | Artefacto (bytes) | Parámetros | Pico RSS del proceso (MiB) |
|---|---:|---:|---:|
'''
    for model, report in reports.items():
        cpu = report['CPU_profile']
        page += f'| {model} | {report["export_bytes"]} | {cpu["parameters"]} | {numeric(cpu["process_peak_RSS_MiB"],2)} |\n'
    page += '''
Cada carpeta `artifacts/e3_handoff/<modelo>/` contiene el artefacto, `metadata.json` y su reporte de ida y vuelta, con versiones, semilla, hiperparámetros, normalización, esquema de entrada, escala de salida y calibrador. La entrada es un tensor float32 ya estandarizado, de forma lote×7×5; la salida debe multiplicarse por la escala del objetivo. La aplicación deberá validar forma, finitud y dominio antes de invocar el artefacto.

### Pendientes de B

Faltan las exportaciones de los diez candidatos y los tres comparadores estadísticos, el control completo sobre prueba A2 y sus costes CPU comparables. Los tres checkpoints seleccionados inspeccionados no contienen estado del optimizador; no se presentan como checkpoints completos para reanudar entrenamiento. Debe resolverse esa conservación y la política de ajuste final común antes de cerrar el bloque.

No se aplicó cuantización ni se atribuyó un límite de memoria a modelos no medidos. Los modelos que no quepan en el despliegue deberán conservar resultados precalculados y una justificación sustentada; la pérdida por compresión, si se aplica, se evaluará frente a δ. El dashboard sólo cargará artefactos y ejecutará inferencia CPU.

## Evidencias descargables

- {download}`Resumen de auditoría <../_static/e3_bloques_ab_20261115/audit_summary.json>`.
- {download}`Inventario de 81 predictores <../_static/e3_bloques_ab_20261115/audit_predictors.csv>`.
- {download}`Controles ejecutados y pendientes <../_static/e3_bloques_ab_20261115/audit_checks.csv>`.
- {download}`Tabla maestra de selección, recursos y despliegue <../_static/e3_bloques_ab_20261115/deployment_master.csv>`.
- {download}`Informe de selección <../_static/e3_bloques_ab_20261115/selection_report.json>`.
- {download}`Resumen del avance A y B <../_static/e3_bloques_ab_20261115/handoff_summary.json>`.

Los artefactos, metadatos, scripts y pruebas están en el [repositorio](https://github.com/jassan-bit/DeepLearning_202630/tree/main/artifacts/e3_handoff). Las versiones de origen y hashes se conservan en `experiments/e3_bloques_ab_20261115/`.

## Repetición de las comprobaciones

Desde la raíz del repositorio, con Python 3.12, sin red ni entrenamiento:

```bash
python scripts/e3_audit_e2_handoff.py --check
python scripts/e3_select_e2_configurations.py --root . --check
python scripts/e3_build_handoff_book.py --check
python -m unittest discover -s tests -p 'test_e3_*.py' -v
```

La regeneración de exportaciones requiere además PyTorch CPU y los checkpoints originales verificados. Las comprobaciones anteriores validan evidencia guardada; no repiten las latencias ni entrenan controles pendientes. La asistencia de IA incluyó programación de estas herramientas y documentación; el autor deberá comprenderlas y defenderlas.
'''
    outputs = {
        DATA/'deployment_master.csv': csv_bytes(deployment),
        DATA/'deployment_master.json': (json.dumps(deployment,ensure_ascii=False,indent=2)+'\n').encode(),
        DATA/'handoff_summary.json': (json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode(),
        PAGE: page.encode('utf-8'),
    }
    for name in ['audit_summary.json','audit_checks.csv','audit_predictors.csv','selection_report.json']:
        outputs[STATIC/name] = (root/DATA/name).read_bytes()
    for name in ['deployment_master.csv','handoff_summary.json']:
        outputs[STATIC/name] = outputs[DATA/name]
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    outputs=generate(args.root)
    if args.check:
        changed=[str(path) for path,data in outputs.items()
                 if not (args.root/path).is_file() or (args.root/path).read_bytes()!=data]
        if changed:parser.exit(1,'Stale handoff outputs: '+', '.join(changed)+'\n')
        print(f'OK: {len(outputs)} evidence outputs reproduce exactly')
    else:
        for path,data in outputs.items():
            path=args.root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
