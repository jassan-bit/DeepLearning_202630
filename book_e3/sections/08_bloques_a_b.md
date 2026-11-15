# Bloques A y B con evidencias

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
| FCN-1D | `C02` | 0.00419039 | 0.00056359 |
| ResNet-1D | `C05` | 0.00419918 | 0.00046783 |
| Inception-1D | `C05` | 0.00424995 | 0.00056256 |
| LSTM-autoencoder con predictor | `C06` | 0.00552508 | 0.00121672 |
| TimesNet | `C05` | 0.00504385 | 0.00078177 |
| TSMixer | `C03` | 0.00559213 | 0.00141589 |
| TiDE | `C01` | 0.00515069 | 0.00158923 |
| TimeMixer | `C01` | 0.00410480 | 0.00070174 |
| PatchTST | `C02` | 0.00847770 | 0.00108272 |
| iTransformer | `C02` | 0.00585837 | 0.00095135 |
| MLP16 | `C05` | 0.00572968 | 0.00076181 |
| LSTM32 | `C01` | 0.00600944 | 0.00111936 |
| GRU32 | `C03` | 0.00556475 | 0.00109916 |
| Persistence | `fixed-lag1` | No disponible | No disponible |
| SeasonalNaive | `{"period":7}` | 0.01221810 | 0.00190607 |
| ARIMA | `{"order":[2,0,1]}` | 0.00431893 | 0.00074515 |

### Líderes y conjunto de finalistas

TimeMixer C01 lidera los quince sistemas con score de validación recuperado: RMSE agrupada 0,00414459. Persistence tiene receta fija y no publica score CV en su recibo; no se certifica un líder global de validación de los dieciséis.

El líder descriptivo de la A2 de prueba histórica es el LSTM-autoencoder y sí pertenece al MCS90 de E2. TimeMixer no pertenece a ese conjunto. Se reportan ambos hechos sin convertir al ganador de prueba en criterio para cambiar configuraciones o semillas. El MCS sigue condicionado a la cronología conocida y no constituye una evaluación independiente de E3.

(e3-exportacion-referencias)=
## Exportación real de tres referencias

MLP16, LSTM32 y GRU32 se exportaron a **TorchScript** desde sus checkpoints guardados. Se usó la semilla 271801 por ser el menor ID registrado, no por su RMSE. Se conservaron el escalador y el calibrador originales; no se reentrenó ni recalibró.

La ida y vuelta usa las mismas **53 fechas reales de calibración** y una tolerancia absoluta de **10⁻⁶ en unidades de σ₇**. La tabla muestra la RMSE del artefacto y su diferencia frente al modelo cargado en memoria. No sustituye la verificación sobre las 608 fechas de prueba A2, que permanece pendiente.

| Referencia | RMSE calibración exportada | ΔRMSE exportado − memoria | Error absoluto máximo | Mediana CPU (ms) | P95 CPU (ms) |
|---|---:|---:|---:|---:|---:|
| MLP16 | 0.00539178 | 0.00000000 | 0.00000000 | 0.008998 | 0.009874 |
| LSTM32 | 0.00619939 | 0.00000000 | 0.00000000 | 0.058608 | 0.130486 |
| GRU32 | 0.00454688 | 0.00000000 | 0.00000000 | 0.077021 | 0.101142 |

Las mediciones usan CPU, un hilo, lote de una muestra, diez calentamientos y cien llamadas. La latencia corresponde al forward TorchScript con entradas preprocesadas y excluye lectura y preprocesamiento. El pico RSS de cada proceso incluye Python, PyTorch, checkpoint y cohorte; no es memoria incremental del modelo ni una medición del dashboard en Render.

| Referencia | Artefacto (bytes) | Parámetros | Pico RSS del proceso (MiB) |
|---|---:|---:|---:|
| MLP16 | 14187 | 593 | 248.92 |
| LSTM32 | 30413 | 5025 | 257.51 |
| GRU32 | 25511 | 3777 | 251.75 |

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
