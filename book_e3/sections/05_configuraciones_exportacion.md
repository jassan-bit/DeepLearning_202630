# 5 Bloque B Selección de configuraciones y exportación

Este bloque reúne la selección documentada de los **16 sistemas** y la ingeniería de inferencia disponible. Las configuraciones se eligen desde validación. Las métricas de prueba histórica se muestran con su alcance, sin utilizarlas para cambiar la receta o escoger una semilla favorable.

## 5.1 Criterio de selección


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

## 5.2 Reentrenamiento final exportación y serialización

La guía exige una política común de ajuste final y conservar pesos, optimizador y configuración. El avance actual exporta pesos heredados de E2; no se presenta como un nuevo reentrenamiento final de E3 ni como un checkpoint completo con optimizador.

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

### Pendientes para cerrar B

Faltan las exportaciones de los diez candidatos y los tres comparadores estadísticos, el control completo sobre prueba A2 y sus costes CPU comparables. Los tres checkpoints seleccionados inspeccionados no contienen estado del optimizador; no se presentan como checkpoints completos para reanudar entrenamiento. Debe resolverse esa conservación y la política de ajuste final común antes de cerrar el bloque.

No se aplicó cuantización ni se atribuyó un límite de memoria a modelos no medidos. Los modelos que no quepan en el despliegue deberán conservar resultados precalculados y una justificación sustentada; la pérdida por compresión, si se aplica, se evaluará frente a δ. El dashboard sólo cargará artefactos y ejecutará inferencia CPU.



## Suplemento CPU y reproducibilidad

E2 añadió un diagnóstico eager CPU de los 78 predictores neuronales, con dos hilos y un plan propio. Es distinto de estas tres exportaciones TorchScript, medidas con un hilo. Sus latencias, memorias y diferencias respecto a CUDA no se mezclan como si fueran un único experimento. Ninguna de las dos verificaciones sustituye la métrica del artefacto exportado sobre las 608 entradas originales de prueba.

La tabla maestra conserva parámetros, FLOPs core, costes disponibles y campos ausentes, sin rellenarlos con ceros ni sustituir CPU por GPU. Los FLOPs completos, los costes comparables del resto y sus exportaciones siguen pendientes.

- {download}`Tabla maestra de selección recursos y despliegue <../_static/e3_bloques_ab_20261115/deployment_master.csv>`.
- {download}`Informe de selección <../_static/e3_bloques_ab_20261115/selection_report.json>`.
- [Artefactos y metadatos de las tres referencias](https://github.com/jassan-bit/DeepLearning_202630/tree/main/artifacts/e3_handoff).
- [Suplemento CPU actual de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html).

Los scripts `e3_audit_e2_handoff.py`, `e3_select_e2_configurations.py` y `e3_build_handoff_book.py` conservan la reproducción de sus informes y del corte original de A/B. La [página anterior de evidencias](08_bloques_a_b.md) permanece como anexo; no se alteran sus resultados para reorganizar el libro.
