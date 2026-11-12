# 41. Ejecución optimizada: avance verificable de once orígenes

**Corte fijo vigente: 11/11/2026, 21:54:51 de Bogotá (12/11/2026, 02:54:51 UTC).** Se describen registros confirmados en GitHub; esta página no es un panel conectado al kernel de Colab. Se conservan los once orígenes, los catorce sistemas neuronales y todos los requisitos de E2. La propuesta de tres orígenes no es operativa.

## Resultados confirmados

El [commit de campaña 1cc1da2a43a4](https://github.com/jassan-bit/DeepLearning_202630/commit/1cc1da2a43a4ed056de18677875f99e3cb7a869a) contiene **1.067 resultados nuevos, todos completed y sin fallos entre esos registros**. Se conservaron las huellas de los 881 recibos del corte anterior y se comprobaron los 186 añadidos. Suman **1.265.888 pasos conocidos de optimizador**. Son resultados lógicos de ajustes HPO; la duración y los pasos del intento original interrumpido siguen desconocidos.

| Modelo o referencia | Enero 2024 | Febrero 2024 | Marzo 2024 | Abril 2024 | Mayo 2024 |
|---|---:|---:|---:|---:|---:|
| FCN-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | **23/36** |
| ResNet-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| Inception-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| TSMixer | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| iTransformer | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| MLP16, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| LSTM32, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| GRU32, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |

El denominador **36 = 18 configuraciones × 2 semillas**, para un modelo y un origen; las semillas son 271201 y 271202. La primera ola de ocho redes/referencias con cuerpo fijo tiene cobertura HPO cerrada **de enero a abril**. Faltan otros modelos, orígenes, NAS y la evaluación final.

† FCN/ResNet/Inception de enero se acreditan con el historial compatible y recibos de selección anteriores. Sus 108 intentos no se suman a los 1.067 nuevos. No se vuelven a auditar aquí los estados individuales de los 1.802 registros históricos.

El total combinado es **1.802 anteriores + 1.067 nuevos = 2.869 registros de resultados**. No representa modelos finales ni 2.869 éxitos auditados. La reutilización compatible tampoco se cuenta como entrenamiento nuevo.

La última corrida cerrada es **FCN-1D, mayo de 2024, configuración hpo-011, semilla 271201**: [evento 2145, job_finished](https://github.com/jassan-bit/DeepLearning_202630/blob/1cc1da2a43a4ed056de18677875f99e3cb7a869a/campaigns/e2-cuda-crop-11-v2/events/000002145.json). El corte termina con ese resultado; no acredita la salud actual de la VM. El status derivado todavía conserva 331 resultados y budget_extended_awaiting_resume: está desactualizado frente al diario y no demuestra una pausa actual.

## Recuperación observada y límite autorizado

La desconexión dejó **889 resultados guardados y un intento abierto**. La recuperación se vinculó al mismo trabajo lógico: **TSMixer, abril de 2024, hpo-012, semilla 271202**. Hubo un reintento físico registrado que terminó completed, con **174 pasos conocidos y 13,79 segundos de ajuste físico**. Se observó a las **20:36:15 de Bogotá** y se confirmó el cierre a las **20:36:22**. Ese resultado elevó el contador a 890; después se confirmaron **otras 177 corridas**, hasta 1.067.

El intento original permanece con desenlace, duración y pasos desconocidos; pudo haber realizado actualizaciones parciales. Los recibos separan ese intento de la ejecución de recuperación observada. El resultado recuperado se cuenta una sola vez como trabajo lógico. Los 889 recibos existentes y el plan se conservaron.

El usuario introdujo explícitamente **64 horas TOTALES**, registradas antes del ajuste recuperado. La secuencia presupuestal es **48 → 56 → 64 horas acumuladas**: la última enmienda añade ocho horas a las 56, conserva el gasto anterior y quedó en el [evento 1784](https://github.com/jassan-bit/DeepLearning_202630/blob/1cc1da2a43a4ed056de18677875f99e3cb7a869a/campaigns/e2-cuda-crop-11-v2/events/000001784.json). La autorización prospectiva y los recibos proceden del [commit fijo de recuperación 741e6d9477d3](https://github.com/jassan-bit/DeepLearning_202630/commit/741e6d9477d3eb276e1d489c96b3f2c8f2b16ac7).

- [Autorización con el nuevo total](../_static/optimized_progress_20261112_025451/recovery_authorization.json).
- [Inicio físico registrado](../_static/optimized_progress_20261112_025451/recovery_started.json).
- [Resultado físico observado](../_static/optimized_progress_20261112_025451/recovery_observed.json).
- [Recuperación confirmada](../_static/optimized_progress_20261112_025451/recovery_completed.json).

## Contabilidad al corte

| Concepto | Valor |
|---|---:|
| Techo acumulado autorizado | **64 h** |
| Consumo heredado de la campaña anterior | 156.553,03 s |
| Cargos de los 1.067 resultados nuevos | 29.662,94 s |
| Overhead del primer cierre | 2,17 s |
| Cargo para completar la asignación conservadora de la sesión perdida | 12.608,04 s |
| Overhead de la recuperación observada | 130,99 s |
| Consumo combinado contabilizado | **198.957,17 s, aproximadamente 55 h 16 min** |
| Saldo no contabilizado: límite superior | **31.442,83 s, aproximadamente 8 h 44 min** |

La sesión perdida se carga con su **asignación completa de ocho horas**: los cargos ya registrados en ella más el complemento de 12.608,04 s. La duración real de esa sesión sin cierre sigue desconocida. El ajuste físico de recuperación se añade mediante su resultado y su overhead observados. Los tres cierres suman 12.741,20 s y se incluyen en el resumen estructurado.

El saldo excluye el intervalo activo y el cierre futuro; no es una estimación del tiempo necesario para terminar E2. El controlador puede pausar con saldo positivo si no alcanza para reservar otro ajuste completo. La continuación conserva el plazo físico restante de la invocación de ocho horas. El supervisor publicado en [§40](40_calendario_original.md) mantiene su tope de 56 y se detiene ante este total de 64; esta actualización documental no cambia ese código ni evita cierres de Colab.

## Configuraciones, costes y trazabilidad

El [CSV de 1.067 ensayos](../_static/optimized_progress_20261112_025451/trials.csv) incluye modelo, origen, configuración, semilla, batch, LR, weight decay, dropout/warmup aplicables, estado, pasos y tiempos. Las **1.067 configuraciones coinciden exactamente con las huellas de sus resultados**, usando la [exportación de 252 configuraciones del protocolo activo](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json), regenerada desde el código y protocolo congelados de A3.

El CSV identifica la fila de recuperación y conserva unknown para el desenlace del intento original. Los segundos de esa fila corresponden al ajuste físico observado de recuperación. Los cargos conservadores y overhead de cierre se reportan por separado; los tiempos del CSV por sí solos no son latencia de inferencia, FLOPs, factura ni energía.

Se verificaron los **378 eventos nuevos** desde el corte anterior: 185 inicios y 186 cierres de ajuste, una ampliación, una interrupción, una pausa, dos cierres de sesión y dos aperturas. La cadena de hashes enlaza con el corte conservado. Los metadatos de ejecución mantienen **Tesla T4, PyTorch 2.8.0+cu126, CUDA 12.6 y FP32 determinista**, con AMP/TF32 desactivados. El perfil sintético TimesNet no acredita una aceleración global de E2.

## Pendientes de E2

Faltan selecciones de otros modelos/orígenes y NAS, dimensionamiento de R con piloto compatible, comparadores y reajustes finales, evaluación común, inferencia temporal, robustez, incertidumbre, explicabilidad, ablaciones y reproducción independiente. **No hay ranking final ni cumplimiento integral acreditado por estos conteos.** Las 608 fechas históricas conocidas mantienen alcance exploratorio. [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

## Fuentes del corte vigente

- Commit fijo de campaña: 1cc1da2a43a4ed056de18677875f99e3cb7a869a.
- Plan: 877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742.
- [Resumen estructurado](../_static/optimized_progress_20261112_025451/summary.json) · [Índice de 1.067 resultados y 16 eventos de referencia](../_static/optimized_progress_20261112_025451/source_index.json).
- [CSV de ensayos](../_static/optimized_progress_20261112_025451/trials.csv) · [Configuraciones activas regeneradas](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json).
- Fuente de recuperación: 741e6d9477d3eb276e1d489c96b3f2c8f2b16ac7; código registrado: 9d70bdad62af54316c9089d9c22e0781c3dd55a2.

(e2-cuda-cut-20261111-193218)=
## Corte anterior conservado: 19:32:18 de Bogotá

El commit 38211c3ca8289539ac8697a1b8d406cb0d7480b7 registraba **881 resultados nuevos y 2.683 combinados**. El corte vigente añade **186 resultados lógicos**, incluido el recuperado. Las huellas de aquellos 881 recibos permanecen intactas.

- [Resumen histórico de 881 ajustes](../_static/optimized_progress_20261112_003218/summary.json).
- [Índice histórico de 881 resultados](../_static/optimized_progress_20261112_003218/source_index.json).
- [CSV histórico de 881 ensayos](../_static/optimized_progress_20261112_003218/trials.csv).

(e2-cuda-cut-20261111-163112)=
## Corte anterior conservado: 16:31:12 de Bogotá

El corte del commit fe7af8ea2c03843a184d0c890a5d4d3734c2db20 registraba **546 resultados nuevos y 2.348 combinados**. El corte de las 19:32:18 añadió **335 resultados nuevos**, sin cambios en los blobs de esos 546 recibos. Los datos y enlaces del corte anterior se conservan.

- [Resumen histórico de 546 ajustes](../_static/optimized_progress_20261111_213112/summary.json).
- [Índice histórico de 546 resultados y ocho eventos](../_static/optimized_progress_20261111_213112/source_index.json).
- [CSV histórico de 546 ensayos](../_static/optimized_progress_20261111_213112/trials.csv).
- [Configuraciones activas regeneradas, fuente común de ambos CSV](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json).

(e2-cuda-cut-20261111-133153)=
## Corte anterior conservado: 13:31:53 de Bogotá

El corte del commit df4415af338e2fab793c74b7732392c51c0fa6f6 registraba **234 resultados nuevos y 2.036 combinados**. El corte de las 16:31:12 añadió 312 resultados a ese registro documental; cada corte conserva su fecha, fuente y alcance.

- [Resumen histórico de 234 ajustes](../_static/optimized_progress_20261111_183153/summary.json).
- [Índice histórico de 234 resultados y cuatro eventos](../_static/optimized_progress_20261111_183153/source_index.json).

Los recibos y las descargas anteriores permanecen con sus bytes originales; no se reescribe el historial.
