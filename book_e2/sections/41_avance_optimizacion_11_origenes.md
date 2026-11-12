# 41. Ejecución optimizada: avance verificable de once orígenes

**Alcance posterior al corte:** desde el 12/11/2026 el [catálogo obligatorio es de 16 sistemas (§43)](43_catalogo_obligatorio.md), sin híbrido adicional ni Ridge. Los contadores de este capítulo corresponden a la campaña sellada anterior de catorce redes. Conservan sus fuentes y significado; no prueban que Colab haya aplicado la enmienda.

**Corte fijo vigente: 12/11/2026, 02:29:43 de Bogotá (07:29:43 UTC).** La campaña quedó **pausada por límite de sesión, con cierre limpio**. Este corte describe registros de GitHub y no comprueba el kernel de Colab en vivo. Se conservan los once orígenes, los catorce sistemas neuronales y todos los requisitos de E2. La propuesta de tres orígenes no es operativa.

## Resultados confirmados

El [commit de campaña dfd7d3c035c0](https://github.com/jassan-bit/DeepLearning_202630/commit/dfd7d3c035c015aac9cad454b1aec30f190a631d) contiene **1.484 resultados nuevos, todos completed y sin fallos entre esos registros**. Se conservaron las huellas de los 1.067 recibos anteriores y se comprobaron los **417 añadidos**. Suman **1.845.434 pasos conocidos de optimizador**. Son resultados lógicos de ajustes HPO; la duración y los pasos del intento original interrumpido siguen desconocidos.

| Modelo o referencia | Enero 2024 | Febrero 2024 | Marzo 2024 | Abril 2024 | Mayo 2024 | Junio 2024 |
|---|---:|---:|---:|---:|---:|---:|
| FCN-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** |
| ResNet-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** |
| Inception-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** |
| TSMixer | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** |
| iTransformer | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** | **8/36** |
| MLP16, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| LSTM32, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |
| GRU32, referencia E1 | **36/36** | **36/36** | **36/36** | **36/36** | **36/36** | Sin resultado nuevo |

El denominador **36 = 18 configuraciones × 2 semillas**, para un modelo y un origen; las semillas son 271201 y 271202. La primera ola de ocho redes/referencias tiene cobertura HPO cerrada **de enero a mayo**. El informe registra **44 selecciones cerradas de pares modelo/origen**, incluidas las compatibles del historial. Faltan otros modelos, orígenes, NAS y la evaluación final.

† FCN/ResNet/Inception de enero se acreditan con el historial compatible y recibos de selección anteriores. Sus 108 intentos no se suman a los 1.484 nuevos. No se vuelven a auditar aquí los estados individuales de los 1.802 registros históricos.

El total combinado es **1.802 anteriores + 1.484 nuevos = 3.286 registros de resultados**. No representa modelos finales ni 3.286 éxitos auditados. La reutilización compatible tampoco se cuenta como entrenamiento nuevo.

## Estado de la campaña y pausa limpia

La última corrida cerrada es **iTransformer, junio de 2024, configuración hpo-003, semilla 271202**, completada a las **02:29:41 de Bogotá**: [evento 2979](https://github.com/jassan-bit/DeepLearning_202630/blob/dfd7d3c035c015aac9cad454b1aec30f190a631d/campaigns/e2-cuda-crop-11-v2/events/000002979.json).

El [evento 2980](https://github.com/jassan-bit/DeepLearning_202630/blob/dfd7d3c035c015aac9cad454b1aec30f190a631d/campaigns/e2-cuda-crop-11-v2/events/000002980.json) registra la pausa session_budget y el [2981](https://github.com/jassan-bit/DeepLearning_202630/blob/dfd7d3c035c015aac9cad454b1aec30f190a631d/campaigns/e2-cuda-crop-11-v2/events/000002981.json) confirma el cierre limpio a las **02:29:43**. Quedaban **7.453,14 s de esa sesión** y el siguiente ajuste exigía una reserva completa de **7.500 s**. La sesión de continuación utilizó aproximadamente 5 h 53 min; la política de reserva explica la pausa antes de consumir todo el límite físico.

El [informe de estado al cierre](../_static/optimized_progress_20261112_072943/status_report.json) está actualizado: paused_session_budget, 1.484 resultados y cero fallos. Sustituye el alcance del informe anterior de 331 resultados. **No hay trabajos lógicos pendientes de conciliación ni asignaciones de sesiones sin cierre en este corte.** El intento físico original del incidente anterior conserva su desenlace desconocido en sus recibos.

La reanudación ordinaria puede usar el saldo actual, tras verificar la T4, la preparación y la ausencia de otro controlador; [procedimiento y límites en §40](40_calendario_original.md). Esta actualización documental no inicia entrenamiento ni acredita una reanudación posterior al corte.

## Recuperación observada y límite autorizado

La desconexión dejó **889 resultados guardados y un intento abierto**. La recuperación se vinculó al mismo trabajo lógico: **TSMixer, abril de 2024, hpo-012, semilla 271202**. Hubo un reintento físico registrado que terminó completed, con **174 pasos conocidos y 13,79 segundos de ajuste físico**. Se observó el **11/11/2026 a las 20:36:15 de Bogotá** y se confirmó el cierre a las **20:36:22**. Ese resultado elevó el contador a 890; después se confirmaron **otras 594 corridas**, hasta 1.484.

El intento original permanece con desenlace, duración y pasos desconocidos; pudo haber realizado actualizaciones parciales. Los recibos separan ese intento de la ejecución de recuperación observada. El resultado recuperado se cuenta una sola vez como trabajo lógico. Los 889 recibos existentes y el plan se conservaron.

El usuario introdujo explícitamente **64 horas TOTALES**, registradas antes del ajuste recuperado. La secuencia presupuestal es **48 → 56 → 64 horas acumuladas**: la última enmienda añade ocho horas a las 56, conserva el gasto anterior y quedó en el [evento 1784](https://github.com/jassan-bit/DeepLearning_202630/blob/1cc1da2a43a4ed056de18677875f99e3cb7a869a/campaigns/e2-cuda-crop-11-v2/events/000001784.json). La autorización prospectiva y los recibos proceden del [commit fijo de recuperación 741e6d9477d3](https://github.com/jassan-bit/DeepLearning_202630/commit/741e6d9477d3eb276e1d489c96b3f2c8f2b16ac7).

- [Autorización con el nuevo total](../_static/optimized_progress_20261112_025451/recovery_authorization.json).
- [Inicio físico registrado](../_static/optimized_progress_20261112_025451/recovery_started.json).
- [Resultado físico observado](../_static/optimized_progress_20261112_025451/recovery_observed.json).
- [Recuperación confirmada](../_static/optimized_progress_20261112_025451/recovery_completed.json).


## Contabilidad al cierre del corte

| Concepto | Valor |
|---|---:|
| Techo acumulado autorizado | **64 h** |
| Consumo heredado de la campaña anterior | 156.553,03 s |
| Cargos de los 1.484 resultados nuevos | 46.152,84 s |
| Overhead del primer cierre | 2,17 s |
| Complemento de la asignación conservadora de la sesión perdida | 12.608,04 s |
| Overhead de la recuperación observada | 130,99 s |
| Overhead del cierre limpio actual | 1,55 s |
| Consumo combinado contabilizado | **215.448,63 s, aproximadamente 59 h 51 min** |
| Saldo disponible al cierre | **14.951,37 s, aproximadamente 4 h 9 min** |

Los **cuatro cierres suman 12.742,76 s**. La sesión perdida conserva su asignación completa de ocho horas: los cargos ya registrados más el complemento de 12.608,04 s. El ajuste físico de recuperación se añade mediante su resultado y overhead observados. La duración real de la sesión perdida sigue desconocida.

El saldo es el del diario al cierre limpio, sin intervalo activo pendiente de contabilizar. La futura reanudación consumirá ese saldo con la política registrada. No es una estimación del tiempo necesario para terminar E2: la admisión sigue exigiendo que quepa la reserva completa de cada ajuste. **El límite sigue en 64 horas; este corte no registra una nueva ampliación.** El supervisor de [§40](40_calendario_original.md) conserva su tope de 56 y se detiene ante un total de 64.

## Configuraciones, costes y trazabilidad

El [CSV de 1.484 ensayos](../_static/optimized_progress_20261112_072943/trials.csv) incluye modelo, origen, configuración, semilla, batch, LR, weight decay, dropout/warmup aplicables, estado, pasos y tiempos. Los payloads de las **1.484 filas coinciden con sus configuraciones registradas**, usando la [exportación de 252 configuraciones del protocolo activo](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json), regenerada desde el código y protocolo congelados de A3.

El CSV identifica la fila recuperada y conserva unknown para el desenlace del intento original. Los segundos de esa fila corresponden a la recuperación física observada. Los cargos conservadores y overhead de cierre se reportan por separado; los tiempos del CSV por sí solos no son latencia de inferencia, FLOPs, factura ni energía.

Se verificaron los **836 eventos nuevos** desde el corte anterior: 417 inicios y 417 cierres de ajuste, una pausa y un cierre de sesión. La cadena de hashes enlaza con el corte conservado. Los metadatos mantienen **Tesla T4, PyTorch 2.8.0+cu126, CUDA 12.6 y FP32 determinista**, con AMP/TF32 desactivados. El perfil sintético TimesNet no acredita una aceleración global de E2.

## Pendientes de E2

Faltan selecciones de otros modelos/orígenes y NAS, dimensionamiento de R con piloto compatible, comparadores y reajustes finales, evaluación común, inferencia temporal, robustez, incertidumbre, explicabilidad, ablaciones y reproducción independiente. **No hay ranking final ni cumplimiento integral acreditado por estos conteos.** Las 608 fechas históricas conocidas mantienen alcance exploratorio. [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

## Fuentes del corte vigente

- Commit fijo de campaña: dfd7d3c035c015aac9cad454b1aec30f190a631d.
- Plan: 877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742.
- [Resumen estructurado](../_static/optimized_progress_20261112_072943/summary.json) · [Índice de 1.484 resultados y 20 eventos de referencia](../_static/optimized_progress_20261112_072943/source_index.json).
- [CSV de ensayos](../_static/optimized_progress_20261112_072943/trials.csv) · [Estado publicado al cierre](../_static/optimized_progress_20261112_072943/status_report.json).
- Fuente de recuperación: 741e6d9477d3eb276e1d489c96b3f2c8f2b16ac7; código registrado: 9d70bdad62af54316c9089d9c22e0781c3dd55a2.

(e2-cuda-cut-20261111-215451)=
## Corte anterior conservado: 11/11, 21:54:51 de Bogotá

El commit 1cc1da2a43a4ed056de18677875f99e3cb7a869a registraba **1.067 resultados nuevos y 2.869 combinados**. El corte vigente añade **417 resultados**, conservando las huellas de aquellos 1.067 recibos.

- [Resumen histórico de 1.067 ajustes](../_static/optimized_progress_20261112_025451/summary.json).
- [Índice histórico de 1.067 resultados](../_static/optimized_progress_20261112_025451/source_index.json).
- [CSV histórico de 1.067 ensayos](../_static/optimized_progress_20261112_025451/trials.csv).

(e2-cuda-cut-20261111-193218)=
## Corte anterior conservado: 19:32:18 de Bogotá

El commit 38211c3ca8289539ac8697a1b8d406cb0d7480b7 registraba **881 resultados nuevos y 2.683 combinados**. El corte de las 21:54:51 añadió **186 resultados lógicos**, incluido el recuperado. Las huellas de aquellos 881 recibos permanecen intactas.

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
