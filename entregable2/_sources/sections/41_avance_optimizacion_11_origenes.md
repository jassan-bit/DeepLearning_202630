# 41. Ejecución optimizada: avance verificable de once orígenes

**Corte fijo vigente: 11/11/2026, 19:32:18 de Bogotá (12/11/2026, 00:32:18 UTC).** Se describen registros confirmados en GitHub; esta página no es un panel conectado al kernel de Colab. Se conservan once orígenes y todos los requisitos. La propuesta de tres no es operativa.

## Resultados confirmados

Se revisaron los **881 archivos nuevos de resultados** del [commit de campaña 38211c3ca828](https://github.com/jassan-bit/DeepLearning_202630/commit/38211c3ca8289539ac8697a1b8d406cb0d7480b7). Los **881 tienen estado completed; no hay fallos entre esos resultados**. Registran 959.115 pasos conocidos de optimizador. Son ajustes HPO, no 881 modelos finales.

| Modelo o referencia | Enero 2024 | Febrero 2024 | Marzo 2024 | Abril 2024 |
|---|---:|---:|---:|---:|
| FCN-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** |
| ResNet-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** |
| Inception-1D | 36/36 histórico† | **36/36** | **36/36** | **36/36** |
| TSMixer | **36/36** | **36/36** | **36/36** | **17/36** |
| iTransformer | **36/36** | **36/36** | **36/36** | Sin resultado nuevo en este corte |
| MLP16, referencia E1 | **36/36** | **36/36** | **36/36** | Sin resultado nuevo en este corte |
| LSTM32, referencia E1 | **36/36** | **36/36** | **36/36** | Sin resultado nuevo en este corte |
| GRU32, referencia E1 | **36/36** | **36/36** | **36/36** | Sin resultado nuevo en este corte |

El denominador **36 = 18 configuraciones × 2 semillas**, para un modelo y un origen. Las semillas son 271201 y 271202. La primera ola de ocho redes/referencias fijas tiene cobertura HPO cerrada para enero, febrero y marzo; no equivale a cerrar todos los modelos, los once orígenes, NAS o el benchmark final.

† La selección de enero para FCN/ResNet/Inception se acredita con el historial compatible y los recibos de selección anteriores; sus intentos no se suman a los 881 nuevos. Esta actualización no vuelve a auditar los estados individuales de todos los 1.802 registros históricos.

El total entre historiales es **1.802 anteriores + 881 nuevos = 2.683 registros de resultados**. No se presenta como 2.683 éxitos auditados, modelos diferentes o modelos finales. La reutilización tampoco se contabiliza como entrenamiento nuevo.

## Última evidencia y presupuesto autorizado

La última corrida cerrada en el snapshot corresponde a **TSMixer, abril de 2024**, configuración hpo-008 y semilla 271201. El [evento final, job_started, secuencia 1767](https://github.com/jassan-bit/DeepLearning_202630/blob/38211c3ca8289539ac8697a1b8d406cb0d7480b7/campaigns/e2-cuda-crop-11-v2/events/000001767.json) inicia otro ajuste a las **19:32:17**; todavía no tiene resultado en el corte. Acredita ese inicio publicado, no la salud actual de la VM. El status derivado conserva el resumen anterior de 331 resultados y budget_extended_awaiting_resume; está desactualizado frente a estos eventos/resultados y no acredita una pausa actual.

La pausa anterior fue **por presupuesto**, tras 331 corridas nuevas, sin fallos. Quedaban aproximadamente 7.498,24 segundos y la siguiente reserva completa requería 7.500. Volver a ejecutar sin cambiar el saldo habría producido la misma pausa.

El usuario autorizó explícitamente pasar de **48 a 56 horas acumuladas**. Se registró antes de nuevos ajustes mediante [el evento de enmienda 665](https://github.com/jassan-bit/DeepLearning_202630/blob/38211c3ca8289539ac8697a1b8d406cb0d7480b7/campaigns/e2-cuda-crop-11-v2/events/000000665.json), a las **15:07:41 de Bogotá**, y quedó confirmado en el [commit de autorización](https://github.com/jassan-bit/DeepLearning_202630/commit/25fcf0282cc67465056883f03684184ac8b2bdb6). Se añaden ocho horas; no se reinician el plan, el historial ni el consumo.

| Contabilidad al corte | Valor |
|---|---:|
| Techo acumulado autorizado | **56 horas** |
| Consumo previo heredado | 156.553,03 s |
| Cargos de los 881 resultados nuevos | 24.613,16 s |
| Overhead de la sesión cerrada | 2,17 s |
| Consumo combinado conocido | 181.168,36 s, aproximadamente 50 h 19 min |
| Saldo todavía no contabilizado: límite superior | **20.431,64 s, aproximadamente 5 h 41 min** |

El saldo es provisional: excluye el intervalo activo y el cierre futuro; no es una predicción de terminación. El controlador puede pausar con saldo positivo si no alcanza para reservar el próximo ajuste completo. El supervisor opcional de [§40](40_calendario_original.md) sólo reanuda pausas limpias de sesión dentro del techo de 56; no añade horas ni evita cierres de Colab. Su activación en la sesión privada no se verifica con este corte.

## Tracking de configuraciones y recursos

Se añade un [CSV de los 881 ensayos](../_static/optimized_progress_20261112_003218/trials.csv) con modelo, origen, ID, semilla, batch, LR, weight decay, dropout/warmup aplicables, estado, pasos y tiempos. Cada configuración fue regenerada con **el código y protocolo congelados de la fuente A3** y su payload se cotejó contra la huella del resultado: **881 de 881 coincidencias exactas**.

La [exportación de las 252 configuraciones del protocolo activo](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json) identifica su fuente y serialización. El artefacto HPO anterior de [§27](27_nas_hpo_presupuesto.md) conserva una versión de diseño y no se usó como autoridad para este CSV. No se seleccionaron configuraciones nuevas ni se ejecutaron ajustes para generar el reporte.

Los segundos del CSV son cargos por intento y tiempo de callback; no equivalen por sí solos a latencia de inferencia, FLOPs, factura o huella energética. El entorno registrado sigue siendo **Tesla T4, PyTorch 2.8.0+cu126, CUDA 12.6 y FP32 determinista**, con AMP/TF32 desactivados. La optimización selectiva TimesNet deriva de un perfil sintético; no se declara aceleración global de E2.

## Pendientes de E2

Faltan selecciones de modelos/orígenes y NAS, dimensionamiento de R con piloto compatible, comparadores y reajustes finales, evaluación común, inferencia temporal, robustez, incertidumbre, explicabilidad, ablaciones y reproducción independiente. **No hay ranking final ni cumplimiento integral acreditado por estos conteos.** Las 608 fechas conocidas mantienen alcance exploratorio. [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

Se comprobaron también los **670 eventos añadidos desde el corte anterior**: 335 inicios y 335 cierres de ajuste, con enlaces de hash consecutivos. No hay una nueva enmienda de presupuesto ni cierre de sesión en ese tramo.

## Fuentes del corte vigente

- Commit fijo: 38211c3ca8289539ac8697a1b8d406cb0d7480b7.
- Plan: 877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742.
- [Resumen estructurado](../_static/optimized_progress_20261112_003218/summary.json) · [Índice de 881 resultados y ocho eventos de referencia](../_static/optimized_progress_20261112_003218/source_index.json).
- [Ensayos/configuraciones CSV](../_static/optimized_progress_20261112_003218/trials.csv) · [Configuraciones activas regeneradas](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json).

(e2-cuda-cut-20261111-163112)=
## Corte anterior conservado: 16:31:12 de Bogotá

El corte del commit fe7af8ea2c03843a184d0c890a5d4d3734c2db20 registraba **546 resultados nuevos y 2.348 combinados**. El corte vigente añade **335 resultados nuevos**, sin cambios en los blobs de esos 546 recibos. Los datos y enlaces del corte anterior se conservan.

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
