# 41. Ejecución optimizada: avance verificable de once orígenes

**Corte fijo: 11/11/2026, 13:31:53 de Bogotá (18:31:53 UTC).** Este capítulo describe registros confirmados en GitHub, no un panel conectado en vivo al kernel de Colab. La campaña continúa bajo el calendario original de once orígenes; la propuesta de tres no es operativa.

## Resultados nuevos confirmados

Se revisaron los **234 archivos nuevos de resultados** presentes en el [commit de campaña `df4415af338e`](https://github.com/jassan-bit/DeepLearning_202630/commit/df4415af338e2fab793c74b7732392c51c0fa6f6). Los 234 tienen `outcome.status=completed`; no hay fallos entre esos resultados. Se registran 238.032 pasos conocidos de optimizador.

| Modelo | Origen de selección | Ajustes HPO nuevos completados | Configuraciones cubiertas |
|---|---|---:|---:|
| TSMixer | Enero de 2024 | **36/36** | 18 |
| iTransformer | Enero de 2024 | **36/36** | 18 |
| MLP16, referencia E1 | Enero de 2024 | **36/36** | 18 |
| LSTM32, referencia E1 | Enero de 2024 | **36/36** | 18 |
| GRU32, referencia E1 | Enero de 2024 | **36/36** | 18 |
| FCN-1D | Febrero de 2024 | **36/36** | 18 |
| ResNet-1D | Febrero de 2024 | **18/36** | 9 |

Cada configuración utiliza las semillas **271201 y 271202**. El denominador 36 corresponde a 18 configuraciones × 2 semillas **para un modelo y un origen**. Cerrar esa cobertura no equivale a completar los once orígenes, NAS o el reajuste final del modelo.

El historial compatible anterior aporta **1.802 registros** fijados en el commit `0ecd8b851059abbbe370899d03093191c9815f7f`. Junto con los 234 nuevos, suman **2.036 registros de resultados**. Los estados de esos 1.802 anteriores no se reauditán aquí: el total combinado no se presenta como 2.036 éxitos, modelos diferentes ni modelos finales. La reutilización de resultados históricos tampoco se cuenta como entrenamiento nuevo.

## Última evidencia de actividad

El último evento del snapshot es [`job_started`, secuencia 469](https://github.com/jassan-bit/DeepLearning_202630/blob/df4415af338e2fab793c74b7732392c51c0fa6f6/campaigns/e2-cuda-crop-11-v2/events/000000469.json), publicado a las **13:31:53 de Bogotá**. Ese trabajo todavía no tiene resultado en el corte. El resultado anterior cerró a las 13:31:51.

Esto acredita el último inicio publicado, no la salud actual de la VM. En este snapshot no existe `status.json`: el ejecutor lo genera al cerrar o pausar la invocación. Su ausencia durante la secuencia de corridas no equivale por sí sola a un error.

## Entorno y corrección técnica

La admisión registra **Tesla T4, PyTorch 2.8.0+cu126, CUDA 12.6**, cálculo float32 determinista y AMP/TF32 desactivados. El perfil sintético pasó las comprobaciones de salidas y gradientes. Sólo los siete casos ancho/batch con mejora medida de al menos 5 % habilitan el cálculo optimizado de TimesNet; los demás conservan el cálculo original. Los factores del microperfil no son una aceleración de extremo a extremo de E2.

Un fallo previo rechazó `TorchVersion`, el tipo especial usado para la versión de PyTorch, al crear el diario estricto. La [corrección registrada antes de nuevas corridas](https://github.com/jassan-bit/DeepLearning_202630/blob/8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9/experiments/e2_implementation_optimization_11_origins_20261111/registration_json_metadata_fix.json) convierte el perfil a tipos JSON básicos, preservando sus valores, bytes canónicos y hash. No cambia arquitectura, búsqueda, semillas, épocas ni presupuesto. La regresión reproduce el fallo con un perfil real y verifica creación y lectura del diario; la suite v2 tiene 16 pruebas.

El [notebook vigente](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_optimizacion_11_origenes.ipynb) captura salida y errores del controlador y conserva un log local separado de la campaña pública. No debe iniciarse un segundo controlador mientras exista otro activo.

## Presupuesto y pendientes

Se conserva el techo acumulado registrado de **48 horas**, sin reinicio ni ampliación. Al admitir la campaña se transfirieron **16.246,97 segundos** (aproximadamente 4 h 31 min), descontando el consumo anterior.

Los nuevos resultados contabilizan **5.316,81 segundos**. La diferencia de **10.930,16 segundos** (aproximadamente 3 h 2 min) es un **límite superior provisional** del saldo todavía no contabilizado: excluye el intervalo abierto y el cierre. No es una promesa de tiempo restante ni una estimación para terminar E2. La reserva publicada para el último trabajo es de 7.500 segundos; el controlador puede pausar antes de saldo cero si no puede admitir la siguiente reserva completa.

Siguen pendientes la selección de los modelos y orígenes restantes, el NAS faltante, dimensionar R, comparadores y reajustes finales en soporte común, inferencia temporal, robustez, incertidumbre, interpretabilidad, ablaciones y reproducción integral. **No hay ranking final ni cumplimiento integral de la guía acreditado por este contador.** [Matriz de requisitos y evidencias](38_cumplimiento_guia.md).

## Fuentes y reproducción del corte

- Campaña: `campaigns/e2-cuda-crop-11-v2`.
- Commit fijo: `df4415af338e2fab793c74b7732392c51c0fa6f6`.
- Plan: `877a8ab4a0d34599f43c5dd16a90fe87fe03d37fe535466bf8e009c379cce742`.
- [Resumen estructurado de este corte](../_static/optimized_progress_20261111_183153/summary.json).
- [Índice de 234 resultados y cuatro eventos de referencia](../_static/optimized_progress_20261111_183153/source_index.json), con IDs, estados, identidades y referencias Git, sin copiar predicciones.
- [Calendario y límites del ejecutor](40_calendario_original.md); los cortes históricos de [§36](36_avance_campana_cuda.md) se conservan con sus fechas.

Los archivos descargables de este corte tienen una URL nueva. No sustituyen los recibos ni descargas de cortes anteriores.
