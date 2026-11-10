# 36. Avance parcial de la campaña CUDA

**Corte congelado del 10/11/2026, 18:00:19 de Bogotá (23:00:19 UTC): 1.497 resultados científicos completados, cero resultados fallidos, cinco modelos y el primer origen de selección.** La búsqueda HPO → NAS-1 → NAS-2 usa una Tesla T4 de Google Colab y el código científico A3. Este capítulo sustituye el corte de 906 resultados de las 09:56:42, conservado como [resumen histórico](../_static/cuda_campaign_progress_20261110/summary.json). Las cifras proceden del commit fijado y no son telemetría en vivo.

El corte contiene **180 resultados HPO, 1.117 NAS-1 y 200 NAS-2**. El LSTM-autoencoder con predictor registra sus 1.000 intentos NAS-1 y 200 NAS-2 del primer origen. TimesNet ya tiene 36 resultados HPO y 117 de los 1.000 intentos NAS-1 de ese origen. Los nueve sistemas restantes del plan, incluida la propuesta CNN-1D + LSTM + atención temporal, todavía no aportan resultados en este corte. No hay un ganador del benchmark final ni superioridad demostrada del candidato 11.

:::{note}
**Actividad reciente confirmada:** el último trabajo terminó a las **18:00:17 de Bogotá**; el siguiente quedó registrado como iniciado a las **18:00:19**. Esto acredita avance de la campaña GPU hasta el corte. La inspección de procesos del servidor CPU de Codex Cloud describe otro entorno y otra ejecución; no permite concluir que Colab esté detenido. La publicación del libro no reinicia ni interviene en la sesión de Colab.
:::

## Qué cuenta este avance

Un resultado corresponde a **un intento lógico de configuración y semilla**, dentro de un modelo, una etapa y un origen. No representa un modelo evaluado en todos los cortes. En el LSTM-autoencoder, cada intento incluye tres etapas de optimización; no se multiplica por tres el conteo de resultados.

| Evidencia del corte | Cantidad o alcance |
|---|---|
| Resultados lógicos confirmados | **1.497** |
| Estado de esos resultados | **1.497 `completed`; 0 `failed`** |
| HPO / NAS-1 / NAS-2 | **180 / 1.117 / 200** |
| Pasos de optimización conocidos | **2.160.176**; ningún resultado confirmado con pasos desconocidos |
| Modelos con resultados | **5 de los 14** del plan |
| Orígenes con resultados | Sólo `2024-01-01`, el primero de once |
| Eventos del diario | **3.015** |
| Último resultado terminado | **10/11/2026, 18:00:17 de Bogotá**; NAS-1 de TimesNet |
| Inicio posterior al último cierre | **18:00:19 de Bogotá**, evento `000003014` |
| Inicios sin resultado confirmado en este commit | **Uno**, iniciado al final del corte; por sí solo no es un fallo ni un atasco |

Los cero resultados con estado `failed` describen los resultados lógicos confirmados. No prueban que todos los intentos físicos anteriores hayan terminado bien: se conservan dos intentos interrumpidos de desenlace y coste desconocidos, descritos más adelante.

## Estado guardado y actividad de la sesión

`status.json` se escribe cuando el ejecutor cierra o pausa una sesión; **no es un heartbeat que se actualice tras cada entrenamiento**. En el commit de este capítulo conserva `paused_session_budget` y **1.280 trabajos**, correspondientes a un cierre anterior. El diario y los resultados posteriores confirman **1.497** y el siguiente inicio; por tanto, esa pausa guardada no identifica el estado de la sesión posterior al corte antiguo.

| Señal | Qué permite concluir |
|---|---|
| Colab muestra «En ejecución» y una T4 | El runtime indica una ejecución activa; el contador de la interfaz no es el número de modelos completados |
| Nuevos eventos `job_finished` y recibos en GitHub | Confirman trabajos científicos terminados hasta sus fechas |
| Último evento `job_started` sin resultado al corte | Hay un trabajo abierto en ese instante; no demuestra un fallo |
| `status.json` con menos resultados que el diario | Resume una salida anterior y debe contrastarse con los eventos posteriores |
| No aparecen procesos en Codex Cloud | Describe el servidor CPU inspeccionado, no la máquina de Colab |

El [resumen descargable](../_static/cuda_campaign_progress_20261110/summary_20261110_180019.json) distingue estas señales y conserva sus fechas. Este capítulo no afirma una utilización GPU medida en tiempo real ni garantiza continuidad del kernel después del corte.

## Cobertura de los catorce sistemas

El plan reúne **diez modelos base, un candidato propuesto y tres controles neuronales de E1**. Los cuatro comparadores restantes del benchmark de dieciocho sistemas mantienen su tratamiento separado y no forman parte de esta tabla de selección CUDA.

Todos los resultados de la tabla pertenecen al origen `2024-01-01`. En cada modelo, el HPO de un origen comprende 18 configuraciones y dos semillas: **36 intentos**. El guion indica que NAS no aplica a una estructura que el protocolo mantiene fija; su conteo NAS es cero.

| Modelo | Papel | HPO | NAS-1 | NAS-2 | Estado al corte |
|---|---|---:|---:|---:|---|
| FCN-1D | Base | 36 | — | — | HPO del primer origen completo; quedan diez orígenes |
| ResNet-1D | Base | 36 | — | — | HPO del primer origen completo; quedan diez orígenes |
| Inception-1D | Base | 36 | — | — | HPO del primer origen completo; quedan diez orígenes |
| LSTM-autoencoder con predictor | Base | 36 | 1.000 | 200 | Etapas del primer origen confirmadas; no equivale a completar los once orígenes |
| TimesNet | Base | 36 | 117 | 0 | HPO del primer origen completo; NAS-1 parcial, 117/1.000 |
| TSMixer | Base | 0 | — | — | Sin resultados confirmados todavía en esta campaña |
| TiDE | Base | 0 | 0 | 0 | Sin resultados confirmados todavía en esta campaña |
| TimeMixer | Base | 0 | 0 | 0 | Sin resultados confirmados todavía en esta campaña |
| PatchTST | Base | 0 | 0 | 0 | Sin resultados confirmados todavía en esta campaña |
| iTransformer | Base | 0 | — | — | Sin resultados confirmados todavía en esta campaña |
| CNN-1D + LSTM + atención temporal | Candidato 11 | 0 | 0 | 0 | Sin resultados confirmados; superioridad no evaluada |
| MLP16 | Control E1 | 0 | — | — | Sin resultados confirmados todavía en esta campaña |
| LSTM32 | Control E1 | 0 | — | — | Sin resultados confirmados todavía en esta campaña |
| GRU32 | Control E1 | 0 | — | — | Sin resultados confirmados todavía en esta campaña |
| **Total confirmado** | | **180** | **1.117** | **200** | **1.497 resultados del primer origen** |

Los 200 resultados NAS-2 del autoencoder corresponden a cuarenta finalistas y cinco semillas. El máximo de 205 por modelo y origen admite una referencia adicional cuando no esté ya entre los finalistas; no implica que aquí falten cinco resultados. Completar una selección de desarrollo de un origen no permite declarar un ganador entre los catorce sistemas o del benchmark final. Los ganadores y sus métricas sólo se publican al resolver sus dominios y semillas correspondientes.

## Dominio pendiente y presupuesto

Se conservan los dominios registrados: **5.544 oportunidades HPO**, **66.000 NAS-1** y **hasta 13.530 NAS-2**, para catorce sistemas y once orígenes. El máximo conjunto de **85.074 intentos** es un límite del espacio enumerado; no es un número de entrenamientos realizados ni una previsión temporal. NAS-2 tiene cardinalidad condicional, y el plan no incluye los reajustes ni comparadores finales: no se presenta una fracción simple de ese máximo como porcentaje de terminación del proyecto. Los diez orígenes restantes aún no aportan resultados al corte.

Los límites **iniciales** fueron **8 horas por sesión y 24 horas acumuladas**. El diario registra una ampliación global de **24 a 36 horas**, el **10/11/2026 a las 11:52:56 de Bogotá (16:52:56 UTC)**, con motivo `operator_requested_budget_extension`. La evidencia es el [evento `000002303`](https://github.com/jassan-bit/DeepLearning_202630/blob/3628e28dd9724939e0554c16c93917a784dcf5aa/campaigns/e2-cuda-selection-v1/events/000002303.json), SHA256 `899ab1bc8c440b0e90be19abc38e3f22785aaa87902b8b531752080ad444cf9a`. El límite de sesión permanece en **8 horas**. Documentar esta extensión ya registrada no modifica el presupuesto ni concede una ampliación adicional.

Estos límites pueden provocar pausas con candidatos, semillas u orígenes pendientes; no prometen completar el espacio. La carga conservadora de sesiones sin cierre confirmado es una regla contable: no se presenta como duración GPU observada ni consumo facturado. El presupuesto de `status.json` es el del cierre antiguo; tampoco se usa para inferir el consumo observado de la sesión abierta.

El trabajo confirmado usa CUDA, FP32 y política determinista, sin AMP ni TF32. La fuente y el desarrollo permanecen fijados; los valores finales de mercado de 2025–2026 no se abren para esta selección. `final_R` sigue sin fijarse. Antes de los reajustes y la evaluación final corresponde completar la selección, dimensionar las recetas y fijar R mediante el protocolo de potencia.

## Recuperaciones registradas y costes desconocidos

La campaña conserva dos reconciliaciones manuales de intentos que habían quedado iniciados sin resultado recuperable. Cada una autorizó un nuevo intento físico del mismo trabajo lógico, con configuración y semilla originales. Los nuevos resultados están confirmados; los intentos interrumpidos anteriores conservan **desenlace, duración y pasos desconocidos**.

| Reconciliación | Nuevo resultado confirmado, UTC | Controlador fijado | Confirmación en la campaña |
|---|---|---|---|
| R1 | 10/11/2026, 11:43:09 | [Fuente R1](https://github.com/jassan-bit/DeepLearning_202630/tree/b2d8cc9ca196f08670c1121efb89549c1838e38d) | [Commit del resultado](https://github.com/jassan-bit/DeepLearning_202630/commit/27caf0f482ce68c874d9add5a749a351ef39bd59) |
| R2 | 10/11/2026, 12:37:00 | [Fuente R2](https://github.com/jassan-bit/DeepLearning_202630/tree/20248e2917746962f92dccbcf89ae1874f6cbad2) | [Commit del resultado](https://github.com/jassan-bit/DeepLearning_202630/commit/72e989fa930bba4e02300622fc1c8d63fb41c4ee) |

Un límite inferior conocido de cero pasos no equivale a observar coste cero. El tiempo del nuevo intento tampoco reconstruye el anterior. Los recibos se conservan separados y no habilitan reintentos automáticos de otros trabajos.

Los **29 perfiles y 872 pasos de §§33–34**, los **123 casos sintéticos de §35** y los resultados de la ejecución CPU anterior tienen otros ámbitos. Ninguno se suma a los 1.497 resultados de esta campaña. Su estado tampoco describe la actividad del Colab actual.

## Evidencia reproducible del corte

El [resumen público JSON](../_static/cuda_campaign_progress_20261110/summary_20261110_180019.json) incluye conteos por modelo/etapa, pasos conocidos, último resultado, inicio abierto, extensión del presupuesto, diferencia con el estado guardado y comprobaciones de integridad. Su SHA256 es:

```text
606e91de0ebe0e62a6e79fd8a2978dd8ab71c03a3013af6756b3e4793786335f
```

| Identidad | Referencia fijada |
|---|---|
| Estado de campaña utilizado | [Commit `3628e28dd9724939e0554c16c93917a784dcf5aa`](https://github.com/jassan-bit/DeepLearning_202630/tree/3628e28dd9724939e0554c16c93917a784dcf5aa/campaigns/e2-cuda-selection-v1) |
| Último resultado confirmado | [Evento `000003013`](https://github.com/jassan-bit/DeepLearning_202630/blob/3628e28dd9724939e0554c16c93917a784dcf5aa/campaigns/e2-cuda-selection-v1/events/000003013.json), 18:00:17 de Bogotá |
| Inicio posterior | [Evento `000003014`](https://github.com/jassan-bit/DeepLearning_202630/blob/3628e28dd9724939e0554c16c93917a784dcf5aa/campaigns/e2-cuda-selection-v1/events/000003014.json), 18:00:19 de Bogotá |
| Código científico A3 | [Commit `564ba4762f84a800b02fa4c734f5752a169f5223`](https://github.com/jassan-bit/DeepLearning_202630/tree/564ba4762f84a800b02fa4c734f5752a169f5223) |
| Registro científico | [Archivo fijado en A3](https://github.com/jassan-bit/DeepLearning_202630/blob/564ba4762f84a800b02fa4c734f5752a169f5223/experiments/e2_cuda_search_20261109/registration.json), SHA256 `78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9` |
| Plan sellado | SHA256 `4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda` |

La lectura validó los esquemas de registros, inventario Merkle, enlaces del diario y de las dos recuperaciones, los **61 hashes científicos** y la conciliación de todos los conteos. La actualización del libro no ejecutó ajustes GPU, no abrió CSV ni escribió en la rama de campaña. El [corte previo de 906 resultados](../_static/cuda_campaign_progress_20261110/summary.json) se conserva identificado; no es el estado más reciente de esta página.
