# 36. Avance parcial de la campaña CUDA

**Corte congelado del 10/11/2026, 09:56:42 de Bogotá (14:56:42 UTC): 906 resultados científicos confirmados, correspondientes a cuatro modelos y al primer origen de selección.** La búsqueda se ejecuta en una Tesla T4 de Google Colab con el código científico A3. Este capítulo incorpora evidencia posterior a la [preparación histórica de §35](35_busqueda_cuda_reanudable.md); sus cifras describen el commit indicado, no un contador en vivo.

El corte contiene **144 resultados HPO y 762 NAS-1; todavía no hay resultados NAS-2**. Los diez modelos restantes del plan, incluida la propuesta CNN-1D + LSTM + atención temporal, no tienen resultados en este corte. No se ha completado la búsqueda ni la comparación final del benchmark, y no se ha demostrado superioridad del candidato 11.

:::{warning}
**Recuperar la vista de la sesión activa no requiere iniciar otra búsqueda.** Si el navegador perdió la conexión, vuelve a conectar la sesión existente de Colab y conserva su caché. No pulses **Ejecutar todas**, no abras otra máquina para entrenar la misma campaña ni inicies un reintento por una pérdida de conexión de la interfaz. Una campaña que sigue confirmando resultados en GitHub puede continuar aunque el navegador muestre «Conectando». Los límites siguen siendo 8 horas por sesión y 24 horas acumuladas, sin ampliación autorizada.
:::

## Qué cuenta este avance

Un resultado corresponde a **un intento lógico de configuración y semilla**, dentro de un modelo, una etapa y un origen. No representa un modelo evaluado en todos los cortes. En el LSTM-autoencoder con predictor, cada intento incluye sus tres etapas de optimización; no se multiplica por tres el conteo de resultados.

| Evidencia del corte | Cantidad o alcance |
|---|---|
| Resultados lógicos confirmados | 906 |
| Estado de esos resultados | 906 `completed`; 0 `failed` |
| HPO / NAS-1 / NAS-2 | 144 / 762 / 0 |
| Modelos con resultados | 4 de los 14 del plan |
| Orígenes con resultados | Sólo `2024-01-01`, el primero de once |
| Eventos del diario | 1825 |
| Último resultado terminado | 10/11/2026, 14:56:42 UTC; NAS-1 del LSTM-autoencoder |
| Inicios sin resultado confirmado en este commit | Ninguno: el corte quedó entre trabajos |

La ausencia de un inicio pendiente en ese commit **no significa que terminó la campaña**: el siguiente trabajo puede comenzar después del corte. Tampoco los cero resultados con estado `failed` prueban que todos los intentos físicos anteriores hayan terminado bien; se conservan dos intentos de desenlace desconocido, descritos más adelante.

## Cobertura de los catorce sistemas

El plan reúne **diez modelos base, un candidato propuesto y tres controles neuronales de E1**. Los cuatro comparadores restantes del benchmark de dieciocho sistemas mantienen su tratamiento separado y no forman parte de esta tabla de selección CUDA.

Todos los resultados de la tabla pertenecen al origen `2024-01-01`. En cada modelo, el HPO de un origen comprende 18 configuraciones y dos semillas: **36 intentos**. La familia SAE lleva 762 de las 1000 arquitecturas de NAS-1 de ese origen. El guion indica que NAS no aplica a una arquitectura que el protocolo mantiene fija; su conteo de resultados NAS es cero.

| Modelo | Papel | HPO | NAS-1 | NAS-2 | Estado al corte |
|---|---|---:|---:|---:|---|
| FCN-1D | Base | 36 | — | — | HPO del primer origen completo; quedan los demás orígenes |
| ResNet-1D | Base | 36 | — | — | HPO del primer origen completo; quedan los demás orígenes |
| Inception-1D | Base | 36 | — | — | HPO del primer origen completo; quedan los demás orígenes |
| LSTM-autoencoder con predictor | Base | 36 | 762 | 0 | HPO del primer origen completo; NAS-1 parcial, 762/1000 |
| TimesNet | Base | 0 | 0 | 0 | Sin resultados confirmados todavía |
| TSMixer | Base | 0 | — | — | Sin resultados confirmados todavía |
| TiDE | Base | 0 | 0 | 0 | Sin resultados confirmados todavía |
| TimeMixer | Base | 0 | 0 | 0 | Sin resultados confirmados todavía |
| PatchTST | Base | 0 | 0 | 0 | Sin resultados confirmados todavía |
| iTransformer | Base | 0 | — | — | Sin resultados confirmados todavía |
| CNN-1D + LSTM + atención temporal | Candidato 11 | 0 | 0 | 0 | Sin resultados confirmados; superioridad no evaluada |
| MLP16 | Control E1 | 0 | — | — | Sin resultados confirmados todavía |
| LSTM32 | Control E1 | 0 | — | — | Sin resultados confirmados todavía |
| GRU32 | Control E1 | 0 | — | — | Sin resultados confirmados todavía |
| **Total confirmado** | | **144** | **762** | **0** | **906 resultados del primer origen** |

Completar el HPO de ese primer origen permite aplicar la regla interna de selección de su configuración. No acredita que FCN-1D, ResNet-1D o Inception-1D hayan completado los once orígenes, ni permite compararlos como ganadores del benchmark final. El candidato 11 sigue siendo una hipótesis por contrastar; sus cero resultados no se sustituyen por los del piloto histórico.

## Dominio pendiente y presupuesto

Se conservan los dominios registrados: **5544 oportunidades HPO**, **66.000 NAS-1** y **hasta 13.530 NAS-2**, para catorce sistemas y once orígenes. El máximo conjunto de **85.074 intentos** es un límite del espacio enumerado, no una cantidad de entrenamientos ya realizados ni una previsión temporal. Los diez orígenes restantes aún no aportan resultados al corte.

La ejecución continúa con los límites originales de **8 horas por sesión y 24 horas acumuladas**. No se registró una ampliación. Esos límites pueden provocar una pausa con candidatos, semillas u orígenes pendientes; no garantizan terminar el espacio. La asignación conservadora de una sesión sin cierre confirmado es una regla contable: no se presenta como tiempo de entrenamiento observado ni como consumo facturado por Colab.

El trabajo numérico confirmado usa CUDA, FP32 y política determinista, sin AMP ni TF32. La fuente y los datos de desarrollo permanecen fijados; los valores finales de mercado de 2025–2026 no se abren para esta selección. `final_R` sigue sin fijarse. Una pausa no habilita elegir entre candidatos o semillas cuya comparación registrada esté pendiente. Antes de los reajustes y la evaluación final todavía corresponde resolver la selección, dimensionar las recetas seleccionadas y fijar R mediante el protocolo de potencia.

## Recuperaciones registradas y costes desconocidos

La campaña conserva dos reconciliaciones manuales de intentos que habían quedado iniciados sin un resultado recuperable. Cada una autorizó un nuevo intento físico del mismo trabajo lógico, con su configuración y semilla originales. Los nuevos resultados están confirmados y enlazados al diario; los intentos interrumpidos anteriores conservan **desenlace, duración y pasos de optimización desconocidos**.

| Reconciliación | Nuevo resultado confirmado, UTC | Controlador fijado | Confirmación en la campaña |
|---|---|---|---|
| R1 | 10/11/2026, 11:43:09 | [Fuente R1](https://github.com/jassan-bit/DeepLearning_202630/tree/b2d8cc9ca196f08670c1121efb89549c1838e38d) | [Commit del resultado](https://github.com/jassan-bit/DeepLearning_202630/commit/27caf0f482ce68c874d9add5a749a351ef39bd59) |
| R2 | 10/11/2026, 12:37:00 | [Fuente R2](https://github.com/jassan-bit/DeepLearning_202630/tree/20248e2917746962f92dccbcf89ae1874f6cbad2) | [Commit del resultado](https://github.com/jassan-bit/DeepLearning_202630/commit/72e989fa930bba4e02300622fc1c8d63fb41c4ee) |

Un límite inferior conocido de cero pasos **no equivale a observar un coste cero**. Los tiempos del nuevo intento físico tampoco reconstruyen lo ocurrido en el anterior. La contabilidad y los recibos de ambas recuperaciones se mantienen separados; el éxito de la reconciliación no habilita reintentos automáticos de otros trabajos ni una extensión del presupuesto.

Los **29 perfiles de recursos y 872 pasos conocidos de §§33–34** pertenecen a los pilotos históricos, y los **123 casos de preparación de §35** son validaciones de software. Ninguno de esos conteos se añade a los 906 resultados científicos de esta campaña.

## Evidencia reproducible del corte

El [resumen público en JSON](../_static/cuda_campaign_progress_20261110/summary.json) contiene los conteos por modelo y etapa, el último trabajo confirmado, los enlaces entre recuperaciones y resultados y las comprobaciones de integridad. Su SHA256 es:

```text
f5f1868524eceddc6cd23e7978ab9856d96e185080b4b65dc52f041b3fc4cac1
```

| Identidad | Referencia fijada |
|---|---|
| Estado de campaña utilizado | [Commit `0cbd8f4a115d76d208668d199c6cda84f5d72583`](https://github.com/jassan-bit/DeepLearning_202630/tree/0cbd8f4a115d76d208668d199c6cda84f5d72583/campaigns/e2-cuda-selection-v1) |
| Código científico A3 | [Commit `564ba4762f84a800b02fa4c734f5752a169f5223`](https://github.com/jassan-bit/DeepLearning_202630/tree/564ba4762f84a800b02fa4c734f5752a169f5223) |
| Registro científico | [Archivo fijado en A3](https://github.com/jassan-bit/DeepLearning_202630/blob/564ba4762f84a800b02fa4c734f5752a169f5223/experiments/e2_cuda_search_20261109/registration.json), SHA256 `78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9` |
| Plan sellado | SHA256 `4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda` |

La lectura del corte verificó el esquema público de los registros, el inventario Merkle, los enlaces del diario mediante el código A3 y los enlaces de las dos recuperaciones. También comprobó los **61 hashes de implementación científica**. La actualización del libro no ejecutó ajustes GPU, no abrió CSV ni escribió en la rama de campaña; documenta los resultados que Colab ya había confirmado.
