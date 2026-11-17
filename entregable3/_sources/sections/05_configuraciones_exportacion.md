# 5 Bloque B Selección de configuraciones y exportación

Este bloque reúne la selección documentada de los **16 sistemas de E2** y la ingeniería de inferencia disponible. La CNN-LSTM con atención de E3 se incorpora a la comparación posterior, pero todavía no tiene una exportación de producción verificada. Las configuraciones se eligen desde validación. Las métricas de prueba histórica se muestran con su alcance, sin utilizarlas para cambiar la receta o escoger una semilla favorable.

## 5.1 Criterio de selección


Se recuperaron trece recibos de selección originales y se verificaron sus seis configuraciones, tres folds temporales, parámetros y enlace al ajuste final. E2 eligió por RMSE agrupada en validación; la media aritmética de las tres RMSE elige la misma configuración en **13/13 redes**. No se usó la métrica de prueba ni la mejor semilla para elegir recetas. Los recibos conservan la arquitectura y el punto de HPO elegidos en E2; el sesgo de selección del ganador no está cuantificado. La revisión de esos recibos verifica la decisión registrada, sin certificar una nueva búsqueda de arquitectura ni igualdad de HPO con E3.

La desviación que aparece abajo describe tres folds temporales correlacionados. No es dispersión de semillas independientes; por ello no se inventa un IC de validación. La tabla maestra descargable conserva el corte histórico de E2: media, DE e IC95 % de prueba entre seis semillas, límites condicionales a una cronología, parámetros, FLOPs core y costes existentes. La ampliación posterior a diez semillas por red se informa por separado abajo y en la [sección 8](08_verificacion_estadistica.md); no cambia esta selección de configuraciones.

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

En el corte histórico de seis semillas, el líder descriptivo de la A2 de prueba es el LSTM-autoencoder y sí pertenece al MCS90 de E2; TimeMixer no pertenece a ese conjunto histórico. Esos hechos no se usan para cambiar configuraciones o semillas. El MCS posterior que incorpora E3 es otro análisis, descrito a continuación, y no sustituye ni reescribe ese corte.

### Comparación posterior con las recetas congeladas

La ampliación conserva las seis semillas históricas y añade 271807–271810 a las trece redes de E2, sin repetir HPO ni seleccionar una semilla por desempeño. Hay **diez semillas por red**, más Persistence, SeasonalNaive y ARIMA con una trayectoria determinista cada uno: **133 predictores de E2**. Se comparan con diez semillas de la **CNN causal → LSTM unidireccional → atención temporal, con cabeza positiva aditiva y 1210 parámetros**, sobre las mismas **608 fechas** de prueba histórica. La primaria es la media aritmética de las RMSE individuales, no la RMSE de un ensamble.

La comparación conserva una diferencia de entrenamiento: E2 usa 655 ejemplos y 53 fechas de calibración; E3 usa 532 ejemplos, 116 de validación y **ninguna fecha de calibración**. No se atribuye igualdad de presupuesto o de política de ajuste final. Los 48 contrastes primarios rechazan tras Holm global, pero sus tres procedimientos tienen ámbitos de incertidumbre distintos. El MCS90 auxiliar conserva **diez sistemas, incluido E3**; no identifica un ganador único ni prueba el margen de RMSE. El análisis es posterior sobre un calendario conocido y no acredita R1, R2 o R3 ni potencia suficiente. La guía exige tres pruebas distintas y pertinentes por contraste; no exige literalmente ámbitos de incertidumbre idénticos. Su pertinencia y combinación deben justificarse para la afirmación F2a.

La [sección 8](08_verificacion_estadistica.md) delimita los procedimientos y sus resultados. Se publican las {download}`métricas por sistema <../_static/e3_estadistica_posterior_20261117/metrics.csv>` y los {download}`48 contrastes primarios <../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv>`, sin modificar las recetas heredadas.

## 5.2 Reentrenamiento final exportación y serialización

La guía exige una política común de ajuste final: entrenar sobre el conjunto de entrenamiento completo con las épocas fijadas por validación y conservar pesos, optimizador y configuración. **Esa política común no está cerrada.** Las redes de E2 se ajustaron con 655 ejemplos y épocas de fase previamente fijadas; E3 conserva 532 ejemplos y parada guiada por 116 de validación. La ampliación de semillas no cambia esa diferencia. Los resultados históricos no se vuelven comparables en política de entrenamiento por exportar sus pesos; una evaluación futura con política común debe documentarse como otra etapa.

Es necesario distinguir tres objetos conservados:

| Objeto | Contenido y evidencia | Alcance |
|---|---|---|
| `selected_model.pt` de E2 | Pesos seleccionados, escalador, arquitectura y enlace al pedido de ajuste. Los tres usados para TorchScript carecen de estado de optimizador. | Artefacto de pesos para reconstruir inferencia; no equivale por sí solo al checkpoint completo del entrenamiento. |
| Archivos de fase de la extensión E2 | El inventario inmutable registra 52 modelos seleccionados y 180 checkpoints de fase: inicial, último y mejor para 60 fases, incluidas las fases del autoencoder. El entrenador serializa optimizador, RNG, configuración, época y recursos. | El inventario y el código no sustituyen deserializar y comprobar cada binario. La API declara `resume_exact_supported=False`; conservar estados no certifica reanudación exacta. |
| Checkpoints de las diez semillas E3 | El ejecutor guarda pesos actuales y mejores, optimizador, RNG, generador e historia, vinculados al plan y manifiesto; exige su publicación antes de evaluar. | Evidencia de conservación y reanudación del ejecutor científico; no es aún un artefacto de servicio ni una política de refit común con E2. |

Los [archivos finales de E2](https://github.com/jassan-bit/DeepLearning_202630/tree/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1) y los [checkpoints de E3](https://github.com/jassan-bit/DeepLearning_202630/tree/510c3bddfdc9926322e3210ac61ee1f5eeffad2e/runs/estabilidad_aditiva_v1) mantienen esa procedencia. El conteo de archivos es una auditoría del inventario, no una nueva comprobación de todos sus contenidos.

MLP16, LSTM32 y GRU32 se exportaron a **TorchScript** desde sus checkpoints guardados. Se usó la semilla 271801 por ser el menor ID registrado, no por su RMSE. Se conservaron el escalador y el calibrador originales; no se reentrenó ni recalibró.

La ida y vuelta usa las mismas **53 fechas reales de calibración** y una tolerancia absoluta de **10⁻⁶ en unidades de σ₇**. La tabla muestra la RMSE del artefacto y su diferencia frente al modelo cargado en memoria. No sustituye la verificación del artefacto exportado sobre las 608 fechas de prueba A2, que permanece pendiente. Las 608 ventanas ya están disponibles en la cohorte auditada; la antigua falta de entradas no describe el estado actual, pero disponer de ellas no ejecuta la ida y vuelta.

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

Cada carpeta `artifacts/e3_handoff/<modelo>/` contiene el artefacto, `metadata.json` y su reporte de ida y vuelta, con versiones, semilla, hiperparámetros, normalización, esquema de entrada, escala de salida y calibrador. Los metadatos incluyen la métrica de selección por validación y el cuantil residual del predictor, pero dejan la métrica de prueba exportada en `null` y los rangos de dominio sin estimar; esos campos siguen pendientes. No se sustituyen por resultados de otro artefacto o semilla. La entrada es un tensor float32 ya estandarizado, de forma lote×7×5; la salida debe multiplicarse por la escala del objetivo. La aplicación deberá validar forma, finitud y dominio antes de invocar el artefacto.

### Pendientes para cerrar B

Faltan las exportaciones de los diez modelos Top 10 de E2, los tres comparadores deterministas y la CNN-LSTM con atención de E3, el control completo sobre prueba A2 y sus costes CPU comparables. Los checkpoints de entrenamiento deben vincularse al artefacto seleccionado y verificarse con su configuración y procedencia; no se declara que falten todos los estados de optimizador por inspeccionar sólo los tres archivos de pesos. La política de ajuste final común sigue pendiente.

No se aplicó cuantización ni se atribuyó un límite de memoria a modelos no medidos. Los modelos que no quepan en el despliegue deberán conservar resultados precalculados y una justificación sustentada; la pérdida por compresión, si se aplica, se evaluará frente a δ. El dashboard sólo cargará artefactos y ejecutará inferencia CPU. La guía permite TorchScript como formato de exportación; el perfil de servicio descrito en la [sección 10](10_despliegue_render.md) requiere además resolver ONNX/ONNX Runtime sin el framework de entrenamiento. Los tres TorchScript verifican un avance de exportación, no el cierre de ese perfil de producción.



## 5.3 Revisión del bloque frente a la guía

| Exigencia de la sección 5 | Evidencia al 17 de noviembre de 2026 | Para cerrar |
|---|---|---|
| Selección por validación para cada sistema | 13 recibos neuronales verificados; agrupación y media de folds coinciden en 13/13. Dos comparadores tienen score CV y Persistence receta fija sin score recuperado. | Completar el score faltante o su justificación y la incertidumbre de validación; documentar el sesgo de selección. |
| Mejor global y pertenencia a finalistas | TimeMixer lidera los 15 con score recuperado y queda fuera del MCS histórico. El líder descriptivo de prueba, autoencoder, sí pertenece; el MCS posterior con E3 es otro corte. | Resolver y explicar esa discordancia sin elegir mediante prueba un sustituto del ganador por validación. |
| Tabla maestra: media, DE, IC95 %, parámetros y costes | La tabla histórica de E2 tiene 16 sistemas y seis semillas neuronales; la tabla posterior tiene diez semillas por red y E3. Los IC entre semillas son condicionales al calendario. | Integrar una tabla de despliegue vigente con el mismo alcance por campo, costes comparables y ausencias explícitas. |
| Ajuste final común y checkpoint completo | E2: 655 ejemplos y épocas fijadas; E3: 532 y parada por validación. Hay pesos de inferencia y archivos científicos más completos, con ámbitos distintos. | Fijar la política común y vincular pesos, optimizador, configuración y procedencia; no reetiquetar el experimento ya evaluado. |
| Exportación y metadatos | Tres TorchScript de referencias E2; faltan 14 sistemas si se incluye E3. Métrica exportada de prueba y rangos de dominio siguen ausentes. | Exportar o justificar resultados precalculados, completar contrato y metadatos, sin entrenar en el servidor. |
| Ida y vuelta con tolerancia y métrica | Tres paridades exactas sobre 53 fechas de calibración con tolerancia 10⁻⁶; las 608 entradas de prueba ya están disponibles. | Verificar cada artefacto y su preprocesamiento sobre el conjunto fijo de prueba, con RMSE y diferencia frente a memoria. |
| Compresión cuando haga falta | No se aplicó cuantización, poda o destilación ni se verificó equivalencia. | Medir primero recursos; si se comprime, contrastar la pérdida con δ y usar TOST cuando sea viable. |
| Inferencia comparable y viabilidad de servicio | Tres latencias TorchScript de un hilo y diagnóstico eager CPU histórico de dos hilos, con métodos distintos. FLOPs core no cubren todos los kernels. | Unificar entorno, lote e hilos para mediana/P95, memoria y cómputo del catálogo; medir el servicio final aparte. |

La {download}`matriz descargable del bloque B <../_static/e3_revision_bc_20261117/block_b_requirements.csv>` y su {download}`registro de fuentes y alcance <../_static/e3_revision_bc_20261117/block_b_readout.json>` conservan esta revisión. No se ejecutaron entrenamiento, inferencia, exportación ni nuevas pruebas estadísticas para producirla.

## Suplemento CPU y reproducibilidad

E2 añadió un diagnóstico eager CPU de los **78 predictores neuronales del corte histórico de seis semillas**, con dos hilos y un plan propio. No cubre automáticamente los 130 predictores neuronales de la ampliación a diez semillas. Es distinto de estas tres exportaciones TorchScript, medidas con un hilo. Sus latencias, memorias y diferencias respecto a CUDA no se mezclan como si fueran un único experimento. Ninguna de las dos verificaciones sustituye la métrica del artefacto exportado sobre las 608 entradas originales de prueba.

La tabla maestra conserva parámetros, FLOPs core, costes disponibles y campos ausentes, sin rellenarlos con ceros ni sustituir CPU por GPU. Los FLOPs completos, los costes comparables del resto y sus exportaciones siguen pendientes.

- {download}`Tabla maestra de selección recursos y despliegue <../_static/e3_bloques_ab_20261115/deployment_master.csv>`.
- {download}`Informe de selección <../_static/e3_bloques_ab_20261115/selection_report.json>`.
- [Artefactos y metadatos de las tres referencias](https://github.com/jassan-bit/DeepLearning_202630/tree/main/artifacts/e3_handoff).
- [Suplemento CPU actual de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html).

Los scripts `e3_audit_e2_handoff.py`, `e3_select_e2_configurations.py` y `e3_build_handoff_book.py` conservan la reproducción de sus informes y del corte original de A/B. La [página anterior de evidencias](08_bloques_a_b.md) permanece como anexo; no se alteran sus resultados para reorganizar el libro.
