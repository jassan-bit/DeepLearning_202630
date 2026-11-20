# 5 Bloque B Selección de configuraciones y exportación

Este bloque reúne la selección documentada de los **16 sistemas de E2** y los artefactos de inferencia. Las vistas originales habilitan cinco ONNX CPU de la semilla 271801: MLP16, LSTM32, GRU32, CNN-LSTM aditiva E3 y TSMixer. La ampliación `/perfil-14` incorpora las otras nueve redes históricas y pasó la {ref}`aceptación pública de 10.10 <e3-render-public-acceptance-20261120>` sobre Render H. Los tres clásicos conservan historia/estado adicional al contrato de siete barras. Pesos, escaladores y cuantiles no cambian; V3 sigue experimental y deshabilitada. Las configuraciones se eligen desde validación. La aceptación del producto no modifica la receta, las medias científicas o la elección de semilla.

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

La comparación conserva una diferencia de entrenamiento: E2 usa 655 ejemplos y 53 fechas de calibración; E3 usa 532 ejemplos y 116 de validación y, en ese corte científico original, **ninguna fecha de calibración**. La adopción posterior de un cuantil sobre 53 fechas separadas se documenta abajo; no cambia pesos, presupuesto o política de ajuste final. Los 48 contrastes primarios rechazan tras Holm global, pero sus tres procedimientos tienen ámbitos de incertidumbre distintos. El MCS90 auxiliar conserva **diez sistemas, incluido E3**; no identifica un ganador único ni prueba el margen de RMSE. El análisis es posterior sobre un calendario conocido y no acredita R1, R2 o R3 ni potencia suficiente. La guía exige tres pruebas distintas y pertinentes por contraste; no exige literalmente ámbitos de incertidumbre idénticos. Su pertinencia y combinación deben justificarse para la afirmación de la ruta temporal F2.

La [sección 8](08_verificacion_estadistica.md) delimita los procedimientos y sus resultados. Se publican las {download}`métricas por sistema <../_static/e3_estadistica_posterior_20261117/metrics.csv>` y los {download}`48 contrastes primarios <../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv>`, sin modificar las recetas heredadas.

## 5.2 Reentrenamiento final exportación y serialización

La guía exige una política común de ajuste final: entrenar sobre el conjunto de entrenamiento completo con las épocas fijadas por validación y conservar pesos, optimizador y configuración. **Esa política común no está cerrada.** Las redes de E2 se ajustaron con 655 ejemplos y épocas de fase previamente fijadas; E3 conserva 532 ejemplos y parada guiada por 116 de validación. La ampliación de semillas no cambia esa diferencia. Los resultados históricos no se vuelven comparables en política de entrenamiento por exportar sus pesos; una evaluación futura con política común debe documentarse como otra etapa.

Es necesario distinguir tres objetos conservados:

| Objeto | Contenido y evidencia | Alcance |
|---|---|---|
| `selected_model.pt` de E2 | Pesos seleccionados, escalador, arquitectura y enlace al pedido de ajuste. Los tres usados para TorchScript carecen de estado de optimizador. | Artefacto de pesos para reconstruir inferencia; no equivale por sí solo al checkpoint completo del entrenamiento. |
| Archivos de fase de la extensión E2 | El inventario inmutable registra 52 modelos seleccionados y 180 checkpoints de fase: inicial, último y mejor para 60 fases, incluidas las fases del autoencoder. El entrenador serializa optimizador, RNG, configuración, época y recursos. | El inventario y el código no sustituyen deserializar y comprobar cada binario. La API declara `resume_exact_supported=False`; conservar estados no certifica reanudación exacta. |
| Checkpoints de las diez semillas E3 | El ejecutor guarda pesos actuales y mejores, optimizador, RNG, generador e historia, vinculados al plan y manifiesto; exige su publicación antes de evaluar. | Evidencia de conservación y reanudación del ejecutor científico. El ONNX de servicio de la semilla 271801 es un producto derivado con su propia paridad; no convierte los checkpoints en una política de refit común con E2. |

Los [archivos finales de E2](https://github.com/jassan-bit/DeepLearning_202630/tree/5e9023ea821bfa9fb0b92340d518413f6ab89161/runs/e2_extension_10_semillas_v1) y los [checkpoints de E3](https://github.com/jassan-bit/DeepLearning_202630/tree/510c3bddfdc9926322e3210ac61ee1f5eeffad2e/runs/estabilidad_aditiva_v1) mantienen esa procedencia. El conteo de archivos es una auditoría del inventario, no una nueva comprobación de todos sus contenidos.

MLP16, LSTM32 y GRU32 se exportaron a **TorchScript** desde sus checkpoints guardados. Se usó la semilla 271801 por ser el menor ID registrado, no por su RMSE. Se conservaron el escalador y el calibrador originales; no se reentrenó ni recalibró.

La ida y vuelta de estos **tres TorchScript históricos** usa las mismas **53 fechas reales de calibración** y una tolerancia absoluta de **10⁻⁶ en unidades de σ₇**. La tabla muestra la RMSE de cada TorchScript y su diferencia frente al modelo cargado en memoria. Sus recibos no verifican esos mismos archivos TorchScript sobre las 608 fechas de prueba A2. La etapa posterior sí comprobó las 608 ventanas para **cuatro artefactos ONNX distintos**, incluido E3, como se describe en las [exportaciones ONNX verificadas](#exportaciones-onnx-verificadas) y en la [sección 9](09_dashboard_predictivo.md); ese resultado conserva su formato, semilla y recibos propios.

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

Cada carpeta `artifacts/e3_handoff/<modelo>/` contiene el TorchScript histórico, `metadata.json` y su reporte de ida y vuelta, con versiones, semilla, hiperparámetros, normalización, esquema de entrada, escala de salida y calibrador. Estos metadatos dejan la métrica de prueba exportada en `null` y los rangos de dominio sin estimar; esos campos siguen pendientes para estos archivos y no se sustituyen por los de otro formato o semilla. Su entrada es un tensor float32 ya estandarizado, de forma lote×7×5; su salida debe multiplicarse por la escala del objetivo. El runtime actual usa los ONNX y sus metadatos propios, con validación de forma y finitud y avisos de dominio descritos en la [sección 9](09_dashboard_predictivo.md).

(exportaciones-onnx-verificadas)=
### Exportaciones ONNX verificadas

La etapa posterior del dashboard añadió cuatro exportaciones ONNX: MLP16, LSTM32, GRU32 y la CNN-LSTM aditiva de E3. Su paridad raw→preprocesamiento→ORT se verificó sobre las 608 ventanas comunes, con semilla fija 271801 y tolerancia absoluta 10⁻⁶; la [sección 9](09_dashboard_predictivo.md) conserva los errores y la diferencia frente a las predicciones congeladas. La identidad de los tres TorchScript históricos y su control sobre 53 fechas de calibración se mantiene por separado.

La actualización de cierre añade **TSMixer como quinta exportación**, sin alterar los cuatro artefactos anteriores: ONNX de 165263 bytes y 35044 parámetros, checkpoint original de la semilla 271801, escalador y calibrador conservados. La {download}`paridad TSMixer <../_static/e3_cierre_integral_20261118/serving/export_capabilities/tsmixer/roundtrip.json>` verifica 608 ventanas de evaluación y 53 de calibración; el error máximo frente a las predicciones guardadas es **2,285×10⁻⁸ σ₇**, inferior a 10⁻⁶. El catálogo se amplía mediante `artifacts/e3_serving_extensions_v2/`; se conservan la {download}`procedencia y los hashes <../_static/e3_cierre_integral_20261118/serving/export_capabilities/tsmixer/provenance.json>`.

E3 adopta en `artifacts/e3_serving_calibration_v2/` el cuantil posterior existente **q = 0,005402463770651868**, ligado mediante hashes al mismo ONNX, checkpoint, metadatos, escala y semilla. Sus 53 fechas de calibración van del 8 de noviembre al 30 de diciembre de 2024, después del ajuste y la validación; el servicio sólo lo aplica a fechas objetivo posteriores al 30 de diciembre. El intervalo nominal del 90 % mantiene la predicción puntual y usa `[max(0, predicción−q), predicción+q]`. La cobertura histórica del **predictor individual 271801** es **94,08 % en 608 fechas** y **85,56 % en 2026Q1**; no son medias de diez semillas ni una garantía temporal o por grupo. La {download}`verificación ORT de 661 ventanas <../_static/e3_cierre_integral_20261118/serving/calibration_runtime_verification.json>` y las {download}`notas de adopción <../_static/e3_cierre_integral_20261118/serving/README.md>` explican también el efecto numérico de frontera en una observación de calibración. No se reentrenó ni recalibró para esta adopción y el análisis histórico no se reescribe.

### Pendientes para cerrar B

El cierre posterior verificó **las nueve exportaciones neuronales restantes**, con sus recibos y pruebas de paridad offline en las 608 ventanas de prueba y 53 de calibración. Junto con las cinco existentes, completan catorce representantes neuronales ONNX verificados. Esa exportación no implicó su habilitación automática; el despliegue y la aceptación pública de las catorce redes se ejecutaron después y están comprobados en 10.10. El {ref}`cierre técnico <e3-completion-20261120>` conserva los archivos y sus límites. La {download}`matriz de capacidades del corte del 18 de noviembre <../_static/e3_cierre_integral_20261118/serving/export_capabilities/capability_matrix.csv>` mantiene su estado histórico: TSMixer exportado, tres clásicos justificados bajo el contrato de siete barras y nueve redes todavía pendientes en aquel corte. Los checkpoints deben vincularse al artefacto seleccionado y verificarse con su configuración y procedencia; conservar estados o completar conversiones no cierra la política común de ajuste final. Los costes completos comparables conservan sus campos pendientes; la integración y aceptación técnica de las nueve redes adicionales están aprobadas para Render H en 10.10. Exportar una réplica no cambia la comparación científica de medias entre diez semillas.

No se aplicó cuantización ni se atribuyó un límite de memoria a modelos no medidos. Los modelos que no quepan en el despliegue deberán conservar resultados precalculados y una justificación sustentada; la pérdida por compresión, si se aplica, se evaluará frente a δ. La guía permite TorchScript. El perfil CPU sin framework de entrenamiento de la [sección 10](10_despliegue_render.md) dispone de cinco ONNX para la aplicación local; los tres TorchScript conservan su alcance histórico. El servicio público y los pendientes señalados siguen sin completar ese cierre.



## 5.3 Revisión del bloque frente a la guía

| Exigencia de la sección 5 | Evidencia histórica y actualización de cierre | Para cerrar |
|---|---|---|
| Selección por validación para cada sistema | 13 recibos neuronales verificados; agrupación y media de folds coinciden en 13/13. Dos comparadores tienen score CV y Persistence receta fija sin score recuperado. | Completar el score faltante o su justificación y la incertidumbre de validación; documentar el sesgo de selección. |
| Mejor global y pertenencia a finalistas | TimeMixer lidera los 15 con score recuperado y queda fuera del MCS histórico. El líder descriptivo de prueba, autoencoder, sí pertenece; el MCS posterior con E3 es otro corte. | Resolver y explicar esa discordancia sin elegir mediante prueba un sustituto del ganador por validación. |
| Tabla maestra: media, DE, IC95 %, parámetros y costes | La tabla histórica de E2 tiene 16 sistemas y seis semillas neuronales; la tabla posterior tiene diez semillas por red y E3. Los IC entre semillas son condicionales al calendario. | Integrar una tabla de despliegue vigente con el mismo alcance por campo, costes comparables y ausencias explícitas. |
| Ajuste final común y checkpoint completo | E2: 655 ejemplos y épocas fijadas; E3: 532 y parada por validación. Hay pesos de inferencia y archivos científicos más completos, con ámbitos distintos. | Fijar la política común y vincular pesos, optimizador, configuración y procedencia; no reetiquetar el experimento ya evaluado. |
| Exportación y metadatos | Tres TorchScript históricos y catorce representantes neuronales ONNX verificados offline, incluidos los nueve del cierre posterior; el perfil vigente sigue habilitando cinco. Escaladores, esquema y hashes conservan su versión; E3 adopta por separado su intervalo posterior individual. | Completar la política final común, los costes comparables y, si procede, integrar y aceptar los nuevos artefactos en el producto. Los tres clásicos conservan su justificación de contrato. |
| Ida y vuelta con tolerancia y métrica | Los cinco ONNX del perfil vigente conservan su comprobación en 608 ventanas y tolerancia absoluta 10⁻⁶; TSMixer también sobre 53 de calibración. Las nueve redes adicionales ya tienen paridad offline en ambas particiones. Las tres paridades TorchScript históricas conservan su alcance. | Mantener la evidencia por formato/semilla y verificar la integración del producto; no confundir una réplica exportada con la media de diez semillas. |
| Compresión cuando haga falta | No se aplicó cuantización, poda o destilación ni se verificó equivalencia. | Medir primero recursos; si se comprime, contrastar la pérdida con δ y usar TOST cuando sea viable. |
| Inferencia comparable y viabilidad de servicio | Cuatro ONNX comparten el perfil histórico de latencia; TSMixer tiene una medición separada y no se incorpora a esa frontera. Los tres perfiles TorchScript y el diagnóstico eager CPU conservan métodos distintos. FLOPs core no cubren todos los kernels. | Completar memoria y cómputo comparables de toda la familia y medir el servicio final aparte. |

La {download}`matriz descargable del bloque B <../_static/e3_revision_bc_20261117/block_b_requirements.csv>` y su {download}`registro de fuentes y alcance <../_static/e3_revision_bc_20261117/block_b_readout.json>` conservan el corte documental del **17 de noviembre de 2026**, anterior a las cuatro exportaciones ONNX. No se ejecutaron entrenamiento, inferencia, exportación ni nuevas pruebas estadísticas para producir esos documentos; sus campos de exportación describen los tres TorchScript sobre 53 fechas. La evidencia ONNX posterior está en la [sección 9](09_dashboard_predictivo.md) y sus recibos, sin reescribir los archivos históricos.

## Suplemento CPU y reproducibilidad

E2 añadió un diagnóstico eager CPU de los **78 predictores neuronales del corte histórico de seis semillas**, con dos hilos y un plan propio. No cubre automáticamente los 130 predictores neuronales de la ampliación a diez semillas. Es distinto de estas tres exportaciones TorchScript, medidas con un hilo. Sus latencias, memorias y diferencias respecto a CUDA no se mezclan como si fueran un único experimento. Esos dos controles históricos no verifican los TorchScript sobre las 608 entradas originales de prueba; la paridad y las métricas de los cuatro ONNX posteriores sí cubren esas 608 ventanas, con su alcance propio.

La tabla maestra conserva parámetros, FLOPs core, costes disponibles y campos ausentes, sin rellenarlos con ceros ni sustituir CPU por GPU. Los FLOPs completos y los costes comparables que faltan siguen pendientes. Las nueve exportaciones adicionales ya están verificadas offline; esa conversión no completa la contabilidad de recursos ni la aceptación del servicio.

- {download}`Tabla maestra de selección recursos y despliegue <../_static/e3_bloques_ab_20261115/deployment_master.csv>`.
- {download}`Informe de selección <../_static/e3_bloques_ab_20261115/selection_report.json>`.
- [Artefactos y metadatos de las tres referencias](https://github.com/jassan-bit/DeepLearning_202630/tree/main/artifacts/e3_handoff).
- [Suplemento CPU actual de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html).

Los scripts `e3_audit_e2_handoff.py`, `e3_select_e2_configurations.py` y `e3_build_handoff_book.py` conservan la reproducción de sus informes y del corte original de A/B. La [página anterior de evidencias](08_bloques_a_b.md) permanece como anexo; no se alteran sus resultados para reorganizar el libro.


(e3-completion-20261120)=
## 5.4 Cierre técnico de exportaciones y medición comparable

El cierre del 20 de noviembre de 2026 recuperó los checkpoints históricos
de la semilla representante **271801** y verificó las **nueve exportaciones
ONNX restantes**: FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con
predictor, TimesNet, TiDE, TimeMixer, PatchTST e iTransformer. Sus archivos
suman **14.312.258 bytes**. Cada red se comprobó sobre **608 ventanas de
prueba y 53 de calibración**, 661 por artefacto, sin reentrenar, escoger otra
semilla o reajustar escaladores y cuantiles. El error absoluto máximo de
los nueve recibos es **3,1492137758×10⁻⁷ σ₇**, inferior a la tolerancia
**10⁻⁶**. La {download}`revisión independiente de procedencia y recibos <../_static/e3_completion_20261120/qa/nine_export_provenance_review.json>`
contrasta hashes de checkpoints y ONNX y conserva el alcance de cada
verificación; esa revisión no reejecutó los modelos.

Se dispone así de **catorce representantes neuronales ONNX verificados
offline**. En aquel corte offline, el perfil original habilitaba **cinco**, con los
otros **doce sistemas precalculados**. El cierre posterior de 10.10 ya
verifica las catorce redes en `/perfil-14`; los tres clásicos mantienen
los requisitos de historia o estado que no se determinan con siete barras. La V3 experimental es otro artefacto y no amplía automáticamente
el registro del dashboard. Las conversiones no modifican las medias de
diez semillas ni el alcance de los experimentos científicos anteriores.

El autor aprobó explícitamente publicar el paquete revisado. El [archivo público de las nueve redes](https://github.com/jassan-bit/DeepLearning_202630/releases/download/e3-nine-onnx-4ef050d047b1/e3-nine-onnx-20261120.zip) tiene **12.987.835 bytes**, SHA256 `0fb857c407250949bfc8c96fd7b9525bab624b650b5c94dd4d1d1d8c60f62c66` y 66 miembros. Su manifiesto de publicación se añade a los binarios y recibos originales; por eso el ZIP tiene un hash distinto del transporte revisado de 12.980.054 bytes (`df995ef4e5185022de21cf49dbe13544111f1e5fb4503fd869c3f2df962cc332`). **Los nueve ONNX conservan exactamente sus hashes**, pesos, escaladores y calibradores. El {download}`recibo de descarga pública independiente <../_static/e3_completion_publication_20261120/public_download_integrity.json>` verificó ZIP y manifiesto sin credenciales. Los {download}`localizadores de origen <../_static/e3_completion_20261120/exports/public_source_unauthenticated_verification.json>` conservan la procedencia de los checkpoints públicos.

La recreación previa de FCN-1D en otro runner produjo un SHA distinto y se detuvo antes de publicar. Se conserva el {download}`fallo anterior <../_static/e3_completion_20261120/exports/remote_attempt/failure_observation.json>`; no se rebajó la guarda ni se atribuyó su causa a metadatos o hardware sin demostrarlo. El procedimiento publicado después **ensambla los bytes ya aceptados y los comprueba sin reexportar**. La [ejecución remota aprobada](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44770549685) pasó **5949 casos, nueve redes por 661 ventanas**, con error máximo **3,099255844×10⁻⁷ σ₇**, inferior al umbral fijado 10⁻⁶, y cero discrepancias de cobertura. El {download}`replay ORT completo <../_static/e3_completion_publication_20261120/E3_ONNX_PARITY_REPORT.json>` registra Python 3.12.14, NumPy 2.3.5 y ONNX Runtime 1.23.2 de CPU, sin Torch, entrenamiento o recalibración. Este resultado no transforma en aprobada la reexportación binaria idéntica que falló.

El [perfil V3 experimental](https://github.com/jassan-bit/DeepLearning_202630/tree/4ef050d047b1f85d39f9b47fce86a784820880c4/artifacts/e3_experimental_v3_20261120) está archivado aparte: **15.368 bytes**, SHA256 `881dd637837265829aa96900d7a0c864dcdab113a57a91485feb20fd2a443abc`. La {download}`lectura pública de V3 <../_static/e3_completion_publication_20261120/V3_public_raw_integrity.json>` verificó HTTP 200 y todos sus bytes, sin ejecutar el modelo. No tiene calibrador ni se incorpora al registro del dashboard.

La exportación offline y la aceptación pública son comprobaciones distintas. El cierre posterior de 10.10 acredita Render H y los nueve modelos integrados, con los clientes de cinco y catorce; no mide capacidad remota concurrente. V3 experimental sigue sin calibrador ni promoción.

### Medición CPU bajo el mismo procedimiento

Se ejecutó una medición nueva para **quince perfiles**: los catorce
representantes históricos y `atom_crps` experimental. Cada artefacto corrió
en un proceso separado, fijado a una CPU lógica del **AMD EPYC 9V74**,
con **ONNX Runtime 1.23.2**, un hilo, batch uno, diez calentamientos y cien
llamadas medidas. Se utilizó la misma ventana OHLCV de siete barras y el
escalador congelado propio de cada modelo; la ventana es un fixture de
tiempo, no una prueba científica nueva ni una selección de modelo. El
{download}`recibo completo <../_static/e3_completion_20261120/costs/measured/receipt.json>`
identifica versiones, afinidad, hashes, carga de sesión, primera llamada,
p95 y observaciones individuales. La tabla reproduce las medianas de
`session.run` y el RSS máximo del proceso:

| Artefacto | Mediana forward (ms) | RSS máximo del proceso (bytes) |
|---|---:|---:|
| CNN-LSTM aditiva E3 | 0,0234455 | 49.197.056 |
| FCN-1D | 0,2047125 | 50.065.408 |
| GRU32 | 0,0112420 | 48.406.528 |
| Inception-1D | 0,4333315 | 53.075.968 |
| LSTM-autoencoder con predictor | 0,0173460 | 48.791.552 |
| LSTM32 | 0,0109110 | 48.427.008 |
| MLP16 | 0,0053785 | 48.062.464 |
| PatchTST | 0,0995195 | 50.032.640 |
| ResNet-1D | 0,3134065 | 52.555.776 |
| TSMixer | 0,0681220 | 49.803.264 |
| TiDE | 0,0323635 | 49.664.000 |
| TimeMixer | 0,2278025 | 51.212.288 |
| TimesNet | 0,3686445 | 67.100.672 |
| V3 `atom_crps` experimental | 0,0279120 | 49.463.296 |
| iTransformer | 0,2206770 | 54.951.936 |

El {download}`CSV de costes <../_static/e3_completion_20261120/costs/measured/costs.csv>`
conserva los valores sin redondear y el RSS basal por separado. El RSS
máximo incluye Python, NumPy, ONNX Runtime y la sesión: **no es memoria
incremental del modelo**. Se mide el forward, sin incluir preprocesamiento,
validación del CSV, explicaciones, Dash, navegador o transporte HTTP. Las
cachés de archivos no se vaciaron, por lo que la carga observada tampoco
es una estimación garantizada de arranque en frío. No se midieron FLOPs,
energía, GPU de entrenamiento o concurrencia. Esas condiciones impiden
trasladar las cifras directamente a la latencia pública de Render.

Este cierre documenta ingeniería ejecutada con IA declarada. El
{download}`control de manifiestos <../_static/e3_completion_20261120/qa/module_manifest_review.json>`
separa los bytes auditados de los archivos no incluidos en cada paquete.
No acredita R2/R3 ni cambia el {ref}`dictamen documental favorable y acotado de R1 para V3 <e3-r1-substantive-verdict-20261120>`, que permanece
separado de la aditiva y de la evaluación institucional.

El {ref}`perfil14 optativo <e3-followup-profile14>` integra los catorce ONNX históricos y conserva pesos, escaladores y cuantiles. Las pruebas locales conservan su alcance; la aceptación pública posterior de ambos clientes está aprobada en 10.10. V3 continúa separada y deshabilitada.
