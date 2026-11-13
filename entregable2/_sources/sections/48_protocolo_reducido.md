# 48. Protocolo reducido y seguimiento de la ejecución

**Registro inicial del 12/11/2026 y seguimiento posterior.** La enmienda conserva los dieciséis sistemas, la métrica y el contrato de E1; reduce las oportunidades HPO antes de nuevas ejecuciones. El registro inicial de reglas, datos, caché y presupuesto no acreditaba entrenamiento nuevo ni aceptación de todos los modelos en GPU. La evidencia posterior se presenta con sus propios recibos y cortes fechados.

La versión científica anterior y sus recibos permanecen en [§45](45_enmienda_protocolo.md). El [informe de §47](47_entrega_consolidada.md) distingue esos antecedentes de los resultados del protocolo reducido. La procedencia histórica se conserva sin renombrar resultados de otra receta.

## Estado vigente: benchmark común completado

**Actualización posterior de diagnóstico:** [§49](49_diagnosticos_f2a.md) conserva una auditoría F2a de los pronósticos y curvas ya guardados, con procedencia y familia exploratoria propias. No modifica el protocolo primario, la congelación, el presupuesto autorizado ni el número de ajustes. Los diagnósticos ejecutados y sus limitaciones se distinguen de los requisitos todavía abiertos.

**Verificación del cierre: 13/11/2026, 11:58:36 de Bogotá (16:58:36 UTC).** Están completados los **78 ajustes finales A100**, los **diez pilotos de desarrollo**, los **tres predictores clásicos finales** y la evaluación común de **16 sistemas sobre 608 fechas**. La [A2 medida](../_static/e2_benchmark_final_20261113/benchmark_A2.csv), el [análisis](../_static/e2_benchmark_final_20261113/benchmark_analysis.json) y la [procedencia](../_static/e2_benchmark_final_20261113/source_evidence.json) documentan el cierre. El resultado es retrospectivo; no acredita por sí solo todos los diagnósticos y entregables de la guía.

Las trece redes aportan seis semillas finales cada una: **60/60 ajustes de los diez candidatos y 18/18 de MLP16/LSTM32/GRU32**. Los pilotos son cinco semillas de iTransformer y cinco de GRU32 y se contabilizan aparte. Persistence, SeasonalNaive y ARIMA aportan un predictor determinista cada uno. La [congelación física](../_static/e2_benchmark_final_20261113/final_freeze.json) vinculó **81 predictores**, sus escaladores y sus calibradores antes de admitir los datos de la evaluación actual.

La selección HPO conserva **231 oportunidades exitosas de 234: 170 entrenamientos nuevos y 61 reutilizaciones verificadas**, además de dos fallos conocidos y un desenlace desconocido. Las trece recetas ganadoras tienen sus tres folds de validación completos. Los tres intentos sin resultado válido no se repiten ni se convierten en éxitos. El [corte de avance del 13/11 a las 09:40](../_static/e2_progress_final_20261113_v2/progress_snapshot.json), su [procedencia](../_static/e2_progress_final_20261113_v2/source_evidence.json) y el [CSV por red](../_static/e2_progress_final_20261113_v2/neural_progress.csv) permanecen como descargas históricas.

| Registro | Alcance vigente |
|---|---|
| [Motor científico](https://github.com/jassan-bit/DeepLearning_202630/commit/1e8745497b484518f766b6da5c19be8362d9add7) | Datos, recetas, dominios, semillas y criterios registrados. |
| [Primera ampliación](https://github.com/jassan-bit/DeepLearning_202630/blob/f540227c23c572b9ca7fb36920588727b029484a/experiments/e2_budget_extension_20261112/registration.json) | Antecedente de **64 a 72 horas acumuladas**, conservando los cargos originales. |
| [Continuación de 80 horas](https://github.com/jassan-bit/DeepLearning_202630/blob/41da88cf600aadc0319214465c94a85335e2b1b7/experiments/e2_budget_continuation_20261113/registration.json) | **80 horas totales acumuladas**: +8 al límite anterior. El código publicado requería confirmación en Colab; el consentimiento y el ancla aplicados ya constan en el control remoto del corte. No reinicia saldo ni resultados. |
| [Política prospectiva de recursos](https://github.com/jassan-bit/DeepLearning_202630/blob/b352ce7f27cd7b55e4141fbfb1548f506ed80987/experiments/e2_resource_continuation_20261112/registration.json) | **1.800 segundos por fase nueva**, máximo 200 épocas y paciencia 20. Las reservas completas son 1.860 segundos para un ajuste normal y 5.460 para SAE. Los resultados antiguos conservan el límite con que se obtuvieron. |
| [Recuperación y transición A100](https://github.com/jassan-bit/DeepLearning_202630/blob/72c04635b7a746d06066273ce61e891975feb510/experiments/e2_a100_recovery_20261113/registration.json) | Conserva los metadatos históricos T4 con su procedencia; usa el perfil A100 observado para oportunidades nuevas. Mantiene GitHub como destino persistente. |
| [Reanudación con transición histórica sellada](https://github.com/jassan-bit/DeepLearning_202630/blob/caf650b67a68226d350ccd6a1f96b10940e0c6a9/experiments/e2_final80_resume_fix_20261113/registration.json) | Reutiliza los tres artefactos históricos y el consentimiento de 80 horas ya aplicado. Verifica la copia local contra GitHub; no concede tiempo adicional ni modifica el motor científico. |
| [Compatibilidad de recibos ARIMA](https://github.com/jassan-bit/DeepLearning_202630/blob/f12cc0904efefde6854342ef18a6c447d5572b3e/experiments/e2_arima_compat_20261113/registration.json) | Corrige el acceso de diagnóstico `raw_value → raw` sin modificar los pronósticos ni repetir el ajuste fallido. Mantiene la elegibilidad por tres folds completos. |
| [Adquisición del periodo E1 registrado](https://github.com/jassan-bit/DeepLearning_202630/blob/24344bd6db20f0546b26167eda4c3076a38c0f4b/experiments/e2_score_source_recovery_20261113/registration.json) | Continúa la adquisición desde los veinte archivos brutos registrados, tras un 404 del derivado privado y antes del primer scoring científico completado. Conserva datos, congelación y ausencia de reajuste. |
| [Recibo terminal del fallo de adquisición](https://github.com/jassan-bit/DeepLearning_202630/blob/125b95263c5f0164b77fb3cb78e9a933d7362e00/experiments/e2_score_receipt_compat_20261113/registration.json) | Respalda la disposición anterior mediante un recibo `failed`; no crea un resultado científico ni reejecuta modelos o evaluación. |

La transición ya consta en el diario remoto. El intento **TSMixer C02 de julio de 2024**, identificado por `f086a89ea56abb834542d12ccee473a6386d12b9de8f456043d78010ada01ede`, perdió su sesión T4 sin un resultado recuperable. Se cerró con **desenlace desconocido**, sin repetirlo, e imputando conservadoramente su reserva original de **1.860 segundos**. Esa imputación es contabilidad conservadora y **no una duración medida**: conserva `optimizer_steps=null`, duración medida desconocida y posible actualización parcial. La conciliación preservó los 97 respaldos existentes entonces. Los respaldos posteriores de selección, ajustes y cierres se identifican en la instantánea actual; el desconocido conserva su disposición de recuperación, sin inventar un manifiesto científico ni una puntuación.

La auditoría valida la caché contra el perfil histórico T4 y los datos, fuentes y política que originaron esos resultados. Los 78 ajustes finales y los diez pilotos se completaron en **NVIDIA A100-SXM4-80GB**, Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6, con FP32 determinista, sin AMP ni TF32. No se afirma igualdad numérica entre GPU distintas. Sus recibos y checkpoints se conservan en GitHub; el tiempo global incluye selección, preparación, fallos, transporte y análisis y no equivale al tiempo de entrenamiento de un único modelo.

Al último cierre del diario, **13/11/2026 a las 11:56:30 de Bogotá**, el consumo acumulado era de **269.830,91 segundos, unas 74,95 horas**, y quedaban **18.169,09 segundos, aproximadamente 5 h 03 min**, del límite autorizado de 80 horas, sin reservas abiertas. Este saldo terminal es posterior al informe de scoring, que aún mostraba 18.189,37 segundos antes de los últimos costes de cierre. Los cargos anteriores no se reiniciaron; el saldo no se utiliza para ampliar R o reabrir HPO a partir de estos resultados ni estima el trabajo documental pendiente.

La pausa anterior ocurrió al intentar regenerar una referencia histórica protegida después de cerrar HPO. El [arreglo de reanudación](https://github.com/jassan-bit/DeepLearning_202630/blob/caf650b67a68226d350ccd6a1f96b10940e0c6a9/experiments/e2_final80_resume_fix_20261113/README.md) valida y reutiliza el perfil, auditoría y recibo sellados, con una admisión separada para una observación A100 compatible. No modifica sus bytes, no repite la selección y conserva la autoridad presupuestaria original de 80 horas. La verificación o restauración de archivos no se cuenta como entrenamiento.

### Comparadores y adquisición: incidencias conservadas

ARIMA recibió **18 órdenes × 3 folds = 54 oportunidades**: 51 produjeron una RMSE válida, dos registraron falta de convergencia y una falló por implementación. La primera oportunidad, orden **(0,0,0) en enero de 2024**, había ajustado antes del fallo de recibo y consumió **14,505 segundos**. Se conserva sin RMSE y no se repite. Quedaron **15 de las 18 órdenes elegibles** con sus tres folds válidos; la receta seleccionada fue **ARIMA(2,0,1)**. SeasonalNaive completó sus nueve oportunidades y eligió periodo 7; Persistence conserva lag 1. Los tres ajustes clásicos finales se completaron antes de congelar los predictores.

La solicitud inicial del CSV privado de E1 devolvió 404 antes de la inferencia final. La continuación utilizó los **veinte archivos mensuales registrados de enero de 2025 a agosto de 2026**, comprobó sus SHA y registró timestamps en **microsegundos** al reconstruir el calendario UTC. Los archivos brutos y sus valores originales no se sustituyeron; el CSV derivado regenerado lleva su propio hash y no se presenta como la misma serialización binaria del derivado histórico. No se abrieron septiembre ni octubre de 2026.

La disposición del fallo de adquisición ya estaba conservada; el respaldo rechazó su estado no terminal. El adaptador posterior archivó esa misma disposición con un recibo terminal **`failed`**, sin alterar el fichero, su cargo ni sus campos desconocidos. Fue una corrección de persistencia, no una nueva corrida científica ni una conversión de fallo en éxito. La evaluación completada utiliza los mismos 81 predictores congelados, sin HPO, reajuste o recalibración después de acceder al periodo de prueba actual.

### Qué significa completar esta etapa

**HPO es selección de hiperparámetros con entrenamiento y validación interna.** No es el testing final. Las **180 oportunidades del Top 10 están contabilizadas**: 177 éxitos, dos fallos conocidos y un desconocido. Las tres referencias E1 completaron sus 54 oportunidades, para **234 contabilizadas y trece recetas ganadoras**. Cada ganador conserva sus tres folds y 249 predicciones de validación; las configuraciones con un fold sin resultado válido no reciben una puntuación completa. Los fallos no se sustituyen por intentos adicionales.

La sección 14 de la guía, pp. 52–55, limita la NAS de dos fases a modelos cuyo artículo no define el cuerpo utilizado. En esos casos requiere las **1.000 arquitecturas** de cribado y la confirmación de los finalistas con nuevas semillas. **Las seis configuraciones HPO actuales no equivalen a ese protocolo NAS.** Las identidades publicadas y sus adaptaciones D7 deben permanecer justificadas; TiDE utiliza la extensión **TIDE-D7-PUBLISHED-R0-v2** documentada más abajo, separada de su antigua variante NAS.

El benchmark común de 16 sistemas y la A2 ya están medidos; los requisitos complementarios conservan su estado en la [matriz de la guía](38_cumplimiento_guia.md). Las 608 fechas heredadas de E1 ya fueron conocidas: la evaluación es **retrospectiva** y la congelación de esta campaña no recupera cegamiento ni constituye confirmación prospectiva independiente. [Resultados y límites](06_benchmark_final.md) · [Discusión](47_entrega_consolidada.md).

## Antecedentes del ejecutor y del almacenamiento


El [motor de ejecución reducido](https://github.com/jassan-bit/DeepLearning_202630/commit/1e8745497b484518f766b6da5c19be8362d9add7) y la [extensión de persistencia en GitHub](https://github.com/jassan-bit/DeepLearning_202630/commit/7ad290ada4ef9908aecfe042b610251206d9b099) quedaron registrados antes de las nuevas corridas. La campaña ya publica su [estado remoto en GitHub](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/control/github-primary-v1/head.json) y recibos de reutilización de caché y de ajustes GPU nuevos. Un [recibo FCN-1D completado](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/runs/08472340feeb25d36cfe22449165e62a5d11029724c94b645cf324ac8109b639/receipt.json) acredita pasos de optimizador en HPO. Los recibos distinguen estos ajustes de la caché; ninguno sustituye la evaluación final común.

El [diagnóstico posterior de Colab, recibido a las 13:11 de Bogotá](../_static/reduced_protocol_20261112/colab_runtime_20261112T181117.json), confirma **Tesla T4, Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6**, con una pequeña prueba CUDA correcta. Su hash del informe es `57e7fd79eba741b06c5304dd0edfefd891ce9c7ea2697fba285b8d99f60f1f38`. La incompatibilidad observada a las 12:59 quedó resuelta; la aceptación de todos los kernels y la política efectiva de cada ajuste siguen requiriendo sus propios recibos.

TiDE tiene ahora la receta **TIDE-D7-PUBLISHED-R0-v2**, que conserva el grafo fuente completo y la rama r=0 inactiva, con esa limitación declarada. La [comparación contra TensorFlow oficial](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_tide_published_r0_20261112/source_parity.json) verificó salida, pérdida, gradiente de entrada y **27 gradientes de parámetros**, con error absoluto máximo **2,38×10⁻⁷**, sin optimizador ni datos de mercado. La [extensión de arquitectura](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_tide_published_r0_20261112/protocol_extension.json) fija su identidad antes del ajuste; no hereda resultados de la antigua variante NAS ni acredita paridad de kernels GPU.

El dominio se mantuvo en **234 oportunidades lógicas HPO**. Antes de ejecutar había 61 candidatos de caché y 173 oportunidades nuevas, incluidos los 18 ajustes de TiDE que habían estado bloqueados. Al corte actual, los **61 resultados reutilizados fueron admitidos** y los 173 intentos nuevos comprenden 170 éxitos y tres incidencias conservadas. La caché de selección no aporta pesos finales reutilizables. Los reajustes finales, los pilotos y los comparadores se contabilizan aparte.

**GitHub es el único destino persistente de la ejecución actual.** El [lanzador registrado](https://github.com/jassan-bit/DeepLearning_202630/blob/7ad290ada4ef9908aecfe042b610251206d9b099/notebooks/run_e2_reduced_github_colab.py) utiliza `/content` únicamente como espacio temporal. Guarda resultados y metadatos en la rama configurada [`results/e2-reduced-delivery-v1`](https://github.com/jassan-bit/DeepLearning_202630/tree/results/e2-reduced-delivery-v1), y pesos/archivos en la versión [`e2-reduced-results-20261112-v1`](https://github.com/jassan-bit/DeepLearning_202630/releases/tag/e2-reduced-results-20261112-v1). El [estado actual](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/control/github-primary-v1/head.json) acredita el respaldo exclusivo en GitHub y permite seguir los recibos sin fijar contadores que envejezcan en esta página. La existencia de recibos de caché no debe contarse como corridas GPU nuevas. El control remoto respalda presupuesto y estado antes de admitir otro trabajador; una reserva desconocida o una subida incompleta detiene la ejecución para conciliación, sin reiniciar saldo ni repetir el ajuste automáticamente.

Los manifiestos preservan las URL originales y señalan los tags particionados **`e2-reduced-results-20261112-v1-p0` a `-pf`** usados para archivos posteriores. La ubicación válida de cada peso es la URL y el hash de su manifiesto; no se supone que todos los pesos pertenezcan a la versión inicial.

El techo inicial fue de **64 horas acumuladas**. Tras la confirmación explícita del usuario, la [ampliación registrada](https://github.com/jassan-bit/DeepLearning_202630/blob/f540227c23c572b9ca7fb36920588727b029484a/experiments/e2_budget_extension_20261112/registration.json) elevó el límite efectivo a **72 horas totales (+8 horas)**, sin reiniciar cargos ni resultados. El ancla aplicada quedó respaldada en el [control de GitHub](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/control/github-primary-v1/head.json). La reanudación concilió un SAE completado cuyo respaldo había fallado por transporte: verificó sus 12.824 pasos, resultado y cargo existentes antes de subirlo, sin repetir el ajuste. Su [recibo recuperado](https://github.com/jassan-bit/DeepLearning_202630/blob/results/e2-reduced-delivery-v1/runs/53b42160571fade6ffa045980ad3351acd40d78d7d285b9bfc81cc029e16587e/receipt.json) permanece separado de las métricas finales. La [migración inicial de reservas](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_reduced_execution_20261112/budget_migration.json) fijó originalmente: **600 segundos por fase**, con reservas completas de **660 segundos** para un ajuste normal y **1.860 segundos** para SAE. Conserva el horizonte máximo de 200 épocas y paciencia 20; alcanzar el límite se registra como fallo. El saldo histórico de 7.472,01 segundos sigue siendo un corte observado que debe conciliarse al ejecutar.

## Contrato y alcance conservados

La tarea sigue siendo **D7×5 OHLCV → σ₇ poblacional de siete retornos logarítmicos de Close**, horizonte de un día y sin anualización. Se conserva la **media de las RMSE individuales por semilla en unidades originales** como métrica primaria y **δ=0,00025772842664981106**. No se sustituye por RMSE del ensamble ni se mezclan escalas entre folds.

El catálogo contiene diez candidatos de cuatro familias, las referencias **MLP16, LSTM32 y GRU32**, y **Persistence, SeasonalNaive y ARIMA**: **13 redes + 3 comparadores = 16 sistemas**. El híbrido adicional y Ridge permanecen opcionales/históricos.

## Seis plazas HPO fijadas antes de comparar resultados

Cada red recibe las mismas seis plazas **C01–C06**, una semilla efectiva de optimización, **271201**, y los mismos tres folds temporales internos. Los puntos se extraen de las configuraciones originales de cada receta sin modificar sus valores numéricos.

| Plaza común | Posición en el dominio original de 18 puntos, base cero |
|---|---:|
| C01 | 5 |
| C02 | 7 |
| C03 | 8 |
| C04 | 9 |
| C05 | 10 |
| C06 | 15 |

**Plazas comunes no significa seis tuplas numéricas idénticas entre arquitecturas.** El generador original es específico por receta. Se conserva una regla común, el número de oportunidades y los espacios aplicables; el [CSV de configuraciones](../_static/reduced_protocol_20261112/hpo_configurations_registered_bytes_v2.csv) expone LR, weight decay, batch y demás campos por modelo/plaza. Los cuatro cuerpos restaurados mantienen los parámetros estructurales y dropout de su fuente.

La elección usó exclusivamente metadatos HPO: cobertura de batch **16/32/64**, weight decay cero y positivo, y warmup cero; después maximizó la menor cobertura normalizada del intervalo logarítmico de LR entre recetas, con desempates deterministas. No utilizó RMSE, rankings de modelos ni tiempos observados para elegir los seis puntos. El subconjunto no representa cobertura completa del dominio original.

Son **13 × 6 × 1 × 3 = 234 ajustes lógicos HPO**, antes de reutilización comprobada. SAE añade 36 etapas auxiliares: **270 etapas optimizadoras** en selección. Los reajustes finales, pilotos, análisis y cualquier NAS aplicable se contabilizan aparte. Las 18 oportunidades y dos semillas del protocolo anterior se conservan como procedencia; el nuevo ejecutor debe usar únicamente la semilla efectiva registrada.

## Tres folds internos y una receta global para la evaluación posterior

Las fechas de mercado son días UTC del contrato E1. Las marcas de registro también se presentan en hora de Bogotá.

| Origen del fold interno | Entrenamiento interno hasta | Validación interna, inclusive |
|---|---|---|
| 01/01/2024 | 25/09/2023 | 03/10–24/12/2023 |
| 01/07/2024 | 25/03/2024 | 02/04–23/06/2024 |
| 01/10/2024 | 25/06/2024 | 03/07–23/09/2024 |

Se mantienen los desplazamientos **o−98** para entrenamiento y **o−90…o−8** para validación, con embargo de siete días. Cada transformación se ajusta únicamente con su entrenamiento admisible.

La receta de cada red se seleccionó como la plaza con **menor RMSE sobre las predicciones de validación concatenadas**, conservando la agregación del contrato E1. Se calcula `sqrt(sum(n_fold × RMSE_fold²) / sum(n_fold))`: 83 fechas comunes por fold, 249 en total. La media aritmética de las tres RMSE queda sólo como diagnóstico. El empate exacto se resuelve por ID estable. Las épocas de reajuste son las medianas de las tres mejores épocas ganadoras; SAE conserva por separado AE1, AE2 y supervisado. Se mantienen los límites de entrenamiento y el horizonte original del scheduler.

La decisión requiere el dominio completo **seis plazas × tres folds × una semilla** con recibos y estados explícitos. Un fallo consume su oportunidad; no autoriza otra semilla ni una configuración de reemplazo. Un error sistemático bloquea la selección.

Esta selección global sólo se aplica a la evaluación que comienza en enero de 2025, posterior a los tres folds. **No se atribuye esa receta global a pronósticos externos de enero–octubre de 2024.** Los resultados internos siguen siendo resultados de selección.

## Ajuste, calibración y evaluación común

| Etapa | Fechas y regla |
|---|---|
| Entrenamiento de pesos y transformaciones finales | Etiquetas admisibles hasta **31/10/2024**, con el mismo corte para las trece redes y referencias |
| Puente de embargo | **01–07/11/2024** |
| Calibración y piloto de desarrollo | **08/11–30/12/2024**, 53 días calendario; recetas previamente congeladas, sin reajuste HPO |
| Puente posterior | **31/12/2024**, observaciones causales para ventanas D7 y estado de comparadores |
| Evaluación heredada de E1 | **01/01/2025–31/08/2026**, 608 fechas calendario y una máscara común verificable |

Reservar calibración dentro de 2023–2024 es una subpartición del desarrollo; no cambia el periodo externo heredado. Todos los sistemas deben respetar esa separación. Los pesos, escaladores y calibradores utilizados en calibración permanecen iguales durante la evaluación: **no se realiza otro refit con etiquetas de diciembre después de calibrar**.

El 31 de diciembre puede alimentar las entradas históricas y la persistencia; ARIMA puede actualizar su estado con coeficientes congelados cuando corresponda. No puede ajustar de nuevo pesos, hiperparámetros, escaladores o calibradores. Su admisión y el soporte de las 608 fechas deben comprobarse antes de evaluar: el loader de desarrollo limitado al 30 de diciembre no basta para el primer pronóstico de enero.

La guía pide heredar la partición de E1; **no exige cien días prospectivos nuevos**. La exposición previa de E1 queda declarada. La campaña actual congeló pesos, transformaciones, calibradores y análisis antes de su scoring completado. Las recuperaciones previas a la inferencia no se presentan como evaluaciones adicionales ni autorizan cambiar decisiones tras los resultados. No se afirma haber recuperado el cegamiento histórico ni una confirmación prospectiva.

## R fijado por recursos y resultado del piloto

**R=6 se conserva como decisión operativa acotada por recursos.** El [piloto ejecutado](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json) completó cinco semillas pareadas de iTransformer y GRU32 sobre las mismas **53 fechas de desarrollo**, sin utilizar sus etiquetas para HPO. Estimó una desviación de las ganancias de **0,000252506**. Con α=0,0025 y R=6, la planificación t no central calcula potencia **0,122953** para la ganancia planificada de **2δ**, inferior al objetivo 0,80. La ganancia mínima para ese objetivo es **0,000878674**, un exceso de **0,000620945** sobre δ.

La estimación depende de sólo cinco pares, dos modelos y un mismo calendario de desarrollo; no garantiza potencia temporal ni la potencia de todos los contrastes del Top 10. **No se alcanzó el objetivo de potencia en esa planificación.** Se informa la limitación con R=6: no se amplían semillas automáticamente después del scoring ni se convierte ausencia de significancia en equivalencia.

## Inferencia registrada y alcance del cierre

La primaria continúa siendo la **media de las RMSE individuales por semilla**, en la escala original de σ₇; no es la RMSE de la predicción promediada. La [A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) declara seis repeticiones por red y una por comparador determinista, con 608 fechas para cada sistema.

La familia registrada contiene **20 contrastes**: cada candidato frente a GRU32 y frente a Persistence. El bootstrap pareado usa bloques móviles de **14 días**, **4.999 remuestreos**, la ganancia de RMSE y el margen δ, con ajuste Holm sobre la familia. **Ninguno de los veinte contrastes primarios rechaza a FWER 0,05.** Las sensibilidades con bloques de 7 y 28 días son diagnósticas sin ajuste. Los contrastes t y cambio de signo por semilla se condicionan a ese mismo calendario; no crean seis mercados independientes ni certifican automáticamente la batería de §7.3.

El [análisis y sus supuestos](../_static/e2_benchmark_final_20261113/benchmark_analysis.json) se conserva tal como se calculó. Los 4.999 remuestreos superan el mínimo de 2.000 de la guía para IC, pero no su umbral de 10.000 para el remuestreo pareado de Tabla C: es una desviación declarada. La revisión adicional debe usar predicciones y pesos ya fijados, identificar los análisis posteriores y conservar la familia primaria sin retocar recetas, selección o márgenes.

La menor RMSE descriptiva corresponde al **LSTM-autoencoder, 0,00380343**. Ese orden no prueba superioridad frente a todos los modelos, equivalencia, ausencia de sobreajuste o calibración suficiente. [A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) · [Contrastes primarios](../_static/e2_benchmark_final_20261113/primary_contrasts.csv) · [Congelación](../_static/e2_benchmark_final_20261113/final_freeze.json) · [Procedencia](../_static/e2_benchmark_final_20261113/source_evidence.json) · [Manifiesto](../_static/e2_benchmark_final_20261113/manifest.json).

Al cierre del registro inicial, doce redes tenían cuerpo definido y **TiDE permanecía bloqueado** hasta verificar una receta publicada compatible y registrar su identidad propia. Ese bloqueo se cerró posteriormente para la nueva receta descrita en la actualización anterior. Las seis filas TiDE heredadas identifican puntos HPO de procedencia; no convierten la variante NAS en un cuerpo publicado. El registro inicial mantenía cualquier requisito NAS aplicable, sin eliminarlo por urgencia.

El [diagnóstico de Colab recibido a las 12:59 p. m. de Bogotá](../_static/reduced_protocol_20261112/colab_runtime_20261112T175910.json) confirma una **Tesla T4 de 15.360 MiB** y una pequeña prueba CUDA correcta, sin controladores de búsqueda activos. El usuario pegó el informe de su sesión privada y su hash se verificó; el agente no accedió directamente al kernel. **No estaba entrenando**: cero pasos de optimizador.

La disponibilidad de hardware no cierra la compatibilidad: el intérprete seleccionado era **Python 3.13.16 / Torch 2.14.1+cu130 / CUDA 13.0**, mientras la política registrada requiere **Python 3.12 / Torch 2.8.0+cu126 / CUDA 12.6**. Ese diagnóstico requirió preparar el intérprete aislado y repetir la comprobación; el informe posterior de las 13:11, enlazado arriba, confirmó las versiones compatibles. La [celda de preparación del entorno](https://github.com/jassan-bit/DeepLearning_202630/blob/main/notebooks/preparar_e2_reducido_runtime_colab.py) instala las dependencias fijadas en una carpeta nueva, conserva el kernel y repite la comprobación; no inicia HPO ni cambia el presupuesto. Los flags observados antes de la prueba no acreditan la política de ajuste; la prueba de matriz tampoco valida todos los kernels de los modelos. El [recibo de procedencia](../_static/reduced_protocol_20261112/colab_runtime_receipt.json) conserva estas limitaciones.

El informe local del registro inicial identificaba **Python 3.12.14 y Torch 2.8.0+cpu, sin CUDA**. El [notebook de diagnóstico](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_comprobar_protocolo_reducido.ipynb) no entrena ni instala paquetes. El informe recibido no estaba conservado en Drive; esta copia pública preserva su evidencia, no los datos ni pesos de la sesión.

## Qué acredita la comprobación de caché y presupuesto

**Resultado histórico del registro inicial del paso 1:** 61 resultados HPO coincidían con las nuevas plazas y las firmas de datos preparados; seguían condicionados a verificar el trabajador. En ese corte quedaban 155 ajustes sin resultados de la receta vigente y 18 plazas TiDE bloqueadas. La extensión posterior admite una receta TiDE nueva: si se aprueban los 61 candidatos, son 173 ajustes HPO nuevos pendientes. Había **cero reutilizaciones aprobadas y cero pesos finales reutilizables** en ese corte inicial.

La auditoría conserva la trazabilidad de ambos diarios y comprueba los archivos de desarrollo registrados, los parámetros de cada plaza y los recibos de ajuste. Distingue una **candidatura reutilizable** de una reutilización totalmente verificada: todavía deben coincidir datos preparados, receta, semilla y política de ejecución efectiva.

La campaña histórica guardó **metadatos de selección**, no pesos finales. Un resultado HPO compatible puede evitar repetir esa oportunidad de selección; no sustituye los reajustes finales, sus curvas ni sus checkpoints. Tampoco se reutiliza directamente un ganador elegido entre 18 puntos como ganador del nuevo dominio de seis.

En aquel corte inicial, el techo era de **64 horas acumuladas**, con todos los cargos anteriores: **222.927,99 segundos consumidos y 7.472,01 segundos de saldo observado**. En el registro inicial, la siguiente reserva histórica de 7.500 segundos no cabía; aquel registro no la modificó. El [recibo de conciliación](../_static/reduced_protocol_20261112/budget_reconciliation.json) fija el corte observado, no un saldo garantizado en vivo. Ese registro inicial no reinició el presupuesto ni concedió tiempo; las ampliaciones posteriores autorizaron primero 72 y después **80 horas acumuladas**, preservando todos los cargos. La migración registrada arriba fija las reservas nuevas y debe validarse antes de admitir cada ajuste; reducir el dominio HPO por sí solo no modificaba la política anterior.

## Archivos de este registro

- [Protocolo reducido](../_static/reduced_protocol_20261112/protocol.json) · [configuraciones completas](../_static/reduced_protocol_20261112/configurations.json) · [configuraciones CSV](../_static/reduced_protocol_20261112/hpo_configurations_registered_bytes_v2.csv) · [calendario](../_static/reduced_protocol_20261112/calendar_registered_bytes_v2.csv).
- [Auditoría de datos y caché](../_static/reduced_protocol_20261112/data_cache_audit.json) · [presupuesto conciliado](../_static/reduced_protocol_20261112/budget_reconciliation.json) · [entorno local observado](../_static/reduced_protocol_20261112/environment_local.json).
- [Recibo de fuentes y hashes](../_static/reduced_protocol_20261112/registration.json) · [registro Git y fecha real](../_static/reduced_protocol_20261112/git_registration.json) · [verificación](../_static/reduced_protocol_20261112/verification.json).

El cierre de este primer paso es un registro verificable de reglas y comprobaciones. Los recibos posteriores deberán acreditar la política GPU efectivamente aplicada, el presupuesto admitido, los ajustes y sus resultados. La verificación de arquitectura y el diagnóstico compatibles añadidos arriba no reemplazan los recibos de entrenamiento.

## Procedencia de las descargas CSV

Las descargas vigentes de configuraciones y calendario conservan los bytes CRLF vinculados por el registro científico. La primera publicación normalizó esos saltos de línea a LF sin cambiar valores; sus URL se preservan como historial: [configuraciones de la primera publicación](../_static/reduced_protocol_20261112/hpo_configurations.csv) y [calendario de la primera publicación](../_static/reduced_protocol_20261112/calendar.csv). La [conciliación de URL y hashes](../_static/reduced_protocol_20261112/publication_url_versioning.json) permite distinguir ambas representaciones. No cambia el protocolo ni sus fuentes registradas.

