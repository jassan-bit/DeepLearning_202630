# 7. Limitaciones Inferenciales de Trabajar con Tres Modelos

Un benchmark con tres modelos y pocas ejecuciones puede tener **potencia muy baja**. El Entregable 1 exige reconocerlo y cuantificarlo, en lugar de interpretar los p-values como si cualquier diseño pudiera detectar diferencias pequeñas.

Este capítulo contrasta las páginas 32–34 de la guía con la validación forward chaining ejecutada mediante `timeseries-cv`, proyecto de DidierRLopes con coautoría de Filipe R. Ramos. El [método de CV](../_static/dl_r2_cv_20261104/method.json), los [folds](../_static/dl_r2_cv_20261104/folds.json) y el [resumen de ejecución](../_static/dl_r2_cv_20261104/execution_summary.json) acreditan **60 trayectorias, 720 segmentos modelo–mes y 1.440 ajustes efectivos**, con diez semillas por red y variante y 366 predicciones diarias por trayectoria. El [complemento estadístico](../_static/filipe_statistics_20261104/summary.json) añade cálculos sobre las mismas predicciones, sin nuevos entrenamientos ni acceso al test. Su enmienda se registró después de los descriptivos; los resultados conservan alcance exploratorio. Los límites discretos de Wilcoxon que siguen son teóricos, no ejecuciones de ese test.

## 7.1. Potencia, tamaño de muestra y multiplicidad

### 7.1.1. Friedman con tres modelos y cinco bloques

Con tres modelos y cinco bloques, Friedman puede tener poca potencia y el post-hoc de Nemenyi ser conservador. La ausencia de significación no demuestra equivalencia ni descarta diferencias de importancia práctica.

Esa advertencia no es una estimación numérica de potencia ni significa que sea imposible detectar cualquier efecto. La potencia depende también del efecto alternativo, la dispersión, los empates, la unidad de bloque y la calibración. Con pocos bloques se identificará la referencia exacta, de permutación o aproximada utilizada y se justificarán sus condiciones, como exige §5.4.3.

Los bloques deben representar unidades defendibles del experimento. Las 366 fechas de validación, las semillas y las configuraciones no se mezclarán como si fueran réplicas independientes de la misma naturaleza. Se conservarán el emparejamiento, el alcance de la inferencia y la dependencia temporal descritos en §5.4.

Un bloque completo contiene los resultados de la misma unidad bajo las alternativas comparadas. Nemenyi controla simultáneamente su familia; no se añade automáticamente Holm como una segunda corrección equivalente. El análisis actual utiliza diez bloques-semilla para las tres redes relativas. Friedman dio χ²=4,2, p=0,122456 bajo referencia asintótica aproximada. Las comparaciones pareadas registradas no quedaron condicionadas a rechazar ese omnibus: la familia global contiene doce hipótesis por semillas, con Holm-12. No se ejecutó Nemenyi ni se aplica Holm como una segunda corrección de ese post-hoc.

### 7.1.2. Límite del Wilcoxon pareado

Para $R$ diferencias pareadas independientes bajo el alcance declarado, sin ceros ni empates en sus valores absolutos, y el modelo nulo de signos equiprobables del Wilcoxon bilateral exacto, el menor p-value posible, cuando todas tienen la misma dirección, es:

$$
p_{\min}=\frac{2}{2^R}.
$$

Con **cinco pares**, $p_{\min}=0{,}0625>0{,}05$: ese Wilcoxon bilateral no puede rechazar al nivel indicado ni en el caso más extremo. Con **seis**, el mínimo es $0{,}03125$. R cuenta pares de resultados de A y B, no la suma de corridas de los tres modelos. Ceros, empates, método exacto o aproximado y convención del software deben declararse; el número de diferencias útiles puede diferir del nominal.

La guía establece cinco ejecuciones como mínimo general y seis para pruebas por rangos, pero el número definitivo deberá responder al piloto y la potencia. No se elegirá una prueba paramétrica solo para conseguir significación: sus supuestos deben justificarse.

Tampoco se cambiará retrospectivamente a un contraste unilateral ni se escogerá un bootstrap porque produzca un resultado favorable. Wilcoxon no contrasta automáticamente la misma media que una t pareada; para una interpretación de desplazamiento requiere condiciones sobre las diferencias, incluida simetría. El bootstrap necesita su propio diseño y puede tener calibración limitada con tan pocas unidades.

### 7.1.3. Corrección de Holm con tres pares

Tres modelos producen tres comparaciones pareadas dentro de una configuración. En Holm, el menor p-value se compara primero con $\alpha/3\approx0{,}0167$ cuando $\alpha=0{,}05$.

Por tanto, **con seis pares es imposible superar ese primer umbral** con el Wilcoxon bilateral exacto descrito. Con siete ya es aritméticamente posible:

| Pares útiles R | Menor p-value bilateral exacto | ¿Permite rechazo a 0,05 sin ajuste? | ¿Permite rechazo en el primer paso de Holm, 0,05/3? |
|---|---:|---|---|
| 5 | 0,0625 | No | No |
| 6 | 0,03125 | Sí | No |
| 7 | 0,015625 | Sí | Sí |
| 8 | 0,0078125 | Sí | Sí |

“Puede” indica únicamente que la resolución discreta permite rechazo en un caso extremo. **Siete no es un tamaño recomendado ni el R definitivo del proyecto**: no garantiza rechazo o potencia 0,80, y una familia más amplia cambia el primer umbral. Holm es secuencial: tras el primer paso emplea α/2 y α, si los anteriores rechazan; no impone α/3 a los tres p-values. Con cinco o seis pares ninguno puede superar el primer paso bajo estas condiciones.

La familia actual reúne doce contrastes por semillas; su primer umbral Holm es α/12≈0,004167. El ejemplo de tres pares anterior explica la resolución de Wilcoxon y no describe la multiplicidad efectivamente aplicada. Con diez pares sin ceros ni empates, su mínimo bilateral hipotético sería 0,001953; eso solo permite aritméticamente un rechazo, no garantiza potencia, y Wilcoxon no se ejecutó. Si se amplían ventanas, objetivos u horizontes, la familia deberá reflejarlo.

## 7.2. Cómo abordar estas limitaciones

### 7.2.1. Repeticiones y validación respetando el tiempo

La guía propone considerar validación cruzada repetida con correcciones de varianza, por ejemplo Nadeau–Bengio, cuando el diseño y sus supuestos lo permiten. Esa recomendación **no autoriza aplicar K-fold aleatorio a BTCUSDT** ni convertir un esquema genérico 5 × 5 en una regla automática para series de tiempo.

El calendario se mantiene: **entrenamiento inicial de 2023, validación forward chaining durante 2024, ajuste final con desarrollo 2023–2024 después de cerrar configuraciones y test previsto de enero de 2025 a agosto de 2026**. Septiembre–octubre de 2026 no se incorpora automáticamente al test. Los índices de `split_train_val_forwardChaining` alimentan entrenamiento y escaladores en cada uno de **12 cortes mensuales**. Los orígenes diarios de la función se agrupan en bloques de calendario; no se presentan como 366 reajustes independientes. Dentro de cada mes los pesos quedan fijos y cada pronóstico h=1 utiliza solo OHLCV ya observado.

El entrenamiento de un mes admite únicamente resultados u≤inicio del mes−8, con D=7 y embargo adicional también antes de la validación interna. Los escaladores se estiman de nuevo solo con su entrenamiento. Hay diez semillas de trayectoria, 4101–4110, emparejadas entre variantes; las doce etapas de una trayectoria comparten una historia de mercado dependiente. No son 120 mercados ni una corrección automática de varianza de una CV aleatoria. No se aplica Nadeau–Bengio sin justificar su adecuación a este diseño.

Se ejecutó [bootstrap circular pareado de fechas](../_static/filipe_statistics_20261104/block_bootstrap.csv): 2.000 réplicas, longitud principal de 14 días y sensibilidades 7/28. En cada muestra se recalcula la RMSE por semilla y su media. Remuestrear predicciones almacenadas condiciona a esos ajustes: no remuestrea semillas ni incorpora nuevos entrenamientos. La dependencia y posibles cambios de régimen limitan su calibración; aumentar las filas de resultados no aumenta información independiente.

Los índices de origen móvil no son repeticiones de una evaluación completa. Las semillas pueden estudiar variación del entrenamiento **condicionada al mismo calendario**, pero no crean nuevas trayectorias de mercado. Ni una corrección de varianza ni Holm reparan fuga temporal, selección con la evaluación o p-values mal calibrados. Se respetarán los cortes internos, el embargo y la separación selección/test de §4–§6.

### 7.2.2. Diferencia mínima detectable y tamaños de efecto

La [sensibilidad de diseño](../_static/filipe_statistics_20261104/design_sensitivity.csv) utiliza la dispersión observada entre las diez diferencias pareadas de RMSE. Es una **calibración retrospectiva condicionada al calendario**, no un piloto anterior al experimento ni potencia calculada con el efecto observado para certificar suficiencia.

Para una alternativa hipotética frente a cero y una aproximación normal, se calculó:

$$
\mathrm{MDE}_0=\frac{(z_{1-\alpha_{\mathrm{plan}}/2}+z_{0,80})\,s_d}{\sqrt{R}},
\qquad \alpha_{\mathrm{plan}}=0,05/12,\quad R=10.
$$

| Comparación | DE de diferencias RMSE | MDE frente a cero | δ de planificación, 5 % de persistencia | R orientativo |
|---|---:|---:|---:|---:|
| MLP frente a persistencia | 0,000278 | 0,000326 | 0,000239 | 19 |
| LSTM frente a persistencia | 0,000187 | 0,000219 | 0,000239 | 9 |
| CNN1D frente a persistencia | 0,000337 | 0,000395 | 0,000239 | 28 |

El δ de esta tabla es aproximadamente 0,00023945, referido al RMSE anual de persistencia en el mismo desarrollo. Los R=19/9/28 son estimaciones orientativas a partir de una desviación incierta, no números decididos antes de la ejecución. Que diez supere el R aproximado de LSTM no acredita potencia 0,80: se usó dispersión observada, una aproximación y un calendario compartido. El dimensionamiento prospectivo debe corresponder al procedimiento y alternativa que se evaluarán.

El archivo también conserva las MDE de los tres pares entre redes; sus R para una diferencia de interés quedan no definidos porque no se ha fijado un margen entre redes. El 5 % frente a persistencia no crea ese margen, una banda de equivalencia ni un umbral frente a un modelo avanzado.

Detectar frente a cero un efecto de tamaño δ es distinto de demostrar una ganancia superior al 5 %. Esta última planificación necesita una alternativa de ganancia mayor que el borde; no se inventa su potencia. Tampoco se dimensiona la incertidumbre temporal o el número de mercados aumentando semillas.

Los [contrastes por semillas](../_static/filipe_statistics_20261104/global_seed_tests.csv) reportan diferencias, IC marginales y tamaños d_z. Los [intervalos por fechas](../_static/filipe_statistics_20261104/block_bootstrap.csv) respetan emparejamiento temporal y condicionan a modelos guardados. Los IC no se vuelven simultáneos por ajustar p-valores. Un intervalo amplio no demuestra equivalencia, y una diferencia pequeña no demuestra ausencia de efecto.

### 7.2.3. Probabilidad de superioridad

Puede complementarse la comparación con una probabilidad de superioridad definida explícitamente. Para métricas pareadas en las que menor es mejor, una estimación descriptiva entre las ejecuciones observadas es:

$$
\widehat p_{A<B}=\frac1R\sum_{s=1}^{R}
\left[\mathbb I(m_{A,s}<m_{B,s})+
\frac12\mathbb I(m_{A,s}=m_{B,s})\right].
$$

La [estimación calculada](../_static/filipe_statistics_20261104/probability_superiority.csv) usa diez pares de semillas y RMSE anual; no hubo empates. Los IC del 95 % son binomiales exactos de Clopper–Pearson, bajo el supuesto de pares de aleatorización independientes **condicionados al mismo calendario**.

| Comparación | Semillas con menor RMSE de A | Fracción observada | IC 95 % condicional |
|---|---:|---:|---|
| MLP frente a LSTM | 9/10 | 0,90 | [0,555; 0,997] |
| MLP frente a CNN1D | 6/10 | 0,60 | [0,262; 0,878] |
| MLP frente a Persistencia | 0/10 | 0,00 | [0,000; 0,308] |
| LSTM frente a CNN1D | 5/10 | 0,50 | [0,187; 0,813] |
| LSTM frente a Persistencia | 0/10 | 0,00 | [0,000; 0,308] |
| CNN1D frente a Persistencia | 0/10 | 0,00 | [0,000; 0,308] |

Cero victorias observadas frente a persistencia no es una probabilidad universal cero: su límite superior condicional es aproximadamente 0,308. Tampoco nueve victorias entre diez prueban una ganancia del 5 %, equivalencia o superioridad en otros mercados. Esta proporción no es un p-value ni una probabilidad posterior de que la diferencia media sea favorable. Las unidades semilla, fecha y bloque temporal no se mezclan.

### 7.2.4. Equivalencia requiere un análisis separado

Para sostener equivalencia se utilizará un procedimiento predefinido, como **TOST** con una banda de equivalencia predefinida o un análisis bayesiano con **ROPE**, con sus supuestos y reglas respectivos. El margen de mejora del 5 % frente a persistencia no es automáticamente esa banda de equivalencia. El margen no se elegirá después de ver los resultados y la ausencia de significación no lo sustituye.

En TOST con α=0,05 sin otro ajuste, ambas pruebas unilaterales deben rechazar; la regla usual corresponde al IC del **90 %** completamente dentro de la banda. Se distingue del IC del 95 % de reporte general y se adapta al procedimiento de multiplicidad aplicable. ROPE requiere modelo, priors, posterior y regla de decisión; no es el mismo contraste ni se obtiene contando semillas ganadoras. Los detalles están en §5.4.8.

Se responderán por separado la diferencia frente a cero y la equivalencia práctica. Una diferencia pequeña y precisa puede ser distinguible de cero y estar dentro de una banda de equivalencia. Si no hay evidencia suficiente de ninguna afirmación, el resultado es inconcluso. No se forzará la interpretación a “gana” o “empata”.

## 7.3. Objetivo y criterio de honestidad inferencial

El propósito del entregable es **establecer el baseline y su variabilidad con evidencia**, no obtener $p<0{,}05$ a toda costa. Un resultado no concluyente deberá explicar el efecto estimado, su intervalo, la sensibilidad del diseño y las limitaciones.

La referencia operativa podrá seleccionarse con desarrollo conforme a §5 aunque no se demuestre superioridad frente a todas las alternativas. Se conservarán su identificador y su incertidumbre antes de evaluar test; el ganador observado en test no reabre esa elección. La falta de un ganador estadístico claro no obliga a inventarlo ni impide reportar una referencia justificada.

No se cambiarán retrospectivamente la métrica, los pares, las semillas, los periodos, las reglas de exclusión o las pruebas para conseguir un resultado favorable. Los análisis añadidos después se identificarán como exploratorios y se distinguirán de los confirmatorios.

La representación relativa obtiene R² medios anuales de **0,615670 para MLP, 0,578892 para LSTM y 0,592088 para CNN1D**. Persistencia alcanza **0,759571** y conserva el menor error anual. En julio–diciembre, CNN1D relativa alcanza R²=0,689848 frente a 0,669995 de persistencia, con una reducción puntual de RMSE de aproximadamente **3,10 %**, inferior al margen operativo del 5 %. Esa ventaja semestral no se presenta como confirmación temporal ni generalización al test.

Las configuraciones H/LR/D se eligieron usando desarrollo ya observado, incluido enero–junio de 2024. Aunque cada fold respeta la disponibilidad temporal para ajustar pesos y escalas, el resumen de 2024 es **exploratorio y no una evaluación anidada de toda la selección**. El resultado anual y el subconjunto de 184 fechas de julio–diciembre se identifican por separado. La familia global Holm-12 no distingue los tres pares entre redes al 5 %; las tres redes tienen mayor RMSE que persistencia bajo la inferencia condicional entre semillas. Los DM principales tampoco rechazan al 5 % tras Holm-6. Los IC bootstrap marginales de redes menos persistencia son positivos. DM compara medias de pérdidas cuadradas, el bootstrap compara medias de RMSE por semilla y sus IC son marginales. Estos resultados se reportan con sus estimandos y multiplicidad distintos, sin elegir el más favorable ni interpretarlos como confirmaciones independientes. Un R² mejor no acredita una ganancia financiera.

## 7.4. Entregables y estado documental real

Los siete productos de la página 33 se identifican por separado. “Parcial” o “pendiente” describe la evidencia disponible; no asigna una calificación.

| Producto exigido por la guía | Evidencia actual y estado | Pendiente o límite |
|---|---|---|
| Informe en Jupyter Book | Disponible y versionado; incorpora problema, EDA, literatura, diseño y CV forward chaining en §4–§5 | No certifica cumplimiento académico total ni evaluación final reservada. |
| Código completo —EDA, tres modelos base y análisis estadístico—, ejecutable de extremo a extremo y versionado | [Scripts versionados](https://github.com/jassan-bit/DeepLearning_202630/tree/main/scripts): adquisición, EDA, `prepare_dl_r2_cv.py`, `dl_data_v2.py`, `dl_models.py`, `train_dl_r2_cv.py`, `export_filipe_book.py`, `analyze_filipe_statistics.py` y verificación de artefactos | Cubre desarrollo n7/h1; no ejecuta test ni las sensibilidades n14/21/28. Construir HTML no vuelve a entrenar modelos. |
| Dataset o instrucciones exactas y scripts de obtención según la licencia | Adquisición y hashes documentados; [instrucciones](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md), script y [términos conservados](../_static/data_source/terms_snapshot.json); cohortes de desarrollo registradas | Brutos fuera del checkout; muestras supervisadas de test pendientes. Acceso público no implica uso irrestricto. |
| Tabla de síntesis de literatura con modelos extraídos y diagrama de flujo de búsqueda | [Síntesis](../_static/literature_e1/study_synthesis.csv), [flujo](../_static/literature_e1/flow.svg) y [selección](../_static/literature_e1/selection_evidence.json): cinco textos incluidos y 81 cifras de sus autores | Respaldo aplicado de MLP/CNN1D simple, verificación individual de indexación y extracción definitiva incompletos. Tres documentos grises. Ejecutar las redes no completa el respaldo bibliográfico. |
| Tabla comparativa de los tres modelos con métricas, definición del baseline e inferencia | [CV anual](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), [julio–diciembre](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2_cv_only.csv) y [contrastes globales](../_static/filipe_statistics_20261104/global_seed_tests.csv), [DM](../_static/filipe_statistics_20261104/temporal_dm.csv), [diagnósticos](../_static/filipe_statistics_20261104/diagnostics.csv) y [bootstrap](../_static/filipe_statistics_20261104/block_bootstrap.csv), con periodos y unidades identificados en §5 | MLP relativa es la red de menor RMSE anual observada. CV exploratoria, sin baseline confirmado en test ni margen avanzado cerrado. Sus referencias temporales son aproximadas y el test reservado sigue pendiente. |
| Archivo de dependencias, semillas y configuración de cada corrida | `requirements-training.lock.txt`, entorno separado de CV, [runtime](../_static/dl_r2_cv_20261104/runtime.json), [método de CV](../_static/dl_r2_cv_20261104/method.json), [folds reales](../_static/dl_r2_cv_20261104/folds.json), [corridas](../_static/dl_r2_cv_20261104/runs.csv) y [configuraciones](../_static/dl_r2_cv_20261104/configuration.json) | CPU determinista; costos locales bajo concurrencia, sin garantía de identidad entre plataformas. Protocolo final de test pendiente. |
| Declaración a priori de métrica primaria, margen δ, unidad de independencia y partición | Registro de CV: n7/h1/D7, MSE/RMSE, 5 % solo frente a persistencia, cortes causales y semillas; enmienda posterior de análisis con familia global de doce contrastes | Registro posterior al EDA y al periodo histórico; no es prospectivo. Sensibilidad retrospectiva MDE/R disponible, sin piloto previo o potencia certificada. Márgenes entre redes, equivalencia y avanzado abiertos. |

Como evidencia complementaria, el [bloque C](../_static/eda_temporal_specific/summary.json) conserva 140 relaciones con rezagos sobre 308 fechas comunes. Las [cuatro Ridge](../_static/eda_temporal_reference/summary.json) aportan 366 residuos de validación por historia, 1.464 en total, como diagnóstico de EDA separado del benchmark de redes. Los [12 bloques de Filipe](../_static/dl_r2_cv_20261104/folds.csv) alimentan ajustes mensuales reales con embargo D; no establecen independencia de etiquetas móviles. La [reserva bruta de test](../_static/data/test_raw_reserve_20250101_20260831.json) identifica 20 ZIP desde metadatos sin abrirlos en este ciclo; no congela máscaras supervisadas, actualización o evaluación. Se conserva la exclusión estricta del 24/03/2023 y el calendario.

El registro de noviembre de 2026 antecedió a los ajustes de CV, pero es posterior a los periodos de mercado y al EDA observado. La evaluación reservada futura será un backtest, no predicciones emitidas prospectivamente en 2025–2026. La exposición de 2024 durante el diseño impide tratar la CV como confirmación independiente. La enmienda del complemento se fechó después de los descriptivos y antes de sus cálculos; no añadió entrenamientos ni cambió predicciones. Las decisiones de configuración, ejecución y análisis conservan su cronología.

La disponibilidad del libro no acredita que todos los productos estén completos. Los resultados inexistentes no se reemplazarán por ejemplos, tablas vacías presentadas como mediciones o conclusiones de literatura atribuidas al experimento propio.

## 7.5. Criterios de evaluación del entregable

Los pesos de la rúbrica suministrada son:

| Criterio | Peso |
|---|---:|
| EDA | 30 % |
| Revisión de literatura y extracción de los tres modelos base | 30 % |
| Implementación de los tres modelos base | 20 % |
| Benchmark, definición del baseline y análisis estadístico | 20 % |
| **Total** | **100 %** |

Estos valores son **pesos de evaluación**, no porcentajes de avance ni una calificación obtenida. Haber redactado una sección o superado la construcción HTML no equivale a satisfacer el criterio científico correspondiente.

El 40 % asociado a implementación y benchmark conserva sus requisitos empíricos. El 60 % restante tampoco se declara aprobado por disponer de EDA y texto: la revisión bibliográfica y la extracción definitiva mantienen sus límites. No se calcula una nota ni un porcentaje de cumplimiento a partir de esta matriz.

## 7.6. Buenas prácticas y cierre

La revisión de cada criterio identifica evidencia y límites; **revisado no significa cumplido en su totalidad**.

| Ítem de buenas prácticas | Estado actual | Evidencia verificable | Límite o siguiente requisito |
|---|---|---|---|
| 1. Múltiples ejecuciones y tamaño defendible | Parcial | Diez semillas por red/representación, doce folds reales; [sensibilidad MDE/R](../_static/filipe_statistics_20261104/design_sensitivity.csv) | El mínimo de repeticiones se supera, sin piloto prospectivo ni potencia definitiva acreditada. |
| 2. Media, DE, IC y tamaños de efecto | Ejecutado en desarrollo | [Métricas CV](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), diferencias y d_z por semillas; IC temporales marginales | Alcances condicionales y métodos distintos; no IC simultáneos ni resultados de test. |
| 3. Pruebas, supuestos y multiplicidad | Ejecutado con límites | Holm-12, Friedman, DM/HAC/HLN, Holm-15 para residuos y sensibilidad de bloques | Referencias aproximadas ante dependencia y cambios de régimen; el análisis sigue siendo exploratorio. |
| 4. Consistencia entre modelos | Ejecutado para la comparación declarada | Mismas covariables, objetivos, folds, semillas y reglas de ajuste; dos representaciones controladas | Configuraciones fijas elegidas con desarrollo expuesto; no se acredita una nueva búsqueda anidada óptima. |
| 5. Reproducción de extremo a extremo | Disponible para desarrollo | Código, dependencias, fuentes/hash, 720 checkpoints y 21.960 predicciones; complemento sin reentrenar | La disponibilidad de scripts no acredita un experimento de test que todavía no se ejecutó. |
| 6. Métrica, margen, unidad y protocolo previos | Parcial, cronología explícita | RMSE y margen del 5 % frente a persistencia; CV registrada antes de ajustar y enmienda fechada antes del complemento | El EDA y los descriptivos ya estaban observados; no se afirma un preregistro anterior a todo contacto con datos. |
| 7. Baseline y margen para modelos avanzados | Parcial | MLP relativa es candidata por menor RMSE neuronal anual; persistencia tiene menor error anual | Cierre del pipeline, margen avanzado y equivalencia, ajuste final y evaluación reservada pendientes. |

La literatura aplicada de MLP/CNN1D simple, el ruido externo y el mecanismo de faltantes mantienen sus límites. No se asigna una nota ni un porcentaje de cumplimiento. La contribución de los resultados, sus intervalos y los diagnósticos se comunica sin forzar una red ganadora o prometer una mejora futura.

La confirmación deberá identificar referencia, configuración, periodo y cobertura; desempeño, incertidumbre y costo; y relevancia práctica bajo una evaluación separada. El margen avanzado no se sustituye por el 5 % frente a persistencia. Si un resultado de test se usa para diseñar después una propuesta, hará falta otra evaluación independiente, conforme a §5.5.5.
