# 7. Limitaciones Inferenciales de Trabajar con Tres Modelos

Un benchmark con tres modelos y pocas ejecuciones puede tener **potencia muy baja**. El Entregable 1 exige reconocerlo y cuantificarlo, en lugar de interpretar los p-values como si cualquier diseño pudiera detectar diferencias pequeñas.

Este capítulo contrasta las páginas 32–34 de la guía con la validación forward chaining ejecutada mediante `timeseries-cv`, proyecto de DidierRLopes con coautoría de Filipe R. Ramos. El [método de CV](../_static/dl_r2_cv_20261104/method.json), los [folds](../_static/dl_r2_cv_20261104/folds.json) y el [resumen de ejecución](../_static/dl_r2_cv_20261104/execution_summary.json) acreditan **60 trayectorias, 720 segmentos modelo–mes y 1.440 ajustes efectivos**, con diez semillas por red y variante y 366 predicciones diarias por trayectoria. Los límites discretos de Wilcoxon que siguen son resultados teóricos; el test reservado no se evaluó.

## 7.1. Potencia, tamaño de muestra y multiplicidad

### 7.1.1. Friedman con tres modelos y cinco bloques

Con tres modelos y cinco bloques, Friedman puede tener poca potencia y el post-hoc de Nemenyi ser conservador. La ausencia de significación no demuestra equivalencia ni descarta diferencias de importancia práctica.

Esa advertencia no es una estimación numérica de potencia ni significa que sea imposible detectar cualquier efecto. La potencia depende también del efecto alternativo, la dispersión, los empates, la unidad de bloque y la calibración. Con pocos bloques se identificará la referencia exacta, de permutación o aproximada utilizada y se justificarán sus condiciones, como exige §5.4.3.

Los bloques deben representar unidades defendibles del experimento. Las 366 fechas de validación, las semillas y las configuraciones no se mezclarán como si fueran réplicas independientes de la misma naturaleza. Se conservarán el emparejamiento, el alcance de la inferencia y la dependencia temporal descritos en §5.4.

Un bloque completo contiene los resultados de la misma unidad bajo las alternativas comparadas. Nemenyi controla simultáneamente su familia; no se añade automáticamente Holm como una segunda corrección equivalente. El análisis ejecutado contiene **tres comparaciones pareadas de RMSE entre escalado absoluto y relativo**, una por red, con diez semillas emparejadas y Holm sobre las tres. Friedman y Nemenyi no se han ejecutado para estas predicciones de CV; sus condiciones se describen como métodos de análisis posibles.

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

Si se realizan afirmaciones sobre varias ventanas, objetivos u horizontes, la familia de comparaciones debe reflejarlo. No se ocultará esa ampliación tratando cada tabla como si fuera la única explorada.

## 7.2. Cómo abordar estas limitaciones

### 7.2.1. Repeticiones y validación respetando el tiempo

La guía propone considerar validación cruzada repetida con correcciones de varianza, por ejemplo Nadeau–Bengio, cuando el diseño y sus supuestos lo permiten. Esa recomendación **no autoriza aplicar K-fold aleatorio a BTCUSDT** ni convertir un esquema genérico 5 × 5 en una regla automática para series de tiempo.

El calendario se mantiene: **entrenamiento inicial de 2023, validación forward chaining durante 2024, ajuste final con desarrollo 2023–2024 después de cerrar configuraciones y test previsto de enero de 2025 a agosto de 2026**. Septiembre–octubre de 2026 no se incorpora automáticamente al test. Los índices de `split_train_val_forwardChaining` alimentan entrenamiento y escaladores en cada uno de **12 cortes mensuales**. Los orígenes diarios de la función se agrupan en bloques de calendario; no se presentan como 366 reajustes independientes. Dentro de cada mes los pesos quedan fijos y cada pronóstico h=1 utiliza solo OHLCV ya observado.

El entrenamiento de un mes admite únicamente resultados u≤inicio del mes−8, con D=7 y embargo adicional también antes de la validación interna. Los escaladores se estiman de nuevo solo con su entrenamiento. Hay diez semillas de trayectoria, 4101–4110, emparejadas entre variantes; las doce etapas de una trayectoria comparten una historia de mercado dependiente. No son 120 mercados ni una corrección automática de varianza de una CV aleatoria. No se aplica Nadeau–Bengio sin justificar su adecuación a este diseño.

El remuestreo temporal por bloques puede complementar el análisis cuando se justifiquen unidad, longitud y supuestos. No se ha ejecutado bootstrap temporal sobre las predicciones actuales de CV. Remuestrear predicciones almacenadas describiría incertidumbre condicionada a esos ajustes, sin incorporar automáticamente la variación de volver a entrenar. Aumentar las filas de resultados no aumenta por sí solo información independiente.

Los índices de origen móvil no son repeticiones de una evaluación completa. Las semillas pueden estudiar variación del entrenamiento **condicionada al mismo calendario**, pero no crean nuevas trayectorias de mercado. Ni una corrección de varianza ni Holm reparan fuga temporal, selección con la evaluación o p-values mal calibrados. Se respetarán los cortes internos, el embargo y la separación selección/test de §4–§6.

### 7.2.2. Diferencia mínima detectable y tamaños de efecto

Se reportará la **MDE con el número de ejecuciones disponible**, su relación con el margen del contraste correspondiente y los supuestos del dimensionamiento de §4.1. El margen aceptado del 5 % frente a persistencia no establece por sí solo un margen entre dos redes. Si el estudio no puede distinguir una diferencia de interés práctico, se ampliará el diseño o se explicitará esa limitación.

La MDE respecto de cero y la potencia para superar el umbral práctico son preguntas distintas, como precisa §4.1.5–§4.1.7. El número de semillas dimensiona variación del entrenamiento condicionada al calendario; no dimensiona por sí solo la incertidumbre temporal ni el número de días de test.

Las comparaciones incluirán tamaños de efecto e intervalos de confianza, conservando el emparejamiento. Los intervalos temporales deberán respetar bloques cuando proceda; no se remuestrearán días dependientes como si fueran i.i.d.

Se priorizará la diferencia en unidades de la métrica, con dirección y denominador explícitos si se expresa de forma relativa. Cualquier efecto estandarizado identificará su fórmula y unidad. El ciclo CV utiliza **diez semillas por modelo y variante** como presupuesto exploratorio: no dispone de un piloto propio ni de dimensionamiento que acredite potencia 0,80 para la mejora mínima de interés. La MDE del procedimiento mensual queda pendiente; no se calcula potencia retrospectiva con el efecto observado para certificar suficiencia.

Las doce particiones temporales no convierten diez semillas en 120 repeticiones independientes. Un dimensionamiento futuro deberá corresponder a las configuraciones, cohortes, política de actualización, estimando y multiplicidad que realmente se evaluarán.

No se interpretará un intervalo amplio como equivalencia ni una diferencia media pequeña como ausencia de efecto. La precisión de la estimación forma parte del resultado y debe figurar junto a su valor puntual.

### 7.2.3. Probabilidad de superioridad

Puede complementarse la comparación con una probabilidad de superioridad definida explícitamente. Para métricas pareadas en las que menor es mejor, una estimación descriptiva entre las ejecuciones observadas es:

$$
\widehat p_{A<B}=\frac1R\sum_{s=1}^{R}
\left[\mathbb I(m_{A,s}<m_{B,s})+
\frac12\mathbb I(m_{A,s}=m_{B,s})\right].
$$

Se reportarán unidad, tratamiento de empates, tamaño e incertidumbre. Esta proporción describe las repeticiones consideradas y no se presentará como probabilidad universal de ganar en cualquier mercado o periodo. Si se estima sobre fechas o bloques, se cambiará y justificará la unidad, sin mezclarla con semillas.

No es un p-value ni una probabilidad posterior de que la diferencia media sea favorable. Una proporción alta tampoco demuestra una ganancia superior al 5 %: debe acompañarse de su magnitud y su incertidumbre. Con pocas repeticiones tiene resolución limitada. Para una métrica donde mayor sea mejor, se invierte la desigualdad y se registra esa dirección antes de comparar.

### 7.2.4. Equivalencia requiere un análisis separado

Para sostener equivalencia se utilizará un procedimiento predefinido, como **TOST** con una banda de equivalencia predefinida o un análisis bayesiano con **ROPE**, con sus supuestos y reglas respectivos. El margen de mejora del 5 % frente a persistencia no es automáticamente esa banda de equivalencia. El margen no se elegirá después de ver los resultados y la ausencia de significación no lo sustituye.

En TOST con α=0,05 sin otro ajuste, ambas pruebas unilaterales deben rechazar; la regla usual corresponde al IC del **90 %** completamente dentro de la banda. Se distingue del IC del 95 % de reporte general y se adapta al procedimiento de multiplicidad aplicable. ROPE requiere modelo, priors, posterior y regla de decisión; no es el mismo contraste ni se obtiene contando semillas ganadoras. Los detalles están en §5.4.8.

Se responderán por separado la diferencia frente a cero y la equivalencia práctica. Una diferencia pequeña y precisa puede ser distinguible de cero y estar dentro de una banda de equivalencia. Si no hay evidencia suficiente de ninguna afirmación, el resultado es inconcluso. No se forzará la interpretación a “gana” o “empata”.

## 7.3. Objetivo y criterio de honestidad inferencial

El propósito del entregable es **establecer el baseline y su variabilidad con evidencia**, no obtener $p<0{,}05$ a toda costa. Un resultado no concluyente deberá explicar el efecto estimado, su intervalo, la sensibilidad del diseño y las limitaciones.

La referencia operativa podrá seleccionarse con desarrollo conforme a §5 aunque no se demuestre superioridad frente a todas las alternativas. Se conservarán su identificador y su incertidumbre antes de evaluar test; el ganador observado en test no reabre esa elección. La falta de un ganador estadístico claro no obliga a inventarlo ni impide reportar una referencia justificada.

No se cambiarán retrospectivamente la métrica, los pares, las semillas, los periodos, las reglas de exclusión o las pruebas para conseguir un resultado favorable. Los análisis añadidos después se identificarán como exploratorios y se distinguirán de los confirmatorios.

La representación relativa obtiene R² medios anuales de **0,615670 para MLP, 0,578892 para LSTM y 0,592088 para CNN1D**. Persistencia alcanza **0,759571** y conserva el menor error anual. En julio–diciembre, CNN1D relativa alcanza R²=0,689848 frente a 0,669995 de persistencia, con una reducción puntual de RMSE de aproximadamente **3,10 %**, inferior al margen operativo del 5 %. Esa ventaja semestral no se presenta como confirmación temporal ni generalización al test.

Las configuraciones H/LR/D se eligieron usando desarrollo ya observado, incluido enero–junio de 2024. Aunque cada fold respeta la disponibilidad temporal para ajustar pesos y escalas, el resumen de 2024 es **exploratorio y no una evaluación anidada de toda la selección**. El resultado anual y el subconjunto de 184 fechas de julio–diciembre se identifican por separado. Las únicas pruebas ejecutadas comparan ambas representaciones dentro de cada red; no prueban superioridad entre redes o frente a persistencia. Un R² mejor no acredita por sí solo una ganancia financiera.

## 7.4. Entregables y estado documental real

Los siete productos de la página 33 se identifican por separado. “Parcial” o “pendiente” describe la evidencia disponible; no asigna una calificación.

| Producto exigido por la guía | Evidencia actual y estado | Pendiente o límite |
|---|---|---|
| Informe en Jupyter Book | Disponible y versionado; incorpora problema, EDA, literatura, diseño y CV forward chaining en §4–§5 | No certifica cumplimiento académico total ni evaluación final reservada. |
| Código completo —EDA, tres modelos base y análisis estadístico—, ejecutable de extremo a extremo y versionado | [Scripts versionados](https://github.com/jassan-bit/DeepLearning_202630/tree/main/scripts): adquisición, EDA, `prepare_dl_r2_cv.py`, `dl_data_v2.py`, `dl_models.py`, `train_dl_r2_cv.py`, `analyze_dl_r2_v2.py` y verificación de artefactos | Cubre desarrollo n7/h1; no ejecuta test ni las sensibilidades n14/21/28. Construir HTML no vuelve a entrenar modelos. |
| Dataset o instrucciones exactas y scripts de obtención según la licencia | Adquisición y hashes documentados; [instrucciones](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md), script y [términos conservados](../_static/data_source/terms_snapshot.json); cohortes de desarrollo registradas | Brutos fuera del checkout; muestras supervisadas de test pendientes. Acceso público no implica uso irrestricto. |
| Tabla de síntesis de literatura con modelos extraídos y diagrama de flujo de búsqueda | [Síntesis](../_static/literature_e1/study_synthesis.csv), [flujo](../_static/literature_e1/flow.svg) y [selección](../_static/literature_e1/selection_evidence.json): cinco textos incluidos y 81 cifras de sus autores | Respaldo aplicado de MLP/CNN1D simple, verificación individual de indexación y extracción definitiva incompletos. Tres documentos grises. Ejecutar las redes no completa el respaldo bibliográfico. |
| Tabla comparativa de los tres modelos con métricas, definición del baseline e inferencia | [CV anual](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), [julio–diciembre](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2_cv_only.csv) y [comparación controlada de escalado](../_static/dl_r2_cv_20261104/analysis/comparisons.csv), con periodos y unidades identificados en §5 | MLP relativa es la red de menor RMSE anual observada. CV exploratoria, sin baseline confirmado en test ni margen avanzado cerrado. DM, BDS, Ljung–Box y bootstrap temporal siguen pendientes para estas predicciones. |
| Archivo de dependencias, semillas y configuración de cada corrida | `requirements-training.lock.txt`, entorno separado de CV, [runtime](../_static/dl_r2_cv_20261104/runtime.json), [método de CV](../_static/dl_r2_cv_20261104/method.json), [folds reales](../_static/dl_r2_cv_20261104/folds.json), [corridas](../_static/dl_r2_cv_20261104/runs.csv) y [configuraciones](../_static/dl_r2_cv_20261104/configuration.json) | CPU determinista; costos locales bajo concurrencia, sin garantía de identidad entre plataformas. Protocolo final de test pendiente. |
| Declaración a priori de métrica primaria, margen δ, unidad de independencia y partición | Registro previo a los ajustes CV: n7/h1/D7, MSE/RMSE, 5 % solo frente a persistencia, cortes causales, semillas y familia de tres contrastes | Registro posterior al EDA y al periodo histórico; no es prospectivo. Diez semillas sin piloto ni dimensionamiento propios. Márgenes entre redes, equivalencia y avanzado abiertos. |

Como evidencia complementaria, el [bloque C](../_static/eda_temporal_specific/summary.json) conserva 140 relaciones con rezagos sobre 308 fechas comunes. Las [cuatro Ridge](../_static/eda_temporal_reference/summary.json) aportan 366 residuos de validación por historia, 1.464 en total, como diagnóstico de EDA separado del benchmark de redes. Los [12 bloques de Filipe](../_static/dl_r2_cv_20261104/folds.csv) alimentan ajustes mensuales reales con embargo D; no establecen independencia de etiquetas móviles. La [reserva bruta de test](../_static/data/test_raw_reserve_20250101_20260831.json) identifica 20 ZIP desde metadatos sin abrirlos en este ciclo; no congela máscaras supervisadas, actualización o evaluación. Se conserva la exclusión estricta del 24/03/2023 y el calendario.

El registro de noviembre de 2026 antecedió a los ajustes de CV, pero es posterior a los periodos de mercado y al EDA observado. La evaluación reservada futura será un backtest, no predicciones emitidas prospectivamente en 2025–2026. La exposición de 2024 durante el diseño impide tratar la CV como confirmación independiente. Las enmiendas se fechan y separan de las decisiones anteriores a cada análisis. La aclaración documental de la cohorte no cambió índices, pesos ni optimización.

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

El cierre experimental que describe la guía debe comprobar que:

1. Se realizaron múltiples ejecuciones, con el mínimo aplicable y el número que exija la potencia.
2. Se reportaron media, desviación muestral, IC del 95 % y tamaños de efecto con unidades identificadas.
3. Se justificaron las pruebas, sus supuestos, correcciones y limitaciones de independencia.
4. Se mantuvo consistencia experimental entre los tres modelos: información disponible, particiones, semillas, reglas comunes y esfuerzo comparable de selección; las adaptaciones por arquitectura quedaron justificadas.
5. El código, las dependencias, los datos o sus instrucciones, y las configuraciones permiten reproducir el procedimiento completo.
6. La métrica principal, $\delta$, la unidad inferencial y el protocolo quedaron fijados antes de la evaluación pertinente.
7. El baseline y el margen exigible a modelos avanzados se concluyeron desde resultados verificables, distinguiendo significación y relevancia práctica.

El análisis esperado es exhaustivo y reconoce la potencia limitada de comparar solo tres modelos. La revisión documental, el EDA y la CV ejecutada se evalúan mediante sus artefactos. La evidencia no permite afirmar cumplimiento total del PDF original: persisten el respaldo aplicado incompleto, la falta de confirmación reservada y un dimensionamiento de potencia suficiente para el procedimiento pertinente.

La conclusión futura deberá identificar referencia, configuración, periodo y cobertura; desempeño, incertidumbre y costo; evidencia de diferencia o de equivalencia cuando proceda; y margen práctico para una propuesta avanzada. Ese margen avanzado sigue abierto y no se sustituye por el 5 % aceptado frente a persistencia. Si el test se usa para diseñar después una nueva propuesta, su confirmación requiere una evaluación independiente no utilizada en ese diseño, conforme a §5.5.5. Por ahora existe una comparación DL de desarrollo con CV real y un ranking anual medido, sin evaluación final reservada. No se cambia retrospectivamente el protocolo para producir una red ganadora.