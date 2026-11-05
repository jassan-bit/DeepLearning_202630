# 7. Limitaciones Inferenciales de Trabajar con Tres Modelos

Un benchmark con tres modelos y pocas ejecuciones puede tener **potencia muy baja**. El Entregable 1 exige reconocerlo y cuantificarlo, en lugar de interpretar los p-values como si cualquier diseño pudiera detectar diferencias pequeñas.

Este capítulo contrasta las páginas 32–34 de la guía con la CV de Filipe y la evaluación reservada ejecutadas. La comparación principal usa D=7/14/21/28, n=7, h=1 y diez semillas: **120 trayectorias CV, 1.440 segmentos modelo–mes, 2.880 ajustes y 43.920 predicciones**, incluidas treinta trayectorias relativas D7 reutilizadas. La [congelación](../_static/filipe_multihistory_20261104/freeze.json) antecede al ajuste y firma de 120 modelos finales y a la apertura posterior de test. El [método vigente](../_static/filipe_multihistory_20261104/method.json) y sus auditorías distinguen CV, test, complemento inicial D7 y [suplemento inferencial final post-hoc](../_static/e1_audit_improvements_20261104/inference/summary.json). Los límites discretos de Wilcoxon siguientes son teóricos. No se ejecutó ese test en el suplemento inicial D7 ni en el piloto nuevo; la [sensibilidad Wilcoxon del test final](../_static/e1_audit_improvements_20261104/inference/wilcoxon_sensitivity.csv) se conserva como análisis posterior separado.

## 7.1. Potencia, tamaño de muestra y multiplicidad

### 7.1.1. Friedman con tres modelos y cinco bloques

Con tres modelos y cinco bloques, Friedman puede tener poca potencia y el post-hoc de Nemenyi ser conservador. La ausencia de significación no demuestra equivalencia ni descarta diferencias de importancia práctica.

Esa advertencia no es una estimación numérica de potencia ni significa que sea imposible detectar cualquier efecto. La potencia depende también del efecto alternativo, la dispersión, los empates, la unidad de bloque y la calibración. Con pocos bloques se identificará la referencia exacta, de permutación o aproximada utilizada y se justificarán sus condiciones, como exige §5.4.3.

Los bloques deben representar unidades defendibles del experimento. Las 366 fechas de validación, las semillas y las configuraciones no se mezclarán como si fueran réplicas independientes de la misma naturaleza. Se conservarán el emparejamiento, el alcance de la inferencia y la dependencia temporal descritos en §5.4.

Un bloque completo contiene los resultados de la misma unidad bajo las alternativas comparadas. Nemenyi controla simultáneamente su familia; no se añade automáticamente Holm como una segunda corrección equivalente. El análisis del ciclo inicial D7 utilizó diez bloques-semilla para las tres redes relativas. Friedman dio χ²=4,2, p=0,122456 bajo referencia asintótica aproximada. Las comparaciones pareadas registradas no quedaron condicionadas a rechazar ese omnibus: la familia global contiene doce hipótesis por semillas, con Holm-12. No se ejecutó Nemenyi ni se aplica Holm como una segunda corrección de ese post-hoc.

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

La familia del complemento inicial D7 reúne doce contrastes por semillas; su primer umbral Holm es α/12≈0,004167. El ejemplo de tres pares anterior explica la resolución de Wilcoxon y no describe la multiplicidad efectivamente aplicada. Con diez pares sin ceros ni empates, su mínimo bilateral hipotético sería 0,001953; eso solo permite aritméticamente un rechazo, no garantiza potencia, y Wilcoxon no se ejecutó. La ampliación de historias conserva ese contraste histórico y declara por separado sus familias BDS: 2.880 hipótesis de CV y 4.800 de test; no se reaplica Holm-12 como si cubriera todas las nuevas preguntas. El suplemento posterior del test declara otras familias: seis pares por semillas, seis DM principales y nueve Ljung–Box. Cada familia conserva su objetivo y unidad; no se combinan sus p-values como corroboraciones independientes.

El nuevo piloto de §4.1 planifica otra familia de doce pares neuronales, tres por D. Sus cinco semillas estiman dispersión y no se utilizan para escoger una prueba favorable. Para un futuro Wilcoxon bilateral exacto de esa familia, sin ceros ni empates, **nueve pares** son el mínimo que permite aritméticamente superar α/12: con ocho el mínimo es 0,0078125 y con nueve 0,00390625. Esto no demuestra potencia suficiente; el dimensionamiento principal registrado usa la referencia t no central y declara sus supuestos.

## 7.2. Cómo abordar estas limitaciones

### 7.2.1. Repeticiones y validación respetando el tiempo

La guía propone considerar validación cruzada repetida con correcciones de varianza, por ejemplo Nadeau–Bengio, cuando el diseño y sus supuestos lo permiten. Esa recomendación **no autoriza aplicar K-fold aleatorio a BTCUSDT** ni convertir un esquema genérico 5 × 5 en una regla automática para series de tiempo.

El calendario se mantiene: **2023 como historia inicial, forward chaining 2024, ajuste final con desarrollo permitido de 2023–2024 y test enero de 2025–agosto de 2026**. Septiembre–octubre de 2026 queda fuera. Los doce cortes mensuales de CV usan índices reales de `split_train_val_forwardChaining`; dentro del mes los pesos quedan fijos y cada pronóstico h=1 usa OHLCV hasta d−1. El test usa los pesos y escaladores finales ya congelados, sin reajustes; sus veinte meses son bloques de reporte, no folds de selección.

El entrenamiento mensual admite u≤inicio del mes−D−1 para D=7/14/21/28, con embargo propio antes de la validación interna. Cada escalador se aprende solo con su entrenamiento. Hay diez semillas 4101–4110 por red e historia, condicionadas a una misma trayectoria de mercado; las 120 combinaciones no son 120 mercados. No se aplica Nadeau–Bengio automáticamente a esta CV temporal.

En el ciclo inicial D7 se ejecutó [bootstrap circular pareado de fechas](../_static/filipe_statistics_20261104/block_bootstrap.csv): 2.000 réplicas, longitud principal de 14 días y sensibilidades 7/28. En cada muestra se recalcula la RMSE por semilla y su media. Remuestrear predicciones almacenadas condiciona a esos ajustes: no remuestrea semillas ni incorpora nuevos entrenamientos. La dependencia y posibles cambios de régimen limitan su calibración; aumentar las filas de resultados no aumenta información independiente.

Los índices de origen móvil no son repeticiones de una evaluación completa. Las semillas pueden estudiar variación del entrenamiento **condicionada al mismo calendario**, pero no crean nuevas trayectorias de mercado. Ni una corrección de varianza ni Holm reparan fuga temporal, selección con la evaluación o p-values mal calibrados. Se respetarán los cortes internos, el embargo y la separación selección/test de §4–§6.

### 7.2.2. Diferencia mínima detectable y tamaños de efecto

El [nuevo piloto entre baselines](../_static/e1_filipe_pilot_20261105/protocol.json), detallado en §4.1, registra cinco semillas nuevas por familia e historia, conserva la receta de Filipe y estima la dispersión de doce pares de RMSE anual. Se adopta antes de sus corridas δ=0,00023944847859645605 y α=0,05/12. La [planificación](../_static/e1_filipe_pilot_20261105/analysis/paired_dimensioning.csv) reporta R puntual y conservador con límite superior marginal de DE, y MDE con cinco pares y con el R común. Es prospectivo respecto de esas inicializaciones, condicionado a 2024 ya conocido; no acredita normalidad, potencia alcanzada o generalización a otro periodo. Las cinco semillas del piloto quedan fuera de la futura fase definitiva.

El máximo común es **302 semillas con DE puntual y 1.680 con el límite superior de DE**, por modelo e historia de una fase futura, determinado por MLP–LSTM/D21. El R grande cuantifica el costo de detectar el margen fijado con multiplicidad y dispersión incierta; no se cambia el margen después de verlo. No se ejecutó esa fase definitiva. Las cinco semillas piloto no se incorporan adaptativamente a sus contrastes.

La [sensibilidad de diseño inicial D7](../_static/filipe_statistics_20261104/design_sensitivity.csv) utiliza la dispersión observada entre las diez diferencias pareadas de RMSE. Es una **calibración retrospectiva condicionada al calendario**, no un piloto anterior al experimento ni potencia calculada con el efecto observado para certificar suficiencia.

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

Ese archivo histórico también conserva las MDE de tres pares entre redes; allí R queda no definido porque aquel protocolo no fijaba un margen entre redes. El nuevo piloto adopta expresamente ahora un margen absoluto para doce pares, sin reescribir ese antecedente. El 5 % frente a persistencia no crea por sí solo una banda de equivalencia ni la frontera de un modelo avanzado.

Detectar frente a cero un efecto de tamaño δ es distinto de demostrar una ganancia superior al 5 %. Esta última planificación necesita una alternativa de ganancia mayor que el borde; no se inventa su potencia. Tampoco se dimensiona la incertidumbre temporal o el número de mercados aumentando semillas.

Los [contrastes por semillas](../_static/filipe_statistics_20261104/global_seed_tests.csv) reportan diferencias, IC marginales y tamaños d_z. Los [intervalos por fechas](../_static/filipe_statistics_20261104/block_bootstrap.csv) respetan emparejamiento temporal y condicionan a modelos guardados. Los IC no se vuelven simultáneos por ajustar p-valores. Un intervalo amplio no demuestra equivalencia, y una diferencia pequeña no demuestra ausencia de efecto.

La planificación de una futura propuesta conserva [36 escenarios de sensibilidad](../_static/e1_audit_improvements_20261104/planning/advanced_margin_scenarios.csv) de 1/2,5/5/10 % frente al ancla MLP/D7 de CV: estiman el R para detectar δ frente a cero bajo dispersiones y correlaciones hipotéticas, no para superar una frontera práctica. El {download}`alcance original de esos escenarios <../_static/e1_audit_improvements_20261104/planning/README.md>` mantiene esa distinción.

El **criterio operativo del benchmark académico** queda definido para el futuro: reducción de RMSE **superior al 5 % frente a MLP/D7 y frente a persistencia**, considerando incertidumbre y ambas comparaciones. No se presenta como margen previo al test actual ni como utilidad financiera. Los [27 escenarios de frontera práctica](../_static/e1_audit_improvements_20261104/planning/academic_joint_power_scenarios.csv) emplean un mismo promedio avanzado hipotético 7,5/10/15 % menor que persistencia, razones σ_A/σ_B=0,5/1/2 y correlaciones 0/0,5/0,9. Se calcula R por t no central unilateral con α=0,025 y potencia 0,90 para cada referencia, usando al menos cinco ejecuciones. Bajo esos supuestos, la cota de la unión limita a 0,20 la probabilidad de fallar alguna comparación; no presupone independencia entre ellas. La [síntesis académica](../_static/e1_audit_improvements_20261104/planning/academic_planning_summary.json) no acredita potencia alcanzada: faltan dispersión real, piloto prospectivo y una evaluación nueva.

### 7.2.3. Probabilidad de superioridad

Puede complementarse la comparación con una probabilidad de superioridad definida explícitamente. Para métricas pareadas en las que menor es mejor, una estimación descriptiva entre las ejecuciones observadas es:

$$
\widehat p_{A<B}=\frac1R\sum_{s=1}^{R}
\left[\mathbb I(m_{A,s}<m_{B,s})+
\frac12\mathbb I(m_{A,s}=m_{B,s})\right].
$$

La [estimación calculada para D7](../_static/filipe_statistics_20261104/probability_superiority.csv) usa diez pares de semillas y RMSE anual; no hubo empates. Los IC del 95 % son binomiales exactos de Clopper–Pearson, bajo el supuesto de pares de aleatorización independientes **condicionados al mismo calendario**.

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


El propósito es **establecer un baseline y su variabilidad con evidencia**, no obtener p<0,05 a toda costa. La familia neuronal y su historia se eligieron con la RMSE media de CV antes de examinar test: **MLP con D=7**. La elección y las doce combinaciones que se reportarían están en la [congelación](../_static/filipe_multihistory_20261104/freeze.json). El orden observado en test no reabre esa decisión.

Las tablas de §5 distinguen desarrollo expuesto de test, meses de reajuste de meses de reporte y dispersión entre semillas de dependencia temporal. Los IC t entre semillas condicionan al calendario: no son intervalos de generalización a mercados nuevos. El cumplimiento puntual del 5 % frente a persistencia y demostrarlo estadísticamente son afirmaciones diferentes. Los nuevos contrastes de RMSE del test se registraron y calcularon como suplemento post-hoc sobre los pronósticos guardados; no se les atribuye un registro anterior al test ni se reutilizan p-valores del D7 de desarrollo para otras historias.

El BDS mensual de h=1 cubre las cuatro historias, tres redes y diez semillas. Se conservan sus valores sin ajustar, Holm por fase y estados no definidos; media y DE de p-valores son resúmenes descriptivos. Muestras mensuales cortas, solapamiento del objetivo y deriva limitan las referencias asintóticas. Rechazar i.i.d. no demuestra exclusivamente no linealidad explotable ni cuantifica una mejora futura.

El ciclo inicial D7 y su complemento Holm-12, Friedman, DM/HAC/HLN y bootstrap mantienen alcance exploratorio propio. Las configuraciones usaron 2024 previamente expuesto; respetar causalidad dentro de los folds no convierte toda la selección en validación anidada. Evaluar el periodo reservado después de congelar modelos fortalece la separación de etapas, pero no acredita potencia 0,80, utilidad financiera o generalización a otros activos. Si se utiliza este test para diseñar una propuesta posterior, será necesario reservar otra evaluación independiente.

## 7.4. Entregables y estado documental real

La [sección 8](08_entregables.md) desarrolla los siete productos, con enlaces directos, catálogos descargables y una declaración cronológica. La tabla siguiente conserva el resumen de su estado y límites.

Los siete productos de la página 33 se identifican por separado. Ejecutar las tres redes y el test no completa los límites bibliográficos, de potencia o de definición de márgenes.

| Producto exigido | Evidencia actual | Límite restante |
|---|---|---|
| Informe en Jupyter Book | Problema, EDA, literatura, CV de cuatro historias, cierre y evaluación final; numeración preservada | No es una calificación ni acredita cumplimiento académico total. |
| Código completo ejecutable y versionado | Preparación, entrenamiento, análisis, cierre, evaluación y réplica privada aislada; [manual actualizado](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) | La réplica restaura y verifica evidencia guardada; no se afirma haber repetido todos los entrenamientos en otra máquina. Construir HTML no vuelve a entrenar. |
| Dataset o instrucciones exactas | [Adquisición](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md), hashes, derivados y manifiesto común de test; [términos](../_static/data_source/terms_snapshot.json) | Brutos fuera del checkout; acceso público no equivale a uso irrestricto. |
| Síntesis y flujo de literatura | Seis textos y 36 referencias; Miura et al. (2019) acredita comparadores MLP/CNN independientes y conserva sus cifras publicadas | La falta de fuente aplicada para esas familias queda resuelta; recurrencia de una fuente en seis, dimensionalidad CNN no reportada, recetas propias no replicadas e indexación individual sin atribución automática. |
| Comparación, baseline e inferencia | CV y test por D/modelo/semilla; BDS por bloque; [selección previa](../_static/filipe_multihistory_20261104/freeze.json); suplemento final con pares/DM/LB/bootstrap | Inferencia final post-hoc; un mercado, etiquetas solapadas, BDS mensual aproximado; criterio académico futuro definido, pero potencia real y nueva evaluación pendientes. Equivalencia condicional. |
| Dependencias, semillas y configuraciones | Versiones, diez semillas comunes y manifiestos públicos; índices completos, escaladores, pesos y curvas conservados localmente; {download}`alcance de los resúmenes publicados <../_static/filipe_multihistory_20261104/README.md>` | Determinismo documentado no garantiza identidad numérica entre plataformas. Los paquetes completos no se ofrecen como descarga pública. |
| Métrica, margen, unidad y partición previos | Registro histórico antes de test; nuevo piloto registra sus semillas, doce pares, margen y análisis antes de sus corridas | No es previo al EDA o al benchmark histórico; piloto condicionado a 2024 conocido y sin potencia alcanzada. |

La evidencia complementaria conserva las cuatro Ridge diagnósticas de EDA y sus 1.464 predicciones; no son otro competidor DL. Los bloques de Filipe alimentan ajustes reales, sin establecer independencia de etiquetas. La [auditoría de test](../_static/filipe_multihistory_20261104/test/raw_test_audit.json) y la [cohorte](../_static/filipe_multihistory_20261104/test/cohort_manifest.json) acreditan 608 fechas comunes y 72960 predicciones después de firmar todos los modelos finales.

El registro de noviembre de 2026 es posterior a los periodos históricos; el test es un backtest con acceso a resultados separado de la selección de desarrollo, no predicciones emitidas en vivo. La existencia de resultados finales no elimina ruido externo desconocido, dependencia, límites de cobertura de literatura o incertidumbre del dimensionamiento.

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


La revisión identifica evidencia y límites; **revisado no significa cumplido en su totalidad**.

| Ítem | Estado actual | Evidencia | Límite |
|---|---|---|---|
| 1. Repeticiones y tamaño defendible | Nuevo piloto registrado y planificación condicional | Cinco semillas nuevas por red/historia, doce folds y doce pares para R/MDE en §4.1; diez semillas históricas conservadas aparte | Ejecución definitiva futura con semillas nuevas; no potencia lograda ni otra trayectoria de mercado. |
| 2. Media, DE, IC y efectos | Ejecutado con alcance condicional | Tablas anuales/globales, seis pares finales con d_z/IC y bootstrap temporal; antecedente D7 separado | No son intervalos de nuevos mercados ni confirmación prospectiva. |
| 3. Pruebas y multiplicidad | Ejecutado con límites | BDS por bloque, pares finales Holm-6, DM-6, LB-9 y suplemento inicial separado | Estadística final post-hoc; muestras BDS cortas y promedio p exclusivamente descriptivo. |
| 4. Consistencia | Ejecutada en las cuatro historias | Mismas covariables, n7/h1, fechas comunes, semillas y reglas causales dentro de cada fase | Cambiar D cambia capacidad y embargo; desarrollo previamente expuesto. |
| 5. Reproducción | Ruta de entrega y verificación aislada validada | Manual vigente, paquete privado con hashes y auditoría de una copia restaurada | Se verifica el backtest guardado sin borrar su marcador ni repetir pronósticos; pesos/vectores completos no se publican. |
| 6. Protocolo anterior a test | Cerrado antes de abrir test | Congelación, épocas, selección, calidad y firma de 120 modelos antes de acceso | No es un registro anterior al EDA o al periodo de mercado. |
| 7. Baseline y margen avanzado | Baseline cerrado y criterio académico futuro definido | **MLP/D7** seleccionado con CV; futura mejora >5 % frente a MLP/D7 y persistencia, con sensibilidad y planificación conjunta | Definición posterior al test actual; faltan piloto/potencia reales y otra evaluación. La equivalencia sigue siendo condicional. |

La literatura aplicada incluye una fuente primaria para MLP y CNN independientes, con recurrencia de una en seis y dimensión CNN no reportada; las recetas son adaptaciones. El nuevo piloto registra su diseño antes de las corridas y dimensiona futuras semillas sobre el calendario conocido; no modifica la selección histórica ni certifica potencia lograda. Persisten la ejecución definitiva nueva, el piloto de una propuesta avanzada, la evaluación independiente y los límites de ruido/faltantes. No se asigna una nota o porcentaje de cumplimiento ni se fuerza superioridad universal.

La evaluación final identifica selección previa, configuración, periodo y cobertura. Si los resultados se usan para diseñar después otro modelo, se necesitará una evaluación nueva independiente, conforme a §5.5.5.
