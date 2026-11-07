# 7. Limitaciones Inferenciales de Trabajar con Tres Modelos

**Dos fases del estudio.** Las tablas de cuatro entradas, diez semillas y test enero de 2025–agosto de 2026 de este capítulo corresponden a la **fase I exploratoria**. La **fase II D7** añade búsqueda equitativa, piloto anterior a su ejecución definitiva, dimensionamiento y evaluación preespecificada sobre el 01/09–30/10/2026. Sus recetas, recuentos y resultados se identifican por separado en el [capítulo 9](09_estudio_planificado.md).

El benchmark inicial de fase I compara **GRU, LSTM y TCN** con D=7/14/21/28, n=7, h=1 y diez semillas. Se completaron **120 trayectorias de CV, 1.440 segmentos mensuales, 2.880 ajustes y 43.920 predicciones**. El cierre final verificó 120 checkpoints: 80 ajustes nuevos y 40 LSTM idénticos reutilizados. El test histórico contiene 608 fechas y 72.960 predicciones. Los [manifiestos de procedencia](../_static/gru_lstm_tcn_20261105/protocols_manifest.json) distinguen fases, hashes y reutilización.

**El periodo de test ya había sido observado en el proyecto.** La nueva configuración se cerró antes de puntuar sus checkpoints, pero esa secuencia no vuelve desconocido el mercado ni convierte el análisis en confirmación temporal independiente. La comparación dentro de la terna es posterior a la evaluación y exploratoria. Un benchmark con tres modelos y pocas semillas puede no detectar diferencias pequeñas; ese límite debe cuantificarse y no ocultarse tras el promedio.

## 7.1. Potencia, tamaño de muestra y multiplicidad

### 7.1.1. Friedman con tres modelos y cinco bloques

Con tres modelos y cinco bloques, Friedman puede tener poca potencia y Nemenyi ser conservador. La ausencia de significación no demuestra equivalencia. Tampoco significa que ningún efecto sea detectable: intervienen la alternativa, la dispersión, los empates, la unidad de bloque y la referencia exacta, de permutación o aproximada utilizada.

Un bloque completo debe contener la misma unidad bajo los tres modelos. Semillas, fechas, meses y ventanas no se mezclan como réplicas independientes de la misma naturaleza. En este experimento una semilla identifica una trayectoria de aleatorización condicionada al calendario; los doce meses de CV no son doce mercados. La interpretación de un Friedman entre semillas tendría ese mismo alcance limitado.

El [Friedman D7 vigente](../_static/gru_lstm_tcn_20261105/friedman_D7.csv) se calculó **post-hoc** con tres modelos y diez bloques-semilla, bajo referencia χ² aproximada con dos grados de libertad:

| Fase | χ² de Friedman | p aproximado | p Holm2, sensibilidad entre fases |
|---|---:|---:|---:|
| CV 2024 | 18,2 | 0,0001116658 | 0,0002233316 |
| Test histórico observado | 16,8 | 0,0002248673 | 0,0002248673 |

Estos resultados corresponden a aleatorización sobre fechas conocidas, no a diez mercados independientes. Los contrastes pareados y su multiplicidad se declaran por separado en el [plan vigente](../_static/gru_lstm_tcn_20261105/public_analysis_plan.json); no se eligen ni se ocultan según el resultado del omnibus. No se ejecuta Nemenyi: tiene su propio control simultáneo y no se sustituye automáticamente por Holm.

### 7.1.2. Límite del Wilcoxon pareado

Para R diferencias pareadas independientes bajo el alcance declarado, sin ceros ni empates en sus valores absolutos y con signos equiprobables bajo el nulo bilateral exacto, el menor p-value posible es:

$$
p_{\min}=\frac{2}{2^R}.
$$

Con **cinco pares**, el mínimo es 0,0625: ese contraste no puede rechazar al nivel 0,05 ni en el caso más extremo. Con **seis**, el mínimo es 0,03125. R cuenta pares de resultados de dos modelos, no la suma de ejecuciones de las tres redes. Ceros, empates y la convención exacta o aproximada del software deben declararse.

La guía establece cinco ejecuciones como mínimo general y seis para pruebas por rangos. Esos mínimos no garantizan potencia 0,80. Wilcoxon tampoco contrasta automáticamente la misma media que una t pareada; una interpretación de desplazamiento requiere condiciones adicionales, incluida simetría. No se cambia retrospectivamente a una prueba unilateral o a otro método porque produzca un resultado favorable.

### 7.1.3. Corrección de Holm con tres pares

Tres modelos producen **tres comparaciones pareadas por configuración**. En Holm, el menor p-value se compara primero con α/3≈0,0167 para α=0,05; los siguientes pasos usan α/2 y α si los anteriores rechazan.

| Pares útiles R | Menor p-value bilateral exacto | ¿Permite rechazo a 0,05 sin ajuste? | ¿Permite el primer paso de Holm, 0,05/3? |
|---|---:|---|---|
| 5 | 0,0625 | No | No |
| 6 | 0,03125 | Sí | No |
| 7 | 0,015625 | Sí | Sí |
| 8 | 0,0078125 | Sí | Sí |
| 10 | 0,001953125 | Sí | Sí |

Esta resolución discreta no es una recomendación de tamaño ni una estimación de potencia. La comparación principal D7 de GRU–LSTM–TCN contiene tres pares por fase; la sensibilidad a cuatro historias contiene doce. Cada familia conserva su pregunta y corrección, como muestran los [pares CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv), [pares test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv) y familias completas [CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_allD_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_allD_test.csv).

La referencia principal es la t pareada declarada, con diagnósticos de sus supuestos. Ajustar sus p-valores no hace simultáneos unos IC marginales ni vuelve correcto un modelo de independencia inadecuado. Los diagnósticos BDS/Ljung–Box y DM mantienen sus propias unidades y familias; sus rechazos no se suman como confirmaciones independientes.

## 7.2. Cómo abordar estas limitaciones

### 7.2.1. Repeticiones y validación respetando el tiempo

La historia inicial es 2023 y los orígenes de validación son los días de 2024. Los índices reales de `split_train_val_forwardChaining` se agrupan en doce bloques mensuales con u≤inicio del mes−D−1 y validación interna reciente con purga propia. Los pesos quedan fijos dentro del mes; cada pronóstico diario usa únicamente OHLCV hasta el día anterior. La selección de D se realiza con ese desarrollo ya expuesto, no con test.

Hay diez semillas 4101–4110 por red e historia. Permiten describir variación de entrenamiento condicionada al calendario, sin convertir las 120 combinaciones en 120 mercados. La validación cruzada repetida y una corrección de varianza solo son apropiadas si sus supuestos se justifican; no autorizan K-fold aleatorio ni mezcla de fechas en BTCUSDT.

El ajuste final utiliza el desarrollo admisible 2023–2024. Los 120 checkpoints se cierran antes de acceder al derivado del test en este ciclo, con 80 ajustes nuevos y 40 controles LSTM reutilizados. Sus veinte meses son intervalos de reporte de pesos fijos, sin adaptación o reescalado con test. Septiembre–octubre de 2026 no se incorporó a fase I. Su asignación posterior y explícita a fase II D7 se documenta en el capítulo 9.

Remuestrear fechas o semillas requiere declarar la unidad y la dependencia que se intenta representar. Un bootstrap sobre predicciones guardadas condiciona a esos modelos; no es un nuevo entrenamiento o mercado. Ni esa operación ni Holm reparan fuga temporal, selección con la evaluación o un p-value mal calibrado.

### 7.2.2. Diferencia mínima detectable y tamaños de efecto

**Fase I:** no se ejecutó un piloto prospectivo para dimensionar sus repeticiones. Las diez semillas fueron un presupuesto fijo. **Fase II D7:** las recetas se seleccionan mediante HPO comparable, el piloto usa diez semillas nuevas por familia y el dimensionamiento precede a los ajustes definitivos. El [capítulo 9](09_estudio_planificado.md) conserva R, MDE y supuestos. El piloto de D7 no se transfiere a D14/21/28 ni acredita potencia temporal o conjunta del 80 %.

Una planificación pareada debe estimar la desviación de las diferencias de la métrica primaria bajo el procedimiento futuro, fijar δ y la familia de comparaciones antes de las nuevas ejecuciones, y justificar la unidad de repetición. Bajo una aproximación normal independiente:

$$
R\approx\frac{(z_{1-\alpha_*/2}+z_{1-\beta})^2\widehat\sigma_d^2}{\delta^2},
$$

donde α* refleja la multiplicidad de planificación. Con muestras pequeñas debe revisarse la referencia t, la incertidumbre de la DE y, si procede, una alternativa por simulación o remuestreo compatible con el diseño. Un cálculo retrospectivo de MDE es sensibilidad del diseño; no acredita un piloto previo o potencia alcanzada.

Después del test de fase I se definió una convención académica del 5 % frente a GRU/D7 y persistencia para un futuro modelo avanzado. El ancla neuronal de CV es RMSE 0,005154568533 y da δ≈0,00025772842665; fase II registró ese valor antes de su reserva. El [registro del margen futuro](../_static/gru_lstm_tcn_20261105/future_margin.json) conserva esa cronología. La [justificación ampliada de §1.1.4](01_problema_dataset.md) responde al uso confirmado de **comparación académica de modelos**: exige una reducción apreciable del error antes de aceptar mayor complejidad y sitúa el 5 % entre sensibilidades del 2,5 % y del 10 %.

El {download}`suplemento de contexto académico <../_static/e1_academic_margin_20261106/README.md>` usa sólo desarrollo 2023–2024: δ equivale a 1,60 DE del RMSE CV entre semillas de GRU/D7 y al 2,64 % de la DE observada del objetivo en 2024. Esas escalas contextualizan una decisión normativa; no derivan un umbral óptimo ni prueban relevancia temporal o económica. La fundamentación es posterior y conserva δ, R, recetas y baseline: no se retrofecha como margen pretest de fase I, no añade pruebas y no establece equivalencia o utilidad financiera.

Las [diferencias de CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv) reportan magnitud, dispersión e intervalos junto a p-valores ajustados y sin ajustar. El tamaño de efecto pareado describe señal respecto a variación entre semillas; no sustituye la diferencia en unidades de σ₇ ni una valoración contextual del margen práctico.

### 7.2.3. Probabilidad de superioridad

Para métricas en las que menor es mejor, una proporción descriptiva entre las R ejecuciones emparejadas puede definirse como:

$$
\widehat p_{A<B}=\frac1R\sum_{s=1}^{R}
\left[\mathbb I(m_{A,s}<m_{B,s})+
\frac12\mathbb I(m_{A,s}=m_{B,s})\right].
$$

Su unidad es la semilla bajo las mismas fechas y recetas. Una fracción de victorias no es un p-value ni una probabilidad posterior de que la diferencia media sea favorable. Un intervalo binomial necesita declarar cómo se tratan los empates y asumir pares de aleatorización independientes condicionados al calendario. No incluye variabilidad de otros periodos o mercados.

En el [test histórico D7](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv), la fracción observada de semillas con menor RMSE de A y los intervalos exactos de Clopper–Pearson son:

| Comparación A frente a B | Victorias de A | Fracción | IC 95 % condicional |
|---|---:|---:|---|
| GRU frente a LSTM | 10/10 | 1,00 | [0,692; 1,000] |
| GRU frente a TCN | 8/10 | 0,80 | [0,444; 0,975] |
| LSTM frente a TCN | 0/10 | 0,00 | [0,000; 0,308] |

No hubo empates bajo la tolerancia declarada. Diez victorias de diez no prueban superioridad universal, una reducción superior al 5 % o equivalencia; cero victorias tampoco significa probabilidad universal cero. Las [fracciones de CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv) se reportan por separado. Los contrastes y los IC de magnitud de §5 mantienen esas preguntas separadas.

### 7.2.4. Equivalencia requiere un análisis separado

La ausencia de significación no demuestra equivalencia. Para sostenerla se necesitaría un procedimiento predefinido, como **TOST** con una banda justificada o un análisis bayesiano con **ROPE**, con sus supuestos y reglas.

En TOST con α=0,05 sin otro ajuste, ambas pruebas unilaterales deben rechazar; la regla usual corresponde al IC del **90 %** completamente dentro de la banda. Se distingue del IC del 95 % de reporte general y se adapta a la multiplicidad aplicable. ROPE requiere modelo, priors, posterior y decisión; no se obtiene contando semillas ganadoras.

El margen de mejora frente al baseline no crea automáticamente una banda de equivalencia. Una diferencia pequeña y precisa puede ser distinguible de cero y estar dentro de una banda práctica. Si no hay evidencia suficiente para ninguna afirmación, el resultado es inconcluso. Este libro **no afirma equivalencia** y no declara TOST/ROPE ejecutados.

## 7.3. Objetivo y criterio de honestidad inferencial

El propósito es establecer un baseline, su variabilidad y sus límites. **GRU/D7** fue seleccionado por la media de RMSE de CV antes de puntuar el nuevo ciclo en test. La [selección registrada](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) mantiene las cuatro historias y no se modifica por el ranking del test.

Las cuarenta trayectorias LSTM de CV se reentrenaron y reprodujeron exactamente 14.640 predicciones; en test se reutilizaron 24.320 predicciones de checkpoints idénticos verificados. Esa identidad es un control de reproducción, no una nueva confirmación de generalización o cuarenta inferencias adicionales.

Los resultados tienen matices: persistencia obtiene menor RMSE que GRU en CV. En test GRU tiene menor RMSE, mientras persistencia conserva mejores MAE y MAPE. Se reportan ambos hechos. Una mejora en la métrica primaria no demuestra superioridad en todas las métricas ni utilidad financiera.

Los IC entre semillas condicionan al calendario y los contrastes dentro de la terna son **post-hoc**. El periodo de test ya era conocido; congelar el nuevo código y los pesos mejora la trazabilidad sin borrar esa exposición. Se requiere otra evaluación independiente si los resultados se usan para diseñar una propuesta posterior. La dependencia residual también puede provenir del solapamiento del objetivo, actualización mensual o deriva; no cuantifica automáticamente una mejora explotable.

## 7.4. Entregables y estado documental real

| Producto | Evidencia vigente | Límite |
|---|---|---|
| Informe en Jupyter Book | Capítulos centrados en GRU–LSTM–TCN, sus cuatro entradas y el baseline GRU/D7 | Publicación y revisión no equivalen a cumplimiento científico total. |
| Código EDA, redes y estadística | Recetas, contratos de tensor, semillas, índices, escalado y análisis de predicciones guardadas | Adaptaciones compactas, no réplicas exactas de redes entrenadas en artículos. |
| Datos y adquisición | Fuente oficial, scripts, manifiestos y reglas de calidad | Una serie BTCUSDT; el periodo de test ya había sido observado. |
| Literatura y flujo | N7 y flujo de 244 ocurrencias/182 grupos conservados; bitácora actual de WoS, Scopus e IEEE acreditada; ampliación gratuita y síntesis acumulada con lecturas y roles separados | Las consultas acreditadas no equivalen a cribado exhaustivo de todos sus resultados. Distinguir familias autónomas, híbridos, activos, objetivos y versiones; no inferir Q1 o indexación desde editorial. |
| Benchmark e inferencia | Fase I: 120 trayectorias CV y 120 resultados finales; fase II D7: HPO, piloto, ajuste dimensionado y nueva reserva (§9) | Fase I posterior y exploratoria; fase II preespecificada y condicionada al calendario, con límites temporales y de generalización. |
| Dependencias y configuraciones | [Catálogo CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv), [catálogo final](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv) y manifiestos | El cierre final incluye 80 ajustes nuevos y 40 controles reutilizados, no 120 entrenamientos nuevos. |
| Métrica, margen, unidad y partición | Cronologías separadas: fase I exploratoria y fase II D7 registrada antes de su reserva, con piloto y R=465 | La potencia planificada es marginal entre semillas; no acredita potencia temporal, conjunta o de otro mercado. |

Los archivos descargables del [capítulo 8](08_entregables.md) distinguen resúmenes públicos, código y evidencia completa. El registro de noviembre de 2026 es posterior al mercado: estos son backtests históricos, no predicciones emitidas en vivo.

La {download}`ampliación bibliográfica del 06/11 <../_static/literature_e1/native_search_20261106/README.md>` acredita exportaciones nativas de WoS y Scopus para E1. Sus 182 grupos no equivalen a textos completos incluidos: un resultado de precio, dirección o volatilidad implícita puede corresponder a otro objetivo. La {download}`recuperación posterior de fuentes <../_static/literature_e1/internet_closure_20261106/README.md>` revisa los 22 casos inicialmente abiertos, con cuatro originales completos y una transcripción completa diferenciada del PDF editorial; cerró cinco decisiones de exclusión/contexto y conservó **17 elegibilidades pendientes en ese corte**. Las fuentes complementarias y sus roles se mantienen separados. Las frecuencias **GRU 3/7, LSTM 5/7 y TCN 1/7** pertenecen al corte verificado del **05/11**: este seguimiento no las actualiza por simple acceso a registros o textos.

La {download}`sustitución posterior por fuentes de acceso gratuito <../_static/literature_e1/open_access_replacement_20261107/README.md>` incorpora las lecturas de los tres PDF entregados. Poudel y Tang aportan comparadores recurrentes, pero sus insuficiencias de objetivo/arquitectura/protocolo impiden usarlos como recetas estrictas; Nokar se conserva como contexto porque no acredita una evaluación identificable de volatilidad continua para las redes finales. Las ocho decisiones verificadas y los 14 informes no recuperados suman los 22 casos originales. Estos últimos tienen una disposición terminal de recuperación al corte, no catorce compras o lecturas obligatorias pendientes. La inaccesibilidad no se convierte en exclusión metodológica y las alternativas no se contabilizan como textos recuperados de esos 14.

La {download}`bitácora histórica de recuperación <../_static/literature_e1/search_recovery_20261106/README.md>` registró cuatro accesos nativos bloqueados y doce consultas propuestas que no se ejecutaron en esa etapa; no recuperó el historial privado ni acreditó entonces una consulta de IEEE. Los totales de aquellos intentos quedaron desconocidos, no en cero. Dos fechas First Online de capítulos nominalmente de 2027 se verificaron en 2026; otras dos fechas sólo corresponden al volumen relacionado y tres fechas efectivas individuales siguen sin resolver. La ronda acreditada posterior se documenta por separado y no reconstruye esas sesiones antiguas.

La primera ampliación de {download}`dos originales adicionales revisados <../_static/literature_e1/primary_reviews_20261106/README.md>` elevó la bibliografía de 40 a **42 referencias** y aportó evidencia en estratos separados: CVI de volatilidad implícita con TCN/LSTM y modelos híbridos GARCH con componentes neuronales. Su [matriz de evidencia](../_static/literature_e1/primary_reviews_20261106/evidence_matrix.csv) documenta el papel de cada red y los límites del diseño. No justifican tratar nueve estudios como un conjunto homogéneo de comparadores autónomos del objetivo σ₇.

La {ref}`bitácora provisional de §3.9 <provisional-native-log-e1>` conserva las cotas WoS ≥8/≥59 y Scopus ≥19/≥158 derivadas de registros únicos exportados, por familia y archivo. Eran valores de preparación, sin total nativo observado ni fecha de ejecución recuperada; no se incorporan al flujo verificado como nuevos resultados. IEEE no tenía una base para estimar su total en aquel registro.

La {ref}`bitácora de ejecución actual <current-native-execution-e1>` acredita consultas realizadas el **06/11/2026, Colombia**, en las tres bases: **271 resultados de WoS, 538 de Scopus y 32 de IEEE**, con 271/538/25 registros recibidos respectivamente. Los siete resultados IEEE no exportados y su filtro de acceso abierto se declaran. Las 834 ocurrencias de esta ronda forman 620 grupos provisionales; no se afirma que todos hayan sido leídos o evaluados para inclusión. La documentación de consulta está entregada, separada de la exhaustividad del cribado.

La {ref}`ampliación gratuita posterior <free-model-expansion-e1>` incorpora la selección de siete trabajos con TCN y seis con MLP: tres obras ya citadas y diez referencias adicionales. La {ref}`síntesis acumulada <homogeneous-expansion-e1>` reconcilia identidades, versiones, lectura, objetivos y calidad antes de calcular frecuencias. La cuota de búsqueda por arquitectura no representa prevalencia en la literatura. Los trabajos de bolsa o VIX se conservan como evidencia relacionada; un componente TCN de un híbrido no se contabiliza automáticamente como base autónoma de volatilidad cripto.

El experimento principal compara **GRU, LSTM y TCN**. Las menciones de MLP corresponden a candidatos bibliográficos o antecedentes históricos identificados; no añaden un cuarto modelo al benchmark ni cambian sus resultados.

## 7.5. Criterios de evaluación del entregable

Los pesos de la rúbrica suministrada son:

| Criterio | Peso |
|---|---:|
| EDA | 30 % |
| Revisión de literatura y extracción de los tres modelos base | 30 % |
| Implementación de los tres modelos base | 20 % |
| Benchmark, definición del baseline y análisis estadístico | 20 % |
| **Total** | **100 %** |

Son pesos de evaluación, no porcentajes de avance ni una calificación obtenida. Disponer de texto, obtener una métrica favorable o superar el build no demuestra por sí solo cumplimiento. La elección de tres nombres no acredita recurrencia, simplicidad, no redundancia o diseño prospectivo. Este libro no calcula una nota a partir de la matriz.

## 7.6. Buenas prácticas y cierre

| Ítem | Estado y evidencia | Límite pendiente |
|---|---|---|
| Repeticiones | Fase I: diez semillas por familia e historia. Fase II D7: R calculado desde su piloto, sin contar días o meses como mercados | La potencia planificada es marginal entre semillas y supone transferencia de dispersión; no es potencia temporal. |
| Media, DE, IC y efecto | Tablas completas de seis métricas y diferencias pareadas | Incertidumbre condicional al calendario; no intervalos de otros mercados. |
| Multiplicidad y supuestos | Familia D7 de tres pares por fase, sensibilidad de doce pares y diagnósticos separados | Análisis posterior y referencias aproximadas en bloques residuales cortos. |
| Consistencia | Mismas covariables, n7/h1, semillas, fechas y reglas causales dentro de cada fase; HPO comparable en fase II | Cambiar D cambia la purga y cohorte. Las métricas de dos periodos distintos no identifican una mejora causada por HPO. |
| Reproducción | Controles LSTM de fase I idénticos; curvas nuevas de recetas finales en §9.10 y comparación con manifiesto histórico | La reproducción posterior de recetas finales no coincide en curvas/checkpoints históricos; identidad de configuración no es identidad de entrenamiento. |
| Cierre anterior al acceso de este ciclo | Configuraciones y 120 checkpoints firmados antes de leer el derivado test | El periodo ya era conocido; no es confirmación temporal independiente. |
| Baseline y margen avanzado | GRU/D7 por CV; 5 % fijado, con justificación académica de desarrollo ampliada después | La justificación no retrofecha el margen ni acredita utilidad; la propuesta futura necesita piloto, protocolo y evaluación propios. |

El [estado de pendientes](protocolo_pendiente.md) distingue el cierre del piloto/HPO D7 del cribado de la ampliación bibliográfica, la limitada recurrencia histórica de TCN, la justificación de GRU/LSTM y los límites de fuerza inferencial. La selección de familias debe responder a los textos y mecanismos documentados; el resultado del benchmark no decide qué estudios se incluyen. E1 define un umbral y sus límites; no se inventa una evaluación avanzada ya ejecutada ni se fuerza superioridad universal. La equivalencia sigue siendo condicional a esa afirmación.

**Alcance de la nueva fase D7.** El [capítulo 9](09_estudio_planificado.md) reúne el diseño público previo, la rejilla equitativa de ocho candidatos por familia, el piloto de 60 días, el cálculo de R y los resultados sobre el 01/09–30/10/2026. Esa fase mantiene n=7/h=1 y covariables OHLCV, con D7 fijo. Los resultados de fase I y sus limitaciones permanecen identificados; la potencia planificada entre semillas no representa potencia temporal ni confirma otros mercados.

**Resultado y supuestos de fase II.** La reserva nueva tiene menor RMSE puntual para persistencia y, entre las redes, para TCN; el baseline GRU/D7-c03 se mantiene porque fue elegido con desarrollo. Shapiro–Wilk rechaza normalidad de los tres pares definitivos. Las t y sus IC se leen como aproximaciones para 465 pares; el objetivo de potencia del diseño no se transforma en potencia alcanzada. Ninguna comparación DM rechaza tras Holm-6 y ningún límite temporal familiar del bloque principal acredita una mejora superior al 5 %. Estos resultados y las sensibilidades se conservan en [§9.6](09_estudio_planificado.md), sin elegir después otro contraste o longitud de bloque.

**Suplemento posterior del 06/11.** Los Q–Q de diferencias, el Friedman global de fase II y la sensibilidad Wilcoxon/Holm se calculan sobre los RMSE por semilla ya publicados y se distinguen del análisis preespecificado en el [capítulo 9](09_estudio_planificado.md). Friedman aporta χ²(2)=391,6258 y Kendall W=0,4211, con alcance condicionado a las mismas sesenta fechas. Los tres pares Wilcoxon rechazan tras Holm bajo sus supuestos; esto no valida normalidad, no convierte semillas en fechas independientes ni sustituye DM o el bootstrap temporal. No se reabre la selección del baseline.

**Curvas nuevas del modelo final.** La {ref}`reproducción posterior de §9.10 <curvas-finales-reproducidas>` publica entrenamiento/validación interna y reajuste de **1.395 trayectorias, 2.790 etapas y 59.210 registros de pérdida**, con protocolo publicado antes de estos ajustes, 465 semillas por familia y sólo desarrollo 2023–2024. Las figuras y CSV muestran curvas observadas de esa ejecución; no se reconstruyen a partir de métricas agregadas ni se vuelve a evaluar la reserva.

**Límite de reproducción que persiste.** Coinciden 4.185/4.185 archivos de configuración, preprocesamiento y partición, pero **0/2.790 CSV de curvas y 0/1.395 checkpoints** con los hashes históricos. La época elegida coincide en 1.366/1.395 trayectorias; difieren 29 TCN. Esto no acredita reproducción exacta del entrenamiento. Sin los originales no se mide la discrepancia numérica ni se determina su causa; un hash distinto de checkpoint tampoco cuantifica diferencias entre pesos. El {download}`resumen y comparador públicos <../_static/e1_curve_reproduction_20261106/README.md>` conservan coincidencias y discrepancias. Los originales privados siguen sin recuperarse y la reproducción independiente de contrastes temporales desde las observaciones originales permanece abierta. Las nuevas curvas no añaden otro mercado o muestra de evaluación ni cambian baseline, margen, métricas o conclusiones históricas.

La {ref}`auditoría numérica actual <auditoria-numerica-e1>` añade un control técnico de seis celdas fijadas previamente, con dos procesos independientes y **18/18 coincidencias canónicas** de pérdidas y pesos. Acredita repetibilidad actual de ese subpanel, sin extenderla al entrenamiento original completo. El procedimiento, dependencias, semillas, configuraciones y curvas están entregados; la recuperación de archivos privados idénticos bit a bit no es una exigencia añadida de la guía.
