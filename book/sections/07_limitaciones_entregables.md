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

Para un modelo avanzado futuro se define **ahora, después del test actual**, una convención operativa del 5 % frente a GRU/D7 y persistencia. El ancla neuronal de CV es RMSE 0,005154568533 y da δ≈0,00025772842665. El [registro del margen futuro](../_static/gru_lstm_tcn_20261105/future_margin.json) conserva su alcance y sensibilidades. No se aplica retrospectivamente como margen pretest ni como banda de equivalencia o utilidad financiera.

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
| Literatura y flujo | Corte 05/11: siete estudios cripto verificados. Ampliación 06/11: cuatro exportaciones nativas WoS/Scopus, 244 ocurrencias y 182 grupos provisionales | Cribar los registros, verificar textos completos y fechas efectivas, completar trazabilidad IEEE y justificar recurrencia y mecanismos; no inferir Q1 o indexación desde editorial. |
| Benchmark e inferencia | Fase I: 120 trayectorias CV y 120 resultados finales; fase II D7: HPO, piloto, ajuste dimensionado y nueva reserva (§9) | Fase I posterior y exploratoria; fase II preespecificada y condicionada al calendario, con límites temporales y de generalización. |
| Dependencias y configuraciones | [Catálogo CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv), [catálogo final](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv) y manifiestos | El cierre final incluye 80 ajustes nuevos y 40 controles reutilizados, no 120 entrenamientos nuevos. |
| Métrica, margen, unidad y partición | Cronologías separadas: fase I exploratoria y fase II D7 registrada antes de su reserva, con piloto y R=465 | La potencia planificada es marginal entre semillas; no acredita potencia temporal, conjunta o de otro mercado. |

Los archivos descargables del [capítulo 8](08_entregables.md) distinguen resúmenes públicos, código y evidencia completa. El registro de noviembre de 2026 es posterior al mercado: estos son backtests históricos, no predicciones emitidas en vivo.

La {download}`ampliación bibliográfica del 06/11 <../_static/literature_e1/native_search_20261106/README.md>` acredita nuevas exportaciones nativas de WoS y Scopus para E1. Sus 182 grupos son provisionales y no equivalen a textos completos incluidos: un resultado que pronostica precio, dirección o un índice de volatilidad puede corresponder a otro objetivo. Las frecuencias **GRU 3/7, LSTM 5/7 y TCN 1/7** pertenecen al corte verificado del **05/11**. Incorporar registros o resúmenes no las actualiza. La ampliación tampoco demuestra cobertura exhaustiva de las bases ni cierra la búsqueda IEEE; los registros con año nominal 2027 requieren comprobar Early Access o fecha efectiva anterior al corte.

Los {download}`dos originales adicionales revisados <../_static/literature_e1/primary_reviews_20261106/README.md>` elevan la bibliografía actual a **42 referencias** y aportan evidencia en estratos separados: CVI de volatilidad implícita con TCN/LSTM y modelos híbridos GARCH con componentes neuronales. Su [matriz de evidencia](../_static/literature_e1/primary_reviews_20261106/evidence_matrix.csv) documenta el papel de cada red y los límites del diseño. No justifican tratar nueve estudios como un conjunto homogéneo de comparadores autónomos del objetivo σ₇.

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
| Reproducción | Controles LSTM idénticos, hashes y separación de ajustes/inferencias reutilizados | Identidad local no garantiza bit a bit en otras plataformas. |
| Cierre anterior al acceso de este ciclo | Configuraciones y 120 checkpoints firmados antes de leer el derivado test | El periodo ya era conocido; no es confirmación temporal independiente. |
| Baseline y margen avanzado | GRU/D7 por CV; futuro 5 % frente a GRU/D7 y persistencia | Margen definido después del test actual; la propuesta futura necesita piloto, protocolo y evaluación propios. |

El [estado de pendientes](protocolo_pendiente.md) distingue el cierre del piloto/HPO D7 del cribado de la ampliación bibliográfica, la limitada recurrencia histórica de TCN, la justificación de GRU/LSTM y los límites de fuerza inferencial. La selección de familias debe responder a los textos y mecanismos documentados; el resultado del benchmark no decide qué estudios se incluyen. E1 define un umbral y sus límites; no se inventa una evaluación avanzada ya ejecutada ni se fuerza superioridad universal. La equivalencia sigue siendo condicional a esa afirmación.

**Alcance de la nueva fase D7.** El [capítulo 9](09_estudio_planificado.md) reúne el diseño público previo, la rejilla equitativa de ocho candidatos por familia, el piloto de 60 días, el cálculo de R y los resultados sobre el 01/09–30/10/2026. Esa fase mantiene n=7/h=1 y covariables OHLCV, con D7 fijo. Los resultados de fase I y sus limitaciones permanecen identificados; la potencia planificada entre semillas no representa potencia temporal ni confirma otros mercados.

**Resultado y supuestos de fase II.** La reserva nueva tiene menor RMSE puntual para persistencia y, entre las redes, para TCN; el baseline GRU/D7-c03 se mantiene porque fue elegido con desarrollo. Shapiro–Wilk rechaza normalidad de los tres pares definitivos. Las t y sus IC se leen como aproximaciones para 465 pares; el objetivo de potencia del diseño no se transforma en potencia alcanzada. Ninguna comparación DM rechaza tras Holm-6 y ningún límite temporal familiar del bloque principal acredita una mejora superior al 5 %. Estos resultados y las sensibilidades se conservan en [§9.6](09_estudio_planificado.md), sin elegir después otro contraste o longitud de bloque.
