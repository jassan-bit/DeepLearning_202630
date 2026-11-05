# 6. Guía de Procedimiento por Tipo de Problema

La formulación actual corresponde al **tipo 4: regresión temporal**, con la ruta del **apartado 6.4 de la guía del Entregable 1**. Las redes reciben ventanas de covariables OHLCV y no rezagos explícitos de la volatilidad objetivo. La clasificación describe el contrato de entrada vigente; el diagnóstico de cuatro referencias Ridge ejecutado en el bloque C encuentra dependencia residual, sin cambiar automáticamente ese contrato. El orden cronológico y la dependencia entre fechas se conservan en la partición y la inferencia.

La numeración de este capítulo es continua dentro del libro. La referencia 6.4 identifica la ruta de la guía original; no obliga a reproducir las rutas de otras modalidades.

La guía contiene **once rutas numeradas, de 6.1 a 6.11**. Su lista inicial agrupa etiquetado secuencial y secuencia a secuencia bajo el ítem 10; aquí se conservan esos rótulos sin convertirlos en una nueva enumeración de tipos. Para el contrato vigente se desarrolla únicamente la ruta 6.4:

| Ruta de la guía | Tipo de tarea | Aplicabilidad al proyecto |
|---|---|---|
| 6.1 | Clasificación tabular | No: el objetivo es una magnitud continua, sin clases. |
| 6.2 | Regresión tabular | No es la ruta principal: que OHLCV sea numérico no elimina el reloj de información ni permite barajar fechas. |
| 6.3 | Series de tiempo | No es la ruta principal mientras no se incluyan rezagos explícitos del objetivo; se revisará si cambia ese contrato. |
| **6.4** | **Regresión temporal** | **Aplicable: OHLCV rezagadas, salida continua, disponibilidad causal y partición cronológica con embargo.** |
| 6.5 | Regresión espacial | No hay coordenadas, vecindad o unidades espaciales de predicción. |
| 6.6 | Regresión espacio-temporal | No se predicen ubicaciones ni se construye un tensor o grafo espacial. |
| 6.7 | Clasificación de imágenes | Las entradas no son imágenes y no hay clases visuales. |
| 6.8 | Segmentación de imágenes | No existen máscaras ni objetivos por píxel. |
| 6.9 | Clasificación de texto | No se incorporan documentos o etiquetas textuales; citar estudios con Twitter no añade esa modalidad. |
| 6.10 | Etiquetado secuencial | No se asigna una etiqueta a cada elemento de la ventana de entrada. |
| 6.11 | Aprendizaje secuencia a secuencia | La salida es una volatilidad escalar de un día, no una secuencia de salida. |

Usar LSTM o Transformer y una entrada D×5 no cambia por sí solo el tipo de tarea. La clasificación como tipo 4 tampoco afirma independencia o exogeneidad de OHLCV respecto a la volatilidad: los precios intervienen en construir retornos y pueden aportar información sobre volatilidad pasada. La distinción operativa de la guía se aplica a las entradas explícitas registradas.

El procedimiento principal utiliza validación forward chaining de `timeseries-cv`, con coautoría de Filipe R. Ramos, y las mismas tres redes GRU/LSTM/TCN para D=7/14/21/28. Los doce bloques mensuales de 2024 generan 120 trayectorias primarias y 43.920 predicciones; las cuatro historias se reentrenan y LSTM actúa como control reproducido. El [registro vigente](../_static/gru_lstm_tcn_20261105/protocols_manifest.json) distingue esos ajustes del cierre de 120 checkpoints finales y de la evaluación posterior sobre test ya observado.

## 6.1. Aplicación de la ruta 6.4 de la guía: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test final comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

En notación de calendario, para el resultado del día d se emite al inicio UTC de d, se usa OHLCV de d−D,…,d−1 y se predice σ7(d), conocida al terminar d. El campo de datos σ7(d) incluye los retornos $r_{d-6},\ldots,r_d$; en la convención de la fórmula del curso de §1.1.1, que indexa la volatilidad por el día siguiente al último retorno incluido, es $Y_d=\sigma_{d+1}=\sigma_7(d)$. Ambas escrituras identifican la misma etiqueta del día de resultado d. El OHLCV completo de d no es entrada disponible al comenzar d. Origen, día de resultado y disponibilidad de etiqueta se conservan como campos distintos.

Se mantiene **2023 como entrenamiento inicial, 2024 como validación forward chaining y 2023–2024 para el ajuste final tras cerrar las configuraciones**. El test evaluado comprende enero de 2025–agosto de 2026; septiembre–octubre de 2026 permanece sin asignación automática.

Se utiliza la función real `split_train_val_forwardChaining(np.arange(731),D,1,1)` de `timeseries-cv` 0.1.5 para D=7/14/21/28. Sus orígenes diarios se reúnen en **doce bloques mensuales**: cada red, historia y semilla se ajusta al comienzo del mes con índices de la función filtrados por embargo y calidad. Los pesos quedan fijos dentro del mes y cada pronóstico h=1 usa OHLCV hasta el día anterior. La agrupación mensual es una adaptación explícita, no una API nativa ni 366 reajustes.

Las cohortes crecen con el pasado permitido de 2024. Los manifiestos por historia y las tablas de CV conservan los índices realmente usados. Se reúnen las 366 predicciones por trayectoria para RMSE/R² anual; no se promedian R² de orígenes unitarios. La selección de historia y familia usa ese desarrollo ya expuesto: no es validación anidada independiente de todo el procedimiento.

La ruta 6.4 exige un **embargo igual al máximo rezago de las covariables**. Para $X_t=(z_{t-D},\ldots,z_{t-1})^\top$, los rezagos son de uno a $D$ días de calendario y se fija $E_D=D$. Se define embargo como el número de días de resultado inmediatamente anteriores a $t$ que se omiten del ajuste: $t-D,\ldots,t-1$. Así, los candidatos de entrenamiento deben cumplir $u\le t-D-1$, además de tener entradas válidas y etiquetas conocidas al emitir el pronóstico. Esta convención deja verificable la diferencia entre días omitidos y distancia entre dos fechas.

| Historia de covariables $D$ | Máximo rezago y embargo $E_D$ | Último día de resultado admisible en el ajuste del origen $t$ |
|---|---|---|
| 7 días | 7 días de calendario omitidos | $t-8$ |
| 14 días | 14 días de calendario omitidos | $t-15$ |
| 21 días | 21 días de calendario omitidos | $t-22$ |
| 28 días | 28 días de calendario omitidos | $t-29$ |

Como ejemplo de frontera para **t=01/01/2024 y D=7**, los resultados de 25–31/12/2023 se omiten del ajuste y el último resultado admisible es 24/12/2023. Las covariables completas de 25–31/12 sí forman la entrada conocida del pronóstico de 01/01. Su etiqueta se conoce al terminar 01/01, al inicio UTC de 02/01. Es una ilustración de índices y disponibilidad, no una corrida nueva ni una decisión de reajuste diario.

**Preparación corregida.** La versión 2 de `results/cv_forward_chaining/folds.csv` aplica esta convención y conserva **5.856 filas: 366 orígenes por cada una de las 16 configuraciones**. El último día de resultado del ajuste es $t-D-1$ en todas las filas. Se retiró una muestra de ajuste adicional por fila respecto de la versión anterior, que terminaba en $t-D$ y omitía $D-1$ días intermedios. Los índices, fechas y hashes se comprobaron mediante una reconstrucción independiente; los CSV de desarrollo y la exclusión del 24/03/2023 permanecen intactos. El [manifiesto vigente](../_static/cv_forward_chaining/folds.csv) y su [resumen de verificación](../_static/cv_forward_chaining/summary.json) documentan la regla, el hash y los recuentos corregidos. Esta preparación no entrena modelos ni abre archivos de test.

La historia observable anterior a validación o test puede formar parte de sus entradas, incluidos los días cuyo resultado se omite del ajuste por embargo. Se revisan por separado el soporte temporal y la disponibilidad de entradas y etiquetas. El embargo de covariables no elimina automáticamente el solapamiento entre volatilidades móviles: con la convención actual, si $n>D+1$, la última etiqueta de ajuste y la de validación comparten $n-D-1$ retornos. Esto ocurre en **2.196 filas de seis configuraciones de sensibilidad** del manifiesto; se registra como dependencia, no como consulta del retorno futuro. Esa dependencia debe conservarse en el diseño inferencial. Incorporar una etiqueta pasada de 2024, cuando ya se conoce y cumple los filtros del entrenamiento permitido, es diferente de consultar la etiqueta futura que se evalúa.

### 6.1.2. Ventanas y preprocesamiento causal

La adquisición es de un minuto y el modelado es **diario UTC**. La CV y el test comparan cinco covariables OHLCV con **D=7/14/21/28 días**, manteniendo la desviación poblacional de **n=7 retornos diarios** y horizonte h=1. Las volatilidades n=14/21/28 son sensibilidad del objetivo explorada en EDA, no objetivos entrenados del benchmark.

La ruta de regresión temporal requiere estudiar frecuencia, desfases y disponibilidad de covariables, su relación con el objetivo, cambios de régimen y autocorrelación de los errores de una referencia. Los diagnósticos de ACF/PACF y estacionariedad ya presentes en el EDA son información temporal complementaria; no constituyen por sí solos una selección de ventana o un resultado de modelos. La correspondencia de los requisitos específicos y la evidencia debe mantenerse explícita en el capítulo 2.

El bloque C de la guía exige diagnosticar residuos de una referencia y reconsiderar el planteamiento si queda autocorrelación importante. La sección 2.2 aporta el contrato de disponibilidad, correlaciones cruzadas de las cinco covariables con 28 rezagos sobre las mismas 308 fechas elegibles de 2023, relaciones móviles de 60 días con al menos 50 pares y residuos fuera de entrenamiento de la referencia definida en §6.1.5. Los máximos de correlación y las variaciones móviles son descriptivos: no seleccionan una historia de entrada ni demuestran cambios de régimen.

Los errores de las cuatro referencias muestran autocorrelación. Las etiquetas consecutivas de volatilidad de siete retornos comparten seis retornos; la especificación lineal y la ausencia de adaptación de esa referencia también limitan la interpretación. Esta evidencia exige conservar la dependencia en la evaluación y revisar la especificación, pero no establece por sí sola que el problema necesite autorregresión del objetivo. Si se introducen rezagos de la volatilidad como entradas, el proyecto pasaría al tipo 3 y habría que actualizar su ruta metodológica.

La reconsideración se documenta en esta revisión del **3 de noviembre de 2026**: se mantiene el tipo 4 porque continúa el contrato OHLCV sin historia explícita de σ como entrada. Se reconoce la dependencia residual y no se atribuye a una causa única. Si se decide añadir σ rezagada, antes de ejecutarlo se versionarán entrada, formulación, disponibilidad, partición y comparación bajo la ruta 6.3; el diagnóstico actual no demuestra que ese cambio sea necesario ni que ya haya ocurrido.

La representación principal `relative_window` resta a los cuatro logaritmos OHLC el logaritmo del último cierre de la propia ventana, conocido en d−1; conserva log1p del volumen y aprende escalas con posiciones de ventanas de entrenamiento. El objetivo se divide por su media positiva de entrenamiento y se invierte para evaluar; no se transfieren estadísticas ni PCA globales del EDA.

En la parada interna del origen t, el bloque de validación termina como máximo en t−D−1. Si v es su primer resultado, el subentrenamiento usa u≤v−D−1. Escaladores y escala de Y se ajustan **solo al subentrenamiento**; la cohorte interna reciente de 90 días deja **90−D etiquetas**, es decir 83/76/69/62. Después de elegir duración se descartan los pesos internos y se reajusta una red nueva con todo el entrenamiento mensual permitido. La etapa final toma la duración ya seleccionada en el último corte de CV, usa desarrollo permitido y congela sus pesos antes de abrir test.

Si las métricas de 2024 intervienen en elegir hiperparámetros o D, son **resultados de selección**. Una evaluación externa de todo el procedimiento requeriría selección anidada en el pasado de cada origen; una parada causal por sí sola no corrige la selección hecha con todo ese año. La evaluación histórica no elige reglas después de puntuar los nuevos checkpoints. El periodo ya observado conserva alcance exploratorio.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La terna vigente es **GRU–LSTM–TCN**. Su selección se documenta con publicaciones y manuscritos leídos y con el papel real de cada modelo. Akbiyik et al. (2023) compara las tres arquitecturas como referencias independientes de volatilidad de Bitcoin; Miura et al. (2019) añade evidencia directa para GRU y LSTM {cite:p}`akbiyik2023,miura2019`.

| Modelo base | Receta compacta ejecutada | Sesgo y alcance |
|---|---|---|
| **GRU** | Una capa unidireccional de 32 unidades, estado final, dropout 0,1 y salida Softplus; LR 0,003; **3.777 parámetros** | Compuertas de actualización y reinicio; memoria integrada en el estado. |
| **LSTM** | Una capa unidireccional de 32 unidades, estado final, dropout 0,1 y Softplus; LR 0,003; **5.025 parámetros** | Celda de memoria separada y compuertas de entrada, olvido y salida. |
| **TCN** | Tres bloques residuales, dos convoluciones causales por bloque, kernel 3, ancho 16 y dilataciones 1/2/4; lectura del último instante y Softplus; LR 0,001; **4.289 parámetros** | Convoluciones causales dilatadas; campo receptivo de **29 días**, que cubre las cuatro ventanas sin cambiar su profundidad. |

Las [especificaciones](../_static/gru_lstm_tcn_20261105/model_specs.json) distinguen las recetas propias de las redes de los artículos. No se afirma réplica exacta. Las frecuencias estrictas son GRU 3/7, LSTM 5/7 y TCN 1/7 dentro del corpus completo; no todos esos registros son artículos de revista indexados. La recurrencia de TCN requiere fortalecerse. GRU y LSTM pertenecen a la misma familia recurrente: sus diferencias de compuertas, memoria, capacidad y costo deben sostener la justificación de no redundancia; una frecuencia o una métrica favorable no la resuelve por sí sola.

Las tres redes usan las mismas covariables, etiquetas, orígenes, semillas y presupuesto de entrenamiento. No se añaden rezagos del objetivo o metadatos a una familia para mejorar su resultado. Los hiperparámetros se mantienen fijos al comparar D; no se ejecuta otra búsqueda después de ver test.

### 6.1.4. Pérdida y estrategia de pronóstico

La pérdida es **MSE** y la métrica principal **RMSE**. El objetivo se divide por su media positiva de entrenamiento y todas las métricas se calculan tras invertir la escala. La salida Softplus conserva volatilidad no negativa. La ruta 6.4 permite otras pérdidas, pero no obliga a probarlas: Huber o MAE necesitarían un procedimiento nuevo, declarado antes de su evaluación.

Cada salida es una magnitud escalar de horizonte **h=1**. El pronóstico se emite al inicio del día d con OHLCV hasta d−1 y su etiqueta se observa al cierre. La etiqueta σ₇ comparte seis retornos con la del día previo; no representa siete días enteramente futuros. El ejemplo sintético 4/3/2 de la librería CV no modifica ese contrato.

### 6.1.5. Métricas y referencias de contexto

Las [tablas de CV](../_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y [test histórico](../_static/gru_lstm_tcn_20261105/metric_summary_test.csv) contienen RMSE, MAE, MSE, R², MAPE y MASE por historia y modelo, con media, DE, cobertura e IC. RMSE y R² se calculan sobre las fechas de cada trayectoria y después se resumen sus diez semillas; no son métricas de un ensemble ni medias de R² por origen unitario.

MAPE admite etiquetas estrictamente positivas, sin epsilon oculto, e informa cobertura. MASE conserva una escala de diferencias de un día de calendario estimada únicamente con entrenamiento permitido. En CV puede cambiar por origen; se promedian errores escalados, sin reemplazarlos por una MAE dividida por una escala elegida después. Una escala nula o sin pares suficientes produce un valor no definido.

Persistencia es una referencia determinista adicional, no uno de los tres puestos DL. Para σ₇ previa requiere ocho cierres; esa información se distingue de D=7 OHLCV. En CV su RMSE es menor que el de GRU; en test GRU tiene menor RMSE, pero persistencia mantiene mejores MAE y MAPE. No se concluye superioridad en todas las métricas.

La ruta 6.4 conserva también las **cuatro Ridge diagnósticas del EDA**, con intercepto libre y penalización fija α=1, ajustadas una vez con entrenamiento admisible de 2023. Sus 1.464 predicciones y residuos de 2024 se documentan en el [resumen de la referencia temporal](../_static/eda_temporal_reference/summary.json). No se convierten en un cuarto modelo DL de test ni seleccionan la ventana de las redes.

El baseline neuronal vigente es **GRU/D7**, elegido con la media de RMSE de CV antes de acceder al contenido del test en este ciclo. El [registro de selección](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) conserva esa decisión y las cuatro historias. El ranking de test no la reabre.

Para una propuesta avanzada futura se define, **después del test actual**, una convención académica: superar el 5 % de reducción relativa de RMSE frente a GRU/D7 y persistencia, con incertidumbre y multiplicidad. El ancla neuronal CV da δ≈0,00025772842665. El [margen futuro](../_static/gru_lstm_tcn_20261105/future_margin.json) identifica su fecha y alcance. No es una tolerancia financiera validada ni un parámetro previo a los resultados actuales. Se requiere piloto y evaluación independiente propios de esa propuesta.

### 6.1.6. Inferencia y diagnóstico: pregunta, unidad y estado

La comparación **entre GRU, LSTM y TCN** se calcula sobre predicciones guardadas y se declara **post-hoc/exploratoria**. La configuración de las redes y la elección de D se cerraron antes de puntuar este ciclo; el periodo de test ya era conocido y los contrastes actuales no son confirmación temporal independiente.

| Pregunta | Unidad y alcance | Evidencia vigente |
|---|---|---|
| Diferencia de RMSE entre las tres recetas D7 | Diez diferencias pareadas por semilla, t bilateral, IC y corrección Holm para los tres pares por fase | [Pares CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv) y [pares test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv) |
| Sensibilidad a las cuatro entradas | Doce pares por fase; multiplicidad propia, sin convertir D o semillas en otros mercados | [Familia completa CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_allD_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_allD_test.csv) |
| Precisión predictiva temporal | Diferencias por fecha de pérdidas cuadradas medias entre semillas, HAC y corrección HLN, h=1 | [Diebold–Mariano CV](../_static/gru_lstm_tcn_20261105/dm_pairs_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/dm_pairs_test.csv) |
| Independencia de los residuos | BDS por bloque y semilla, con dimensiones, radio, multiplicidad y estados no definidos documentados | [BDS CV](../_static/gru_lstm_tcn_20261105/bds_by_fold_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/bds_by_fold_test.csv) |
| Autocorrelación remanente | Ljung–Box por bloque, con rezagos y grados de libertad documentados | [Ljung–Box CV](../_static/gru_lstm_tcn_20261105/ljung_box_by_fold_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/ljung_box_by_fold_test.csv) |

La t entre semillas estudia variación del entrenamiento bajo el calendario fijo. DM estudia pérdidas en fechas correlacionadas. Sus p-valores responden preguntas distintas y no son corroboraciones independientes. Se conservan la normalidad como diagnóstico, el tamaño de efecto y los intervalos; no se cambia el contraste para obtener el resultado más favorable.

BDS rechaza i.i.d., no exclusivamente ausencia de no linealidad aprovechable. Los meses tienen entre 28 y 31 fechas y las etiquetas se solapan: sus referencias asintóticas son aproximadas. La media de p-valores BDS se presenta solo como descripción. Ljung–Box no resta el número de pesos de una red como si fueran parámetros ARMA. Las reglas exactas y la fecha posterior del análisis constan en el [plan público](../_static/gru_lstm_tcn_20261105/public_analysis_plan.json).

No se afirma equivalencia. TOST/ROPE y una banda definida antes de sus cálculos solo serían necesarios para sostener esa afirmación. La ausencia de significación no significa igualdad. Tampoco el número de semillas o una corrección de multiplicidad acreditan potencia 0,80 o generalización a otra trayectoria de mercado.

## 6.2. Correspondencia entre procedimiento y evidencia

| Componente | Evidencia vigente | Límite |
|---|---|---|
| Tipo, frecuencia y entrada | Tipo 4; OHLCV diaria UTC desde fuente de un minuto; D=7/14/21/28 | No se entrenan redes a frecuencia de un minuto ni se añaden rezagos explícitos del objetivo. |
| Objetivo | σ₇ poblacional, ddof0, sin anualizar; h=1 | n=14/21/28 son sensibilidad de EDA, no objetivos del benchmark. |
| CV de Filipe | Doce bloques mensuales de 2024 y entrenamiento creciente con u≤origen−D−1 | Desarrollo previamente expuesto; no validación anidada de toda la selección. |
| Ejecución nueva | 120 trayectorias, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones | Las 40 trayectorias LSTM se reentrenan y reproducen 14.640 predicciones; no son otra serie. |
| Cierre final | 80 ajustes GRU/TCN nuevos y 40 LSTM idénticos reutilizados; 120 checkpoints firmados | La reutilización no se presenta como entrenamiento nuevo o costo intrínseco cero. |
| Test histórico | 608 fechas y 72.960 predicciones; 80 inferencias nuevas y 40 trayectorias reutilizadas | Periodo ya observado; 20 intervalos de reporte, sin reajustes o selección en test. |
| Selección | GRU/D7 por RMSE media CV | No se elige otra ventana por el resultado de test. |
| Inferencia entre la terna | Tres pares D7 por fase y sensibilidad de doce pares, con Holm | Análisis posterior; incertidumbre condicional a fechas y recetas. |
| Piloto y potencia | Diez semillas constituyen el presupuesto ejecutado | Piloto prospectivo y dimensionamiento de esta terna pendientes; no potencia alcanzada. |
| Literatura | Referencias independientes GRU 3/7, LSTM 5/7 y TCN 1/7 | Fortalecer recurrencia TCN y justificar las diferencias entre las recurrentes. |

Los [manifiestos de procedencia](../_static/gru_lstm_tcn_20261105/protocols_manifest.json), [catálogo CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv) y [catálogo final](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv) permiten seguir las fases sin volver a entrenar.

## 6.3. Secuencia de cierre y ejecución posterior

El registro de noviembre de 2026 es posterior a los periodos de mercado. Son resultados de un **backtest histórico**, no pronósticos emitidos en vivo en 2025–2026.

1. Se congelaron código, recetas, datos de desarrollo, cuatro entradas, diez semillas e índices de Filipe antes de ajustar la nueva terna.
2. Se completaron las 120 trayectorias de CV y se comprobó la identidad de LSTM con su control. Todas las recetas e historias permanecen reportadas.
3. Se seleccionó D=7 por familia y GRU/D7 como baseline por la media de RMSE de desarrollo; se cerraron las épocas finales a partir del corte de diciembre de 2024.
4. Se realizaron 80 ajustes finales nuevos y se verificaron 40 checkpoints LSTM idénticos. El manifiesto de 120 pesos y escaladores precede al acceso de este ciclo al contenido del test.
5. Se usó el derivado diario ya auditado, con 608 fechas y sin abrir ZIP crudos. Hubo 80 inferencias nuevas y reutilización firmada de 40 trayectorias LSTM; no se incorporó septiembre–octubre de 2026.
6. Se publican los análisis agregados y los contrastes dentro de la terna como **posteriores a la evaluación y exploratorios**. No se reabren las configuraciones para mejorar el test.

El siguiente cierre científico es un piloto y dimensionamiento específicos de GRU–LSTM–TCN, además del refuerzo bibliográfico. Si se diseña otro modelo con estos resultados, deberá reservar una evaluación independiente y registrar sus decisiones antes de producirla. La guía E1 anuncia desarrollos posteriores, pero no permite inventar los requisitos completos de documentos E2/E3 que no se han aportado.
