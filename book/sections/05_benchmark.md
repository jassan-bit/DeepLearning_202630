# 5. Benchmark de los Modelos Base y Definición del Baseline

Se completó el benchmark de **MLP, LSTM y CNN1D con entradas de 7, 14, 21 y 28 días**. El objetivo permanece en **siete retornos diarios, volatilidad poblacional sin anualizar**, horizonte **h=1 día** y cinco covariables OHLCV. La fuente de un minuto se agrega a diario antes de formar entradas.

La comparación vigente utiliza **D=7,14,21,28**, objetivo **n=7** y horizonte **h=1 diario**, con representación `relative_window` y diez semillas por combinación. La elección de historia/modelo se cerró con CV de 2024 antes de abrir test. El [cierre previo](../_static/filipe_multihistory_20261104/freeze.json) conserva las decisiones, fuentes y hashes.

La CV primaria utiliza doce meses de 2024, entrenamiento inicial de 2023 y pasado permitido de 2024: **120 trayectorias y 43.920 predicciones**. D7 reutiliza sus treinta trayectorias relativas verificadas; las otras noventa son nuevas. La comparación inicial D7 con `absolute_log` se mantiene como antecedente y no se amplió a otros D.

El cierre seleccionó **MLP, D=7, `relative_window`** por la media de RMSE de CV. Después se realizaron **120 ajustes finales**, se firmaron pesos/escaladores y se evaluó el test **01/01/2025–31/08/2026**, con **608 fechas por corrida**. Los veinte meses del test son intervalos de reporte de esos pesos fijos, no veinte reajustes. Septiembre–octubre de 2026 permanece fuera.

Se publican las métricas requeridas y **BDS por fold/mes para todos los D, modelos y semillas**. El promedio de p-valores es descriptivo. Los contrastes y bootstrap del suplemento inicial D7 conservan su alcance de desarrollo y no se presentan como inferencia sobre el test nuevo.

La publicación ampliada ofrece métricas agregadas por semilla y mes, diagnósticos y manifiestos de trazabilidad. Los vectores de predicción individuales, series de mercado, pesos y paquetes completos se conservan localmente; no se ofrecen como descargas públicas de esta ampliación.

## 5.1. Comparación Tabular

Las comparaciones comparten fuente, calendario, objetivo n=7, horizonte y cohorte dentro de cada fase. **CV y test se muestran por separado**: no se mezclan para seleccionar la configuración ni se interpreta su diferencia como efecto causal de D.

Las tablas completas [de CV](../_static/filipe_multihistory_20261104/analysis_cv/metrics_summary.csv), [de test](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv), [CV por fold](../_static/filipe_multihistory_20261104/analysis_cv/fold_metrics_summary.csv) y [test por mes](../_static/filipe_multihistory_20261104/test/analysis/fold_metrics_summary.csv) conservan media, DE, IC, cobertura y número de valores definidos.

### 5.1.1. Tabla de comparación de los tres modelos base

Cada entrada es $D\times5$ con D=7,14,21,28. El objetivo es σ7, `ddof=0`, sin anualizar; h=1. En CV los índices diarios de Filipe se agrupan en doce meses y se purgan resultados u>t−D−1 en el comienzo mensual t. La validación interna usa únicamente pasado permitido y no recibe etiquetas del mes externo.

Para N fechas de una fase y R=10 semillas se calcula primero:

$$
m_{D,m,s}=\sqrt{\frac1N\sum_{d}(y_d-\widehat y_{D,m,s,d})^2},\qquad
\bar m_{D,m}=\frac1{10}\sum_s m_{D,m,s}.
$$

Se utilizan N=366 en CV y la cohorte común registrada en test. R² se calcula desde la media y sumas de cuadrados del periodo completo, sin promediar R² mensuales. La media entre semillas no equivale a la RMSE de una predicción ensemble. El test emplea pesos finales fijos, con entradas observables hasta d−1.

#### 5.1.1.1. Modelo base

**CV forward de 2024, diez semillas por D–familia.** Los IC son t marginales entre semillas, condicionados al calendario.

| D (días) | Modelo | RMSE, media ± DE | IC marginal 95 % de RMSE | R², media ± DE | Parámetros |
|---:|---|---:|---|---:|---:|
| 7 | MLP | 0,0060491 ± 0,0002780 | [0,0058502; 0,0062479] | 0,615670 ± 0,035544 | 593 |
| 7 | LSTM | 0,0063354 ± 0,0001870 | [0,0062016; 0,0064692] | 0,578892 ± 0,024807 | 5.025 |
| 7 | CNN1D | 0,0062296 ± 0,0003373 | [0,0059883; 0,0064709] | 0,592088 ± 0,042687 | 1.057 |
| 7 | Persistencia | 0,0047890 | No aplica a semillas | 0,759571 | 0 |
| 14 | MLP | 0,0069510 ± 0,0005341 | [0,0065689; 0,0073331] | 0,490785 ± 0,080012 | 1.153 |
| 14 | LSTM | 0,0077018 ± 0,0001999 | [0,0075588; 0,0078448] | 0,377765 ± 0,032441 | 5.025 |
| 14 | CNN1D | 0,0092696 ± 0,0003187 | [0,0090416; 0,0094976] | 0,098246 ± 0,061482 | 1.057 |
| 14 | Persistencia | 0,0047890 | No aplica a semillas | 0,759571 | 0 |
| 21 | MLP | 0,0083825 ± 0,0004626 | [0,0080515; 0,0087134] | 0,261352 ± 0,081648 | 1.713 |
| 21 | LSTM | 0,0087796 ± 0,0003524 | [0,0085275; 0,0090317] | 0,190751 ± 0,063743 | 5.025 |
| 21 | CNN1D | 0,0098231 ± 0,0002269 | [0,0096608; 0,0099854] | -0,012060 ± 0,047010 | 1.057 |
| 21 | Persistencia | 0,0047890 | No aplica a semillas | 0,759571 | 0 |
| 28 | MLP | 0,0093860 ± 0,0008195 | [0,0087998; 0,0099723] | 0,070100 ± 0,161632 | 2.273 |
| 28 | LSTM | 0,0093129 ± 0,0002589 | [0,0091277; 0,0094980] | 0,090149 ± 0,050359 | 5.025 |
| 28 | CNN1D | 0,0099066 ± 0,0002304 | [0,0097418; 0,0100714] | -0,029350 ± 0,047984 | 1.057 |
| 28 | Persistencia | 0,0047890 | No aplica a semillas | 0,759571 | 0 |

Persistencia no tiene pesos ni variación de inicialización. Se repite su etiqueta por D para explicitar el comparador y denominador MASE de cada grupo; **no son nuevas corridas independientes**. La referencia necesita ocho cierres anteriores; D7 recibe siete días OHLCV y D14/21/28 reciben más historia. Ese soporte distinto se declara al interpretar los resultados.

![RMSE y MAPE de CV por historia de entrada](../_static/filipe_multihistory_20261104/analysis_cv/metrics_by_input_history.png)

Las barras indican media ± DE entre semillas; la referencia persistente es determinista.

#### 5.1.1.2. Métricas: media ± desviación estándar

**Test final 01/01/2025–31/08/2026, 608 fechas elegibles por corrida.** Cada fila neuronal resume diez pesos finales ajustados solo con desarrollo y firmados antes de leer test.

| D (días) | Modelo | RMSE, media ± DE | IC marginal 95 % de RMSE | R², media ± DE | Parámetros |
|---:|---|---:|---|---:|---:|
| 7 | MLP | 0,0047932 ± 0,0003988 | [0,0045079; 0,0050785] | 0,764012 ± 0,042175 | 593 |
| 7 | LSTM | 0,0050298 ± 0,0002637 | [0,0048412; 0,0052184] | 0,741108 ± 0,026619 | 5.025 |
| 7 | CNN1D | 0,0044427 ± 0,0003568 | [0,0041874; 0,0046979] | 0,797351 ± 0,033620 | 1.057 |
| 7 | Persistencia | 0,0040856 | No aplica a semillas | 0,829605 | 0 |
| 14 | MLP | 0,0052448 ± 0,0001439 | [0,0051419; 0,0053478] | 0,719001 ± 0,015490 | 1.153 |
| 14 | LSTM | 0,0057094 ± 0,0007524 | [0,0051711; 0,0062476] | 0,662048 ± 0,095907 | 5.025 |
| 14 | CNN1D | 0,0074212 ± 0,0006584 | [0,0069502; 0,0078922] | 0,433815 ± 0,100196 | 1.057 |
| 14 | Persistencia | 0,0040856 | No aplica a semillas | 0,829605 | 0 |
| 21 | MLP | 0,0058746 ± 0,0006002 | [0,0054453; 0,0063040] | 0,644395 ± 0,078879 | 1.713 |
| 21 | LSTM | 0,0061970 ± 0,0008139 | [0,0056148; 0,0067792] | 0,601894 ± 0,106170 | 5.025 |
| 21 | CNN1D | 0,0086565 ± 0,0006164 | [0,0082155; 0,0090975] | 0,231560 ± 0,110033 | 1.057 |
| 21 | Persistencia | 0,0040856 | No aplica a semillas | 0,829605 | 0 |
| 28 | MLP | 0,0074423 ± 0,0004717 | [0,0071049; 0,0077798] | 0,432546 ± 0,073317 | 2.273 |
| 28 | LSTM | 0,0077325 ± 0,0011902 | [0,0068811; 0,0085839] | 0,376622 ± 0,180465 | 5.025 |
| 28 | CNN1D | 0,0088902 ± 0,0004757 | [0,0085499; 0,0092305] | 0,191122 ± 0,086580 | 1.057 |
| 28 | Persistencia | 0,0040856 | No aplica a semillas | 0,829605 | 0 |

Las métricas adicionales requeridas conservan escala original; **MAPE se expresa en porcentaje**:

| D (días) | Modelo | MAE, media ± DE | MAPE (%), media ± DE | MSE, media ± DE | MASE, media ± DE |
|---:|---|---:|---:|---:|---:|
| 7 | MLP | 0,0036400 ± 0,0001764 | 21,5699 ± 0,7252 | 0,000023118 ± 0,000004132 | 1,34304 ± 0,06509 |
| 7 | LSTM | 0,0039189 ± 0,0002196 | 25,3406 ± 3,4056 | 0,000025361 ± 0,000002608 | 1,44595 ± 0,08102 |
| 7 | CNN1D | 0,0034670 ± 0,0003380 | 22,8304 ± 2,5527 | 0,000019852 ± 0,000003293 | 1,27923 ± 0,12470 |
| 7 | Persistencia | 0,0023532 | 13,7553 | 0,000016692 | 0,86825 |
| 14 | MLP | 0,0039266 ± 0,0000828 | 23,0465 ± 0,6938 | 0,000027527 ± 0,000001517 | 1,44893 ± 0,03056 |
| 14 | LSTM | 0,0042049 ± 0,0004001 | 24,6377 ± 1,6635 | 0,000033106 ± 0,000009395 | 1,55164 ± 0,14765 |
| 14 | CNN1D | 0,0055150 ± 0,0004639 | 34,1032 ± 4,8418 | 0,000055464 ± 0,000009815 | 2,03506 ± 0,17120 |
| 14 | Persistencia | 0,0023532 | 13,7553 | 0,000016692 | 0,86834 |
| 21 | MLP | 0,0044246 ± 0,0002915 | 25,4061 ± 2,0566 | 0,000034836 ± 0,000007727 | 1,62350 ± 0,10696 |
| 21 | LSTM | 0,0045288 ± 0,0004498 | 26,6124 ± 3,0421 | 0,000038999 ± 0,000010401 | 1,66172 ± 0,16505 |
| 21 | CNN1D | 0,0063372 ± 0,0003112 | 40,4725 ± 4,0513 | 0,000075277 ± 0,000010779 | 2,32528 ± 0,11419 |
| 21 | Persistencia | 0,0023532 | 13,7553 | 0,000016692 | 0,86343 |
| 28 | MLP | 0,0055578 ± 0,0002967 | 33,3174 ± 3,7242 | 0,000055588 ± 0,000007182 | 2,04064 ± 0,10894 |
| 28 | LSTM | 0,0053131 ± 0,0007067 | 30,8444 ± 6,5454 | 0,000061067 ± 0,000017679 | 1,95080 ± 0,25949 |
| 28 | CNN1D | 0,0064983 ± 0,0003872 | 41,6831 ± 5,8112 | 0,000079239 ± 0,000008481 | 2,38598 ± 0,14216 |
| 28 | Persistencia | 0,0023532 | 13,7553 | 0,000016692 | 0,86401 |

Las tablas [por cada mes](../_static/filipe_multihistory_20261104/test/analysis/fold_metrics_summary.csv) muestran media/DE entre las diez semillas, además de las métricas globales anteriores. La métrica global se recalcula agrupando fechas; no es el promedio simple de los meses. Los CSV guardan cobertura y casos indefinidos; no se añaden epsilons para ocultar denominadores nulos.

![RMSE y MAPE final por historia de entrada](../_static/filipe_multihistory_20261104/test/analysis/metrics_by_input_history.png)

Las medias de RMSE de las doce configuraciones neuronales son mayores que la de persistencia en este test. Es una comparación descriptiva del periodo, sin declarar una inferioridad estadística no contrastada.

#### 5.1.1.3. Intervalos de confianza del 95 % y método de construcción

Para una métrica por semilla, el IC marginal del 95 % es:

$$
\bar m\pm t_{0{,}975;9}\frac{s_m}{\sqrt{10}}.
$$

Este intervalo describe variabilidad del entrenamiento condicionada a las fechas, datos y configuración. No incorpora toda la dependencia temporal, la selección de D ni el cambio de mercado. Persistencia no recibe un IC entre semillas.

El bootstrap circular de fechas del suplemento inicial se conserva exclusivamente para **D7/2024**; no proporciona intervalos temporales para el test o las nuevas historias. Un IC marginal tampoco se convierte en simultáneo porque otro análisis ajuste p-valores con Holm.

#### 5.1.1.4. Tiempo de entrenamiento y de inferencia

Los registros separan entrenamiento CV, ajuste final y puntuación de test. Cada trayectoria CV suma sus 24 ajustes, mientras cada corrida final añade un solo ajuste de desarrollo por la duración cerrada. Se utilizó CPU y se conservaron épocas, hardware, versiones y condiciones de concurrencia.

D7 reutiliza los tiempos de su ejecución original. El [resumen de ajustes finales](../_static/filipe_multihistory_20261104/final_fit_summary.csv) publica las duraciones registradas de esa etapa; los registros locales de CV conservan sus mediciones completas. Las [métricas CV por semilla](../_static/filipe_multihistory_20261104/analysis_cv/run_metrics.csv) y los manifiestos vinculan cada configuración con sus resultados, sin presentar diferencias entre ejecuciones como un benchmark controlado de velocidad. **Test se puntuó una vez por checkpoint**; no se repitió su evaluación para fabricar medidas de latencia.

#### 5.1.1.5. Número de parámetros

Los conteos reales son:

| D | MLP (H=16) | LSTM (H=32) | CNN1D (H=16) |
|---:|---:|---:|---:|
| 7 | 593 | 5.025 | 1.057 |
| 14 | 1.153 | 5.025 | 1.057 |
| 21 | 1.713 | 5.025 | 1.057 |
| 28 | 2.273 | 5.025 | 1.057 |

MLP aumenta el tamaño de la capa de entrada según D: $80D+33$ parámetros. LSTM y CNN1D comparten pesos temporales y conservan sus conteos. Todas mantienen la profundidad y anchura registradas; persistencia no tiene parámetros ajustados.

## 5.2. Ranking de Modelos Base

La selección se realizó con la media anual de RMSE de **CV2024**, no con el test. Se distingue el orden observado del procedimiento elegido previamente y de una superioridad estadística no probada. La ausencia de otra prueba no se sustituye por afirmar que la menor media gana universalmente.

### 5.2.1. Orden de los tres modelos según la métrica principal

| D | Orden neuronal por RMSE de CV | Menor RMSE neuronal | RMSE de persistencia |
|---:|---|---:|---:|
| 7 | MLP → CNN1D → LSTM | 0,0060491 | 0,0047890 |
| 14 | MLP → LSTM → CNN1D | 0,0069510 | 0,0047890 |
| 21 | MLP → LSTM → CNN1D | 0,0083825 | 0,0047890 |
| 28 | LSTM → MLP → CNN1D | 0,0093129 | 0,0047890 |

La mejor historia de cada familia se eligió únicamente con desarrollo:

| Familia | D elegido con CV | RMSE CV, media ± DE | RMSE test, media ± DE | R² test, media ± DE |
|---|---:|---:|---:|---:|
| MLP | 7 | 0,0060491 ± 0,0002780 | 0,0047932 ± 0,0003988 | 0,764012 ± 0,042175 |
| LSTM | 7 | 0,0063354 ± 0,0001870 | 0,0050298 ± 0,0002637 | 0,741108 ± 0,026619 |
| CNN1D | 7 | 0,0062296 ± 0,0003373 | 0,0044427 ± 0,0003568 | 0,797351 ± 0,033620 |

El baseline neuronal cerrado es **MLP, D=7, `relative_window`**, RMSE CV **0,0060491 ± 0,0002780**. El desempate registrado usa tolerancia 10⁻¹², D menor, parámetros y nombre de familia. El test informa el desempeño de esta elección y las otras once configuraciones predeclaradas; un orden diferente en test **no cambia el baseline seleccionado**.

En este procedimiento, las tres familias eligieron D7. Las entradas mayores no mejoraron su media anual de CV; también modificaron purga, muestras admitidas y validación interna, por lo que no se atribuye el cambio exclusivamente a la longitud de contexto.

Las pruebas Holm-12/Friedman y frecuencias de bootstrap del estudio inicial corresponden solo a las redes D7 de 2024. No acreditan diferencias entre estas doce configuraciones o entre sus resultados finales.

### 5.2.2. Análisis del trade-off

Se comparan error, historia, parámetros y costo de entrenamiento. Las tablas permiten evaluar compromisos observados, sin suponer que más contexto o más parámetros deben mejorar el error. La selección sigue la regla registrada de RMSE de CV, no un criterio de costo añadido después del test.

#### 5.2.2.1. Precisión frente a costo computacional

Una ventana mayor puede aumentar cómputo y cambiar el número de épocas elegido. Los tiempos reflejan ambas cosas y la concurrencia local. La CV y el ajuste final tienen alcances distintos; no se suman mediciones de inferencia repetida del estudio inicial a la única pasada de test como si fueran una prueba de despliegue.

#### 5.2.2.2. Precisión frente a complejidad del modelo

Al pasar D7→D28, MLP crece de 593 a 2.273 parámetros; LSTM permanece en 5.025 y CNN1D en 1.057. Una mejora al aumentar D en LSTM/CNN1D cambia contexto sin aumentar pesos, pero también cambia purga y entrenamiento elegible. No se atribuye la diferencia exclusivamente al tamaño de representación sin considerar ese procedimiento completo.

Persistencia sigue siendo una referencia sin pesos y con soporte histórico declarado. Superar solo otra red no equivale a mejorar esa referencia.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

El baseline es una arquitectura, D, representación, regla de selección, épocas, escalado y ajuste final completos. Se seleccionó con CV y quedó firmado antes del test; no es la fila de menor error elegida después de abrirlo.

La referencia neuronal cerrada es **MLP, D=7, `relative_window`**. El test conserva todas las configuraciones predeclaradas y permite comprobar si esta elección mejora o empeora persistencia fuera de desarrollo, sin sustituirla por otra historia al observar el resultado.

### 5.3.1. Selección del baseline de referencia

La selección considera las 366 fechas agrupadas de 2024 y las diez semillas, sin elegir un mes o una inicialización de menor error. El ajuste final conserva la regla de duración definida con diciembre de 2024 y no utiliza early stopping sobre test.

#### 5.3.1.1. Identificación del mejor modelo base

**MLP, D=7, `relative_window`** es el **baseline neuronal seleccionado antes de test**. Mantiene los hiperparámetros de su familia, AdamW, dropout 0,1, weight decay 10⁻⁴ y batch 32. La duración de cada semilla final consta en el cierre.

Su elección se apoya en RMSE media de CV **0,0060491 ± 0,0002780**. La CV utilizó desarrollo ya explorado y no es una búsqueda anidada. Persistencia permanece como referencia contextual fuerte y debe acompañar cualquier afirmación de mejora.

#### 5.3.1.2. Desempeño con media, desviación e IC del 95 %

| Periodo del baseline cerrado | RMSE, media ± DE | IC marginal 95 % | R², media ± DE |
|---|---:|---|---:|
| CV2024 | 0,0060491 ± 0,0002780 | [0,0058502; 0,0062479] | 0,615670 ± 0,035544 |
| Test2025–agosto2026 | 0,0047932 ± 0,0003988 | [0,0045079; 0,0050785] | 0,764012 ± 0,042175 |

Persistencia obtiene RMSE test **0,0040856** y R² **0,829605** sobre las mismas fechas. Su ausencia de DE entre semillas refleja determinismo, no ausencia de incertidumbre temporal. CV y test tienen calendarios y política de ajuste diferentes; no se estima una ganancia causal tomando su diferencia.

### 5.3.2. Cuantificación de las diferencias entre modelos base

Se publican diferencias en unidades originales y porcentajes con denominador y dirección explícitos, manteniendo el mismo objetivo y periodo.

#### 5.3.2.1. Regresión y series de tiempo: diferencias respecto al peor y al mejor modelo

Para una media de error por semilla $\bar M_m$ y referencia r se reporta:

$$
\Delta(m;r)=100\frac{\bar M_r-\bar M_m}{\bar M_r},\qquad \bar M_r>0.
$$

Un valor positivo favorece a m; referencia cero implica razón indefinida. La razón de medias no es la media de razones. La tabla siguiente es **descriptiva del test** y compara dentro del mismo D, sin reseleccionar la configuración:

| D | Red | Reducción RMSE frente a peor red (%) | Reducción frente a mejor red (%) | Reducción frente a persistencia (%) |
|---:|---|---:|---:|---:|
| 7 | MLP | 4,704 | -7,890 | -17,319 |
| 7 | LSTM | 0,000 | -13,216 | -23,110 |
| 7 | CNN1D | 11,673 | 0,000 | -8,740 |
| 14 | MLP | 29,326 | 0,000 | -28,374 |
| 14 | LSTM | 23,067 | -8,857 | -39,744 |
| 14 | CNN1D | 0,000 | -41,495 | -81,643 |
| 21 | MLP | 32,136 | 0,000 | -43,789 |
| 21 | LSTM | 28,412 | -5,487 | -51,679 |
| 21 | CNN1D | 0,000 | -47,354 | -111,879 |
| 28 | MLP | 16,286 | 0,000 | -82,160 |
| 28 | LSTM | 13,022 | -3,899 | -89,263 |
| 28 | CNN1D | 0,000 | -19,454 | -117,598 |

Las diferencias absolutas se obtienen de las métricas en escala original de §5.1; los CSV mantienen precisión sin redondear. El orden puntual del test no constituye un contraste de hipótesis o prueba de equivalencia.

#### 5.3.2.2. Clasificación, etiquetado secuencial y segmentación

**No aplican al proyecto actual:** se pronostica volatilidad continua, sin clases, etiquetas por token o máscaras de imagen. Para una tarea con puntuación S donde mayor sea mejor se definirían:

$$
\Delta_{\rm abs}(m;r)=\bar S_m-\bar S_r,\qquad
\Delta_{\rm rel}(m;r)=100\frac{\bar S_m-\bar S_r}{\bar S_r},\quad\bar S_r>0.
$$

La diferencia en unidades conserva signo; si S se expresa en porcentaje, se distinguen puntos porcentuales de cambio relativo. Con referencia cero, la razón queda indefinida. Estas expresiones no aportan mediciones de clasificación, etiquetado o segmentación del proyecto.

#### 5.3.2.3. Secuencia a secuencia

**No aplica:** la salida es una volatilidad escalar y no una secuencia generada. En una tarea de generación se fijarían métrica, escala, tokenización, referencias y versión antes de comparar BLEU, ROUGE, chrF o exactitud completa. Se mantendrían referentes, parejas y tratamiento de denominadores cero, sin atribuir esas métricas a Bitcoin.

#### 5.3.2.4. Tamaño de efecto de cada diferencia

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Para diferencias pareadas de RMSE por semilla $d_s=m_{A,s}-m_{B,s}$ se reporta:

$$
d_z=\frac{\bar d}{s_d}.
$$

Δ<0 favorece al primer procedimiento cuando se compara error. El denominador es la DE de las diferencias, no una DE marginal. Con dispersión nula se deja indefinido. Los efectos estandarizados y sus diferencias en unidades originales se conservan para las seis comparaciones entre redes y persistencia y para los tres controles de representación.

Los contrastes de ganancia relativa emplean $g_s=1-m_{A,s}/m_{\rm pers}$; allí un valor positivo favorece a la red y la frontera práctica es 0,05. La distancia estandarizada a esa frontera no se confunde con el efecto respecto a cero. Estas medidas describen variación entre inicializaciones sobre un calendario fijo; no sustituyen el margen aplicado ni cuantifican variación entre mercados.

La ampliación publica diferencias de error y dispersión entre semillas, sin añadir una familia de contrastes de desempeño sobre test. No se reutiliza el d_z del D7 inicial como tamaño de efecto de otras entradas o del periodo final.

### 5.3.3. Inferencia sobre las diferencias

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

El análisis reúne comparaciones de las tres redes relativas, persistencia, la frontera práctica del 5 % y los controles de representación. La [familia global](../_static/filipe_statistics_20261104/global_seed_tests.csv) contiene **doce hipótesis por semillas**; Holm global-12 es el ajuste principal de las tablas. Los registros conservan los ajustes de componentes para trazabilidad, sin tratarlos como corroboraciones independientes.

Para el benchmark ampliado y test se reportan métricas, diferencias descriptivas e IC marginales entre semillas. No se generaron nuevos p-valores de superioridad RMSE ni se cambió una prueba al observar test.

#### 5.3.3.1. Comparaciones pareadas y corrección por multiplicidad

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

La familia de doce incluye tres diferencias bilaterales entre redes relativas, tres bilaterales de red frente a persistencia, tres pruebas unilaterales de ganancia relativa superior a 0,05 y los tres controles bilaterales relativa−absoluta. Se utilizan diez pares de semillas base y las mismas 366 fechas. Para persistencia, la puntuación fija se resta de cada ejecución de la red; repetirla algebraicamente no crea diez realizaciones de la referencia.

El complemento y su multiplicidad se registraron antes de calcular estas pruebas, pero después de examinar resultados descriptivos del desarrollo. La t conserva nueve grados de libertad y las comparaciones son exploratorias; no acreditan una selección anidada ni confirmación en datos nuevos.

#### 5.3.3.2. Diferencia media, IC del 95 % y p-value ajustado y sin ajustar

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Diferencias de **RMSE anual media por semilla**, con Δ=primer procedimiento−segundo; Δ<0 favorece al primero. IC del 95 % marginales, t(9):

| Comparación | Δ RMSE | IC 95 % de Δ | d_z | p sin ajustar | p Holm global-12 |
| --- | --- | --- | --- | --- | --- |
| MLP − LSTM | -0,0002863 | [-0,0005087; -0,0000640] | -0,9212 | 0,017225365 | 0,10335219 |
| MLP − CNN1D | -0,0001805 | [-0,0004783; 0,0001173] | -0,4336 | 0,20351666 | 1 |
| MLP − Persistencia | 0,0012601 | [0,0010613; 0,0014589] | 4,5334 | 1,670790e-07 | 1,336632e-06 |
| LSTM − CNN1D | 0,0001058 | [-0,0002203; 0,0004320] | 0,2321 | 0,48167197 | 1 |
| LSTM − Persistencia | 0,0015464 | [0,0014127; 0,0016802] | 8,2693 | 8,441270e-10 | 9,285398e-09 |
| CNN1D − Persistencia | 0,0014406 | [0,0011993; 0,0016819] | 4,2712 | 2,794885e-07 | 1,956420e-06 |

La persistencia tiene menor error en los diez recorridos de cada red y sus diferencias rechazan cero bajo la referencia condicional por semillas. Ninguna pareja de redes rechaza con Holm-12. Los IC marginales no se interpretan como intervalos simultáneos; en particular, un IC marginal puede excluir cero mientras su p ajustado no rechaza.

**Controles de preprocesamiento bajo la misma CV**, integrados en la misma familia global:

| Comparación | Δ RMSE | IC 95 % marginal | d_z | p sin ajustar | p Holm global-12 |
| --- | --- | --- | --- | --- | --- |
| MLP: relativa − absoluta | -0,0043982 | [-0,0046984; -0,0040981] | -10,4826 | 1,019002e-10 | 1,222802e-09 |
| LSTM: relativa − absoluta | -0,0033185 | [-0,0036149; -0,0030221] | -8,0095 | 1,121134e-09 | 1,121134e-08 |
| CNN1D: relativa − absoluta | -0,0044668 | [-0,0050626; -0,0038709] | -5,3624 | 3,871572e-08 | 3,484415e-07 |

Las tres representaciones relativas reducen la RMSE frente a su control absoluto con el ajuste global. El cambio combina representación y ponderación del escalado; no aísla una causa única.

**Pruebas unilaterales de superar el margen práctico**, $H_0:\mathbb E[g_s]\le0,05$ frente a $H_1:\mathbb E[g_s]>0,05$:

| Red | Ganancia media vs persistencia | IC 95 % marginal de ganancia | p sin ajustar | p Holm global-12 |
| --- | --- | --- | --- | --- |
| MLP | -26,313 % | [-30,465; -22,161] % | 0,99999998 | 1 |
| LSTM | -32,292 % | [-35,085; -29,498] % | 1 | 1 |
| CNN1D | -30,082 % | [-35,121; -25,044] % | 0,99999996 | 1 |

Las ganancias anuales son negativas y no se demuestra una mejora superior al 5 %. Los IC anteriores describen la ganancia media; no son por sí solos un procedimiento de equivalencia. Los [resultados completos](../_static/filipe_statistics_20261104/global_seed_tests.csv) incluyen estadísticos t, grados de libertad, Shapiro, precisión numérica y familias. Los IC de ΔR² relativa−absoluta siguen siendo secundarios descriptivos en el [control de representación](../_static/dl_r2_cv_20261104/analysis/comparisons.csv).

### 5.3.4. Umbral de mejora para modelos avanzados

El test permite medir el baseline previamente cerrado, pero no determina automáticamente un margen para una arquitectura avanzada no evaluada. Esa comparación requiere su propia diferencia pareada, variabilidad y margen aplicado, junto con una evaluación posterior independiente si el diseño utiliza los resultados de este test.

#### 5.3.4.1. Mejora mínima detectable frente al baseline

La MDE depende del contraste, potencia objetivo, emparejamiento y dispersión de las diferencias. Para una propuesta avanzada A frente al baseline B:

$$
\operatorname{Var}(m_A-m_B)=\operatorname{Var}(m_A)+\operatorname{Var}(m_B)-2\operatorname{Cov}(m_A,m_B).
$$

La DE o el IC de B aislado no identifican esa varianza. **No se ha dimensionado una MDE de un modelo avanzado que aún no se ha evaluado**, ni se obtiene sumando una DE al baseline neuronal cerrado.

Para los métodos del estudio inicial D7/2024, §5.4.7 calcula sensibilidad orientativa frente a cero usando la dispersión observada entre diez pares. Es una calibración retrospectiva y condicionada al calendario, no un piloto anterior al entrenamiento ni una garantía de potencia. Sus valores no se trasladan a una nueva arquitectura o a la pregunta de superar el 5 % sin definir una alternativa apropiada.

#### 5.3.4.2. Margen de relevancia práctica

El criterio operativo documentado es una **reducción relativa de RMSE del 5 % frente a persistencia**, sobre las mismas fechas y objetivo. Es una referencia descriptiva de relevancia y no una garantía de utilidad financiera; no define automáticamente equivalencia o el margen de un modelo avanzado frente al baseline neuronal.

Un cambio puntual del error superior o inferior al 5 % no es un contraste estadístico de superar ese límite. No se modificó el margen al abrir test; el margen propio de una propuesta avanzada sigue pendiente de justificación.

#### 5.3.4.3. Cota de referencia del problema

Persistencia es una referencia observable: su RMSE de CV2024 es 0,00478897 y en el test de la cohorte del baseline es **0,0040856**. Ninguna es una cota irreducible universal. El acuerdo humano no es una referencia aplicable a esta etiqueta automática y no se estimó una volatilidad latente sin ruido.

La ventana móvil del objetivo, el retorno todavía desconocido del día y el ruido del proveedor deben distinguirse del error reducible de la red. El BDS residual no convierte una diferencia observada en una ganancia alcanzable certificada.

### 5.3.5. Interpretación

Las conclusiones combinan magnitud de error, incertidumbre, costo y alcance del diseño; los p-valores por sí solos no fijan una referencia operacional definitiva.

#### 5.3.5.1. Significación estadística y relevancia práctica

Las diferencias de medias y sus IC de semillas no sustituyen un contraste temporal ni garantizan relevancia económica. El test se evalúa con la configuración cerrada y todas sus semillas; no se escoge una configuración distinta por su error final. Un resultado no significativo tampoco demuestra equivalencia.

#### 5.3.5.2. Desempeño frente a complejidad, parámetros y costo

Las tablas permiten relacionar error final con D y número de pesos. MLP aumenta parámetros con D; LSTM/CNN1D no, aunque su cómputo puede crecer. Los costos se interpretan con duración y concurrencia registradas, sin añadir una utilidad de despliegue no medida.

#### 5.3.5.3. Conclusión explícita del baseline y margen disponible

El baseline neuronal **seleccionado antes de test** es **MLP, D=7, `relative_window`**; su RMSE final es **0,0047932 ± 0,0003988**. La reducción puntual frente a persistencia es **-17,319 %**; un signo negativo indica mayor error de la red.

Esa medida es descriptiva del test y no prueba estadísticamente una ganancia superior al 5 %. Una propuesta avanzada deberá justificar su margen y comparar el procedimiento completo, manteniendo la referencia persistente y una confirmación independiente posterior.

#### 5.3.5.4. Estado del benchmark

**Completado:** entradas D7/14/21/28, CV efectiva, selección cerrada con desarrollo, 120 ajustes finales, tablas de test y BDS por fold/mes para todas las semillas. Se conservan cohortes, hashes, resultados indefinidos y orden de eventos.

**Pendiente específico:** apoyo aplicado de MLP/CNN1D, piloto prospectivo, dimensionamiento definitivo, márgenes de equivalencia y de propuesta avanzada. Abrir el test no resuelve estos puntos ni crea otra reserva desconocida.

## 5.4. Análisis Estadístico y Pruebas de Hipótesis

La ampliación incorpora **BDS por fold de CV y por mes de reporte del test**. Cada fase conserva su familia planificada y p-valores sin ajustar/con Holm. La media de p-valores se publica solo como descripción.

Los contrastes de desempeño por semillas, DM, bootstrap y Ljung–Box del suplemento inicial se conservan en los incisos correspondientes con alcance explícito **D7/2024**. No se añadieron contrastes RMSE de test ni se trasladaron familias antiguas a otro conjunto de resultados. Estas familias no constituyen un control único de toda la investigación.

### 5.4.1. Hipótesis, unidad de análisis y supuestos

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Para error, $d_s=\mathrm{RMSE}_{A,s}-\mathrm{RMSE}_{B,s}$ y la hipótesis bilateral es $H_0:\mathbb E[d_s]=0$; Δ<0 favorece a A. Para mejora práctica, $g_s=1-\mathrm{RMSE}_{{\rm red},s}/\mathrm{RMSE}_{\rm pers}$ y se contrasta unilateralmente superar la frontera 0,05. Los controles de representación conservan el orden relativa−absoluta.

La unidad de las pruebas por semillas es la aleatorización de entrenamiento condicionada a datos, configuración y calendario compartidos. Las doce particiones mensuales son dependientes y no crean doce mercados. La t exige un diseño de pares defendible y una distribución apropiada de sus diferencias; Shapiro es un diagnóstico, sin escoger retrospectivamente la prueba que produzca significación.

### 5.4.2. Comparaciones pareadas: t y Wilcoxon

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

La t pareada utiliza diez diferencias y nueve grados de libertad. Se conservan estadístico, IC marginal, p sin ajustar y p Holm global-12. No se sustituyó por otra prueba según el resultado observado.

Wilcoxon no se ejecutó. Su eventual aplicación debe fijar hipótesis, ceros, empates, referencia exacta o aproximada y supuestos; no contrasta automáticamente la misma media que la t. Su resolución discreta y diez semillas no garantizan potencia suficiente.

### 5.4.3. Comparación conjunta de tres modelos

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se ejecutó **Friedman sobre diez bloques-semilla completos** de las tres redes relativas, condicionado al calendario de 2024. Se obtuvo estadístico **4,2000**, con referencia asintótica χ² de dos grados de libertad y **p=0,12245643**. Los rangos medios fueron MLP=1,5, LSTM=2,4 y CNN1D=2,1.

No se rechaza su hipótesis a α=0,05; esto no demuestra equivalencia ni igualdad de desempeño fuera de desarrollo. Se declara la referencia aproximada y el número reducido de bloques. El contraste global no se utilizó como filtro para ejecutar las comparaciones pareadas registradas. Fuente: [summary.json](../_static/filipe_statistics_20261104/summary.json).

### 5.4.4. Comparaciones post-hoc y corrección por multiplicidad

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Holm global-12 ajusta conjuntamente los seis contrastes bilaterales entre métodos relativos/persistencia, los tres unilaterales de ganancia >5 % y los tres controles de representación. La tabla principal utiliza ese ajuste; los CSV retienen los ajustes de componentes para trazabilidad, sin convertirlos en pruebas adicionales independientes.

No se ejecutó Nemenyi ni ANOVA. Las comparaciones pareadas estaban especificadas en el complemento y no quedaron condicionadas al rechazo de Friedman. DM utiliza su familia primaria de seis comparaciones de pérdidas por fecha, y los diagnósticos su familia de quince; no se mezclan sus estimandos ni se exige una segunda corrección equivalente sobre un post-hoc ya simultáneo.

### 5.4.5. Intervalos de confianza y tamaño de efecto

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se reportan IC t de diferencias entre semillas y percentiles de un bootstrap temporal de fechas. Ambos son marginales y condicionan aspectos diferentes del procedimiento. Las diferencias en unidades de RMSE se priorizan frente a estandarizaciones aisladas; para cada resultado se identifica el signo y la unidad.

#### 5.4.5.1. Bootstrap de las diferencias pareadas

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se aplicó **bootstrap circular pareado de fechas**, 2.000 réplicas, L=14 principal y sensibilidad L=7/28, semilla 20261104. Los índices de cada réplica son comunes a los cuatro procedimientos. Se recalcula RMSE por semilla y luego su media, y la razón de mejora usa el mismo denominador remuestreado. No se concatena semilla×fecha como observaciones independientes.

IC percentiles marginales del 95 % con L14:

| Comparación | IC de diferencia RMSE | IC de ganancia relativa |
| --- | --- | --- |
| MLP − LSTM | [-0,0006277; 0,0000720] | [-1,291; 9,828] % |
| MLP − CNN1D | [-0,0008532; 0,0003781] | [-6,368; 13,062] % |
| MLP − Persistencia | [0,0003343; 0,0022573] | [-50,797; -6,548] % |
| LSTM − CNN1D | [-0,0007565; 0,0007734] | [-13,093; 11,627] % |
| LSTM − Persistencia | [0,0005281; 0,0026421] | [-59,589; -10,508] % |
| CNN1D − Persistencia | [0,0003919; 0,0024583] | [-56,008; -7,287] % |

Los IC entre redes incluyen cero. Los IC marginales red−persistencia excluyen cero con signo de mayor error de la red. Estos IC no están ajustados por multiplicidad y su funcional es la media de RMSE; DM/Holm-6 contrasta otra pregunta, sobre pérdidas cuadráticas y una familia simultánea (§5.4.9).

![Intervalos temporales de diferencias de RMSE](../_static/filipe_statistics_20261104/figures/temporal_bootstrap_intervals.png)

Los [18 intervalos](../_static/filipe_statistics_20261104/block_bootstrap.csv) conservan las tres longitudes, sin elegir la favorable. El remuestreo condiciona a los modelos guardados: **no remuestrea semillas ni reentrena** sobre historias nuevas. Su calibración depende de una representación adecuada de dependencia y estabilidad local; las etiquetas solapadas y cambios mensuales de red limitan una lectura universal.

#### 5.4.5.2. Tamaño de efecto junto con el p-value

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Las diferencias de RMSE, IC t marginales y d_z se reportan para los pares de métodos y controles de representación (§5.3.3). El signo sigue primer método−segundo para error; una ganancia relativa mantiene su dirección propia. La DE es la de diferencias emparejadas.

No se calculó delta de Cliff ni rango-biserial; no se presentan como equivalentes automáticos a d_z. Los tamaños grandes entre inicializaciones no describen variabilidad entre mercados ni garantizan relevancia económica.

### 5.4.6. Advertencia sobre independencia

Compartir calendario y ventanas genera dependencia que debe distinguirse de la variación de inicialización.

#### 5.4.6.1. Ejecuciones que comparten el conjunto de prueba

Las diez corridas de cada configuración comparten las mismas fechas de CV y, después, las mismas fechas de test. Las semillas representan variabilidad del entrenamiento, no diez mercados. Sus IC no cuantifican toda la variación temporal entre periodos.

Las σ7 consecutivas comparten seis retornos; los meses de CV comparten entrenamiento y los de test conservan pesos finales. No se concatenan semilla×fecha o mes×semilla como observaciones independientes. Persistencia es un único pronóstico determinista por fecha.

#### 5.4.6.2. Validación repetida y remuestreo por bloques

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se ejecutó forward chaining creciente con doce bloques mensuales y horizonte diario, manteniendo pesos dentro de cada mes. No se utilizó K-fold aleatorio ni múltiples particiones aleatorias repetidas.

El bootstrap de fechas usa bloques L14 y sensibilidad L7/L28. Aun así, no reproduce volver a entrenar sobre otras trayectorias de mercado. No se aplica una corrección de varianza derivada de otro diseño como si reparara selección, dependencia o fuga de información. Las 366 fechas, diez semillas y doce bloques conservan sus significados distintos.

### 5.4.7. Reporte de potencia y diferencia mínima detectable

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Las diez semillas constituyen un presupuesto exploratorio. Se calcula ahora **sensibilidad de planificación con dispersión retrospectiva**, condicionada al calendario, para mostrar lo que ese tamaño puede distinguir bajo una aproximación normal. No se presenta como piloto anterior al entrenamiento, potencia observada a partir del efecto obtenido o certificado de suficiencia.

Para $\alpha_{\rm plan}=0,05/12$ y objetivo hipotético $1-\beta=0,80$:

$$
\operatorname{MDE}_0\approx
\frac{(z_{1-\alpha_{\rm plan}/2}+z_{0,80})\widehat\sigma_d}{\sqrt R},\qquad
R_{\rm orient}=\left\lceil
\frac{(z_{1-\alpha_{\rm plan}/2}+z_{0,80})^2\widehat\sigma_d^2}{\delta_{\rm plan}^2}
\right\rceil.
$$

Para red frente a persistencia, $\delta_{\rm plan}=0,05\,\mathrm{RMSE}_{\rm pers}=0,0002394485$ es el efecto a detectar **frente a cero**, sin afirmar que se supera el 5 %:

| Red vs persistencia | DE de diferencia | MDE0 con R=10 | δplan frente a cero | R orientativo |
| --- | --- | --- | --- | --- |
| MLP | 0,0002780 | 0,0003258 | 0,0002394485 | 19 |
| LSTM | 0,0001870 | 0,0002192 | 0,0002394485 | 9 |
| CNN1D | 0,0003373 | 0,0003954 | 0,0002394485 | 28 |

Bajo estos supuestos y esta calibración, diez pares quedan por debajo del R orientativo de MLP y CNN1D; el valor orientativo de LSTM no acredita un piloto prospectivo ni potencia exacta. La dispersión se estima con solo diez pares y tiene incertidumbre. Para pares entre redes se publican MDE0, pero no R orientativo porque no se definió su margen propio.

Fuente: [design_sensitivity.csv](../_static/filipe_statistics_20261104/design_sensitivity.csv). No se dimensiona aquí demostrar ganancia >5 % sin una alternativa $q_{\rm alt}>0,05$, ni un modelo avanzado aún inexistente. Dimensionar aleatorizaciones de entrenamiento tampoco dimensiona la incertidumbre temporal o las fechas del test.

### 5.4.8. Ausencia de significación y equivalencia

No rechazar una diferencia no demuestra equivalencia. **No se ejecutó TOST ni ROPE**, y falta un margen de equivalencia propio.

TOST requiere una banda práctica predefinida y ambas pruebas unilaterales; con α=0,05 sin otro ajuste, su regla usual corresponde al IC del 90 % dentro de la banda. Ese intervalo no se confunde con los IC del 95 % de reporte. ROPE exige un modelo bayesiano, priors, posterior y regla de decisión. El 5 % de mejora sobre persistencia no define automáticamente una banda de equivalencia entre modelos ni se ensancha después para obtener una conclusión.

### 5.4.9. Test de Diebold–Mariano

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se ejecutó Diebold–Mariano para las **seis parejas** de las tres redes relativas y persistencia, sobre las mismas 366 fechas. El estimando y la corrección se fijaron en el complemento exploratorio. Fuente: [temporal_dm.csv](../_static/filipe_statistics_20261104/temporal_dm.csv).

#### 5.4.9.1. Comparación de precisión predictiva por horizonte

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Para h=1 se contrasta la media de diferencias de pérdida cuadrática por fecha:

$$
d_t=\frac1R\sum_{s=1}^{R}\left(e_{A,s,t}^2-e_{B,s,t}^2\right),\qquad H_0:\mathbb E[d_t]=0.
$$

Persistencia aporta una trayectoria determinista; las redes aportan la media de pérdidas de las diez semillas. No es la pérdida de un ensemble ni la media de RMSE por trayectoria: promediar raíces difiere de promediar pérdidas. Tampoco se elige la semilla de menor error.

El signo negativo favorece al primer método en pérdida cuadrática. Los cambios mensuales de parámetros, etiquetas solapadas y posibles cambios de régimen limitan las referencias estacionarias de la prueba; se conservan como evidencia exploratoria de este calendario.

#### 5.4.9.2. Corrección Harvey–Leybourne–Newbold y pérdida coherente

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se utilizó HAC de Bartlett con **rezago 14 principal** y sensibilidades 7/28, horizonte h=1 y corrección Harvey–Leybourne–Newbold. Para varianza de largo plazo positiva:

$$
DM=\frac{\sqrt T\,\bar d}{\sqrt{\widehat\Omega_d}},\qquad
DM_{\rm HLN}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}.
$$

Aquí T=366 y el factor HLN es $\sqrt{(T-1)/T}$; h no se sustituye por n=7 o D=7. Se usa referencia t de Student con 365 grados de libertad, aproximada, y Holm sobre las seis comparaciones principales:

| Comparación | DM HLN, HAC14 | p sin ajustar | p Holm-6 |
| --- | --- | --- | --- |
| MLP − LSTM | -1,53375 | 0,12595787 | 0,37787361 |
| MLP − CNN1D | -0,59704 | 0,55084908 | 1 |
| MLP − Persistencia | 2,39216 | 0,017254737 | 0,069018947 |
| LSTM − CNN1D | 0,24639 | 0,80551675 | 1 |
| LSTM − Persistencia | 2,64512 | 0,0085190747 | 0,051114448 |
| CNN1D − Persistencia | 2,55372 | 0,011063495 | 0,055317477 |

**Ninguna pareja rechaza tras Holm-6 a α=0,05.** Aunque las tres comparaciones red−persistencia presentan p nominales inferiores a 0,05, sus valores ajustados son superiores. No se sustituye el ajuste por el resultado nominal ni se elige otra banda HAC para lograr rechazo.

Las sensibilidades HAC7/28 se conservan descriptivamente en el CSV; no reabren la selección del análisis principal. La corrección HLN no resuelve por sí sola no estacionariedad, selección o todas las formas de dependencia.

### 5.4.10. Test BDS (Brock–Dechert–Scheinkman)

Se ejecutó **BDS sobre residuos de h=1 por cada fold/mes, D=7/14/21/28, modelo y las diez semillas**, con dimensiones de incrustación **2 y 3** y radio **1,5 DE muestral** del residuo del bloque. El residuo es observado−predicho.

CV: **2.880 filas planificadas, 2.880 p-valores definidos**; test: **4.800 filas planificadas, 4.800 definidos**.

Se publican [BDS por fold de CV](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold.csv), [resumen por fold de CV](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold_summary.csv), [BDS por mes de test](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold.csv) y [resumen por mes de test](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold_summary.csv). Los resúmenes [de CV](../_static/filipe_multihistory_20261104/analysis_cv/bds_pvalues_descriptive_summary.csv) y [de test](../_static/filipe_multihistory_20261104/test/analysis/bds_pvalues_descriptive_summary.csv) conservan media/DE de p, cantidad planificada/definida y estados. Los casos degenerados o no finitos quedan registrados y no se sustituyen por cero.

**Resumen descriptivo de p por D, familia y dimensión**, sobre las semillas y meses de cada fase; los valores por cada fold se conservan en los enlaces anteriores:

| D | Red | Dimensión | p medio ± DE, CV | p medio ± DE, test | Rechazos Holm CV/test |
|---:|---|---:|---:|---:|---:|
| 7 | MLP | 2 | 0,187543 ± 0,298378 | 0,076018 ± 0,189438 | 54 de 120 ; 97 de 200 |
| 7 | MLP | 3 | 0,187511 ± 0,299019 | 0,094188 ± 0,207507 | 50 de 120 ; 100 de 200 |
| 7 | LSTM | 2 | 0,104282 ± 0,216751 | 0,050000 ± 0,184918 | 55 de 120 ; 135 de 200 |
| 7 | LSTM | 3 | 0,115311 ± 0,220061 | 0,051976 ± 0,158720 | 48 de 120 ; 124 de 200 |
| 7 | CNN1D | 2 | 0,189190 ± 0,289672 | 0,105102 ± 0,221707 | 54 de 120 ; 84 de 200 |
| 7 | CNN1D | 3 | 0,216587 ± 0,305149 | 0,118528 ± 0,232420 | 54 de 120 ; 76 de 200 |
| 14 | MLP | 2 | 0,057291 ± 0,193246 | 0,041902 ± 0,134103 | 76 de 120 ; 121 de 200 |
| 14 | MLP | 3 | 0,064782 ± 0,182154 | 0,043533 ± 0,130718 | 66 de 120 ; 115 de 200 |
| 14 | LSTM | 2 | 0,086290 ± 0,195357 | 0,017436 ± 0,095400 | 73 de 120 ; 147 de 200 |
| 14 | LSTM | 3 | 0,114850 ± 0,233369 | 0,024050 ± 0,104575 | 69 de 120 ; 133 de 200 |
| 14 | CNN1D | 2 | 0,040308 ± 0,151474 | 0,004959 ± 0,055291 | 95 de 120 ; 184 de 200 |
| 14 | CNN1D | 3 | 0,036687 ± 0,144666 | 0,015195 ± 0,095313 | 98 de 120 ; 179 de 200 |
| 21 | MLP | 2 | 0,031146 ± 0,128393 | 0,034867 ± 0,126915 | 93 de 120 ; 143 de 200 |
| 21 | MLP | 3 | 0,046430 ± 0,150166 | 0,028493 ± 0,113573 | 92 de 120 ; 141 de 200 |
| 21 | LSTM | 2 | 0,093804 ± 0,200501 | 0,012820 ± 0,064395 | 80 de 120 ; 153 de 200 |
| 21 | LSTM | 3 | 0,123288 ± 0,262537 | 0,023574 ± 0,104464 | 76 de 120 ; 142 de 200 |
| 21 | CNN1D | 2 | 0,000347 ± 0,003615 | 0,017463 ± 0,099971 | 116 de 120 ; 191 de 200 |
| 21 | CNN1D | 3 | 0,004778 ± 0,037973 | 0,015312 ± 0,096886 | 114 de 120 ; 191 de 200 |
| 28 | MLP | 2 | 0,044668 ± 0,153850 | 0,010747 ± 0,086350 | 88 de 120 ; 186 de 200 |
| 28 | MLP | 3 | 0,053219 ± 0,168621 | 0,008909 ± 0,061053 | 90 de 120 ; 181 de 200 |
| 28 | LSTM | 2 | 0,063567 ± 0,181633 | 0,013675 ± 0,078342 | 84 de 120 ; 179 de 200 |
| 28 | LSTM | 3 | 0,065453 ± 0,182395 | 0,020768 ± 0,108517 | 83 de 120 ; 176 de 200 |
| 28 | CNN1D | 2 | 0,000531 ± 0,004030 | 0,008418 ± 0,061554 | 113 de 120 ; 191 de 200 |
| 28 | CNN1D | 3 | 0,002424 ± 0,023429 | 0,007675 ± 0,048608 | 109 de 120 ; 191 de 200 |

Los conteos de rechazo son pruebas individuales bajo la familia de su fase, no réplicas independientes ni una estimación de probabilidad universal.

Cada fase conserva la familia planificada de sus **2.880 pruebas CV o 4.800 pruebas test**, con Holm y α=0,05. El promedio de p-valores solicitado es **solo descriptivo: no es una prueba combinada ni permite declarar independencia global**.

Los bloques mensuales contienen 28–31 residuos y las referencias asintóticas son frágiles con esas muestras pequeñas. El solapamiento de σ7, cambios de régimen y dependencia lineal también pueden producir rechazo. BDS no demuestra exclusivamente no linealidad aprovechable ni cuantifica una mejora futura.

### 5.4.11. Test de Ljung–Box

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Se ejecutó Ljung–Box sobre los mismos residuos de la semilla 4101, con rezagos **7, 14 y 28** y `model_df=0`. No se descuentan 593, 5.025 o 1.057 pesos como si fueran órdenes ARMA. Las referencias χ² son aproximadas para errores externos de redes reajustadas mensualmente.

| Red | p Holm-15, rezago 7 | p Holm-15, rezago 14 | p Holm-15, rezago 28 |
| --- | --- | --- | --- |
| MLP | 4,242223e-92 | 1,153786e-89 | 2,008099e-101 |
| LSTM | 6,532286e-71 | 1,724077e-66 | 3,641182e-81 |
| CNN1D | 2,076805e-88 | 4,728694e-86 | 3,017740e-100 |

Los nueve contrastes rechazan ausencia conjunta de autocorrelación hasta sus rezagos. Se conservan estadísticos Q, grados de libertad y p sin ajustar en el CSV. La evidencia no distingue por sí sola entre solapamiento de σ7, errores de especificación, cambios de régimen u otras fuentes. No se realizó aquí una prueba ARCH ni un contraste separado sobre residuos al cuadrado.

### 5.4.12. Interpretación de los residuos

Se analizan residuos con signo observado−predicho, conservando fecha, D, familia, semilla y bloque. La estructura residual limita interpretar errores como i.i.d. y no determina por sí sola cuál arquitectura avanzada funcionará.

El BDS mensual ampliado tiene una referencia aproximada con muestras pequeñas. Ljung–Box y ACF del suplemento inicial se refieren a D7/2024; no se los presenta como ejecutados para los otros D o test. No se añadió una prueba específica de heterocedasticidad condicional.

#### 5.4.12.1. Estructura residual y posible margen de mejora

**Alcance de esta evidencia:** el suplemento siguiente corresponde exclusivamente a la CV inicial **D=7, año 2024**. Sus contrastes no se trasladan a D=14/21/28 ni al test final.

Para la semilla designada, con signo residual **observado−predicho**:

| Red relativa | Media residual | DE residual | RMSE de esta semilla |
| --- | --- | --- | --- |
| MLP | 0,0010027 | 0,0064693 | 0,0065378 |
| LSTM | -0,0006784 | 0,0065510 | 0,0065772 |
| CNN1D | 0,0003229 | 0,0063742 | 0,0063736 |

Estas cifras corresponden a una sola inicialización, no a las medias anuales de diez semillas de §5.1. La descomposición por todas las trayectorias continúa disponible en [bias_decomposition.csv](../_static/dl_r2_cv_20261104/analysis/bias_decomposition.csv); allí el sesgo se define como predicho−observado, con signo opuesto al residuo usado en esta tabla.

![Autocorrelación residual, semilla 4101](../_static/filipe_statistics_20261104/figures/residual_acf_seed4101.png)

![Distribución de residuos, semilla 4101](../_static/filipe_statistics_20261104/figures/residual_distribution_seed4101.png)

ACF, Ljung–Box y BDS muestran estructura remanente, pero **no cuantifican una mejora futura alcanzable** ni prueban una causa arquitectónica. Los modelos avanzados deben demostrar su aprovechamiento con un diseño posterior y una evaluación separada.

![Curvas internas reales del último bloque mensual](../_static/filipe_statistics_20261104/figures/internal_epoch_curves_december_seed4101.png)

El ejemplo de parada corresponde a diciembre de 2024, fijado en el complemento, semilla 4101: mejores épocas **25/24/99** de MLP/LSTM/CNN1D, con **37/36/100** épocas registradas. CNN1D llega al máximo de 100 en ese corte; no se declara convergencia garantizada. Son curvas por época en subentrenamiento y validación interna, no curvas frente a tamaño muestral ni pérdidas del test. La inversión de escala permite comparar entrenamiento y validación dentro de cada ajuste.

### 5.4.13. Buenas Prácticas Estadísticas

Las decisiones, fuentes, código y fechas de cierre distinguen desarrollo exploratorio de evaluación final. Se conservan resultados favorables/desfavorables y estados indefinidos. El test no se utiliza para cambiar D, hiperparámetros, épocas o semillas después de puntuarlo.

#### 5.4.13.1. Nivel de significancia declarado antes de analizar resultados

Se fijó **α=0,05**. BDS conserva por separado las familias planificadas CV (2.880) y test (4.800), con Holm. Las medias de p-valores no se contrastan con α como si fueran p combinados.

El suplemento inicial D7 mantiene Holm-12 para semillas, Holm-6 para DM principal y Holm-15 para sus diagnósticos. Son análisis de otro alcance, no confirmaciones independientes de las nuevas tablas. El protocolo de ampliación y el cierre anteceden a la apertura del test, pero son posteriores al desarrollo ya conocido.

#### 5.4.13.2. P-valores numéricos, ajustados y sin ajustar

Los CSV publican p-valores BDS sin ajustar y con Holm, estadístico, dimensión, radio, tamaño, logaritmos cuando corresponden y estado. Un p no definido conserva su motivo; un valor que desborda la precisión numérica no se interpreta como probabilidad matemática exactamente cero. La media/DE se identifica como resumen descriptivo.

Los p-valores t/DM/Ljung–Box iniciales conservan su conjunto D7/2024 y no se reutilizan para otras entradas o test.

#### 5.4.13.3. Intervalos de confianza y tamaños de efecto

Las tablas nuevas incluyen IC t marginales entre diez semillas, condicionados al calendario. Los efectos d_z y bootstrap de diferencias del suplemento inicial siguen referidos a D7/2024. No se calculó una nueva familia confirmatoria de diferencias RMSE de test ni una banda de equivalencia.

Un ajuste Holm en BDS no convierte los IC de desempeño en simultáneos. Los IC de semillas tampoco cubren automáticamente incertidumbre temporal o selección del modelo.

#### 5.4.13.4. Verificación de supuestos

Se verificaron cronología, continuidad, máscaras, escaladores, firmas y métricas. El BDS por mes registra T, dimensión, radio y casos no definidos. Con 28–31 observaciones y volatilidades solapadas, su referencia asintótica es exploratoria, no una certificación de independencia.

Los diagnósticos Shapiro/Q–Q, ACF y Ljung–Box del estudio inicial conservan sus fechas y D7. No se declara normalidad de las nuevas diferencias a partir de ese examen ni se escoge una prueba retrospectiva para producir significación.

#### 5.4.13.5. Múltiples ejecuciones documentadas

Se conservan las **diez semillas 4101–4110** en cada D–familia. La CV primaria cuenta 120 trayectorias; el ajuste final cuenta 120 modelos nuevos. Ninguna semilla se excluye por error. Los 1.440 segmentos de CV y veinte intervalos de test no son semillas adicionales o mercados independientes. Diez repeticiones no certifican potencia suficiente.

#### 5.4.13.6. Métrica principal y comparaciones declaradas con antelación

El protocolo fijó RMSE principal, σ7/ddof0, h1, cuatro entradas, representación relativa y diez semillas. La selección se cerró con la media anual de CV, desempate registrado y épocas de diciembre. El [cierre previo](../_static/filipe_multihistory_20261104/freeze.json) antecede a lectura de test.

El análisis BDS por fase/fold y sus parámetros se registraron. No se cambió el margen del 5 %, se eligió un semestre favorable ni se seleccionó una configuración a partir de test. La exposición previa del desarrollo sigue declarada.

#### 5.4.13.7. Registro y trazabilidad de cada contraste

| Registro ampliado | Contenido |
|---|---|
| [Protocolo](../_static/filipe_multihistory_20261104/protocol.json) | D, objetivo, CV, semillas, selección, ajuste final y análisis |
| [Cierre](../_static/filipe_multihistory_20261104/freeze.json) | Decisiones y hashes previos a abrir test |
| [Métricas CV](../_static/filipe_multihistory_20261104/analysis_cv/metrics_summary.csv) | Desempeño anual de cuatro D, DE e IC entre semillas |
| [Métricas test](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv) | Desempeño final sobre cohorte registrada |
| [CV por fold](../_static/filipe_multihistory_20261104/analysis_cv/fold_metrics_summary.csv) | Métricas por mes y dispersión de semillas |
| [Test por mes](../_static/filipe_multihistory_20261104/test/analysis/fold_metrics_summary.csv) | Desglose de los mismos pesos finales, sin reajuste |
| [BDS CV](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold.csv) | Pruebas planificadas, estadístico, parámetros, p y estados |
| [BDS test](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold.csv) | Diagnósticos mensuales del test, con multiplicidad |

Los [resultados del suplemento inicial](../_static/filipe_statistics_20261104/summary.json) permanecen reproducibles exclusivamente para D7/2024. Wilcoxon, Nemenyi, TOST y ROPE no se ejecutaron; no se sustituyen con tablas inventadas. DeLong, McNemar, segmentación y generación de lenguaje no aplican al objetivo continuo.

### 5.4.14. Interpretación

Las tablas nuevas describen el procedimiento cerrado en CV y test; los p-valores BDS evalúan una hipótesis sobre residuos y no comparan precisión entre modelos. La selección del baseline precede al test. Un menor error medio o un rechazo residual no equivale a superioridad estadística o utilidad práctica demostrada.

#### 5.4.14.1. Diferencias estadísticamente significativas

No se ejecutó una nueva familia de contrastes de superioridad RMSE sobre test. Los rankings de las doce configuraciones son descriptivos y no permiten declarar un ganador significativo entre arquitecturas.

Los contrastes iniciales D7/2024 conservaron sus resultados: ninguna comparación neuronal rechazó con Holm-12 y ningún DM principal con Holm-6. Ese resultado no se extrapola a las otras entradas o al periodo final. Los BDS ampliados se interpretan según su propia familia, estados y referencia aproximada.

#### 5.4.14.2. Significancia estadística y relevancia práctica

El baseline cerrado cambia puntualmente el RMSE de persistencia en **-17,319 %** durante test. Un valor negativo significa mayor error neuronal. El margen operativo del 5 % conserva su valor previo; no se redefine para favorecer el resultado.

La cifra no es una prueba de superar ese margen y no acredita utilidad financiera. No rechazar una diferencia tampoco demuestra equivalencia, cuya banda aplicada sigue pendiente.

#### 5.4.14.3. Conclusiones respaldadas por diferencias, incertidumbre y efectos

Se informa media, dispersión, IC y periodo de cada configuración, junto con la referencia persistente y BDS por bloque. Las diferencias entre CV y test pueden reflejar mercado, tamaño de entrenamiento y actualización distinta; no se atribuyen a una causa única.

El test ya evaluado no se reutiliza como confirmación desconocida si sus resultados orientan nuevos cambios. La evidencia inicial por semillas/fechas y la ampliación final se conservan con su alcance y sin escoger solo la conclusión favorable.

#### 5.4.14.4. Estado del análisis estadístico

**Ejecutado:** cuatro entradas, diez semillas por familia, CV real, cierre previo, ajustes finales, test, métricas globales/por mes y BDS por fold para todas las semillas. El suplemento inicial D7 conserva contrastes y diagnósticos adicionales sobre desarrollo.

**Pendiente específico:** piloto prospectivo/dimensionamiento definitivo, márgenes de equivalencia y de modelo avanzado, apoyo aplicado MLP/CNN1D y pruebas de robustez no ejecutadas. No se acredita potencia 0,80 con dispersión retrospectiva ni se añade una superioridad de test no contrastada.

## 5.5. Discusión Crítica

La discusión utiliza la selección previa de desarrollo, la evaluación final y los diagnósticos residuales con sus límites. Las doce configuraciones de test se publican completas; el procedimiento elegido no cambia porque otra fila tenga menor error final.

### 5.5.1. Desempeño y explicación de las diferencias

El desarrollo seleccionó **MLP, D=7, `relative_window`**, con RMSE CV **0,0060491 ± 0,0002780**. En test obtiene **0,0047932 ± 0,0003988**. Los demás D y familias se mantienen visibles para examinar el orden observado, sin llamarlo selección confirmatoria posterior.

La menor RMSE neuronal puntual del test corresponde a **CNN1D, D=7**, con **0,0044427**. Ese resultado posterior no sustituye la selección previa de CV. Ninguna de las doce configuraciones tiene menor RMSE media de test que persistencia.

MLP aplana el contexto; LSTM incorpora estado recurrente y CNN1D comparte filtros causales locales con promedio temporal. Son sesgos inductivos descritos, no mecanismos causales de victoria probados. Cambiar D modifica contexto y purga; en MLP además cambia parámetros. Ninguna ventaja se atribuye exclusivamente a una de esas diferencias.

### 5.5.2. Generalización y estabilidad

El test permite evaluar fuera del calendario de desarrollo el procedimiento previamente cerrado, con pesos fijos y todas las semillas. No acredita generalización universal a otros activos, fuentes o regímenes; se trata de un único tramo temporal de Bitcoin.

La estabilidad se examina en las tablas mensuales. Una media global positiva de R² no excluye meses negativos; su promedio mensual tampoco sustituye el R² global. La comparación con CV incorpora una política de actualización distinta: modelos mensuales crecientes en 2024 frente a pesos finales fijos durante test.

### 5.5.3. Significación estadística y relevancia práctica

La diferencia puntual del baseline cerrado frente a persistencia en test es **-17,319 %**. El margen operativo del 5 % se interpreta sin cambiar su signo o denominador. Las tablas de semillas no son por sí mismas un contraste temporal de superar ese margen.

Los p-valores BDS identifican desviaciones respecto a i.i.d. bajo una referencia aproximada; su promedio no es una prueba global ni una comparación de precisión. Los resultados inferenciales iniciales D7/2024 mantienen su alcance y no convierten el nuevo ranking en superioridad estadística demostrada.

### 5.5.4. Desempeño frente a costo y complejidad

Los cuatro D permiten contrastar error observado con tamaño de entrada y pesos. MLP pasa de 593 a 2.273 parámetros; LSTM/CNN1D mantienen 5.025/1.057. Se incluyen duraciones locales con sus condiciones, sin convertirlas en evaluación de despliegue.

La persistencia no requiere pesos entrenados y conserva su soporte histórico declarado. Un aumento de complejidad debe acompañarse de una mejora útil y reproducible; más parámetros por sí solos no la demuestran.

### 5.5.5. Baseline y umbral para un modelo avanzado

El baseline neuronal previo a test es **MLP, D=7, `relative_window`**, RMSE final **0,0047932 ± 0,0003988**, IC marginal entre semillas **[0,0045079; 0,0050785]**. La referencia persistente obtiene RMSE **0,0040856** en las mismas fechas.

Un modelo avanzado necesita un margen aplicado y diseño propios, además de considerar persistencia. Si su diseño utiliza estos resultados finales, deberá confirmarse en otro conjunto independiente. La sensibilidad MDE retrospectiva del D7 inicial no dimensiona automáticamente esa nueva comparación.

### 5.5.6. Brecha con la literatura y margen de mejora potencial

Las diferencias publicadas en la literatura sirven como contexto cuando se declaran objetivos, escala, fuente y partición. No se trasladan a la σ7 diaria de este proyecto como una ganancia esperada. El corpus mantiene apoyo aplicado pendiente para MLP y CNN1D simple; la ejecución propia no completa esa evidencia ni convierte la CNN en una réplica de la TCN publicada.

### 5.5.7. Amenazas a la validez

| Amenaza | Control/evidencia | Límite |
|---|---|---|
| Interna | Índices reales de Filipe, purgas D, escaladores train, firma previa al test | Desarrollo ya conocido; sin búsqueda anidada |
| Externa | Test posterior a desarrollo, periodo y cohorte documentados | Un activo/proveedor; no asegura otro régimen |
| De construcción | σ7/ddof0, h1, soporte de entradas y persistencia explícitos | Proxy solapado; ruido de mercado/proveedor no separado |
| De conclusión | Diez semillas, IC marginales, BDS con familias planificadas y estados | Dependencia temporal; bloques BDS pequeños; no superioridad RMSE de test contrastada |

Los meses CV comparten historia y los meses test comparten pesos finales; ninguno crea mercados independientes. El cierre demuestra el orden de decisiones respecto al test, no ausencia de toda exposición anterior de desarrollo. El test ya consultado no podrá usarse como nueva reserva desconocida para cambios orientados por él.

### 5.5.8. Limitaciones del estudio

El estudio se limita a BTCUSDT, un proveedor, objetivo n=7, horizonte de un día, cuatro historias y diez semillas por combinación. El test tiene pesos finales fijos; otra política de actualización sería otro procedimiento. No se investigaron ruido simulado, otros objetivos, horizontes o activos.

BDS mensual usa 28–31 residuos y una referencia asintótica sensible a muestras pequeñas, solapamiento y no estacionariedad. Los p no definidos se conservan; su promedio es descriptivo. Los IC entre semillas son marginales, no intervalos completos de incertidumbre temporal.

Siguen abiertos el respaldo aplicado de MLP/CNN1D, piloto prospectivo, potencia definitiva y márgenes de equivalencia/avanzado. No se modificaron objetivo, fechas, fórmula de R² o configuración al observar el test.

### 5.5.9. Recomendaciones futuras

Las siguientes acciones derivan de fallos y límites observados; no se promete obtener un p-valor favorable o un R² predeterminado.

#### 5.5.9.1. Recomendaciones derivadas de las limitaciones

Completar el apoyo aplicado de las dos familias pendientes y definir una comparación avanzada con margen y piloto propios. Investigar meses difíciles y estructura residual mediante hipótesis registradas, manteniendo información causal y una referencia persistente adecuada.

Si los hallazgos del test orientan cambios de D, arquitectura o actualización, evaluarlos después en otro tramo independiente. Más semillas no sustituyen otros periodos/activos. La dependencia residual no garantiza que añadir complejidad produzca una ganancia determinada.

#### 5.5.9.2. Matriz de conclusiones y evidencia pendiente

| Pregunta | Evidencia actual | Conclusión o límite |
|---|---|---|
| ¿Se completaron las entradas? | D7/14/21/28, mismos modelos y objetivo | Sí; n=7 y h1 permanecen fijos |
| ¿Qué baseline se eligió? | CV anual, diez semillas y cierre previo | MLP, D=7; no elegido con test |
| ¿Cómo rindió fuera de desarrollo? | Tablas del test con 608 fechas | RMSE del baseline 0,0047932 ± 0,0003988; alcance a este periodo |
| ¿Se supera persistencia? | Misma cohorte y métrica | Cambio puntual -17,319 %; no prueba de superioridad estadística |
| ¿Se calculó BDS por fold? | Todos los D/modelos/semillas y dimensiones 2/3 | Sí; p medios descriptivos y referencias frágiles con T mensual |
| ¿Qué variación se reporta? | Media/DE/IC entre semillas; meses identificados | Inicialización condicional; no nuevos mercados |
| ¿Qué falta? | Limitaciones y corpus | Apoyo aplicado, piloto/potencia prospectivos, márgenes futuros y otra confirmación si se diseña usando test |
