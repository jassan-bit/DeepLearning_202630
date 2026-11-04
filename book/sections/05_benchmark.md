# 5. Benchmark de los Modelos Base y Definición del Baseline

El benchmark utiliza **validación forward chaining con `timeseries-cv`**, proyecto de DidierRLopes desarrollado con coautoría de Filipe R. Ramos. Las tres redes son **MLP, LSTM y CNN1D poco profunda**. La representación principal es `relative_window`; `absolute_log` se conserva como control de preprocesamiento dentro de la misma CV.

Se ejecutaron **doce bloques mensuales de 2024**, con entrenamiento inicial de 2023 e incorporación del pasado permitido de 2024. Cada trayectoria reúne 366 predicciones diarias de horizonte h=1. Las diez semillas por red y variante producen **60 trayectorias, 720 segmentos modelo–mes y 1.440 ajustes efectivos**: selección interna de duración y reajuste nuevo en cada segmento. No son 366 reajustes diarios ni pronósticos de un mes adelante.

El [método de CV](../_static/dl_r2_cv_20261104/method.json) y las [configuraciones](../_static/dl_r2_cv_20261104/configuration.json), los [folds](../_static/dl_r2_cv_20261104/folds.csv) y las [corridas](../_static/dl_r2_cv_20261104/runs.csv) permiten comprobar el procedimiento. La configuración se fijó usando desarrollo ya explorado: la evaluación de 2024 es **exploratoria**, sin selección anidada ni confirmación externa. El ajuste final con desarrollo completo y el test de enero de 2025–agosto de 2026 siguen pendientes; septiembre–octubre de 2026 permanece sin asignar al test.

**Resultado anual:** MLP relativo tiene la menor RMSE entre las redes y las tres redes relativas obtienen R² medio positivo. **Persistencia conserva menor error en todo 2024.** Las cifras siguientes identifican periodo, representación y unidad de incertidumbre.

## 5.1. Comparación Tabular

Las tablas utilizan las mismas fuentes BTCUSDT de Binance Spot, calendario UTC, objetivo y cohortes dentro de cada comparación.

### 5.1.1. Tabla de comparación de los tres modelos base

La entrada es una ventana **D=7 días × cinco canales OHLCV**, el objetivo es volatilidad poblacional de siete retornos diarios, sin anualizar y con `ddof=0`, y el horizonte es **h=1**. En cada comienzo mensual b se admiten etiquetas de entrenamiento hasta $u\le b-D-1$. La ventana interna de noventa días antes de b deja **83 etiquetas después de la purga**; sus escaladores se ajustan solo con el subentrenamiento. El reajuste completo usa escaladores propios y toda la cohorte mensual admitida.

La función `split_train_val_forwardChaining(np.arange(731),7,1,1)` proporciona los índices diarios. El adaptador los reúne en doce bloques mensuales y agrega controles explícitos de embargo y calidad. El entrenador consume esos índices para ajustar cada red. Los pesos quedan fijos durante el mes, mientras cada pronóstico diario usa OHLCV observable de los días anteriores.

Para fechas $\mathcal V$ y semillas $\mathcal S$, la puntuación principal es:

$$
m_{m,s}=\sqrt{\frac1N\sum_{t\in\mathcal V}(Y_t-\widehat Y_{m,s,t})^2},\qquad
S_m=\frac1R\sum_{s\in\mathcal S}m_{m,s}.
$$

Primero se reúnen las 366 predicciones anuales de cada trayectoria y se calcula su RMSE; después se promedian las diez semillas. No se puntúa un ensemble ni se mezclan semillas×fechas como observaciones independientes. El R² anual también se calcula desde las 366 predicciones, **sin promediar R² diarios o mensuales**.

#### 5.1.1.1. Modelo base

**Representación principal `relative_window`, 1 de enero–31 de diciembre de 2024.** Cada fila neuronal resume diez trayectorias completas. Los IC t del 95 % son marginales entre semillas y condicionan el calendario.

| Modelo | RMSE, media ± DE | IC 95 % de RMSE | MAE, media ± DE | R², media ± DE | Parámetros |
| --- | --- | --- | --- | --- | --- |
| MLP | 0,0060491 ± 0,0002780 | [0,0058502; 0,0062479] | 0,0043804 ± 0,0002110 | 0,615670 ± 0,035544 | 593 |
| LSTM | 0,0063354 ± 0,0001870 | [0,0062016; 0,0064692] | 0,0047263 ± 0,0001245 | 0,578892 ± 0,024807 | 5.025 |
| CNN1D | 0,0062296 ± 0,0003373 | [0,0059883; 0,0064709] | 0,0046780 ± 0,0002212 | 0,592088 ± 0,042687 | 1.057 |
| Persistencia | 0,0047890 | No aplica a semillas | 0,0029435 | 0,759571 | 0 |

Fuente: [metrics_summary.csv](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv). Persistencia es una referencia determinista, sin pesos ni variabilidad de inicialización; no se cuenta como diez entrenamientos ni como un cuarto modelo DL. Necesita ocho cierres para obtener la última σ7 conocida, mientras las redes reciben siete días OHLCV. Esa diferencia de información histórica observable se conserva en la interpretación.

![R² anual de ambos preprocesamientos con la misma CV](../_static/dl_r2_cv_20261104/analysis/figures/r2_cv_full2024.png)

Las barras representan DE, no IC. La línea de persistencia no tiene variabilidad entre semillas.

#### 5.1.1.2. Métricas: media ± desviación estándar

Todas las métricas se expresan después de invertir la escala positiva de Y. RMSE es la principal; las siguientes complementan la lectura de la representación relativa:

| Modelo | IC 95 % de R² | MAPE (%), media ± DE | MASE, media ± DE |
| --- | --- | --- | --- |
| MLP | [0,590243; 0,641096] | 19,3303 ± 0,7369 | 1,66168 ± 0,08422 |
| LSTM | [0,561147; 0,596638] | 21,7863 ± 0,8045 | 1,78982 ± 0,04645 |
| CNN1D | [0,561551; 0,622624] | 21,9803 ± 1,0000 | 1,77956 ± 0,08783 |
| Persistencia | No aplica a semillas | 13,5521 | 1,11835 |

MAPE y MASE tienen cobertura del 100 %. MASE emplea en cada mes una escala calculada únicamente con su entrenamiento admisible, registrada por predicción; no se espera MASE=1 para persistencia en las fechas de validación. R² positivo significa menor MSE que la referencia algebraica de la media observada del periodo, sin implicar superioridad frente a persistencia.

**Control de representación dentro de la misma CV anual.** Ambos preprocesamientos comparten modelos, configuraciones, folds y diez semillas:

| Representación CV | Red | RMSE, media ± DE | R², media ± DE |
| --- | --- | --- | --- |
| `absolute_log` | MLP | 0,0104473 ± 0,0005993 | -0,147616 ± 0,130856 |
| `absolute_log` | LSTM | 0,0096539 ± 0,0004292 | 0,021226 ± 0,087144 |
| `absolute_log` | CNN1D | 0,0106964 ± 0,0008499 | -0,206245 ± 0,191124 |
| `relative_window` | MLP | 0,0060491 ± 0,0002780 | 0,615670 ± 0,035544 |
| `relative_window` | LSTM | 0,0063354 ± 0,0001870 | 0,578892 ± 0,024807 |
| `relative_window` | CNN1D | 0,0062296 ± 0,0003373 | 0,592088 ± 0,042687 |

`absolute_log` usa logaritmos de precios y volumen, con escala por canal sobre días únicos de entrenamiento. `relative_window` expresa OHLC respecto al último cierre observable de cada ventana y ajusta escalas sobre posiciones de ventanas de entrenamiento. Esta comparación cambia representación y ponderación del escalado; no aísla causalmente una sola transformación.

**Subperiodo julio–diciembre: 184 fechas de la misma CV.** Se informa por separado del año, sin intercambiar denominadores ni cohortes:

| Red relativa | RMSE, media ± DE | R², media ± DE |
| --- | --- | --- |
| MLP | 0,0049062 ± 0,0002224 | 0,657808 ± 0,032033 |
| LSTM | 0,0052429 ± 0,0002265 | 0,609298 ± 0,033741 |
| CNN1D | 0,0046732 ± 0,0001429 | 0,689848 ± 0,018888 |
| Persistencia | 0,0048225 | 0,669995 |

Fuente del subperiodo: [metrics_common_h2_cv_only.csv](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2_cv_only.csv). CNN1D relativa obtiene una reducción puntual de RMSE del **3,095 %** frente a persistencia, inferior al margen operativo del 5 %. El orden semestral no se extiende al año completo ni se interpreta como superioridad temporal confirmada.

![R² de los dos preprocesamientos con CV en julio–diciembre](../_static/dl_r2_cv_20261104/analysis/figures/r2_common_h2_cv_only.png)

#### 5.1.1.3. Intervalos de confianza del 95 % y método de construcción

Los IC marginales entre semillas emplean t de Student con **nueve grados de libertad**:

$$
\bar m\pm t_{0{,}975;9}\frac{s_m}{\sqrt{10}}.
$$

Describen la media de desempeño del procedimiento de entrenamiento condicionada a los datos, configuración y fechas observadas. Los IC de modelos aislados no sustituyen el IC de su diferencia emparejada. Aplicar Holm a p-valores tampoco convierte estos IC en intervalos simultáneos.

El bootstrap temporal por bloques y los contrastes de pérdidas por fecha **no se han ejecutado** para estas predicciones. Su diseño requiere unidad, longitud de bloque, sensibilidad y supuestos apropiados (§5.4); los intervalos entre semillas no los reemplazan.

#### 5.1.1.4. Tiempo de entrenamiento y de inferencia

Los tiempos corresponden a una trayectoria anual completa: entrenamiento suma el ajuste interno y reajuste de los doce bloques; inferencia suma las mediciones de las 366 predicciones. Se usaron CPU, un hilo PyTorch por trabajador y tres trabajadores concurrentes.

| Representación | Red | Entrenamiento de 12 bloques (s), media ± DE | Inferencia de 366 casos (ms), media ± DE |
| --- | --- | --- | --- |
| `absolute_log` | MLP | 9,189 ± 1,617 | 0,6397 ± 0,0896 |
| `absolute_log` | LSTM | 12,010 ± 1,738 | 3,7515 ± 0,6540 |
| `absolute_log` | CNN1D | 17,661 ± 2,955 | 4,0847 ± 0,3203 |
| `relative_window` | MLP | 21,352 ± 1,234 | 0,6227 ± 0,0413 |
| `relative_window` | LSTM | 18,505 ± 2,305 | 3,2121 ± 0,3623 |
| `relative_window` | CNN1D | 44,350 ± 2,425 | 3,6905 ± 0,2351 |

La inferencia se midió por bloques después de cinco calentamientos y con veinte pasadas, lote máximo 256. Entrenamiento incluye la evaluación de parada; las medidas excluyen preprocesamiento y serialización. Son tiempos locales, sensibles a concurrencia y número de épocas, sin una evaluación de despliegue. Fuentes: [runs.csv](../_static/dl_r2_cv_20261104/runs.csv) y [runtime](../_static/dl_r2_cv_20261104/runtime.json).

#### 5.1.1.5. Número de parámetros

Los conteos de instancias reales son **593 parámetros entrenables en MLP, 5.025 en LSTM y 1.057 en CNN1D**; no hay parámetros no entrenables. Persistencia no tiene pesos ajustados.

MLP tiene una capa oculta de 16 unidades; LSTM una capa recurrente unidireccional de 32; CNN1D dos convoluciones causales ordinarias de 16 canales y kernel 3, seguidas de promedio temporal. Los dos preprocesamientos mantienen esas arquitecturas. La CNN propia no reproduce la TCN publicada en la literatura.

## 5.2. Ranking de Modelos Base

El ranking se define por la media de RMSE por trayectoria, con representación y periodo identificados. Es descriptivo de esta CV de desarrollo.

### 5.2.1. Orden de los tres modelos según la métrica principal

En **todo 2024**, el orden dentro de `relative_window` es **MLP, CNN1D, LSTM**. En `absolute_log` es LSTM, MLP, CNN1D. La persistencia anual tiene menor RMSE que las seis combinaciones.

En julio–diciembre, el orden relativo es CNN1D, MLP, LSTM; esta diferencia respecto al año exige identificar el periodo. No se ejecutaron contrastes entre las tres arquitecturas para certificar el ranking. Los tres contrastes disponibles comparan representaciones dentro de cada red. No se interpreta un IC solapado como equivalencia ni una posición descriptiva como un ganador del test.

### 5.2.2. Análisis del trade-off

Se comparan error, parámetros y costos bajo la misma CV. La dominancia sobre medias describe estas estimaciones y no garantiza dominancia poblacional ni utilidad financiera.

#### 5.2.2.1. Precisión frente a costo computacional

La representación relativa reduce RMSE con los mismos parámetros, pero exige más tiempo medio de entrenamiento que la absoluta en las tres redes. Las diferencias de duración incorporan épocas seleccionadas y concurrencia, sin demostrar eficiencia arquitectónica intrínseca.

Entre las redes relativas, MLP tiene la menor RMSE anual y la menor duración de inferencia; LSTM tiene menor tiempo medio de entrenamiento. Las condiciones de §5.1.1.4 se mantienen y no se inventa una restricción de latencia operacional.

#### 5.2.2.2. Precisión frente a complejidad del modelo

MLP relativa combina el menor error anual neuronal con el menor conteo de parámetros. LSTM utiliza aproximadamente 8,47 veces sus pesos y CNN1D 1,78 veces, sin obtener menor error anual. La ventaja semestral de CNN1D no modifica esa comparación anual.

Persistencia obtiene mejor desempeño anual sin pesos aprendidos, con la diferencia de información histórica observable indicada. Más parámetros o mayor tiempo no se consideran una mejora por sí mismos.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

El baseline se expresa como una arquitectura y procedimiento completos: fuentes, objetivo, representación, entrenamiento creciente, reglas de parada y calendario. Se distingue la referencia neuronal descriptiva de la confirmación final pendiente.

### 5.3.1. Selección del baseline de referencia

La elección descriptiva se apoya en el desempeño anual completo y no en seleccionar el mes o la semilla de menor error.

#### 5.3.1.1. Identificación del mejor modelo base

**MLP con `relative_window`, D=7, H=16 y learning rate=0,001 es el candidato a baseline DL por menor RMSE anual media.** Comparte AdamW, dropout 0,1, weight decay 10⁻⁴, batch 32, parada interna y reajuste mensual de §4.

La configuración completa y el margen para una propuesta avanzada deberán cerrarse antes del ajuste final y del test. La CV usa desarrollo ya explorado; no acredita selección anidada o generalización externa. Persistencia conserva el papel de referencia contextual fuerte, con menor error anual que las redes.

#### 5.3.1.2. Desempeño con media, desviación e IC del 95 %

MLP relativa obtiene RMSE **0,0060491 ± 0,0002780**, IC t del 95 % **[0,0058502; 0,0062479]**, y R² **0,615670 ± 0,035544**, sobre diez trayectorias y 366 fechas. Persistencia obtiene RMSE **0,0047890** y R² **0,759571**; no tiene dispersión de inicialización.

La lectura es condicional al calendario de 2024. No se sustituye el año por el semestre favorable a una arquitectura ni se presenta este candidato como una cota universal de error.

### 5.3.2. Cuantificación de las diferencias entre modelos base

Se publican diferencias en unidades originales y porcentajes con denominador y dirección explícitos, manteniendo el mismo objetivo y periodo.

#### 5.3.2.1. Regresión y series de tiempo: diferencias respecto al peor y al mejor modelo

Para medias de error por trayectoria $\bar M_m$, una reducción relativa frente a r es:

$$
\Delta(m;r)=100\frac{\bar M_r-\bar M_m}{\bar M_r},\qquad\bar M_r>0.
$$

Un valor positivo favorece a m; si el denominador es cero, se reporta indefinido sin epsilon. La razón de medias no es la media de razones por semilla. Mejor y peor se identifican por métrica, no se trasladan automáticamente de RMSE a MAE.

En la representación relativa anual, MLP y LSTM son los extremos mejor y peor entre redes por RMSE. Los porcentajes siguientes usan datos sin redondear:

| Red relativa | Reducción RMSE frente a LSTM (%) | Reducción frente a MLP (%) | Reducción frente a su variante absoluta (%) | Reducción frente a persistencia anual (%) |
| --- | --- | --- | --- | --- |
| MLP | 4,520 | 0,000 | 42,099 | -26,313 |
| LSTM | 0,000 | -4,734 | 34,375 | -32,292 |
| CNN1D | 1,670 | -2,984 | 41,760 | -30,082 |

Todos los valores anuales frente a persistencia son negativos: las redes tienen mayor error. En el semestre julio–diciembre, CNN1D relativa obtiene una reducción puntual del 3,095 %, inferior al 5 % y sin prueba temporal de superioridad ejecutada.

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

Para diferencias pareadas por semilla $d_s=m_{{\rm rel},s}-m_{{\rm abs},s}$ se reporta:

$$
d_z=\frac{\bar d}{s_d}.
$$

Un valor negativo favorece la representación relativa; el denominador es la DE de las diferencias, sin sustituirla por una DE marginal. Si $s_d=0$, el efecto estandarizado queda indefinido. El tamaño de efecto entre inicializaciones condiciona el mismo calendario; no mide variación entre mercados ni reemplaza el margen aplicado. Los efectos de los tres contrastes están en §5.3.3.

### 5.3.3. Inferencia sobre las diferencias

La inferencia ejecutada evalúa únicamente el cambio de preprocesamiento dentro de cada arquitectura bajo el mismo forward chaining.

#### 5.3.3.1. Comparaciones pareadas y corrección por multiplicidad

Se ejecutaron **tres contrastes bilaterales**, uno por red, de RMSE(`relative_window`)−RMSE(`absolute_log`), con las diez mismas semillas base y las mismas 366 fechas. La corrección Holm se aplica conjuntamente a los tres, sin dividir por signo o resultado.

No se ejecutaron pruebas entre arquitecturas ni contra persistencia. La exposición del desarrollo y el diseño exploratorio limitan el alcance: causalidad temporal de cada fold no convierte toda la selección en confirmación independiente.

#### 5.3.3.2. Diferencia media, IC del 95 % y p-value ajustado y sin ajustar

Δ<0 favorece `relative_window`. La t y los IC marginales se refieren a diez diferencias de RMSE por trayectoria, con nueve grados de libertad:

| Red | Δ RMSE relativa − absoluta | IC 95 % marginal | t(9) | d_z | p sin ajustar | p Holm, familia 3 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | -0,00439825 | [-0,00469839; -0,00409810] | -33,14874 | -10,48255 | 1.019002e-10 | 3.057005e-10 |
| LSTM | -0,00331851 | [-0,00361490; -0,00302213] | -25,32833 | -8,00952 | 1.121134e-09 | 2.242268e-09 |
| CNN1D | -0,00446676 | [-0,00506264; -0,00387089] | -16,95733 | -5,36238 | 3.871572e-08 | 3.871572e-08 |

Los tres contrastes favorecen la representación relativa bajo esa referencia y su alcance condicionado al calendario. Shapiro–Wilk sobre las diferencias produce p=0,425763, 0,237020 y 0,573491 para MLP, LSTM y CNN1D; no rechazarlo no demuestra normalidad. Los efectos grandes describen variación entre estas inicializaciones, sin acreditar utilidad financiera.

El cambio de R² se informa como intervalo secundario descriptivo, sin otra familia de pruebas seleccionada por esa métrica:

| Red | Δ R² relativa − absoluta | IC 95 % marginal descriptivo |
| --- | --- | --- |
| MLP | 0,763286 | [0,687570; 0,839002] |
| LSTM | 0,557666 | [0,498273; 0,617059] |
| CNN1D | 0,798333 | [0,665593; 0,931073] |

Fuente: [comparisons.csv](../_static/dl_r2_cv_20261104/analysis/comparisons.csv). Los cambios de representación y ponderación de escalas se estudian conjuntamente. Estos contrastes no demuestran superar el 5 % frente a persistencia ni generalización temporal.

### 5.3.4. Umbral de mejora para modelos avanzados

El margen práctico y la diferencia mínima detectable son cantidades distintas; ambas requieren definición para la comparación concreta que vaya a realizarse.

#### 5.3.4.1. Mejora mínima detectable frente al baseline

La MDE depende del contraste, potencia objetivo, emparejamiento y variabilidad de las diferencias. Para una propuesta avanzada A frente al baseline B:

$$
\operatorname{Var}(m_A-m_B)=\operatorname{Var}(m_A)+\operatorname{Var}(m_B)-2\operatorname{Cov}(m_A,m_B).
$$

La DE o el IC de B aislado no identifican esa varianza. **No existe una MDE numérica dimensionada para la CV actual o para una propuesta avanzada aún no evaluada.** Se requiere un piloto apropiado al procedimiento creciente y al estimando, o supuestos de planificación justificados; no se obtiene el umbral sumando una DE al baseline.

#### 5.3.4.2. Margen de relevancia práctica

El criterio operativo aceptado es una **reducción relativa de RMSE de al menos 5 % frente a persistencia**, sobre las mismas fechas. No se interpreta como utilidad financiera calibrada ni se transfiere automáticamente a una comparación entre redes o de un modelo avanzado frente al candidato MLP.

En todo 2024 ninguna red supera a persistencia; en julio–diciembre la reducción puntual de CNN1D relativa es 3,095 %, inferior a ese margen. Alcanzar un porcentaje puntual tampoco demostraría que la mejora excede el margen con incertidumbre. Para la propuesta avanzada falta fijar su propio margen y diseñar la comparación antes de evaluarla.

#### 5.3.4.3. Cota de referencia del problema

Persistencia es una referencia contextual fuerte y observable, con RMSE anual 0,0047890. Su error no constituye una cota irreducible universal. El acuerdo humano no es una referencia aplicable a esta etiqueta automática y no se ha estimado una volatilidad latente sin ruido.

No se deduce de estos datos una cota certificada de desempeño alcanzable. La ventana móvil del objetivo, el retorno del día aún desconocido y el ruido del proveedor deben distinguirse del error reducible del modelo.

### 5.3.5. Interpretación

Las conclusiones combinan magnitud de error, incertidumbre, costo y alcance del diseño; los p-valores por sí solos no fijan una referencia operacional definitiva.

#### 5.3.5.1. Significación estadística y relevancia práctica

Los tres contrastes relativa–absoluta son significativos bajo Holm-3 y la referencia t utilizada. Esa evidencia condiciona el calendario e inicializaciones; no demuestra superioridad temporal de una red sobre otra ni cumplimiento del margen práctico frente a persistencia.

#### 5.3.5.2. Desempeño frente a complejidad, parámetros y costo

La representación relativa mejora error anual sin añadir pesos. MLP relativa combina el menor RMSE neuronal y los menos parámetros, mientras LSTM relativa necesita menos tiempo medio de entrenamiento. El mayor costo de CNN1D no se traduce en el menor error anual. Los compromisos son descriptivos de la CV y de la CPU registrada.

#### 5.3.5.3. Conclusión explícita del baseline y margen disponible

El **candidato a baseline DL es MLP con precios relativos y forward chaining mensual**, por menor RMSE media anual. Persistencia continúa como la referencia contextual de menor error anual y no se ha demostrado una mejora de las redes del 5 % sobre ella.

El procedimiento final y el margen de una propuesta avanzada quedan por cerrar. El resultado de desarrollo no establece un umbral universal ni autoriza elegir retrospectivamente el periodo más favorable.

#### 5.3.5.4. Estado del benchmark

Se dispone de tres redes implementadas, dos preprocesamientos comparables, diez trayectorias por combinación, 366 fechas, costos, parámetros, IC marginales y tres contrastes emparejados. Los [registros agregados](../_static/dl_r2_cv_20261104/runs.csv) y las [predicciones](../_static/dl_r2_cv_20261104/predictions.csv) enlazan los 720 segmentos mensuales.

Permanecen pendientes el dimensionamiento específico, la inferencia temporal, el cierre del procedimiento y margen avanzado, el ajuste final y el test. La ejecución no completa el respaldo bibliográfico aplicado de MLP/CNN1D simple.

## 5.4. Análisis Estadístico y Pruebas de Hipótesis

El análisis ejecutado consiste en IC t por semillas y tres contrastes pareados de RMSE relativa–absoluta con Holm. Los demás procedimientos de los incisos siguientes se documentan como métodos y pendientes, sin atribuirles resultados inexistentes de esta CV.

### 5.4.1. Hipótesis, unidad de análisis y supuestos

Para cada arquitectura, con diez pares de trayectorias, $d_s=\mathrm{RMSE}_{{\rm rel},s}-\mathrm{RMSE}_{{\rm abs},s}$, se contrasta bilateralmente $H_0:\mathbb E[d_s]=0$. Un valor negativo favorece el procedimiento relativo.

La unidad inferencial es la aleatorización de entrenamiento condicionada al calendario y configuración compartidos. Las doce particiones mensuales son dependientes y no son doce series independientes. Las condiciones de la t incluyen un diseño de semillas defendible y una distribución de diferencias apropiada; la normalidad no se declara probada por un diagnóstico aislado.

### 5.4.2. Comparaciones pareadas: t y Wilcoxon

La t pareada fue el procedimiento utilizado, con nueve grados de libertad y sin cambiar de prueba según el p-valor observado. Wilcoxon no se ejecutó. Si se usa después, deberán fijarse hipótesis, tratamiento de ceros/empates, referencia exacta o aproximada y supuestos, incluida simetría para una interpretación de desplazamiento.

Wilcoxon no contrasta automáticamente la misma media que la t. Ni su resolución discreta ni disponer de diez semillas garantizan potencia suficiente.

### 5.4.3. Comparación conjunta de tres modelos

No se ejecutó un contraste conjunto entre MLP, LSTM y CNN1D. El ranking anual de §5.2 es descriptivo.

Si se aplica Friedman, cada bloque debe contener resultados comparables de las tres redes bajo una unidad defendible. Debe identificarse la referencia exacta, de permutación o aproximada y respetarse la dependencia. Un contraste sobre semillas condicionaría el calendario; no aportaría mercados nuevos ni resolvería por sí solo la incertidumbre temporal.

### 5.4.4. Comparaciones post-hoc y corrección por multiplicidad

La familia ejecutada contiene las tres comparaciones de representaciones dentro de arquitectura y usa **Holm con α=0,05**. No se realizó Nemenyi, ANOVA ni post-hoc entre las redes.

Si se añaden hipótesis de arquitecturas, variantes, periodos o horizontes, la familia y su procedimiento deberán definirse antes del análisis correspondiente. No se dividirán familias para conservar significación ni se añadirá Holm automáticamente como segunda corrección de un post-hoc ya simultáneo.

### 5.4.5. Intervalos de confianza y tamaño de efecto

Los intervalos por semillas y los efectos pareados disponibles se presentan con sus supuestos. La incertidumbre temporal requiere un diseño separado.

#### 5.4.5.1. Bootstrap de las diferencias pareadas

**Pendiente para estas predicciones.** Un bootstrap temporal deberá remuestrear fechas o bloques conjuntamente entre los procedimientos y recalcular métricas y razones con los mismos índices, respetando calendario y dependencia. La longitud, las sensibilidades, las réplicas y la semilla se justificarán y registrarán.

Remuestrear predicciones almacenadas condiciona a los modelos realizados: no reproduce automáticamente la variación de reentrenar sobre historias remuestreadas. Tampoco se mezclarán días dependientes como si fueran i.i.d. o se elegirá la longitud por su resultado favorable.

#### 5.4.5.2. Tamaño de efecto junto con el p-value

Los tres contrastes informan diferencias en unidades originales, IC t marginales y d_z pareado (§5.3.3). Se mantiene el signo relativa−absoluta y la DE de las diferencias. No se calculó delta de Cliff ni rango-biserial; una medida alternativa deberá conservar el emparejamiento y declarar ceros/empates.

### 5.4.6. Advertencia sobre independencia

Compartir calendario y ventanas genera dependencia que debe distinguirse de la variación de inicialización.

#### 5.4.6.1. Ejecuciones que comparten el conjunto de prueba

El inciso expresa una precaución general: **la ejecución actual comparte validación de desarrollo, no el test reservado**. Las diez trayectorias de cada combinación utilizan las mismas fechas, etiquetas y reglas. Sus IC no cuantifican la variación entre periodos de mercado independientes.

Las σ7 consecutivas comparten seis retornos. Las predicciones mantienen fecha, red, variante, semilla y fold; no se concatenan todas las filas como observaciones independientes. Persistencia se trata como un único pronóstico determinista.

#### 5.4.6.2. Validación repetida y remuestreo por bloques

Se ejecutó forward chaining creciente con **doce bloques mensuales**, pesos constantes dentro de cada mes y horizonte diario. No se utilizó K-fold aleatorio ni se ejecutaron múltiples particiones aleatorias repetidas.

El remuestreo por bloques y una corrección de varianza requieren adecuación al diseño creciente y al solapamiento del objetivo. No se aplica una corrección derivada para otro esquema como si garantizara independencia. El bootstrap temporal sigue pendiente.

### 5.4.7. Reporte de potencia y diferencia mínima detectable

Las diez semillas son un **presupuesto exploratorio**, sin un piloto específico que acredite potencia 0,80. No se publica una MDE o potencia retrospectiva calculada con el efecto observado para certificar suficiencia.

Un futuro dimensionamiento pareado deberá fijar efecto alternativo $\delta$, α, multiplicidad y unidad, y podría partir, si son adecuados sus supuestos, de:

$$
R\approx\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\widehat\sigma_d^2}{\delta^2}.
$$

Se evaluarán la incertidumbre de la dispersión piloto y una alternativa apropiada si la aproximación no procede. Detectar frente a cero un efecto del tamaño de un margen no equivale a demostrar que se supera ese margen. Dimensionar semillas tampoco dimensiona automáticamente las fechas dependientes o las pruebas temporales.

### 5.4.8. Ausencia de significación y equivalencia

No rechazar una diferencia no demuestra equivalencia. **No se ejecutó TOST ni ROPE**, y falta un margen de equivalencia propio.

TOST requiere una banda práctica predefinida y ambas pruebas unilaterales; con α=0,05 sin otro ajuste, su regla usual corresponde al IC del 90 % dentro de la banda. Ese intervalo no se confunde con los IC del 95 % de reporte. ROPE exige un modelo bayesiano, priors, posterior y regla de decisión. El 5 % de mejora sobre persistencia no define automáticamente una banda de equivalencia entre modelos ni se ensancha después para obtener una conclusión.

### 5.4.9. Test de Diebold–Mariano

**Pendiente para las predicciones de esta CV.** El método se conserva como requisito de evaluación temporal, sin estadísticos o p-valores atribuidos a estos modelos.

#### 5.4.9.1. Comparación de precisión predictiva por horizonte

Para h=1 y una política de agregación de semillas previamente fijada, podría compararse la serie de diferencias de pérdida cuadrática:

$$
d_t=\frac1R\sum_{s=1}^{R}\left(e_{A,s,t}^2-e_{B,s,t}^2\right),\qquad H_0:\mathbb E[d_t]=0.
$$

Esta serie no es la pérdida de un ensemble ni concatena semilla×fecha como observaciones independientes. Tampoco contrasta directamente la media de RMSE por trayectoria: promediar raíces difiere de promediar pérdidas. Los cambios mensuales de modelo, la dependencia y posibles cambios de régimen deben entrar en la justificación. No se elegirá la semilla de menor error para contrastar.

#### 5.4.9.2. Corrección Harvey–Leybourne–Newbold y pérdida coherente

Antes de ejecutarlo se deberán registrar pérdida, estimación HAC de largo plazo, kernel, rezagos, hipótesis, familia de comparaciones y sensibilidades. Para $\widehat\Omega_d>0$:

$$
DM=\frac{\sqrt T\,\bar d}{\sqrt{\widehat\Omega_d}},\qquad
DM_{\rm HLN}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}.
$$

Con h=1, el factor HLN es $\sqrt{(T-1)/T}$; h no se sustituye por n=7 o D=7. La corrección para muestras pequeñas no resuelve por sí sola dependencia, no estacionariedad o selección. Su referencia y los grados de libertad serán identificados como exactos o aproximados según corresponda. Ninguna especificación HAC está ejecutada aquí.

### 5.4.10. Test BDS (Brock–Dechert–Scheinkman)

**Pendiente.** BDS puede examinar desviaciones de i.i.d. en una serie residual, con dimensiones, radio, tamaño útil, política de semillas y referencia definidos antes del contraste. No se atribuyen estadísticos BDS a las predicciones actuales.

Un eventual rechazo no identifica exclusivamente no linealidad aprovechable: dependencia lineal, heterocedasticidad, deriva, estimación de modelos y etiquetas móviles también afectan la interpretación. La multiplicidad no repara sus supuestos ni convierte un diagnóstico en una ganancia predictiva demostrada.

### 5.4.11. Test de Ljung–Box

**Pendiente.** Ljung–Box evalúa ausencia conjunta de autocorrelación hasta rezagos definidos, por ejemplo mediante:

$$
Q=T(T+2)\sum_{k=1}^{K}\frac{\widehat\rho_k^2}{T-k}.
$$

Los rezagos, el tramo continuo, la agregación de semillas, los grados de libertad y la familia deberán justificarse. El descuento de grados de libertad no se igualará al número de pesos de una red como si fueran parámetros ARMA. Un análisis de residuos al cuadrado responde otra pregunta y se declarará aparte. No hay resultados Ljung–Box calculados para esta CV.

### 5.4.12. Interpretación de los residuos

El diagnóstico descriptivo está disponible; la inferencia temporal formal sobre sus residuos sigue pendiente.

#### 5.4.12.1. Estructura residual y posible margen de mejora

Se examinó la descomposición por trayectoria:

$$
\mathrm{MSE}=\overline e^{\,2}+\operatorname{Var}_{\rm pobl}(e).
$$

En este diagnóstico, $e_t=\widehat Y_t-Y_t$: sesgo negativo indica subestimación y positivo sobreestimación. Los siguientes promedios resumen las diez trayectorias relativas anuales; la correlación es descriptiva:

| Red relativa | Sesgo medio | DE de predicción | DE del objetivo | Correlación media |
| --- | --- | --- | --- | --- |
| MLP | -0,0004849 | 0,008404 | 0,009767 | 0,7900 |
| LSTM | 0,0004575 | 0,008048 | 0,009767 | 0,7657 |
| CNN1D | 0,0001687 | 0,008042 | 0,009767 | 0,7725 |

Fuente: [bias_decomposition.csv](../_static/dl_r2_cv_20261104/analysis/bias_decomposition.csv). No son un contraste de autocorrelación o i.i.d. La figura utiliza la semilla designada 4101 y no una inicialización elegida por menor error:

![Predicciones diarias de las dos representaciones CV, semilla 4101](../_static/dl_r2_cv_20261104/analysis/figures/predictions_designated_seed.png)

![Descomposición descriptiva del sesgo en las trayectorias CV anuales](../_static/dl_r2_cv_20261104/analysis/figures/bias_cv_full2024.png)

La estructura residual podría reflejar etiquetas solapadas, deriva, heterocedasticidad o capacidad del modelo. No se cuantifica una mejora futura por observarla; una propuesta avanzada deberá demostrar su aprovechamiento con una evaluación separada.

### 5.4.13. Buenas Prácticas Estadísticas

Las reglas siguientes mantienen separados resultados ejecutados, diagnósticos descriptivos y procedimientos aún pendientes.

#### 5.4.13.1. Nivel de significancia declarado antes de analizar resultados

Se utiliza α=0,05 para la familia de tres contrastes relativa–absoluta, con ajuste Holm. El protocolo se registró antes de sus ajustes, pero la configuración y el desarrollo ya habían sido explorados; ese registro no se presenta como anterior a todo contacto con los datos.

#### 5.4.13.2. P-valores numéricos, ajustados y sin ajustar

Los CSV publican p-valores numéricos ajustados y sin ajustar, estadísticos, grados de libertad, logaritmos y límites de precisión. La notación científica evita reducirlos a «p<0,05». Un valor numérico no convierte la referencia t en una prueba exacta de aleatorización.

#### 5.4.13.3. Intervalos de confianza y tamaños de efecto

Se reportan IC t marginales y d_z pareados para las tres diferencias de RMSE. Los IC secundarios de ΔR² son descriptivos. No se calcula una banda simultánea por el hecho de ajustar p-valores ni un IC de equivalencia que no se haya definido.

#### 5.4.13.4. Verificación de supuestos

Los índices, continuidad, cobertura, escaladores y métricas fueron verificados. Shapiro–Wilk de las diferencias no rechaza sus tres hipótesis a 0,05; eso no prueba normalidad ni independencia global. ACF, Ljung–Box y BDS de las nuevas series residuales siguen pendientes. No se selecciona una prueba o semilla para favorecer el resultado.

#### 5.4.13.5. Múltiples ejecuciones documentadas

Se ejecutaron **diez trayectorias completas por red y variante**, semillas base 4101–4110. Cada mes utiliza una semilla derivada y el reajuste otra inicialización nueva, con configuración registrada. No se descartaron trayectorias por su error.

Los 720 segmentos mensuales y los 1.440 ajustes no son 720 o 1.440 semillas independientes. Diez supera mínimos aritméticos habituales, pero no demuestra potencia suficiente ni aumenta el número de mercados observados.

#### 5.4.13.6. Métrica principal y comparaciones declaradas con antelación

RMSE principal, σ7/ddof0, D7, h1, dos representaciones, diez semillas y tres contrastes emparejados constan en el protocolo. No hubo una nueva búsqueda anidada de hiperparámetros. La exposición del desarrollo se declara y las nuevas comparaciones requerirán su propia definición de familia, sin elegir ventanas o métricas por sus p-valores.

#### 5.4.13.7. Registro y trazabilidad de cada contraste

| Registro | Contenido verificable |
|---|---|
| [folds.json](../_static/dl_r2_cv_20261104/folds.json) | Índices reales de la función de Filipe, adaptación mensual, purgas, fechas y hashes |
| [runs.csv](../_static/dl_r2_cv_20261104/runs.csv) | Sesenta trayectorias, configuración, semilla, métricas, segmentos y costos |
| [comparisons.csv](../_static/dl_r2_cv_20261104/analysis/comparisons.csv) | Diez pares por red, Δ RMSE, t(9), IC, d_z, Shapiro y Holm-3; IC descriptivo de ΔR² |
| [fold_metrics.csv](../_static/dl_r2_cv_20261104/analysis/fold_metrics.csv) | 720 métricas mensuales, cohortes y cantidades por fold |
| [bias_decomposition.csv](../_static/dl_r2_cv_20261104/analysis/bias_decomposition.csv) | Descomposición descriptiva por trayectoria anual |
| [summary.json](../_static/dl_r2_cv_20261104/analysis/summary.json) | Fuentes, hashes, recuentos, métodos y alcance de desarrollo |

No hay tablas de resultados para DM, BDS, Ljung–Box, bootstrap temporal, Friedman, Wilcoxon, Nemenyi, TOST o ROPE de esta CV. DeLong, McNemar, segmentación y generación de lenguaje no aplican a la volatilidad continua de este proyecto.

### 5.4.14. Interpretación

La interpretación separa diferencias dentro del mismo procedimiento CV, ranking descriptivo y preguntas que exigen inferencia temporal o evaluación externa.

#### 5.4.14.1. Diferencias estadísticamente significativas

La representación relativa reduce la media de RMSE de cada red frente a la absoluta con los tres p-valores Holm inferiores a 0,05. La inferencia condiciona el mismo calendario y diseño de semillas; no hay una prueba ejecutada de superioridad entre redes o contra persistencia.

#### 5.4.14.2. Significancia estadística y relevancia práctica

La significación de relativa–absoluta no demuestra una mejora de al menos 5 % sobre persistencia. El referente anual conserva menor error y la ventaja puntual semestral de CNN1D relativa es 3,095 %. No se afirma utilidad económica o equivalencia.

#### 5.4.14.3. Conclusiones respaldadas por diferencias, incertidumbre y efectos

Se consideran diferencias, IC, tamaños de efecto, cobertura y variación por meses, sin basarse únicamente en promedios. Los d_z grandes entre inicializaciones no se extrapolan a variabilidad entre mercados. El resultado favorable de un subperiodo no sustituye el anual.

#### 5.4.14.4. Estado del análisis estadístico

**Ejecutado:** métricas por trayectoria y mes, IC t por semillas, tres contrastes pareados relativa–absoluta con Holm, IC descriptivos de ΔR² y descomposición de sesgo.

**Pendiente:** inferencia temporal, comparaciones inferenciales entre arquitecturas o contra persistencia, dimensionamiento de potencia, equivalencia, margen avanzado, ajuste final con desarrollo y test. No se acredita potencia 0,80 con las diez semillas.

## 5.5. Discusión Crítica

La discusión responde a las preguntas del entregable con evidencia de la CV efectiva y límites explícitos, sin convertir el desarrollo exploratorio en confirmación de generalización.

### 5.5.1. Desempeño y explicación de las diferencias

MLP relativa tiene menor RMSE neuronal anual; CNN1D relativa lidera el semestre julio–diciembre. MLP opera sobre la historia aplanada, LSTM incorpora estado recurrente y CNN1D filtros temporales compartidos. Estos sesgos inductivos son descripciones de arquitectura y no una explicación causal demostrada del orden observado.

El contraste de representación muestra mejora relativa–absoluta en cada red bajo folds y semillas comunes. También cambia la ponderación del escalado, por lo que no identifica un mecanismo único. No se atribuye toda diferencia a profundidad o capacidad.

### 5.5.2. Generalización y estabilidad

No se determina qué red generaliza mejor al test reservado: no fue evaluado por este ciclo. La CV incorpora únicamente pasado permitido en cada mes, pero su diseño utiliza desarrollo ya conocido y configuraciones escogidas con ese desarrollo.

La estabilidad mensual es limitada: el R² medio entre semillas es negativo en **2 de doce meses para MLP relativa, 3 para LSTM y 4 para CNN1D**. La media anual positiva no elimina esos fallos. Los [resultados por fold](../_static/dl_r2_cv_20261104/analysis/fold_metrics.csv), el activo y proveedor únicos y la exclusión del día incompleto delimitan la extrapolación.

### 5.5.3. Significación estadística y relevancia práctica

La mejora de preprocesamiento está respaldada por diferencias de RMSE, IC y Holm-3 condicionados a las semillas. La relevancia frente a persistencia permanece sin demostración del margen del 5 %. Un menor promedio semestral no sustituye una comparación temporal registrada ni confirma el desempeño fuera de desarrollo.

### 5.5.4. Desempeño frente a costo y complejidad

Las redes relativas conservan el mismo conteo de pesos que sus controles absolutos, con menor error y más tiempo medio de ajuste. MLP relativa combina menor RMSE anual y menos parámetros; LSTM relativa presenta menor tiempo medio de entrenamiento. No se deriva una decisión de despliegue ni una ventaja poblacional a partir de costos concurrentes locales.

### 5.5.5. Baseline y umbral para un modelo avanzado

El candidato neuronal es **MLP relativa bajo CV mensual**, con RMSE anual 0,0060491. Una propuesta avanzada deberá compararse con un procedimiento completamente cerrado, emparejamiento adecuado y un margen práctico propio; mejorar el candidato neuronal no exime de considerar la persistencia anual más fuerte.

El ajuste final con desarrollo y el protocolo del test deben cerrarse antes de consultarlo. Si los resultados del test se usan después para diseñar otra propuesta, será necesaria otra evaluación independiente para confirmarla.

### 5.5.6. Brecha con la literatura y margen de mejora potencial

Las diferencias publicadas en la literatura sirven como contexto cuando se declaran objetivos, escala, fuente y partición. No se trasladan a la σ7 diaria de este proyecto como una ganancia esperada. El corpus mantiene apoyo aplicado pendiente para MLP y CNN1D simple; la ejecución propia no completa esa evidencia ni convierte la CNN en una réplica de la TCN publicada.

### 5.5.7. Amenazas a la validez

| Amenaza | Evidencia y control | Límite |
|---|---|---|
| Interna | Folds de Filipe, filtros de disponibilidad, purga, escaladores de train y configuración por segmento | La exploración y elección de configuración usan desarrollo ya conocido; no hay selección anidada |
| Externa | Año completo y subperiodo identificados, predicciones por fecha | Un activo, proveedor y periodo; test no evaluado |
| De construcción | σ7/ddof0 y RMSE/R² definidos; persistencia observable | Proxy móvil con ruido y solapamiento; no es volatilidad latente observada sin error |
| De conclusión estadística | IC marginales, pares de semillas y Holm-3 | Dependencia temporal, diez trayectorias y potencia no dimensionada; pruebas temporales pendientes |

Los bloques mensuales crecen y comparten historia de entrenamiento; no son independientes entre sí. La comparación de representaciones mantiene folds y semillas, pero cambia representación y ponderación del escalado. La información histórica de persistencia difiere en un cierre adicional.

### 5.5.8. Limitaciones del estudio

El estudio comprende un activo y proveedor, objetivo móvil solapado, configuraciones fijas y diez semillas por combinación. Se ejecutan dos representaciones sin una nueva búsqueda anidada; 2024 estaba expuesto al EDA y a decisiones de configuración. Los meses con R² negativo persisten y las pruebas temporales no se han realizado.

La referencia anual de menor error es persistencia. La ventaja puntual de CNN1D relativa en el semestre es inferior al 5 % y no confirma mejora financiera ni generalización. El respaldo bibliográfico aplicado, dimensionamiento, margen avanzado, ajuste final y test permanecen pendientes.

### 5.5.9. Recomendaciones futuras

Las siguientes acciones derivan de fallos y límites observados; no se promete obtener un p-valor favorable o un R² predeterminado.

#### 5.5.9.1. Recomendaciones derivadas de las limitaciones

Cerrar representación, configuración, reglas de actualización y márgenes antes del ajuste final y del test. Completar el respaldo aplicado de MLP/CNN1D simple y un piloto de variabilidad coherente con el procedimiento creciente.

Examinar los meses de R² negativo, la estructura residual y diferencias de información con persistencia. Registrar inferencia temporal apropiada y comparar propuestas con presupuesto justo. Más semillas no sustituyen periodos externos independientes ni permiten usar el test para diseñar sin reservar otra evaluación confirmatoria.

#### 5.5.9.2. Matriz de conclusiones y evidencia pendiente

| Pregunta | Evidencia disponible | Conclusión o pendiente |
|---|---|---|
| ¿Qué red funciona mejor? | RMSE anual de diez trayectorias por representación | MLP relativa tiene menor error neuronal anual; el ranking semestral cambia |
| ¿Qué R² se obtuvo? | Métricas anuales y de julio–diciembre de la misma CV | Medias relativas positivas en las tres redes, con meses negativos restantes |
| ¿Se supera persistencia? | Referencia determinista en las mismas fechas | Persistencia es mejor anual; CNN1D mejora 3,095 % puntual en el semestre, por debajo del 5 % |
| ¿Qué diferencia está contrastada? | Tres pares de representaciones con Holm-3 | Mejora relativa–absoluta dentro de cada red; no hay pruebas temporales ni entre arquitecturas |
| ¿Compensa la complejidad? | Parámetros y costos de la CV | MLP relativa combina menor error anual y menos pesos; no se acredita utilidad de despliegue |
| ¿Cuál es el baseline y margen avanzado? | Candidato MLP relativo y protocolo de desarrollo | Cierre del procedimiento, margen aplicado y dimensionamiento pendientes |
| ¿Se puede extrapolar al test? | Test reservado sin evaluación | No; falta ajuste final y evaluación bajo protocolo cerrado |
| ¿Qué falta de la revisión? | Corpus aplicado con lagunas MLP/CNN1D simple | La implementación propia no sustituye sus referencias bibliográficas |
