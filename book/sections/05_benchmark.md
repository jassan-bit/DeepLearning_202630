# 5. Benchmark de los Modelos Base y Definición del Baseline

El capítulo conserva **dos ciclos experimentales de desarrollo**, con los mismos tres modelos —MLP, LSTM y CNN1D poco profunda— y alcances distintos. **V1** es el benchmark original: entrenamiento estático en 2023, búsqueda en enero–junio de 2024 y comparación en julio–diciembre con treinta semillas. Sus R² negativos y demás resultados se conservan. **V2-CV** responde a ese fallo observado y aplica validación hacia adelante de Filipe en los doce meses de 2024, con dos preprocesamientos y diez semillas por modelo/variante. Los resultados anuales de V2 no sustituyen cifras de V1 calculadas en otro periodo.

Ambos ciclos usan **desarrollo de 2024 ya expuesto** al EDA y a decisiones de diseño; la CV posterior tampoco es confirmación independiente. El test de **enero de 2025 a agosto de 2026 permanece cerrado**. Septiembre–octubre de 2026 no tienen una función experimental asignada y el ajuste final sobre desarrollo completo todavía no se ha ejecutado.

En V1 el protocolo se registró a las 14:28:17 UTC del 4 de noviembre de 2026 y se cerró tras el piloto a las 14:37:41 UTC. La búsqueda fijó **D=7 y MLP como referencia operacional**; LSTM fue el líder descriptivo posterior entre redes y persistencia superó a las tres. El protocolo V2 se registró a las 15:17:28 UTC, después de observar V1 y antes de sus propios ajustes. Mantiene arquitectura, anchos y tasas, sin nueva búsqueda ni sustitución retrospectiva de MLP.

Los archivos originales de V1 siguen disponibles: [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json), [protocolo final](../_static/dl_benchmark_20261104/protocol_final.json), [configuración](../_static/dl_benchmark_20261104/selected_configuration.json), [piloto](../_static/dl_benchmark_20261104/pilot_summary.json), [corridas](../_static/dl_benchmark_20261104/runs.csv), [predicciones](../_static/dl_benchmark_20261104/predictions/development_benchmark.csv) y {download}`alcance V1 <../_static/dl_benchmark_20261104/analysis/README.md>`. V2 tiene [protocolo propio](../_static/dl_r2_cv_20261104/protocol.json), [erratum descriptivo](../_static/dl_r2_cv_20261104/protocol_erratum.json) y [particiones verificables](../_static/dl_r2_cv_20261104/folds.csv). El entrenamiento propio no resuelve el respaldo aplicado pendiente para MLP y CNN1D simple en §3.

**Resultado del ciclo V2-CV:** se completaron las sesenta trayectorias, 720 segmentos mensuales y 1.440 ajustes, con 21.960 predicciones anuales. `relative_window` obtiene R² medios positivos —MLP 0,615670; LSTM 0,578892; CNN1D 0,592088—, frente a la persistencia anual 0,759571. La mejora respecto de `absolute_log` está cuantificada dentro de la CV; no borra el resultado negativo original ni constituye una validación del test.

## 5.1. Comparación Tabular



### 5.1.1. Tabla de comparación de los tres modelos base

**V1: benchmark original estático, julio–diciembre de 2024.**

La comparación corresponde a BTCUSDT de Binance Spot, datos diarios UTC, entrada OHLCV de **D=7 días**, objetivo de volatilidad **$\sigma_7$ poblacional (`ddof=0`) sin anualizar** y horizonte h=1. Son las mismas **184 fechas, del 1 de julio al 31 de diciembre de 2024**, y semillas 1001–1030 para cada red; no hay fechas ni corridas faltantes en esta tabla.

Los modelos son ajustes estáticos con cohortes de entrenamiento de 2023 y embargo de D días. La búsqueda de D e hiperparámetros, el escalado, la validación interna y la parada siguen §4; no se reajustan con los resultados de julio–diciembre. La transformación logarítmica de OHLC y `log1p` de volumen, sus escalas por canal y la escala positiva de Y se ajustan solo con los datos de entrenamiento admitidos.

Para orígenes comunes $\mathcal V_c$ y semillas $\mathcal S_c$, la métrica de una corrida y la puntuación principal son:

$$
m_{m,s,c}=\sqrt{\frac1{N_c}\sum_{t\in\mathcal V_c}
(Y_t-\widehat Y_{m,s,t})^2},\qquad
S_{m,c}=\frac1{R_c}\sum_{s\in\mathcal S_c}m_{m,s,c}.
$$

Se ordena por **media de RMSE por corrida**, con DE muestral entre semillas. No es la RMSE del promedio de predicciones ni una única RMSE sobre filas semilla×fecha. Promediar predicciones definiría un ensemble adicional que no se evalúa aquí. No se comparan directamente RMSE de ventanas de volatilidad distintas.

**V2-CV: dos procedimientos sobre las mismas 366 fechas de 2024.** Se utiliza `timeseries-cv==0.1.5`, función `split_train_val_forwardChaining(7,1,1)`, con los índices reales de la librería. Sus orígenes diarios se agrupan en **doce bloques mensuales**: se ajusta con pasado creciente de 2023 y 2024 ya observado, y se pronostica cada día a h=1 sin reajustar dentro del mes. No son 366 reajustes ni pronósticos a un mes.

Para origen mensual b, se admiten etiquetas de entrenamiento hasta $u\leq b-D-1$ y se conserva el embargo interno. La ventana interna de los últimos noventa días antes de b deja **83 etiquetas tras la purga**; por ejemplo, en enero son del 3 de octubre al 24 de diciembre de 2023. Se elige época con ese pasado y se reajusta desde cero sobre toda la cohorte mensual admitida. El [erratum](../_static/dl_r2_cv_20261104/protocol_erratum.json) corrige una frase heredada que decía solo 2023; los índices y ajustes ya utilizaban el pasado creciente correcto.

Las variantes mantienen cinco canales, D=7, n=7, h=1 y las configuraciones de V1:

| Variante V2 | Representación y ajuste de escalas |
|---|---|
| `absolute_log` | Log de OHLC, `log1p` de volumen y escala por canal ajustada sobre días únicos de entrada del entrenamiento; conserva la representación de V1 |
| `relative_window` | Log de cada precio relativo al último cierre conocido de su ventana; `log1p` de volumen y escala sobre todas las posiciones de ventanas admitidas de train |

La segunda variante cambia **representación y ponderación de posiciones repetidas en el escalado**; no aísla causalmente una sola transformación. No añade rezagos explícitos de volatilidad ni usa precios futuros. Y conserva el escalado positivo aprendido solo en train y se invierte antes de puntuar. Las diez semillas base 4101–4110 se comparten entre variantes/modelos; cada mes usa semillas derivadas nuevas, con un ajuste interno y un refit por segmento: **24 ajustes por trayectoria anual**.

La RMSE y el R² anuales se calculan concatenando las predicciones fuera del entrenamiento de cada trayectoria sobre las 366 etiquetas de 2024. Después se resumen las diez trayectorias; **no se promedian R² diarios ni R² mensuales para obtener el anual**. Los doce meses no añaden doce mercados independientes. El subconjunto julio–diciembre permite contexto descriptivo con las mismas 184 etiquetas de V1, sin emparejar sus treinta semillas con las diez distintas de V2.

#### 5.1.1.1. Modelo base

**Resultados originales V1; se conservan como referencia del primer ciclo.**

Resultados de desarrollo en unidades originales de volatilidad. Los IC de esta tabla son t de Student al 95 %, marginales y condicionados al mismo calendario; no son intervalos de generalización temporal.

| Modelo | Corridas | RMSE, media ± DE | IC 95 % de RMSE | MAE, media ± DE | Parámetros entrenables |
|---|---:|---:|---|---:|---:|
| MLP, H=16, LR=0,001 | 30 | 0,0159903 ± 0,0028545 | [0,0149244; 0,0170562] | 0,0138174 ± 0,0028091 | 593 |
| LSTM, H=32, LR=0,003 | 30 | 0,0089035 ± 0,0004854 | [0,0087222; 0,0090847] | 0,0068832 ± 0,0007473 | 5.025 |
| CNN1D, H=16, LR=0,001 | 30 | 0,0175867 ± 0,0051921 | [0,0156479; 0,0195255] | 0,0152235 ± 0,0052040 | 1.057 |
| Persistencia, referencia determinista | No aplica | 0,0048225; sin DE entre semillas | No aplica a semillas | 0,0028727; sin DE entre semillas | 0 |

Fuente: [metrics_summary.csv](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv). Persistencia no es un cuarto modelo DL ni treinta entrenamientos repetidos. Pronostica con la última σ7 conocida y necesita ocho cierres anteriores; las redes reciben siete días de OHLCV. Esa diferencia de información debe conservarse al interpretar la comparación.

**V2-CV anual: del 1 de enero al 31 de diciembre de 2024, 366 fechas por trayectoria.** Cada fila DL resume diez semillas. Los IC t del 95 % son marginales entre trayectorias, condicionados a ese calendario; persistencia es un único pronóstico determinista.

| Variante | Modelo | RMSE, media ± DE | R², media ± DE | IC 95 % de R² | MAE, media ± DE |
|---|---|---:|---:|---|---:|
| `absolute_log` | MLP | 0,0104473 ± 0,0005993 | -0,147616 ± 0,130856 | [-0,241225; -0,054007] | 0,0079127 ± 0,0003336 |
| `absolute_log` | LSTM | 0,0096539 ± 0,0004292 | 0,021226 ± 0,087144 | [-0,041113; 0,083565] | 0,0073090 ± 0,0003436 |
| `absolute_log` | CNN1D | 0,0106964 ± 0,0008499 | -0,206245 ± 0,191124 | [-0,342967; -0,069523] | 0,0079353 ± 0,0005051 |
| `relative_window` | MLP | 0,0060491 ± 0,0002780 | 0,615670 ± 0,035544 | [0,590243; 0,641096] | 0,0043804 ± 0,0002110 |
| `relative_window` | LSTM | 0,0063354 ± 0,0001870 | 0,578892 ± 0,024807 | [0,561147; 0,596638] | 0,0047263 ± 0,0001245 |
| `relative_window` | CNN1D | 0,0062296 ± 0,0003373 | 0,592088 ± 0,042687 | [0,561551; 0,622624] | 0,0046780 ± 0,0002212 |
| Referencia determinista | Persistencia | 0,0047890 | 0,759571 | No aplica a semillas | 0,0029435 |

Fuente: [metrics_summary.csv V2](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), que conserva también IC de RMSE y demás métricas. Los R² negativos de MLP/CNN1D con `absolute_log` no desaparecen por actualizar mensualmente; la variante relativa mejora los tres modelos en esta CV. Un R² positivo representa menor MSE que la referencia algebraica de la media observada del mismo periodo, no superioridad frente a persistencia.

![R² anual de ambas variantes CV, media y DE entre diez semillas](../_static/dl_r2_cv_20261104/analysis/figures/r2_cv_full2024.png)

Las barras muestran DE, no IC; la línea de persistencia no tiene variabilidad de inicialización.

#### 5.1.1.2. Métricas: media ± desviación estándar

**V1, mismas 184 fechas de julio–diciembre.**

El CSV anterior conserva media, DE muestral, número de valores definidos, cobertura e IC de cada métrica. Las siguientes cifras son complementarias a RMSE; todas se calculan después de invertir la escala de Y:

| Modelo | R², media ± DE | MAPE (%), media ± DE | MASE, media ± DE |
|---|---:|---:|---:|
| MLP | −2,73998 ± 1,29673 | 69,6558 ± 14,4619 | 5,56565 ± 1,13150 |
| LSTM | −0,12809 ± 0,12276 | 33,7315 ± 5,0528 | 2,77255 ± 0,30100 |
| CNN1D | −3,75860 ± 3,02634 | 76,6621 ± 27,0778 | 6,13203 ± 2,09619 |
| Persistencia | 0,67000 | 13,2578 | 1,15711 |

MAPE y MASE tienen cobertura del 100 % en esta cohorte; sus denominadores y máscaras siguen §4.4. El denominador de MASE es histórico, por lo que no se debe esperar MASE=1 para la persistencia en estas fechas de evaluación. El R² negativo de las redes describe el error en este tramo frente a la media observada usada en la definición de R²; no identifica por sí solo la causa del mal desempeño.

**Comparación descriptiva en las mismas 184 fechas de julio–diciembre.** Se recalculan las métricas de V2 sobre esas fechas; no se comparan sus resultados anuales directamente con el semestre de V1. V1 usa treinta semillas y V2 diez distintas: no son pares de entrenamientos.

| Ciclo/variante | Modelo | Semillas | RMSE, media ± DE | R², media ± DE |
|---|---|---:|---:|---:|
| V1 estático original | MLP | 30 | 0,0159903 ± 0,0028545 | -2,739982 ± 1,296729 |
| V1 estático original | LSTM | 30 | 0,0089035 ± 0,0004854 | -0,128090 ± 0,122763 |
| V1 estático original | CNN1D | 30 | 0,0175867 ± 0,0051921 | -3,758601 ± 3,026344 |
| V2 `absolute_log` | MLP | 10 | 0,0097415 ± 0,0005256 | -0,350098 ± 0,146099 |
| V2 `absolute_log` | LSTM | 10 | 0,0077408 ± 0,0006330 | 0,144631 ± 0,143471 |
| V2 `absolute_log` | CNN1D | 10 | 0,0093454 ± 0,0007712 | -0,246885 ± 0,202259 |
| V2 `relative_window` | MLP | 10 | 0,0049062 ± 0,0002224 | 0,657808 ± 0,032033 |
| V2 `relative_window` | LSTM | 10 | 0,0052429 ± 0,0002265 | 0,609298 ± 0,033741 |
| V2 `relative_window` | CNN1D | 10 | 0,0046732 ± 0,0001429 | 0,689848 ± 0,018888 |
| Mismas fechas en ambos ciclos | Persistencia | No aplica | 0,0048225 | 0,669995 |

Fuente: [metrics_common_h2.csv](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2.csv). El R² original queda preservado: −2,739982, −0,128090 y −3,758601; en V2 relativo pasa a 0,657808, 0,609298 y 0,689848. Esta comparación cambia además política de ajuste, cohortes y semillas, por lo que no identifica causalmente una sola corrección ni lleva p-valores pareados V1–V2.

![R² de V1 y las dos variantes CV sobre julio–diciembre común; tamaños de semillas distintos](../_static/dl_r2_cv_20261104/analysis/figures/r2_common_h2_context.png)

En V2, MASE usa la escala histórica de entrenamiento admitido de cada mes, registrada por predicción. V1 usa su escala fija de 2023. Esas políticas se conservan diferenciadas en [persistence_by_scale_policy.csv](../_static/dl_r2_cv_20261104/analysis/persistence_by_scale_policy.csv); una diferencia de MASE entre ciclos no se atribuye enteramente al pronóstico.

#### 5.1.1.3. Intervalos de confianza del 95 % y método de construcción

En V1, los IC marginales entre semillas emplean t de Student con 29 grados de libertad. Describen la media de desempeño del procedimiento de entrenamiento condicionada a los datos, partición e hiperparámetros observados. El solapamiento de IC marginales de modelos no sustituye un IC de su diferencia pareada.

El análisis temporal complementario de V1 remuestrea conjuntamente fechas en bloques circulares: 2.000 réplicas, longitud principal 14 días y sensibilidades 7 y 28 (§5.4.5). Recalcula la RMSE de cada semilla y luego su media; no remuestrea semillas ni reentrena modelos. Ambos intervalos contestan preguntas distintas. Los IC del 95 % publicados son marginales; Holm sobre p-valores no los convierte en intervalos simultáneos.

En V2, los IC t marginales usan nueve grados de libertad y condicionan las mismas fechas de 2024. No se ejecutaron bootstrap temporal, DM, BDS ni Ljung–Box para V2. Los resultados originales de esas pruebas en §5.4 pertenecen exclusivamente a V1; no se trasladan a las nuevas predicciones.

#### 5.1.1.4. Tiempo de entrenamiento y de inferencia

**Costos originales V1, ejecución serial.**

Las mediciones se hicieron en CPU con un hilo PyTorch, sin GPU y con ejecución serial de las corridas del benchmark. El tiempo de entrenamiento suma selección de época mediante validación interna y reajuste nuevo; excluye preprocesamiento y serialización. La inferencia se midió después de cinco calentamientos, con veinte pasadas y lote máximo de 256: cada duración corresponde a **las 184 predicciones**, no a una predicción individual.

| Modelo | Entrenamiento (s), media ± DE | Inferencia de 184 casos (ms), media ± DE |
|---|---:|---:|
| MLP | 0,34480 ± 0,13838 | 0,03979 ± 0,01082 |
| LSTM | 0,22704 ± 0,05764 | 0,46344 ± 0,08246 |
| CNN1D | 0,63489 ± 0,49260 | 0,35376 ± 0,05189 |

El registro completo está en [runs.csv](../_static/dl_benchmark_20261104/runs.csv). Los tiempos de la búsqueda, ejecutada con tres trabajadores, no se comparan directamente con estas mediciones seriales. Son tiempos locales y muy pequeños, sensibles a hardware, calentamiento y época de parada; no son una evaluación de despliegue ni de entrenamiento acelerado.

**Alcance V2:** las trayectorias CV se ejecutan con tres trabajadores concurrentes en CPU y doce reajustes mensuales por trayectoria. Sus tiempos no permiten una comparación directa de velocidad con V1 serial, ni se asigna retrospectivamente el costo anual al subconjunto de 184 fechas. La cuenta correcta distingue trayectoria, segmento mensual y los dos ajustes de cada segmento.

**Costos medidos V2 por trayectoria anual completa.** El entrenamiento suma los dos ajustes de cada uno de los doce meses; la inferencia suma las mediciones de los doce bloques. La ejecución tuvo tres trabajadores concurrentes y no es comparable en velocidad con el único ajuste estático serial V1.

| Variante | Modelo | Entrenamiento de 12 bloques (s), media ± DE | Inferencia total de 366 casos (ms), media ± DE |
|---|---|---:|---:|
| `absolute_log` | MLP | 9,189 ± 1,617 | 0,6397 ± 0,0896 |
| `absolute_log` | LSTM | 12,010 ± 1,738 | 3,7515 ± 0,6540 |
| `absolute_log` | CNN1D | 17,661 ± 2,955 | 4,0847 ± 0,3203 |
| `relative_window` | MLP | 21,352 ± 1,234 | 0,6227 ± 0,0413 |
| `relative_window` | LSTM | 18,505 ± 2,305 | 3,2121 ± 0,3623 |
| `relative_window` | CNN1D | 44,350 ± 2,425 | 3,6905 ± 0,2351 |

Fuente: [runs.csv V2](../_static/dl_r2_cv_20261104/runs.csv) y [resumen de métricas](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv). Se conservan 24 ajustes por trayectoria anual, sesenta trayectorias y 1.440 ajustes en total; no se presentan como una única corrida ni como 1.440 semillas independientes.

#### 5.1.1.5. Número de parámetros

El conteo proviene de las instancias reales de PyTorch: **593 en MLP, 5.025 en LSTM y 1.057 en CNN1D**, todos entrenables; cada modelo tiene cero parámetros no entrenables. La persistencia no tiene pesos ajustados. Las fórmulas y pruebas del conteo están en la implementación de §4.2.

MLP tiene una capa oculta; LSTM una capa recurrente unidireccional; CNN1D dos convoluciones causales ordinarias de kernel 3, sin dilatación ni residuales, seguidas de promedio temporal. La CNN ejecutada no reproduce la TCN de S1.

V2 reutiliza las mismas instancias arquitectónicas y cantidades de parámetros; la CV cambia cohortes, preprocesamiento y pesos aprendidos por mes. No se añade una capa ni se repite la búsqueda de anchos, tasas o D.

## 5.2. Ranking de Modelos Base



### 5.2.1. Orden de los tres modelos según la métrica principal

**Resultados y decisiones originales V1.**

La búsqueda exhaustiva compartió cuatro historias D∈{7,14,21,28}, anchos {16,32}, tasas {0,001;0,003} y semillas {101,202}. Se seleccionó una configuración por familia y D, y después el D común que minimiza la media de los tres ganadores. Los desempates registrados prefieren menos parámetros y menor D cuando corresponde. Resultó **D=7**.

| Familia seleccionada en D=7 | RMSE de búsqueda enero–junio, media de 2 semillas | Puesto en búsqueda | Puesto descriptivo julio–diciembre |
|---|---:|---:|---:|
| MLP | 0,01052383 | 1 | 2 |
| LSTM | 0,01077513 | 2 | 1 |
| CNN1D | 0,01348078 | 3 | 3 |

Por la regla previa, **MLP es la referencia operacional seleccionada**. El orden descriptivo posterior es LSTM, MLP, CNN1D; persistencia obtiene menor error que todas. La discrepancia entre búsqueda y comparación se reporta sin sustituir MLP post hoc. La selección no exige demostrar superioridad frente a todas las alternativas para poder identificar una referencia operacional, y estos resultados expuestos no establecen un ganador en test.

La frecuencia de menor RMSE en el [bootstrap temporal](../_static/dl_benchmark_20261104/analysis/ranking_bootstrap.csv) es 1,000 para LSTM dentro de las tres redes y 1,000 para persistencia al incluirla, para las tres longitudes de bloque. Son frecuencias de remuestreo de estos modelos almacenados, no probabilidades posteriores ni una demostración de ranking poblacional.

**Ranking descriptivo V2, sin nueva selección.** En todo 2024 la menor RMSE entre las seis combinaciones corresponde a `relative_window`–MLP; dentro de esa variante el orden es MLP, CNN1D, LSTM. En `absolute_log` el orden es LSTM, MLP, CNN1D. La persistencia anual conserva menor RMSE que todas. En las 184 fechas comunes, la variante relativa cambia de orden: CNN1D, MLP, LSTM.

La diferencia de orden entre periodos demuestra por qué deben identificarse cohorte y variante. No se calculó una familia de contrastes entre las tres redes V2 para certificar ese ranking ni se reemplazó la referencia operacional MLP de V1; los tres contrastes V2 ejecutados comparan preprocesamientos dentro de cada arquitectura.

### 5.2.2. Análisis del trade-off

**Resultados y decisiones originales V1.**

Se comparan los frentes error–tiempo y error–parámetros usando las mediciones de §5.1. La dominancia sobre medias es descriptiva de esta CPU y este calendario, con la dispersión de corridas documentada; no acredita dominancia poblacional.

En V2, la variante relativa reduce error con los mismos parámetros que la absoluta, pero exige más tiempo medio de entrenamiento en las tres arquitecturas. Los cambios de época de parada y las condiciones concurrentes forman parte del costo; no se interpreta esa diferencia como eficiencia arquitectónica intrínseca. MLP relativo combina la menor RMSE anual DL, menos parámetros y la inferencia más corta entre las redes medidas; no se ensayó una regla adicional de despliegue.

#### 5.2.2.1. Precisión frente a costo computacional

**Resultados y decisiones originales V1.**

LSTM tiene la menor RMSE entre redes y el menor tiempo medio de entrenamiento observado, pero la mayor duración de inferencia. MLP tiene la inferencia más corta. En medias, MLP supera a CNN1D en error y en ambos tiempos medidos; esa comparación de costo no convierte la diferencia de RMSE entre semillas en significativa (§5.3.3).

La parada temprana cambia el número de épocas y contribuye al tiempo total. Por ello no se atribuye el menor tiempo de entrenamiento de LSTM a una eficiencia intrínseca de su arquitectura. Tampoco se establece una exigencia operacional de latencia que no haya sido definida para el proyecto.

#### 5.2.2.2. Precisión frente a complejidad del modelo

**Resultados y decisiones originales V1.**

LSTM usa aproximadamente 8,47 veces los parámetros de MLP y 4,75 veces los de CNN1D. Su menor error observado se obtiene con más pesos, pero no con mayor tiempo medio de entrenamiento en estas corridas. La inferencia sí es más lenta que en MLP; parámetros y latencia no son intercambiables.

MLP y LSTM presentan un compromiso descriptivo entre RMSE y parámetros; CNN1D queda dominada por MLP en esas medias. Persistencia obtiene menor error sin pesos entrenados, aunque con la diferencia de historia disponible ya declarada. No se infiere utilidad económica ni una decisión de despliegue a partir de estos frentes.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

Se distinguen la referencia DL congelada por la búsqueda, el líder descriptivo posterior, la referencia ingenua y la sensibilidad del diseño. Así se evita convertir la elección del menor error observado en una confirmación independiente.

### 5.3.1. Selección del baseline de referencia



#### 5.3.1.1. Identificación del mejor modelo base

La regla de menor RMSE media en enero–junio seleccionó **MLP, D=7, H=16 y LR=0,001**. La configuración comparte AdamW, dropout=0,1, decaimiento de pesos=10⁻⁴, lote 32 y las reglas de parada/reajuste de §4. Se conserva como **baseline DL operacional de desarrollo**.

LSTM es la mejor red por la media observada en julio–diciembre, pero esa evidencia no autoriza una sustitución retrospectiva. Persistencia es la referencia de menor error en esta evaluación; su ventaja impide afirmar que el uso de DL haya mejorado el pronóstico. Ninguna de estas denominaciones constituye una cota universal de error ni una elección confirmada con el test reservado.

La identidad operacional seleccionada en V1 se conserva durante V2-CV. Un líder observado en el nuevo ciclo se identificará como descriptivo de esa CV expuesta; no se presenta como reselección confirmada ni como ganador del test.

#### 5.3.1.2. Desempeño con media, desviación e IC del 95 %

**Resultados y decisiones originales V1.**

El baseline operacional MLP obtiene RMSE **0,0159903 ± 0,0028545**, IC t marginal del 95 % **[0,0149244; 0,0170562]**, en treinta semillas sobre 184 fechas expuestas. LSTM obtiene **0,0089035 ± 0,0004854**, IC **[0,0087222; 0,0090847]**, y persistencia **0,0048225**, sin variabilidad de inicialización. Se reportan selección y desempeño posterior con sus nombres y fases diferenciados.

**Medición V2 de la familia MLP, sin reselección:** con `relative_window`, D=7/H=16/LR=0,001 y ajuste mensual, la RMSE anual es 0,0060491 ± 0,0002780, IC t [0,0058502; 0,0062479]; R² 0,615670 ± 0,035544. Es un procedimiento de preprocesamiento y actualización distinto de V1, no una nueva medición intercambiable del mismo pipeline congelado.

### 5.3.2. Cuantificación de las diferencias entre modelos base



#### 5.3.2.1. Regresión y series de tiempo: diferencias respecto al peor y al mejor modelo

**Resultados y decisiones originales V1.**

Para la métrica de error k se usarán sus medias por corrida $\bar M_{m,k,c}$ sobre repeticiones y fechas comparables. Los extremos descriptivos se identifican **por métrica**:

$$
r^{\mathrm{mejor}}_{k,c}=\arg\min_m\bar M_{m,k,c},\qquad
r^{\mathrm{peor}}_{k,c}=\arg\max_m\bar M_{m,k,c}.
$$

El mejor o peor por RMSE no tiene por qué serlo por MAE. Las tablas identificarán qué referente se usa para cada porcentaje. Si se calcula MAE frente a los identificadores elegidos por RMSE como análisis complementario, se rotulará así y no se los llamará extremos MAE. Los empates conservarán sus identificadores según §5.2.1.

La reducción relativa descriptiva se calculará **sobre las medias por corrida**:

$$
\Delta_k(m;r,c)=100\,
\frac{\bar M_{r,k,c}-\bar M_{m,k,c}}{\bar M_{r,k,c}},
\qquad \bar M_{r,k,c}>0.
$$

Un valor positivo favorece al modelo m frente a r; negativo indica mayor error. Respecto al mejor error de esa métrica, las otras filas tendrán reducciones no positivas. Si el referente vale cero, el porcentaje es indefinido, sin epsilon. Se publicarán también diferencias en unidades originales, con signo y referentes identificados.

La razón de medias no equivale a la media de porcentajes por semilla. Esta última sería otro estimando; no se mezcla con el principal de la tabla. Las comparaciones pareadas usan las métricas individuales y las predicciones, sin sustituirlas por cifras redondeadas. Para un IC de la razón se recalculan numerador y denominador conjuntamente bajo la unidad de inferencia registrada.

En julio–diciembre, LSTM y CNN1D son respectivamente los extremos mejor y peor entre las redes tanto por RMSE como por MAE. Son extremos descriptivos de desarrollo, separados de la referencia operacional MLP. Elegirlos en esas mismas fechas no elimina la incertidumbre de selección.

| Modelo | Reducción de RMSE frente a CNN1D (%) | Reducción de RMSE frente a LSTM (%) | Reducción de MAE frente a CNN1D (%) | Reducción de MAE frente a LSTM (%) |
|---|---:|---:|---:|---:|
| MLP | 9,077 | -79,597 | 9,236 | -100,741 |
| LSTM | 49,374 | 0,000 | 54,786 | 0,000 |
| CNN1D | 0,000 | -97,526 | 0,000 | -121,169 |

Frente a persistencia, la reducción relativa de RMSE es **−231,579 % en MLP, −84,624 % en LSTM y −264,682 % en CNN1D**: son aumentos de error, no mejoras. Las diferencias originales y sus IC se presentan en §§5.3.3 y 5.4.5. Estos porcentajes se calculan con cifras sin redondear del CSV.

**Diferencias descriptivas V2 sobre todo 2024.** Los porcentajes usan las medias de RMSE por trayectoria; no promedian cocientes entre semillas.

| Red | Reducción de RMSE de relativa frente a absoluta (%) | Reducción de RMSE relativa frente a persistencia anual (%) |
|---|---:|---:|
| MLP | 42,099 | -26,313 |
| LSTM | 34,375 | -32,292 |
| CNN1D | 41,760 | -30,082 |

En el semestre común, CNN1D relativa obtiene RMSE 0,0046732 frente a 0,0048225 de persistencia: reducción puntual de **3,095 %**. Es inferior al margen del 5 %, pertenece a un subperiodo ya expuesto y no tiene aquí un contraste temporal de superioridad registrado/ejecutado. No se declara cumplida la hipótesis de mejora ni se generaliza esa diferencia al año completo.

#### 5.3.2.2. Clasificación, etiquetado secuencial y segmentación

Estas modalidades **no aplican al proyecto actual**:

- **Clasificación:** se predice volatilidad continua, sin clases ni probabilidades de clase.
- **Etiquetado secuencial:** no se predicen etiquetas por token ni entidades delimitadas.
- **Segmentación:** no hay imágenes de entrada, máscaras ni etiquetas por píxel.

Para documentar la guía, cuando la métrica principal sea F1-score, AUC-ROC, F1 de entidad, mIoU o Dice y **un valor mayor sea mejor**, sean $\bar S_{m,k,c}$ y $\bar S_{r,k,c}$ sus medias por corrida bajo la misma evaluación. La diferencia en unidades de la métrica y la mejora relativa frente a la referencia r serían:

$$
\Delta_{\mathrm{abs},k}(m;r,c)=\bar S_{m,k,c}-\bar S_{r,k,c}.
$$

$$
\Delta_{\mathrm{rel},k}(m;r,c)=100\,
\frac{\bar S_{m,k,c}-\bar S_{r,k,c}}{\bar S_{r,k,c}},
\qquad \bar S_{r,k,c}>0.
$$

«Diferencia absoluta» designa aquí el cambio en unidades de la métrica, **con signo**; no el valor absoluto matemático. Un valor positivo favorece a m. Se declararán la escala, el promedio y el referente: si la métrica se expresa en porcentaje, la diferencia de esos valores se comunica en puntos porcentuales, mientras la segunda expresión es un cambio porcentual relativo. El mejor y el peor se identifican por la mayor y la menor puntuación de esa métrica, respectivamente, conservando los empates. Con referencia cero, la diferencia relativa queda indefinida y se informa el motivo, sin añadir epsilon; la diferencia en unidades puede seguir definida.

Se conservarán las mismas muestras y repeticiones emparejadas. La razón entre medias no se confundirá con la media de razones por semilla, y la incertidumbre de una razón recalculará conjuntamente numerador y denominador. Estas definiciones no aportan resultados de clasificación, etiquetado o segmentación del proyecto.

#### 5.3.2.3. Secuencia a secuencia

**No aplica al proyecto actual:** la salida es una volatilidad escalar y no una secuencia generada. Para una tarea de generación se documentarían las diferencias en unidades y relativas de su métrica principal —por ejemplo, BLEU, ROUGE, chrF o exactitud de secuencia completa— con la convención de §5.3.2.2 cuando un valor mayor sea mejor.

Se declararán la escala, tokenización, versión, referencias, parámetros y agregación de la métrica antes de comparar puntuaciones. Se identificarán los referentes mejor y peor de esa métrica y se mantendrán el emparejamiento y el tratamiento de referencias cero. No se mezclarán escalas o configuraciones de cálculo distintas ni se presentarán esas diferencias como mediciones propias de volatilidad.

#### 5.3.2.4. Tamaño de efecto de cada diferencia

En el proyecto, las cantidades m siguientes son métricas de error, como RMSE. Para puntuaciones donde un valor mayor sea mejor se declarará el sentido del signo correspondiente. Una alternativa basada en rangos es la correlación rango-biserial pareada, con su convención de ceros y empates explícita.

Cada diferencia relevante se acompañará de una medida de efecto apropiada y su definición. Para diferencias pareadas por ejecución $d_s=m_{A,s}-m_{B,s}$ puede informarse:

$$
d_z=\frac{\bar d}{s_d},
$$

con signo, denominador y unidad de emparejamiento identificados. Si $s_d=0$, el cociente no está definido. No se sustituirá la desviación de las diferencias por la desviación de un solo modelo. Si se usa una medida basada en rangos, se especificará el cálculo y el tratamiento de empates.

El efecto entre semillas mide esa variabilidad condicionada a los mismos datos. Para efectos sobre errores temporalmente dependientes se justificará el método de bloques o diseño correspondiente. No se atribuirá independencia a todos los días por disponer de muchas predicciones.

Con $d_s=m_{A,s}-m_{B,s}$, un $d_z$ negativo favorece a A; la reducción porcentual anterior usa el signo opuesto, positiva cuando mejora m. Se rotularán ambas convenciones. El d_z pareado no se calcula con una desviación marginal o combinada de grupos independientes. Tampoco se sustituye automáticamente por el delta de Cliff basado en todos los cruces A×B: eso perdería el emparejamiento. El efecto de rangos elegido deberá conservar las parejas y su convención de ceros/empates. Los umbrales convencionales de efecto pequeño o grande no reemplazan el margen práctico.

### 5.3.3. Inferencia sobre las diferencias



#### 5.3.3.1. Comparaciones pareadas y corrección por multiplicidad

**Resultados y decisiones originales V1.**

La familia prevista contiene **nueve contrastes entre semillas**: tres pares bilaterales entre redes, tres comparaciones bilaterales red–persistencia y tres contrastes unilaterales de ganancia relativa superior al 5 % frente a persistencia. Se aplica Holm conjuntamente a los nueve, sin dividirlos según sus resultados. Cada pareja de redes conserva la misma semilla y calendario; la persistencia es un valor fijo, no treinta ajustes independientes.

Son análisis de desarrollo: el calendario de 2024 ya se había expuesto al EDA y al diseño. Registrar el protocolo antes de entrenar y ajustar multiplicidad no vuelve confirmatorio ese calendario ni elimina el optimismo de selección. Una evaluación externa requerirá el test reservado o un procedimiento externo que haya aislado todo el diseño.

**Familia V2 independiente por pregunta:** tres contrastes bilaterales, uno por arquitectura, de RMSE(`relative_window`)−RMSE(`absolute_log`) sobre las diez mismas semillas base y las mismas 366 fechas; ajuste Holm-3. La multiplicidad no se divide por signo o resultado y no se suman estas pruebas como confirmaciones independientes de V1. No se ejecutaron contrastes pareados contra V1, contra persistencia ni entre arquitecturas de V2.

#### 5.3.3.2. Diferencia media, IC del 95 % y p-value ajustado y sin ajustar

**Resultados y decisiones originales V1.**

Para cada par se usa Δ=RMSE(A)−RMSE(B); un valor negativo favorece A. Los IC son t marginales del 95 %, con 29 grados de libertad, y los efectos son d_z pareados. Los p-valores numéricos se calculan con la referencia t y sus condiciones, sin presentarlos como pruebas exactas de aleatorización.

| A − B | Δ media | IC 95 % marginal | d_z | p sin ajustar | p Holm, familia 9 |
|---|---:|---|---:|---:|---:|
| MLP − LSTM | 0,00708685 | [0,00598691; 0,00818678] | 2,40585 | 9,02388×10⁻¹⁴ | 5,41433×10⁻¹³ |
| MLP − CNN1D | −0,00159638 | [−0,00373267; 0,00053991] | −0,27903 | 0,137264 | 0,549057 |
| LSTM − CNN1D | −0,00868323 | [−0,01058312; −0,00678333] | −1,70661 | 2,98184×10⁻¹⁰ | 1,49092×10⁻⁹ |
| MLP − Persistencia | 0,01116783 | [0,01010196; 0,01223370] | 3,91241 | 2,50849×10⁻¹⁹ | 2,00679×10⁻¹⁸ |
| LSTM − Persistencia | 0,00408098 | [0,00389972; 0,00426225] | 8,40675 | 1,13894×10⁻²⁸ | 1,02504×10⁻²⁷ |
| CNN1D − Persistencia | 0,01276421 | [0,01082544; 0,01470298] | 2,45838 | 5,25204×10⁻¹⁴ | 3,67643×10⁻¹³ |

Fuente completa: [paired_tests.csv](../_static/dl_benchmark_20261104/analysis/paired_tests.csv), con estadísticos, diagnósticos y los nueve contrastes. En los tres unilaterales de mejora superior al 5 %, p sin ajustar y Holm son 1,000: las estimaciones favorecen a persistencia. No rechazar esa superioridad no demuestra equivalencia.

LSTM supera a las otras dos redes bajo este análisis condicionado a las semillas; MLP–CNN1D es inconcluso. Los diagnósticos Shapiro–Wilk de MLP–LSTM (p=0,020789) y MLP–persistencia (p=0,016355) muestran desviaciones de normalidad que limitan la calibración de la referencia t. Se conserva el análisis previamente elegido y se informa la limitación; no se sustituye por la prueba que resulte más favorable.

**V2: contraste controlado entre procedimientos CV.** Δ<0 favorece `relative_window`; t e IC se refieren a diez diferencias de RMSE por trayectoria, con nueve grados de libertad.

| Red | Δ RMSE relativa − absoluta | IC 95 % marginal | t(9) | d_z | p sin ajustar | p Holm, familia 3 |
|---|---:|---|---:|---:|---:|---:|
| MLP | -0,00439825 | [-0,00469839; -0,00409810] | -33,14874 | -10,48255 | 1,019002e-10 | 3,057005e-10 |
| LSTM | -0,00331851 | [-0,00361490; -0,00302213] | -25,32833 | -8,00952 | 1,121134e-09 | 2,242268e-09 |
| CNN1D | -0,00446676 | [-0,00506264; -0,00387089] | -16,95733 | -5,36238 | 3,871572e-08 | 3,871572e-08 |

Los tres contrastes favorecen la variante relativa bajo la referencia t y su alcance condicionado al calendario. Shapiro–Wilk sobre las diferencias da p=0,425763, 0,237020 y 0,573491 para MLP, LSTM y CNN1D; no rechazarlo no demuestra normalidad. Los d_z grandes describen variación entre estas inicializaciones y no utilidad financiera o independencia de mercados.

El cambio de R² se resume como intervalo secundario descriptivo, **sin una segunda familia de pruebas seleccionada por R²**:

| Red | Δ R² relativa − absoluta | IC 95 % marginal descriptivo |
|---|---:|---|
| MLP | 0,763286 | [0,687570; 0,839002] |
| LSTM | 0,557666 | [0,498273; 0,617059] |
| CNN1D | 0,798333 | [0,665593; 0,931073] |

Fuente: [comparisons.csv V2](../_static/dl_r2_cv_20261104/analysis/comparisons.csv). La comparación evalúa conjuntamente representación relativa y ponderación de escalado, manteniendo política CV y configuraciones; no aísla un mecanismo causal único. No demuestra superioridad del 5 % frente a persistencia ni generalización temporal.

### 5.3.4. Umbral de mejora para modelos avanzados

Se distinguirán dos cantidades:

#### 5.3.4.1. Mejora mínima detectable frente al baseline

- **Mejora mínima detectable:** depende del diseño, la variabilidad pareada y la potencia; se estima con el procedimiento de §4.1, respetando sus supuestos.

La desviación entre corridas o la anchura de un IC por sí solas no bastan para declarar una mejora detectable: se necesitan contraste, potencia, emparejamiento y variabilidad de las diferencias. Si la mejora detectable supera el margen registrado para el contraste correspondiente, se ampliará el diseño o se explicitará esa limitación.

Para una propuesta avanzada A frente a un baseline B, en la misma unidad pareada:

$$
\operatorname{Var}(m_A-m_B)=\operatorname{Var}(m_A)
+\operatorname{Var}(m_B)-2\operatorname{Cov}(m_A,m_B).
$$

La DE o el IC de B aislado no identifica esa varianza ni una MDE verificable de A−B. Se necesitan estimando, política de entrenamiento o referencia fija, emparejamiento y datos de planificación o supuestos justificados sobre ambas cantidades. No existe una MDE numérica válida para una propuesta avanzada todavía no definida ni ejecutada. La planificación numérica disponible corresponde a cada red frente a persistencia (§5.4.7), no a una futura red avanzada frente a MLP.

Se mantendrá la distinción de §4.1.5–§4.1.7: detectar un efecto frente a cero es distinto de demostrar que supera un margen práctico. La MDE por semillas es condicional al calendario y no acredita por sí sola potencia temporal ni potencia conjunta de toda una familia. El umbral avanzado no se define sumando una DE al baseline o tomando un extremo de su IC.

#### 5.3.4.2. Margen de relevancia práctica

**Resultados y decisiones originales V1.**

La regla registrada es una **reducción relativa de RMSE del 5 % frente a persistencia**. Con RMSE de persistencia 0,00482247875 en la calibración, su conversión de planificación es **δ=0,00024112394** y el nivel puntual correspondiente sería RMSE≤0,00458135481. Esta conversión usa desarrollo expuesto, no test, y ninguna red se aproxima a ese nivel en esta ejecución.

Detectar un efecto de esa magnitud frente a cero no es demostrar que la ganancia supera el 5 %. Los contrastes unilaterales sobre la frontera 0,05 se conservan separados del dimensionamiento aproximado frente a cero. La estimación puntual tampoco sustituye un contraste e intervalo sobre el margen.

El 5 % no se traslada automáticamente a una comparación entre redes ni a un modelo avanzado frente al baseline MLP. Esa comparación requerirá un margen aplicado propio, estimando, política de entrenamiento y planificación pareada registrados antes de evaluarla. La RMSE o el extremo de un IC de MLP no bastan para fijar ese umbral.

**V2:** no se registró un margen nuevo a partir de sus cifras. Persistencia sigue siendo mejor en todo 2024; la mejora puntual de CNN1D relativa del 3,095 % en julio–diciembre no alcanza el 5 % ni sustituye un contraste sobre ese margen. Un R² positivo o un p-valor pequeño contra `absolute_log` no resuelve esa comparación.

#### 5.3.4.3. Cota de referencia del problema

Cuando exista una cota de referencia verificable del problema, se discutirá junto con sus supuestos y su comparabilidad. Actualmente no se ha estimado ruido irreducible ni una cota de error para esta volatilidad financiera. La persistencia aporta contexto, pero **no es una cota teórica de error mínimo**; un resultado publicado en otro dataset tampoco lo es.

El acuerdo entre anotadores no aplica porque la etiqueta se construye mediante una fórmula sobre retornos, sin anotación humana. No se dispone de una medición de desempeño humano comparable que sirva como cota.

### 5.3.5. Interpretación



#### 5.3.5.1. Significación estadística y relevancia práctica

**Resultados y decisiones originales V1.**

Entre semillas, las diferencias LSTM–MLP y LSTM–CNN1D son distinguibles con Holm bajo los supuestos t; MLP–CNN1D no lo es. Las tres redes tienen mayor error que persistencia y ninguna cumple la mejora práctica del 5 %. Por tanto, una ventaja relativa de LSTM dentro del grupo DL no representa mejora útil frente a la referencia ingenua en estas fechas.

La inferencia temporal y el remuestreo por bloques aportan otra perspectiva sobre las mismas predicciones, no confirmaciones independientes. No se afirma equivalencia, causalidad arquitectónica ni generalización a 2025–2026, otros activos o proveedores.

**V2:** la reducción de error relativa–absoluta es distinguible bajo los tres contrastes entre semillas, pero persistencia mantiene menor error anual. La única ventaja puntual de CNN1D sobre persistencia aparece en el semestre común, es inferior al 5 % y carece de confirmación temporal en este ciclo. La interpretación cambia según referente y calendario, que se mantienen visibles.

#### 5.3.5.2. Desempeño frente a complejidad, parámetros y costo

**Resultados y decisiones originales V1.**

LSTM reduce RMSE respecto de MLP un 44,320 % en medias de julio–diciembre, a costa de 8,47 veces sus parámetros y una inferencia mayor. Su entrenamiento observado es más corto por el procedimiento y las épocas efectivas; no se interpreta como eficiencia arquitectónica general. CNN1D no aporta una mejora frente a MLP en las medias observadas y presenta mayor variabilidad. Ninguna red justifica una mejora predictiva frente a persistencia en este experimento.

**V2:** se obtiene una mejora de la variante relativa sin añadir parámetros. Aumenta su tiempo de entrenamiento en las tres redes bajo las mediciones concurrentes y cambian épocas efectivas; no se infiere una relación causal simple entre número de pesos, costo y R².

#### 5.3.5.3. Conclusión explícita del baseline y margen disponible

**Resultados y decisiones originales V1.**

El **baseline DL operacional de desarrollo es MLP, D=7, H=16, LR=0,001**, elegido en enero–junio. Su RMSE posterior es 0,0159903 ± 0,0028545; LSTM lidera descriptivamente las redes con 0,0089035 y persistencia obtiene 0,0048225. No se proclama un baseline ganador confirmado en test.

No hay un margen cuantificado para una arquitectura avanzada frente a MLP. El único margen práctico fijado, 5 % frente a persistencia, no se alcanza; el piloto tampoco acredita sensibilidad de 0,80 para detectar su magnitud con treinta semillas. Los residuos motivan investigación, sin cuantificar una mejora aprovechable garantizada.

**Actualización V2:** MLP relativo lidera descriptivamente el año CV y CNN1D relativo el semestre común; la referencia operacional V1 no se reselecciona. Los R² positivos aportan una corrección empírica dentro del desarrollo, pero persisten el límite frente a persistencia anual, la falta de un margen avanzado y la ausencia de confirmación en el test reservado.

#### 5.3.5.4. Estado del benchmark

| Producto | Estado verificable |
|---|---|
| Tres modelos y repeticiones | Ejecutados: 30 semillas por red, 90 corridas finales |
| Búsqueda común | 96 corridas; D=7; MLP seleccionado antes del piloto final |
| Piloto | 5 semillas por red; requerimiento máximo 1.212; límite registrado 30 |
| Métricas, tiempos y parámetros | Medidos; CSV y predicciones disponibles |
| Ranking posterior | LSTM, MLP, CNN1D; persistencia mejor que las tres |
| Inferencia | t pareada, Friedman, Holm, DM-HLN, bootstrap y diagnósticos ejecutados con alcance de desarrollo |
| Margen práctico | 5 % frente a persistencia; no alcanzado |
| Potencia del 80 % para esa magnitud | No acreditada con el límite de recursos |
| Baseline operacional | MLP congelado por búsqueda; no reseleccionado con julio–diciembre |
| Confirmación en test | No ejecutada; 2025–2026 sigue reservado |

La ejecución completa reúne 201 corridas y 402 ajustes al contar búsqueda, piloto, parada interna y reajuste. Es un benchmark medido de desarrollo con limitaciones explícitas, no una evaluación del test ni la resolución del respaldo bibliográfico pendiente.

**Estado adicional V2-CV:**

| Producto V2 | Estado verificable |
|---|---|
| CV con índices de Filipe | Doce bloques mensuales reales y 366 fechas por trayectoria |
| Dos variantes × tres redes × diez semillas | 60 trayectorias completas, 720 segmentos y 1.440 ajustes |
| Predicciones y métricas | 21.960 predicciones; resumen anual, semestre común y 720 filas por fold |
| Contraste controlado | Tres t pareadas relativa–absoluta, Holm-3, e IC descriptivos de Δ R² |
| DM/BDS/Ljung–Box/bootstrap V2 | No ejecutados; sus resultados V1 no se trasladan |
| Selección y potencia | Sin HPO nuevo, sin reselección de referencia y sin nuevo piloto de potencia |
| Test y ajuste final | Test 2025–agosto de 2026 cerrado; ajuste final de desarrollo aún no ejecutado |

Archivos V2: [corridas](../_static/dl_r2_cv_20261104/runs.csv), [predicciones](../_static/dl_r2_cv_20261104/predictions.csv), [resumen del análisis](../_static/dl_r2_cv_20261104/analysis/summary.json) y {download}`alcance V2 <../_static/dl_r2_cv_20261104/analysis/README.md>`.

## 5.4. Análisis Estadístico y Pruebas de Hipótesis

**Separación de ciclos.** Las pruebas originales t/Friedman/Holm-9, DM-HLN/Holm-6, BDS/Ljung–Box/Holm-15 y bootstrap por bloques de esta sección corresponden a **V1**. Para V2 solo se ejecuta la comparación controlada de sus dos variantes con las mismas diez semillas por arquitectura: tres contrastes de RMSE, Holm-3, e IC descriptivos de diferencia de R². No se reutilizan p-valores de V1 para calificar V2.

El análisis distingue variación entre entrenamientos, diferencias de pérdidas por fecha y diagnósticos residuales. Se utilizan las mismas predicciones de desarrollo para preguntas distintas; sus resultados no constituyen confirmaciones independientes ni evidencia del test reservado.

### 5.4.1. Hipótesis, unidad de análisis y supuestos

**V1: resultados y alcance originales.**

El protocolo inicial fijó RMSE, h=1, $\alpha=0{,}05$, direcciones y familias antes del primer entrenamiento DL, pero después de conocer el desarrollo histórico. El protocolo final conserva el límite de treinta semillas y registra el resultado del piloto. Son inferencias exploratorias de desarrollo, con los siguientes estimandos:

| Pregunta | Unidad y agregación ejecutadas | Familia principal |
|---|---|---|
| Variación por entrenamiento | Diferencias de métricas de las semillas 1001–1030; mismo calendario | 9 contrastes, Holm |
| Pérdida predictiva en el tiempo | 184 diferencias diarias de pérdidas cuadráticas promediadas sobre semillas | 6 pares, Holm; HAC 14 |
| Estructura residual | 184 residuos de la semilla predefinida 1001, sin concatenar semillas | 15 diagnósticos, Holm |

Se calcularon Shapiro–Wilk sobre diferencias pareadas, ACF residual y las verificaciones de continuidad/cobertura. No se publican gráficos Q–Q ni se presenta Shapiro como prueba suficiente de normalidad o de independencia. Las desviaciones de normalidad señaladas en §5.3.3 limitan la referencia t, y los diagnósticos temporales documentan dependencia.

Las treinta inicializaciones aportan variación del entrenamiento condicionada a estos datos, no treinta mercados independientes. Las 184 fechas tampoco se consideran i.i.d., por el solapamiento de σ7 y la dependencia adicional. Las configuraciones D no se convierten en bloques intercambiables ni las filas semilla×fecha en un tamaño muestral independiente de 5.520.

**Unidad V2:** diez trayectorias anuales por red/variante, emparejadas por semillas base; la diferencia principal es RMSE(`relative_window`)−RMSE(`absolute_log`). Los tres contrastes t bilaterales forman una familia Holm propia. V1 solo aparece como contexto descriptivo en las 184 fechas comunes: sus semillas y procedimiento de ajuste difieren y no se calculan p-valores pareados V1–V2.

### 5.4.2. Comparaciones pareadas: t y Wilcoxon

**V1: resultados y alcance originales.**

Se ejecutó **t pareada**, con 30 pares y 29 grados de libertad, para las diferencias de RMSE entre redes. Frente a persistencia, el mismo cálculo compara la métrica de cada semilla con una referencia determinista, sin inventar repeticiones independientes de esa referencia. El resultado completo, incluidos estadístico, IC y Shapiro–Wilk, está en [paired_tests.csv](../_static/dl_benchmark_20261104/analysis/paired_tests.csv).

La t contrasta la media de las diferencias; no exige igualdad de varianzas marginales de las redes. La normalidad pertinente es la de las diferencias y la independencia se refiere a los entrenamientos condicionados al calendario. Las desviaciones diagnosticadas acompañan los resultados; no se seleccionó después una prueba según su p-valor.

**Wilcoxon no se ejecutó.** Su uso requeriría definir hipótesis de desplazamiento, simetría pertinente, tratamiento de ceros/empates y cálculo exacto o aproximado; no es un reemplazo automático de la prueba de la media. Con cinco pares sin ceros ni empates, el menor p bilateral exacto es $2/2^5=0{,}0625$; con seis es $0{,}03125$, todavía mayor que $0{,}05/3$. Por eso cumplir un mínimo de semillas no garantiza potencia tras multiplicidad. Aquí se ejecutaron treinta semillas, pero tampoco se alcanzó el tamaño exigido por el piloto.

**V2:** las tres comparaciones dentro de cada arquitectura ejecutadas con diez semillas y Holm-3 están en §5.3.3.2. Se contrasta diferencia media de RMSE entre los dos procedimientos CV; no se ejecutó Wilcoxon ni se emparejaron semillas de V1 con V2.

### 5.4.3. Comparación conjunta de tres modelos

**V1: resultados y alcance originales.**

Se ejecutó **Friedman** sobre treinta bloques completos de semillas de las tres redes: estadístico χ² aproximado **45,26667**, con dos grados de libertad y **p=1,48070×10⁻¹⁰**. El resumen está en [summary.json](../_static/dl_benchmark_20261104/analysis/summary.json).

La conclusión es condicional al calendario común: no acredita independencia de mercados ni identifica por sí sola qué par difiere. Se conservan treinta bloques completos sin imputación ni selección de semillas. ANOVA de medidas repetidas no se ejecutó; habría requerido justificar la estructura de covarianza y las correcciones correspondientes.

### 5.4.4. Comparaciones post-hoc y corrección por multiplicidad

**V1: resultados y alcance originales.**

Las comparaciones pareadas principales ya estaban previstas y **no dependen de que Friedman actúe como puerta de entrada**. Se aplicó Holm a los nueve contrastes entre semillas; los seis pares temporales y los quince diagnósticos tienen sus familias diferenciadas por pregunta, registradas antes de interpretarlos.

Nemenyi y Bonferroni como análisis alternativos de resultados no se ejecutaron. El nivel conservador α=0,05/9 solo se utilizó para planificar tamaño/MDE, mientras los p-valores observados se ajustan por Holm. No se añade automáticamente una segunda corrección a un procedimiento ya simultáneo. La corrección de multiplicidad no repara supuestos inválidos ni convierte desarrollo expuesto en confirmación externa.

**V2:** sus tres comparaciones principales de preprocesamiento fueron declaradas en el nuevo protocolo y usan Holm-3. No son post-hoc de un Friedman V2: tal omnibus no se ejecutó. No se heredan los ajustes Holm-9, Holm-6 ni Holm-15 del ciclo original.

### 5.4.5. Intervalos de confianza y tamaño de efecto



#### 5.4.5.1. Bootstrap de las diferencias pareadas

**V1: resultados y alcance originales.**

Se ejecutó un **bootstrap circular pareado de fechas**, con 2.000 réplicas, semilla de remuestreo 20261104 y longitud principal de 14 días. Las sensibilidades de 7 y 28 días usan el mismo procedimiento y permanecen identificadas en [block_bootstrap.csv](../_static/dl_benchmark_20261104/analysis/block_bootstrap.csv). Se remuestrean juntas etiquetas y predicciones de todos los métodos; cada réplica calcula la RMSE de cada semilla y después su media. Las razones recalculan conjuntamente el denominador de persistencia.

| A − B | Diferencia de RMSE | IC percentil 95 %, bloques de 14 días |
|---|---:|---|
| MLP − LSTM | 0,0070868463 | [0,0030695119; 0,01100505] |
| MLP − CNN1D | -0,0015963798 | [-0,0019706962; -0,0012442331] |
| MLP − Persistencia | 0,01116783 | [0,0076314113; 0,014521738] |
| LSTM − CNN1D | -0,0086832262 | [-0,012660688; -0,0044586473] |
| LSTM − Persistencia | 0,0040809842 | [0,0022340656; 0,0057947357] |
| CNN1D − Persistencia | 0,01276421 | [0,0090669865; 0,016202049] |

Son intervalos marginales de variación temporal de evaluación **condicionada a los modelos almacenados**: no se reentrena ni se remuestrean semillas, por lo que no incorporan conjuntamente toda la incertidumbre del entrenamiento ni garantizan generalización temporal.

Para MLP–CNN1D, el IC temporal excluye cero aunque el IC entre semillas lo incluye. No hay contradicción: una unidad cambia las fechas manteniendo los entrenamientos observados; la otra cambia inicializaciones manteniendo las fechas. Se publican ambos sin seleccionar el que favorezca una conclusión. El IC de diferencias de RMSE tampoco es el IC de una diferencia de pérdidas cuadráticas de DM.

#### 5.4.5.2. Tamaño de efecto junto con el p-value

**V1: resultados y alcance originales.**

Se informa d_z pareado para cada diferencia de métricas en §5.3.3 y en el CSV, usando la DE de las diferencias, no una DE marginal o combinada de grupos independientes. Δ y d_z negativos favorecen A; la reducción porcentual usa la convención opuesta.

Los efectos LSTM–MLP y LSTM–CNN1D reflejan diferencias grandes respecto de la variación entre estas inicializaciones, sin cuantificar utilidad financiera o variación entre mercados. El margen aplicado sigue siendo el 5 % frente a persistencia. No se calculó delta de Cliff ni rango-biserial; sustituir el emparejamiento por todos los cruces A×B cambiaría el estimando.

### 5.4.6. Advertencia sobre independencia



#### 5.4.6.1. Ejecuciones que comparten el conjunto de prueba

**V1: resultados y alcance originales.**

El nombre del inciso corresponde al requisito general; **esta ejecución comparte desarrollo/calibración, no el test reservado**. Las treinta semillas de cada red comparten las mismas 184 fechas, etiquetas, configuración y entrenamiento admitido. Sus IC y pruebas describen la variación entre entrenamientos bajo ese calendario.

Las etiquetas consecutivas σ7 comparten seis retornos; aumentar semillas no aumenta el número de periodos de mercado observados. Las predicciones conservan identificadores de fecha, modelo y semilla, y no se agrupan como observaciones i.i.d. El carácter determinista de persistencia también se conserva.

#### 5.4.6.2. Validación repetida y remuestreo por bloques

**V1: resultados y alcance originales.**

No se ejecutó validación cruzada repetida ni K-fold aleatorio. Se mantuvo validación interna cronológica y un ajuste estático; no se aplicó una corrección de varianza derivada para otro diseño como si resolviera la dependencia temporal.

El remuestreo temporal ejecutado usa bloques circulares pareados de 14 días, con 7 y 28 como sensibilidad. Mantiene la dependencia local aproximada y las parejas entre métodos, pero los bordes circulares y la duración limitada del tramo son supuestos del procedimiento. Las frecuencias de ranking y los IC de bloques son evidencia condicionada a las predicciones guardadas, no garantías de independencia o estabilidad futura.

### 5.4.7. Reporte de potencia y diferencia mínima detectable

**V1: resultados y alcance originales.**

El piloto usó cinco semillas distintas de las finales —11, 23, 37, 53 y 71— sobre la misma calibración de julio–diciembre. Para detectar frente a cero una diferencia de magnitud $\delta=0{,}00024112394$, equivalente al 5 % de la RMSE de persistencia en desarrollo, se planificó potencia 0,80 y α conservador 0,05/9 mediante la aproximación normal pareada de §4.1.

| Red | DE piloto de RMSE frente a referencia fija | R requerido aproximado | R final | MDE frente a cero con R=30, potencia objetivo 0,80 |
|---|---:|---:|---:|---:|
| MLP | 0,00232148 | 1.212 | 30 | 0,00153200 |
| LSTM | 0,00087432 | 172 | 30 | 0,00057698 |
| CNN1D | 0,00179438 | 724 | 30 | 0,00118415 |

Fuente: [pilot_summary.json](../_static/dl_benchmark_20261104/pilot_summary.json). Prevaleció el **límite de recursos registrado de treinta semillas**, por debajo del máximo requerido de 1.212. Se declara expresamente el incumplimiento del tamaño que exigiría la guía para esa potencia. Las tres MDE superan δ: **no se acredita potencia del 80 % para detectar una magnitud del 5 %**. Las estimaciones aproximadas de planificación para esa magnitud con R=30 son 0,0142, 0,1034 y 0,0211, respectivamente; no son potencias retrospectivas calculadas con los efectos finales.

Esta planificación condiciona el calendario y supone normalidad de diferencias y la referencia fija. No dimensiona automáticamente fechas dependientes, DM, BDS ni una propuesta avanzada frente a MLP. Detectar una diferencia frente a cero tampoco acredita que supere el margen práctico del 5 %; esa frontera requiere su alternativa y planificación propias.

**V2 no hereda ese dimensionamiento:** diez semillas responden al presupuesto de la exploración CV, no a un nuevo piloto ni a una acreditación de potencia de 0,80. No se añade un margen práctico nuevo ni se aplica automáticamente el 5 % frente a persistencia a una comparación entre preprocesamientos.

### 5.4.8. Ausencia de significación y equivalencia

**No rechazar una diferencia no demuestra equivalencia.** Si se desea afirmar equivalencia entre modelos, se planteará un análisis separado, por ejemplo **TOST** con margen práctico predefinido. Una alternativa bayesiana con **ROPE** deberá declarar modelo, distribución posterior y regla de decisión; no se presentará como si fuera la misma prueba frecuentista.

El margen de equivalencia debe fijarse con sentido aplicado, escala, referencia y unidades claras. El 5 % de mejora frente a persistencia es un criterio de superioridad práctica y no define automáticamente una banda de equivalencia entre dos modelos. No se ensanchará después para conseguir equivalencia. Se distinguirán ausencia de evidencia, evidencia de diferencia y evidencia de equivalencia bajo el diseño correspondiente.

En un TOST al nivel 0,05, bajo el método y la calibración aplicables, ambas pruebas unilaterales deben rechazar; su regla usual corresponde al **IC del 90 % completamente dentro de la banda**. Se distinguirá ese intervalo de decisión del IC del 95 % requerido para el reporte general. Si cambia el nivel por comparación debido a multiplicidad, el intervalo de decisión debe corresponder al procedimiento realmente utilizado.

Las dos unilaterales de un TOST forman una prueba de intersección–unión; no se les aplica por defecto Bonferroni como si fueran dos afirmaciones independientes. Las equivalencias entre varios pares o configuraciones sí requieren su tratamiento de familia. Para ROPE se registrarán también dependencia, priors, posterior y criterio de decisión, sin inventar ahora sus valores.

**Resultado actual:** MLP–CNN1D no rechaza la igualdad de medias entre semillas con Holm (p=0,549057); se informa como inconcluso bajo ese diseño. No se ejecutó TOST ni ROPE y no se afirma equivalencia. Los contrastes unilaterales frente al 5 % tampoco son pruebas de equivalencia.

### 5.4.9. Test de Diebold–Mariano



#### 5.4.9.1. Comparación de precisión predictiva por horizonte

**V1: resultados y alcance originales.**

Se ejecutó Diebold–Mariano únicamente para el horizonte registrado **h=1**, sobre las 184 fechas consecutivas de julio–diciembre. La pérdida es cuadrática, coherente con RMSE. Para errores $e_{A,s,t}=Y_t-\widehat Y_{A,s,t}$, la serie registrada es la diferencia de pérdidas promediadas entre semillas:

$$
d_t=\frac1{30}\sum_{s=1001}^{1030}
\left(e_{A,s,t}^2-e_{B,s,t}^2\right),\qquad H_0:\mathbb E[d_t]=0.
$$

Una media negativa favorece A. Persistencia aporta su misma pérdida determinista a cada pareja; no se trata como treinta procesos entrenados. La serie no es la pérdida del promedio de predicciones ni concatena 184×30 observaciones. Tampoco contrasta directamente la media de RMSE por semilla usada en el ranking, pues promediar raíces y tomar la raíz de una pérdida media son operaciones distintas.

No se eligió la semilla de menor error para ejecutar DM. Agregar horizontes exigiría registrar su papel y la ampliación de la familia antes de evaluar.

#### 5.4.9.2. Corrección Harvey–Leybourne–Newbold y pérdida coherente

**V1: resultados y alcance originales.**

El análisis principal usa **HAC de Bartlett con rezago 14**, h=1 y la corrección Harvey–Leybourne–Newbold. Las sensibilidades HAC 7 y 28 se conservan separadas; no se eligió el rezago por su significación. Los p-valores bilaterales usan una referencia t con T−1=183 grados de libertad, **aproximada**, con Holm sobre los seis pares principales.

Para T fechas consecutivas y varianza de largo plazo admisible $\widehat\Omega_d>0$:

$$
DM=\frac{\sqrt T\,\bar d}{\sqrt{\widehat\Omega_d}},\qquad
DM_{\mathrm{HLN}}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}.
$$

Con h=1 el factor es $\sqrt{(T-1)/T}$. **h no se sustituye por n=7 ni D=7**, y ese factor no corrige por sí solo la dependencia de σ7 ni garantiza calibración de cualquier HAC.

| A − B | Diferencia media de pérdida cuadrática | DM-HLN | p sin ajustar | p Holm, familia 6 |
|---|---:|---:|---:|---:|
| MLP − LSTM | 0,00018406691 | 2,893437 | 0,004273305 | 0,00663043 |
| MLP − CNN1D | -7,1784771e-05 | -4,807961 | 3,171305e-06 | 1,902783e-05 |
| MLP − Persistencia | 0,00024031006 | 4,02422 | 8,35857e-05 | 0,0003343428 |
| LSTM − CNN1D | -0,00025585168 | -3,306286 | 0,001137843 | 0,003413529 |
| LSTM − Persistencia | 5,6243148e-05 | 2,975981 | 0,003315215 | 0,00663043 |
| CNN1D − Persistencia | 0,00031209483 | 4,277476 | 3,041587e-05 | 0,0001520794 |

Fuente: [temporal_dm.csv](../_static/dl_benchmark_20261104/analysis/temporal_dm.csv). En la especificación principal se rechaza igualdad de pérdidas para los seis pares: LSTM favorecida frente a MLP/CNN1D, MLP frente a CNN1D y persistencia frente a todas. Esto no contradice el resultado inconcluso MLP–CNN1D entre semillas: cambian estimando y unidad de incertidumbre.

La referencia requiere momentos y dependencia adecuados para estimar la varianza de largo plazo; un tramo de 184 fechas, deriva, modelos seleccionados y etiquetas solapadas limitan la interpretación. HAC, HLN y Holm no convierten esa evidencia expuesta en confirmación externa ni prueban superación del 5 %. Las sensibilidades no se presentan como pruebas principales adicionales.

### 5.4.10. Test BDS (Brock–Dechert–Scheinkman)

**V1: resultados y alcance originales.**

Se ejecutó BDS sobre los residuos ordenados de la **semilla predefinida 1001**, h=1 y las mismas 184 fechas, sin prefiltrado ni agrupación de semillas. Se usan dimensiones 2 y 3, con 183 y 182 vectores de inmersión respectivamente, y ε=1,5 veces la DE muestral residual de cada red. Los seis BDS y nueve Ljung–Box forman la familia de quince diagnósticos ajustada con Holm.

| Red | ε | Dimensión | Estadístico BDS | p sin ajustar | p Holm, familia 15 |
|---|---:|---:|---:|---:|---:|
| MLP | 0,014471341 | 2 | 18,5201 | 1,421675e-76 | 9,951727e-76 |
| MLP | 0,014471341 | 3 | 18,37325 | 2,151413e-75 | 1,290848e-74 |
| LSTM | 0,012537396 | 2 | 15,34825 | 3,638829e-53 | 7,277658e-53 |
| LSTM | 0,012537396 | 3 | 15,07379 | 2,408773e-51 | 2,408773e-51 |
| CNN1D | 0,015044008 | 2 | 20,54051 | 9,358426e-94 | 7,48674e-93 |
| CNN1D | 0,015044008 | 3 | 20,63415 | 1,355199e-94 | 1,219679e-93 |

Fuente: [diagnostics.csv](../_static/dl_benchmark_20261104/analysis/diagnostics.csv), con versión, radio, tamaño útil y advertencias. Bajo su referencia normal aproximada, se rechaza i.i.d. en ambas dimensiones de las tres redes.

El rechazo de BDS no identifica exclusivamente no linealidad estructural ni capacidad predictiva aprovechable. Dependencia lineal, heterocedasticidad, deriva, estimación de modelos y etiquetas solapadas también afectan su interpretación y calibración; Holm no repara esas condiciones. No se extrapola el resultado de esta semilla a todos los entrenamientos.

### 5.4.11. Test de Ljung–Box

**V1: resultados y alcance originales.**

Se ejecutó Ljung–Box sobre los mismos residuos de la semilla 1001 en los rezagos registrados **7, 14 y 28**. Se usa **model_df=0**: no se resta el número de pesos de la red como si fueran parámetros ARMA. Los grados de libertad de la referencia χ² son, respectivamente, 7, 14 y 28; sus p-valores son aproximados.

| Red | Rezago | Q | p sin ajustar | p Holm, familia 15 |
|---|---:|---:|---:|---:|
| MLP | 7 | 509,5771 | 6,996479e-106 | 9,095422e-105 |
| MLP | 14 | 576,4649 | 5,40088e-114 | 7,561232e-113 |
| MLP | 28 | 659,4703 | 5,716454e-121 | 8,574681e-120 |
| LSTM | 7 | 352,8016 | 3,097088e-72 | 1,548544e-71 |
| LSTM | 14 | 353,5511 | 7,404031e-67 | 2,961612e-66 |
| LSTM | 28 | 360,8814 | 1,606715e-59 | 4,820145e-59 |
| CNN1D | 7 | 492,8041 | 2,824187e-102 | 2,824187e-101 |
| CNN1D | 14 | 536,6333 | 1,569977e-105 | 1,883972e-104 |
| CNN1D | 28 | 582,9916 | 4,684525e-105 | 5,152978e-104 |

Los nueve resultados rechazan ausencia conjunta de autocorrelación hasta esos rezagos bajo la referencia usada. No son nueve confirmaciones independientes ni demuestran que un modelo avanzado pueda eliminarla. No se ejecutó aquí Ljung–Box sobre residuos al cuadrado; requeriría identificar esa pregunta como diagnóstico adicional.

### 5.4.12. Interpretación de los residuos



#### 5.4.12.1. Estructura residual y posible margen de mejora

**V1: resultados y alcance originales.**

La ACF de la semilla 1001 es elevada en el primer rezago: **0,870991 en MLP, 0,831449 en LSTM y 0,877100 en CNN1D**. En la figura, MLP y CNN1D conservan correlaciones positivas a rezagos más largos; LSTM cae antes hacia cero. Las estimaciones completas están en [acf.csv](../_static/dl_benchmark_20261104/analysis/acf.csv).

![ACF residual de las tres redes en desarrollo, semilla predefinida 1001](../_static/dl_benchmark_20261104/analysis/figures/residual_acf_designated_seed.png)

Ljung–Box y BDS documentan estructura residual y desviaciones de i.i.d. bajo sus referencias aproximadas. El solapamiento del objetivo, la deriva y la heterocedasticidad pueden contribuir; no se atribuye toda esa estructura a falta de capacidad de las redes. No se cuantifica una ganancia futura por observarla ni se identifica un mecanismo causal. Una propuesta avanzada deberá demostrar su aprovechamiento con predicciones y comparación justas en evaluación no usada para diseñarla.

### 5.4.13. Buenas Prácticas Estadísticas



#### 5.4.13.1. Nivel de significancia declarado antes de analizar resultados

Se registró $\alpha=0{,}05$ con familias y direcciones en el protocolo inicial previo al entrenamiento. Se informa el contexto de desarrollo ya conocido y el cierre posterior al piloto; no se presenta ese registro como anterior a todo contacto con los datos. El dimensionamiento usa α=0,05/9, y los contrastes observados usan las familias Holm documentadas.

#### 5.4.13.2. P-valores numéricos, ajustados y sin ajustar

Las tablas y CSV conservan p-valores numéricos ajustados y sin ajustar, logaritmos y límites de precisión. La notación científica evita reducirlos a “p<0,05”. Ningún p-valor se muestra como cero por conveniencia. Las referencias t, χ² de Friedman/Ljung–Box, normal de BDS y t aproximada de DM mantienen sus condiciones; “numérico” no significa una prueba matemáticamente exacta.

#### 5.4.13.3. Intervalos de confianza y tamaños de efecto

Se publican IC t marginales entre semillas, d_z pareados e IC percentiles de fechas por bloques. Cada tabla identifica método, unidad y signo. Los primeros condicionan el calendario y los segundos los modelos almacenados; no se suman automáticamente sus incertidumbres. No se calculó un IC de equivalencia ni se confunden los IC marginales con una banda simultánea de la familia.

#### 5.4.13.4. Verificación de supuestos

Se publican Shapiro–Wilk de las diferencias, continuidad y cobertura, ACF, Ljung–Box y BDS. Shapiro señala desviaciones en MLP–LSTM y MLP–persistencia; la dependencia residual es patente en la corrida designada. Estos resultados limitan las referencias inferenciales y no autorizan a declarar “supuestos verificados” de forma global. No se seleccionó la prueba, semilla o rezago por obtener el menor p-valor.

#### 5.4.13.5. Múltiples ejecuciones documentadas

**V1: resultados y alcance originales.**

Se ejecutaron **treinta semillas finales por red, 1001–1030**, además de cinco semillas piloto distintas y dos de búsqueda. Cada corrida inicializa desde cero; el reajuste posterior a la parada interna usa una semilla derivada independiente y registrada. Las noventa corridas finales tienen los mismos orígenes y no se descartaron según su error.

Treinta supera el mínimo general de cinco y el de seis para rangos, pero queda por debajo de lo solicitado por el piloto: el límite de recursos no demuestra potencia suficiente. Las fechas y los dos ajustes internos de cada corrida no se contabilizan como nuevas semillas del benchmark.

**V2:** diez trayectorias completas por combinación de variante y red, con inicializaciones derivadas por mes. Las 720 combinaciones trayectoria–mes no se cuentan como 720 semillas o mercados independientes; los 1.440 ajustes incluyen parada interna y refit.

#### 5.4.13.6. Métrica principal y comparaciones declaradas con antelación

RMSE, h=1, búsqueda común, selección de MLP antes del piloto y las familias de nueve, seis y quince contrastes se conservan en los protocolos enlazados. D e hiperparámetros se seleccionan con enero–junio; julio–diciembre cuantifica el resultado posterior y el piloto sin reabrirlos. La exposición previa de 2024 al EDA se declara, por lo que no se reivindica confirmación independiente. El test 2025–2026 sigue cerrado.

#### 5.4.13.7. Registro y trazabilidad de cada contraste

| Registro | Contenido verificable |
|---|---|
| [paired_tests.csv](../_static/dl_benchmark_20261104/analysis/paired_tests.csv) | Nueve contrastes: modelos, estimando, hipótesis, pares, diferencia, DE, t, grados de libertad, IC, d_z, Shapiro y p-valores |
| [temporal_dm.csv](../_static/dl_benchmark_20261104/analysis/temporal_dm.csv) | Seis pares principales y sensibilidades, fechas, h, HAC, kernel, HLN, pérdida media y p-valores |
| [diagnostics.csv](../_static/dl_benchmark_20261104/analysis/diagnostics.csv) | Semilla, rezagos, model_df, dimensión, ε, vectores útiles y familia Holm de quince |
| [block_bootstrap.csv](../_static/dl_benchmark_20261104/analysis/block_bootstrap.csv) | Longitud, réplicas, semilla de remuestreo, diferencias/razones e IC marginales |
| [summary.json](../_static/dl_benchmark_20261104/analysis/summary.json) | Versiones, hashes, cohorte, Friedman, fuentes, alcance y limitaciones |
| [pilot_summary.json](../_static/dl_benchmark_20261104/pilot_summary.json) | Dispersión de planificación, δ, R requerido, límite y MDE |

Cada resultado identifica fase y unidad de inferencia; no se omiten pares desfavorables ni se separan familias después de observar p-valores. Los contrastes de DeLong, McNemar, segmentación o lenguaje no aplican a la volatilidad continua de esta serie. TOST, ROPE, Wilcoxon, Nemenyi y ANOVA se distinguen explícitamente como no ejecutados.

| Registro V2 | Contenido verificable |
|---|---|
| [comparisons.csv](../_static/dl_r2_cv_20261104/analysis/comparisons.csv) | Diez pares, Δ RMSE, t(9), IC, d_z, Shapiro, p y Holm-3; IC secundario descriptivo de Δ R² |
| [fold_metrics.csv](../_static/dl_r2_cv_20261104/analysis/fold_metrics.csv) | 720 métricas mensuales con fechas/cantidad por fold |
| [bias_decomposition.csv](../_static/dl_r2_cv_20261104/analysis/bias_decomposition.csv) | Descomposición por trayectoria anual; diagnóstico descriptivo, no una prueba temporal |
| [bias_decomposition_common_h2.csv](../_static/dl_r2_cv_20261104/analysis/bias_decomposition_common_h2.csv) | Mismo diagnóstico en las 184 fechas comunes, manteniendo V1 separado |
| [summary.json](../_static/dl_r2_cv_20261104/analysis/summary.json) | Fuentes, hashes, recuentos y constancia de no evaluar test ni reseleccionar referencia |

### 5.4.14. Interpretación



#### 5.4.14.1. Diferencias estadísticamente significativas

**V1: resultados y alcance originales.**

En el análisis entre semillas con Holm, LSTM obtiene menor RMSE que MLP y CNN1D; MLP–CNN1D es inconcluso. Las tres redes tienen mayor error que persistencia. DM-HLN sobre pérdidas cuadráticas medias por fecha distingue los seis pares, con el orden de pérdidas favorable a persistencia, LSTM, MLP y CNN1D. Las afirmaciones mantienen sus supuestos, estimandos y alcance de desarrollo.

**V2:** el único resultado inferencial principal nuevo es la menor RMSE de `relative_window` frente a `absolute_log` dentro de cada red, con Holm-3 y alcance entre diez trayectorias sobre calendario fijo. No se afirma significación temporal, entre arquitecturas, contra V1 o frente a persistencia para V2.

#### 5.4.14.2. Significancia estadística y relevancia práctica

**V1: resultados y alcance originales.**

La ventaja de LSTM dentro de las redes no demuestra utilidad frente a persistencia: su RMSE media es un 84,624 % mayor. Ninguna red cumple el 5 % de reducción fijado y los contrastes de superación del margen tienen p=1,000. MLP–CNN1D no se declara equivalente por no rechazar entre semillas. Tampoco se fija un margen práctico entre redes a posteriori.

**V2:** los tres p-valores ajustados pequeños comparan procedimientos CV, no el margen del 5 % contra persistencia. La mejora puntual de CNN1D en julio–diciembre es menor a ese margen. No se concluye equivalencia de las demás redes ni se interpreta un R² anual positivo como cumplimiento de una hipótesis aplicada.

#### 5.4.14.3. Conclusiones respaldadas por diferencias, incertidumbre y efectos

Las conclusiones se apoyan en treinta repeticiones, dispersión, diferencias pareadas, IC, d_z, Holm, análisis temporal y limitaciones del piloto. Las pruebas reutilizan predicciones y no son confirmaciones independientes. No se atribuyen ventajas causales a recurrencia/convolución ni generalización al test no evaluado. El límite de treinta semillas impide afirmar la potencia objetivo, aunque se observen diferencias grandes.

#### 5.4.14.4. Estado del análisis estadístico

| Producto | Estado |
|---|---|
| Predicciones y residuos | Disponibles para 30 semillas por red y 184 fechas de desarrollo |
| Familias y parámetros | Registrados: Holm 9 entre semillas, Holm 6 DM principal y Holm 15 diagnósticos |
| Normalidad y dependencia | Shapiro calculado; ACF y diagnósticos publicados; límites explícitos, sin Q–Q publicado |
| Comparación entre semillas | t pareada y Friedman ejecutados; Wilcoxon/ANOVA/Nemenyi no ejecutados |
| DM, BDS y Ljung–Box | Ejecutados con parámetros y referencias aproximadas documentados |
| IC y efectos | t marginal, d_z y bootstrap temporal pareado publicados |
| Potencia/MDE | Piloto ejecutado; máximo R=1.212 frente al límite 30; potencia objetivo no acreditada |
| Equivalencia | TOST y ROPE no ejecutados; no se afirma equivalencia |
| Confirmación externa | Pendiente; test 2025–2026 no leído ni evaluado por este experimento |

El análisis cuantifica un experimento real de desarrollo; no transforma su exposición previa o los límites de diseño en evidencia confirmatoria.

**V2 se añade sin sustituir lo anterior:** tres t pareadas y Holm-3 ejecutados; IC marginales de métricas y diferencias de R² publicados; descomposición de sesgo y métricas mensuales disponibles. No se ejecutaron nuevos DM, BDS, Ljung–Box, Friedman, equivalencia ni bootstrap temporal. Diez semillas no proceden de un estudio de potencia y no acreditan 0,80.

## 5.5. Discusión Crítica

La discusión se apoya en mediciones propias, sus unidades de incertidumbre y literatura verificable. Las comparaciones descriptivas se identifican como tales; las afirmaciones inferenciales mantienen supuestos y límites. Una explicación arquitectónica plausible no sustituye evidencia causal ni se presenta como resultado probado.

### 5.5.1. Desempeño y explicación de las diferencias

**Lectura del ciclo original V1.**

LSTM tiene la menor RMSE posterior entre las redes y sus diferencias frente a MLP y CNN1D son distinguibles bajo el análisis t condicionado a semillas, con Holm e IC publicados. La ventaja no se conserva frente a persistencia, que obtiene menor RMSE, MAE y mejor R². MLP permanece como referencia operacional porque fue seleccionado con la fase de búsqueda, no por ser el menor error posterior.

![Predicciones reales en julio–diciembre de 2024, semilla predefinida 1001](../_static/dl_benchmark_20261104/analysis/figures/predictions_designated_seed.png)

En la corrida designada, LSTM produce una trayectoria muy suave que omite picos; MLP y CNN1D también suavizan y muestran niveles persistentemente elevados en partes del tramo. Persistencia sigue la etiqueta con un día de rezago. La figura ilustra errores sistemáticos reales de esa semilla, sin resumir la variación de las treinta ni demostrar utilidad económica. La historia adicional disponible para persistencia se declara en §5.1.

![Curvas reales de entrenamiento y validación interna, semilla 1001](../_static/dl_benchmark_20261104/analysis/figures/learning_curves_designated_seed.png)

Las curvas corresponden a la **parada interna**, no a un ajuste contra julio–diciembre ni contra test. Se seleccionan las épocas 22 para MLP, 3 para LSTM y 8 para CNN1D; se registran 34, 15 y 20 épocas, respectivamente. En LSTM el error interno de entrenamiento sigue disminuyendo mientras la validación empeora después de su mínimo, lo que justifica conservar la regla de parada. Esto no identifica por sí solo por qué LSTM obtiene menor error en la calibración posterior.

Recurrencia, convolución causal y representación plana incorporan sesgos distintos, pero atribuirles causalmente estas diferencias requeriría ablaciones o comparaciones adicionales controladas. No se afirma que una familia sea superior por su nombre ni por una única semilla.

**Diagnóstico V1 y resultado V2 del problema de R².** Para la misma cohorte,

$$
R^2=1-\frac{\operatorname{MSE}}{\operatorname{Var}_{0}(Y)},\qquad
\operatorname{MSE}=\left(\overline{\widehat Y-Y}\right)^2+
\operatorname{Var}_{0}(\widehat Y-Y).
$$

Un R² negativo significa que el error cuadrático supera la varianza poblacional de las etiquetas de ese periodo; no es por sí solo un fallo de cálculo. El sesgo de nivel siguiente se define como **predicción−objetivo**, signo opuesto al residuo Y−predicción usado en los diagnósticos V1:

| V1, julio–diciembre | Sesgo medio de nivel | Media de sesgo²/MSE por semilla (%) | Media de DE(predicción)/DE(objetivo) |
|---|---:|---:|---:|
| MLP | 0,0120251 | 55,285 | 0,9112 |
| LSTM | 0,0026207 | 12,531 | 0,0948 |
| CNN1D | 0,0131235 | 56,819 | 0,9073 |

El diagnóstico cuantifica la desviación de nivel en MLP/CNN1D y la escasa variación de los pronósticos LSTM originales. Son identidades descriptivas y razones promediadas por semilla, no una atribución causal única. En V2 relativo, el sesgo anual medio es −0,0004849, 0,0004575 y 0,0001687 para MLP, LSTM y CNN1D; sus razones de DE son 0,8605, 0,8240 y 0,8234.

![Sesgo anual de ambas variantes CV, media y DE entre diez semillas](../_static/dl_r2_cv_20261104/analysis/figures/bias_cv_full2024.png)

![Predicciones CV reales de la semilla 4101; V1 semilla 1001 solo como contexto distinto](../_static/dl_r2_cv_20261104/analysis/figures/predictions_designated_seed.png)

En la semilla designada 4101, la variante relativa sigue más cambios del objetivo que la absoluta, pero conserva desviaciones y picos sobrestimados. El resultado visual acompaña las métricas de diez semillas; no prueba que se hayan eliminado la dependencia residual o los errores estructurales, porque DM/BDS/Ljung–Box no se repitieron para V2.

La comparación controlada favorece el procedimiento relativo en la CV, pero combina representación y ponderación de escala. Frente a V1 cambia además el pasado admitido y el reajuste mensual; las cifras no permiten atribuir toda la mejora exclusivamente a CV o exclusivamente a normalización.

### 5.5.2. Generalización y estabilidad

**Lectura del ciclo original V1.**

LSTM presenta la menor DE entre semillas de la RMSE en este calendario: 0,0004854, frente a 0,0028545 de MLP y 0,0051921 de CNN1D. Esa estabilidad es del entrenamiento condicionado a las mismas fechas, no una prueba de generalización temporal.

![RMSE media y desviación estándar entre las treinta semillas; persistencia determinista](../_static/dl_benchmark_20261104/analysis/figures/rmse_seed_variability.png)

Las barras muestran **DE, no IC del 95 %**. Persistencia carece de variación de inicialización. El bootstrap de fechas mantiene el orden descriptivo en las longitudes examinadas, pero condiciona los modelos ya ajustados y no representa nuevos regímenes de mercado.

No se determina qué red generaliza mejor a 2025 o enero–agosto de 2026: ese test no se ha utilizado para modelado ni evaluado por este ciclo. La diferencia de orden entre enero–junio y julio–diciembre puede incluir selección, deriva, historia y variación de ajuste; no identifica por sí sola sobreajuste de una arquitectura. Un solo activo y proveedor, la máscara de elegibilidad y el día incompleto excluido del desarrollo limitan además la extrapolación a otros datos.

**Estabilidad V2 por meses.** El R² anual positivo no implica que todos los meses sean positivos. El resumen siguiente toma primero la media de R² entre diez semillas en cada mes y conserva su rango; **no es el cálculo del R² anual**.

| Variante | Red | Mínimo R² mensual medio | Máximo R² mensual medio | Meses con media R² negativa / 12 |
|---|---|---:|---:|---:|
| `absolute_log` | MLP | -8,0063 | 0,1017 | 11 / 12 |
| `absolute_log` | LSTM | -6,0273 | 0,1368 | 8 / 12 |
| `absolute_log` | CNN1D | -3,4703 | -0,1019 | 12 / 12 |
| `relative_window` | MLP | -1,7225 | 0,6876 | 2 / 12 |
| `relative_window` | LSTM | -2,1486 | 0,6832 | 3 / 12 |
| `relative_window` | CNN1D | -1,6783 | 0,7665 | 4 / 12 |

Fuente: [fold_metrics.csv](../_static/dl_r2_cv_20261104/analysis/fold_metrics.csv), con 29, 30 o 31 fechas según el mes. El procedimiento relativo reduce los fallos mensuales, pero conserva meses de bajo desempeño y no demuestra estabilidad futura. Las doce cohortes se solapan en su entrenamiento y los días mantienen dependencia; ni folds ni semillas crean mercados nuevos.

### 5.5.3. Significación estadística y relevancia práctica

**Lectura del ciclo original V1.**

La diferencia MLP−LSTM es 0,00708685, IC entre semillas [0,00598691; 0,00818678], d_z=2,40585 y p Holm=5,41433×10⁻¹³; la calibración t se limita por Shapiro p=0,020789. MLP−CNN1D tiene IC que cruza cero y p Holm=0,549057, aunque su pérdida temporal media favorece MLP. Se conservan ambas perspectivas con su estimando explícito, sin escoger la favorable.

Los aumentos de RMSE frente a persistencia son 231,579 %, 84,624 % y 264,682 % para MLP, LSTM y CNN1D. Por tanto, la ventaja de LSTM entre redes no alcanza el criterio aplicado de mejora del 5 % frente a persistencia. No hay evidencia de equivalencia ni potencia de 0,80 acreditada para detectar la magnitud planificada con el límite de treinta semillas.

**V2:** los tres contrastes emparejados relativa–absoluta tienen p Holm entre 3,05700×10⁻¹⁰ y 3,87157×10⁻⁸. Son evidencia del procedimiento en ese calendario, no de mejora del 5 % frente a persistencia. Persistencia gana en todo 2024; CNN1D relativa mejora puntualmente 3,095 % en el semestre común, sin alcanzar el margen ni disponer de un contraste temporal nuevo. Los R² mensuales negativos restantes impiden presentar la corrección como éxito uniforme.

### 5.5.4. Desempeño frente a costo y complejidad

**Lectura del ciclo original V1.**

LSTM combina menor error entre redes y menor tiempo medio de entrenamiento medido, pero usa más parámetros y mayor tiempo de inferencia. MLP tiene menor costo de inferencia y menos pesos; CNN1D presenta mayor error medio y costo que MLP en las condiciones medidas. La ausencia de significación entre semillas para MLP–CNN1D impide convertir ese orden descriptivo en una afirmación de superioridad poblacional de RMSE.

Las épocas seleccionadas y la validación interna forman parte del costo: no se comparó una época idéntica de cada arquitectura. Las mediciones locales de fracciones de segundo no establecen un requisito de producción. La mejor referencia de error sigue siendo persistencia, sin pesos, por lo que más complejidad DL no se tradujo aquí en una mejora predictiva frente a ella.

**V2:** la representación relativa mejora RMSE anual sin aumentar el número de pesos, pero tarda más en entrenamiento bajo el mismo entorno concurrente. MLP relativo es la red anual de menor error y la de menos parámetros; CNN1D relativa lidera solo el semestre común. No se mezclan esos rankings ni se compara directamente el costo de un ajuste V1 con doce reajustes V2.

### 5.5.5. Baseline y umbral para un modelo avanzado

**Lectura del ciclo original V1.**

La referencia DL operacional es **MLP, D=7, H=16, LR=0,001, n=7 y h=1**, seleccionada en enero–junio. Su resultado de desarrollo posterior es RMSE 0,0159903 ± 0,0028545, IC t marginal [0,0149244; 0,0170562]. LSTM es el líder descriptivo posterior dentro de las redes; persistencia obtiene RMSE 0,0048225. No se reemplaza MLP ni se afirma un ganador de test.

Una propuesta avanzada necesita un margen aplicado propio frente a esa referencia, variabilidad pareada y evaluación comparable. Hoy no existe una MDE válida para una propuesta todavía no especificada. El 5 % registrado frente a persistencia corresponde a otra comparación, no se cumple y no debe transferirse a MLP. El piloto indica además que el límite de treinta semillas no cubre la magnitud de planificación con potencia de 0,80.

Si los resultados de una evaluación orientan el diseño de la propuesta avanzada, esa evaluación deja de ser confirmación independiente para ella. Se deberá aislar el procedimiento completo y evaluar también el baseline congelado en las mismas fechas. Esto no reasigna retrospectivamente un periodo nuevo ni autoriza abrir ahora el test reservado.

**Cierre del nuevo ciclo:** V2 acredita una mejora del procedimiento relativo frente al absoluto dentro de desarrollo y mantiene MLP como identidad operacional heredada, sin afirmar que su pipeline original y el actualizado sean el mismo estimador. Antes del ajuste final y del test debe cerrarse qué representación y regla de actualización se evaluarán. El ajuste final sobre 2023–2024 no se ejecutó en esta CV; 2025–agosto de 2026 sigue cerrado y septiembre–octubre continúa sin asignación. No se fijó un margen avanzado nuevo ni se acreditó potencia de 0,80.

### 5.5.6. Brecha con la literatura y margen de mejora potencial

La brecha entre modelos avanzados publicados y las referencias básicas se examinará con los estudios incluidos en la revisión. Se comprobarán definición de volatilidad, escala, frecuencia, horizonte, activo, proveedor, periodos, particiones, selección y número de repeticiones.

Solo bajo comparabilidad suficiente se cuantificará una diferencia directa. Si los estudios usan otros datos o protocolos, se presentarán como contexto y se explicará por qué no permiten calcular un margen numérico trasladable al experimento propio. No se restará una RMSE publicada de la RMSE local sin verificar primero unidades y condiciones.

Se discutirá la credibilidad de la brecha considerando intervalos, variabilidad, sesgos de publicación y calidad metodológica. Una mejora publicada no constituye una garantía de que otra arquitectura la reproduzca en BTCUSDT bajo este calendario.

En §3.4.3 ya se cuantificaron las reducciones puntuales de RMSE de D-TCN frente a TCN en S1: **User 8,53 %, Tweet 5,04 %, VADER −4,65 % y Count −26,74 %**. Proceden de medias redondeadas; cambian arquitectura e información social, con cuarenta reinicios frente a veinte del comparador. Son contexto bibliográfico, no un margen numérico transferible a OHLCV diario, n7/h1 o al baseline propio medido aquí con otro protocolo.

Los resultados actuales tampoco permiten trasladar esas brechas: ninguna red mejoró persistencia en V1 y ninguna la mejora en la evaluación de todo 2024 bajo V2; la ventaja puntual de CNN1D relativa en el semestre común es inferior al 5 % y no está confirmada temporalmente. La literatura puede motivar hipótesis, pero no proporciona una reducción futura garantizada ni resuelve la ausencia de respaldo aplicado de MLP/CNN1D simple en el corpus revisado.

### 5.5.7. Amenazas a la validez

| Dimensión | Riesgo | Control o evidencia ejecutada | Límite restante |
|---|---|---|---|
| Interna | Fuga temporal por ventanas, escalado o parada | Ventanas causales, embargo D, escaladores solo de train y parada interna pasada | Verificaciones de implementación no eliminan errores de especificación económica |
| Interna | Optimización desigual o selección de semillas | Mismo presupuesto: cuatro configuraciones por familia/D y dos semillas; treinta semillas finales completas | Los espacios mínimos no agotan el potencial de cada familia |
| Interna | Selección posterior | MLP y D=7 fijados con búsqueda; no sustituidos por LSTM posterior | Desarrollo de 2024 ya conocido por EDA y calibración reutilizada para piloto |
| Interna | Cobertura diferente | Mismas 184 fechas por red y semilla; sin fallas ni exclusiones por desempeño | Cohorte de seis meses y máscaras de elegibilidad restringen representatividad |
| Externa | Un activo, proveedor y régimen | BTCUSDT de Binance Spot y fechas declarados | Test 2025–2026 cerrado; sin otros activos o exchanges |
| De construcción | Proxy σ7 e información de referencia | Fórmula, escala y h explícitos; persistencia causal | Solapamiento de etiquetas; persistencia requiere ocho cierres frente a D=7 de OHLCV |
| De construcción | Métricas y costos | RMSE/MAE/R²/MAPE/MASE, cobertura completa y tiempos locales definidos | No se mide utilidad financiera, riesgo operacional ni ruido irreducible |
| De conclusión estadística | Dependencia, normalidad y potencia | Holm, IC/efectos, HAC-HLN, bloques, Shapiro y diagnósticos | Referencias aproximadas, 184 fechas y límite 30 frente a R requerido máximo 1.212 |

Los controles y archivos acreditan lo que se ejecutó, no una garantía absoluta de validez. Varias pruebas sobre las mismas predicciones no constituyen réplicas externas independientes.

**Amenazas específicas V2:** la corrección se diseñó después de observar el R² V1 y sobre 2024 previamente expuesto; sus folds causales evitan usar futuro en cada ajuste, pero no eliminan la exposición del proceso de diseño. No hay selección anidada nueva, los hiperparámetros se heredan de V1, las cohortes mensuales crecen y los diez recorridos comparten fechas. Los cambios de representación y ponderación de escalas van juntos. El semestre común permite contexto con etiquetas iguales, pero su comparación V1–V2 cambia semillas y política de ajuste, por lo que permanece descriptiva.

### 5.5.8. Limitaciones del estudio

El experimento original V1 es real pero acotado: un activo y proveedor, objetivo móvil solapado, cuatro historias candidatas, espacios pequeños de búsqueda y solo 184 fechas posteriores de desarrollo. El piloto usa esa misma calibración y el límite de treinta semillas queda por debajo de lo que exige su aproximación de potencia. Los modelos V1 son ajustes estáticos con entrenamiento admitido de 2023; los resultados no constituyen evaluación de test; su política de actualización sigue pendiente.

Los intervalos entre semillas no capturan mercados nuevos y el bootstrap temporal no reentrena. Los diagnósticos se limitan a la semilla 1001 y emplean referencias aproximadas. Las figuras revelan suavizado y desajustes de nivel en esa corrida, pero no permiten atribuir causas sin análisis adicional. Persistencia dispone de un cierre histórico adicional respecto de la ventana OHLCV de las redes.

La ejecución no completa el respaldo de §3: el corpus aplicado mantiene **0/5 estudios para MLP y 0/5 para CNN1D simple**; LSTM sí tiene apoyo documental. La CNN propia no es la TCN publicada. Los resultados no sustituyen las referencias ausentes ni justifican exagerar la calidad de fuentes suplementarias. La persistencia gana en V1 y en el año completo V2; no se ha demostrado mejora financiera o generalización en el test reservado.

V2 añade evidencia de CV mensual real sobre todo 2024, pero sigue siendo una modificación motivada por resultados ya observados. No es validación anidada del proceso de selección, no repite la búsqueda de hiperparámetros y no confirma generalización. El contraste entre sus dos variantes cambia representación y ponderación del escalado; el contraste descriptivo con V1 cambia además reajuste, cohortes y semillas. Ninguno aísla una causa única del R².

La CV V2 completó doce reajustes mensuales reales por trayectoria y conserva meses con R² medio negativo incluso con la representación relativa. No repitió las pruebas temporales originales y sus diez semillas no resuelven el déficit de potencia. Tampoco convierte la pequeña ventaja puntual de CNN1D relativa en julio–diciembre en cumplimiento del margen o evidencia de superioridad futura.

### 5.5.9. Recomendaciones futuras



#### 5.5.9.1. Recomendaciones derivadas de las limitaciones

Cerrar explícitamente el procedimiento posterior a V2 —representación, política de ajuste, configuración y márgenes— antes de cualquier ajuste final o apertura del test. Conservar V1 como resultado original y V2 como corrección exploratoria con fechas, semillas y artefactos distintos. Completar además el respaldo aplicado de MLP/CNN1D simple sin sustituirlo por resultados propios.

Estudiar los meses de R² negativo y los errores restantes con análisis registrados, información comparable de persistencia y diagnósticos temporales adecuados a las nuevas predicciones; no trasladar los p-valores residuales V1. Para una propuesta avanzada, registrar su margen frente al baseline, estimando, emparejamiento y presupuesto con planificación defendible. Aumentar semillas cuando sea viable no reemplaza periodos externos independientes.

El ajuste final de desarrollo y el test permanecen pendientes de cierre del procedimiento; septiembre–octubre de 2026 sigue sin asignación. Noticias, otros mercados y arquitecturas avanzadas requieren nueva justificación y no se presentan como mejoras garantizadas.

#### 5.5.9.2. Matriz de conclusiones y evidencia pendiente

| Pregunta | Evidencia actual | Conclusión y límite |
|---|---|---|
| ¿Qué red funciona mejor? | V1 y dos variantes CV, con periodos identificados | LSTM lideró V1; MLP relativo lidera el año CV y CNN1D relativo el semestre común. No hay un ganador universal ni explicación causal única |
| ¿Se corrigió el R² negativo? | R² anual CV y comparación sobre 184 fechas comunes | Relativo alcanza medias positivas en las tres redes; absolutos y meses concretos conservan fallos. V1 no se borra |
| ¿Cuál generaliza mejor? | CV causal en desarrollo ya expuesto | No determinado en test u otros mercados; las doce ventanas no convierten la selección previa en confirmación |
| ¿Qué diferencia está contrastada? | V1 con sus familias y V2 relativa–absoluta con Holm-3 | Mejora del procedimiento relativo dentro de cada red; no hay p-valores V2 contra V1, entre redes o contra persistencia |
| ¿Se supera persistencia con el margen práctico? | Año CV y semestre común | Persistencia gana en el año; CNN1D relativa mejora puntualmente alrededor del 3,1 % en el semestre, por debajo del 5 % y sin confirmación temporal |
| ¿Compensa la complejidad? | Parámetros iguales y costos por ciclo | V2 relativo mejora error sin más pesos, con más tiempo de entrenamiento observado; no se comparan costos seriales V1 y concurrentes V2 directamente |
| ¿Cuál es el baseline y umbral avanzado? | MLP elegido en V1 y nuevo protocolo exploratorio | Identidad MLP conservada; cierre del pipeline actualizado y margen avanzado pendientes, sin nueva potencia de 0,80 |
| ¿Qué margen sugiere la literatura? | Brechas de §3.4.3 y diferencias de protocolo | Contexto bibliográfico no transferible numéricamente; falta respaldo aplicado de MLP/CNN1D simple |
| ¿Qué falta ejecutar? | Protocolos, predicciones y cierre de acceso | Ajuste final y test 2025–agosto de 2026 todavía no ejecutados; septiembre–octubre sin asignación |

Se conservan los resultados desfavorables originales y se incorpora una corrección CV verificable. El nuevo R² positivo se interpreta con su calendario, procedimiento y límites; no sustituye la evaluación reservada.
