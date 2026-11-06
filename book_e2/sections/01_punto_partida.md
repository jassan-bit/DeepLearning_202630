# 1. Definición del Problema, Preguntas de Investigación y Dataset

Esta sección desarrolla las páginas 2–3 de la guía completa de 65 páginas. La formulación y la ficha de datos se conservan; la ruta se cierra como **F2a, series de tiempo**, en la {ref}`sección 7 <e2-ruta-temporal>`. Las preguntas y las reglas del procedimiento E2 están registradas en el [commit previo al piloto `599a29f`](https://github.com/jassan-bit/DeepLearning_202630/tree/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0), con alcance **exploratorio** porque los periodos heredados ya se conocían. El benchmark final sigue pendiente; el [piloto P0 de desarrollo y su protocolo](09_protocolo_registrado.md) se ejecutan por separado y no constituyen sus resultados. Los diagnósticos propios del EDA se presentan en la [sección 2](02_eda.md).

## 1.1 Definición formal del problema

(e2-contrato-heredado)=
### 1.1.1 Contrato heredado y versión de referencia

Se adopta explícitamente la **fase II D7** del [Entregable 1, commit `94a4b03`](https://github.com/jassan-bit/DeepLearning_202630/tree/94a4b0324e93361e8928d708fdfeb50110e3686f). El [contrato descargable](../_static/e1_reference.json) identifica sus artefactos. No se mezclan sus recetas con las redes de la fase I.

| Elemento heredado | Definición conservada en E2 |
|---|---|
| Problema | Volatilidad histórica de Bitcoin, BTCUSDT en Binance Spot; formulación heredada inicialmente denominada regresión temporal, con ruta E2 resuelta como F2a, series de tiempo |
| Entrada | D=7 días de cinco canales OHLCV; horizonte h=1 día |
| Objetivo | σ₇ poblacional de siete retornos logarítmicos diarios de Close, sin anualización |
| Modelos base | GRU/D7-c03, LSTM/D7-c02 y TCN/D7-c02, con sus configuraciones de fase II |
| Referencia seleccionada en desarrollo | GRU-c03; se conserva aunque otro modelo obtenga menor error en test |
| Métrica primaria | Media entre semillas de la RMSE de cada trayectoria sobre las fechas elegibles comunes, en escala original |
| Margen absoluto | δ = **0,00025772842664981106** unidades de volatilidad diaria |
| Unidad de independencia | Una serie BTCUSDT de un proveedor; las fechas y etiquetas solapadas tienen dependencia temporal |
| Partición | Calendario de fase II descrito abajo; no se reasignan sus periodos |

| Modelo heredado | Receta seleccionada en E1 | Parámetros |
|---|---|---:|
| GRU-c03 | Una capa recurrente, 16 unidades, learning rate 0,003, dropout 0,1 | 1.121 |
| LSTM-c02 | Una capa recurrente, 16 unidades, learning rate 0,003, dropout 0 | 1.489 |
| TCN-c02 | Ancho 8; tres bloques residuales, dos Conv1D por bloque, kernel 3, dilataciones 1/2/4; learning rate 0,003, dropout 0 | 1.185 |

Las recetas completas, escalado, optimizador y entrenamiento están en el [diseño versionado de fase II](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/experiments/gru_lstm_tcn_planned_20261105/design.json). La TCN tiene campo receptivo nominal 29; la ventana efectiva de este contrato sigue siendo D=7.

| Función | Periodo UTC y regla heredada |
|---|---|
| Historia inicial | 01/01–31/12/2023 |
| Optimización | Enero–octubre de 2024: diez bloques mensuales forward chaining comunes, con índices de `timeseries-cv` de DidierRLopes y Filipe R. Ramos, embargo D=7 y validación interna de 90 días |
| Piloto de fase II | 01/11–30/12/2024: 60 fechas de desarrollo |
| Ajuste final | Desarrollo permitido de 2023–2024; origen de cierre 01/01/2025 y etiquetas de entrenamiento hasta el 24/12/2024 por embargo |
| Reserva de fase II | 01/09–30/10/2026: 60 fechas, **ya evaluadas en E1** |
| Archivo de fase I | 01/01/2025–31/08/2026: 608 fechas previamente evaluadas; conserva su condición de antecedente conocido |

El embargo excluye resultados de entrenamiento demasiado próximos al origen según la regla D=7; no elimina siete días aleatorios ni cambia los periodos anteriores. Los metadatos conservan los detalles y las exclusiones. No se asigna una semilla a la partición cronológica determinista.

### 1.1.2 Entradas, salida y disponibilidad causal

Sea $P_t>0$ el cierre del día UTC $t$. Se define

$$
r_t=\log(P_t/P_{t-1}),\qquad
\bar r_t=\frac{1}{7}\sum_{j=0}^{6}r_{t-j},\qquad
Y_t=\sigma_7(t)=\sqrt{\frac{1}{7}\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

Se necesitan **ocho cierres consecutivos para siete retornos**. El denominador es 7 (`ddof=0`). No se anualiza ni se sustituye esta variable por volatilidad intradía o por el precio.

Sea $Z_t=(O_t,H_t,L_t,C_t,V_t)$ el OHLCV agregado del día. El pronóstico se emite al comienzo del día $t$, con

$$
X_t=(Z_{t-7},\ldots,Z_{t-1})\in\mathcal X\subseteq\mathbb R^{7\times5},
\qquad Y_t\in\mathcal Y=\mathbb R_{\ge0}.
$$

Así, $r_t$ y $Y_t$ solo se conocen al terminar el día pronosticado. Esta convención equivale a escribir «entrada hasta $t$, objetivo $\sigma_7(t+1)$» desplazando el índice; no añade información del futuro. La volatilidad pasada puede usarse para diagnósticos por régimen, pero no se añade aquí como sexto canal del modelo.

El preprocesamiento heredado usa logaritmos de OHLC relativos al último cierre conocido de la ventana y `log1p` del volumen; los escaladores se ajustan exclusivamente en entrenamiento permitido. El objetivo se divide por su media de entrenamiento y las métricas se calculan después de invertir esa transformación. Se conserva el calendario y se excluyen ventanas que dependan de días inválidos, sin unir artificialmente fechas separadas por huecos.

### 1.1.3 Espacio de hipótesis, pérdida y riesgo

Para el catálogo de redes $\mathcal M$, con los diez modelos Top10, tres variantes GRU/LSTM/TCN ajustables en E2 y tres anclas heredadas, el espacio de hipótesis es $\mathcal H=\bigcup_{m\in\mathcal M}\mathcal H_m$, con $f_\theta:\mathcal X\to\mathbb R$. Junto con Ridge, persistencia, naive semanal y ARIMA, el panel registrado reúne **veinte configuraciones**: quince familias ajustables, tres anclas y dos referencias naive sin HPO ficticio. Las recetas registradas y sus comprobaciones de desarrollo no equivalen a un benchmark final ejecutado. La variable objetivo es no negativa; el protocolo registra las salidas de cada arquitectura y el enlace puntual de Ridge/ARIMA sin cambiar las recetas heredadas.

Se conserva la pérdida cuadrática:

$$
\mathcal L(f(X),Y)=(Y-f(X))^2,\qquad
R_P(f)=\mathbb E_{(X,Y)\sim P}[\mathcal L(f(X),Y)].
$$

Sobre una cohorte elegible común $\mathcal T$, su versión empírica es

$$
\widehat R_{\mathcal T}(f)=\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}(Y_t-f(X_t))^2,
\qquad
\widehat\mu_m=\frac{1}{R}\sum_{s=1}^{R}\sqrt{\widehat R_{\mathcal T}(f_{m,s})}.
$$

$\widehat\mu_m$ es la métrica primaria. No se reemplaza por la RMSE de un ensamble ni por la raíz de la media de todos los errores mezclando semillas. La dependencia y el cambio de distribución impiden tratar el riesgo empírico histórico como una estimación iid de cualquier mercado futuro. La guía pide prueba sellada; **los periodos heredados ya conocidos no cumplen esa condición para una confirmación nueva de E2**.

### 1.1.4 Tipo de tarea y relevancia

La clasificación metodológica de E2 es **F2a, series de tiempo**, resuelta tras recibir la guía completa. El {ref}`diagnóstico residual de §2.2.3 <e2-diagnostico-forward>` aporta autocorrelación fuerte de la referencia Ridge; RT1 (PDF p32) exige reclasificar, además del diagnóstico del EDA C (p6). La {ref}`sección 7 <e2-ruta-temporal>` registra esa resolución y los baselines específicos. Se conserva la tarea predictiva, D7×5, σ₇, h=1, δ y partición de E1; la reclasificación no añade Y rezagada como canal de las redes. Persistencia, naive semanal y ARIMA declaran su información endógena y sus soportes distintos del D7.

Tres trabajos respaldan la relevancia científica del pronóstico de volatilidad en criptomonedas: Miura, Pichl y Kaizoji estudian redes neuronales para volatilidad realizada {cite:p}`miura2019`; Guo, Bifet y Antulov-Fantulin estudian la información de órdenes de compra y venta {cite:p}`guo2018`; Akbiyik y colaboradores incorporan información de Twitter al pronóstico de volatilidad de Bitcoin {cite:p}`akbiyik2023`.

Estos antecedentes motivan investigar memoria temporal, covariables y generalización. **No se afirma que utilicen nuestro snapshot, idéntico objetivo σ₇ ni exactamente las entradas aquí disponibles.** Tampoco constituyen por sí solos la revisión sistemática de E2 ni prueban uso del dataset exacto en una revista Q1. El interés aplicado es medir capacidad predictiva y coste bajo un benchmark académico; no se concluye rentabilidad, reducción de riesgo financiero o utilidad de trading.

### 1.1.5 Métricas, margen y unidad de independencia

La primaria es RMSE; las secundarias heredadas son MAE, MSE, R², MAPE y MASE. MAPE se define solo donde el objetivo sea positivo y se reportará su cobertura; no se añadirá un épsilon arbitrario. MASE usa el denominador histórico ingenuo de lag 1 sobre pares de días consecutivos elegibles de entrenamiento; no se calcula comprimiendo huecos ni usando test para escalar.

El margen absoluto heredado procede de $0,05\times0,005154568532996221$, RMSE CV de la GRU/D7 de fase I. **No se recalcula como el 5 % de una nueva RMSE.** Se conserva aparte la regla relativa académica de E1 frente a GRU y persistencia; no se confunde una reducción porcentual con δ absoluto. Su origen y fecha están documentados: heredar el margen no lo vuelve una decisión previa a los resultados históricos iniciales.

Hay **una única serie de mercado** ($N_{\text{series}}=1$). Las semillas describen variabilidad del entrenamiento condicionada a los datos; no son nuevas series independientes. Las etiquetas σ₇ comparten retornos y las ventanas comparten observaciones. El protocolo registrado empareja modelos por semilla y fecha y distingue remuestreo de trayectorias y de calendario: bootstrap temporal de bloques de siete días, sensibilidades fijadas de catorce y veintiocho días y 99.999 remuestreos, sin escoger el menor p-valor. El R=465 de E1 no acredita potencia para las nuevas comparaciones. El dimensionamiento P1 deberá evaluar capacidad temporal y entre semillas con las recetas que queden seleccionadas por desarrollo; P0 comprueba viabilidad y no acredita esa potencia.

## 1.2 Preguntas de investigación e hipótesis

Las tres preguntas siguientes son **exploratorias y registradas antes del piloto P0 de E2** en `599a29f`, con hipótesis, márgenes y reglas explícitas. El registro mejora el orden prospectivo del procedimiento nuevo; no borra que el test heredado ya fue inspeccionado ni lo convierte en una confirmación independiente. La [versión registrada de las reglas](https://github.com/jassan-bit/DeepLearning_202630/blob/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0/experiments/e2_remediation_20261106/preregistration.json) distingue ese estado del registro de ejecución posterior.

Para cada uno de los diez candidatos Top10 registrados, sea $g_m=\mu_{\mathrm{GRU}}-\mu_m$: valores positivos indican menor RMSE que la GRU-c03 de referencia heredada. Las diferencias se calcularán en las mismas fechas y con semillas emparejadas; no se seleccionarán comparadores después de mirar el resultado. Las mismas tres preguntas se replican en una familia secundaria frente a GRU ajustada en E2 bajo el HPO común, distinguiendo ambos comparadores.

### 1.2.1 RQ1 · Ganancia predictiva relevante

**Pregunta:** ¿algún candidato reduce la RMSE respecto a GRU-c03 en más que el margen heredado δ?

$$
H_{0,m}:g_m\le\delta,\qquad H_{1,m}:g_m>\delta.
$$

El efecto mínimo es δ en unidades originales de volatilidad. Se registra un contraste unilateral pareado del efecto respecto a δ mediante bootstrap temporal por bloques, con las mismas fechas remuestreadas para ambos modelos y la variabilidad entre trayectorias tratada separadamente. La distribución nula se centra en la frontera $g_m=\delta$; no basta probar una diferencia contra cero. Para planificar potencia, P1 usa una ganancia verdadera de **2δ**, cuyo exceso respecto a esa frontera es δ; no promete potencia del 80 % exactamente sobre la hipótesis nula.

La decisión requerirá un p-valor ajustado inferior al **α=0,02 de su familia** y un límite inferior simultáneo para $g_m$ superior a δ, usando la cola asignada por el registro. Se reportarán efecto absoluto, reducción relativa, intervalos y p-valores ajustados y sin ajustar. Una ganancia puntual superior a δ, por sí sola, no prueba mejora práctica con incertidumbre controlada. Persistencia conserva su papel de referencia adicional: superar GRU no significa superar el pronóstico ingenuo.

### 1.2.2 RQ2 · Deterioro de la ventaja en alta volatilidad

**Pregunta:** ¿la ventaja de un candidato frente a GRU disminuye en más de δ durante días de alta volatilidad conocida al emitir el pronóstico?

Se registra el estrato causal $A_t=\mathbf1\{\sigma_7(t-1)>q_{0,75}^{2023}\}$, con $q_{0,75}^{2023}=0,024247163097617024$, obtenido únicamente de las 350 etiquetas válidas σ₇ de 2023. Su [perfil de entrenamiento está versionado en E1](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/eda_general_quality/target_profiles.csv). El estrato usa información pasada; no es un nuevo canal de entrada ni un umbral optimizado en test.

Con RMSE calculada por trayectoria dentro de cada estrato, sean $g_{m,N}$ y $g_{m,A}$ las ganancias en régimen normal y alto, y $d_m=g_{m,N}-g_{m,A}$:

$$
H_{0,m}:d_m\le\delta,\qquad H_{1,m}:d_m>\delta.
$$

El efecto mínimo es una erosión absoluta δ de la ventaja. Se registra bootstrap temporal pareado sobre el calendario común, conservando las etiquetas de régimen y recalculando las métricas dentro de cada estrato. P1 planifica capacidad bajo una erosión verdadera de 2δ; la frontera contrastada sigue siendo δ. Se reportarán cantidades de fechas y bloques útiles por estrato; si son insuficientes, el resultado será no estimable o inconcluso. Rechazar indica deterioro relevante; **no rechazar no demuestra robustez ni equivalencia**.

### 1.2.3 RQ3 · Mejora conjunta de error y coste

**Pregunta:** ¿algún candidato mejora la RMSE en más de δ y reduce el coste de inferencia frente a GRU-c03?

Sea $c_m$ la diferencia de latencia media por ventana, candidato menos GRU, bajo el mismo hardware, precisión y batch size:

$$
H_{0,m}:(g_m\le\delta)\ \text{o}\ (c_m\ge0),\qquad
H_{1,m}:(g_m>\delta)\ \text{y}\ (c_m<0).
$$

Se registra una prueba de intersección-unión: ambas condiciones deben quedar respaldadas; el p-valor conjunto será el máximo de los dos p-valores unilaterales válidos. El margen de error sigue siendo δ; el umbral de coste es cero, sin inventar una tolerancia financiera. P1 planifica una reducción de latencia del 20 % respecto a cada referencia, sin convertir ese objetivo de capacidad en un margen nuevo de la hipótesis. Se reportará también la magnitud e intervalo del ahorro para valorar su importancia operativa.

La latencia se medirá con entradas de desarrollo o sintéticas de forma $7\times5$, sin volver a puntuar test para medir tiempos. El protocolo fija CPU común, un thread, batch1, 256 llamadas de calentamiento y 61 bloques de 256 llamadas, con orden intercalado reproducible; declara por separado precisión neuronal FP32 y clásica float64. Esos bloques serán las unidades de repetición del coste. Las repeticiones del cronómetro no aumentan el tamaño muestral de la evaluación predictiva.

### 1.2.4 Multiplicidad y criterio de decisión común

El registro fija **K=10 candidatos Top10** y dos familias distintas de **3K=30 hipótesis**: la principal frente a GRU-c03 heredada y la secundaria frente a GRU ajustada en E2. Cada una aplica Holm con **α=0,02** a sus p-valores válidos; para RQ3 se obtiene primero el p-valor conjunto de intersección-unión. Los límites simultáneos utilizan la cola conservadora **α de la familia / 30**, no 0,05/30. Los IC marginales del 95 % se distinguen de esos límites simultáneos.

Una tercera familia contiene **199 diagnósticos registrados**, con Holm y **α=0,01**. La suma 0,02+0,02+0,01 conserva un presupuesto global 0,05 mediante la cota de la unión para esas tres familias, bajo validez de sus p-valores; no cubre automáticamente todos los análisis exploratorios, calibración, incertidumbre o NAS. Los veinte integrantes del panel producirían 190 pares si se compararan todos: ese número no es la familia principal de treinta hipótesis ni la familia de 199 diagnósticos.

La tolerancia de brecha train–validación queda registrada como **δg=δ**, en unidades originales de RMSE; es una decisión operativa prospectiva, no un criterio que certifique retrospectivamente E1. [El protocolo de §9](09_protocolo_registrado.md) define que P1 dimensionará tras HPO con recetas congeladas, alternativas de ganancia/erosión 2δ y capacidad entre semillas y calendario por separado. P0 no selecciona arquitecturas ni acredita esa potencia: completó 220 trayectorias/440 ajustes de desarrollo y se informa como una etapa distinta, todavía sin resultados finales de benchmark. Aunque se controle multiplicidad, el test conocido mantiene el alcance exploratorio; no se cambia la partición heredada para simular una reserva desconocida.

## 1.3 Justificación del dataset

### 1.3.1 Selección, tamaño y calidad

Se conserva **Binance Spot BTCUSDT**, obtenido de [Binance Public Data](https://github.com/binance/binance-public-data) y [Binance Vision](https://data.binance.vision/). Es pertinente por ofrecer una serie continua de un mercado de criptomonedas con OHLCV y trazabilidad de archivos; permite reproducir el benchmark de E1. El tamaño bruto no se interpreta como millones de muestras supervisadas independientes.

El snapshot inventariado contiene **75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas de doce campos**, entre el 01/01/2023 y el 30/10/2026. Los 1.399 días del calendario corresponden a 2.014.560 minutos esperados. Se detectaron **80 minutos ausentes** y una anomalía adicional de `close_time` el 24/03/2023. Por ello, ese día no es elegible para OHLCV diario completo; no se imputa ni se elimina el hueco del calendario.

El desarrollo 2023–2024 tiene 731 días de calendario y 730 días OHLCV completos. En 2023 hay 350 etiquetas σ₇ válidas; en 2024, 366. Las pérdidas adicionales de ventanas dependen de D, el objetivo y el embargo: los conteos de filas brutas, días y ejemplos supervisados son distintos. El ajuste final D7 de E1 utilizó 709 ejemplos elegibles; no se atribuye ese número a futuras configuraciones de E2.

La literatura citada respalda el problema y distintas fuentes informativas. **No está verificado el uso de esta versión exacta del dataset en literatura Q1.** Esa comprobación y la matriz de estudios de E2 siguen pendientes de la revisión sistemática; no se usa la reputación de una editorial como sustituto de evidencia.

### 1.3.2 Procedencia, licencia, ética y privacidad

Los archivos originales y sus checksums se descargan del proveedor; el [manifiesto descargable](../_static/dataset_manifest_e1.json) identifica cada objeto y su integridad. Las [instrucciones y scripts de adquisición de E1](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/data/README.md) permiten reconstruir la fuente sin incorporar precios individuales a este capítulo.

La copia de términos revisada en E1 el 03/11/2026 corresponde a **Vision Dataset Terms v1.0, actualizados el 26/08/2026**: establece CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Véanse el [texto conservado](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/TERMS_AND_CONDITIONS.md) y su [registro de procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/terms_snapshot.json). No se presenta como una licencia MIT de los datos ni se retrofecha su consulta.

El uso es académico y no comercial. Las velas agregadas no incluyen identificadores personales; no se intentará identificar operadores. La ausencia de datos personales no elimina sesgos del mercado ni convierte los resultados en asesoría financiera. Se mantienen la atribución y la separación entre licencia de datos y código.

### 1.3.3 Representatividad, sesgos y partición pública

La población observada es un único par, exchange y periodo. BTCUSDT no representa otras criptomonedas, otros proveedores ni cotizaciones en moneda fiduciaria; USDT añade exposición a una stablecoin. La selección de un activo superviviente, los cambios de régimen, las interrupciones del proveedor y la cobertura histórica limitan la generalización. El volumen refleja negociación de ese mercado, sin medir todo el mercado mundial.

No se ha identificado una **partición oficial de machine learning para este snapshot** en la fuente utilizada. Los archivos mensuales y diarios del proveedor son una organización de descarga, no folds de evaluación. Se conserva la partición temporal de E1 y su forward chaining para respetar disponibilidad causal y comparabilidad. Los periodos conocidos se documentan como tales; no se barajan fechas ni se crean nuevos grupos independientes artificiales.

## 1.4 Descripción técnica · Data Card

Descargar la [Data Card en JSON](../_static/dataset_card.json), el [contrato heredado](../_static/e1_reference.json) y el [manifiesto de archivos](../_static/dataset_manifest_e1.json).

| Campo | Descripción verificable |
|---|---|
| Identificador | Snapshot E1 de Binance Spot BTCUSDT, velas de un minuto, 01/01/2023–30/10/2026 |
| Fuente y recolección | Objetos históricos mensuales/diarios de Binance Vision; adquisición y hashes registrados en el manifiesto heredado, sin afirmar una adquisición nueva de E2 |
| Versión | Commit E1 `94a4b0324e93361e8928d708fdfeb50110e3686f` y manifiesto SHA-256 `af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2` |
| Estructura bruta | 75 ZIP; 2.014.480 filas; doce campos de kline |
| Esquema | Timestamps de apertura/cierre; OHLC, volumen, volumen cotizado, número de operaciones, volúmenes taker base/cotizado y campo ignore |
| Tipos | Timestamps normalizados a UTC, conteo de operaciones entero y magnitudes numéricas; la unidad temporal original se verifica por archivo antes de convertir |
| Entrada del modelo | Siete días consecutivos × cinco variables OHLCV; 35 valores antes de transformaciones |
| Objetivo y horizonte | σ₇ poblacional de retornos diarios; día siguiente al último día de entrada; ocho cierres necesarios |
| Etiquetado | Cálculo determinista desde cierres diarios completos, sin anotadores humanos; acuerdo entre anotadores no aplica |
| Ruido del objetivo | σ₇ es un estimador histórico dependiente del cierre y ventana elegidos; no es observación directa de volatilidad latente ni etiqueta sin error |
| Cobertura | Mercado BTC/USDT en Binance Spot, operación 24/7, calendario diario UTC; no hay unidades geográficas ni población de personas |
| Filtros | Validación de timestamps y velas, agregación de días completos, retornos sobre días consecutivos, exclusión de ventanas afectadas por huecos y embargo heredado; sin imputación |
| Calidad conocida | 80 minutos ausentes y una anomalía adicional de cierre el 24/03/2023; un día diario incompleto |
| Desarrollo | 2023–2024: 731 días de calendario, 730 OHLCV completos; etiquetas σ₇ válidas 350 en 2023 y 366 en 2024 |
| Evaluación heredada | Reserva fase II 01/09–30/10/2026, 60 fechas ya evaluadas; fase I 2025–agosto de 2026 también conocida |
| Licencia | Términos Vision conservados en E1, CC BY-NC-SA 4.0 con condiciones y excepciones indicadas por la fuente |
| Límites de uso | Benchmark académico de una sola serie; sin afirmación de utilidad financiera ni generalización a otras criptomonedas |

El hash anterior identifica **el archivo manifiesto**; no es un único checksum de todos los precios concatenados. Cada ZIP tiene su comprobación individual en ese archivo. La Data Card y el contrato documentan la transferencia; el benchmark final Bitcoin D7 continúa pendiente y su ejecución se distingue del [P0 de desarrollo registrado en §9](09_protocolo_registrado.md). El [recibo de las veinte corridas externas de ETTh1](../_static/paper_guide_evidence/execution_receipt.json) tiene otro alcance.

### 1.4.1 Estado de cierre de esta sección

| Requisito visible | Estado |
|---|---|
| Formulación matemática, tarea, métricas, δ y unidad | Documentados con la herencia de E1 |
| Relevancia con al menos tres referencias | Tres referencias de contexto citadas; no sustituyen la revisión sistemática |
| Tres RQ, hipótesis, efecto y reglas | Registrados antes de P0, con dos familias de treinta hipótesis y alcance exploratorio |
| Justificación y Data Card | Documentadas con procedencia, versión, hashes, calidad y límites |
| Uso del dataset exacto en Q1 | No verificado |
| Catálogo y protocolo | Veinte configuraciones registradas; quince familias ajustables, tres anclas y dos naive; ejecución y validación final pendientes |
| Prueba desconocida y confirmación nueva | No disponibles en los periodos heredados ya evaluados |
