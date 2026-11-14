# 1. Definición del Problema, Preguntas de Investigación y Dataset

**Estado de ejecución al 13/11/2026:** el [benchmark reducido](06_benchmark_final.md) está completado para dieciséis sistemas y 608 fechas, con seis semillas por red. [§48](48_protocolo_reducido.md) documenta las recetas y el protocolo ejecutados; [§49](49_diagnosticos_f2a.md) contiene los diagnósticos posteriores. La arquitectura vigente de TiDE es **TIDE-D7-PUBLISHED-R0-v2**, acreditada antes de sus ajustes; las variantes NAS anteriores mantienen otra identidad.

**Antecedente conservado:** **Vigencia documental (12/11/2026):** existe selección HPO/NAS ejecutada del plan anterior de catorce redes. No acredita las cuatro recetas restauradas de SAE, TimesNet, TimeMixer y PatchTST, ni el benchmark final del catálogo vigente. k, R y el cierre inferencial siguen pendientes. [Catálogo obligatorio (§43)](43_catalogo_obligatorio.md) · [Arquitecturas (§44)](44_arquitecturas_publicadas.md) · [Enmienda registrada (§45)](45_enmienda_protocolo.md) · [Corte histórico de resultados (§41)](41_avance_optimizacion_11_origenes.md).

**Catálogo obligatorio vigente: dieciséis sistemas.** Diez modelos de literatura y adaptaciones, las referencias MLP16, LSTM32 y GRU32 de E1, y Persistence, SeasonalNaive y ARIMA. Las referencias E1 se cuentan aparte del Top 10 y forman parte de los dieciséis. El híbrido CNN-1D + LSTM + atención temporal y Ridge quedan como antecedentes opcionales. La [enmienda del 12/11/2026](45_enmienda_protocolo.md) conserva el antecedente de transición; el cuerpo TiDE publicado, el calendario reducido y la ejecución efectiva están documentados en §48.

Esta sección desarrolla las páginas 2–3 de la guía completa de 65 páginas. Se conservan la formulación y la ficha de datos; la ruta es **F2a, series de tiempo**, como explica la {ref}`sección 7 <e2-ruta-temporal>`. El [registro histórico `599a29f`](https://github.com/jassan-bit/DeepLearning_202630/tree/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0) y el [piloto P0 ejecutado](09_protocolo_registrado.md) se conservan para trazabilidad, sin trasladar sus candidatos, presupuesto o resultados al catálogo vigente. Los periodos heredados ya se conocían: su reutilización tiene alcance exploratorio. Los diagnósticos del EDA se presentan en la [sección 2](02_eda.md).

## 1.1 Definición formal del problema

(e2-contrato-heredado)=
### 1.1.1 Contrato vigente y procedencia histórica

**La especificación vigente se sincroniza con el E1 principal de 608 fechas, no con la reserva complementaria de 60 días.** Véase [§17](17_contrato_sincronizado.md): representación relativa, escalado de entrenamiento, referencias MLP/LSTM/GRU y límites de evaluación conocida. Las tablas siguientes documentan únicamente la procedencia del registro previo y no fijan el calendario de nuevas ejecuciones.

El contrato histórico utilizado por E2 procede de la **fase II D7** del [Entregable 1, commit `94a4b03`](https://github.com/jassan-bit/DeepLearning_202630/tree/94a4b0324e93361e8928d708fdfeb50110e3686f). El [contrato descargable histórico](../_static/e1_reference.json) identifica esa procedencia. Las recetas y resultados de ese registro conservan su identidad; no definen la nueva selección de modelos ni se reasignan a las referencias actuales MLP, LSTM y GRU de E1.

| Elemento heredado | Definición conservada en E2 |
|---|---|
| Problema | Volatilidad histórica de Bitcoin, BTCUSDT en Binance Spot; formulación heredada inicialmente denominada regresión temporal, con ruta E2 resuelta como F2a, series de tiempo |
| Entrada | D=7 días de cinco canales OHLCV; horizonte h=1 día |
| Objetivo | σ₇ poblacional de siete retornos logarítmicos diarios de Close, sin anualización |
| Referencias de E1 | MLP16, LSTM32 y GRU32, fuera del Top 10 e incluidas en los dieciséis sistemas; receta y calendario común definidos en §43–§45 |
| Métrica primaria | Media entre semillas de la RMSE de cada trayectoria sobre las fechas elegibles comunes, en escala original |
| Margen absoluto | δ = **0,00025772842664981106** unidades de volatilidad diaria |
| Unidad de independencia | Una serie BTCUSDT de un proveedor; las fechas y etiquetas solapadas tienen dependencia temporal |
| Partición | Calendario de fase II descrito abajo; no se reasignan sus periodos |

Las recetas del experimento de origen se consultan en el [diseño versionado de fase II](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/experiments/gru_lstm_tcn_planned_20261105/design.json). Se conserva su trazabilidad sin reproducir aquí una segunda selección de modelos.

El siguiente calendario describe la procedencia histórica de los datos. La [enmienda registrada de §45](45_enmienda_protocolo.md) identifica el uso actual de cada periodo y de cada referencia sin retrofechar decisiones.

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

El Top 10 vigente contiene **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**. Son modelos de la literatura o adaptaciones de familias; no se presentan como los nombres literales del curso. **CNN-1D + LSTM + atención temporal** conserva una propuesta histórica opcional, sin superioridad demostrada y sin obligación de completar nuevas corridas para el catálogo requerido.

Para esos modelos, el espacio de hipótesis es $\mathcal H=\bigcup_{m\in\mathcal M}\mathcal H_m$. Las diez redes y las tres referencias E1 parten del contrato $\mathcal X\subseteq\mathbb R^{7\times5}$. No incorporan historia mensual ni etiquetas rezagadas como entradas adicionales. La variable objetivo es no negativa; el enlace de salida y cualquier posprocesamiento se fijan por receta antes de puntuar. MLP16, LSTM32 y GRU32 mantienen sus cuerpos E1, se reinicializan y conservan identidad experimental propia dentro del catálogo obligatorio.

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

La clasificación metodológica de E2 es **F2a, series de tiempo**, resuelta tras recibir la guía completa. El {ref}`diagnóstico residual de §2.2.3 <e2-diagnostico-forward>` aporta autocorrelación fuerte de la referencia histórica Ridge; RT1 (PDF p32) exige reclasificar, además del diagnóstico del EDA C (p6). La {ref}`sección 7 <e2-ruta-temporal>` conserva esa resolución y los controles causales. Se conserva la tarea predictiva, D7×5, σ₇, h=1 y δ; la reclasificación no añade Y rezagada como canal de las redes. El calendario y el soporte de los comparadores obligatorios se fijan en §45; retirar Ridge del alcance obligatorio no elimina su diagnóstico histórico.

Tres trabajos respaldan la relevancia científica del pronóstico de volatilidad en criptomonedas: Miura, Pichl y Kaizoji estudian redes neuronales para volatilidad realizada {cite:p}`miura2019`; Guo, Bifet y Antulov-Fantulin estudian la información de órdenes de compra y venta {cite:p}`guo2018`; Akbiyik y colaboradores incorporan información de Twitter al pronóstico de volatilidad de Bitcoin {cite:p}`akbiyik2023`.

Estos antecedentes motivan investigar memoria temporal, covariables y generalización. **No se afirma que utilicen nuestro snapshot, idéntico objetivo σ₇ ni exactamente las entradas aquí disponibles.** Tampoco constituyen por sí solos la revisión sistemática de E2 ni prueban uso del dataset exacto en una revista Q1. El interés aplicado es medir capacidad predictiva y coste bajo un benchmark académico; no se concluye rentabilidad, reducción de riesgo financiero o utilidad de trading.

### 1.1.5 Métricas, margen y unidad de independencia

La primaria es RMSE; las secundarias heredadas son MAE, MSE, R², MAPE y MASE. MAPE se define solo donde el objetivo sea positivo y se reportará su cobertura; no se añadirá un épsilon arbitrario. MASE usa el denominador histórico ingenuo de lag 1 sobre pares de días consecutivos elegibles de entrenamiento; no se calcula comprimiendo huecos ni usando test para escalar.

El margen absoluto heredado procede de $0,05\times0,005154568532996221$, RMSE CV de la GRU/D7 de fase I. **No se recalcula como el 5 % de una nueva RMSE.** Se conserva aparte la regla relativa académica de E1 frente a GRU y persistencia; no se confunde una reducción porcentual con δ absoluto. Su origen y fecha están documentados: heredar el margen no lo vuelve una decisión previa a los resultados históricos iniciales.

Hay **una única serie de mercado** ($N_{\text{series}}=1$). Las semillas describen variabilidad del entrenamiento condicionada a los datos; no son nuevas series independientes. Las etiquetas σ₇ comparten retornos y las ventanas comparten observaciones. La comparación debe emparejar fechas y distinguir variación entre semillas de incertidumbre temporal. El remuestreo por bloques evita suponer independencia diaria, pero su diseño y sensibilidad requieren justificación en desarrollo. Ningún número de semillas de E1 o del P0 histórico acredita potencia para las nuevas comparaciones. §45 registra el procedimiento de dimensionamiento; su ejecución con las recetas seleccionadas y el valor de R siguen pendientes.

## 1.2 Preguntas de investigación e hipótesis

Las tres preguntas conservan la formulación científica original. La [enmienda de §45](45_enmienda_protocolo.md) concreta RQ1 en **veinte contrastes primarios del Top 10 frente a GRU32 y Persistence**, con ajuste Holm; RQ2 y RQ3 se mantienen como análisis secundarios exploratorios y no reciben automáticamente la potencia ni el control familiar de RQ1. El [registro anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0/experiments/e2_remediation_20261106/preregistration.json) conserva su alcance histórico. Las hipótesis del híbrido opcional quedan fuera de la familia primaria vigente.

Para cada candidato $m$, sea $g_m=\mu_{\mathrm{ref}}-\mu_m$: valores positivos indican menor RMSE que la referencia predefinida. Las diferencias deben usar las mismas fechas elegibles y un tratamiento explícito de las semillas. MLP16, LSTM32 y GRU32 aportan referencias separadas; su receta, cohorte y papel inferencial no se escogen después de mirar los resultados. Ningún candidato recibe la etiqueta de ganador por construcción.

### 1.2.1 RQ1 · Ganancia predictiva relevante

**Pregunta:** ¿algún candidato reduce la RMSE respecto a la referencia predefinida en más que el margen heredado δ?

$$
H_{0,m}:g_m\le\delta,\qquad H_{1,m}:g_m>\delta.
$$

El efecto mínimo es δ en unidades originales de volatilidad. §45 registra un contraste unilateral pareado respecto a δ mediante bootstrap temporal por bloques, con las mismas fechas remuestreadas para ambos modelos y la variabilidad entre trayectorias tratada separadamente. La distribución nula se centra en la frontera $g_m=\delta$; no basta probar una diferencia contra cero. La alternativa registrada para planificar potencia es $2\delta$; no se promete potencia de detección en la frontera de la hipótesis nula ni potencia ya demostrada.

La decisión requerirá un contraste ajustado y un límite de incertidumbre compatibles con el margen δ y con el control de multiplicidad que se registre. Se reportarán efecto absoluto, reducción relativa, intervalos y p-valores ajustados y sin ajustar. Una ganancia puntual superior a δ, por sí sola, no prueba mejora práctica con incertidumbre controlada.

### 1.2.2 RQ2 · Deterioro de la ventaja en alta volatilidad

**Pregunta:** ¿la ventaja de un candidato frente a la referencia disminuye en más de δ durante días de alta volatilidad conocida al emitir el pronóstico?

El antecedente metodológico define el estrato causal $A_t=\mathbf1\{\sigma_7(t-1)>q_{0,75}^{2023}\}$, con $q_{0,75}^{2023}=0,024247163097617024$, obtenido únicamente de las 350 etiquetas válidas σ₇ de 2023. Su [perfil de entrenamiento está versionado en E1](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/eda_general_quality/target_profiles.csv). El estrato usa información pasada; no es un nuevo canal de entrada ni un umbral optimizado en test.

Con RMSE calculada por trayectoria dentro de cada estrato, sean $g_{m,N}$ y $g_{m,A}$ las ganancias en régimen normal y alto, y $d_m=g_{m,N}-g_{m,A}$:

$$
H_{0,m}:d_m\le\delta,\qquad H_{1,m}:d_m>\delta.
$$

El efecto mínimo es una erosión absoluta δ de la ventaja. Se requiere bootstrap temporal pareado sobre el calendario común, conservando las etiquetas de régimen y recalculando las métricas dentro de cada estrato. La alternativa para dimensionar este contraste sigue pendiente; su frontera es δ. Se reportarán cantidades de fechas y bloques útiles por estrato; si son insuficientes, el resultado será no estimable o inconcluso. Rechazar indica deterioro relevante; **no rechazar no demuestra robustez ni equivalencia**.

### 1.2.3 RQ3 · Mejora conjunta de error y coste

**Pregunta:** ¿algún candidato mejora la RMSE en más de δ y reduce el coste de inferencia frente a la referencia predefinida?

Sea $c_m$ la diferencia de latencia media por ventana, candidato menos referencia, bajo el mismo hardware, precisión y batch size:

$$
H_{0,m}:(g_m\le\delta)\ \text{o}\ (c_m\ge0),\qquad
H_{1,m}:(g_m>\delta)\ \text{y}\ (c_m<0).
$$

La afirmación conjunta requiere una prueba de intersección-unión: ambas condiciones deben quedar respaldadas; el p-valor conjunto será el máximo de los dos p-valores unilaterales válidos. El margen de error sigue siendo δ; el umbral de coste es cero, sin inventar una tolerancia financiera. Su dimensionamiento específico no está cerrado; no se inventa un ahorro observado ni se transfiere a RQ3 la potencia planificada para RQ1. Se reportará también la magnitud e intervalo del ahorro para valorar su importancia operativa.

La latencia debe medirse con entradas de desarrollo o sintéticas, sin volver a puntuar test para medir tiempos. Se deben documentar hardware, precisión, batch, calentamiento, bloques y orden intercalado de cada medición. El híbrido opcional conserva sus propios controles si se analiza. Las repeticiones del cronómetro no aumentan el tamaño muestral de la evaluación predictiva.

### 1.2.4 Multiplicidad y criterio de decisión común

La familia primaria vigente contiene veinte comparaciones del Top 10 frente a GRU32 y Persistence y utiliza Holm con nivel familiar 0,05, según §45. **No se heredan los recuentos ni las asignaciones estadísticas del catálogo retirado.** Las referencias E1 y los comparadores sí forman parte de los dieciséis sistemas obligatorios; el híbrido y Ridge no forman parte de esa familia primaria.

RQ2 y RQ3 conservan alcance exploratorio. Si se comunica el contraste conjunto de RQ3, se obtiene primero el p-valor de intersección-unión y se declara su multiplicidad y sus límites; no se incorpora silenciosamente al ajuste primario. Los IC marginales del 95 % se distinguen de los límites simultáneos. Las pruebas de equivalencia, calibración, robustez y diagnóstico necesitan su propio alcance; no quedan cubiertas automáticamente por el ajuste de las preguntas principales.

El margen absoluto δ se mantiene como referencia metodológica; la tolerancia de brecha train–validación y las alternativas de potencia deberán fijarse explícitamente. El [P0 histórico de §9](09_protocolo_registrado.md) completó 220 trayectorias y 440 ajustes de desarrollo, pero no selecciona ni dimensiona el catálogo vigente. Aunque se controle multiplicidad, reutilizar un test conocido mantiene el alcance exploratorio.

## 1.3 Justificación del dataset

### 1.3.1 Selección, tamaño y calidad

Se conserva **Binance Spot BTCUSDT**, obtenido de [Binance Public Data](https://github.com/binance/binance-public-data) y [Binance Vision](https://data.binance.vision/). Es pertinente por ofrecer una serie continua de un mercado de criptomonedas con OHLCV y trazabilidad de archivos; permite reproducir el benchmark de E1. El tamaño bruto no se interpreta como millones de muestras supervisadas independientes.

El snapshot inventariado contiene **75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas de doce campos**, entre el 01/01/2023 y el 30/10/2026. Los 1.399 días del calendario corresponden a 2.014.560 minutos esperados. Se detectaron **80 minutos ausentes** y una anomalía adicional de `close_time` el 24/03/2023. Por ello, ese día no es elegible para OHLCV diario completo; no se imputa ni se elimina el hueco del calendario.

El desarrollo 2023–2024 tiene 731 días de calendario y 730 días OHLCV completos. En 2023 hay 350 etiquetas σ₇ válidas; en 2024, 366. Las pérdidas adicionales de ventanas dependen de D, el objetivo y el embargo: los conteos de filas brutas, días y ejemplos supervisados son distintos. El ajuste final D7 de E1 utilizó 709 ejemplos elegibles; no se atribuye ese número a futuras configuraciones de E2.

La literatura citada respalda el problema y distintas fuentes informativas. **El uso previo de esta fuente o dataset en literatura Q1 requiere aún una referencia verificable y el cuartil de su año.** El criterio de la guía no exige que un artículo haya utilizado exactamente nuestro snapshot, sus fechas o hashes; esa identidad no se confunde con el respaldo de la fuente. La matriz de estudios de E2 mantiene pendiente esta comprobación, sin usar la reputación de una editorial como sustituto de evidencia.

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
| Tres RQ, hipótesis, efecto y reglas | RQ1 registrada con veinte contrastes primarios; RQ2/RQ3 exploratorias; ejecución y potencia pendientes |
| Justificación y Data Card | Documentadas con procedencia, versión, hashes, calidad y límites |
| Uso previo de la fuente/dataset en Q1 | Pendiente de referencia y cuartil por año; no exige identidad exacta del snapshot |
| Catálogo y protocolo | Dieciséis sistemas obligatorios; nueve cuerpos definidos y TiDE NAS; enmienda registrada, transición y evaluación pendientes |
| Prueba desconocida y confirmación nueva | No disponibles en los periodos heredados ya evaluados |

La [revisión bibliográfica de §3](03_literatura.md) distingue las fuentes de los modelos, su adaptación a regresión de σ₇ y la evidencia aplicada a criptomonedas. Una arquitectura publicada o un original leído no acreditan su funcionamiento en este dataset. La comprobación de uso de la fuente en Q1 y el cierre bibliográfico se mantienen separados de la ejecución del nuevo benchmark.
