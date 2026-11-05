# 1. Definición del Problema y Dataset

**Dos fases del estudio.** Las tablas de cuatro entradas, diez semillas y test enero de 2025–agosto de 2026 de este capítulo corresponden a la **fase I exploratoria**. La **fase II D7** añade búsqueda equitativa, piloto anterior a su ejecución definitiva, dimensionamiento y evaluación preespecificada sobre el 01/09–30/10/2026. Sus recetas, recuentos y resultados se identifican por separado en el [capítulo 9](09_estudio_planificado.md).

**Correspondencia con los requisitos de las páginas 2–3**

La siguiente tabla permite localizar las respuestas solicitadas en la guía. La formulación incorpora las decisiones aceptadas para la validación forward chaining de Filipe, diferenciando la selección con desarrollo de 2024 de la evaluación histórica ya observada, con configuraciones del nuevo ciclo congeladas antes de puntuarlo.

| Requisito del entregable | Respuesta y ubicación |
|---|---|
| Espacios de entrada y salida, función y pérdida | §1.1.1: ventana OHLCV diaria, salida no negativa, predictor supervisado y MSE |
| Tipo de problema admitido | §1.1.2: tipo 4, regresión temporal; ruta 6.4 de la guía, con entrada OHLCV sin rezagos explícitos de volatilidad |
| Pregunta de investigación e hipótesis operativa | §1.1.3: comparar el error de pronóstico con persistencia |
| Métricas primaria y secundarias | §1.1.4: RMSE principal; MAE, MSE, R², MAPE y MASE complementarias, con sus condiciones explícitas |
| Margen práctico e independencia | §1.1.4: criterio futuro del 5 % frente a GRU/D7 y persistencia, declarado después del test; inferencia entre semillas condicionada al calendario |
| Relevancia científica o aplicada | §1.2.1: pronóstico de incertidumbre de Bitcoin y antecedente científico |
| Disponibilidad, licencia y procedencia | §1.2.2 y §1.3.2: Binance Spot BTCUSDT, archivos oficiales, términos y reglas de agregación diaria documentados |
| Uso previo del dataset en literatura | §1.2.3: no verificado; el artículo citado es antecedente del problema |
| Tamaño, dimensionalidad y almacenamiento | §1.3.1: 75 ZIP, 2.014.480 filas CSV de 12 columnas y esquema de entrada definido; muestras de desarrollo y test supervisado materializadas, con elegibilidad y hashes |
| Tipo de datos y unidad de observación | §1.3.2: vela de un minuto en los brutos, día UTC en los derivados y origen temporal en las muestras supervisadas |
| Etiquetado y ruido esperado | §1.3.3, con fórmula en §1.1.1: cálculo automático del siguiente valor de volatilidad histórica móvil y fuentes de ruido |
| Dependencia entre observaciones | §1.3.4, con unidad declarada en §1.1.4: solapamiento temporal, partición cronológica y prevención de fuga |
| Versión y hash del dataset | §1.3.5: instantánea del 02/11/2026, manifiesto y SHA-256 de los 75 ZIP; derivados e índices de desarrollo disponibles; manifiesto supervisado de test y congelación final disponibles |

**Fase I, conservada como benchmark exploratorio:** se comparan **GRU, LSTM y TCN** con entradas D=7/14/21/28, objetivo fijo n=7 y horizonte h=1. La CV utiliza doce bloques mensuales de Filipe y diez semillas por red e historia: **120 trayectorias nuevas y 43.920 predicciones**. LSTM se reentrenó como control y reprodujo 14.640 predicciones. Tras cerrar las configuraciones se realizaron 80 ajustes finales nuevos y se verificó la reutilización de 40 LSTM: 120 checkpoints, 608 fechas de test y 72.960 predicciones. El periodo 2025–agosto de 2026 ya había sido observado: esta comparación es exploratoria. La [selección con CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) identifica **GRU/D7** como baseline. El capítulo 3 documenta su respaldo aplicado y los límites de recurrencia y adaptación.

## 1.1. Definición del problema

### 1.1.1. Espacios de entrada y salida, función y pérdida

**Retornos diarios y reloj de información**

Sea $P_d>0$ el cierre del día UTC $d$. El retorno de modelado es **diario**:

$$
r_d=\log\left(\frac{P_d}{P_{d-1}}\right).
$$

El pronóstico del día $t$ se emite **después del cierre del día $t-1$ y antes de observar el cierre del día $t$**. En ese momento solo están disponibles OHLCV y retornos de días con índice menor o igual a $t-1$. Un OHLCV diario no está disponible al comenzar ese mismo día.

**Historias de entrada evaluadas**

El usuario confirmó que los tamaños representan **días**, no minutos. La comparación principal de las tres redes utiliza **D∈{7,14,21,28}** observaciones diarias consecutivas, manteniendo el mismo objetivo n=7 y horizonte h=1. Las cuatro entradas se reentrenaron bajo la misma regla de Filipe; las predicciones LSTM se cotejaron con su control histórico:

| Historia $D$ | Velas de un minuto que originan esos días | Observaciones de entrada al modelo diario |
|---|---:|---:|
| 7 días | 10.080 | 7 |
| 14 días | 20.160 | 14 |
| 21 días | 30.240 | 21 |
| 28 días | 40.320 | 28 |

No se confunde la cantidad de velas brutas con el número de pasos diarios que recibe la red. El **Close** es la variable central para los retornos y las etiquetas. Para la configuración OHLCV diaria aportada, sea $z_d=(O_d,H_d,L_d,P_d,V_d)$:

$$
\mathcal X_D\subset\mathbb R^{D\times5},\qquad
X_t=(z_{t-D},\ldots,z_{t-1})^\top.
$$

Cualquier transformación o variable derivada debe calcularse causalmente. El contrato conserva cinco covariables OHLCV. La representación principal `relative_window` usa logaritmos de OHLC menos el logaritmo del último cierre de la propia ventana, conocido en d−1; volumen conserva log(1+V). Escalas de entrada y media positiva de Y se estiman únicamente con entrenamiento permitido. Se mantienen n=7, h=1 y ausencia de rezagos explícitos de volatilidad.

**Volatilidad histórica y objetivo de un día adelante**

Se usa la definición histórica del material específico del curso recibido, con divisor poblacional $n$:

$$
\bar r_t=\frac1n\sum_{i=1}^{n}r_{t-i},\qquad
\sigma_t=\sqrt{\frac1n\sum_{i=1}^{n}(r_{t-i}-\bar r_t)^2}.
$$

Aquí $n\ge2$ es la cantidad de **retornos diarios** usada para calcular la volatilidad histórica y es distinta de los $D$ días de entrada. La definición principal aceptada es **$n=7$**; $n=14,21,28$ son sensibilidad sobre objetivos diferentes. El EDA ya exploró las cuatro definiciones. Las comparaciones de modelos deben mantenerse dentro de un mismo $n$: una RMSE menor para una volatilidad más suavizada no permite elegir otro objetivo como si se tratara de la misma tarea.

Con la convención de emisión al inicio del día $t$, $\sigma_t$ ya es conocida. El horizonte principal de un día consiste en pronosticar:

$$
Y_t=\sigma_{t+1},\qquad
\mathcal Y=\mathbb R_{\ge0},\qquad
f_\theta:\mathcal X_D\longrightarrow\mathcal Y,\qquad
\widehat Y_t=f_\theta(X_t).
$$

$\sigma_{t+1}$ incorpora $r_t$, que todavía no se conoce al emitir el pronóstico y se observa al cerrar ese día. Esta indexación evita presentar como pronóstico una cantidad ya calculable con los datos disponibles. Es una convención operacional explícita de la fórmula del curso; no se anualiza ni multiplica por 100 en el cálculo. Las gráficas pueden mostrar porcentaje con la conversión indicada.

El objetivo es el siguiente valor de una **volatilidad histórica móvil**. Las ventanas consecutivas comparten $n-1$ retornos; no se está pronosticando la volatilidad de $n$ días enteramente futuros. La volatilidad es un proxy de dispersión de retornos observados, con incertidumbre de muestreo y posible ruido del proveedor. No se confunde con volatilidad latente observada sin error. Las etiquetas necesitan $n$ retornos válidos consecutivos y, por tanto, $n+1$ cierres diarios; no se forman atravesando días incompletos.

**Pérdida y referencia contextual**

Se fija MSE como pérdida supervisada, sobre los orígenes elegibles de entrenamiento:

$$
\mathcal L(\theta)=\frac1{|\mathcal T_{\mathrm{train}}|}
\sum_{t\in\mathcal T_{\mathrm{train}}}(Y_t-f_\theta(X_t))^2.
$$

La referencia de persistencia para el mismo horizonte será $\widehat Y_t^{\mathrm{pers}}=\sigma_t$, calculada únicamente con el pasado. No ocupa uno de los tres puestos de modelos de Deep Learning. Se comparará en los mismos orígenes elegibles para cada tamaño de entrada. Para $n=7$ resume ocho cierres; una entrada de siete días OHLCV no incorpora explícitamente esa misma estadística. La comparación entre redes conservará entradas idénticas, mientras persistencia se identificará como referencia contextual con información histórica disponible.

La ruta 6.4 exige además una **regresión lineal regularizada con las mismas covariables rezagadas**. Como diagnóstico del bloque C se ejecutaron cuatro referencias Ridge con intercepto libre y penalización fija $\alpha=1$, una para cada $D=7,14,21,28$, sobre las $5D$ covariables de su ventana. Cada referencia se ajustó una vez con muestras admisibles de 2023, aplicando el embargo del primer origen de 2024, y se conservó fija para obtener 366 predicciones fuera de entrenamiento en 2024: **1.464 en total**. La estandarización utiliza exclusivamente la media y desviación poblacional del ajuste permitido. El [resumen reproducible](../_static/eda_temporal_reference/summary.json) y el bloque C de §2.2.3 documentan predicciones y residuos; no se ajustó $\alpha$ ni se ordenaron las historias por rendimiento.

La autocorrelación residual detectada exige reconocer la insuficiencia de esta referencia y la dependencia del objetivo móvil; no demuestra por sí sola que deban añadirse rezagos de volatilidad. El contrato vigente sigue siendo tipo 4. Añadir esos rezagos requeriría actualizarlo al tipo 3 y revisar la ruta. La referencia lineal no ocupa uno de los tres puestos DL ni congela la política de reajuste del benchmark posterior; persistencia conserva el papel adicional de referencia para el margen operativo aceptado del 5 %.

Escalado y cualquier transformación aprendida se ajustan solo con el entrenamiento permitido. La CV de Filipe utiliza 2023 y, progresivamente, el pasado admisible de 2024. Cada mes selecciona duración con validación interna reciente y realiza un reajuste nuevo con la cohorte mensual admitida. Los pesos quedan fijos dentro del mes, mientras cada pronóstico diario usa solo OHLCV de días anteriores. La política excluye los días incompletos y sus ventanas afectadas, sin imputación. Las métricas se expresan en la escala original y las tres redes implementan salida Softplus no negativa.

### 1.1.2. Tipo de problema admitido

**Tipo admitido número 4: regresión temporal; ruta 6.4 de la guía**, con salida continua y aprendizaje supervisado. La entrada definida es una secuencia de covariables OHLCV; no incluye rezagos explícitos de la volatilidad objetivo. El problema sigue siendo un pronóstico temporal y exige disponibilidad causal, partición cronológica con embargo y validación de origen móvil. La clasificación depende del contrato de entrada y no de usar LSTM, TCN o atención. Si se incorporasen rezagos del objetivo como entrada, correspondería revisar la clasificación como series de tiempo, tipo 3, y registrar el cambio antes de ejecutarlo.

### 1.1.3. Pregunta de investigación e hipótesis operativa

**Pregunta de investigación:** ¿pueden tres modelos básicos de Deep Learning, cuya selección debe acreditarse en la literatura y usando 7, 14, 21 o 28 días de información OHLCV, pronosticar el siguiente valor de la volatilidad histórica de siete retornos diarios con menor RMSE que persistencia bajo evaluación cronológica con embargo?

**Regla de selección y alcance:** se elige la historia de cada familia y el baseline neuronal por la media de RMSE de CV2024. La regla seleccionó **GRU con D=7** antes de acceder al contenido del test en este ciclo. Las diferencias frente a persistencia se reportan en las mismas fechas. El test ya había sido observado en el proyecto; no se formula retrospectivamente una hipótesis confirmatoria previa a sus resultados. El margen académico para futuros modelos y la evaluación actual se distinguen en §1.1.4 y §5.

### 1.1.4. Métricas, margen práctico y unidad de independencia

Para cada modelo $m$, con historia $D$ y configuración fijadas, se distinguen diferencia absoluta y mejora relativa; el subíndice $m$ abrevia esa configuración y los resultados de distintas ventanas se mantienen separados:

$$
\Delta_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m,
\qquad
g_m=\frac{\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m}
{\mathrm{RMSE}_{\mathrm{pers}}},
\quad \mathrm{RMSE}_{\mathrm{pers}}>0.
$$

La mejora relativa frente a persistencia se conserva como **descripción del mismo objetivo y las mismas fechas**. Una estimación puntual favorable, una diferencia estadísticamente distinguible de cero y una mejora que supera un margen práctico son afirmaciones diferentes. Los análisis nuevos entre la terna se declaran posteriores a las observaciones y exploratorios; no se atribuye su protocolo al periodo de mercado. La fase II registra sus propias comparaciones antes de puntuar una reserva distinta; no cambia esta cronología.

**Margen operativo para una propuesta futura.** Después de evaluar esta terna se propone una reducción relativa de RMSE **superior al 5 % frente a GRU/D7 y frente a persistencia** en una nueva evaluación independiente, con incertidumbre y multiplicidad declaradas. El ancla neuronal de desarrollo es RMSE = 0,005154568533, por lo que el margen absoluto de planificación neuronal es:

$$
\delta_{\mathrm{futuro}}=0{,}05\times0{,}005154568533
\simeq0{,}00025772842665.
$$

Esta definición es **posterior al test actual**. No constituye un margen pretest de sus contrastes ni una tolerancia de riesgo o rentabilidad validada. Las sensibilidades al 2,5 % y al 10 % son escenarios operativos académicos. La comparación futura tendrá que registrar su pérdida, dos referencias, unidad de repetición, alternativa y tamaño de muestra antes de producir resultados. El promedio por sí solo no demuestra superar la frontera práctica.

Las diez semillas de la CV de fase I fueron un presupuesto fijo y no procedieron de un piloto. La sensibilidad retrospectiva no sustituye ese requisito. La fase II D7 incorpora un piloto antes de sus ajustes definitivos y dimensionamiento con las diferencias pareadas GRU–LSTM–TCN, como se detalla en [§9.4–9.5](09_estudio_planificado.md). Su potencia es de planificación entre semillas, condicionada al calendario; no acredita potencia temporal ni otros mercados. No se han entrenado modelos avanzados.
| Elemento | Decisión y alcance |
|---|---|
| Objetivo principal | Desviación poblacional de siete retornos diarios; próximo valor de la medida móvil |
| Horizonte | Un día |
| Historias de entrada | D=7/14/21/28 días OHLCV evaluadas, con n=7 fijo |
| Métrica primaria | RMSE en la escala original del objetivo |
| Pérdida implementada | MSE; escala positiva del objetivo ajustada solo con entrenamiento e inversión para evaluar |
| Métricas complementarias | MAE, MSE, MAPE, R² y MASE, con cobertura y casos no definidos |
| Margen operativo futuro | 5 % frente a GRU/D7 y persistencia; definición posterior al test actual, para una evaluación nueva |
| Nivel de significación del análisis | $\alpha=0{,}05$; familias por fase, con comparaciones dentro de la terna declaradas post-hoc y corrección de Holm |
| Partición | Historia inicial 2023; validación 2024; test 01/01/2025–31/08/2026 |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad de independencia entre mercados | Una sola serie BTCUSDT de Binance Spot; $N_{\mathrm{series}}=1$, sin réplicas independientes de otros mercados |
| Unidad inferencial dentro de la serie | Diez trayectorias por red/variante condicionadas al calendario; los folds mensuales y las semillas no representan mercados independientes |
| Referencias de contexto | Regresión lineal regularizada con las mismas covariables, exigida por la ruta 6.4; persistencia adicional para el margen del 5 % |
| Selección de modelos | GRU, LSTM y TCN; configuración congelada por fase y selección de D/baseline con CV. Las recetas son adaptaciones compactas con respaldo aplicado de familia, no réplicas exactas |

Para $N$ predicciones evaluables:

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_t(Y_t-\widehat Y_t)^2},
\qquad
\mathrm{MAE}=\frac1N\sum_t|Y_t-\widehat Y_t|.
$$

**MAPE:** se calculó sobre etiquetas estrictamente positivas, con 366 casos y cobertura del 100 % en 2024; no hubo ceros en esta cohorte. La política informa cantidad de ceros, casos incluidos y cobertura en cualquier otra evaluación. No se añade epsilon ni se eliminan casos sin declararlo. Si no hay etiquetas positivas, la métrica se marca no definida. La posible sensibilidad a valores positivos pequeños se discutirá junto a RMSE y MAE.

**MASE:** se fija un rezago de referencia de **un día de calendario**, coherente con persistencia diaria y sin imponer una estacionalidad no demostrada. Para el entrenamiento permitido en el origen $t$:

$$
a_{\mathrm{train}(t),1}=
\frac{1}{|\mathcal J_t|}\sum_{j\in\mathcal J_t}|Y_j-Y_{j-1}|,
\qquad
\mathrm{MASE}=\frac1N\sum_t\frac{|Y_t-\widehat Y_t|}{a_{\mathrm{train}(t),1}}.
$$

$\mathcal J_t$ contiene pares de etiquetas válidas separados por un día de calendario y completamente observados en ese entrenamiento. El denominador es común a los modelos en el mismo origen. Si vale cero o faltan pares, MASE queda no definida y se informa su cobertura. No se comprimen huecos ni se sustituye esta media de errores escalados por dividir la MAE global entre un promedio distinto de denominadores.

**Reporte por cada tamaño de entrada**

Las historias **D=7/14/21/28** tienen tablas separadas de **MAPE, MAE, RMSE y MSE**, con media y DE entre las diez semillas. R², MASE, cobertura e IC complementan ese reporte. Los archivos [resúmenes de CV](../_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/metric_summary_test.csv) permiten reconstruirlo. n=14/21/28 no se entrenó: esas sensibilidades cambiarían el objetivo, no solo la entrada.

El **BDS de residuos de h=1 se calculó por bloque y semilla** para las tres redes y cuatro entradas, con dimensión de incrustación 2 y radio 1,5 DE muestral. Se reportan valores sin ajustar y Holm en cada familia declarada, estados no definidos y media/DE descriptivas de p-valores. El promedio no es un contraste combinado ni una probabilidad de independencia. Las muestras mensuales cortas, etiquetas solapadas y cambios de régimen limitan la referencia asintótica.

**Estado de cierre**

La definición principal queda fijada en **n=7, h=1 y D=7/14/21/28**. La [selección de desarrollo](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) identifica GRU/D7 y la política de pesos finales fijos. El contenido del test se leyó en este ciclo después de cerrar los pesos; sus veinte meses son intervalos de reporte. El periodo ya había sido observado, por lo que esta separación no constituye confirmación temporal independiente.

La CV utiliza desarrollo expuesto y no constituye selección anidada ni potencia 0,80 acreditada. El diagnóstico Ridge permanece separado. La [matriz de cierre](protocolo_pendiente.md) distingue el piloto de fase II D7 de la ausencia de piloto previo en fase I, la limitada recurrencia de TCN y la necesidad de justificar las diferencias de sesgo entre GRU y LSTM. Una propuesta avanzada requiere su propia dispersión y otra evaluación independiente. TOST/ROPE y una banda de equivalencia solo corresponden si se pretende afirmar equivalencia.

## 1.2. Justificación del Dataset

### 1.2.1. Relevancia científica y aplicada

Se propone evaluar si el historial de covariables OHLCV diarias de Bitcoin contiene información útil para pronosticar el siguiente valor de su volatilidad histórica móvil. Los retornos logarítmicos de cierre se utilizan para construir el objetivo; las entradas definidas son las covariables OHLCV. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema {cite:p}`huang2024`. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. La revisión documental y las 40 referencias vigentes se desarrollan en el capítulo 3 y en [Referencias](referencias.md). El catálogo vigente distingue siete textos completos sobre criptomonedas de las fuentes metodológicas de otros mercados. Los siete incluyen Bitcoin; uno estudia además otros 19 activos, por lo que esos subconjuntos se solapan.

**Decisiones confirmadas:** BTCUSDT, agregación diaria UTC de velas de un minuto, volatilidad poblacional de siete retornos diarios, horizonte de un día, entradas **D=7/14/21/28** y RMSE primaria. Las volatilidades n=14/21/28 fueron sensibilidades de EDA y no son objetivos entrenados del benchmark.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

### 1.2.2. Disponibilidad pública, licencia y procedencia

**Fuente seleccionada:** Binance Spot, par BTCUSDT, velas de un minuto adquiridas directamente de los archivos públicos oficiales. La adquisición, verificación y auditoría se describen en §2.3 del [EDA](02_eda.md). Se verificaron 75 ZIP oficiales para el periodo 01/01/2023–30/10/2026 inclusive UTC: 2.014.480 filas. El manifiesto contiene los SHA-256 y la auditoría de timestamps. El EDA de calidad, precios, retornos y cuatro definiciones de volatilidad de desarrollo está ejecutado y documentado en la sección 2. La definición principal ya es $n=7$; sus muestras de desarrollo y su uso para entrenamiento están materializados en los [folds de CV](../_static/cv_forward_chaining/folds_D7_registered.json). La cohorte final y las firmas del ciclo vigente se identifican en el [manifiesto de procedencia](../_static/gru_lstm_tcn_20261105/protocols_manifest.json).

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Nombre y proveedor | Binance Spot; archivos públicos oficiales de velas de un minuto |
| Activo y par | Bitcoin, cotizado en USDT; par BTCUSDT |
| Disponibilidad pública | Archivos oficiales accesibles y descargados desde data.binance.vision; la API devuelve HTTP 451 por ubicación y no se utiliza |
| Licencia y redistribución | Binance Vision Dataset Terms v1.0, actualizados el 26/08/2026: CC BY-NC-SA 4.0 salvo indicación específica, con condiciones adicionales; copia y hash conservados en la revisión, ver §2.3.4 |
| Procedencia | Archivos oficiales de klines del mercado Spot BTCUSDT; esquema temporal y agregación diaria descritos debajo |

**Generación y agregación.** Los archivos de Binance Public Data contienen klines del mercado Spot BTCUSDT: cada registro incluye hora de apertura, OHLC, volumen, hora de cierre y campos complementarios de volumen/operaciones. La documentación de referencia es [Binance Public Data](https://github.com/binance/binance-public-data) y el esquema de [klines de Binance Spot](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data). El par pertenece a un mercado y proveedor concretos; no es un índice agregado de precios de múltiples exchanges.

Los términos de uso se conservan como {download}`copia documental <../_static/data_source/TERMS_AND_CONDITIONS.md>`, con su [registro de procedencia y SHA-256](../_static/data_source/terms_snapshot.json). Esta captura de la revisión del 03/11/2026 se distingue de la fecha de adquisición y de la consulta anterior; la disponibilidad pública se interpreta junto a esas condiciones.

### 1.2.3. Uso previo del dataset en la literatura

**Uso previo y comparabilidad:** la [matriz actual de evidencia](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.csv) identifica las siete fuentes completas del corpus cripto, **S1, S2, S5, S6, S7, S8 y S9**, y separa sus registros de transferencia financiera. No se acreditó que una publicación utilice la misma versión histórica de Binance Spot BTCUSDT y el mismo objetivo σ₇/h1. Las diferencias de proveedor, medida, frecuencia, horizonte e información se documentan sin declarar equivalencia entre datasets.

| Fuente del corpus completo | Datos y mercado documentados | Objetivo/horizonte y límite para comparar |
|---|---|---|
| S1 · Akbiyik et al. | Bitcoin cada 15 minutos y Twitter | RV del próximo día agregada de 96 retornos intradiarios; cambia definición, escala, información y periodo. |
| S2 · Zeri | Binance Futures BTCUSDT perpetuo, vía CryptoLake | Log RV futura de 30 minutos; futuros no es Spot y su microestructura no coincide con OHLCV diario. |
| S5 · Gao et al. | Yahoo Finance BTC-USD diario | Ventana de 30 días sin centrar y desplazamiento de siete días; cambia proveedor, par, normalización y horizonte. |
| S6 · Wang et al. | Bitcoin diario; proveedor insuficientemente especificado | RV1D sin fórmula suficientemente recuperada; no se equipara a σ₇. |
| S7 · Alakhouch | BTC/USD diario de Investing.com/Kaggle, según manuscrito/código | Código de volatilidad futura de 30 días anualizada; no es la etiqueta móvil σ₇/h1 del proyecto. |
| S8 · Miura et al. | BTCUSD de un minuto, fuente Kaggle Bitcoin historical data | Suma de retornos cuadrados en intervalos de tres horas, sin raíz; ese intervalo no acredita por sí solo un horizonte explícito de pronóstico. |
| S9 · Djanga et al. | Veinte pares cripto USDT de Binance vía Tardis, incluido BTC; segmento Spot/futuros no explícito | RV intradiaria de 30 minutos y trayectoria del día siguiente en rejilla de cinco minutos; definición y horizonte no se equiparan a σ₇ diario. |

Las pistas S3 y S4 de revisiones previas no se cuentan como originales completos dentro de ese denominador de siete. La [matriz de diez registros](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.json) conserva además T1/T2/T3 como transferencia financiera con sus niveles de lectura y verificación separados.

La recuperación también encontró software que declara Binance Spot/BTCUSDT a un minuto, pero con dispersión futura de diez minutos y sin un artículo verificable: aporta una pista de procedencia, no uso científico acreditado del dataset propio. Las cifras de las fuentes siguen siendo contexto; no se ordenan junto a las métricas propias como si procedieran del mismo experimento. El {download}`informe de recuperación <../_static/literature_e1/recovery_audit_20261104/report.md>` conserva ubicaciones, calidad y limitaciones. La igualdad de la versión exacta sigue abierta; la **matriz de diferencias ya está documentada**.

## 1.3. Descripción de la técnica

### 1.3.1. Tamaño, estructura y almacenamiento

La cuadrícula de desarrollo 2023–2024 contiene 731 días UTC y 730 días OHLCV completos; el 24/03/2023 se mantiene como posición ausente. Son recuentos de días, diferentes de las filas de un minuto y de las muestras supervisadas.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Tamaño | 75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas brutas; ventanas de desarrollo y test materializadas con manifiestos de elegibilidad |
| Estructura de modelado | OHLCV diario; CV y evaluación final con entradas $(N,D,5)$, D=7/14/21/28, y salida escalar de volatilidad de un día adelante |
| Formato de almacenamiento | CSV oficiales de 12 columnas dentro de ZIP; timestamps normalizados solo para auditar cobertura, brutos intactos |

Con $T$ días OHLCV consecutivos válidos y sin pérdidas por partición, construir entradas de $D$ días y una etiqueta con $n$ retornos diarios permite como máximo $\max\{0,T-\max(D,n+1)\}$ orígenes de un día adelante, incluyendo la historia necesaria para persistencia. Es una relación teórica, **no el tamaño real**; huecos, filtros y separación entre particiones pueden reducirlo.

### 1.3.2. Tipo de datos y unidad de observación

Se distingue **adquisición a un minuto** de **modelado diario**. Los archivos oficiales BTCUSDT se agregan en días UTC completos. Cada observación diaria contiene apertura, máximo, mínimo, cierre y volumen (OHLCV); un día incompleto se marca no elegible en los derivados, sin modificar los ZIP originales.

La unidad bruta es una vela de un minuto; la unidad derivada, un día UTC. La muestra supervisada corresponde al origen temporal t, con ventana histórica de entrada y etiqueta futura.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Periodo temporal y zona horaria | Descarga: 01/01/2023–30/10/2026 inclusive UTC; timestamps de apertura en milisegundos antes de 2025 y microsegundos desde 2025. Calendario experimental en §1.3.4 |
| Frecuencia | Brutos de un minuto; observaciones de modelado y retornos del objetivo diarios, cierre UTC |

La agregación usada por el proyecto agrupa por día UTC: toma la primera apertura, el máximo de los máximos, el mínimo de los mínimos, el último cierre y la suma de volúmenes. El cierre diario $P_d$ corresponde al último minuto del día, disponible al concluirlo. Se respeta el cambio documentado de unidades de timestamps: milisegundos antes de 2025 y microsegundos desde 2025; las unidades se normalizan para procesar tiempos, conservando intactos los ZIP.

La configuración aportada establece modelado diario, cierre UTC y horizonte principal de un día. Las noticias solo se incorporarían con una fuente verificable y fechas de publicación anteriores al pronóstico; no se han adquirido ni forman parte de la entrada implementada de este avance.

### 1.3.3. Variable objetivo, etiquetado y ruido esperado

**Etiquetas:** Cálculo automático de $Y_t$ según fórmula aprobada; sin anotación manual.

El objetivo se deriva de precios observados, por lo que puede reflejar errores del proveedor, precios anómalos, cambios de liquidez y ruido de microestructura. El tamaño finito de la ventana y la definición elegida introducen incertidumbre como proxy. No se ha medido todavía la magnitud de estos efectos.

El EDA ya comprobó orden temporal, duplicados, cierres no positivos, huecos, frecuencia y significado del timestamp sobre desarrollo; sus hallazgos y límites figuran en §2. En la construcción definitiva se conservarán esos controles. Un retorno calculado a través de un día ausente no se tratará como retorno diario consecutivo; tampoco se confunde un salto entre minutos separados con un retorno regular de un minuto.

La regla actual exige 1.440 minutos elegibles para acreditar un OHLCV diario completo. Los **80 minutos ausentes del 24/03/2023** y la fila con cierre temporal no estándar se conservan en el bruto, y el derivado diario afectado queda ausente. Se mantienen las posiciones de calendario y se excluyen las ventanas que necesiten esos datos, sin interpolar, rellenar volumen con cero ni fabricar etiquetas. Esta es la política vigente de E1; cualquier recuperación desde otra granularidad o cambio de imputación requiere documentar por separado fuente, validación y versión antes de usarla.

### 1.3.4. Dependencia entre observaciones, partición y calendario

**Dependencia:** Temporal y por ventanas solapadas; diagnósticos de desarrollo documentados en §2, sin presuponer independencia.

Las ventanas y las volatilidades móviles comparten información. Una fecha, una inicialización y un bloque temporal representan fuentes diferentes de variabilidad. Las semillas y los doce folds mensuales sobre un mercado no se contarán como mercados independientes. Los IC por semillas condicionan al calendario observado; remuestrear fechas requeriría un diseño que represente su dependencia y seguiría condicionado a los modelos guardados. Los intervalos entre semillas no se presentan como incertidumbre de otros periodos. Un único activo limita la generalización a otros mercados.

Así se declara la dependencia conocida, sin presentar como resuelto el requisito de independencia: ningún número de filas, folds o semillas convierte una serie en varios mercados. La inferencia temporal debe justificar sus supuestos, longitud de bloque y estabilidad en el procedimiento que se evalúa. El análisis de fase I utiliza 4.999 remuestras pareadas de las diez semillas como sensibilidad del tamaño de efecto $d_z$, con alcance post-hoc y condicionado al calendario y a los modelos guardados. La fase I no ejecutó bootstrap temporal para esta terna. La fase II registra antes de la nueva reserva un bootstrap circular temporal con bloque principal siete y sensibilidad tres/catorce, separado del remuestreo de semillas (§9.6.3).

**Separación temporal y prevención de fuga**

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- En cada origen, excluir del ajuste toda etiqueta que todavía no esté disponible; reservar el test para la evaluación posterior y no utilizarlo para selección.
- Ajustar escalamiento, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos; mantener la exclusión vigente sin imputación.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $D$, $n$ y la disponibilidad de etiquetas.
- Congelar y verificar el test por hash antes de evaluación definitiva; no usarlo para elegir arquitectura, hiperparámetros o preprocesamiento.

**Calendario proporcionado**

El calendario de fase I, conservado sin cambios, es el siguiente:

| Etapa | Periodo | Uso |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Construir las primeras muestras elegibles de entrenamiento |
| Validación forward chaining | Objetivos del 01/01/2024 al 31/12/2024 | Entrenar con el pasado y acumular errores de validación para comparar configuraciones |
| Ajuste final | Datos elegibles de 2023–2024 | Reajustar la configuración seleccionada |
| Test histórico de fase I | 01/01/2025–31/08/2026 | Evaluar las recetas cerradas; periodo ya observado, sin convertirlo en confirmación nueva |

Se reportan **2025 y enero–agosto de 2026 por separado**, con 365 y 243 fechas, respectivamente. La [tabla anual](../_static/gru_lstm_tcn_20261105/annual_metric_summary_test.csv) calcula cada métrica sobre sus fechas y luego resume semillas; 2026 no representa el año completo ni se obtiene la RMSE global promediando dos RMSE anuales.

Los 366 días de 2024 se evalúan mediante doce bloques mensuales de forward chaining, utilizando solo el pasado permitido para cada ajuste. Ese año ya estuvo disponible en el EDA y en la elección de configuraciones, por lo que sus resultados son desarrollo exploratorio. El ajuste final se realizó sobre el desarrollo permitido de 2023–2024, excluyendo los últimos D resultados antes de enero de 2025 por embargo; las épocas se fijaron con la validación interna de diciembre de 2024.

**Regla de disponibilidad para la preparación posterior:** cada muestra se asigna al periodo de su día de resultado $t$. La entrada termina en $t-1$ y el objetivo se conoce al cerrar $t$. Una etiqueta solo puede utilizarse para ajustar un modelo cuando su instante de disponibilidad es anterior o igual al origen que se pretende evaluar. Si necesita retornos de 2025, no pertenece al ajuste con desarrollo 2023–2024. Una etiqueta del 31/12/2024, ya observable al comienzo del 01/01/2025, se juzga por esa disponibilidad y no se excluye solo por la escritura $\sigma_{t+1}$.

La historia previa al inicio de validación o test puede formar parte de las entradas cuando ya sea observable. Los índices de Filipe se utilizan para entrenar una red al comienzo de cada mes, con pasado permitido de 2023 y de 2024; los pesos se conservan durante el mes y se predice cada día a h=1. Son doce procedimientos mensuales por trayectoria, no un reajuste por cada origen diario. La política final cerrada no actualiza pesos ni escaladores durante test: los veinte meses de 2025–agosto de 2026 son periodos de reporte de los mismos modelos finales.

La preparación corregida explicita el embargo de la ruta 6.4: para $X_t=(z_{t-D},\ldots,z_{t-1})$, el máximo rezago es $D$ días. Se omiten los días de resultado $t-D,\ldots,t-1$ del ajuste y se admiten candidatos $u\le t-D-1$, sujetos además a disponibilidad y calidad. Los 5.856 registros de partición —366 por cada una de las 16 configuraciones— aplican esa regla. El embargo no elimina toda la dependencia entre etiquetas móviles cuando $n>D+1$, que deberá conservarse en la inferencia.

Los intervalos se expresan en días UTC. El calendario contiene 365 días iniciales de 2023, 366 de validación de 2024 y **608 de test** hasta agosto de 2026; son días de calendario, no recuentos definitivos de muestras elegibles. Los huecos y la historia requerida reducen esa elegibilidad. Los índices, transformaciones y configuraciones se conservaron por corrida. La congelación anterior a la apertura de test, la auditoría de los veinte ZIP y el manifiesto de cohorte permiten verificar su ejecución final. Septiembre–octubre de 2026 se adquirieron sin asignarse a fase I. La fase II asigna explícitamente el 01/09–30/10/2026 bajo un nuevo registro previo; los porcentajes y cohortes iniciales permanecen intactos.

La evaluación prevista es un backtest histórico documentado después del periodo de mercado considerado. La reserva organiza el uso posterior de los datos; no se presenta como un pronóstico que se hubiera emitido en tiempo real durante 2025–2026.

### 1.3.5. Versión y hash del dataset

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Versión / fecha de extracción | Instantánea verificada el 02/11/2026; versión identificada por el manifiesto y SHA-256 individuales |
| Hash de datos | SHA-256 de los 75 ZIP contrastados con checksums oficiales; manifiesto enlazado en §2.3.2. Derivados y etiquetas de desarrollo documentados en §2; configuraciones e índices conservados en §4; congelación y manifiestos finales de test disponibles |

El SHA-256 del [manifiesto de adquisición](../_static/data/binance_spot_BTCUSDT_1m_20230101_20261030.json) es `af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2`; cada ZIP conserva además su huella individual.

La identidad de los brutos de test queda fijada en el [manifiesto de reserva de 20 ZIP mensuales](../_static/data/test_raw_reserve_20250101_20260831.json), construido inicialmente desde metadatos. El proyecto ya había observado ese periodo. El ciclo GRU–LSTM–TCN cerró sus configuraciones y checkpoints antes de acceder de nuevo al derivado diario auditado, sin reabrir ZIP brutos. El [manifiesto vigente de protocolos](../_static/gru_lstm_tcn_20261105/protocols_manifest.json) conserva fechas, hashes, reutilización y cohorte; la reserva inicial no se sobrescribe ni convierte esta comparación en un test independiente.

**Aplicación en la fase II D7.** El [capítulo 9](09_estudio_planificado.md) reúne el diseño público previo, la rejilla equitativa de ocho candidatos por familia, el piloto de 60 días, el cálculo de R y los resultados sobre el 01/09–30/10/2026. Esa fase mantiene n=7/h=1 y covariables OHLCV, con D7 fijo. Los resultados de fase I y sus limitaciones permanecen identificados; la potencia planificada entre semillas no representa potencia temporal ni confirma otros mercados.
