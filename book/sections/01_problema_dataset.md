# 1. Definición del Problema y Dataset

**Correspondencia con los requisitos de las páginas 2–3**

La siguiente tabla permite localizar las respuestas solicitadas en la guía. La formulación incorpora las decisiones ya aceptadas por el usuario. La evidencia documental, el EDA y el primer experimento de desarrollo ejecutado el 04/11/2026 se distinguen de la evaluación final aún pendiente.

| Requisito del entregable | Respuesta y ubicación |
|---|---|
| Espacios de entrada y salida, función y pérdida | §1.1.1: ventana OHLCV diaria, salida no negativa, predictor supervisado y MSE |
| Tipo de problema admitido | §1.1.2: tipo 4, regresión temporal; ruta 6.4 de la guía, con entrada OHLCV sin rezagos explícitos de volatilidad |
| Pregunta de investigación e hipótesis operativa | §1.1.3: comparar el error de pronóstico con persistencia |
| Métricas primaria y secundarias | §1.1.4: RMSE principal; MAE, MSE, R², MAPE y MASE complementarias, con sus condiciones explícitas |
| Margen práctico e independencia | §1.1.4: mejora relativa mínima del 5 % frente a persistencia; inferencia entre semillas condicionada al calendario y remuestreo temporal por bloques registrado para desarrollo |
| Relevancia científica o aplicada | §1.2.1: pronóstico de incertidumbre de Bitcoin y antecedente científico |
| Disponibilidad, licencia y procedencia | §1.2.2 y §1.3.2: Binance Spot BTCUSDT, archivos oficiales, términos y reglas de agregación diaria documentados |
| Uso previo del dataset en literatura | §1.2.3: no verificado; el artículo citado es antecedente del problema |
| Tamaño, dimensionalidad y almacenamiento | §1.3.1: 75 ZIP, 2.014.480 filas CSV de 12 columnas y esquema de entrada definido; muestras de desarrollo materializadas, test supervisado pendiente |
| Tipo de datos y unidad de observación | §1.3.2: vela de un minuto en los brutos, día UTC en los derivados y origen temporal en las muestras supervisadas |
| Etiquetado y ruido esperado | §1.3.3, con fórmula en §1.1.1: cálculo automático del siguiente valor de volatilidad histórica móvil y fuentes de ruido |
| Dependencia entre observaciones | §1.3.4, con unidad declarada en §1.1.4: solapamiento temporal, partición cronológica y prevención de fuga |
| Versión y hash del dataset | §1.3.5: instantánea del 02/11/2026, manifiesto y SHA-256 de los 75 ZIP; derivados e índices de desarrollo disponibles; manifiestos supervisados definitivos de test pendientes |

**Alcance de esta actualización:** la fuente, adquisición y EDA están documentados; el objetivo principal es volatilidad de siete retornos diarios y horizonte de un día. Las cuatro referencias Ridge del bloque C se conservan como diagnósticos de EDA. El 04/11/2026 el usuario autorizó correr MLP, LSTM y CNN1D poco profunda y actualizar E1, ampliando el alcance previo. El primer ciclo completó búsqueda, piloto y comparación de desarrollo con D=7 seleccionado, bajo el [protocolo registrado](../_static/dl_benchmark_20261104/protocol_initial.json). Sus resultados se conservan como **V1** en los capítulos 4–5. El ciclo CV posterior entrenó las tres redes con los índices reales de `split_train_val_forwardChaining`, desarrollado con coautoría de Filipe R. Ramos: doce bloques mensuales de 2024, dos representaciones OHLCV y diez semillas por red/variante. Su [protocolo](../_static/dl_r2_cv_20261104/protocol.json) y [ejecución](../_static/dl_r2_cv_20261104/execution_summary.json) registran 60 trayectorias, 720 modelos mensuales en total y 1.440 ajustes. El ajuste final con desarrollo 2023–2024 y el test reservado siguen pendientes. Esta ejecución no completa por sí sola todos los requisitos de la guía, incluida la cobertura bibliográfica aplicada.

## 1.1. Definición del problema

### 1.1.1. Espacios de entrada y salida, función y pérdida

**Retornos diarios y reloj de información**

Sea $P_d>0$ el cierre del día UTC $d$. El retorno de modelado es **diario**:

$$
r_d=\log\left(\frac{P_d}{P_{d-1}}\right).
$$

El pronóstico del día $t$ se emite **después del cierre del día $t-1$ y antes de observar el cierre del día $t$**. En ese momento solo están disponibles OHLCV y retornos de días con índice menor o igual a $t-1$. Un OHLCV diario no está disponible al comenzar ese mismo día.

**Entradas: 7, 14, 21 y 28 días de historia**

El usuario confirmó que los tamaños representan **días**, no siete o veintiocho minutos. Se consideran $D\in\{7,14,21,28\}$ observaciones diarias consecutivas:

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

Cualquier transformación o variable derivada debe calcularse causalmente. El contrato de entrada conserva las cinco covariables OHLCV; el EDA descriptivo no autoriza cambiar ese conjunto mirando el test. El primer ciclo registró transformaciones fijas log(O/H/L/C), log(1+V), escalas de entrada aprendidas solo con los días admisibles de entrenamiento y escala positiva del objetivo obtenida del mismo entrenamiento. El ciclo CV compara ese preprocesamiento absoluto con `relative_window`: logaritmos de OHLC menos el logaritmo del último cierre de la propia ventana, conocido en d−1; volumen conserva log(1+V). Las escalas de la variante relativa se ajustan solo a las posiciones de ventanas de entrenamiento. Se conservan cinco canales OHLCV, D7, n7 y h1, sin añadir rezagos explícitos del objetivo. Los detalles de ambas variantes figuran en §4.2.

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

Escalado y cualquier transformación aprendida se ajustan solo con el entrenamiento permitido. V1 aplicó cortes internos y un ajuste nuevo con 2023, sin actualizar pesos durante 2024. El ciclo CV utiliza el pasado permitido de 2023 y, progresivamente, de 2024. Cada mes selecciona duración con validación interna reciente y realiza un reajuste nuevo con la cohorte mensual admisible. Los pesos quedan fijos dentro del mes, mientras cada pronóstico diario usa solo OHLCV de días anteriores. La política vigente excluye los días incompletos y sus ventanas afectadas, sin imputación. Las métricas propias se expresan en la escala original del objetivo y las tres redes implementan salida Softplus no negativa.

### 1.1.2. Tipo de problema admitido

**Tipo admitido número 4: regresión temporal; ruta 6.4 de la guía**, con salida continua y aprendizaje supervisado. La entrada definida es una secuencia de covariables OHLCV; no incluye rezagos explícitos de la volatilidad objetivo. El problema sigue siendo un pronóstico temporal y exige disponibilidad causal, partición cronológica con embargo y validación de origen móvil. La clasificación depende del contrato de entrada y no de usar LSTM, TCN o atención. Si se incorporasen rezagos del objetivo como entrada, correspondería revisar la clasificación como series de tiempo, tipo 3, y registrar el cambio antes de ejecutarlo.

### 1.1.3. Pregunta de investigación e hipótesis operativa

**Pregunta de investigación:** ¿pueden tres modelos básicos de Deep Learning, cuya selección debe acreditarse en la literatura y usando 7, 14, 21 o 28 días de información OHLCV, pronosticar el siguiente valor de la volatilidad histórica de siete retornos diarios con menor RMSE que persistencia bajo evaluación cronológica con embargo?

**Hipótesis operativa de diseño:** al menos una configuración seleccionada exclusivamente con desarrollo alcanzará una reducción relativa de RMSE de al menos el 5 % frente a persistencia, sobre las mismas fechas elegibles del periodo de evaluación. La primera comparación de desarrollo no apoya esta hipótesis: en las 184 fechas de julio–diciembre de 2024 las tres redes tienen mayor RMSE que persistencia. Los contrastes de mejora superior al 5 % se reportan en §5.4 con su alcance condicional; no constituyen confirmación en test. LSTM es líder descriptivo entre redes en ese tramo, mientras MLP conserva la identidad de referencia elegida con enero–junio.

### 1.1.4. Métricas, margen práctico y unidad de independencia

Para cada modelo $m$, con historia $D$ y configuración fijadas, se distinguen diferencia absoluta y mejora relativa; el subíndice $m$ abrevia esa configuración y los resultados de distintas ventanas se mantienen separados:

$$
\Delta_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m,
\qquad
g_m=\frac{\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m}
{\mathrm{RMSE}_{\mathrm{pers}}},
\quad \mathrm{RMSE}_{\mathrm{pers}}>0.
$$

El criterio operativo aceptado es **$g_m\ge0{,}05$**, es decir, al menos un **5 % de reducción relativa de RMSE frente a persistencia**, en las mismas fechas y con el mismo objetivo. Es un margen de comparación del estudio; no representa una utilidad financiera calibrada ni una garantía de rentabilidad. La diferencia estadística y la relevancia operativa se reportan por separado para el primer ciclo de desarrollo y deberán conservar esa distinción en la futura evaluación final. Si la RMSE de persistencia es cero, la mejora relativa queda no definida y se reportará; no se introduce un epsilon.

Alcanzar el 5 % con una estimación puntual y demostrar una mejora de ese tamaño son afirmaciones diferentes. Para sostener la segunda se necesitará incertidumbre temporal compatible con superar el margen, además de las reglas de multiplicidad y selección fijadas antes del test. Este porcentaje se refiere únicamente a la comparación con persistencia: no determina automáticamente el margen entre dos redes, una banda de equivalencia ni el umbral futuro de un modelo avanzado frente al baseline DL.

El margen relativo quedó fijado antes del primer experimento DL. Para dimensionarlo se registró la conversión $\delta_{\mathrm{plan}}=0{,}05\,\mathrm{RMSE}_{\mathrm{pers,desarrollo}}$. Con persistencia en julio–diciembre de 2024, el piloto obtuvo $\delta_{\mathrm{plan}}=0{,}00024112394$; esta referencia de calibración y sus supuestos constan en el [informe del piloto](../_static/dl_benchmark_20261104/pilot_summary.json). Es planificación para detectar frente a cero un efecto de tamaño 5 %, una pregunta distinta de demostrar que se supera el margen del 5 %. El máximo requerido fue R=1.212 y el ciclo usó el límite previamente registrado de R=30; se declara la insuficiencia para ese objetivo de planificación. La conversión no es un umbral universal para otros periodos.

| Elemento | Decisión y alcance |
|---|---|
| Objetivo principal | Desviación poblacional de siete retornos diarios; próximo valor de la medida móvil |
| Horizonte | Un día |
| Historias de entrada | Candidatas $D=7,14,21,28$ días OHLCV; D=7 seleccionada como historia común del primer ciclo |
| Métrica primaria | RMSE en la escala original del objetivo |
| Pérdida implementada | MSE; escala positiva del objetivo ajustada solo con entrenamiento e inversión para evaluar |
| Métricas complementarias | MAE, MSE, MAPE, R² y MASE, con cobertura y casos no definidos |
| Margen operativo relativo | $\delta_{\mathrm{rel}}=0{,}05$ frente a persistencia de las mismas fechas |
| Nivel de significación del plan | $\alpha=0{,}05$; V1: Holm en nueve hipótesis por semillas, seis DM y quince diagnósticos. CV: tres contrastes pareados de RMSE entre las variantes, con Holm; no repite los diagnósticos V1 |
| Partición | Historia inicial 2023; validación 2024; test 01/01/2025–31/08/2026 |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad de independencia entre mercados | Una sola serie BTCUSDT de Binance Spot; $N_{\mathrm{series}}=1$, sin réplicas independientes de otros mercados |
| Unidad inferencial dentro de la serie | Semillas condicionadas al calendario: 30 por red en V1 y diez trayectorias por red/variante en CV. El bootstrap L14 y sensibilidad 7/28 corresponde solo a V1; ni semillas ni bloques representan mercados independientes |
| Referencias de contexto | Regresión lineal regularizada con las mismas covariables, exigida por la ruta 6.4; persistencia adicional para el margen del 5 % |
| Selección de modelos | MLP, LSTM y CNN1D poco profunda ejecutadas en desarrollo; configuración congelada por fase en §4. La acreditación bibliográfica de §3 continúa parcial |

Para $N$ predicciones evaluables:

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_t(Y_t-\widehat Y_t)^2},
\qquad
\mathrm{MAE}=\frac1N\sum_t|Y_t-\widehat Y_t|.
$$

**MAPE:** se calculará únicamente sobre etiquetas estrictamente positivas, informando cantidad de ceros, número de casos incluidos y cobertura respecto al total. No se añade epsilon ni se eliminan casos sin declararlo. Si no hay etiquetas positivas, la métrica se marca no definida. La posible sensibilidad a valores positivos pequeños se discutirá junto a RMSE y MAE.

**MASE:** se fija un rezago de referencia de **un día de calendario**, coherente con persistencia diaria y sin imponer una estacionalidad no demostrada. Para el entrenamiento permitido en el origen $t$:

$$
a_{\mathrm{train}(t),1}=
\frac{1}{|\mathcal J_t|}\sum_{j\in\mathcal J_t}|Y_j-Y_{j-1}|,
\qquad
\mathrm{MASE}=\frac1N\sum_t\frac{|Y_t-\widehat Y_t|}{a_{\mathrm{train}(t),1}}.
$$

$\mathcal J_t$ contiene pares de etiquetas válidas separados por un día de calendario y completamente observados en ese entrenamiento. El denominador es común a los modelos en el mismo origen. Si vale cero o faltan pares, MASE queda no definida y se informa su cobertura. No se comprimen huecos ni se sustituye esta media de errores escalados por dividir la MAE global entre un promedio distinto de denominadores.

**Reporte exigido por cada tamaño de entrada**

La futura evaluación de test deberá reportar por separado las historias de 7, 14, 21 y 28 días que se autoricen en su protocolo, con **MAPE, MAE, RMSE y MSE**, media y desviación entre ejecuciones, y MASE con el rezago diario de §1.1.4. El primer ciclo ya calculó estas métricas de desarrollo para el objetivo n=7: buscó las cuatro historias con el mismo presupuesto y fijó D=7 para piloto y comparación con 30 semillas nuevas por red. Esto no completa una comparación final de test por historia ni entrena las sensibilidades del objetivo n=14/21/28. La configuración principal mantiene un horizonte de un día; no se inventan salidas adicionales.

El **BDS de V1** se ejecutó como diagnóstico complementario de los residuos del primer horizonte en desarrollo, conforme al material específico recibido. Se usó la semilla designada 1001, dimensiones 2/3 y radio 1,5 veces la desviación muestral del tramo, con parámetros, tamaño y límites en §5.4.10; no se trató un fold de una sola observación como una serie de residuos. BDS sobre retornos crudos del EDA no sustituye estos errores predictivos ni constituye un diagnóstico del test reservado.

La política de MAPE ya queda definida en §1.1.4: etiquetas positivas, sin epsilon y con cobertura y ceros informados. Las tablas de desempeño propio de desarrollo ya están en §5.1 y se distinguen tanto del test pendiente como de los valores publicados por otros autores.

**Estado de cierre**

La definición documental principal queda fijada en $n=7$, $h=1$, cuatro tamaños de entrada, RMSE y margen relativo del 5 %. Procedencia, reloj de información, política vigente de faltantes y reglas de métricas están descritos. La [lista de decisiones y pendientes](protocolo_pendiente.md) distingue esos acuerdos, la ejecución propia de desarrollo, la selección bibliográfica aún parcial y los trabajos posteriores. El uso previo del mismo conjunto exacto de archivos en la literatura no se ha acreditado; la pertinencia temática de un artículo no se presenta como equivalencia de dataset.

V1 conserva sus tablas, piloto y benchmark. El ciclo CV añade entrenamiento mensual real y métricas agrupadas sobre las 366 fechas de 2024; su comparación con V1 utiliza aparte las mismas 184 fechas de julio–diciembre, con 30 semillas antiguas frente a diez nuevas y alcance descriptivo. Los hiperparámetros V1 ya se eligieron con enero–junio de 2024: esta CV es exploratoria, sin selección anidada ni potencia 0,80 acreditada para sus diez semillas. El diagnóstico Ridge permanece separado de las redes. Ningún resultado de desarrollo se presenta como evaluación del test reservado ni como cumplimiento total de la guía.

## 1.2. Justificación del Dataset

### 1.2.1. Relevancia científica y aplicada

Se propone evaluar si el historial de covariables OHLCV diarias de Bitcoin contiene información útil para pronosticar el siguiente valor de su volatilidad histórica móvil. Los retornos logarítmicos de cierre se utilizan para construir el objetivo; las entradas definidas son las covariables OHLCV. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema {cite:p}`huang2024`. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. La revisión documental y las 35 referencias vigentes se desarrollan en el capítulo 3 y en [Referencias](referencias.md).

**Decisiones confirmadas:** BTCUSDT, modelado diario desde velas de un minuto, entradas de 7/14/21/28 días, objetivo principal de volatilidad de **siete retornos diarios**, horizonte **de un día** y RMSE como métrica primaria. Las volatilidades de 14/21/28 retornos se conservan como sensibilidad del objetivo. La búsqueda propia del primer ciclo eligió $D=7$ como historia común para piloto y comparación; las otras historias conservan sus resultados de búsqueda. Las sensibilidades del objetivo n=14/21/28 siguen sin entrenarse con las redes.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

### 1.2.2. Disponibilidad pública, licencia y procedencia

**Fuente seleccionada:** Binance Spot, par BTCUSDT, velas de un minuto adquiridas directamente de los archivos públicos oficiales. La adquisición, verificación y auditoría se describen en §2.3 del [EDA](02_eda.md). Se verificaron 75 ZIP oficiales para el periodo 01/01/2023–30/10/2026 inclusive UTC: 2.014.480 filas. El manifiesto contiene los SHA-256 y la auditoría de timestamps. El EDA de calidad, precios, retornos y cuatro definiciones de volatilidad de desarrollo está ejecutado y documentado en la sección 2. La definición principal ya es $n=7$; sus muestras de desarrollo y su uso para entrenamiento están materializados en el [registro del primer ciclo](../_static/dl_benchmark_20261104/data_audit.json). Las muestras supervisadas finales de test permanecen pendientes.

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

**Uso previo del mismo dataset:** No verificado; el antecedente temático no demuestra igualdad de dataset.

## 1.3. Descripción de la técnica

### 1.3.1. Tamaño, estructura y almacenamiento

La cuadrícula de desarrollo 2023–2024 contiene 731 días UTC y 730 días OHLCV completos; el 24/03/2023 se mantiene como posición ausente. Son recuentos de días, diferentes de las filas de un minuto y de las muestras supervisadas.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Tamaño | 75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas brutas; ventanas de desarrollo materializadas, ventanas finales de test pendientes |
| Estructura de modelado | OHLCV diario; entradas $(N,D,5)$ para $D=7,14,21,28$; salida escalar de volatilidad de un día adelante |
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

Las ventanas y las volatilidades móviles comparten información. Una fecha, una inicialización y un bloque temporal representan fuentes diferentes de variabilidad. Las ventanas solapadas y las semillas sobre un mismo mercado no se contarán como mercados independientes. El protocolo del primer ciclo registró bloques temporales de 14 días y sensibilidad de 7/28, coherentes con etiquetas σ7 solapadas y acompañados de sus límites. Este remuestreo evalúa modelos guardados sobre fechas de desarrollo; no vuelve a entrenarlos ni remuestrea semillas. Un único activo limita la generalización a otros mercados.

Así se declara la dependencia conocida a priori, pero no se presenta como resuelto el requisito de una unidad inferencial independiente: ningún número de filas, folds o semillas convierte esta única serie en varios mercados. La inferencia temporal del primer ciclo conserva los supuestos de bloques y estabilidad local de su protocolo; la evaluación final requerirá justificar de nuevo su diseño, sin dar por resuelta la independencia.

**Separación temporal y prevención de fuga**

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- En cada origen, excluir del ajuste toda etiqueta que todavía no esté disponible; reservar el test para la evaluación posterior y no utilizarlo para selección.
- Ajustar escalamiento, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos; mantener la exclusión vigente sin imputación.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $D$, $n$ y la disponibilidad de etiquetas.
- Congelar y verificar el test por hash antes de evaluación definitiva; no usarlo para elegir arquitectura, hiperparámetros o preprocesamiento.

**Calendario proporcionado**

La captura aclaratoria suministrada por el usuario especifica este calendario, que reemplaza la referencia histórica menos precisa «selección 2023–2024; evaluación 2025–2026»:

| Etapa | Periodo | Uso |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Construir las primeras muestras elegibles de entrenamiento |
| Validación forward chaining | Objetivos del 01/01/2024 al 31/12/2024 | Entrenar con el pasado y acumular errores de validación para comparar configuraciones |
| Ajuste final | Datos elegibles de 2023–2024 | Reajustar la configuración seleccionada |
| Test externo | 01/01/2025–31/08/2026 | Evaluar sin seleccionar configuraciones a partir del test |

Se reportarán **2025 y 2026 por separado**; el segundo año solo cubre hasta el 31 de agosto, no el año completo.

En el primer ciclo ejecutado, 2024 se separó en **enero–junio para búsqueda** y **julio–diciembre para piloto y comparación con semillas nuevas**. Ambos tramos ya estaban disponibles en el EDA; el segundo también sirvió para dimensionar repeticiones. Sus resultados son desarrollo/calibración expuesto, no una evaluación independiente final. El ajuste con desarrollo 2023–2024 para test que aparece en la tabla todavía no se ha realizado.

**Regla de disponibilidad para la preparación posterior:** cada muestra se asigna al periodo de su día de resultado $t$. La entrada termina en $t-1$ y el objetivo se conoce al cerrar $t$. Una etiqueta solo puede utilizarse para ajustar un modelo cuando su instante de disponibilidad es anterior o igual al origen que se pretende evaluar. Si necesita retornos de 2025, no pertenece al ajuste con desarrollo 2023–2024. Una etiqueta del 31/12/2024, ya observable al comienzo del 01/01/2025, se juzga por esa disponibilidad y no se excluye solo por la escritura $\sigma_{t+1}$.

La historia previa al inicio de validación o test puede formar parte de sus entradas cuando ya sea observable. El primer ciclo DL registró una política **estática**: tras elegir duración con un corte interno cronológico, cada corrida se ajustó de nuevo con las muestras admisibles de 2023 y mantuvo pesos y escalas fijos durante 2024. No hubo reajustes diarios ni uso de etiquetas de 2024 para actualizar esos pesos. El manifiesto histórico documentaba índices de preparación. En el ciclo CV posterior, los índices de Filipe sí se utilizan para entrenar una red por comienzo mensual, con pasado permitido de 2023 y de 2024; se conservan sus pesos durante el mes y se predice cada día a h=1. La actualización mensual es una decisión explícita de este segundo procedimiento, no un reajuste por cada origen diario. La política durante el test futuro sigue pendiente y no hereda automáticamente esta decisión ni la política estática de Ridge.

La preparación corregida explicita el embargo de la ruta 6.4: para $X_t=(z_{t-D},\ldots,z_{t-1})$, el máximo rezago es $D$ días. Se omiten los días de resultado $t-D,\ldots,t-1$ del ajuste y se admiten candidatos $u\le t-D-1$, sujetos además a disponibilidad y calidad. Los 5.856 registros de partición —366 por cada una de las 16 configuraciones— aplican esa regla. El embargo no elimina toda la dependencia entre etiquetas móviles cuando $n>D+1$, que deberá conservarse en la inferencia.

Los intervalos se expresan en días UTC. El calendario contiene 365 días iniciales de 2023, 366 de validación de 2024 y **608 de test** hasta agosto de 2026; son días de calendario, no recuentos definitivos de muestras elegibles. Los huecos y la historia requerida reducen esa elegibilidad. Los índices, transformaciones y configuraciones de desarrollo se conservaron por corrida; las muestras supervisadas y el protocolo definitivo de test aún deben congelarse antes de esa evaluación. Septiembre–octubre de 2026 se adquirieron, pero no se asignan automáticamente a otra etapa ni cambian el calendario confirmado.

La evaluación prevista es un backtest histórico documentado después del periodo de mercado considerado. La reserva organiza el uso posterior de los datos; no se presenta como un pronóstico que se hubiera emitido en tiempo real durante 2025–2026.

### 1.3.5. Versión y hash del dataset

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Versión / fecha de extracción | Instantánea verificada el 02/11/2026; versión identificada por el manifiesto y SHA-256 individuales |
| Hash de datos | SHA-256 de los 75 ZIP contrastados con checksums oficiales; manifiesto enlazado en §2.3.2. Derivados y etiquetas de desarrollo documentados en §2; configuraciones e índices de desarrollo conservados en §4; manifiestos supervisados finales de test pendientes |

El SHA-256 del [manifiesto de adquisición](../_static/data/binance_spot_BTCUSDT_1m_20230101_20261030.json) es `af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2`; cada ZIP conserva además su huella individual.

La identidad de los brutos de test queda fijada en un [manifiesto de reserva de 20 ZIP mensuales](../_static/data/test_raw_reserve_20250101_20260831.json), construido únicamente a partir de los metadatos de adquisición existentes, sin abrir esos ZIP en esta revisión. Su SHA-256 es `d2eb4e84fff85ee957e6b02890cead3cec37fd593ec2a8db487dbe137d2d3a10`. Esta reserva de archivos y calendario no equivale a congelar las muestras supervisadas, máscaras, transformaciones o procedimiento del test, que siguen pendientes. Las configuraciones y los índices del ciclo de desarrollo del 04/11/2026 se conservan por separado.
