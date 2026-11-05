# 1. Definición del Problema y Dataset

**Correspondencia con los requisitos de las páginas 2–3**

La siguiente tabla permite localizar las respuestas solicitadas en la guía. La formulación incorpora las decisiones aceptadas para la validación forward chaining de Filipe, diferenciando la selección con desarrollo de 2024 de la evaluación final reservada, ejecutada después de congelar configuraciones.

| Requisito del entregable | Respuesta y ubicación |
|---|---|
| Espacios de entrada y salida, función y pérdida | §1.1.1: ventana OHLCV diaria, salida no negativa, predictor supervisado y MSE |
| Tipo de problema admitido | §1.1.2: tipo 4, regresión temporal; ruta 6.4 de la guía, con entrada OHLCV sin rezagos explícitos de volatilidad |
| Pregunta de investigación e hipótesis operativa | §1.1.3: comparar el error de pronóstico con persistencia |
| Métricas primaria y secundarias | §1.1.4: RMSE principal; MAE, MSE, R², MAPE y MASE complementarias, con sus condiciones explícitas |
| Margen práctico e independencia | §1.1.4: mejora relativa mínima del 5 % frente a persistencia; inferencia entre semillas condicionada al calendario; bootstrap temporal de fechas ejecutado, condicionado a los modelos guardados |
| Relevancia científica o aplicada | §1.2.1: pronóstico de incertidumbre de Bitcoin y antecedente científico |
| Disponibilidad, licencia y procedencia | §1.2.2 y §1.3.2: Binance Spot BTCUSDT, archivos oficiales, términos y reglas de agregación diaria documentados |
| Uso previo del dataset en literatura | §1.2.3: no verificado; el artículo citado es antecedente del problema |
| Tamaño, dimensionalidad y almacenamiento | §1.3.1: 75 ZIP, 2.014.480 filas CSV de 12 columnas y esquema de entrada definido; muestras de desarrollo y test supervisado materializadas, con elegibilidad y hashes |
| Tipo de datos y unidad de observación | §1.3.2: vela de un minuto en los brutos, día UTC en los derivados y origen temporal en las muestras supervisadas |
| Etiquetado y ruido esperado | §1.3.3, con fórmula en §1.1.1: cálculo automático del siguiente valor de volatilidad histórica móvil y fuentes de ruido |
| Dependencia entre observaciones | §1.3.4, con unidad declarada en §1.1.4: solapamiento temporal, partición cronológica y prevención de fuga |
| Versión y hash del dataset | §1.3.5: instantánea del 02/11/2026, manifiesto y SHA-256 de los 75 ZIP; derivados e índices de desarrollo disponibles; manifiesto supervisado de test y congelación final disponibles |

**Alcance de esta actualización:** se comparan MLP, LSTM y CNN1D con las cuatro historias D=7/14/21/28, objetivo fijo n=7 y horizonte h=1. La CV principal usa doce cortes mensuales de Filipe y diez semillas por red/historia: 120 trayectorias y 43.920 predicciones, incluidas treinta trayectorias D7 reutilizadas. Tras congelar decisiones se ajustaron 120 modelos finales y se evaluó el periodo 2025–agosto de 2026. El [método vigente](../_static/filipe_multihistory_20261104/method.json), la [congelación](../_static/filipe_multihistory_20261104/freeze.json) y los capítulos 4–5 conservan evidencia y cronología. La revisión posterior del capítulo 3 incorpora Miura et al. (2019), que acredita comparadores MLP y CNN independientes para volatilidad de Bitcoin. La dimensionalidad de aquella CNN y las configuraciones exactas siguen sin acreditarse como réplicas de las redes propias; una fuente no demuestra recurrencia amplia.

## 1.1. Definición del problema

### 1.1.1. Espacios de entrada y salida, función y pérdida

**Retornos diarios y reloj de información**

Sea $P_d>0$ el cierre del día UTC $d$. El retorno de modelado es **diario**:

$$
r_d=\log\left(\frac{P_d}{P_{d-1}}\right).
$$

El pronóstico del día $t$ se emite **después del cierre del día $t-1$ y antes de observar el cierre del día $t$**. En ese momento solo están disponibles OHLCV y retornos de días con índice menor o igual a $t-1$. Un OHLCV diario no está disponible al comenzar ese mismo día.

**Historias de entrada evaluadas**

El usuario confirmó que los tamaños representan **días**, no minutos. La comparación principal de las tres redes utiliza **D∈{7,14,21,28}** observaciones diarias consecutivas, manteniendo el mismo objetivo n=7 y horizonte h=1. D7 se reutiliza de la CV relativa registrada; D14/21/28 se entrenaron bajo la misma regla de Filipe en una ampliación identificada por separado:

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

Cualquier transformación o variable derivada debe calcularse causalmente. El contrato conserva cinco covariables OHLCV. La representación principal `relative_window` usa logaritmos de OHLC menos el logaritmo del último cierre de la propia ventana, conocido en d−1; volumen conserva log(1+V). Escalas de entrada y media positiva de Y se estiman únicamente con entrenamiento permitido. El control `absolute_log` pertenece al ciclo inicial D7; no se presenta como ejecutado en las otras historias. Se mantienen n=7, h=1 y ausencia de rezagos explícitos de volatilidad.

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

**Hipótesis operativa de diseño:** al menos una configuración elegida exclusivamente con desarrollo alcanzará una reducción relativa de RMSE de al menos el 5 % frente a persistencia, sobre las mismas fechas elegibles. El baseline neuronal cerrado antes de test es **MLP con D=7**. El cumplimiento puntual del margen y la evidencia estadística se distinguen en §5; el orden de test no reabre la selección. Los resultados iniciales D7 y su desglose semestral conservan su propio alcance.

### 1.1.4. Métricas, margen práctico y unidad de independencia

Para cada modelo $m$, con historia $D$ y configuración fijadas, se distinguen diferencia absoluta y mejora relativa; el subíndice $m$ abrevia esa configuración y los resultados de distintas ventanas se mantienen separados:

$$
\Delta_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m,
\qquad
g_m=\frac{\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m}
{\mathrm{RMSE}_{\mathrm{pers}}},
\quad \mathrm{RMSE}_{\mathrm{pers}}>0.
$$

El criterio operativo **del estudio ya ejecutado** es **$g_m\ge0{,}05$**, es decir, al menos un **5 % de reducción relativa de RMSE frente a persistencia**, en las mismas fechas y con el mismo objetivo. Es un margen de comparación del estudio; no representa una utilidad financiera calibrada ni una garantía de rentabilidad. La diferencia estadística y la relevancia operativa se reportan por separado para la CV de desarrollo y para la evaluación final reservada. Si la RMSE de persistencia es cero, la mejora relativa queda no definida y se reportará; no se introduce un epsilon.

Alcanzar el 5 % con una estimación puntual y demostrar una mejora de ese tamaño son afirmaciones diferentes. Para sostener la segunda se necesitará incertidumbre temporal compatible con superar el margen, además de las reglas de multiplicidad y selección fijadas antes del test. En ese protocolo el porcentaje se refiere únicamente a persistencia: no fijaba un margen entre dos redes, una banda de equivalencia ni el criterio de una propuesta avanzada. La definición académica futura que sigue se registra después de observar este test y no se atribuye retroactivamente al estudio completado.

El margen relativo original del 5 % corresponde solo a persistencia. Para planificar repeticiones puede expresarse como $\delta_{\mathrm{plan}}=0{,}05\,\mathrm{RMSE}_{\mathrm{pers,desarrollo}}$, fijando cohorte y supuestos. Detectar frente a cero un efecto de ese tamaño es distinto de demostrar una mejora superior al 5 %. Las diez semillas históricas fueron un presupuesto exploratorio sin piloto previo de aquella CV; la [sensibilidad original](../_static/filipe_statistics_20261104/design_sensitivity.csv) es retrospectiva. El [nuevo protocolo piloto](../_static/e1_filipe_pilot_20261105/protocol.json), registrado antes de sus cinco semillas nuevas por modelo e historia, adopta ahora un margen absoluto entre redes de **0,00023944847859645605**, con doce contrastes y α=0,05/12. Dimensiona futuras inicializaciones bajo el calendario 2024 conocido; no acredita potencia lograda ni reescribe el diseño histórico.

**Criterio académico para una propuesta futura.** Se define como convención operativa una reducción relativa de RMSE **superior al 5 % frente a cada una de dos referencias: MLP/D7 y persistencia**, sobre las mismas fechas de una nueva evaluación independiente. Es una decisión de protocolo transparente, coherente con el criterio previo contra persistencia y examinada mediante sensibilidad; no se presenta como tolerancia financiera validada ni como una condición anterior al test ya observado. El promedio por sí solo no acredita el cumplimiento: se deben declarar incertidumbre y las dos comparaciones. El {download}`criterio académico y su alcance <../_static/e1_audit_improvements_20261104/planning/academic_margin_README.md>` conserva fecha, referencias y regla.

Se mantienen los [36 escenarios de sensibilidad](../_static/e1_audit_improvements_20261104/planning/advanced_margin_scenarios.csv) de 1/2,5/5/10 % para detectar diferencias frente a cero. Se añaden [27 escenarios de planificación conjunta](../_static/e1_audit_improvements_20261104/planning/academic_joint_power_scenarios.csv) para superar la frontera práctica del 5 %: un mismo promedio avanzado hipotético, 7,5/10/15 % menor que persistencia, tres dispersiones futuras y tres correlaciones pareadas con el baseline. Cada contraste unilateral planifica α=0,025 y potencia 0,90; bajo los supuestos, la cota de la unión proporciona potencia conjunta al menos 0,80 sin suponer independencia entre ambos contrastes. Se impone un mínimo de cinco ejecuciones; usar pruebas de rangos exige además su propio mínimo y dimensionamiento. Son escenarios, no potencia alcanzada: el modelo avanzado requiere su propio piloto y otra evaluación. El piloto nuevo entre baselines de §4.1 no responde por sí solo esa pregunta. No se ha entrenado aquí un modelo avanzado.

| Elemento | Decisión y alcance |
|---|---|
| Objetivo principal | Desviación poblacional de siete retornos diarios; próximo valor de la medida móvil |
| Horizonte | Un día |
| Historias de entrada | D=7/14/21/28 días OHLCV evaluadas, con n=7 fijo |
| Métrica primaria | RMSE en la escala original del objetivo |
| Pérdida implementada | MSE; escala positiva del objetivo ajustada solo con entrenamiento e inversión para evaluar |
| Métricas complementarias | MAE, MSE, MAPE, R² y MASE, con cobertura y casos no definidos |
| Margen operativo relativo | $\delta_{\mathrm{rel}}=0{,}05$ frente a persistencia de las mismas fechas |
| Nivel de significación del plan | $\alpha=0{,}05$; familias declaradas por fase: el ciclo inicial D7 conserva Holm-12/DM-6/diagnósticos-15; BDS mensual se ajusta en sus familias de CV y test |
| Partición | Historia inicial 2023; validación 2024; test 01/01/2025–31/08/2026 |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad de independencia entre mercados | Una sola serie BTCUSDT de Binance Spot; $N_{\mathrm{series}}=1$, sin réplicas independientes de otros mercados |
| Unidad inferencial dentro de la serie | Diez trayectorias por red/variante condicionadas al calendario; los folds mensuales y las semillas no representan mercados independientes |
| Referencias de contexto | Regresión lineal regularizada con las mismas covariables, exigida por la ruta 6.4; persistencia adicional para el margen del 5 % |
| Selección de modelos | MLP, LSTM y CNN1D poco profunda ejecutadas en desarrollo; configuración congelada por fase en §4. §3 acredita respaldo aplicado de las familias MLP y CNN; la Conv1D y las recetas ejecutadas son adaptaciones, no réplicas documentadas |

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

Las historias **D=7/14/21/28** tienen tablas separadas de **MAPE, MAE, RMSE y MSE**, con media y DE entre las diez semillas. R², MASE, cobertura e IC complementan ese reporte. Los archivos [CV por fold y semilla](../_static/filipe_multihistory_20261104/analysis_cv/fold_metrics.csv) y [test por bloque y semilla](../_static/filipe_multihistory_20261104/test/analysis/fold_metrics.csv) permiten reconstruirlo. n=14/21/28 no se entrenó: esas sensibilidades cambiarían el objetivo, no solo la entrada.

El **BDS de residuos de h=1 se calculó por bloque y semilla** para las tres redes y cuatro entradas, con dimensiones 2/3 y radio 1,5 DE muestral. Se reportan valores sin ajustar y Holm en cada familia declarada, estados no definidos y media/DE descriptivas de p-valores. El promedio no es un contraste combinado ni una probabilidad de independencia. Las muestras mensuales cortas, etiquetas solapadas y cambios de régimen limitan la referencia asintótica.

**Estado de cierre**

La definición principal queda fijada en **n=7, h=1 y D=7/14/21/28**, RMSE y margen operativo del 5 % frente a persistencia. La [congelación](../_static/filipe_multihistory_20261104/freeze.json) registra la selección de desarrollo y la política de pesos finales fijos. La prueba se evaluó después de firmar los modelos; sus veinte meses se usan para reportar errores, no para seleccionar o reajustar.

La CV mensual y el control absoluto inicial D7 conservan su cronología. Las decisiones usaron desarrollo expuesto y no constituyen selección anidada ni potencia 0,80 acreditada. El diagnóstico Ridge permanece separado. La [matriz de cierre](protocolo_pendiente.md) distingue el nuevo piloto entre baselines, su dimensionamiento condicional y los límites de recurrencia/adaptación. El criterio avanzado requiere una futura evaluación independiente y su propia dispersión. La equivalencia y su margen son condicionales a esa afirmación, no otro requisito obligatorio pendiente.

## 1.2. Justificación del Dataset

### 1.2.1. Relevancia científica y aplicada

Se propone evaluar si el historial de covariables OHLCV diarias de Bitcoin contiene información útil para pronosticar el siguiente valor de su volatilidad histórica móvil. Los retornos logarítmicos de cierre se utilizan para construir el objetivo; las entradas definidas son las covariables OHLCV. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema {cite:p}`huang2024`. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. La revisión documental y las 39 referencias vigentes se desarrollan en el capítulo 3 y en [Referencias](referencias.md). El catálogo distingue seis textos completos directamente sobre Bitcoin de las fuentes metodológicas transferidas desde otros mercados.

**Decisiones confirmadas:** BTCUSDT, agregación diaria UTC de velas de un minuto, volatilidad poblacional de siete retornos diarios, horizonte de un día, entradas **D=7/14/21/28** y RMSE primaria. Las volatilidades n=14/21/28 fueron sensibilidades de EDA y no son objetivos entrenados del benchmark.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

### 1.2.2. Disponibilidad pública, licencia y procedencia

**Fuente seleccionada:** Binance Spot, par BTCUSDT, velas de un minuto adquiridas directamente de los archivos públicos oficiales. La adquisición, verificación y auditoría se describen en §2.3 del [EDA](02_eda.md). Se verificaron 75 ZIP oficiales para el periodo 01/01/2023–30/10/2026 inclusive UTC: 2.014.480 filas. El manifiesto contiene los SHA-256 y la auditoría de timestamps. El EDA de calidad, precios, retornos y cuatro definiciones de volatilidad de desarrollo está ejecutado y documentado en la sección 2. La definición principal ya es $n=7$; sus muestras de desarrollo y su uso para entrenamiento están materializados en los [folds de CV](../_static/dl_r2_cv_20261104/folds.json). Las muestras supervisadas finales se conservan en el [manifiesto de cohorte](../_static/filipe_multihistory_20261104/test/cohort_manifest.json).

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

**Uso previo y comparabilidad:** la revisión ahora identifica explícitamente los datos y objetivos de siete fuentes en la [matriz de comparabilidad](../_static/e1_audit_improvements_20261104/literature/dataset_comparability.csv). No se acreditó todavía que una publicación use la misma versión histórica Binance Spot/σ7. La documentación concreta de diferencias sustituye la afirmación genérica «No verificado», sin declarar equivalencia entre datasets.

| Fuente | Datos y mercado documentados | Objetivo/horizonte y límite para comparar |
|---|---|---|
| S1 · Akbiyik et al. | Bitcoin cada 15 minutos y Twitter | RV del próximo día agregada de 96 retornos intradiarios; cambia definición, escala, información y periodo. |
| S2 · Zeri | Binance Futures BTCUSDT perpetuo, vía CryptoLake | Log RV futura de 30 minutos; futuros no es Spot y su microestructura no coincide con OHLCV diario. |
| S3 · Herremans/Low | CryptoQuant y whale-alert/Twitter | Picos del día siguiente según autores; fórmula exacta pendiente de artículo completo y covariables diferentes. |
| S4 · Huang et al. | Proveedor y datos no extraídos de texto completo en esta revisión | Definición/horizonte pendientes; el antecedente temático no acredita igualdad numérica. |
| S5 · Gao et al. | Yahoo Finance BTC-USD diario | Ventana de 30 días sin centrar y desplazamiento de siete días; cambia proveedor, par, normalización y horizonte. |
| S6 · Wang et al. | OHLCV de Bitcoin diario; proveedor insuficientemente especificado | RV1D sin fórmula suficientemente recuperada; no se equipara a σ7. |
| S7 · Alakhouch | BTC/USD diario de Investing.com/Kaggle, según manuscrito/código | Código de volatilidad futura de 30 días anualizada; no es la etiqueta móvil σ7/h1 del proyecto. |

La recuperación también encontró software que declara Binance Spot/BTCUSDT a un minuto, pero con dispersión futura de diez minutos y sin un artículo verificable: aporta una pista de procedencia, no uso científico acreditado del dataset propio. Las cifras de las fuentes siguen siendo contexto; no se ordenan junto a las métricas propias como si procedieran del mismo experimento. El {download}`informe de recuperación <../_static/e1_audit_improvements_20261104/literature/report.md>` conserva ubicaciones, calidad y limitaciones. La igualdad de la versión exacta sigue abierta; la **matriz de diferencias ya está documentada**.

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

Las ventanas y las volatilidades móviles comparten información. Una fecha, una inicialización y un bloque temporal representan fuentes diferentes de variabilidad. Las semillas y los doce folds mensuales sobre un mercado no se contarán como mercados independientes. Los IC por semillas condicionan al calendario observado; el bootstrap temporal inicial D7 se conserva como análisis exploratorio condicionado a sus modelos guardados, sin extrapolarlo a las otras entradas. Un único activo limita la generalización a otros mercados.

Así se declara la dependencia conocida, sin presentar como resuelto el requisito de independencia: ningún número de filas, folds o semillas convierte una serie en varios mercados. La inferencia temporal debe justificar sus supuestos, longitud de bloque y estabilidad en el procedimiento que se evalúa. El suplemento final conserva longitud principal 14 y sensibilidades 7/28, con alcance post-hoc y condicionado a sus modelos guardados.

**Separación temporal y prevención de fuga**

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- En cada origen, excluir del ajuste toda etiqueta que todavía no esté disponible; reservar el test para la evaluación posterior y no utilizarlo para selección.
- Ajustar escalamiento, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos; mantener la exclusión vigente sin imputación.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $D$, $n$ y la disponibilidad de etiquetas.
- Congelar y verificar el test por hash antes de evaluación definitiva; no usarlo para elegir arquitectura, hiperparámetros o preprocesamiento.

**Calendario proporcionado**

El calendario confirmado del proyecto es el siguiente:

| Etapa | Periodo | Uso |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Construir las primeras muestras elegibles de entrenamiento |
| Validación forward chaining | Objetivos del 01/01/2024 al 31/12/2024 | Entrenar con el pasado y acumular errores de validación para comparar configuraciones |
| Ajuste final | Datos elegibles de 2023–2024 | Reajustar la configuración seleccionada |
| Test externo | 01/01/2025–31/08/2026 | Evaluar sin seleccionar configuraciones a partir del test |

Se reportan **2025 y enero–agosto de 2026 por separado**, con 365 y 243 fechas, respectivamente. La [tabla anual](../_static/e1_audit_improvements_20261104/descriptive/annual_metrics_summary.csv) calcula cada métrica sobre sus fechas y luego resume semillas; 2026 no representa el año completo ni se obtiene la RMSE global promediando dos RMSE anuales.

Los 366 días de 2024 se evalúan mediante doce bloques mensuales de forward chaining, utilizando solo el pasado permitido para cada ajuste. Ese año ya estuvo disponible en el EDA y en la elección de configuraciones, por lo que sus resultados son desarrollo exploratorio. El ajuste final se realizó sobre el desarrollo permitido de 2023–2024, excluyendo los últimos D resultados antes de enero de 2025 por embargo; las épocas se fijaron con la validación interna de diciembre de 2024.

**Regla de disponibilidad para la preparación posterior:** cada muestra se asigna al periodo de su día de resultado $t$. La entrada termina en $t-1$ y el objetivo se conoce al cerrar $t$. Una etiqueta solo puede utilizarse para ajustar un modelo cuando su instante de disponibilidad es anterior o igual al origen que se pretende evaluar. Si necesita retornos de 2025, no pertenece al ajuste con desarrollo 2023–2024. Una etiqueta del 31/12/2024, ya observable al comienzo del 01/01/2025, se juzga por esa disponibilidad y no se excluye solo por la escritura $\sigma_{t+1}$.

La historia previa al inicio de validación o test puede formar parte de las entradas cuando ya sea observable. Los índices de Filipe se utilizan para entrenar una red al comienzo de cada mes, con pasado permitido de 2023 y de 2024; los pesos se conservan durante el mes y se predice cada día a h=1. Son doce procedimientos mensuales por trayectoria, no un reajuste por cada origen diario. La política final cerrada no actualiza pesos ni escaladores durante test: los veinte meses de 2025–agosto de 2026 son periodos de reporte de los mismos modelos finales.

La preparación corregida explicita el embargo de la ruta 6.4: para $X_t=(z_{t-D},\ldots,z_{t-1})$, el máximo rezago es $D$ días. Se omiten los días de resultado $t-D,\ldots,t-1$ del ajuste y se admiten candidatos $u\le t-D-1$, sujetos además a disponibilidad y calidad. Los 5.856 registros de partición —366 por cada una de las 16 configuraciones— aplican esa regla. El embargo no elimina toda la dependencia entre etiquetas móviles cuando $n>D+1$, que deberá conservarse en la inferencia.

Los intervalos se expresan en días UTC. El calendario contiene 365 días iniciales de 2023, 366 de validación de 2024 y **608 de test** hasta agosto de 2026; son días de calendario, no recuentos definitivos de muestras elegibles. Los huecos y la historia requerida reducen esa elegibilidad. Los índices, transformaciones y configuraciones se conservaron por corrida. La congelación anterior a la apertura de test, la auditoría de los veinte ZIP y el manifiesto de cohorte permiten verificar su ejecución final. Septiembre–octubre de 2026 se adquirieron, pero no se asignan automáticamente a otra etapa ni cambian el calendario confirmado.

La evaluación prevista es un backtest histórico documentado después del periodo de mercado considerado. La reserva organiza el uso posterior de los datos; no se presenta como un pronóstico que se hubiera emitido en tiempo real durante 2025–2026.

### 1.3.5. Versión y hash del dataset

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Versión / fecha de extracción | Instantánea verificada el 02/11/2026; versión identificada por el manifiesto y SHA-256 individuales |
| Hash de datos | SHA-256 de los 75 ZIP contrastados con checksums oficiales; manifiesto enlazado en §2.3.2. Derivados y etiquetas de desarrollo documentados en §2; configuraciones e índices conservados en §4; congelación y manifiestos finales de test disponibles |

El SHA-256 del [manifiesto de adquisición](../_static/data/binance_spot_BTCUSDT_1m_20230101_20261030.json) es `af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2`; cada ZIP conserva además su huella individual.

La identidad de los brutos de test queda fijada en el [manifiesto de reserva de 20 ZIP mensuales](../_static/data/test_raw_reserve_20250101_20260831.json), construido inicialmente solo desde metadatos. El ciclo final los abrió después de cerrar configuraciones y pesos: la [auditoría](../_static/filipe_multihistory_20261104/test/raw_test_audit.json), la [cohorte supervisada](../_static/filipe_multihistory_20261104/test/cohort_manifest.json) y la [congelación](../_static/filipe_multihistory_20261104/freeze.json) conservan hashes, fechas, reglas y exclusiones. La reserva inicial no se sobrescribe ni se confunde con esta evaluación posterior.
