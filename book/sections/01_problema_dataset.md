# 1. Definición del Problema y Dataset

**Correspondencia con los requisitos de las páginas 2–3**

La siguiente tabla permite localizar las respuestas solicitadas en la guía. La formulación incorpora las decisiones ya aceptadas por el usuario. Los campos que requieren ejecución futura se distinguen de la evidencia documental y del EDA disponible.

| Requisito del entregable | Respuesta y ubicación |
|---|---|
| Espacios de entrada y salida, función y pérdida | §1.3: ventana OHLCV diaria, salida no negativa, predictor supervisado y MSE |
| Tipo de problema admitido | §1.2: regresión temporal, ruta 6.4 de la guía; entrada OHLCV sin rezagos explícitos de volatilidad |
| Pregunta de investigación e hipótesis operativa | §1.4: comparar el error de pronóstico con persistencia |
| Métricas primaria y secundarias | §1.4: RMSE, MAE, MAPE y MSE; MASE adicional con condiciones explícitas |
| Margen práctico e independencia | §1.4: mejora relativa mínima del 5 % frente a persistencia; unidades temporales dependientes y bloques por justificar antes de la inferencia |
| Relevancia científica o aplicada | §1.1: pronóstico de incertidumbre de Bitcoin y antecedente científico |
| Disponibilidad, licencia y procedencia | §1.5: Binance Spot BTCUSDT, archivos oficiales, términos y reglas de agregación diaria documentados |
| Uso previo del dataset en literatura | §1.5: no verificado; el artículo citado es antecedente del problema |
| Tamaño, dimensionalidad y almacenamiento | §1.5: 75 ZIP, 2.014.480 filas CSV de 12 columnas y esquema de entrada definido; ventanas supervisadas pendientes |
| Tipo de datos y unidad de observación | §1.2 y §1.5: precios temporales y origen de pronóstico |
| Etiquetado y ruido esperado | §1.3 y §1.5: cálculo automático de dispersión futura y fuentes de ruido |
| Dependencia entre observaciones | §1.4–§1.6: solapamiento temporal, partición cronológica y prevención de fuga |
| Versión y hash del dataset | §1.5: instantánea del 02/11/2026, manifiesto y SHA-256 de los 75 ZIP verificados; etiquetas y derivados supervisados pendientes |

**Alcance de esta actualización:** la fuente, adquisición y EDA están documentados; la definición principal queda fijada en volatilidad de siete retornos diarios y horizonte de un día. Por instrucción del usuario, este trabajo completa la parte documental de E1 y reserva entrenamiento, ajuste de hiperparámetros, piloto y evaluación experimental para E2. El PDF de la guía incluye esos bloques dentro de E1 en sus secciones 4–5: se registra la diferencia de alcance sin atribuir al documento una excepción que no contiene. Las muestras y decisiones de ejecución pendientes se trasladan al plan posterior; no se presentan resultados propios de modelos.

## 1.1. Motivación y alcance

Se propone evaluar si el historial de rendimientos logarítmicos derivados del cierre de Bitcoin contiene información útil para pronosticar su volatilidad futura. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema {cite:p}`huang2024`. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. La revisión documental y las treinta referencias se desarrollan en el capítulo 3 y en [Referencias](referencias.md).

**Decisiones confirmadas:** BTCUSDT, modelado diario desde velas de un minuto, entradas de 7/14/21/28 días, objetivo principal de volatilidad de **siete retornos diarios**, horizonte **de un día** y RMSE como métrica primaria. Las volatilidades de 14/21/28 retornos se conservan como sensibilidad del objetivo. No se fija una longitud de entrada ganadora ni se elige $D=7$ como única configuración principal.

## 1.2. Tipo de problema y reloj de información

**Tipo según la definición de la guía: regresión temporal, ruta 6.4**, con salida continua y aprendizaje supervisado. La entrada definida es una secuencia de covariables OHLCV; no incluye rezagos explícitos de la volatilidad objetivo. El problema sigue siendo un pronóstico temporal y exige disponibilidad causal, partición cronológica y validación de origen móvil. Añadir rezagos de volatilidad sería un cambio del contrato de entrada que debe justificarse y registrarse, no una consecuencia automática de esta clasificación.

Se distingue **adquisición a un minuto** de **modelado diario**. Los archivos oficiales BTCUSDT se agregan en días UTC completos. Cada observación diaria contiene apertura, máximo, mínimo, cierre y volumen (OHLCV); un día incompleto se marca no elegible en los derivados, sin modificar los ZIP originales.

Sea $P_d>0$ el cierre del día UTC $d$. El retorno de modelado es **diario**:

$$
r_d=\log\left(\frac{P_d}{P_{d-1}}\right).
$$

El pronóstico del día $t$ se emite **después del cierre del día $t-1$ y antes de observar el cierre del día $t$**. En ese momento solo están disponibles OHLCV y retornos de días con índice menor o igual a $t-1$. Un OHLCV diario no está disponible al comenzar ese mismo día.

La configuración aportada establece modelado diario, cierre UTC y horizonte principal de un día. Las noticias solo se incorporarían con una fuente verificable y fechas de publicación anteriores al pronóstico; no se han adquirido ni forman parte de la entrada implementada de este avance.

## 1.3. Formulación matemática y decisiones del proyecto

### 1.3.1. Entradas: 7, 14, 21 y 28 días de historia

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

Cualquier transformación o variable derivada debe calcularse causalmente. El EDA no ha seleccionado todavía un subconjunto final de covariables ni su escalado; las comparaciones deberán mantener el mismo pipeline entre modelos.

### 1.3.2. Volatilidad histórica y objetivo de un día adelante

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

### 1.3.3. Pérdida y referencia contextual

Se fija MSE como pérdida del diseño supervisado posterior, sobre los orígenes elegibles de entrenamiento:

$$
\mathcal L(\theta)=\frac1{|\mathcal T_{\mathrm{train}}|}
\sum_{t\in\mathcal T_{\mathrm{train}}}(Y_t-f_\theta(X_t))^2.
$$

La referencia de persistencia para el mismo horizonte será $\widehat Y_t^{\mathrm{pers}}=\sigma_t$, calculada únicamente con el pasado. No ocupa uno de los tres puestos de modelos de Deep Learning. Se comparará en los mismos orígenes elegibles para cada tamaño de entrada. Para $n=7$ resume ocho cierres; una entrada de siete días OHLCV no incorpora explícitamente esa misma estadística. La comparación entre redes conservará entradas idénticas, mientras persistencia se identificará como referencia contextual con información histórica disponible.

Escalado, imputación permitida y cualquier transformación se ajustarán solo con el entrenamiento de cada fold. Las métricas finales se expresarán en la escala original del objetivo y la salida no negativa se garantizará al implementar las redes.

## 1.4. Pregunta, hipótesis y evaluación a priori

**Pregunta de investigación:** ¿pueden tres modelos simples de Deep Learning, seleccionados desde la literatura y usando 7, 14, 21 o 28 días de información OHLCV, pronosticar el siguiente valor de la volatilidad histórica de siete retornos diarios con menor RMSE que persistencia bajo evaluación cronológica?

Para cada modelo $m$ se distinguen diferencia absoluta y mejora relativa:

$$
\Delta_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m,
\qquad
g_m=\frac{\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_m}
{\mathrm{RMSE}_{\mathrm{pers}}},
\quad \mathrm{RMSE}_{\mathrm{pers}}>0.
$$

El criterio operativo aceptado es **$g_m\ge0{,}05$**, es decir, al menos un **5 % de reducción relativa de RMSE frente a persistencia**, en las mismas fechas y con el mismo objetivo. Es un margen de comparación del estudio; no representa una utilidad financiera calibrada ni una garantía de rentabilidad. La diferencia estadística y la relevancia operativa se evaluarán por separado en la fase experimental posterior. Si la RMSE de persistencia es cero, la mejora relativa queda no definida y se reportará; no se introduce un epsilon.

El margen relativo queda fijado antes de ejecutar esa fase. Para dimensionar un experimento podrá convertirse a unidades absolutas con una referencia de desarrollo identificada: $\delta_{\mathrm{plan}}=0{,}05\,\mathrm{RMSE}_{\mathrm{pers,desarrollo}}$. Aquí no se calcula un valor a partir de corridas archivadas ni se presenta esa conversión como un umbral universal para otros periodos.

| Elemento | Decisión y alcance |
|---|---|
| Objetivo principal | Desviación poblacional de siete retornos diarios; próximo valor de la medida móvil |
| Horizonte | Un día |
| Historias de entrada | $D=7,14,21,28$ días OHLCV; sin longitud ganadora fijada |
| Métrica primaria | RMSE en la escala original del objetivo |
| Pérdida prevista | MSE |
| Métricas complementarias | MAE, MSE, MAPE, R² y MASE, con cobertura y casos no definidos |
| Margen operativo relativo | $\delta_{\mathrm{rel}}=0{,}05$ frente a persistencia de las mismas fechas |
| Nivel de significación del plan | $\alpha=0{,}05$; familia y multiplicidad se registrarán antes de la inferencia |
| Partición | Historia inicial 2023; validación 2024; test 01/01/2025–31/08/2026 |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad inferencial | Unidades o bloques temporales justificados; longitud y supuestos pendientes del protocolo posterior |
| Referencia contextual | Persistencia de volatilidad histórica ya observable |
| Selección de modelos | Registro documental, evidencia y estado de cierre en §3 |

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

Las ventanas y las volatilidades móviles comparten información. Una fecha, una inicialización y un bloque temporal representan fuentes diferentes de variabilidad. Las ventanas solapadas y las semillas sobre un mismo mercado no se contarán como mercados independientes. La longitud de bloque y su sensibilidad se justificarán antes de los contrastes posteriores; no se importan longitudes obtenidas de experimentos fuera del alcance de E1. Un único activo limita la generalización a otros mercados.

## 1.5. Dataset: pertinencia y ficha técnica

**Fuente seleccionada:** Binance Spot, par BTCUSDT, velas de un minuto adquiridas directamente de los archivos públicos oficiales. La adquisición, verificación y auditoría se describen en §2.3 del [EDA](02_eda.md). Se verificaron 75 ZIP oficiales para el periodo 01/01/2023–30/10/2026 inclusive UTC: 2.014.480 filas. El manifiesto contiene los SHA-256 y la auditoría de timestamps. El EDA de calidad, precios, retornos y cuatro definiciones de volatilidad de desarrollo está ejecutado y documentado en la sección 2. La definición principal ya es $n=7$; la materialización definitiva de muestras y su uso para entrenamiento pertenecen al plan posterior.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Nombre y proveedor | Binance Spot; archivos públicos oficiales de velas de un minuto |
| Activo y par | Bitcoin, cotizado en USDT; par BTCUSDT |
| Disponibilidad pública | Archivos oficiales accesibles y descargados desde data.binance.vision; la API devuelve HTTP 451 por ubicación y no se utiliza |
| Licencia y redistribución | Binance Vision Dataset Terms v1.0, actualizados el 26/08/2026: CC BY-NC-SA 4.0 salvo indicación específica, con condiciones adicionales; ver §2.3.4 |
| Procedencia | Archivos oficiales de klines del mercado Spot BTCUSDT; esquema temporal y agregación diaria descritos debajo |
| Uso previo del mismo dataset | No verificado; el antecedente temático no demuestra igualdad de dataset |
| Periodo temporal y zona horaria | Descarga: 01/01/2023–30/10/2026 inclusive UTC; timestamps de apertura en milisegundos antes de 2025 y microsegundos desde 2025. Calendario experimental en §1.6 |
| Frecuencia | Brutos de un minuto; observaciones de modelado y retornos del objetivo diarios, cierre UTC |
| Tamaño | 75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas brutas; cierres válidos y ventanas finales pendientes |
| Estructura de modelado | OHLCV diario; entradas $(N,D,5)$ para $D=7,14,21,28$; salida escalar de volatilidad de un día adelante |
| Formato de almacenamiento | CSV oficiales de 12 columnas dentro de ZIP; timestamps normalizados solo para auditar cobertura, brutos intactos |
| Etiquetas | Cálculo automático de $Y_t$ según fórmula aprobada; sin anotación manual |
| Dependencia | Temporal y por ventanas solapadas; diagnósticos de desarrollo documentados en §2, sin presuponer independencia |
| Versión / fecha de extracción | Instantánea verificada el 02/11/2026; versión identificada por el manifiesto y SHA-256 individuales |
| Hash de datos | SHA-256 de los 75 ZIP contrastados con checksums oficiales; manifiesto enlazado en §2.3.2. Derivados y etiquetas de desarrollo documentados en §2; configuraciones y manifiestos supervisados finales pendientes |

**Generación y agregación.** Los archivos de Binance Public Data contienen klines del mercado Spot BTCUSDT: cada registro incluye hora de apertura, OHLC, volumen, hora de cierre y campos complementarios de volumen/operaciones. La documentación de referencia es [Binance Public Data](https://github.com/binance/binance-public-data) y el esquema de [klines de Binance Spot](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data). El par pertenece a un mercado y proveedor concretos; no es un índice agregado de precios de múltiples exchanges.

La agregación usada por el proyecto agrupa por día UTC: toma la primera apertura, el máximo de los máximos, el mínimo de los mínimos, el último cierre y la suma de volúmenes. El cierre diario $P_d$ corresponde al último minuto del día, disponible al concluirlo. Se respeta el cambio documentado de unidades de timestamps: milisegundos antes de 2025 y microsegundos desde 2025; las unidades se normalizan para procesar tiempos, conservando intactos los ZIP.

La regla actual exige 1.440 minutos elegibles para acreditar un OHLCV diario completo. Los **80 minutos ausentes del 24/03/2023** y la fila con cierre temporal no estándar se conservan en el bruto, y el derivado diario afectado queda ausente. Se mantienen las posiciones de calendario y se excluyen las ventanas que necesiten esos datos, sin interpolar, rellenar volumen con cero ni fabricar etiquetas. Esta es la política vigente de E1; cualquier recuperación desde otra granularidad o cambio de imputación requiere documentar por separado fuente, validación y versión antes de usarla.

Con $T$ días OHLCV consecutivos válidos y sin pérdidas por partición, construir entradas de $D$ días y una etiqueta con $n$ retornos diarios permite como máximo $T-\max(D,n+1)$ orígenes de un día adelante, con historia suficiente para la referencia contextual. Es una relación teórica, **no el tamaño real**; huecos, filtros y separación entre particiones pueden reducirlo.

### 1.5.1. Etiquetado, ruido y controles de calidad

El objetivo se derivará de precios observados, por lo que puede reflejar errores del proveedor, precios anómalos, cambios de liquidez y ruido de microestructura. El tamaño finito de la ventana y la definición elegida introducen incertidumbre como proxy. No se ha medido todavía la magnitud de estos efectos.

El EDA ya comprobó orden temporal, duplicados, cierres no positivos, huecos, frecuencia y significado del timestamp sobre desarrollo; sus hallazgos y límites figuran en §2. En la construcción definitiva se conservarán esos controles. Un retorno calculado a través de un día ausente no se tratará como retorno diario consecutivo; tampoco se confunde un salto entre minutos separados con un retorno regular de un minuto.

### 1.5.2. Separación temporal y prevención de fuga

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- En cada origen, excluir del ajuste toda etiqueta que todavía no esté disponible; reservar el test para la evaluación posterior y no utilizarlo para selección.
- Ajustar escalamiento, imputación, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $D$, $n$ y la disponibilidad de etiquetas.
- Congelar y verificar el test por hash antes de evaluación definitiva; no usarlo para elegir arquitectura, hiperparámetros o preprocesamiento.

## 1.6. Calendario proporcionado

La captura aclaratoria suministrada por el usuario especifica este calendario, que reemplaza la referencia histórica menos precisa «selección 2023–2024; evaluación 2025–2026»:

| Etapa | Periodo | Uso |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Construir las primeras muestras elegibles de entrenamiento |
| Validación forward chaining | Objetivos del 01/01/2024 al 31/12/2024 | Entrenar con el pasado y acumular errores de validación para comparar configuraciones |
| Ajuste final | Datos elegibles de 2023–2024 | Reajustar la configuración seleccionada |
| Test externo | 01/01/2025–31/08/2026 | Evaluar sin seleccionar configuraciones a partir del test |

Se reportarán **2025 y 2026 por separado**; el segundo año solo cubre hasta el 31 de agosto, no el año completo.

**Regla de disponibilidad para la preparación posterior:** cada muestra se asigna al periodo de su día de resultado $t$. La entrada termina en $t-1$ y el objetivo se conoce al cerrar $t$. Una etiqueta solo puede utilizarse para ajustar un modelo cuando su instante de disponibilidad es anterior o igual al origen que se pretende evaluar. Si necesita retornos de 2025, no pertenece al ajuste con desarrollo 2023–2024. Una etiqueta del 31/12/2024, ya observable al comienzo del 01/01/2025, se juzga por esa disponibilidad y no se excluye solo por la escritura $\sigma_{t+1}$.

La historia previa al inicio de validación o test puede formar parte de sus entradas cuando ya sea observable. La frecuencia de reajuste y la política durante test se definirán en E2: **no se fija en esta revisión un reajuste mensual, diario o estático**. Los folds existentes documentan orígenes y controles de disponibilidad; no son corridas de entrenamiento ni obligan a reentrenar en cada origen.

Los intervalos se expresan en días UTC. El calendario contiene 365 días iniciales de 2023, 366 de validación de 2024 y **608 de test** hasta agosto de 2026; son días de calendario, no recuentos definitivos de muestras elegibles. Los huecos y la historia requerida reducen esa elegibilidad. Los manifiestos supervisados y la congelación experimental se materializarán antes de la ejecución posterior. Septiembre–octubre de 2026 se adquirieron, pero no se asignan automáticamente a otra etapa ni cambian el calendario confirmado.

## 1.7. Estado de cierre

La definición documental principal queda fijada en $n=7$, $h=1$, cuatro tamaños de entrada, RMSE y margen relativo del 5 %. Procedencia, reloj de información, política vigente de faltantes y reglas de métricas están descritos. La [lista de decisiones y pendientes](protocolo_pendiente.md) distinguirá esos acuerdos de la selección bibliográfica y de los trabajos posteriores que requieren ejecución. El uso previo del mismo conjunto exacto de archivos en la literatura no se ha acreditado; la pertinencia temática de un artículo no se presenta como equivalencia de dataset.

Por alcance de esta intervención, las tablas de resultados propios, el piloto y el entrenamiento corresponden a E2. Esta organización sigue la instrucción del usuario y no modifica lo que exige literalmente el PDF de la guía.

### 1.7.1. Reporte exigido por cada tamaño de entrada

El plan posterior de E2 contempla reportar resultados de test para 7, 14, 21 y 28 días de entrada, idealmente por fold y con media y desviación estándar: **MAPE, MAE, RMSE y MSE**, por horizonte y agregados cuando corresponda. Se añadirá MASE como referencia escalada pertinente al pronóstico temporal, con el rezago diario fijado en §1.4. La configuración principal tiene un horizonte de un día; no se inventan salidas adicionales.

El **BDS** se prevé como diagnóstico complementario de los residuos del primer horizonte, conforme al material específico recibido. Requiere tramos con suficientes residuos fuera de entrenamiento; no se calculará sobre un fold de una sola observación. Se documentarán dimensiones, radios, tamaño y supuestos en la fase posterior. BDS sobre retornos crudos del EDA no sustituye un diagnóstico de errores predictivos.

La política de MAPE ya queda definida en §1.4: etiquetas positivas, sin epsilon y con cobertura y ceros informados. Las tablas de desempeño propio permanecen reservadas para la ejecución posterior; esa ausencia no se cubre con valores publicados por otros autores.
