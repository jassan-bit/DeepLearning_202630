# 8 Bloque D Verificación estadística y evaluación ampliada

**Estado: diseño documental, sin protocolo congelado ni experimentos confirmatorios del candidato.** Este capítulo aplica la Sección 8 de la guía al pronóstico BTCUSDT, D7×5 OHLCV → σ₇ poblacional, sin anualización, horizonte h=1 y ruta temporal F2a. La [implementación de la sección 6](06_modelo_original.md) y la [demostración y verificación de la sección 7](07_diseno_riguroso.md) no establecen superioridad predictiva. El [borrador de protocolo](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_d/protocolo_borrador.md) reúne las decisiones que deben cerrarse antes de ejecutar la campaña definitiva; no es un preregistro ni asigna cómputo.

## 8.1 Protocolo experimental pre-registrado

La métrica primaria heredada es la **media de las RMSE individuales**, en unidades originales de σ₇. Para un modelo j y una ejecución r sobre las mismas fechas admisibles T,

$$
\operatorname{RMSE}_{j,r}=\sqrt{\frac1{|T|}\sum_{t\in T}(y_t-\widehat y_{j,r,t})^2},\qquad
M_j=\frac1{R_j}\sum_{r=1}^{R_j}\operatorname{RMSE}_{j,r}.
$$

No se sustituye esta media por la RMSE de un ensemble ni por la raíz de la MSE agregada entre semillas. La ganancia frente al rival j se orienta como Δj = Mj − Moriginal: positivo favorece al candidato. La hipótesis confirmatoria propuesta es

$$
H_{0,j}:\Delta_j\le\delta,\qquad H_{1,j}:\Delta_j>\delta,
\qquad\delta=0{,}00025772842664981106.
$$

El parámetro poblacional y su estimador deben vincularse a la unidad de incertidumbre elegida; una media observada no basta para rechazar la nula. Antes del registro faltan estas decisiones concretas:

| Registro exigido | Qué debe quedar cerrado | Estado E3 |
|---|---|---|
| Estándar y procedencia | Compromiso teórico, fuente del mecanismo, modificación y relación con el paper guía. | Estándar descrito; registro formal `experiments/estandar.md`, revisión R1 y defensa pendientes. |
| Hipótesis y métricas | Primaria, secundarias, δ, brecha δg, hipótesis D1–D5 y análisis exploratorios separados. | RMSE y δ heredados; catálogo secundario y reglas finales pendientes. |
| Datos | Fuentes, versiones, hashes, exposición previa, cortes, purga/embargo y tres conjuntos adicionales. | Sin selección de los tres conjuntos ni cierre de confirmación. |
| Ejecuciones | Semillas, R, determinismo, versiones, fallos y política común de entrenamiento/calibración. | Lista final y dimensionamiento pendientes. |
| Inferencia | Familia completa, tres procedimientos válidos por rival, supuestos, estadísticos, IC y Holm. | Batería por formular y validar. |
| Recursos | Espacios, intentos, épocas, parada, tiempo, hardware y costes por modelo. | Presupuesto E3 pendiente. |
| Diseño | Variantes, esfuerzo acumulado Nd y separación desarrollo/confirmación. | Cuatro variantes funcionales de software; Nd de mercado igual a cero, según la bitácora. |
| Congelamiento | Documento final, fecha y tag Git reales anteriores a los experimentos definitivos. | No existe cierre confirmatorio de este borrador. |

La [sección 7](07_diseno_riguroso.md) conserva los criterios de desarrollo como propuestas. Toda desviación posterior se registrará con fecha, motivo e impacto; los análisis no previstos serán exploratorios y no podrán sostener por sí solos R3. No se retrofecha un registro después de observar resultados.

## 8.2 Equidad de la comparación

El candidato y cada rival deben compartir particiones, variables, preprocesamiento ajustado sólo en entrenamiento, aumentos admisibles, fuente de pesos si se usan, métrica y política de evaluación. Las transformaciones, épocas y calibradores se fijan antes de consultar la confirmación. La [sección 4](04_correcciones_e2.md) conserva los controles y límites de E2; su existencia no certifica el nuevo pipeline del candidato.

El presupuesto se declarará en **configuraciones y tiempo**, o se justificará otra definición previa. Deben fijarse intentos, espacio relativo, épocas máximas, parada temprana, semillas HPO y tratamiento de fallos. Si el candidato consume más por intento, se igualará el tiempo total o se reportarán ambos análisis. El esfuerzo de idear variantes se contabiliza separado del HPO.

Se reportarán parámetros totales/activos, FLOPs con su alcance y capacidad de todos los modelos. Una diferencia sustancial de tamaño exige una variante comparable o discusión explícita; las ablaciones que cambien parámetros tendrán su control de tamaño igualado conforme a la sección 7.4. Los costes heredados de GPU no se etiquetan como latencias CPU.

Las redes requieren **al menos diez semillas independientes**, o más si el dimensionamiento lo exige, con pareado entre modelos cuando exista aleatoriedad. Las seis semillas históricas de E2 no completan ese requisito. Una campaña nueva deberá conservar recetas y reglas previamente cerradas, sin reajustar rivales después de ver la comparación final.

Persistence, SeasonalNaive y ARIMA son comparadores deterministas en el contrato heredado: copiar diez veces la misma trayectoria no genera diez ajustes independientes. Se representará la variación del candidato contra esa referencia fija y la incertidumbre temporal sin inventar semillas clásicas. Hardware, precisión, tiempo, memoria y costes reales se registrarán en condiciones comparables.

## 8.3 Evaluación en conjuntos de datos adicionales

Se necesitan al menos tres conjuntos públicos de la **misma tarea de pronóstico de volatilidad** y con la misma primaria. Cada uno debe diferir de BTCUSDT en al menos dos ejes documentados, como dominio, tamaño, ruido o dimensionalidad. Se registrarán fuente, licencia, versión, hash, criterio de selección y dependencia entre series antes de observar el desempeño.

| Conjunto requerido | Fuente y diferencias documentadas | Selección y evaluación |
|---|---|---|
| Adicional 1 | Pendiente de elegir y justificar. | No ejecutada. |
| Adicional 2 | Pendiente de elegir y justificar. | No ejecutada. |
| Adicional 3 | Pendiente de elegir y justificar. | No ejecutada. |

El candidato y los **16 rivales** se evaluarán en cada conjunto, sin excepciones por resultados desfavorables, con presupuesto comparable y auditoría de fuga. Si un conjunto se usa para rediseñar el mecanismo, deja de ser evidencia independiente de esa decisión. Los márgenes y sus unidades en escalas adicionales deben justificarse antes de evaluar, sin recalibrarlos según la ventaja observada.

Se reportarán métricas, diferencias, IC y efectos por conjunto; no se promedian RMSE de escalas distintas. La consolidación prevista es Friedman con corrección de Iman–Davenport y pares con Holm, o una alternativa bayesiana con ROPE elegida previamente. No se escogerá el procedimiento más favorable a posteriori. Con sólo tres conjuntos la potencia global puede ser limitada, y series relacionadas pueden compartir dependencia; la consistencia por conjunto seguirá siendo necesaria.

## 8.4 Definición operativa de superioridad R3

La guía exige cumplir simultáneamente las condiciones siguientes. Ninguna está acreditada por las pruebas de software del candidato:

1. **Media favorable frente a cada rival.** Menor primaria media, con comparación sobre soporte y tratamiento comunes.
2. **Tres pruebas válidas y relevancia.** La batería previamente cerrada debe rechazar para cada rival, con p ajustados por Holm menores que α=0,05, orientación favorable y ganancia mayor que δ. Si se usa MCS, su nivel se declara aparte; pertenecer a un conjunto no prueba dirección de superioridad ni ganancia práctica.
3. **Efecto no despreciable e IC.** La guía cita |δCliff| ≥ 0,147 o |dCohen| ≥ 0,2 cuando proceda. Debe elegirse un estimador compatible con pareado, dependencia y unidad, con IC95 % de ganancia cuyo límite inferior sea positivo e idealmente mayor que δ. Un IC marginal no sustituye la corrección familiar ni la prueba de la frontera δ.
4. **Consistencia adicional.** Mejor media en al menos dos de los tres conjuntos y ausencia de inferioridad significativa frente a cualquier rival en cualquiera de ellos. No detectar inferioridad no demuestra equivalencia o inocuidad.
5. **Adecuación del candidato.** Superar el baseline trivial y cumplir los criterios de brecha, calibración, residuos y coherencia validación–prueba. Se adaptarán las Tablas A y B de la ruta F2a al candidato, con aceptación y límites prefijados.
6. **Conclusión robusta.** Conservar la conclusión bajo pruebas alternativas y sensibilidades predefinidas. Si cambia, se reportará como no robusta, sin elegir sólo el análisis favorable.

Para la brecha se conserva **δg=0,00025772842664981106**, registrado previamente en E2. Coincide numéricamente con δ, pero tiene otro significado: g = RMSEvalidación − RMSEtrain, calculada con riesgos comparables. Falta cerrar su transferencia al candidato, el estimando —media o mediana— y el ámbito de incertidumbre. La orientación es H0,g: θg≥δg frente a H1,g: θg<δg, con θg definido previamente como media o mediana. Una prueba sobre la media de g no acredita automáticamente una afirmación sobre su mediana. El [suplemento posterior de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html) conserva sus resultados condicionales; no se trasladan al candidato ni a otro backend sin verificación.

## 8.5 Familia de contrastes

La familia candidata conserva **m=16 rivales de E2**, todos frente al original:

| Grupo | Rivales incluidos |
|---|---|
| Top 10 | FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. |
| Referencias neuronales | MLP16, LSTM32 y GRU32. |
| Comparadores clásicos | Persistence, SeasonalNaive y ARIMA. |

La comparación completa tendrá 17 sistemas contando al candidato. El m≈12 ilustrativo de la guía no autoriza retirar rivales. Las limitaciones de memoria del dashboard tampoco reducen la familia experimental. La afirmación «supera a todos» exige la conjunción de las decisiones de los 16 contrastes; un rival no concluyente no se elimina de la tabla.

Falta cerrar el Bloque A y vincular las recetas confirmatorias con sus artefactos y hashes. Se conserva el benchmark histórico y se registrarán las ejecuciones nuevas con identidad propia; no se alteran sus resultados, recetas o semillas retrospectivamente para favorecer al candidato.

## 8.6 Reglas metodológicas para que las pruebas sean válidas

La unidad temporal no es una fila iid: las ventanas D7 y los objetivos σ₇ comparten soporte, y los residuos pueden estar autocorrelacionados y ser heterocedásticos. El remuestreo deberá mantener bloques sincronizados entre todos los modelos, fechas comunes y la estructura de semillas. Sus longitudes, réplicas, generadores y sensibilidades deben fijarse antes del cálculo final. Las semillas sobre un calendario no son mercados independientes.

Los datos de confirmación se consultarán al final, con diseño, hiperparámetros, umbrales y parada fijados en desarrollo. Las fechas históricas conocidas de E1/E2 no recuperan cegamiento al volver a puntuarlas. La purga, el embargo y la disponibilidad de covariables se derivarán del soporte real; compartir observaciones pasadas disponibles no implica por sí solo fuga, pero debe auditarse.

**Todavía no existe una batería cerrada de tres pruebas distintas admisibles de la Tabla 4 y justificadas para F2a por rival.** Las opciones siguientes se distinguen por estimando y alcance; figurar en esta lista no acredita su admisibilidad automática para la ruta temporal:

| Procedimiento candidato | Hipótesis o incertidumbre que debe justificarse | Límite para R3 |
|---|---|---|
| Bootstrap temporal pareado de la ganancia de RMSE | Contraste de Δj frente a δ, remuestreo por bloques y construcción válida bajo la nula; coherencia con la media de RMSE individuales. | Es candidato natural, no una prueba ya ejecutada del original. Cambiar bloque, B o semilla no crea otra prueba distinta. |
| t pareada de ganancias entre semillas | Media de ganancias condicionada al calendario; independencia de inicializaciones y adecuación de la distribución o aproximación. | No representa por sí sola incertidumbre temporal ni múltiples mercados. |
| Aleatorización de signos de ganancias | Simetría/intercambiabilidad bajo la nula pertinente y frontera del margen correctamente formulada. | Su presencia junto a la t y al bootstrap no acredita tres confirmaciones temporales homogéneas. |
| DM–HLN, Giacomini–White y MCS | Orden temporal, pérdidas, estacionariedad, varianza de largo plazo o condiciones del procedimiento global. | Las versiones sobre MSE diaria e igualdad contra cero no prueban automáticamente ΔRMSE>δ ni completan esa batería por sí mismas. |

La aplicación histórica de E2 conserva bootstrap de RMSE, t y signos entre semillas con sus ámbitos condicionados; sus DM/MCS posteriores tienen familias y estimandos propios. No se trasladan como tres pruebas confirmatorias del candidato. Antes de congelar deben formularse y validarse tres procedimientos coherentes con la afirmación principal o justificarse expresamente su combinación y sus límites.

La media de pérdidas cuadráticas diarias y la media de RMSE por trayectoria son cantidades diferentes. **No se inserta δ de RMSE como un desplazamiento del mismo tamaño en MSE.** DM o una tabla MCS pueden aportar información auxiliar, pero no convertir una hipótesis de igualdad de MSE en una ganancia de RMSE mayor que δ. Clark–West requiere anidamiento pertinente. Nadeau–Bengio corresponde a comparaciones de particiones solapadas que justifiquen su corrección; no se aplica genéricamente a días o semillas de este calendario.

Se verificarán los supuestos de cada procedimiento y se declararán diagnósticos no estimables. Una predicción no finita o un fallo conserva su estado: no se sustituye por cero ni se retira una fecha para mejorar el resultado. Los IC de diferencias temporales deben respetar la dependencia; equivalencia requiere TOST o ROPE con margen previo, no un p no significativo.

## 8.7 Estrategia estadística: familias, potencia y agregación

### 8.7.1 Control del error por familia

La propuesta del borrador aplica Holm a α=0,05 sobre los **16 rivales para cada procedimiento** previamente definido. Con p ordenados de una de esas familias,

$$
p_{(i)}^{\rm Holm}=\min\left\{1,\ \max_{1\le j\le i}(m-j+1)p_{(j)}\right\},\qquad m=16.
$$

Holm controla el error familiar si los p de partida son válidos. La coherencia de hipótesis y ámbitos de incertidumbre debe cerrarse antes de atribuir ese control a la afirmación conjunta. Para el rival r y las pruebas exigidas K, se usará

$$
P_r=\max_{k\in K}p_{r,k}^{\rm Holm}.
$$

Todas deben rechazar y tener dirección favorable; no se escoge el menor p. La decisión «supera a todos» requiere esa regla para **cada r=1,…,16**, además de las condiciones de 8.4. Se reportará el procedimiento que menos rechaza. Si se integra una prueba global como MCS, su regla debe quedar fijada; no se inventa un p por par a partir de pertenencia al conjunto.

Los conjuntos adicionales, las ablaciones, los subgrupos/regímenes, las sensibilidades y los diagnósticos tendrán familias secundarias separadas y enumeradas previamente. Se informarán su tamaño y todos los p crudos/ajustados. No se subdividen ni mezclan después de ver resultados para relajar el contraste principal.

### 8.7.2 Potencia y número de semillas

Se requiere un **piloto de desarrollo de al menos cinco semillas por par**, todavía no ejecutado para el candidato, que estime la dispersión de las diferencias pareadas frente a cada rival. Para clásicos deterministas se conserva su trayectoria fija y se estima la variación pertinente del candidato y del calendario; no se inventan cinco ajustes clásicos independientes.

Como orientación normal bilateral inicial, con nivel conservador α′=0,05/m y diferencia objetivo e respecto a la frontera nula,

$$
R_j\approx\frac{\left(z_{1-\alpha'/2}+z_{1-\beta}\right)^2\widehat\sigma_{D,j}^{\,2}}{e^2},
\qquad R\ge\max\left(10,\max_j\lceil R_j\rceil\right).
$$

La guía ilustra e=δ al detectar una diferencia δ frente a cero. Para la propuesta H0:Δ≤δ, la distancia es **e=Δalt−δ>0**, no δ salvo que se planifique Δalt=2δ. La alternativa y el criterio unilateral/bilateral deben fijarse antes de dimensionar. La fórmula no certifica potencia de Holm, bootstrap, una batería de tres pruebas o fechas dependientes.

El objetivo es **potencia de al menos 0,90 por contraste con la regla completa**, con potencia conjunta de superar a toda la familia reportada. Se dimensionarán tanto R como el calendario y se evaluará el procedimiento real, por ejemplo mediante simulación que conserve las dependencias, con supuestos e incertidumbre del piloto. No se multiplican potencias marginales como si los contrastes que comparten original y calendario fueran independientes.

Si diez semillas no bastan, se aumentan antes de congelar; se reportarán MDE y limitaciones si la potencia objetivo no resulta alcanzable. La falta de rechazo no demuestra equivalencia. **No se han estimado aquí σD, R definitivo, MDE ni potencia del candidato.** El piloto y las seis semillas históricos de E2 no constituyen ese dimensionamiento de toda la familia nueva.

### 8.7.3 Agregación jerárquica y múltiples conjuntos de datos

Se distinguirán serie/conjunto, origen o bloque temporal, modelo y semilla. Las observaciones correlacionadas se agrupan o modelan antes del contraste; no se suman fechas, ventanas y semillas como si todas fueran réplicas independientes. La DE entre inicializaciones y el IC temporal tendrán etiquetas diferentes.

Un modelo de efectos mixtos es una opción cuando el diseño aporte unidades repetidas adecuadas, por ejemplo Yijr = μ + αj + ui + vr + εijr para una métrica por unidad, modelo y semilla. Deben justificarse esas unidades y su covarianza temporal; no basta añadir un intercepto aleatorio para declarar independencia. No se ha ajustado ese modelo ni se lo fija automáticamente para una única serie BTC.

Además de la consolidación entre conjuntos, se reportarán por conjunto y rival la ganancia, IC, efecto y probabilidad de superioridad Pr(RMSEoriginal < RMSErival), estimada sobre unidades pareadas apropiadas y con el tratamiento de empates fijado. No se promedian métricas de escalas distintas. Las sensibilidades con una prueba alternativa, una métrica secundaria y subconjuntos razonables de semillas se definirán antes de evaluar y mostrarán si la conclusión R3 se conserva, sin elegir sólo semillas favorables.

Se incluirán alternativas paramétrica/no paramétrica pertinentes. Se justificará si Nadeau–Bengio aplica al diseño y, en ese caso, se contrastarán las decisiones con y sin la corrección; si no aplica, se documentará el motivo sin calcular una corrección inválida.

## 8.8 Robustez, sensibilidad y calidad probabilística

La Tabla 5 de la guía se adapta a la ruta temporal con el mismo procedimiento para candidato y rivales. Los estudios siguientes están **pendientes de ejecución E3**; las pruebas sintéticas y suplementos de E2 conservan su alcance propio.

| Estudio exigido | Aplicación prevista a BTC/F2a |
|---|---|
| Sensibilidad a hiperparámetros | Curvas sobre radio, penalización, temperatura, tamaño del mecanismo y learning rate en desarrollo; región de estabilidad y comparación con rivales. Fijar rangos antes de confirmar. |
| Semillas y particiones | Distribuciones con al menos diez semillas o R mayor, IC y coeficiente de variación cuando sea interpretable. ANOVA/Kruskal–Wallis sólo con réplicas y supuestos válidos: un único score por semilla no forma grupos con replicación. |
| Perturbaciones controladas | Ruido, faltantes y atípicos con al menos tres severidades y curvas de degradación. Distinguir corrupción artificial de alteraciones que preserven coherencia OHLCV y disponibilidad temporal; reglas de imputación se fijan sin prueba. |
| Tamaño y escalabilidad | Curvas con fracciones de entrenamiento —por ejemplo 10/25/50/100 %—, varias semillas y cortes cronológicos válidos. Tiempo y memoria al variar entrada o tamaño; los porcentajes son propuesta, no corridas realizadas. |
| Cambio de distribución | Periodo, fuente o mercado mantenido fuera del diseño. Registrar qué cambio se evalúa y la exposición previa; otro tramo conocido no se vuelve independiente por renombrarlo. |
| Calibración y puntuación | Calibración puntual/residuos mediante diagnósticos F2a y reglas probabilísticas pertinentes. CRPS/PIT y log-verosimilitud requieren una distribución o densidad adecuada; pinball se informa para cuantiles y niveles definidos. Un único intervalo del 90 % no identifica una distribución completa. No importar ECE de clasificación por defecto. |
| Incertidumbre | Cobertura global y por regímenes/colas prefijados, IC temporales y ancho medio frente al nominal. Declarar límites de intercambiabilidad y distinguir dispersión de semillas de un intervalo predictivo calibrado. |
| Eficiencia y Pareto | Parámetros, FLOPs con operaciones incluidas, CPU/GPU, latencias, memoria, entrenamiento y energía si se estima, con hardware/precisión/hilos/lotes. RSS del proceso no es RAM incremental del modelo ni memoria Render. |
| Errores compartidos | Subgrupos y regímenes donde el candidato falla o no supera al benchmark; pérdidas y errores compartidos, extremos genuinos y explicaciones con fidelidad/estabilidad. Las causas económicas se distinguen de etiquetas automáticas. |

La adecuación temporal incluirá Mincer–Zarnowitz, Ljung–Box, ARCH-LM y, cuando procedan, BDS/Jarque–Bera, estabilidad y brechas comparables. Se fijarán sus hipótesis, reglas y familias; un p no significativo no prueba ausencia de autocorrelación, heterocedasticidad o sobreajuste. Las métricas probabilísticas sólo se usarán cuando el producto emitido permita calcularlas con significado.

## 8.9 Reporte obligatorio de resultados

El informe final deberá incluir los productos siguientes, con versiones, fuentes, estimandos y límites. **En este capítulo no se rellenan métricas, p ni potencias con ceros o cifras ficticias.**

| Tabla o figura | Campos mínimos | Estado del candidato |
|---|---|---|
| Resumen de los 17 sistemas | Modelo/dataset, media±DE, IC bootstrap con unidad, parámetros, FLOPs, tiempo de entrenamiento y latencia. | Comparación pendiente; los parámetros del software se describen en sección 6. |
| Contrastes por rival y procedimiento | Prueba, nula, supuestos verificados, estadístico, p crudo, p Holm, efecto/IC, orientación y conclusión frente a δ. | Batería pendiente; ningún p calculado. |
| MCS con original incluido | Pérdida, nivel, pertenencia, estadístico y alcance temporal. | Pendiente; MCS histórico de E2 no incluye al candidato. |
| Cada conjunto adicional y consolidación | Fuente/versión/hash, métrica, diferencias, IC, efectos, rangos/prueba global y pares. | Tres conjuntos aún sin elegir. |
| Tablas A y B del original | Baseline trivial, ajuste, residuos, calibración, brecha δg y coherencia validación–prueba. | Pendientes; no se heredan resultados de las redes de E2. |
| Potencia | Piloto, σD, efecto alternativo, R, semillas, α, MDE, potencia por contraste y conjunta, supuestos e incertidumbre. | Piloto y dimensionamiento no ejecutados. |
| Estudios de la Tabla 5 | Curvas, distribuciones, degradación, cobertura, errores, costes y Pareto. | Pendientes de evaluación real de E3. |

La tabla de contrastes deberá dejar explícito el resultado menos favorable de cada rival y la decisión conjunta, no sólo los rechazos. Se conservarán resultados negativos, fallos y análisis no concluyentes, con discusión de muestra, ruido, capacidad, variabilidad y potencia. Un resultado honesto sin superioridad no se convierte en equivalencia ni se corrige relajando el estándar.

El {download}`borrador de protocolo <../_static/e3_bloque_d/protocolo_borrador.md>` conserva estos pendientes y las condiciones para un registro futuro. El capítulo organiza el trabajo estadístico; no declara cerrado el Bloque D ni autoriza entrenamientos o una campaña confirmatoria.
