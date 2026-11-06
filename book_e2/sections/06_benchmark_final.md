# 6. Benchmark Final

**Estado: plan de análisis propuesto, no congelado; resultados Bitcoin D7 pendientes.** Se recibieron seis capturas de §6.1–6.4, con páginas **17 y 19** visibles y otros fragmentos sin folio. No se les atribuyen páginas 18 o 20. El [registro de la guía](../_static/guia_e2_recibida.json) conserva treinta requisitos de esta sección.

El objetivo es un benchmark reproducible, justo y comparable con el [contrato de §1](01_punto_partida.md) y el [diseño de §4](04_implementacion.md). Las veinte corridas ETTh1 de [§5](05_paper_guia.md) —cinco de réplica y quince de piloto— son evidencia externa; **no son resultados del benchmark Bitcoin ni se trasladan a sus tablas**. Las métricas históricas de E1 tampoco se reetiquetan como nuevas corridas E2.

Las remisiones a §7.3, §7.4, §9.2, §11.4, §12, §13 y Apéndice A siguen pendientes de texto completo. Se detallan métodos y salidas previstos sin afirmar que todos estén implementados o que sus supuestos ya se cumplan. La reserva heredada ya conocida conserva alcance exploratorio, incluso si se registra una única fase futura de scoring E2.

## 6.1 Tabla comparativa y evidencia requerida

### 6.1.1 Quince configuraciones, sin métricas inventadas

El mínimo recibido en §3.2 es Top 10 más tres recetas heredadas, Ridge y persistencia: **quince configuraciones**. Las variantes retunadas de GRU/LSTM/TCN, si se justifican y registran conforme a §4, añadirían tres filas; no reemplazarían las anclas ni se dan por incorporadas ahora. El orden siguiente agrupa papeles y familias: no es un ranking observado.

| Modelo/configuración | Papel | RMSE, media ± DE | IC95 % de RMSE | MAE, MSE, R², MAPE y MASE |
|---|---|---|---|---|
| TSMixer | Top 10 · MLP | PENDIENTE | PENDIENTE | PENDIENTE |
| TiDE | Top 10 · MLP | PENDIENTE | PENDIENTE | PENDIENTE |
| TimeMixer | Top 10 · MLP multiescala | PENDIENTE | PENDIENTE | PENDIENTE |
| TimesNet | Top 10 · convolucional | PENDIENTE | PENDIENTE | PENDIENTE |
| PatchTST | Top 10 · atención | PENDIENTE | PENDIENTE | PENDIENTE |
| Informer | Top 10 · atención | PENDIENTE | PENDIENTE | PENDIENTE |
| Autoformer | Top 10 · atención/autocorrelación | PENDIENTE | PENDIENTE | PENDIENTE |
| iTransformer | Top 10 · atención entre variables | PENDIENTE | PENDIENTE | PENDIENTE |
| Crossformer | Top 10 · atención temporal/dimensional | PENDIENTE | PENDIENTE | PENDIENTE |
| S-Mamba | Top 10 · estados selectivos | PENDIENTE | PENDIENTE | PENDIENTE |
| GRU/D7-c03 | Ancla E1 y baseline registrado | PENDIENTE | PENDIENTE | PENDIENTE |
| LSTM/D7-c02 | Ancla E1 | PENDIENTE | PENDIENTE | PENDIENTE |
| TCN/D7-c02 | Ancla E1 | PENDIENTE | PENDIENTE | PENDIENTE |
| Ridge | Referencia no profunda | PENDIENTE | PENDIENTE | PENDIENTE |
| Persistencia σ₇(t−1) | Referencia trivial | PENDIENTE | PENDIENTE | PENDIENTE |

| Modelo/configuración | Calibración | Robustez | Tiempo de entrenamiento | Latencia por muestra | Parámetros / FLOPs |
|---|---|---|---|---|---|
| TSMixer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| TiDE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| TimeMixer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| TimesNet | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| PatchTST | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| Informer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| Autoformer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| iTransformer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| Crossformer | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| S-Mamba | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| GRU/D7-c03 | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| LSTM/D7-c02 | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| TCN/D7-c02 | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| Ridge | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| Persistencia σ₇(t−1) | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

**PENDIENTE no significa cero.** Una futura celda no aplicable tendrá motivo explícito, por ejemplo ausencia de ajuste de parámetros en persistencia; no se sustituye tiempo no medido por 0 s. Los conteos heredados de parámetros pueden usarse como referencia de arquitectura, pero la tabla final identificará su procedencia y la configuración efectivamente evaluada. La [plantilla CSV](../_static/benchmark_analysis/table_template.csv) conserva estos estados para completar los campos del Apéndice A cuando se reciba.

### 6.1.2 Métricas, cobertura e intervalos

La primaria sigue siendo la **media de las RMSE por trayectoria** sobre las mismas fechas elegibles, en escala original, como en §1. Para el modelo m y sus R semillas:

$$
q_{m,s}=\sqrt{\frac1{|\mathcal T|}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2},
\qquad \widehat\mu_m=\frac1R\sum_{s=1}^{R}q_{m,s}.
$$

Se publicarán los q por semilla, DE muestral e IC, identificando si describen aleatoriedad de entrenamiento condicionada al calendario o variabilidad temporal. No se reemplaza esta métrica por la RMSE del ensamble ni por una media no ponderada de RMSE mensuales. Persistencia y una Ridge determinista no adquieren R nuevas realizaciones independientes al repetir su misma predicción en el panel de comparación.

MAE, MSE y R² acompañarán la primaria. MAPE se calcula solo con Y positivo y reporta cobertura; MASE usa su denominador ingenuo de entrenamiento, sin comprimir huecos. Se conserva como no definido R² cuando la varianza objetivo sea cero y MASE cuando su escala sea cero. Los casos faltantes, fallos y exclusiones se informan; no se completan con el mejor resultado de otra semilla. Las comparaciones usan la cohorte común fijada antes del scoring y reportan su tamaño, sus fechas y las pérdidas de elegibilidad.

Un **IC de una métrica agregada no es un intervalo predictivo para Y**. Para pronósticos puntuales se proponen sesgo firmado, pendiente/intercepto de calibración y contraste entre promedio predicho y observado en bins fijados en desarrollo, con incertidumbre temporal; son diagnósticos de calibración del punto, no una distribución predictiva. La dispersión entre semillas no proporciona automáticamente cobertura de futuras observaciones. La calibración probabilística requiere definir y validar ese objeto. Se propone, sujeto a §7.4 y al protocolo completo, registrar intervalos predictivos construidos con residuales de desarrollo fuera de muestra y evaluar cobertura nominal/empírica, anchura e interval score. Su calibrador solo podrá usar pasado permitido. Dependencia y cambio de régimen impiden afirmar cobertura exacta del 95 % por invocar conformalización sin justificar sus condiciones. Un predictor puntual no recibe automáticamente una métrica de calibración probabilística; tampoco se le aplica ECE de clasificación como sustituto.

La robustez seguirá las curvas y áreas de degradación registradas en §4.6, con tres severidades y control limpio por perturbación. Se conservarán error absoluto, error relativo, máscaras/semillas de corrupción y cobertura; un área estable de un modelo impreciso no basta para considerarlo mejor. Los intervalos temporales se obtendrán remuestreando bloques de orígenes emparejados, con todas las trayectorias de cada origen juntas. El tamaño de bloque se cerrará sin escoger el que produzca una conclusión favorable.

### 6.1.3 Coste medido bajo condiciones comparables

Entrenamiento, inferencia, parámetros y FLOPs se medirán conforme a §4 y al futuro protocolo §11.4: mismo dispositivo, precisión, batch, threads y entradas; calentamiento, sincronización y orden de temporización intercalado. Se separarán HPO, selección de épocas, reajuste y ajuste final. La latencia por ventana batch 1 y el throughput por lote no son intercambiables.

Se registrarán método y alcance del conteo de FLOPs, MACs u operaciones no soportadas. Los parámetros registrados y activos se distinguirán si el código contiene módulos sin uso. El hardware y las versiones serán parte de cada fila, no notas omitidas al comparar CPU con GPU. Costes externos de ETTh1 o cifras editoriales no se transfieren al D7 de Bitcoin.

## 6.2 Ranking, estabilidad y eficiencia

### 6.2.1 Rangos promedio sobre comparaciones emparejadas

Dentro de cada ejecución emparejada s se ordenará q por modelo, con rango 1 para menor RMSE y rango medio en empates exactos; se preservará la precisión numérica original, sin crear empates redondeando la tabla. Se reportará

$$
\bar r_m=\frac1R\sum_{s=1}^{R}\operatorname{rank}_s(q_{m,s}),
$$

junto con la RMSE media, su intervalo, la distribución de rangos y la frecuencia de ocupar los primeros puestos. Un rango favorable sin mejora mayor que δ no acredita relevancia práctica. Si el orden por rangos medios difiere del orden por RMSE media, se publicará esa discordancia y su incertidumbre; no se cambia el estimando primario para elegir al ganador.

Los rangos por partición son otra descripción posible, con sus propias fechas y tamaños. **No se multiplican R semillas por diez meses y se tratan como R×10 datasets independientes.** Las semillas describen entrenamiento condicionado a una serie; los meses tienen dependencia temporal y entrenamientos solapados. Un ranking descriptivo no autoriza automáticamente Friedman o Nemenyi como si las particiones fueran datasets independientes.

### 6.2.2 Fronteras de Pareto

Se compararán RMSE y, por separado, parámetros, FLOPs y latencia; también se documentará el coste de entrenamiento. Para un vector de criterios a minimizar z, A domina puntualmente a B si todos sus componentes son menores o iguales y alguno es estrictamente menor. Los modelos no dominados forman la frontera para esos criterios concretos. Una frontera RMSE–latencia puede diferir de la de RMSE–parámetros.

Se distinguirá **dominancia de las estimaciones puntuales** de dominancia respaldada por incertidumbre. Se propone examinar estabilidad de pertenencia mediante remuestreo temporal emparejado de errores y bloques de temporización de coste; una frecuencia bootstrap no es por sí sola una probabilidad posterior ni una prueba simultánea de superioridad. No se colapsan precisión, coste, robustez e interpretabilidad en un índice único con pesos elegidos después de los resultados.

El trade-off se discutirá con magnitudes e intervalos: cuánta RMSE se intercambia por cuánto tiempo, memoria o complejidad, y qué ocurre con calibración, robustez y fidelidad de explicaciones. Menos parámetros no implica menor latencia; una atención visualmente atractiva no implica mayor interpretabilidad fiel. El análisis de ablación e interpretabilidad de §4 sustenta las explicaciones sobre mecanismos, sin establecer causalidad económica.

## 6.3 Inferencia estadística formal

### 6.3.1 Estimandos, unidad y familia principal propuesta

Se mantiene **α=0,05**, el baseline GRU/D7-c03 y **δ=0,00025772842664981106** en unidades de volatilidad diaria. El margen no se recalcula como el 5 % de la nueva RMSE ni se sustituye por la tolerancia MSE de la réplica ETTh1. Rechazar igualdad de errores no demuestra superar ese margen.

Como **borrador revisable antes del scoring Bitcoin**, se propone K=10 candidatos del Top 10 frente a GRU-c03 para las tres RQ de §1: **3K=30 hipótesis principales**. LSTM, TCN, Ridge y persistencia permanecen como referencias contextuales. Esta propuesta concreta no afirma que §1 o el protocolo completo ya estén congelados. Superar GRU no equivale a superar persistencia.

Con ganancias g respecto a GRU, deterioro d de esa ganancia en alta volatilidad observable y diferencia de latencia c respecto a GRU, las alternativas son:

- **RQ1:** g > δ; mejora predictiva relevante.
- **RQ2:** d > δ; erosión relevante de la ventaja en el régimen registrado.
- **RQ3:** g > δ **y** c < 0; mejora conjunta de error y coste.

RQ3 es una intersección-unión: su nula es g≤δ **o** c≥0. Se usa el máximo de los p-valores unilaterales válidos de sus dos componentes y después se incluye ese p conjunto en Holm sobre las treinta hipótesis. No se declara cumplida RQ3 porque solo un componente resulte favorable. Holm controla FWER bajo la validez de esos p-valores, sin exigir independencia entre ellos {cite:p}`e2_holm`.

Los **105 pares** posibles entre quince configuraciones, o los 45 entre diez, son familias distintas. No se adoptan automáticamente ni se mezclan con los treinta contrastes principales. Si se añade una familia exploratoria de todos los pares, se registrarán sus preguntas, ajuste y alcance; un control al 5 % por familia no se presentará como control global conjunto. Si se pretende una afirmación conjunta entre familias, deberá cerrarse una familia unificada o una asignación de α cuya suma no exceda 0,05.

### 6.3.2 Intervalos, potencia y resolución numérica

Se propone para los treinta parámetros únicos —g, d y c por candidato— construir límites **unilaterales simultáneos**, con nivel de cola conservador α/30 por límite: inferiores para g y d y superiores para c. La cobertura familiar objetivo es al menos 95 % si cada límite marginal es válido; no se etiqueta un IC95 % marginal como simultáneo. El límite inferior de g se reutiliza donde corresponda, sin inventar un segundo efecto distinto para RQ3.

La distribución nula del contraste práctico debe centrarse en la frontera δ. Una cola de bootstrap sin centrar o probar solo contra cero no responde esa pregunta. Se remuestrean los mismos bloques/fechas para ambos modelos, recalculando la RMSE de cada trayectoria antes de promediar semillas; el coste conserva su unidad de temporización separada. Los IC marginales descriptivos se publican además de los límites simultáneos que sustentan decisiones.

Para esta ampliación de familia se propone **49.999 remuestreos**, con corrección Monte Carlo de sumar uno: resolución mínima 1/50.000=0,00002 y unas 83 realizaciones esperadas en una cola α/30. Es un nuevo parámetro de diseño de §6 por ratificar, no un cambio retroactivo de los 4.999 remuestreos propuestos en §4 o utilizados en §5. Más remuestreos reducen error Monte Carlo; no crean más fechas independientes ni garantizan cobertura.

El R definitivo, efecto mínimo detectable, error familiar y capacidad de los contrastes deben justificarse con el piloto Bitcoin y §9.2. El R de E1 y las cinco semillas ETTh1 no proporcionan esa justificación. Sesenta fechas conocidas pueden dejar muy pocos bloques efectivos; se reportará esa limitación. Una ausencia de rechazo con poca capacidad no es evidencia de igualdad.

### 6.3.3 Supuestos, comparaciones pareadas y efectos

Se documentarán distribución de las diferencias emparejadas, colas/outliers, Shapiro–Wilk, dispersión e independencia pertinente. Las series de residuos y pérdidas se examinan también en su orden temporal. No se elige automáticamente t o Wilcoxon según un único p de normalidad: con pocas semillas Shapiro tiene escasa capacidad, y la no normalidad de residuos no equivale por sí sola a no normalidad de diferencias de RMSE entre semillas.

| Método | Aplicación y condición que debe justificarse |
|---|---|
| t pareado | Diferencias de métricas por semillas independientes del entrenamiento, condicionadas al calendario; normalidad de diferencias o aproximación justificada. No exige igualdad de varianzas de las métricas originales de ambos modelos |
| Wilcoxon signed-rank | Diferencias emparejadas con condiciones de simetría/interpretación de localización y tratamiento declarado de ceros y empates. No es una prueba universal libre de supuestos |
| Levene/Brown–Forsythe | Diagnóstico de dispersión cuando su unidad y dependencia permitan aplicarlo; un Levene no significativo no acredita independencia ni es requisito automático del t pareado |
| Bootstrap temporal pareado | Incertidumbre del calendario con bloques de orígenes y predicciones comunes; la longitud y los supuestos de dependencia/estabilidad deben justificarse |

Se reportará primero el efecto en unidades originales g y su reducción relativa, con denominadores declarados. Como efecto estandarizado pareado se propone d_z=media de diferencias/DE de diferencias, con IC adecuado al pareamiento; si la DE es cero queda no definido. Si se informa delta de Cliff, se especificará su estimando marginal y se conservará el pareamiento en el remuestreo: no se lo renombra como probabilidad de victoria de pares de semillas. Un efecto de rangos pareados puede complementar Wilcoxon, sin sustituir el margen δ.

Los contrastes entre semillas describen variabilidad del entrenamiento, no generalización a mercados independientes. Con cinco pares no nulos y sin empates el mínimo p bilateral exacto de Wilcoxon es 0,0625; con seis es 0,03125. Aumentar R mejora esa resolución condicional, pero no aumenta las fechas ni demuestra potencia temporal.

### 6.3.4 Comparación de diez o más modelos y solapamiento

Si el diseño admite un contraste global válido, se informará como análisis complementario: ANOVA de medidas repetidas solo si se justifican sus supuestos, incluidas unidad independiente y esfericidad/correcciones pertinentes; o Friedman sobre bloques emparejados independientes cuando su diseño lo permita {cite:p}`e2_demsar`. La familia de treinta contrastes principales planificados no se condiciona a rechazar primero Friedman: Holm se aplica a sus p-valores válidos y no necesita esa puerta de entrada. No se aplica mecánicamente a los diez meses porque compartan nombres de fold. En este proyecto con un único mercado no se reportará Friedman/Nemenyi o un diagrama de diferencia crítica como evidencia inferencial de superioridad global. Las comparaciones pareadas entre semillas conservan un alcance condicional al calendario; no convierten las semillas en datasets independientes. Esta entrega no dispone de los bloques independientes necesarios para la interpretación entre datasets de esos procedimientos.

Los post-hoc podrán usar Nemenyi cuando proceda ese diseño, o pares con Holm/Bonferroni en la familia registrada. El diagrama de diferencia crítica solo se generará con bloques y aproximación justificados. Un gráfico post-hoc Holm no se presenta como una única distancia crítica Nemenyi; los vínculos de no rechazo tampoco significan equivalencia. Si faltan bloques suficientes o una calibración fiable de la distribución nula, se reporta el ranking descriptivo con esa limitación en vez de fabricar el diagrama inferencial.

Los entrenamientos solapados y la autocorrelación afectan la varianza. La corrección Nadeau–Bengio es un ejemplo para ciertos diseños de remuestreo; no se trasplanta automáticamente a ventanas temporales con entrenamiento expansivo. La propuesta temporal usa diferencias fuera de muestra alineadas, anidación de selección y bootstrap por bloques, justificando por qué no se tratan meses solapados como réplicas iid. Esa alternativa conserva la dependencia observada de pérdidas, pero no reproduce por sí sola el efecto de volver a entrenar todos los modelos bajo otras historias. La elección y sus límites se cerrarán con §7 y §13.

### 6.3.5 Diebold–Mariano con corrección de muestra finita

Para comparar precisión temporal se propone DM con pérdida cuadrática coherente con la pérdida principal y corrección Harvey–Leybourne–Newbold {cite:p}`e2_dm,e2_hln`. Para cada origen se promedian **pérdidas individuales de las semillas**, no predicciones de un ensamble:

$$
\bar\ell_{m,t}=\frac1R\sum_s(Y_t-\widehat Y_{m,s,t})^2,
\qquad a_t=\bar\ell_{A,t}-\bar\ell_{B,t}.
$$

Se estima la varianza de largo plazo de a mediante HAC y un ancho L>0 predefinido a partir de desarrollo/piloto, con sensibilidades registradas. No se impone automáticamente L=h−1=0: aunque **h=1**, las etiquetas σ₇ y las ventanas solapadas pueden producir dependencia de más de un rezago. D=7 y la ventana del objetivo no se sustituyen por h en la corrección:

$$
DM_{HLN}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}},
\qquad h=1\Rightarrow\sqrt{(T-1)/T}.
$$

La corrección finita no elimina por sí sola autocorrelación, cambios de régimen o el problema de estimar HAC con sesenta fechas. DM contrasta igualdad de la pérdida MSE media; **no contrasta automáticamente la mejora δ de RMSE media entre semillas**. Las RQ prácticas mantienen su funcional y contraste por bloques. DM es una comparación secundaria con familia, cola, HAC y aproximación de p fijados previamente; su resultado no basta para declarar relevancia práctica.

### 6.3.6 Diagnóstico de residuos y cuadrados

Para h=1 se examinarán residuos fuera de muestra por modelo y por una regla de semilla registrada, junto a su estabilidad entre semillas; no se escogerá la semilla más favorable ni se sustituirán por residuos de un ensamble no registrado. No se concatenan las semillas como si fueran una serie temporal más larga ni se promedian sus p-valores.

- **Ljung–Box en residuos:** evalúa autocorrelación conjunta hasta los rezagos declarados. Se informan rezagos, grados de libertad y corrección por multiplicidad; en una red fuera de muestra se propone `model_df=0`, sin restar el número total de pesos como si fuera p+q de ARMA. Los grados de libertad y la aproximación asintótica requieren cautela con sesenta fechas {cite:p}`e2_ljung`.
- **Ljung–Box en residuos al cuadrado:** detecta dependencia de magnitudes compatible con heterocedasticidad dinámica. No constituye por sí solo una prueba de un modelo ARCH específico ni identifica su causa.
- **BDS:** evalúa la hipótesis iid y puede detectar dependencia no lineal remanente. Se registrarán dimensión de embedding, radios, escalado, tamaño efectivo y validez de la aproximación; rechazar puede deberse también a dependencia lineal, heterocedasticidad o cambios de distribución. No prueba por sí solo una mejora alcanzable por una arquitectura avanzada.

La cuadrícula de rezagos, radios y dimensiones formará familias diagnósticas explícitas, con ajuste y alcance exploratorio. No se declara un solo p del modelo después de explorar muchas combinaciones. Si la muestra no soporta la dimensión/rezago o falla una condición numérica, el resultado es no estimable con motivo, no p=1 o p=0. La ruta exacta y los métodos pendientes de §7.3–7.4 deben recibirse antes de cerrar estos diagnósticos; documentarlos no afirma que todos estén implementados en el analizador actual.

### 6.3.7 Reporte completo y alternativas opcionales

Se publicarán tamaño de muestra/unidad, hipótesis, efecto, IC, p numérico sin ajustar y ajustado, tamaño de familia, método y estado de supuestos; nunca solo «p<0,05». La precisión Monte Carlo y los casos no definidos acompañan los p. Un IC nominal que excluya cero puede no sustentar una afirmación familiar después de multiplicidad.

Model Confidence Set es opcional: requiere pérdidas comparables, bootstrap temporal y nivel registrados; pertenecer al conjunto no significa equivalencia ni superar δ. Un análisis bayesiano de rangos con signo y ROPE también es opcional, con modelo/prior, unidad y región práctica declarados antes; sus probabilidades posteriores no se interpretan como p-valores o FWER. Ninguna opción se adopta para sustituir a posteriori un resultado clásico desfavorable.

**No rechazar una diferencia no demuestra equivalencia.** Una afirmación de equivalencia requiere un procedimiento definido, como TOST con margen bilateral justificado y dependencia/multiplicidad tratadas, o evidencia bayesiana dentro de una ROPE registrada. δ es una referencia de mejora absoluta; usarlo como margen simétrico de equivalencia requiere una justificación explícita.

## 6.4 Discusión crítica y estado de las nueve preguntas

Las respuestas siguientes delimitan lo que hoy puede afirmarse. El cierre exigirá magnitudes e incertidumbre procedentes del benchmark Bitcoin, en vez de conclusiones basadas solo en promedios, reputación de una arquitectura o resultados ETTh1.

| Pregunta de la guía | Estado actual | Evidencia necesaria para una respuesta respaldada |
|---|---|---|
| 1. ¿Qué modelo funciona mejor y por qué? | **No determinado en Bitcoin E2** | RMSE y rangos sobre las mismas fechas, efectos/IC y análisis de ablación e interpretabilidad que sustenten el mecanismo |
| 2. ¿Cuál generaliza y calibra mejor? | **No determinado**; un mercado y un test conocido limitan la generalización | Evaluación fuera de muestra identificada; cobertura/anchura o calibración apropiada al objeto predictivo, con incertidumbre y límites por régimen |
| 3. ¿Cuál es más robusto a perturbaciones y cambio de distribución? | **No determinado en D7**; el piloto ETTh1 tiene alcance externo | Curvas de tres severidades, control limpio, áreas, estratos causales y efectos pareados con IC; no extrapolar otro dataset a Bitcoin |
| 4. ¿Las diferencias son significativas y prácticamente relevantes? | **Sin contrastes del benchmark Bitcoin** | Familias ajustadas y límites simultáneos respecto a δ; distinguir mejora frente a GRU y frente a persistencia |
| 5. ¿Existe trade-off entre desempeño y coste? | **Sin frontera Bitcoin medida** | Costes bajo el mismo protocolo, frentes no dominados y magnitudes de ahorro/deterioro con incertidumbre |
| 6. ¿Qué tan cerca está el mejor modelo del techo y cuánto margen queda? | **Techo no estimado; §12 pendiente** | Definición y estimación del techo apropiado a σ₇, con incertidumbre. R²=1 no se adopta como un techo empírico alcanzable; no se resta MSE ETTh1 de RMSE Bitcoin |
| 7. ¿Se confirman o rechazan las RQ de §1? | **Sin decisión nueva**; las RQ siguen exploratorias | Efectos g, d y c, reglas RQ1–RQ3, ajuste y poder/capacidad; un no rechazo no confirma una nula ni elimina una hipótesis por sí solo |
| 8. ¿Qué amenazas afectan la validez? | Riesgos identificados; no constituyen una auditoría ya aprobada | Interna: fuga, test conocido y HPO desigual. Externa: un activo/exchange/periodo. Constructo: σ₇ como proxy y adecuación de métricas. Estadística: dependencia, pseudorreplicación, multiplicidad y poca información temporal |
| 9. ¿Cuáles son las limitaciones y recomendaciones futuras? | Benchmark, calibración, techo y rutas metodológicas sin cerrar | Documentar casos adversos/inconclusos y pendientes; priorizar futuras mejoras según efecto medido, coste y evidencia, sin prometer utilidad financiera |

El baseline operativo heredado sigue siendo **GRU/D7-c03**, elegido antes de la reserva de E1. Un ranking futuro de test no lo renombra retrospectivamente. Los resultados negativos de la réplica o del piloto externo se conservan, pero no determinan cuál de las quince configuraciones es mejor en Bitcoin.

### 6.4.1 Herramientas, descargas y cierre pendiente

`scripts/analyze_e2_benchmark.py` prepara el análisis documental y las plantillas sin recibir predicciones por defecto. Las funciones numéricas comprobadas con fixtures sintéticos validan software; no son evidencia de desempeño, potencia o ausencia de fuga en Bitcoin. El informe debe identificar métodos disponibles, no implementados o no aplicables, en lugar de inventar resultados para cumplir una lista.

- [Informe del estado de análisis](../_static/benchmark_analysis/report.json), [tabla pendiente](../_static/benchmark_analysis/table_template.csv) y [trazabilidad de requisitos](../_static/benchmark_analysis/requirements.csv).
- [Descargar analizador y plantillas](../_static/benchmark_analysis_download/Benchmark_E2_analisis_plantillas.zip).
- [Contrato heredado](../_static/e1_reference.json), [diseño de implementación](04_implementacion.md) y [evidencia externa separada](05_paper_guia.md).

El capítulo aporta un plan de análisis y una estructura revisable. Completarlo exige predicciones Bitcoin verificadas, presupuesto comparable, potencia y familia cerradas, protocolo de eficiencia/calibración, rutas estadísticas y Apéndice A completos. Hasta entonces, sus celdas, rangos, fronteras y conclusiones siguen **PENDIENTES**.
