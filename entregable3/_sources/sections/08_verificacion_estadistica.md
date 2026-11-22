# 8 Bloque D Verificación estadística y evaluación ampliada

**Corte actual, 21 de noviembre de 2026:** {ref}`8.28 <e3-input-only-audit-20261121>` recalcula el piloto V5 desde predicciones guardadas: input-only mejora **2,7595 %** frente a global y gana **4/5** semillas sobre **55 fechas ya expuestas**. Fue seleccionada después de examinar desarrollo; pasa del rango 6 al 1 entre selección y assessment y presenta SD entre semillas **7,27×** la de global. Se continúa su preparación con identidad propia; V5 completo conserva abandono. El protocolo sigue **DRAFT**, sin métodos aprobados, potencia, tamaño final o prueba sellada; **R3 no acreditada**.

**Corte de la auditoría exploratoria del 19 de noviembre de 2026:** la auditoría exploratoria de 19 sistemas en D00–D03 completó **796 trabajos, 76 paneles, 12 tablas de análisis y 16 figuras**. El [cierre de 8.18](#e3-final-extended-audit) contiene sus resultados y limitaciones. Las secciones anteriores conservan sus cortes históricos; los 48 contrastes retrospectivos descritos a continuación pertenecen a otra comparación y no se trasladan a la auditoría ampliada. Esta última no calculó nuevos valores *p*, ajuste de Holm o tres pruebas confirmatorias por rival.

**Corte del 17 de noviembre de 2026: análisis retrospectivo ejecutado sobre predicciones congeladas.** La CNN-LSTM con atención de contenido y salida aditiva positiva tiene diez semillas; cada una de las trece redes rivales también tiene diez. Persistence, SeasonalNaive y ARIMA conservan una trayectoria determinista cada uno. La comparación incluye los diecisiete sistemas sobre las mismas 608 fechas, sin eliminar rivales, semillas ni observaciones.

Los **48 contrastes principales** —dieciséis rivales por tres procedimientos— rechazaron sus nulas frente al margen heredado, con **Holm sobre los 48 p-valores** y máximo p ajustado **0,0019**. Las sensibilidades temporales conservaron esa decisión. Este resultado está condicionado al calendario y a los supuestos de cada método: **no acredita por sí solo R3**, un diseño confirmatorio previo o la originalidad del mecanismo. El MCS auxiliar conserva diez sistemas, incluido E3; no identifica un ganador único.

Los suplementos **8.10–8.12, ejecutados el 18 de noviembre**, añaden una auditoría del piso observable, un diagnóstico retrospectivo de intervalos y dimensionamiento condicional para una evaluación futura. Mantienen separados esos resultados nuevos de las tablas, las predicciones y los 48 contrastes del corte anterior. Incluyen resultados adversos: el control sólo piso tiene menor MAE que E3 y la cobertura posterior cae por debajo del nominal en 2026Q1. Ninguno acredita una campaña confirmatoria o R3.

## 8.1 Protocolo experimental pre-registrado

**No existe un preregistro confirmatorio de esta comparación.** El [borrador de Bloque D](https://github.com/jassan-bit/DeepLearning_202630/blob/c4a4a1e68e9844cdfb81de0075177ced0cd96c07/experiments/e3_bloque_d/protocolo_borrador.md) continúa identificado como borrador. El plan de esta etapa se fijó en el commit [4a21b032](https://github.com/jassan-bit/DeepLearning_202630/commit/4a21b0329b99e864ec64fc2df49260aa874141bf), después de conocer las predicciones y sus métricas medias y antes de calcular estos p-valores. El código probado se fijó en [02132664](https://github.com/jassan-bit/DeepLearning_202630/commit/0213266404a0aa3af985d7306dce6b13892c71ec) antes de ejecutar el análisis.

La cronología permite reproducir las decisiones de esta etapa; no restaura el cegamiento del test. Se conservan el {download}`plan con once insumos y SHA256 <../_static/e3_estadistica_posterior_20261117/plan.json>` y el {download}`protocolo de análisis posterior <../_static/e3_estadistica_posterior_20261117/protocol.md>`.

El {download}`borrador histórico de protocolo <../_static/e3_bloque_d/protocolo_borrador.md>` conserva su archivo y su enlace de descarga anteriores. Su estado de borrador no cambia por esta evaluación posterior.

El [documento canónico de trabajo](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/protocolo.md) organiza ahora los campos exigidos, con fecha real y estado **borrador posterior a los resultados conocidos**. No lleva una etiqueta que certifique un congelamiento confirmatorio anterior. Antes de una campaña futura deben resolverse y registrarse los siguientes campos; crear el archivo no completa sus decisiones:

Se conservan su {download}`copia de borrador para revisión <../_static/e3_revision_bloque_d_20261117/protocol_draft.md>` y la {download}`matriz de preparación del protocolo <../_static/e3_revision_bloque_d_20261117/protocol_readiness.csv>`. Los campos pendientes no se convierten en decisiones cerradas mediante estos archivos.

| Campo mínimo de la sección 8.1 de la guía | Evidencia disponible y cierre necesario |
|---|---|
| Estándar de profundidad y compromiso | [Estándar teórico](02_estandar_profundidad.md) y `experiments/estandar.md`; revisión y registro final con fecha/tag pendientes. |
| Hipótesis principales y exploratorias | La ganancia de RMSE y el margen se especifican abajo. Las hipótesis del mecanismo y predicciones secundarias deben corresponder al aporte defendido; las de penalización/saturación no se trasladan a la salida aditiva. Conservar lo posterior como exploratorio. |
| Métrica primaria, secundarias y margen heredado | Media de RMSE individuales, MAE secundaria y δ heredado. No cambiar de métrica, dirección o margen después de ver la prueba; las métricas descriptivas añadidas conservan su fecha posterior. |
| Fuente del mecanismo y transferencia | [Procedencia y antecedentes](06_modelo_original.md), con componentes conocidos y revisión abierta. No inventar una fuente o adaptación para completar la casilla. |
| Tres conjuntos adicionales | La auditoría exploratoria completada en [8.18](#e3-final-extended-audit) evalúa **D01–D03 y D00 con 19 sistemas en cada conjunto**. Conserva sus dependencias de mercado y temporales; no acredita tres dominios independientes ni el diseño confirmatorio de R3. Los inventarios anteriores conservan sus cortes históricos. |
| Particiones, ejecuciones, semillas y determinismo | Calendario y diez inicializaciones registrados para la comparación histórica; 8.17 fija particiones, purgas y semillas de la auditoría exploratoria. Quedan pendientes la separación confirmatoria y el número final dimensionado por potencia. |
| Familia, pruebas, ajustes y nivel | Los 16 rivales permanecen completos. El análisis posterior tiene Holm global48, α=0,05 y MCS auxiliar al 90 %; justificar las pruebas y familias definitivas antes de confirmar. |
| Optimización, cómputo y potencia | Registrar número de configuraciones, épocas/parada y tiempo real por modelo, con capacidad y hardware. La comparación histórica no declara presupuesto equivalente. La auditoría exploratoria de 8.17 fija rejilla y presupuesto común; la potencia por contraste y conjunta sigue pendiente. |
| Nd y desarrollo frente a confirmación | {ref}`Inventario conciliado de 7.14 <e3-design-inventory-reconciled>`: variantes, controles, propuestas, 183 ensayos HPO y 844 trabajos con ámbitos separados. No se cuentan semillas como diseños ni se inventan decisiones sin registro. |
| Fecha/tag y desviaciones | Registrar la versión final antes de sus experimentos y mantener una bitácora fechada de desviaciones, con motivo e impacto. Un análisis no previsto conserva alcance exploratorio y no sustenta por sí solo R3. |

La ecuación {eq}`eq-e3-primary-mean-rmse-gain` define la métrica primaria como la **media de las RMSE individuales**, en unidades originales de sigma7 poblacional, sin anualización:

$$
M_j=\frac1{R_j}\sum_{r=1}^{R_j}\sqrt{\frac1{608}\sum_{t=1}^{608}(y_t-\widehat y_{j,r,t})^2},\qquad
\widehat\Delta_j=M_j-M_{\mathrm{E3}}.
$$ (eq-e3-primary-mean-rmse-gain)

Positivo favorece a E3. No se sustituye esta media por la RMSE de un ensemble o por la raíz del MSE agregado entre semillas. Se mantiene el margen del borrador:

$$
H_{0,j}:\Delta_j\le\delta,\qquad H_{1,j}:\Delta_j>\delta,
\qquad\delta=0{,}00025772842664981106.
$$ (eq-e3-superiority-hypotheses)

El parámetro y la incertidumbre representada dependen de cada procedimiento. Las semillas son **271801–271810**. La semilla histórica 20261116 queda fuera de la comparación. Las fechas comunes van del **1 de enero de 2025 al 31 de agosto de 2026**; este calendario ya era conocido.

## 8.2 Equidad de la comparación

La auditoría verificó los once archivos fijados, las fechas y objetivos exactos, las 608 máscaras válidas, las coordenadas de semilla y la conservación de los 81 predictores históricos de E2 dentro de su ampliación a **133 predictores**. Las métricas guardadas coinciden con las recalculadas. Se conserva el {download}`informe de admisión <../_static/e3_estadistica_posterior_20261117/input_validation.json>`.

| Tratamiento | E2: trece redes | CNN-LSTM E3 |
|---|---|---|
| Entrenamiento | 655 ejemplos; hasta 2024-10-31 | 532 ejemplos; hasta 2024-06-30 |
| Purga entre ajuste y validación | Receta final de E2 conservada | 7 ejemplos |
| Validación para selección/parada | Épocas finales fijadas por la receta previa | 116 ejemplos; 2024-07-08 a 2024-10-31 |
| Calibración de la campaña histórica | 53 ejemplos; 2024-11-08 a 2024-12-30 | No utilizada en la campaña original; el diagnóstico separado de 8.11 abre esas ventanas después |
| Prueba común | 608 fechas | Las mismas 608 fechas y objetivos |
| Repeticiones estocásticas | 10 por red | 10 |

La ampliación de E2 añadió cuatro semillas por red, **52 ajustes**, sin retunear las recetas. Los tres clásicos siguen almacenados una sola vez. Los ajustes y calibradores nuevos quedaron confirmados antes de producir sus nuevas predicciones de prueba. La CNN-LSTM se reutilizó desde su réplica de diez semillas.

**Tener diez semillas por red no iguala las particiones, la selección ni el presupuesto de entrenamiento.** Estos contrastes comparan las recetas ejecutadas; no aíslan un efecto causal de arquitectura. Se conservan separados los 5130,970719594005 segundos registrados de la ampliación E2 y los 948,439625042 segundos cerrados de E3. La etapa estadística del 17 de noviembre ejecutó cero ajustes y cero inferencias nuevas, en CPU; los 530 forwards de calibración del suplemento 8.11 se registran aparte.

La equidad futura requiere fijar la misma herramienta y número de intentos de HPO, máximo de épocas, criterio de parada y espacio relativo de búsqueda; el mismo número de configuraciones no iguala tiempo cuando su coste difiere. Debe declararse si el presupuesto se mide en configuraciones, cómputo o ambos, e igualar el tiempo total o reportar ambos análisis cuando corresponda. El esfuerzo de diseño se registra aparte del HPO. No se cambia la receta congelada de un rival para favorecer retrospectivamente la comparación.

El preprocesamiento, las variables disponibles, aumentos aplicables, particiones, fuente de pesos preentrenados y métrica deben ser comunes en el protocolo pertinente. Se reportarán parámetros y FLOPs de todos los modelos; si el original resulta sustancialmente mayor que el mejor benchmark, se incluirá una comparación de capacidad comparable o una discusión explícita del efecto del tamaño. Las medidas existentes de GPU, eager CPU y TorchScript tienen condiciones distintas: faltan una contabilidad completa por modelo de entrenamiento/inferencia, hardware, memoria pico y energía estimada con método. Una trayectoria determinista se conserva como tal: duplicarla diez veces no produce incertidumbre de inicialización.

## 8.3 Evaluación en conjuntos de datos adicionales

**La auditoría ampliada ya evaluó los tres conjuntos adicionales.** El [cierre de 8.18](#e3-final-extended-audit) contiene resultados para D01–D03 y el BTC D00 bajo la política común. El análisis histórico del 17 de noviembre, conservado en 8.1–8.12, sólo usó una cronología BTC conocida. Las semillas no son conjuntos de datos independientes; no se ejecuta Friedman o Iman–Davenport usando diez inicializaciones como si fueran diez mercados. El análisis nuevo ofrece rangos descriptivos entre conjuntos, sin una nueva prueba formal entre dominios.

La selección, admisión y comparación exploratorias están ejecutadas; permanecen pendientes los márgenes y la batería confirmatoria con su potencia y adecuación temporal. No se promedian RMSE de escalas distintas ni se denomina confirmación independiente a otro tramo ya expuesto. La presencia de exposiciones cripto compartidas y solapamiento BTC limita la generalización.

El inventario preparatorio inicial del 18 de noviembre, resumido en **8.12**, identificó tres candidatos con **cero conjuntos admitidos o evaluados en ese corte**. La revisión posterior de calendario y política de **8.13** admitió D01 WIKI/AAPL para una tarea RAW entre sesiones: aquel corte fue **1 de 3 archivos admitidos, 0 evaluados por modelos y 2 pendientes**. La actualización 3/3 y los resultados parciales se conservan en 8.17; el cierre completo está en 8.18. Cada etapa mantiene su fecha y alcance.

Cada conjunto debe mantener la **misma tarea y métrica primaria** y diferir del principal en al menos dos ejes pertinentes de la guía: tamaño, dominio, ruido, dimensionalidad o desbalance cuando sea aplicable. En esta regresión no se inventa un desbalance de clases; la distribución del objetivo puede describirse como contexto sin convertirla automáticamente en otro eje acreditado. Su ficha debe identificar fuente pública, licencia/condiciones, versión y hash, periodo/unidad independiente, ejes de diferencia, particiones, auditoría de fuga, presupuesto y margen con unidades. Los 17 sistemas se evalúan en cada conjunto; una limitación de memoria del dashboard no autoriza excluir un rival científico.

La consolidación podrá usar Friedman con corrección de Iman–Davenport y comparaciones por pares con Holm, o una alternativa bayesiana de rangos con signo y ROPE previamente justificada. La elección se fija antes de ver los resultados, con rangos y márgenes pertinentes; con sólo tres conjuntos la potencia global es limitada. Se informarán también diferencias, IC y efectos por conjunto y la consistencia del resultado, sin mezclar RMSE de escalas distintas en un promedio bruto. Un conjunto usado para rediseñar el modelo pasa a desarrollo y deja de aportar confirmación independiente.

## 8.4 Definición operativa de superioridad R3

La guía exige **seis condiciones simultáneas**; ninguna se sustituye por otra. La evidencia de la tabla corresponde al corte retrospectivo BTC del 17 de noviembre; la evaluación adicional posterior está en 8.18 y la nueva implementación controlada y sus límites en {ref}`8.25 <e3-next-statistics-data-20261121>`:

| Condición de la sección 8.4 | Evidencia de este corte | Cierre necesario |
|---|---|---|
| 1. Mejor media primaria frente a cada rival | E3 tiene la menor RMSE media individual de los 17 sistemas históricos; MAE también favorable dentro de esa familia. | Observado en el calendario histórico, con tratamientos de entrenamiento distintos. El control posterior sólo piso de 8.10 conserva su resultado adverso de MAE fuera de esa familia. |
| 2. Al menos tres pruebas distintas admisibles por contraste, control familiar y ganancia mayor que δ | 48/48 rechazos bajo tres procedimientos posteriores, con Holm global48. | Justificar su admisibilidad según la Tabla 4 y sus supuestos. Ni la suma de tres métodos ni el MCS auxiliar sobre otra pérdida completan automáticamente esta condición. |
| 3. Efecto no despreciable e IC favorable | Ganancias, IC temporales y Cohen dz entre semillas disponibles; frente al SAE la ganancia es 0,0009334571. | La guía usa $\lvert\delta_{\rm Cliff}\rvert\ge0,147$ o $\lvert d_{\rm Cohen}\rvert\ge0,2$ cuando sean apropiados, y límite inferior del IC95 % de la diferencia >0, idealmente >δ. Justificar unidad y alcance; un dz condicionado al calendario no mide diferencias entre mercados. Los IC publicados son marginales, no simultáneos. |
| 4. Consistencia adicional | No hay resultados en tres conjuntos adicionales. | Mejor media en al menos dos de los tres y ninguna inferioridad significativa frente a un rival en cualquiera. La ausencia de rechazo de inferioridad no se renombra como equivalencia o no inferioridad demostrada. |
| 5. Adecuación del propio modelo | Esta etapa compara errores guardados; no completa las Tablas A/B de E3. | Superar el baseline trivial, brecha train–validación dentro del margen de adecuación, calibración/residuos aceptables y consistencia validación–prueba bajo datos y riesgos comparables. |
| 6. Robustez a la elección de la prueba | Bloques/rezagos 7/14/28 y MAE secundaria mantienen decisiones posteriores. | Alternativas paramétricas/no paramétricas pertinentes y sensibilidad a las correcciones que el diseño admita. Nadeau–Bengio no se aplica a días o semillas de esta única cronología; esa limitación debe justificarse, sin simular una comparación inexistente. |

Además siguen pendientes el preregistro previo, la comparación justa y la potencia exigida por el estándar. Por tanto, **R3 permanece sin acreditar**. La inferioridad no rechazada, la pertenencia al MCS o un p pequeño no prueban equivalencia, originalidad o cumplimiento del entregable completo.

## 8.5 Familia de contrastes

Se conservaron los dieciséis rivales fijados en el borrador:

| Grupo | Modelos |
|---|---|
| Top 10 de E2 | FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST, iTransformer |
| Referencias neuronales | MLP16, LSTM32, GRU32 |
| Clásicos deterministas | Persistence, SeasonalNaive, ARIMA |

```{csv-table} Comparación ampliada sobre las mismas 608 fechas
:file: ../_static/e3_table_intervals_20261121/metrics_extended_table.csv
:header-rows: 1
```

La tabla incorpora la CNN-LSTM y las diez semillas de cada red, sin elegir una réplica por rendimiento. MAE, sMAPE y R² se calculan por predictor y después se promedian. sMAPE se expresa en porcentaje; RMSE, MAE y anchura usan unidades originales de σ7, sin anualización. Para una trayectoria,

$$
\operatorname{sMAPE}=\frac{200}{n}\sum_t\frac{|y_t-\widehat y_t|}{|y_t|+|\widehat y_t|},\qquad
R^2=1-\frac{\sum_t(y_t-\widehat y_t)^2}{\sum_t(y_t-\overline y)^2},
$$ (eq-e3-smape-rsquared)

con contribución cero cuando ambos valores de un término sMAPE son cero. La media de R² no es el R² de un ensamble. E3 obtiene sMAPE **11,0977 %** y R² medio **0,911779** en este calendario. La [tabla abreviada del corte estadístico](../_static/e3_estadistica_posterior_20261117/ranking_table.csv) conserva sus bytes; el suplemento amplía métricas descriptivas y no cambia sus p-valores.

La cobertura y anchura de E2 se reconstruyen exclusivamente con los cuantiles residuales **ya congelados** de cada predictor, mediante $[\max(0,\widehat y-q),\widehat y+q]$; no se ajustó un nuevo calibrador con prueba. Son medias de las coberturas y anchuras individuales, no de un intervalo de ensamble. **La fila E3 incorpora los intervalos del suplemento retrospectivo de 8.11:** cobertura media **94,490 %** y anchura media **0,010963444 σ₇**, sobre las mismas 608 fechas y las diez semillas 271801–271810. Cada predictor usa su propio cuantil de rango 49, calculado sobre 53 ventanas separadas de calibración; se reutilizan las 6080 predicciones puntuales guardadas, sin entrenar, volver a predecir o recalibrar en esta actualización. Estos intervalos se calcularon después de la campaña original y sobre una evaluación ya conocida: su incorporación es descriptiva y no convierte el análisis en confirmación prospectiva. La DE entre semillas caracteriza la dispersión de los ajustes; el nominal 90 % no constituye una garantía individual bajo dependencia o cambio temporal.

La {download}`tabla histórica del 17 de noviembre <../_static/e3_revision_bloque_d_20261117/metrics_extended_table.csv>` conserva sus bytes y sus dos campos ausentes de aquel corte. El {download}`CSV actualizado de esta tabla <../_static/e3_table_intervals_20261121/metrics_extended_table.csv>` completa exclusivamente esas dos celdas a partir de las {download}`métricas posteriores por semilla <../_static/e3_remediacion_20261118/calibration/results/per_seed_intervals.csv>`; su {download}`recibo de procedencia <../_static/e3_table_intervals_20261121/provenance.json>` registra valores, agregación y hashes. Son medias de diez predictores; el intervalo del representante servido 271801 conserva sus métricas individuales y su versión, como se explica en 8.11 y en el capítulo 9.

```{figure} ../_static/e3_estadistica_posterior_20261117/ranking_rmse.svg
:alt: Los diecisiete sistemas ordenados por RMSE medio; las redes incluyen su desviación estándar entre diez semillas.
:width: 100%

RMSE medio y DE muestral entre inicializaciones. Los clásicos tienen R=1 y no reciben una barra de dispersión artificial. La DE entre semillas no es un intervalo predictivo ni temporal.
```

La CNN-LSTM obtuvo **RMSE medio 0,0029394324** y **MAE medio 0,0018928273**. El mejor rival por RMSE fue LSTM-autoencoder, con **0,0038728895**: la reducción relativa observada de E3 fue **24,10 %**. Frente a ese rival, la ganancia absoluta fue **0,0009334571**, con IC95% temporal marginal **[0,0006329359; 0,0012131746]**.

La afirmación «supera a todos» es una regla de **unión–intersección**: deben satisfacerse todos los contrastes exigidos, no un subconjunto favorable. La familia tiene 16 rivales, aunque la guía ilustre otros tamaños con m≈12; el control posterior abarca sus 48 componentes de prueba. Cualquier exclusión de una futura familia tendría que justificarse antes del congelamiento por razones ajenas al desempeño observado. Las familias de ablación, conjuntos adicionales, subgrupos y horizontes se declaran y controlan por separado; no se usa su separación para hacer más permisiva la afirmación principal.

## 8.6 Reglas metodológicas para que las pruebas sean válidas

| Procedimiento ejecutado | Unidad y supuestos | Interpretación |
|---|---|---|
| Bootstrap temporal circular pareado | 9999 réplicas; bloques de 14 días sincronizados para todos los sistemas y semillas; trayectorias ajustadas fijas; aproximación de dependencia débil y estabilidad temporal | Ruido de la ganancia de RMSE centrado para contrastar la frontera delta; incertidumbre del calendario condicionada a los ajustes |
| Influencia de la media de RMSE con Bartlett-HAC | Derivada de cada raíz antes de promediar, 14 rezagos, aproximación normal y varianza de largo plazo | Alternativa temporal; no es DM–HLN ni un traslado del margen de RMSE a MSE |
| t pareada entre semillas | Diez ganancias por identificador de semilla, 9 grados de libertad; inicializaciones independientes y ganancias aproximadamente normales | Incertidumbre de inicialización condicionada al calendario; el clásico aporta una trayectoria fija, no diez ajustes inventados |

Para HAC se usa la influencia

$$
\psi_{j,t}=\frac1{R_j}\sum_r\frac{e_{j,r,t}^2-\overline{e_{j,r}^2}}{2\operatorname{RMSE}_{j,r}},
\qquad z_j=\frac{\widehat\Delta_j-\delta}{\operatorname{SE}_{\rm HAC}(\psi_j-\psi_{\rm E3})}.
$$ (eq-e3-rmse-influence-hac)

El bootstrap recalcula la métrica completa y compara su ruido centrado con la distancia observada al margen. Usa la corrección Monte Carlo de uno en numerador y denominador, con resolución **0,0001**. Los p no estimables permanecen en la familia mediante una contribución conservadora de uno para el ajuste y se reportan como no estimables.

Los dieciséis diagnósticos Shapiro entre semillas no rechazaron normalidad a 0,05; el menor p fue 0,0564. **Eso no demuestra normalidad con diez observaciones.** Los supuestos de estabilidad temporal y del calendario tampoco quedan certificados por los p de comparación. Holm no corrige la exposición previa al test, la selección del diseño ni un supuesto inválido.

La guía exige al menos tres pruebas distintas y apropiadas y al menos un procedimiento global o con control de comparaciones múltiples; no prescribe literalmente idéntico ámbito de incertidumbre. El análisis posterior aplica Holm global48, pero los procedimientos y sus alcances deben justificarse para la hipótesis de la ruta temporal F2: sus rechazos no acreditan automáticamente toda la Tabla 4 ni R3. No se aplican correcciones Nadeau–Bengio a días o semillas de esta única cronología, ni Clark–West sin justificar anidamiento.

### Relación con las pruebas admisibles de la Tabla 4

La ruta literal del Cuadro 4 de la guía es **F2. Temporal**. «F2a» se conserva sólo como rótulo de algunos documentos históricos; no es una categoría literal de la guía. La comparación actual no se renombra para hacer coincidir métodos diferentes con la tabla de la guía:

| Método de la fila temporal de la guía | Relación con lo ejecutado y decisión que falta |
|---|---|
| MCS | Hay un MCS90 auxiliar de MSE diario medio entre semillas. Es un procedimiento global, pero su estimando no es la media de RMSE ni la ganancia mayor que δ; no sustituye un contraste de ese margen. |
| Diebold–Mariano con corrección HLN por horizonte | No se ejecutó con ese nombre o construcción. El HAC de influencia RMSE es otro procedimiento. La pérdida, el horizonte y la dependencia temporal deberán justificarse; h=1 no elimina la dependencia por ventanas de σ7 solapadas. |
| Giacomini–White o Clark–West según anidamiento | No ejecutados. Exigen el diseño y supuestos pertinentes; no se aplica Clark–West a arquitecturas arbitrarias por compartir datos, ni se añade un instrumento después de mirar qué produce un p favorable. |
| Friedman sobre series u orígenes | No ejecutado con unidades independientes justificadas. Semillas o días correlacionados no se convierten en series independientes para llenar la tabla. |
| Bootstrap estacionario | El análisis usa bloques **circulares de longitud fija**, no el bootstrap estacionario de longitud aleatoria. Debe justificarse esa alternativa y su alcance; no se presenta como un procedimiento idéntico por cambiarle el nombre. |

La t entre inicializaciones añade información condicional al calendario, pero no se convierte en una tercera prueba temporal de la Tabla 4. La pertinencia de la batería final debe cerrarse antes de la confirmación, conservando los resultados posteriores ya calculados. Las filas tabular, espacial, imágenes o lenguaje no se importan como pruebas adicionales de BTC por estar enumeradas en la guía.

### Supuestos, ajuste, calibración y residuos

El pareado de fechas, objetivos y coordenadas de semilla fue verificado. El pareado estricto de una campaña justa incluye además mismas particiones y preprocesamiento; no queda satisfecho por las 608 fechas comunes cuando los tratamientos de ajuste difieren. La prueba sellada se consulta al final con hiperparámetros, umbrales y parada fijados en validación. El calendario actual ya expuesto no se vuelve sellado por volver a puntuarlo.

| Diagnóstico pertinente a F2 | Qué debe comprobarse y alcance actual |
|---|---|
| Dependencia de residuos y de pérdidas | La {ref}`actualización de 8.20 <e3-existing-adequacy-close-20261120>` ejecuta Ljung–Box(7/14) y ARCH-LM(7) en dos versiones de pesos y familias diagnósticas separadas, con supuestos y resultados adversos. BDS no se añade sin justificar su pertinencia. Son diagnósticos retrospectivos y no acreditan adecuación completa. |
| Distribución y calibración puntual | Jarque–Bera como diagnóstico, Mincer–Zarnowitz y residuos con intercepto/pendiente y cohortes claramente definidas. La falta de normalidad de residuos no se sustituye por Shapiro entre diez ganancias de semillas. |
| Ajuste y sobreajuste del original | Tablas A/B propias, baseline trivial, brecha con el margen δg, riesgos comparables y consistencia dentro/fuera de muestra. δg conserva el valor numérico heredado cuando corresponda, pero su nula de brecha no es la de ganancia mayor que δ. Los diagnósticos de otras redes no acreditan a E3. |
| Calidad probabilística | PIT, cobertura, cuantiles, pinball, CRPS o log-verosimilitud sólo cuando exista la salida probabilística y el procedimiento que los defina. La cabeza aditiva es puntual: una colección de semillas no es por sí sola una distribución predictiva calibrada. |
| Cambio y estabilidad | Comparación de errores dentro/fuera de muestra y pruebas de estabilidad como Giacomini–Rossi cuando proceda. Deben distinguir cambio de distribución, mala especificación y fuga; ninguna conclusión se deduce de un p de superioridad. |

Se reportarán estadísticos, supuestos, p crudos/ajustados y unidades. La significancia práctica deberá discutirse con su magnitud, IC y coste adicional, incluso cuando p sea pequeño. Estos controles de adecuación y calidad no se completan mediante la tabla de comparación de modelos.

## 8.7 Estrategia estadística: familias, potencia y agregación

### 8.7.1 Control del error por familia

Se fijó **una familia principal de 48 hipótesis**, con Holm a alfa=0,05. Este control es más amplio que las tres familias de dieciséis propuestas en el borrador. No se eligió la prueba de menor p ni se excluyeron rivales después de ver el resultado. Para la descripción conjunta por rival se exige que los tres p ajustados sean menores que 0,05.

Para p ordenados $p_{(1)}\le\cdots\le p_{(m)}$, el ajuste monótono es

$$
p^{\rm Holm}_{(i)}=\min\left\{1,\max_{1\le k\le i}(m-k+1)p_{(k)}\right\}.
$$ (eq-e3-holm-adjustment)

En este corte $m=48$ en la ecuación {eq}`eq-e3-holm-adjustment`. Se informa por rival el **máximo** de sus tres p ajustados: todos deben rechazar. Holm controla el error familiar bajo p válidos sin exigir independencia entre pruebas; no corrige una prueba inválida o el uso previo del test para diseñar.

```{csv-table} Contrastes principales frente al margen heredado
:file: ../_static/e3_estadistica_posterior_20261117/contrasts_table.csv
:header-rows: 1
```

Los máximos p Holm fueron **0,0019** en bootstrap, **0,0018** en HAC y **0,0007875561** en la t entre semillas. Los 16 rivales registran los tres rechazos exploratorios. La {download}`tabla completa <../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv>` conserva los p crudos, ajustes, estadísticos, incertidumbre, diagnósticos y alcances.

```{figure} ../_static/e3_estadistica_posterior_20261117/gains_ci.svg
:alt: Ganancias de RMSE de E3 frente a los dieciséis rivales con intervalos temporales marginales y el margen heredado.
:width: 100%

Ganancias e IC95% temporales marginales con bloque 14. Una ganancia positiva favorece a E3. Estos intervalos no son simultáneos para toda la familia.
```

Los límites nominales del 95% no deben confundirse con límites simultáneos. La cota inferior Bonferroni48 del bootstrap frente a FCN-1D es **0,0001980411**, menor que delta, aunque el contraste Holm rechace. La conclusión no se sustituye por una afirmación inexistente de que todas las cotas Bonferroni superan el margen.

### 8.7.2 Potencia y número de semillas

**El corte histórico no estimó potencia prospectiva, potencia conjunta ni MDE a partir de este test.** Diez semillas cumplen una cantidad de repeticiones; no demuestran el objetivo de potencia de la guía. Calcular potencia observada usando el efecto de este mismo resultado no reemplazaría un dimensionamiento previo con un piloto adecuado. El suplemento **8.12** añade referencias condicionales de tamaño y MDE usando dispersión histórica y alternativas declaradas; no estima la potencia real de la campaña o de su batería completa.

La guía exige un piloto con **al menos cinco semillas** para estimar la dispersión de diferencias pareadas frente a cada rival y dimensionar la campaña antes de congelarla: potencia de al menos 0,90 por contraste, potencia conjunta reportada y diez semillas como mínimo o más si hace falta. Deben incluirse las decisiones de las tres pruebas y Holm; semillas y calendario no son fuentes de incertidumbre intercambiables. El piloto y la alternativa planificada no se eligen retrospectivamente para declarar suficiente el tamaño ya usado.

La aproximación normal bilateral ilustrada en la guía, con $\alpha'=0,05/m$, es

$$
R\simeq\frac{(z_{1-\alpha'/2}+z_{1-\beta})^2\,\widehat\sigma_d^{\,2}}{g^2},
$$ (eq-e3-power-normal-approximation)

donde $g$ es la distancia de la alternativa planificada a la frontera de la nula y $\widehat\sigma_d$ la dispersión pertinente. Si se contrasta igualdad y se planifica detectar una diferencia δ, entonces $g=\delta$. Para **nuestra nula de relevancia** $\Delta\le\delta$, una alternativa futura $\Delta_1>\delta$ da $g=\Delta_1-\delta$; poner δ en el denominador cambiaría el problema. La elección bilateral/unilateral y su cuantil debe fijarse de acuerdo con la hipótesis, no por el resultado. Esta fórmula no incorpora por sí sola dependencia temporal, estimación de hiperparámetros o incertidumbre conjunta.

Como referencia puramente ilustrativa, con m=12, potencia 0,80 y diseño bilateral normal, una distancia de una DE requiere aproximadamente 14 pares; media DE requiere unos 55. No son el R calculado para E3 ni sustituyen la exigencia 0,90. Si la campaña no alcanza potencia suficiente deberá aumentarse el diseño antes del cierre y, si continúa limitada, reportar su MDE y resultado no concluyente. La ausencia de significancia no demuestra equivalencia: TOST o ROPE requieren margen, procedimiento y datos pertinentes prefijados.

La resolución exacta también importa. Sin empates ni ceros, el menor p bilateral de Wilcoxon con n pares es $2^{1-n}$: con cinco pares es 0,0625 y no puede rechazar a 0,05. Para superar el **primer umbral conservador** de Holm, $0,05/m$, hacen falta al menos nueve pares si m=12, diez si m=16 y once si m=48. Son límites de resolución para esa comparación inicial; un paso posterior de Holm puede tener otro umbral, y estos números no prueban potencia 0,90. No se sustituyen pares independientes por días o semillas×días para alcanzar el conteo.

La potencia de «supera a todos» corresponde al evento conjunto de todas las decisiones exigidas. Multiplicar potencias marginales sólo sería válido bajo independencia: por ejemplo, $0,90^{12}\approx0,28$ es una ilustración de ese supuesto, no una estimación de este experimento. La dependencia por calendario, rival y método exige una planificación conjunta adecuada. La potencia retrospectiva no se obtiene de los p observados.

Las {download}`ilustraciones numéricas de planificación <../_static/e3_revision_bloque_d_20261117/planning_illustrations.json>`, su {download}`script reproducible <../_static/e3_revision_bloque_d_20261117/planning_illustrations.py>` y la {download}`revisión de métodos y alcances <../_static/e3_revision_bloque_d_20261117/methods_review.md>` documentan estas cuentas y sus supuestos. Son ejemplos matemáticos, no un análisis de potencia ejecutado para la campaña E3.

Los consumos históricos se mantienen cerrados y separados. El {download}`registro de ejecución CPU <../_static/e3_estadistica_posterior_20261117/execution.json>` identifica versiones, hashes y tiempo numérico; no se añaden ajustes o inferencias del modelo.

### 8.7.3 Agregación jerárquica y múltiples conjuntos de datos

Las unidades se mantienen diferenciadas: una serie compartida, bloques temporales, diecisiete sistemas y sus inicializaciones. Los 6080 pares semilla-fecha de una red **no se tratan como 6080 casos iid**. La t entre semillas y la inferencia temporal condicionan aspectos distintos; no forman una estimación conjunta de incertidumbre de entrenamiento, selección y mercado.

La {download}`tabla de efectos <../_static/e3_estadistica_posterior_20261117/effects.csv>` incluye Cohen dz de ganancias pareadas entre semillas e intervalos bootstrap nominales, con su alcance condicionado al calendario. No se interpreta como efecto entre mercados independientes.

Una campaña con varias series, particiones o unidades debe declarar su jerarquía antes de inferir: observaciones correlacionadas se agregan a la unidad pertinente o se modelan con dependencia explícita. El esquema de efectos mixtos de la guía,

$$
y_{ijr}=\mu+\alpha_j+u_i+v_r+\varepsilon_{ijr},
$$ (eq-e3-mixed-effects-schema)

se refiere al efecto del modelo $j$, de la unidad $i$ y de la semilla $r$; no autoriza tratar cada día de una serie como un efecto aleatorio iid. Su uso exigiría justificar estructura, correlación y estimabilidad y contrastar al original con cada rival con el control familiar fijado. No se ha ajustado ese modelo aquí. En varios conjuntos se reportarán también diferencias, IC, efectos y probabilidad de superioridad estimada sobre unidades válidas, con el significado de esa probabilidad explícito.

La sensibilidad de la conclusión debe distinguir alternativa de prueba, métrica secundaria registrada y subconjunto razonable de semillas. Se conservan las sensibilidades posteriores de 8.8 sin elegir la más favorable. No se usa «tendencia a la significancia»: falta de evidencia de diferencia y evidencia de equivalencia son afirmaciones distintas.

## 8.8 Robustez, sensibilidad y calidad probabilística

```{csv-table} Sensibilidades temporales prefijadas
:file: ../_static/e3_estadistica_posterior_20261117/sensitivity_table.csv
:header-rows: 1
```

Los bloques/rezagos **7 y 28** conservaron **48/48 rechazos**, cada uno con su familia completa y sin reemplazar la configuración principal de 14. Los máximos p Holm fueron **0,0029** y **0,0018539201**, respectivamente. Se conservan {download}`todos los contrastes de sensibilidad <../_static/e3_estadistica_posterior_20261117/sensitivities.csv>`.

Como métrica secundaria, los **16 contrastes bootstrap de MAE frente a cero rechazaron con Holm16**, con máximo p ajustado **0,0016**. No se fijó ni se inventó un margen práctico de MAE. La {download}`familia secundaria completa <../_static/e3_estadistica_posterior_20261117/mae_secondary.csv>` y los {download}`subconjuntos descriptivos de primeras/últimas cinco semillas <../_static/e3_estadistica_posterior_20261117/seed_subgroups.csv>` permanecen disponibles.

### MCS auxiliar con el modelo E3 incluido

El MCS usa **Tmax, nivel 90%, 4999 réplicas y bloques 14**. Su pérdida es el **MSE diario medio entre semillas**, otro estimando respecto a la media de RMSE primaria. Tras siete exclusiones, conserva **diez sistemas**, con p del último contraste global **0,1442**:

```{csv-table} MCS auxiliar: conjunto retenido y exclusiones secuenciales
:file: ../_static/e3_estadistica_posterior_20261117/mcs_table.csv
:header-rows: 1
```

E3 permanece junto a ARIMA, FCN-1D, GRU32, Inception-1D, LSTM-autoencoder, MLP16, Persistence, ResNet-1D y TimeMixer. **Pertenecer al conjunto no prueba equivalencia ni un ganador único**. Ese resultado global de MSE no se renombra como contraste de ganancia RMSE mayor que delta. El {download}`detalle secuencial del MCS <../_static/e3_estadistica_posterior_20261117/mcs.json>` incluye pérdidas, estadísticos y p.

Continúan pendientes la sensibilidad a hiperparámetros propios, las ablaciones comparables, las perturbaciones y curvas de tamaño de muestra, diagnósticos completos de adecuación, explicaciones verificadas, costes comparables y los tres conjuntos adicionales. La auditoría física de **8.10** y la calibración posterior de **8.11** aportan diagnósticos ejecutados con alcance exploratorio; no completan la validación probabilística confirmatoria o su integración en el servicio. Las sensibilidades de bloques y semillas aquí reportadas conservan su alcance histórico.

### Estudios exigidos por la Tabla 5

| Estudio | Evidencia actual y mínimo que falta |
|---|---|
| Sensibilidad a hiperparámetros | Falta variar hiperparámetros propios con superficies/curvas, región de ventaja y sensibilidad frente a benchmarks. Replicar la receta aditiva no constituye HPO ni este estudio. |
| Estabilidad por semilla y partición | Hay diez inicializaciones de una receta sobre un mismo calendario; se reportan dispersión y subgrupos. Falta variación pertinente de particiones, IC/coeficiente de variación con unidades claras y la comparación ANOVA/Kruskal–Wallis sólo cuando su diseño la justifique. Las semillas no son mercados independientes. |
| Perturbaciones controladas | Falta aplicar a toda la familia al menos tres severidades y curvas de degradación. En OHLCV, ruido, datos faltantes o atípicos deben respetar o declarar la coherencia temporal/física; no se traslada sin cautela el estrés artificial de E2. |
| Tamaño de muestra y escalabilidad | Faltan curvas de aprendizaje —por ejemplo 10 %, 25 %, 50 % y 100 %— con varias semillas, más tiempo y memoria al variar tamaño de entrada. Esos porcentajes ilustran el estudio, no corridas realizadas. |
| Cambio de distribución | Falta un periodo, dominio o fuente fuera de toda decisión de diseño. Los conjuntos adicionales y sus derechos/particiones se eligen antes de ver desempeño; reutilizar un calendario conocido no lo vuelve independiente. |
| Calibración y reglas de puntuación | El suplemento 8.11 calcula diez cuantiles posteriores con 53 ventanas de calibración y reporta score de intervalo sobre pronósticos históricos guardados. La campaña original y la app conservan su perfil sin intervalo E3. Faltan validación confirmatoria y los demás diagnósticos de adecuación; pinball, CRPS o log-score requieren las salidas que los definan. No se inventan ECE o probabilidades de clasificación para este objetivo. |
| Cuantificación de incertidumbre | E2 conserva sus cuantiles históricos; 8.11 añade cobertura, anchura, subgrupos e IC temporales posteriores para E3. La cobertura agregada de 94,49 % cae a 86,22 % en 2026Q1 y no demuestra calibración uniforme. El nominal 90 % no es garantía individual; el bootstrap condiciona en modelos y cuantiles fijos. La DE de semillas no reemplaza un intervalo predictivo. |
| Eficiencia y frontera de Pareto | Parámetros y algunos costes están inventariados; faltan FLOPs completos, latencias CPU/GPU, memoria pico, entrenamiento y energía con método común. Las tres referencias TorchScript no representan el catálogo completo ni una medición de Render. |
| Análisis de errores | El piloto conserva diagnósticos posteriores por régimen: en 18 fechas de objetivo realizado alto la aditiva tuvo RMSE 22,50 % mayor que M00 proyectado. Ese grupo no está disponible al pronosticar y no identifica una causa. Faltan escenarios prefijados, errores compartidos con rivales y discusión de fallos sin ocultar desenlaces adversos. |

Las intervenciones y criterios se fijarán antes de medir la evaluación correspondiente, con el mismo protocolo para todos los rivales. Una explicación por atención requiere además fidelidad y estabilidad comparadas con el mejor benchmark; no se convierte en evidencia causal sólo por visualizar sus pesos.

## 8.9 Reporte obligatorio de resultados

| Producto | Evidencia de este corte | Límite |
|---|---|---|
| Tabla histórica de 17 sistemas | RMSE, DE, MAE, sMAPE y R²; cobertura/anchura E2 reconstruidas de cuantiles congelados e IC bootstrap entre semillas disponibles por separado | Mismas fechas y tratamientos de entrenamiento distintos; campaña original y servicio sin intervalos E3. Los diagnósticos nuevos se separan en 8.10–8.12; recursos completos comparables pendientes |
| Familia completa | 48 contrastes, p crudos/Holm, estadísticos, efectos, IC, diagnósticos y conclusión | Retrospectivos; distintos ámbitos; los supuestos y la admisibilidad no se certifican sólo por rechazar |
| Sensibilidad y MAE | Bloques 7/14/28, subgrupos de semillas y Holm16 secundario | No perturbaciones o nuevos conjuntos |
| MCS con original | Diez sistemas retenidos al 90% | MSE auxiliar; no ganador único o equivalencia |
| Potencia y MDE | Ausentes en el diseño histórico; 8.12 añade 864 escenarios condicionales y referencias de MDE | No son potencia real o conjunta acreditada; requieren un diseño/piloto nuevo pertinente |
| Tablas A/B del original | Criterios y diagnósticos necesarios identificados en 8.6 | Pendientes de E3 con riesgos comparables, interpretación y fuentes; no se heredan de otras redes |
| Tabla por conjunto adicional y prueba consolidada | Especificación de selección en 8.3 e inventario preparatorio de tres candidatos en 8.12 | Cero admitidos/evaluados; faltan selección final y familia completa con controles aplicados a cada conjunto |
| Figuras y tablas de robustez/eficiencia/calidad | Matriz de estudios en 8.8 y antecedentes con su alcance | Faltan los estudios de la Tabla 5 que no cubren bloques/semillas o el piloto histórico |
| Discusión de resultados | Ganancias favorables y desenlaces adversos conservados | Explicar límites de capacidad, ruido, selección, variabilidad y potencia sin atribuir causas no demostradas |

La tabla maestra final debe integrar media±DE e IC95 % bootstrap de la métrica, parámetros, FLOPs, tiempos de entrenamiento/inferencia, hardware y alcance de cada coste. La tabla de hipótesis debe conservar prueba, nula, supuestos comprobados, estadístico, p crudo/ajustado, tamaño del efecto, IC y conclusión, incluidos casos no estimables. El análisis de potencia deberá reportar la dispersión piloto, semillas usadas, MDE y potencia conjunta; las ilustraciones de 8.7 no rellenan esos campos. Una discusión honesta de limitaciones y resultados adversos forma parte del reporte, sin asignar aquí una nota o tope académico.

El {download}`suplemento completo de métricas <../_static/e3_revision_bloque_d_20261117/metrics_extended.csv>` conserva número de predictores, calendario y estado de intervalos. Sus {download}`métricas por predictor <../_static/e3_revision_bloque_d_20261117/metrics_per_predictor.csv>` permiten verificar las medias. La {download}`tabla de incertidumbre de RMSE entre semillas <../_static/e3_revision_bloque_d_20261117/metrics_uncertainty.csv>` usa 10000 remuestras de las diez inicializaciones, con semilla fijada y los mismos índices para las redes. El IC95 % de E3 es **[0,0029132823; 0,0029684040]**, coincidente con el ya publicado para su estabilidad. Los clásicos, con una trayectoria, no reciben un IC de inicialización ficticio.

Estos son intervalos marginales de la media entre inicializaciones **condicionados al calendario conocido**: no describen incertidumbre temporal, error predictivo futuro o cobertura simultánea de la familia. El {download}`informe de fuentes, definiciones y reconciliación <../_static/e3_revision_bloque_d_20261117/metrics_supplement.json>` y su {download}`ejecutor del suplemento <../_static/e3_revision_bloque_d_20261117/metrics_supplement.py>` documentan el cálculo desde las trayectorias congeladas, sin nuevos ajustes, inferencias de mercado o contrastes de hipótesis.

El [workflow del suplemento](https://github.com/jassan-bit/DeepLearning_202630/blob/main/.github/workflows/verify-e3-block-d.yml) ejecuta nueve pruebas unitarias de las definiciones y la integridad, reproduce las ilustraciones de planificación y reconcilia las tablas con once insumos verificados por hash. También comprueba las copias publicadas y el borrador canónico. Su alcance es cálculo sobre pronósticos ya guardados; no entrena, calibra ni estima la potencia real del proyecto.

Las **42 pruebas de métodos e insumos aprobaron en integración continua**, con los archivos inmutables verificados y casos artificiales para los procedimientos numéricos. El {download}`registro de verificación <../_static/e3_estadistica_posterior_20261117/verification.json>` enlaza la ejecución de CI y distingue esa comprobación de una reproducción humana independiente.

Los {download}`resultados numéricos completos <../_static/e3_estadistica_posterior_20261117/result.json>`, el {download}`manifiesto de artefactos <../_static/e3_estadistica_posterior_20261117/manifest.json>` y el {download}`inventario de publicación <../_static/e3_estadistica_posterior_20261117/publication_manifest.json>` permiten comprobar las tablas. El código y sus pruebas se conservan en [experiments/e3_estadistica_posterior_20261117](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_estadistica_posterior_20261117).

Para reproducir, se instalan las versiones de `requirements.txt` y se ejecuta `python experiments/e3_estadistica_posterior_20261117/run.py --download --output work/e3_estadistica_reproduccion`; `report.py` exporta las tablas de lectura desde esa salida. Los datos se descargan desde commits inmutables y se verifican por SHA256. No se reentrena el modelo ni se requieren GPU o Google Drive.

La conclusión admisible es **evidencia retrospectiva favorable a la receta E3 bajo los procedimientos y supuestos declarados**. No se retrofecha el protocolo, no se oculta la diferencia de entrenamiento y no se transforma este análisis en certificación R1, R2 o R3.

(e3-suplemento-piso)=
## 8.10 Suplemento exploratorio: piso observable y proyección física

La auditoría del 18 de noviembre verifica las **143 trayectorias congeladas**, sus once insumos por SHA256 y la cohorte cruda auditada. Conserva los **17 sistemas y las 608 fechas**, sin seleccionar semillas o eliminar observaciones. Sólo realiza operaciones sobre entradas y pronósticos guardados: **cero ajustes, cero nuevos forwards y cero contrastes de hipótesis**. La proyección y el control sólo piso son comparadores exploratorios posteriores, separados del catálogo y de los 48 contrastes anteriores.

### Restricción conocida al emitir el pronóstico

La entrada de una fecha objetivo $t$ tiene siete cierres completos, desde $t-7$ hasta $t-1$. Proporciona seis retornos logarítmicos conocidos, cuya media es $\mu_6$; el retorno $r_t$ todavía no se conoce. La identidad para el objetivo poblacional, sin anualizar, es

$$
A_t=\frac17\sum_{i=t-6}^{t-1}(r_i-\mu_6)^2,
\qquad
\sigma_{7,t}^{2}=A_t+\frac6{49}(r_t-\mu_6)^2,
\qquad b_t=\sqrt{A_t}.
$$ (eq-e3-audit-observable-floor)

El divisor de $A_t$ es siete, aunque use seis retornos; $A_t$ es varianza y $b_t$ está en unidades de σ₇. El piso es alcanzable cuando el retorno desconocido coincide con la media conocida. Se calcula exclusivamente con los siete cierres pasados, sin usar $r_t$, el objetivo realizado o estadísticas ajustadas.

La transformación $p_t^+=\max(p_t,b_t)$ tiene una propiedad elemental: si $y_t\ge b_t$ y $p_t<b_t$, entonces

$$
(p_t-y_t)^2-(b_t-y_t)^2=(b_t-p_t)(2y_t-b_t-p_t)\ge0,
\qquad
|p_t-y_t|-|b_t-y_t|=b_t-p_t\ge0.
$$ (eq-e3-audit-projection-loss)

Por encima del piso la salida queda idéntica. Por ello la proyección no aumenta ninguna de esas pérdidas por observación cuando se cumple la premisa. No demuestra novedad, calibración o superioridad fuera de muestra; cuantifica una incoherencia física de las salidas respecto a este objetivo e historia conocidos.

### Resultado sobre la familia completa

Se encontraron **20.007 incumplimientos entre 86.944 pares predictor–fecha: 23,01 %**. Las trece redes E2 tienen salidas inferiores al piso; E3 y Persistence no tienen ninguna y sus pronósticos quedan idénticos. El conteo principal usa tolerancia absoluta de $10^{-12}$ en σ₇ y conserva también el conteo estricto. La proyección usa el máximo exacto. Las parejas solapadas y las semillas no constituyen observaciones de mercado independientes.

| Sistema de referencia | RMSE antes | RMSE tras proyección | MAE antes | MAE tras proyección |
|---|---:|---:|---:|---:|
| E3 aditiva, diez semillas | 0,0029394324 | 0,0029394324 | 0,0018928273 | 0,0018928273 |
| LSTM-autoencoder, mejor rival bruto | 0,0038728895 | 0,0035990849 | 0,0028608472 | 0,0025607201 |
| TSMixer, mejor rival proyectado | 0,0048215434 | 0,0032780684 | 0,0034346292 | 0,0020128791 |
| Persistence, trayectoria única | 0,0040855959 | 0,0040855959 | 0,0023531639 | 0,0023531639 |

Esta selección de lectura se acompaña de la {download}`tabla completa de los 17 sistemas <../_static/e3_remediacion_20261118/constraint_audit/results/per_model.csv>` y las {download}`143 filas por predictor <../_static/e3_remediacion_20261118/constraint_audit/results/per_predictor.csv>`, con tasas, magnitudes y cambios de rango. RMSE mantiene la media de las RMSE individuales. El máximo aumento observado de pérdida cuadrática y absoluta fue cero en todos los pares.

Al aplicar la misma proyección a todos los sistemas, la ventaja relativa RMSE de E3 frente al mejor rival disponible pasa de **24,10 % a 10,33 %**. La brecha absoluta disminuye de **0,0009334571 a 0,0003386360**, un **63,72 %** descriptivo. Cambia el rival de referencia entre ambas comparaciones; ese porcentaje no atribuye causalmente una fracción de la mejora a una pieza de E3. Muestra que una regla algebraica sencilla mejora las salidas rivales sin cambiar sus pesos y debe considerarse al aislar la contribución defendida.

El control adicional **predecir sólo $b_t$**, sin aprendizaje, tiene RMSE **0,0035275805**, un 20,01 % mayor que E3, pero MAE **0,0018227552**, un **3,70 % menor que E3**. Se conserva este resultado adverso. El control está fuera de la familia histórica y no recibe un contraste añadido retrospectivamente. La diferencia entre RMSE y MAE requiere estudiar errores y colas; estas cifras no identifican una causa.

### Comprobaciones y límites

Los 608 objetivos son idénticos a los del panel fijado y ninguno queda por debajo del piso. Se verifican geometría OHLCV, fechas consecutivas y solapamiento cronológico exacto de seis barras. El piso recalculado coincide con el guardado por E3 y otra identidad por diferencias entre pares verifica la varianza. Un bloque separado reconstruye **607 etiquetas** con el cierre de la ventana siguiente, con discrepancia máxima **1,2577×10⁻¹⁶** en σ₇: ese cierre futuro sólo audita etiquetas y nunca entra en el cálculo del piso o la proyección. No existe una ventana siguiente para esa reconstrucción del último objetivo, cuya admisión conserva el respaldo del NPZ y panel verificados.

El {download}`ejecutor <../_static/e3_remediacion_20261118/constraint_audit/run_audit.py>`, sus {download}`fuentes y admisión <../_static/e3_remediacion_20261118/constraint_audit/results/sources_and_admission.json>`, el {download}`piso por fecha <../_static/e3_remediacion_20261118/constraint_audit/results/floor_by_date.csv>` y el {download}`resumen numérico <../_static/e3_remediacion_20261118/constraint_audit/results/summary.json>` permiten reproducir el diagnóstico. Las {download}`nueve pruebas numéricas <../_static/e3_remediacion_20261118/constraint_audit/test_receipt.json>` verifican unidades, orientación, divisor, futuro excluido, estimando y premisa de la proyección. Heredan el contrato temporal de la cohorte auditada y no cierran sus tres controles parciales. La {download}`descripción íntegra <../_static/e3_remediacion_20261118/constraint_audit/README.md>` conserva los comandos y límites de originalidad y reproducción.

(e3-suplemento-calibracion)=
## 8.11 Suplemento exploratorio: calibración posterior de E3

La campaña original prohibió abrir la calibración y registró `calibration_members_opened:false`. El diagnóstico del **18 de noviembre** verifica y conserva ese historial, pero abre esas ventanas **en una etapa posterior identificada como tal**. Calcula diez cuantiles nuevos, sin reentrenar ni modificar pronósticos puntuales. El calendario de evaluación ya era conocido; el {download}`plan de este diagnóstico <../_static/e3_remediacion_20261118/calibration/analysis_plan.json>` no es un preregistro confirmatorio. **En aquel corte los cuantiles no se activaron en la aplicación**. La ampliación posterior del [capítulo 9](09_dashboard_predictivo.md) adopta sólo el del representante 271801, con metadatos separados y restricción temporal; la cobertura individual del predictor servido no es la media de diez semillas descrita aquí.

Se usan las diez semillas 271801–271810 y sus `best_model_state` congelados, con bytes y SHA256 verificados. Se conservan el escalador de los **532 ejemplos de ajuste** y la selección original con **116 de validación**. Las **53 ventanas de calibración**, de 2024-11-08 a 2024-12-30, se transforman con ese escalador E3; no se reutilizan tensores escalados por E2 ni se actualizan estadísticas. Se ejecutan **530 forwards CPU nuevos de calibración** y cero forwards de evaluación. Para evaluar los intervalos se reutilizan las **6080 predicciones puntuales guardadas** sobre las 608 fechas conocidas.

Con nominal 90 %, cada predictor utiliza el residuo absoluto de rango **49 de 53**, porque `ceil((53+1)×0,90)=49`, sin interpolación. Su intervalo es `[max(0,predicción−q),predicción+q]`, con el cuantil de ese mismo predictor. No se elige la semilla de mejor cobertura ni se forma un ensamble. Además de cobertura y anchura, se calcula el score de intervalo a α=0,10, que penaliza anchura y observaciones fuera de ambos extremos; menor es mejor.

| Resultado posterior E3 | Media de los diez predictores | Alcance |
|---|---:|---|
| Cobertura empírica | **94,4901 %** | 608 fechas históricas; nominal 90 % |
| Anchura media | **0,01096344 σ₇** | Intervalos individuales, sin ensamble |
| Score de intervalo, α=0,10 | **0,01536077 σ₇** | Regla propia que considera anchura y fallos |
| RMSE puntual | **0,0029394324 σ₇** | Idéntica a la campaña original |

La cobertura agregada no se mantiene uniformemente en el tiempo:

| Diagnóstico posterior | Fechas por predictor | Cobertura media |
|---|---:|---:|
| 2025Q1 | 90 | 95,00 % |
| 2025Q2 | 91 | 93,41 % |
| 2025Q3 | 92 | 97,72 % |
| 2025Q4 | 92 | 96,20 % |
| **2026Q1** | **90** | **86,22 %** |
| 2026Q2 | 91 | 97,36 % |
| 2026Q3, julio–agosto | 62 | 95,81 % |
| Objetivo superior al q75 original de ajuste | 86 | **84,53 %** |

Todos los fallos de cobertura ocurrieron por el extremo superior. El umbral q75 procede del ajuste original, pero asignar ese último grupo utiliza el **objetivo ya realizado**: describe fallos retrospectivos y no es una regla disponible para dirigir pronósticos futuros. Ningún grupo modificó q o la selección de semillas.

El bootstrap circular pareado de bloques de 14 días, con 4999 remuestras, da IC95 % marginal **[92,4342 %; 96,3158 %]** para cobertura media y **[0,01313811; 0,01826591] σ₇** para el score. Resamplea conjuntamente los días de las diez semillas y condiciona en los modelos y cuantiles ya estimados: omite la incertidumbre de estimar q con 53 ventanas y no proporciona intervalos simultáneos.

Los objetivos de calibración y prueba son disjuntos y ordenados, pero sus historias móviles se solapan: la primera entrada de enero contiene 2024-12-25–31 y la calibración termina el 30 de diciembre. Son datos conocidos al pronosticar; la dependencia y el cambio temporal no acreditan intercambiabilidad. **La corrección de rango finito y la cobertura global de 94,49 % no garantizan cobertura nominal del 90 % en periodos o casos futuros.** El descenso en 2026Q1 conserva esa limitación de forma observable.

Se publican los {download}`cuantiles y métricas por semilla <../_static/e3_remediacion_20261118/calibration/results/per_seed_intervals.csv>`, los {download}`530 pronósticos de calibración <../_static/e3_remediacion_20261118/calibration/results/calibration_predictions.csv>`, los {download}`diagnósticos de subgrupos <../_static/e3_remediacion_20261118/calibration/results/subgroup_diagnostics.csv>` y el {download}`informe completo <../_static/e3_remediacion_20261118/calibration/results/results.json>`. El {download}`ejecutor <../_static/e3_remediacion_20261118/calibration/run.py>` y el {download}`verificador de resultados <../_static/e3_remediacion_20261118/calibration/verify_results.py>` distinguen el cálculo nuevo de la reconstrucción sin forwards adicionales. Su {download}`recibo de verificación <../_static/e3_remediacion_20261118/calibration/results/verification.json>` comprueba quince entradas, ocho matrices originales, precisiones, cuantiles y métricas; no acredita reproducción humana independiente. La {download}`documentación del diagnóstico <../_static/e3_remediacion_20261118/calibration/README.md>` mantiene comandos, pruebas y limitaciones.

(e3-preparacion-potencia-datos)=
## 8.12 Preparación ejecutada: potencia condicional y admisión de datos

Esta etapa verifica insumos y calcula **864 escenarios de planificación: 432 por semillas y 432 temporales**. No inicia una campaña, no estima potencia observada del resultado, no convierte el test histórico en piloto prospectivo y no acredita potencia conjunta real. Tampoco ajusta modelos, ejecuta inferencias, recalibra o añade contrastes. El {download}`informe de dimensionamiento <../_static/e3_remediacion_20261118/evaluation/results/power_planning.json>` conserva todas las alternativas y sus supuestos.

### Escenarios de semillas y calendario

Se recupera la DE de las ganancias pareadas de RMSE a partir de las diez inicializaciones sobre las 608 fechas conocidas. Frente a un clásico, su trayectoria permanece única: la dispersión de diez ganancias procede del candidato, no de diez reentrenamientos ficticios del rival. La planificación adopta alternativas declaradas `g₁=1,25δ`, `1,5δ` y `2δ`; la distancia frente a la nula es **g₁−δ**, y no se utiliza la ganancia media observada como alternativa.

Para el componente t por semillas se usa la distribución t no central, bajo ganancias pareadas normales e independientes y una DE futura igual a la asumida. El umbral unilateral conservador es `0,05/48`, como referencia para los 48 componentes de Holm; no se calcula la potencia exacta de su procedimiento adaptativo. Con alternativa `g₁=2δ`, los máximos de pares requeridos sobre los 16 rivales son:

| Meta marginal por procedimiento | DE histórica | Cota superior simultánea de DE | Sensibilidad 2×DE |
|---|---:|---:|---:|
| 0,90 | 280 | 1614 | 1103 |
| 0,9666667 | 354 | 2048 | 1399 |
| 0,9979167 | **515** | 2996 | 2046 |

TiDE determina estos máximos. La cota superior de DE usa teoría normal χ² con cola `0,05/16`; con diez pares la incertidumbre de esa dispersión es material. **515 pares sólo satisfarían la meta marginal del componente t bajo el escenario y los supuestos declarados**. No son el tamaño suficiente de la batería completa, una potencia comprobada del proyecto o una recomendación de entrenar cientos de semillas.

Si las 48 decisiones alcanzaran potencia marginal de al menos 0,9979167, la desigualdad de la unión daría una cota conjunta de 0,90 sin exigir independencia. Aquí no se han dimensionado ni validado todos los procedimientos; la potencia conjunta real continúa abierta. Las correlaciones históricas de ganancias y los productos de potencias bajo independencia hipotética conservan un significado descriptivo o ilustrativo.

La referencia temporal se calcula por separado, con influencia del RMSE, HAC Bartlett de rezagos 7/14/28 y aproximación normal. HAC14 con `g₁=2δ` y objetivo marginal 0,90 da una extrapolación máxima de **420.677 fechas**, determinada por SeasonalNaive. Esa cifra extrema no es un horizonte recomendado: prolonga supuestos de estacionariedad y dispersión muy lejos de las 608 fechas y aplica la misma alternativa cercana a la frontera a rivales muy diferentes. Indica que hacen falta un piloto nuevo, alternativas y varianzas justificadas y un diseño viable antes del congelamiento. Aumentar semillas no crea periodos de mercado independientes.

Las {download}`432 filas de planificación por semillas <../_static/e3_remediacion_20261118/evaluation/results/seed_planning.csv>`, las {download}`432 referencias temporales <../_static/e3_remediacion_20261118/evaluation/results/calendar_planning.csv>` y las {download}`dispersiones de entrada <../_static/e3_remediacion_20261118/evaluation/results/seed_variance_inputs.csv>` permiten revisar todos los rivales. El {download}`ejecutor de planificación <../_static/e3_remediacion_20261118/evaluation/power_planning.py>` y el {download}`recibo de reproducción automatizada <../_static/e3_remediacion_20261118/evaluation/results/verification.json>` conservan nueve pruebas numéricas y reconstrucción idéntica de los productos; no certifican potencia real o reproducción humana independiente.

### Tres candidatos, cero conjuntos admitidos

**Encabezado y expediente del corte preparatorio inicial.** Se conserva su ancla y su estado histórico; la admisión derivada posterior de D01 se documenta en 8.13. La admisión de D01 fue un corte intermedio; el estado actualizado 3/3 y los entrenamientos parciales se documentan en 8.17.

La búsqueda preparatoria inicial identificó los siguientes candidatos por metadatos, procedencia y términos. No descargó archivos de mercado, entrenó modelos o observó resultados de rendimiento de esos candidatos **en esa búsqueda**. La descarga y admisión posterior de D01 tienen su propio registro y no modifican retrospectivamente aquel expediente.

| Candidato | Diferencias propuestas frente a BTC Binance | Bloqueos de admisión |
|---|---|---|
| WIKI/AAPL diario, periodo propuesto 2000–2013, mediante Bokeh | Fuente, acción frente a cripto, calendario y sesiones bursátiles | En el inventario inicial faltaban archivo/hash, derechos, calendario y política. Esos controles se cerraron posteriormente para D01 RAW entre sesiones en 8.13; la evaluación de modelos permanece pendiente. |
| HF Data Library / IEX SPY, V2 Clean, minuto a sesión diaria | Fuente/venue, ETF frente a cripto y reloj de sesiones | CC BY 4.0 de compilación no sustituye derechos upstream de IEX; acceso, archivo, limpieza causal y agregación sin verificar |
| Zielak / Bitstamp BTCUSD, minuto a día UTC, calendario anterior propuesto | Exchange/fuente, calendario y USD frente a USDT | Tarjeta indexada declara CC BY-SA 4.0, no CC0; faltan derechos upstream, versión, cobertura y agregación UTC verificadas |

**En ese corte inicial los tres se registraron como no admitidos.** La revisión posterior adopta explícitamente para D01 σ₇ de siete retornos entre sesiones y horizonte próxima sesión: una tarea financiera homóloga, distinta de siete días UTC consecutivos. No se inventan velas en fines de semana o festivos. La admisión no afirma un reloj UTC idéntico ni acredita por sí sola la diversidad y superioridad exigidas por R3; el protocolo común deberá respetar el contrato de cada conjunto. D02 y D03 siguen pendientes.

El {download}`expediente de admisión <../_static/e3_remediacion_20261118/evaluation/dataset_admission.md>` y el {download}`inventario estructurado <../_static/e3_remediacion_20261118/evaluation/dataset_candidates.json>` conservan fuentes, accesos observados y bloqueos exactos. El {download}`informe de preparación de campaña <../_static/e3_remediacion_20261118/evaluation/results/campaign_readiness.json>` audita las 17 identidades y once puntos de entrada por inspección sintáctica; no ejecuta las recetas ni demuestra que todas sean operativas en el entorno actual.

Antes de cualquier confirmación siguen pendientes candidato y mecanismo defendidos, datos admitidos, particiones y escaladores comunes, política de calibración, HPO o recetas fijas con presupuesto comparable, piloto pertinente, potencia de la batería y protocolo fechado. El entrenamiento histórico E2 de 655 ejemplos frente a 532 de E3 conserva su diferencia. La {download}`documentación de preparación <../_static/e3_remediacion_20261118/evaluation/README.md>` describe la reproducción y estos límites. Los tres suplementos añaden evidencia delimitada y resultados adversos; **R3 permanece sin acreditar**.


### Cuaderno preparado para Colab con GPU

El usuario dispone de Colab con GPU. El nuevo [cuaderno de preflight](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_remediacion_preflight_colab.ipynb) prepara un entorno CUDA aislado y comprueba **los 17 perfiles reales con datos artificiales**. La comprobación no modifica el Torch del kernel de Colab ni descarga observaciones de mercado. La versión con persistencia usa un secreto GitHub para guardar sus recibos. Sus descargas de paquetes pueden requerir varios GB y varios minutos. La primera celda fija y registra las fuentes; se selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**, y se ejecuta el cuaderno completo.

La comprobación local **CPU** pasó los 17 perfiles: 14 redes con forward, gradiente finito y recarga idéntica; ARIMA con un ajuste exclusivamente sintético; persistencia y estacional con rezagos artificiales. Posteriormente, la persona autora compartió una salida Colab con **17/17 perfiles aprobados en NVIDIA L4**, verificación global GPU verdadera y código 0, sobre el commit `530479e4dbc4bc60be7fb363291a0f3420c5e7ac`. El ZIP de esa sesión L4 se perdió; su consola conserva alcance de evidencia reportada. La nueva ejecución A100, distinta de ese antecedente, ya dispone de recibos guardados y auditados en el apartado siguiente. El {download}`cuaderno descargable <../_static/e3_remediacion_20261118/evaluation/e3_remediacion_preflight_colab.ipynb>`, la {download}`documentación del preflight <../_static/e3_remediacion_20261118/evaluation/colab_preflight_README.md>` y el {download}`recibo CPU <../_static/e3_remediacion_20261118/evaluation/results/colab_cpu_preflight.json>` delimitan esa evidencia. Estos controles no comparan desempeño ni ejecutan entrenamientos con datos reales. La pasada GPU no cierra las admisiones: tras la revisión posterior D01 queda admitido como archivo, D02/D03 siguen pendientes y el protocolo y el tratamiento/presupuesto/potencia finales continúan abiertos. Los recibos históricos conservan el estado que existía al ejecutarlos.


### Ejecución GPU reportada y corrección del recibo

La {download}`observación estructurada <../_static/e3_gpu_colab_20261118/observation.json>` y la {download}`transcripción normalizada <../_static/e3_gpu_colab_20261118/console_transcript.txt>` registran la salida aportada por la persona autora: **NVIDIA L4**, driver 580.82.07, Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6. El resultado reportado es `passed_synthetic_only`, 17/17 y verificación global GPU verdadera. No se reemplazan los recibos CPU anteriores ni se presenta una nueva ejecución GPU realizada por el asistente.

En esa versión, `environment.GPU_execution_verified=false` pertenecía a la inspección inicial del hardware y no se actualizaba al terminar los modelos. La marca global en `true` describía la pasada final. La corrección elimina el campo anidado y exige, para la verificación global, los 14 recibos neuronales en CUDA con backward y recarga idéntica, además de los 17 perfiles aprobados. El {download}`ejecutor corregido <../_static/e3_gpu_colab_20261118/colab_preflight.py>`, la {download}`prueba de regresión <../_static/e3_gpu_colab_20261118/test_colab_preflight.py>` y el {download}`cuaderno actual <../_static/e3_gpu_colab_20261118/e3_remediacion_preflight_colab.ipynb>` conservan ese cambio de reporte sin modificar los métodos científicos.

El {download}`alcance de la revisión <../_static/e3_gpu_colab_20261118/README.md>` distingue salida compartida de archivo auditado. **El ZIP del antecedente L4 no se recuperó**: para esa sesión no hay SHA-256 del paquete ni validación de casos individuales. Esa limitación histórica no describe el nuevo paquete A100, que sí se guardó y auditó por separado. Ese corte conservaba cero archivos adicionales admitidos; la decisión de 8.13 llevó en aquel corte la admisión a uno de tres archivos, todavía sin evaluación por modelos. El cierre posterior de 8.18 completó los tres y los 796 trabajos de la aditiva; no resuelve la confirmación de V3 ni su potencia prospectiva. Pasar los contratos sintéticos no completa R1/R3 ni la reproducción humana independiente.


### Guardado automático de Colab en GitHub

Tras la pérdida de los archivos temporales de la sesión L4, el usuario solicitó guardar los resultados en GitHub. El cuaderno actual incorpora ese destino: recibos JSON, logs, ZIP y manifiesto de hashes quedan en `runs/colab_preflight/<RUN_ID>/`, dentro de una rama nueva `e3-colab-receipts-<RUN_ID>` del repositorio. El guardado usa un commit atómico, comprueba los bytes remotos y reconoce un reintento idéntico. El código y las evidencias anteriores conservan sus versiones.

Para escribir desde Colab se necesita **GITHUB_TOKEN**, con **Contents: Read and write** limitado a `jassan-bit/DeepLearning_202630`, guardado en Secretos o introducido en un campo oculto. La consulta previa comprueba acceso al repositorio; no certifica por sí sola el alcance de escritura de un token fino. El token se mantiene en memoria y se excluye del entorno científico y los archivos publicados.

El {download}`cuaderno con guardado en GitHub <../_static/e3_github_persistence_20261118/e3_remediacion_preflight_colab.ipynb>`, el {download}`módulo de persistencia <../_static/e3_github_persistence_20261118/colab_github.py>`, sus {download}`pruebas de contrato <../_static/e3_github_persistence_20261118/test_colab_github.py>` y las {download}`instrucciones <../_static/e3_github_persistence_20261118/README.md>` describen el flujo. Al terminar se imprime una URL GitHub para revisar la ejecución. Si falla la escritura, se corrige el secreto y se repite sólo el guardado, con los mismos resultados locales.

**La primera ejecución con este flujo ya está guardada y auditada: A100, run `20261118T172624131741Z`.** El ZIP de la sesión L4 anterior no se recuperó; su consola se mantiene como un antecedente distinto, sin reetiquetarla como la ejecución nueva. En aquel corte, guardar recibos sintéticos no habilitaba la campaña real: estaban pendientes D02/D03, protocolo/tag, tratamiento, presupuesto y potencia. La ejecución exploratoria posterior se documenta en 8.17; no acredita por sí sola R3.


### Primera ejecución A100 guardada y auditada

La ejecución `20261118T172624131741Z` completó **17/17 perfiles sintéticos** y guardó sus [resultados en un commit inmutable](https://github.com/jassan-bit/DeepLearning_202630/tree/47d2f316a50d9268cb6484a547e8a2f0cc0413eb/runs/colab_preflight/20261118T172624131741Z), `47d2f316a50d9268cb6484a547e8a2f0cc0413eb`, cuyo padre es el código ejecutado `1ff136f71c0ffcd1dbd918efbc078a8d9653ed21`. Los 14 recibos neuronales registran CUDA, backward y recarga idéntica; declaran 456 tensores de gradiente finitos. Los tres comparadores clásicos se ejecutaron en CPU, incluido un único ajuste ARIMA artificial. El resultado global es `passed_synthetic_only`; ajustes, predicciones y contrastes nuevos de mercado permanecen en cero.

La auditoría verificó el ZIP de **43.860 bytes**, SHA-256 `44b7fb93a3f9f667bbe3cd7b6a8d4b1a190c0742ac62e78c0dbd127dfa62fd31`: sus **40 miembros** coinciden exactamente con los archivos sueltos. El manifiesto enumera **41 artefactos**, incluido el ZIP; con el propio `storage_manifest.json` son **42 archivos persistidos**, sin discrepancias de bytes. También se cotejaron las siete fuentes del contrato, el preparador de entorno, el lock de 43 paquetes y el ejecutor. El {download}`informe de auditoría <../_static/e3_a100_receipts_20261118/audit.json>`, la {download}`tabla por perfil <../_static/e3_a100_receipts_20261118/per_model.csv>` y la {download}`auditoría de fuentes <../_static/e3_a100_receipts_20261118/source_audit.json>` delimitan esas comprobaciones.

El {download}`ZIP preservado <../_static/e3_a100_receipts_20261118/receipts.zip>`, el {download}`manifiesto de guardado <../_static/e3_a100_receipts_20261118/storage_manifest.json>`, el {download}`verificador <../_static/e3_a100_receipts_20261118/verify.py>` y la {download}`documentación del alcance <../_static/e3_a100_receipts_20261118/README.md>` permiten repetir la auditoría de archivos. **El ZIP no contiene los pesos ni los tensores originales**: no se recalcularon gradientes o salidas GPU ni se reprodujo numéricamente la recarga; se verificaron los recibos y su coherencia con la fuente hasheada. Esta revisión no es reproducción humana independiente ni acredita R1/R3. En el corte A100 había cero archivos adicionales admitidos. Tras aquella revisión de D01, el corte pasó a uno de tres admitido, cero evaluados por modelos y dos pendientes. La ejecución exploratoria posterior de 8.17 ya tiene admisión 3/3, protocolo con tag y resultados parciales; la potencia completa para R3 sigue pendiente.


(d01-admission-session-raw)=
## 8.13 D01 admitido para σ₇ RAW entre sesiones

**Corte inicial de admisión D01: 1 de 3 archivos admitidos, 0 evaluados por modelos y 2 pendientes.** La actualización posterior con admisión 3/3 y entrenamiento real está en 8.17. D01 es el extracto WIKI/AAPL de Bokeh, 3270 sesiones del 1 de marzo de 2000 al 1 de marzo de 2013, admitido para una tarea definida de precio cotizado bruto: desviación poblacional —divisor siete— de siete retornos logarítmicos entre sesiones, con horizonte próxima sesión. La [ejecución CPU del usuario](https://github.com/jassan-bit/DeepLearning_202630/commit/648f2b91b477b5f4226cad3af5928088accd74ed), run `20261118T181551067867Z`, conserva como código ejecutado [`568aee0445a6eaba0f13c11deb2d6d64aa4f3fbd`](https://github.com/jassan-bit/DeepLearning_202630/commit/568aee0445a6eaba0f13c11deb2d6d64aa4f3fbd). Su JSON original registró cero admitidos; **no se modifica ni se presenta como si el usuario hubiera reportado uno**. El cambio a uno es una decisión derivada posterior, después de verificar calendario y fijar la política de datos; tampoco altera los cortes inicial, CPU, L4 o A100.

La fuente queda fijada al commit Bokeh [`3e831d2779ca74c087c7eee05d538015c97b1e52`](https://github.com/bokeh/bokeh_sampledata/tree/3e831d2779ca74c087c7eee05d538015c97b1e52), archivo `stocks.csv.gz`, SHA-256 `5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd`. Su inventario declara dominio público para los datos WIKI y documenta la procedencia; la licencia BSD del software no se usa para relicenciar los datos. El calendario se contrastó con `exchange-calendars==4.11.3`, fuente [`79a4c2cbeed5a16adee5d013edd76f5c8c82d656`](https://github.com/gerrymanoim/exchange_calendars/tree/79a4c2cbeed5a16adee5d013edd76f5c8c82d656): `NASDAQ`/`XNAS` son alias publicados de `XNYS`, no comprobaciones independientes. Coinciden **3270 de 3270 fechas en orden, sin sesiones faltantes ni adicionales**, incluidos los cierres de 9/11, Reagan, Ford y Sandy. Son etiquetas de sesión, no marcas horarias UTC de ejecución; no se rellena ni inventa una vela para días sin negociación.

La política **`RAW_KEEP_ALL`** conserva todas las filas y los OHLCV sin ajustar y excluye `Adj Close` tanto de entradas como de etiquetas. No se corrigen precios por splits, no se eliminan saltos ni se seleccionan etiquetas después de observar resultados de modelos. Los efectos mecánicos de acciones corporativas y los movimientos de mercado permanecen en la serie: **σ₇ describe volatilidad del precio cotizado bruto, no volatilidad económica ajustada por splits/dividendos**. El contrato de sesiones es una adaptación explícita y no se presenta como el reloj diario UTC continuo de BTCUSDT. La {download}`decisión de admisión <../_static/e3_d01_admission_20261118/admission_decision.json>`, la {download}`política RAW <../_static/e3_d01_admission_20261118/policy.md>` y la {download}`documentación del alcance <../_static/e3_d01_admission_20261118/README.md>` fijan esa definición.

La {download}`comprobación de calendario <../_static/e3_d01_admission_20261118/calendar_check.json>`, las {download}`3270 sesiones de referencia <../_static/e3_d01_admission_20261118/NASDAQ_sessions_20000301_20130301.csv>`, la {download}`auditoría de bytes del registro <../_static/e3_d01_admission_20261118/byte_audit.json>` y el {download}`verificador <../_static/e3_d01_admission_20261118/verify.py>` permiten revisar el cierre del archivo. **Admitir D01 no requiere declarar ya cerrado el protocolo MODEL ni habilita entrenamiento.** Los entrenamientos históricos existen y se conservan; esta etapa no añade ajustes, predicciones de mercado ni contrastes de modelos. En ese corte, la campaña futura estaba bloqueada hasta resolver D02/D03 y fijar protocolo, particiones, tratamiento, presupuesto y potencia. La auditoría exploratoria posterior de 8.17 conserva ese historial y el dimensionamiento de potencia aún pendiente. Ese protocolo podrá justificar ajustes futuros; no se reentrena por rutina. La admisión de un archivo no acredita evaluación ampliada, reproducción humana independiente, R1/R3, DOI o aceptación pública de Render.

## 8.14 Admisión física preparada para dos conjuntos adicionales

El nuevo recolector sustituye los candidatos de metadatos aún no admitidos **antes de observar rendimiento de modelos**. Conserva D01 WIKI/AAPL RAW de 8.13 y añade estas especificaciones:

| Identificador | Fuente y periodo fijados | Unidad y tamaño esperado | Estado observado |
|---|---|---|---|
| D01 | WIKI/AAPL, 2000-03-01–2013-03-01. | 3270 sesiones; volumen en acciones. | Archivo y calendario admitidos; cero modelos evaluados. |
| D02 | Binance Vision Spot ETHUSDT, 2018-01-01–2021-12-31. | 1461 días UTC; volumen en ETH. | 48 ZIP y CHECKSUM mensuales por verificar en Colab. |
| D03 | Binance Vision COIN-M BTCUSD_PERP, 2020-09-01–2024-12-31. | 1583 días UTC; volumen en **contratos**, no BTC. | 52 ZIP y CHECKSUM mensuales por verificar en Colab. |

La selección aporta acciones cotizadas, un criptoactivo al contado y un derivado con margen en moneda. **No son tres mercados independientes o globalmente ciegos:** D02/D03 comparten distribuidor y factores cripto; D03 comparte subyacente BTC y fechas 2023–2024 ya conocidas en el ajuste histórico. D01 usa sesiones bursátiles, distintas de días UTC. Estas dependencias deben incorporarse al protocolo y al alcance del veredicto; cambiar de archivo no las elimina.

El {download}`recolector CPU <../_static/e3_cierre_integral_20261118/data/collect.py>` verifica SHA-256 oficial de cada ZIP, nombres seguros, exactamente doce columnas de origen, cierres UTC, milisegundos/microsegundos, OHLCV coherentes y calendario completo sin duplicados o relleno. D01 se vuelve a verificar contra sus bytes y calendario fijados; se conserva `RAW_KEEP_ALL`. El caché mantiene los primeros bytes observados y permite reanudar descargas; una fuente cambiada exige una salida nueva, sin sobrescribir silenciosamente la anterior.

Las {download}`fuentes y derechos <../_static/e3_cierre_integral_20261118/data/source_provenance.json>` fijan `binance/binance-public-data` al commit `d571690eb66760763b0bd6e27e8cd35603c7e244`. Sus términos 1.0, de 26 de agosto de 2026, especifican **CC BY-NC-SA 4.0** e incluyen derivados como modelos, indicadores y gráficos. Su alcance no se convierte en MIT o CC0 y no se traslada automáticamente a la cohorte histórica obtenida por otra vía. La licencia propia del código y los derechos de datos son decisiones distintas.

Las {download}`nueve pruebas offline <../_static/e3_cierre_integral_20261118/data/local_verification.json>` incluyen los 3270 registros físicos de D01, calendarios y rechazos de archivos malformados. Los sondeos de D02/D03 en este entorno devolvieron 403 del proxy: **no hay todavía recibo de admisión de esos dos conjuntos ni evaluación ampliada nueva**. La ejecución real de Colab deberá conservar archivos, respuestas y `admission.json`; tres admisiones físicas tampoco equivalen a tres evaluaciones de modelos.

## 8.15 Piloto común de desarrollo preparado para GPU

La [propuesta V3](06_modelo_original.md) requiere ajustes nuevos de desarrollo para evaluar una modificación que aún no tiene resultados. Se conservan todos los entrenamientos históricos. El ejecutor nuevo utiliza únicamente los miembros de entrenamiento de la cohorte fijada, con **532 ventanas de ajuste, 116 de validación y siete de purga**; el escalador se estima sólo en ajuste. No abre los miembros de calibración o las 608 ventanas de prueba para seleccionar esta propuesta.

La primera ejecución compara V3 con CRPS, V3 sólo MSE, probabilidades fijas, aditiva histórica y M00 proyectado, además de controles analíticos. El modo ampliado contiene **18 sistemas**: Top 10, seis referencias, aditiva histórica y V3. Mantiene a la aditiva como rival; no se retira un competidor fuerte al cambiar de candidato. SAE conserva sus fases de preentrenamiento y ajuste supervisado, y ARIMA pronostica causalmente antes de incorporar el objetivo observado. Los checkpoints registran pesos, optimizador y RNG para reanudar; la igualdad entre plataformas no se presume.

La coincidencia de particiones, preprocesamiento y criterio de validación mejora la comparación del piloto. **Todavía faltan una política final común de HPO/presupuesto y el protocolo confirmatorio**, así como las evaluaciones y estudios de robustez en los conjuntos adicionales. Una victoria de desarrollo no acredita R3 ni recupera el cegamiento del BTC histórico.

## 8.16 Potencia condicional y persistencia de las siguientes ejecuciones

El planificador nuevo sólo admite un panel completo del piloto común de 18 sistemas y al menos diez semillas emparejadas. Para planificación modela tres procedimientos unilaterales sobre diferencias de RMSE entre semillas —t emparejada, Wilcoxon y signo exacto— y aplica **Holm a los 51 valores p de la familia completa**. La decisión conjunta exige el rechazo de los tres procedimientos para cada uno de los 17 rivales. Los procedimientos están relacionados y su interpretación de localización común depende del modelo de planificación; no son tres réplicas independientes de evidencia temporal.

La simulación y sus sensibilidades estiman potencia condicional bajo supuestos y ganancias de desarrollo observadas. **Un piloto de controles incompleto no dimensiona la batería; tampoco certifica potencia de nuevos dominios, validez temporal o superioridad futura.** El código no congela automáticamente el protocolo ni convierte resultados conocidos en un preregistro. Semillas, margen de relevancia, presupuesto, familia final y reglas entre conjuntos deberán fijarse antes de la campaña confirmatoria.

Los [cuadernos de CPU y GPU](https://github.com/jassan-bit/DeepLearning_202630/tree/main/notebooks) fijan el SHA del código y guardan las evidencias en ramas nuevas. La etapa CPU recoge admisiones y prueba Render; la etapa GPU ejecuta desarrollo y conserva snapshots de los trabajos, sus checkpoints y el panel completo. Los {download}`contratos de persistencia <../_static/e3_cierre_integral_20261118/orchestration/storage_verification.json>` comprueban preservación del árbol fuente, bytes, reintentos, conflictos y rechazo de secretos incluso en ZIP/PT/NPZ comprimidos. Los recibos siguen declarando `R1_accredited=false` y `R3_accredited=false`; registrar una ejecución no acredita sus conclusiones científicas.

## 8.17 Auditoría ampliada ejecutable y estudios de robustez

Esta sección conserva el método y **el corte histórico de la parte 46, con 251 trabajos completos sobre D01 y D02**. En aquel corte faltaba el análisis final. La continuación posterior completó los cuatro conjuntos, el análisis y las figuras: véase [8.18](#e3-final-extended-audit). Los tres archivos adicionales tienen admisión pública y el protocolo exploratorio se fijó con tag antes de los ajustes. El [cuaderno Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_evaluacion_ampliada_robustez_colab.ipynb) reúne admisión, preespecificación, ajustes, evaluación, análisis, figuras y continuación desde GitHub. La prueba de Render tiene su propio flujo.

### Comparación y datos fijados antes de evaluar

La familia contiene **19 sistemas**: los 17 históricos —incluido el candidato aditivo de 1210 parámetros— y dos controles analíticos, `floor_only` y `floor_plus_mean_excess`. Este último estima un exceso medio sólo en ajuste; tiene un parámetro estadístico, sin optimizador neuronal. Se conservan los pronósticos crudos y se aplica **la misma proyección `max(pronóstico, sqrt(A))` a todos** como comparación principal. Así el piso físico no se ofrece exclusivamente al candidato. V3 continúa en una investigación de desarrollo separada.

| Conjunto | Contrato de la auditoría | Estado en la parte 46, conservado como histórico |
|---|---|---|
| D00 BTC histórico | Cohorte NPZ fijada; 532 ajuste, 116 validación, 53 calibración y 608 prueba conocida. Historia diaria original para los clásicos. | Los resultados anteriores existen; los nuevos ajustes bajo política común aún no se han ejecutado. |
| D01 WIKI/AAPL RAW | Siete observaciones entre sesiones; `Adj Close` excluido y saltos corporativos conservados. | 3270 filas admitidas y 19 sistemas completos: 199 registros de trabajo. |
| D02 ETHUSDT Spot | Días UTC, 2018–2021; volumen en ETH. | 1461 días admitidos; FCN, ResNet, Inception y SAE completos: 52 registros. TimesNet tiene un HPO iniciado pendiente. |
| D03 BTCUSD_PERP COIN-M | Días UTC, septiembre de 2020–2024; volumen en contratos. | 1583 días admitidos y persistidos, incluida recuperación completa de agosto de 2023; ajustes nuevos pendientes. |

La [admisión Colab `20261119T022547490732Z`](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-closure-cpu-20261119T022547490732Z/runs/e3_closure_cpu/20261119T022547490732Z) conserva el primer corte de **2/3 archivos admitidos y cero ajustes nuevos**. El ZIP D03 de agosto de 2023 coincide con su checksum oficial, pero sólo contiene las observaciones del 1 al 27; faltan los días 28–31. El {download}`diagnóstico de sus bytes <../_static/e3_evaluacion_ampliada_20261119/admission/incident_202308.json>` conserva hashes y procedencia. La corrección incorpora la recuperación del mes completo mediante ZIP diarios oficiales y checksums, con calendario exacto y coincidencia de los 27 días comunes. El recuperador sólo admite D03 cuando verifica todo el mes; el cierre público posterior se identifica a continuación. No se imputan observaciones, se acorta el periodo o se altera el recibo fallido.

La consola de la ejecución siguiente, `20261119T024543618052Z`, informa **3/3 admitidos**, después de completar los 52 meses de D03 con el recuperador. Su guardado se interrumpió en `POST /git/blobs`: GitHub devolvió un límite secundario con `Retry-After: 60` y 4546 solicitudes ordinarias restantes. El {download}`registro de esa incidencia <../_static/e3_evaluacion_ampliada_20261119/admission/storage_secondary_rate_limit_20261119.json>` identifica su origen en la consola y conserva aquel intento fallido. En ese corte aún no había ajustes, predicciones ni protocolo congelado. La ejecución posterior `20261119T031104515520Z` resolvió el guardado y ya dispone de recibo público 3/3.

La historia auxiliar BTC conserva el hueco del 24 de marzo de 2023 y sus objetivos no construibles; no rellena OHLCV o etiquetas. Su comprobación reconstruye las 1316 etiquetas conocidas dentro de precisión numérica. Los clásicos usan historia causal de etiquetas más amplia que las siete barras neuronales. La máscara común de diagnóstico de ajuste se fija por disponibilidad de rezagos antes de conocer errores; no elimina fechas de prueba. Estos contratos se reportan en vez de afirmar idéntica información para todas las familias.

D01–D03 diversifican instrumento, producto, tamaño y periodo, pero **no son tres mercados independientes**. Los dos criptoactivos comparten exposiciones y D03 comparte subyacente y parte de las fechas BTC. La auditoría no convierte sus rangos en evidencia de una población de dominios independientes.

### Corte histórico auditado de la parte 46

La [admisión persistida](https://github.com/jassan-bit/DeepLearning_202630/tree/fe061b59678741d5de06eebb46526748af017fd4/runs/e3_closure_cpu/20261119T031104515520Z) verifica D01, D02 y D03 con 3270, 1461 y 1583 filas, respectivamente. Se cotejaron los bytes de sus CSV contra sus hashes. El {download}`control de admisión <../_static/e3_audit_progress_20261119T050335948088Z/admission_persistence_check.json>` conserva los identificadores y la recuperación de agosto mediante 31 archivos diarios oficiales.

La campaña `20261119T031104515520Z` mantiene la fuente `5a07d6157a9796ca54416dc5dea1e1527e731597` y el [tag anotado previo a los ajustes](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-extended-audit-protocol-20261119T031104515520Z). El hash del protocolo es `5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a`. La ejecución usa NVIDIA A100-SXM4-80GB, Python 3.12.12 y Torch 2.8.0+cu126.

El [guardado de la parte 46](https://github.com/jassan-bit/DeepLearning_202630/tree/c8a700ab2245b70f8932d4a93f7e0faa890cfc9e/runs/e3_extended_audit/20261119T050335948088Z) enlaza **46 partes** y contiene **251 registros `result.json` con estado `completed`**, cuyos SHA Git coinciden con los recibos:

| Tipo de trabajo completo | Cantidad | Alcance |
|---|---:|---|
| HPO neuronal | 54 | Tres tasas prefijadas, con selección por validación. |
| Ajustes neuronales finales | 180 | Trayectorias por modelo, conjunto y semilla. |
| Curvas con tamaño reducido | 12 | Ajustes nuevos de los subconjuntos prefijados. |
| ARIMA | 1 | Ajuste clásico causal. |
| Controles deterministas/estadísticos | 4 | Sin optimizador neuronal; no se cuentan como cuatro entrenamientos neuronales. |
| Total | **251** | **247 trabajos de ajuste numérico: 246 neuronales y un ARIMA; cuatro controles adicionales.** |

El control `floor_plus_mean_excess` estima un escalar sólo con ajuste; se mantiene entre los controles estadísticos y no se cuenta como ejecución de un optimizador neuronal. Los 251 trabajos no son 251 arquitecturas distintas. D01 completó su familia de 19 sistemas; D02 completó cuatro modelos. Los 23 paneles completos por conjunto/modelo contienen 833 990 filas de predicciones, que incluyen particiones, semillas y escenarios y **no son 833 990 observaciones independientes**. D03 y D00 todavía no tienen sus ajustes nuevos bajo esta auditoría.

El {download}`dictamen de revisión de archivos <../_static/e3_audit_progress_20261119T050335948088Z/audit_verification.json>`, el {download}`índice de 251 resultados <../_static/e3_audit_progress_20261119T050335948088Z/result_records_index.json>` y la {download}`consola suministrada <../_static/e3_audit_progress_20261119T050335948088Z/colab_console.txt.gz>` permiten contrastar el corte. El cero de `new_real_data_fits` en la última parte cuenta intentos nuevos identificados en ese delta, no el número acumulado de ajustes terminados. La consola sólo imprimió 129 mensajes de trabajo porque el ejecutor continuó avanzando mientras el controlador guardaba los archivos.

El proceso se interrumpió por `github_transport_network_read_failed operation=git_blob_read`; el registro de interrupción confirma `model_state_preserved=true` y el estado del controlador conserva `analysis_complete=false`. Para continuar está el [cuaderno Colab fijado a esta campaña](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/4a03a0b45761ba1bf02440883de6a7093e71a7cf/runs/e3_read_resume/20261119T050335948088Z/resume_colab.ipynb), también disponible como {download}`archivo descargable <../_static/e3_audit_progress_20261119T050335948088Z/resume_colab.ipynb>`. Conserva fuente, master, protocolo y perfil A100 de 80 GB. Sus {download}`siete casos simulados de recuperación <../_static/e3_audit_progress_20261119T050335948088Z/resume_launcher_verification.json>` comprueban reintentos acotados sólo para ese error de lectura, sin reintentar errores de autenticación, modelos o escrituras inciertas. Los ajustes completos se reutilizan; un ajuste parcial o las inferencias pueden continuar o reconstruirse.

Este corte acredita progreso de ejecución y persistencia. **No contiene el análisis ampliado final ni certifica R1/R3**. Restan completar D02, ejecutar D03 y D00, y producir el análisis conjunto de calibración, perturbaciones, sensibilidad, curvas y recursos.

### Ajuste común, selección y presupuesto

La rejilla prefijada usa tres tasas —0,0003; 0,001; 0,003— y una semilla HPO separada. El ganador se elige por el mismo MSE de validación proyectado. Se fijan diez semillas finales por red y conjunto, con escalas estimadas sólo en ajuste, calibración separada y purgas de siete soportes. La validación adicional sirve exclusivamente al HPO automático prefijado; si sus resultados guían un rediseño humano del mecanismo, esa exposición deberá declararse como desarrollo.

El límite de **120 segundos por ajuste** incluye las tres fases de SAE mediante un presupuesto compartido. Los 168 ajustes HPO, 560 finales y 48 adicionales para curvas suman 776 ajustes neuronales; se añaden cuatro ajustes ARIMA. Su techo nominal es **26 horas**, además de instalación, inferencia, estadísticas y persistencia. Se registran pasos, tiempos, parámetros y excesos en fronteras de lote/iteración; no se llama timeout duro a un límite que no puede interrumpir un kernel activo. El perfil de GPU y software se fija en el protocolo; la continuación con hardware distinto se detiene.

### Estudios especificados y evidencia exigida a sus salidas

| Estudio | Evidencia que generará la ejecución real |
|---|---|
| Evaluación adicional | Panel completo por conjunto, partición, modelo, semilla y fecha; RMSE, MAE, sMAPE y R². Ningún rival se sustituye ni se descartan fechas por rendimiento. |
| Incertidumbre de métricas | IC por bloques temporales de 7/14/28, ganancias emparejadas y dispersión entre semillas con ámbitos diferenciados. No se promedian RMSE de distintas escalas. |
| Calibración y cambio temporal | Cuantil nominal del 90 % obtenido sólo de calibración limpia; cobertura, anchura, score de intervalo y grupos temporales. No se fabrica CRPS/PIT desde un punto y un solo intervalo. |
| Perturbaciones | Tres severidades coherentes de OHLCV con escala estimada en ajuste, objetivos limpios y calibrador congelado. Los clásicos conservan su pronóstico crudo; su proyección puede cambiar con el piso corrupto. No se afirma perturbación idéntica de todo su historial. |
| Sensibilidad HPO | Evaluación secundaria de las tres tasas ya ajustadas, después de guardar la selección por validación; no cambia el ganador por su resultado de prueba. |
| Tamaño de muestra | Ajustes reales al 25/50/100 % para candidato y TSMixer, preseleccionados, con tres semillas; 100 % reutiliza el ajuste final. Los prefijos también cambian la cobertura temporal, por lo que no se atribuye causalmente todo efecto sólo al tamaño. |
| Recursos y figuras | Tiempos y parámetros observados, figuras con IC o dispersión identificada; FLOPs y energía ausentes se conservan como ausencias. |

Los {download}`detalles del método y su recuperación, versión v1 <../_static/e3_evaluacion_ampliada_20261119/README_recovery_transport_v1_20261119.md>` y la {download}`revisión metodológica <../_static/e3_evaluacion_ampliada_20261119/protocol_review/recommendations.md>` identifican supuestos y limitaciones. Se conserva la {download}`descripción inicial sin modificar <../_static/e3_evaluacion_ampliada_20261119/README.md>` como descarga histórica. Los paneles se fragmentan para evitar archivos monolíticos y cada bloque guarda hashes, checkpoints y estados necesarios para continuar una sesión nueva. Un fallo conserva su evidencia y no activa una arquitectura de sustitución.

### Preespecificación verificable y alcance de la conclusión

El controlador ya generó el protocolo con los hashes de los tres archivos admitidos y creó el **tag anotado real antes de los ajustes**; el corte auditado conserva su verificación. Ese registro es una preespecificación de auditoría exploratoria, no un preregistro retroactivo ni una certificación de R3. La batería temporal y entre conjuntos todavía no tiene potencia completa dimensionada; el plan anterior de 18 sistemas y 51 pruebas entre semillas no se reutiliza como dimensionamiento de esta familia.

La documentación y las pruebas sintéticas no completan por sí solas el componente. El corte parcial anterior dejó pendiente producir e inspeccionar el análisis final; la continuación de **8.18** ya aporta sus tablas y figuras reales. Persisten las limitaciones de recursos, adecuación temporal, inferencia y potencia descritas allí. Los recibos finales conservan `R1_accredited=false` y `R3_accredited=false`.

(e3-final-extended-audit)=
## 8.18 Cierre de la auditoría ampliada: resultados y robustez

**Corte final del 19 de noviembre de 2026, 18:43 en Colombia (23:43 UTC).** El autor ejecutó la campaña en Colab y la actualización del libro cotejó los recibos, tablas y figuras guardados. El [commit final `16a4504`](https://github.com/jassan-bit/DeepLearning_202630/tree/16a4504f6fb9208c7d6f89b65017194fa20470da/runs/e3_extended_audit/20261119T234320461006Z), parte 193 del master `20261119T031104515520Z`, declara **`stage_status=completed` y `analysis_complete=true`**. Este cierre sustituye el estado de ejecución parcial de 8.17 y conserva aquel corte como historial.

### Ejecución, integridad y desviaciones declaradas

| Evidencia final | Cantidad y alcance |
|---|---|
| Conjuntos evaluados | D00 BTC histórico y tres adicionales: D01 WIKI/AAPL RAW, D02 ETHUSDT Spot y D03 BTCUSD_PERP COIN-M. |
| Familia completa | 19 sistemas por conjunto; diez semillas finales para cada una de las 14 redes y una trayectoria por control determinista. |
| Trabajos completos | **796**: 168 ajustes HPO neuronales, 560 ajustes neuronales finales, 48 ajustes adicionales de curvas, cuatro ajustes ARIMA y 16 controles. Los 776 ajustes neuronales y cuatro ARIMA suman 780 ajustes numéricos. |
| Paneles | **76** archivos, con **1 874 560 filas** que incluyen particiones, semillas y escenarios; no son réplicas independientes. |
| Análisis | **12 tablas CSV originales**, verificadas por tamaño y SHA-256; 9999 remuestreos por combinación fijada de conjunto y longitud de bloque. |
| Figuras | **16**, cada una exportada a PNG y SVG: 32 imágenes reales verificadas. |
| Estado de evaluación | `completed`, lista de fallos vacía. No se confunde una parada por presupuesto registrada con un fallo borrado. |

Se conserva la fuente `5a07d6157a9796ca54416dc5dea1e1527e731597`, el tag `e3-extended-audit-protocol-20261119T031104515520Z` y el protocolo SHA-256 `5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a`. Los ajustes mantuvieron la cohorte NVIDIA A100-SXM4-80GB, Python 3.12.12, Torch 2.8.0+cu126 y CUDA 12.6. El análisis y las figuras usaron su entorno CPU aislado; esa separación no se presenta como una nueva cohorte de entrenamiento.

La ejecución efectiva tuvo una **enmienda de lectura de D00 posterior al congelamiento**, declarada y guardada antes de sus primeros ajustes: el lector anterior intentaba convertir a número el cierre vacío del 24 de marzo de 2023. La {download}`declaración de la enmienda <../_static/e3_audit_completed_20261119T234320461006Z/D00_parser_amendment.json>` registra 597 trabajos anteriores y cero D00 completos al declararla. Conserva el hueco de calendario, la aritmética original de σ₇ y los bytes de CSV/NPZ, sin imputar valores ni retirar filas de la cohorte. Mantener la fuente fijada no oculta este cambio del ejecutable efectivo.

El último fallo ocurrió **después de terminar el análisis**: Matplotlib rechazó el backend heredado de Colab durante la importación. El {download}`recibo de recuperación de figuras <../_static/e3_audit_completed_20261119T234320461006Z/plot_backend_recovery.json>` documenta la selección de `MPLBACKEND=Agg` antes de iniciar ese proceso. Verificó y reutilizó los resultados y las tablas ya completas: cero ajustes, predicciones o remuestreos nuevos en ese cierre. La asistencia de IA está declarada en los recibos y en el informe.

### Rendimiento en prueba limpia con tratamiento común

La métrica es la **media de las RMSE individuales**, con diez inicializaciones por red; no la RMSE de un ensamble. Todos los sistemas reciben la misma proyección sobre el piso observable y se conservan sus predicciones crudas. El objetivo es σ₇ poblacional sin anualizar; D01 usa sesiones bursátiles y los otros conjuntos días de su calendario. Cada fila se interpreta en su conjunto, sin promediar errores entre escalas.

| Conjunto | Fechas de prueba | RMSE E3 | RMSE mejor rival | Reducción relativa | Puesto E3 |
|---|---|---|---|---|---|
| D00 | 608 | 0,002924166 | 0,003027273 | 3,41 % | 1/19 |
| D01 | 633 | 0,002631154 | 0,002857878 | 7,93 % | 1/19 |
| D02 | 271 | 0,006515450 | 0,006654141 | 2,08 % | 1/19 |
| D03 | 296 | 0,003301771 | 0,003420131 | 3,46 % | 1/19 |

El candidato ocupa el primer puesto descriptivo en los cuatro conjuntos. El rival más cercano en todos es **`floor_plus_mean_excess`**, que estima su exceso medio únicamente con ajuste. Las reducciones relativas son `100 × (RMSE_rival − RMSE_E3) / RMSE_rival`. El {download}`resumen con precisión completa <../_static/e3_audit_completed_20261119T234320461006Z/clean_test_summary.csv>` registra también dispersión entre semillas, cobertura y anchura; las tablas originales conservan MAE y R².

Estas cifras proceden de una **auditoría exploratoria con rejilla y presupuesto comunes**, distinta de la comparación histórica de 17 sistemas. La RMSE histórica de E3 de 8.1–8.12 y sus 48 contrastes no se sustituyen ni se trasladan a estas recetas, semillas y conjuntos.

### Incertidumbre y alcance de los contrastes

Los intervalos por bloques de 7, 14 y 28 observaciones están condicionados a las trayectorias ajustadas: no remuestrean semillas. Las figuras de RMSE muestran IC nominales del 95 %, mientras que las curvas muestran desviación estándar entre tres semillas. No representan el mismo tipo de incertidumbre.

La tabla siguiente resume la ganancia `RMSE_rival − RMSE_E3` frente a `floor_plus_mean_excess`, para bloques de siete observaciones. Positivo favorece al candidato. Los intervalos simultáneos aplican la corrección de Bonferroni de la familia primaria de 72 comparaciones, **no un ajuste de valores p con Holm**.

| Conjunto | Ganancia media | IC nominal 95 % | IC simultáneo, familia 72 |
|---|---|---|---|
| D00 | 0,000103107 | [-0,000013182; 0,000192388] | [-0,000102030; 0,000225230] |
| D01 | 0,000226724 | [0,000135666; 0,000307491] | [0,000081400; 0,000357423] |
| D02 | 0,000138691 | [0,000019600; 0,000271180] | [-0,000055627; 0,000348872] |
| D03 | 0,000118360 | [0,000047501; 0,000186228] | [-0,000000835; 0,000232290] |

El IC nominal de D00 incluye cero. Los intervalos simultáneos frente al rival más cercano incluyen cero en **D00, D02 y D03**, también para bloques 14 y 28; sólo D01 mantiene sus extremos positivos. Además, con 9999 remuestreos, cada cola simultánea tiene apenas **3,47 remuestreos esperados**: la resolución extrema es escasa y el archivo registra esa limitación. No se presenta este cálculo como certificación de superioridad frente a un margen práctico δ.

`paired_effects.csv` contiene **864 filas**: 216 del escenario limpio y 648 de ruido. `noise_contrasts.csv` repite ese subconjunto de ruido y no aporta otras 648 pruebas independientes. Sus campos `p_value` están vacíos y `formal_R3_decision=false`: **esta etapa no produjo nuevos valores p, ajuste de Holm o tres pruebas confirmatorias por rival**. Los rangos entre conjuntos son descriptivos; no se ejecutó una nueva prueba Friedman/Iman–Davenport ni ROPE para certificar R3. Los tres tamaños de bloque son sensibilidad del mismo procedimiento, no tres pruebas diferentes.

### Calibración, cambio temporal y perturbaciones

El calibrador de cada trayectoria se estima sólo con la partición de calibración limpia, a nivel nominal **90 %**; se conserva al evaluar ruido. Las coberturas siguientes son medias entre las diez semillas del candidato sobre prueba limpia, no la cobertura de un único modelo servido en Render.

| Conjunto | Cobertura E3, nominal 90 % | Anchura media, σ₇ |
|---|---|---|
| D00 | 94,51 % | 0,010960896 |
| D01 | 91,50 % | 0,005514413 |
| D02 | 91,88 % | 0,016063960 |
| D03 | 89,29 % | 0,007521472 |

D03 presenta cobertura agregada inferior al nominal. También hay grupos temporales desfavorables: **86,44 % en D00 durante 2026Q1**, **82,10 % en D01 durante 2012Q4** y **85,71 % en D03 durante 2024Q1**, este último con 21 fechas. La cobertura global no garantiza cobertura por trimestre, subgrupo o cambio de distribución. Las tablas conservan fechas, tamaños, anchura y score de intervalo; un punto y un único intervalo no definen una distribución para fabricar CRPS o PIT.

Las tres severidades de ruido perturban coherentemente las entradas D7 OHLCV, mantienen el objetivo limpio y no recalibran sobre prueba. En la severidad máxima se observa:

| Conjunto | RMSE con ruido máximo | Factor frente a limpio | Puesto E3 |
|---|---|---|---|
| D00 | 0,013568025 | 4,64 × | 17/19 |
| D01 | 0,023710133 | 9,01 × | 18/19 |
| D02 | 0,023783958 | 3,65 × | 17/19 |
| D03 | 0,023082500 | 6,99 × | 18/19 |

El deterioro descarta una afirmación general de robustez del candidato al ruido. Los clásicos conservan limpia su historia cruda y puede cambiar su proyección sobre el piso perturbado; el estudio **no equivale a corromper toda la historia de todos los modelos bajo un tratamiento idéntico**. Se publica el resultado adverso sin cambiar la métrica o retirar rivales.

### HPO, tamaño de muestra y coste observado

La sensibilidad HPO contiene **168 filas**, correspondientes a cuatro conjuntos, 14 redes y las tres tasas prefijadas. Se verificó la selección por mínimo de validación con la semilla HPO separada 20261117. Los errores de prueba de esas alternativas se reportan como análisis secundario; no se usaron para volver a seleccionar la receta.

Las curvas comparan al candidato y al TSMixer preseleccionado, con fracciones 25/50/100 % y tres semillas. Hay **72 grupos de ajuste**: 24 reutilizan el ajuste final del 100 % y **48 contienen ajustes redimensionados reales**. Los prefijos anidados cambian conjuntamente el tamaño y la extensión del calendario; no aíslan un efecto causal exclusivo del número de muestras. Sus barras muestran desviación estándar entre tres semillas.

`resources_supplied.csv` conserva **900 registros de fases y mediciones**, no 900 modelos independientes. La suma informada de tiempos de fase de ajuste es **9319,069 segundos**; no es el tiempo total de campaña, instalación, inferencia, estadísticas, persistencia o facturación. Las razones de parada incluyen límites de tiempo, reanudación, épocas y parada temprana. **FLOPs y energía siguen ausentes**, y la comparabilidad completa de las mediciones de hardware no se ha verificado independientemente. Esos campos permanecen desconocidos, sin sustituirlos por cero o estimarlos desde el número de parámetros.

### Figuras de la familia completa y galería

Las cuatro figuras de RMSE incluyen los 19 sistemas y sus IC temporales nominales; no omiten los dos controles analíticos. El solapamiento visual de barras no sustituye al intervalo emparejado de la diferencia descrito arriba.

```{figure} ../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D00.png
:name: fig-e3-final-rmse-d00
:width: 100%

D00: media de RMSE individuales e IC temporal nominal del 95 %, bloques de siete observaciones. Auditoría exploratoria; no acredita R3.
```

```{figure} ../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D01.png
:name: fig-e3-final-rmse-d01
:width: 100%

D01: media de RMSE individuales e IC temporal nominal del 95 %, bloques de siete observaciones. Auditoría exploratoria; no acredita R3.
```

```{figure} ../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D02.png
:name: fig-e3-final-rmse-d02
:width: 100%

D02: media de RMSE individuales e IC temporal nominal del 95 %, bloques de siete observaciones. Auditoría exploratoria; no acredita R3.
```

```{figure} ../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D03.png
:name: fig-e3-final-rmse-d03
:width: 100%

D03: media de RMSE individuales e IC temporal nominal del 95 %, bloques de siete observaciones. Auditoría exploratoria; no acredita R3.
```

La galería completa permite consultar las cuatro clases de figura en cada conjunto y descargar sus originales. Los gráficos de ruido muestran los comparadores preseleccionados; las tablas descargables conservan la familia completa.

| Conjunto | RMSE | Ruido | Cobertura | Tamaño de muestra |
|---|---|---|---|---|
| D00 | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D00.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D00.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D00.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D00.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D00.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D00.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D00.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D00.svg) |
| D01 | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D01.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D01.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D01.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D01.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D01.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D01.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D01.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D01.svg) |
| D02 | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D02.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D02.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D02.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D02.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D02.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D02.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D02.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D02.svg) |
| D03 | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D03.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/rmse_D03.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D03.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/noise_D03.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D03.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/coverage_D03.svg) | [PNG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D03.png) · [SVG](../_static/e3_audit_completed_20261119T234320461006Z/figures/learning_D03.svg) |

### Descargas y trazabilidad del cierre

Se publican copias de los **12 CSV originales y las 32 imágenes, sin alterar sus bytes**, junto con sus recibos. El {download}`manifiesto del análisis <../_static/e3_audit_completed_20261119T234320461006Z/analysis/output_manifest.json>` fija tamaño y SHA-256 de cada tabla, y el {download}`manifiesto de figuras <../_static/e3_audit_completed_20261119T234320461006Z/figures/figures_manifest.json>` enlaza las imágenes con ese análisis. El {download}`control de las doce tablas <../_static/e3_audit_completed_20261119T234320461006Z/tables_verification.json>` y la {download}`lectura derivada declarada <../_static/e3_audit_completed_20261119T234320461006Z/readout.json>` distinguen comprobación y resumen de resultados de nuevos experimentos.

| Producto | Descarga |
|---|---|
| Evaluación y protocolo | {download}`Recibo de evaluación <../_static/e3_audit_completed_20261119T234320461006Z/evaluation_receipt.json>` · {download}`Protocolo <../_static/e3_audit_completed_20261119T234320461006Z/protocol.json>` · {download}`Recibo del tag <../_static/e3_audit_completed_20261119T234320461006Z/freeze_receipt.json>` |
| Identidad de resultados | {download}`Manifiesto de ejecución <../_static/e3_audit_completed_20261119T234320461006Z/run_manifest.json>` · {download}`Manifiesto de paneles <../_static/e3_audit_completed_20261119T234320461006Z/panel_manifest.json>` |
| Cierre y análisis | {download}`Estado final <../_static/e3_audit_completed_20261119T234320461006Z/controller_status.json>` · {download}`Recibo del análisis <../_static/e3_audit_completed_20261119T234320461006Z/analysis/analysis_receipt.json>` · {download}`Recibo de parte 193 <../_static/e3_audit_completed_20261119T234320461006Z/stage_receipt.json>` |
| Intervalos temporales | {download}`calendar_CIs.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/calendar_CIs.csv>` |
| Calibradores | {download}`calibrators.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/calibrators.csv>` |
| Rangos entre conjuntos | {download}`cross_dataset_ranks.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/cross_dataset_ranks.csv>` |
| Sensibilidad HPO | {download}`hpo_sensitivity_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/hpo_sensitivity_metrics.csv>` |
| Métricas de intervalos | {download}`interval_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/interval_metrics.csv>` |
| Métricas por modelo | {download}`model_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/model_metrics.csv>` |
| Efectos bajo ruido | {download}`noise_contrasts.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/noise_contrasts.csv>` |
| Efectos emparejados | {download}`paired_effects.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/paired_effects.csv>` |
| Métricas por semilla | {download}`per_seed_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/per_seed_metrics.csv>` |
| Métricas trimestrales | {download}`quarter_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/quarter_metrics.csv>` |
| Recursos observados | {download}`resources_supplied.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/resources_supplied.csv>` |
| Curvas de tamaño | {download}`sample_size_metrics.csv <../_static/e3_audit_completed_20261119T234320461006Z/analysis/sample_size_metrics.csv>` |

Los paneles completos, trabajos y checkpoints permanecen en la cadena inmutable de GitHub; esta actualización del libro no replica los 277 MB de paneles como descargas del sitio. El manifiesto permite identificar esos archivos. Se conservan la procedencia y las condiciones de los datos descritas en 8.14, incluidas las restricciones de Binance; publicar una figura no cambia sus derechos.

### Qué queda abierto para R3 y para el entregable

La evaluación de los tres archivos adicionales, el tratamiento común, la sensibilidad HPO, las curvas reales y las figuras **ya tienen evidencia ejecutada**. La menor RMSE descriptiva en los cuatro conjuntos mejora la evidencia disponible, sin cerrar las condiciones científicas siguientes:

1. **Inferencia confirmatoria:** justificar y ejecutar las tres pruebas adecuadas por contraste, márgenes de relevancia y control familiar exigidos, con potencia por contraste y conjunta dimensionada prospectivamente. Este análisis de intervalos no los reemplaza.
2. **Generalización:** evaluar la adecuación temporal y las dependencias. D02/D03 comparten exposición cripto y D03 comparte subyacente y fechas con BTC; no son tres dominios independientes. Los resultados ya observados conservan su alcance exploratorio.
3. **Robustez, incertidumbre y coste:** explicar el deterioro con ruido y la infracobertura, y completar las mediciones comparables de recursos, FLOPs y energía. Las tablas no justifican declarar robustez universal.
4. **Contribución y producto:** la novedad sustantiva de R1 y la defensa académica de R2 siguen sus requisitos propios. La reproducción humana del libro y su tabla principal —con el alcance de la guía aclarado en {ref}`11.12 <e3-next-human-book-20261121>`—, el servicio Render, el catálogo exportado y el depósito DOI no se completan por guardar estas figuras; tampoco se impone por ello repetir todos los entrenamientos. El replay Docker histórico conserva su ámbito anterior de 11 CSV y 143 trayectorias.

El cierre operativo está verificado. Los recibos científicos mantienen **`R1_accredited=false` y `R3_accredited=false`**, y la actualización no asigna una calificación automática.

(e3-v3-pilot-completed-20261120)=
## 8.19 Piloto V3 terminado: ablaciones y resultado adverso de desarrollo

**El piloto completó sus 30 ajustes y no superó la heurística de continuación.** El [resultado inmutable](https://github.com/jassan-bit/DeepLearning_202630/tree/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z) corresponde al código `8f65eac34aaf49293924ae8328c235cedbdd2866`, ejecución `20261120T031424176339Z`, snapshot 93. Se conservan sus {download}`recibo final <../_static/e3_evidence_close_20261120/pilot/archive/payload/pilot_receipt.json>`, {download}`plan <../_static/e3_evidence_close_20261120/pilot/archive/payload/plan.json>` y {download}`veredicto original <../_static/e3_evidence_close_20261120/pilot/archive/payload/development_verdict.json>`. Los 796 trabajos anteriores pertenecen a la referencia aditiva y no se convierten en entrenamientos V3.

### Qué se ejecutó y qué se verificó después

El estudio original realizó 15 ajustes HPO y 15 finales: cinco recetas neuronales, tres tasas con una semilla HPO y tres semillas finales por receta; además evaluó los dos controles analíticos. Las 532 ventanas de ajuste y las 54 de selección conservan su función. Tras siete fechas purgadas, la evaluación usa **55 fechas, del 7 de septiembre al 31 de octubre de 2024**, ya expuestas históricamente como desarrollo. Los recibos declaran cero miembros de calibración/prueba abiertos y ningún conjunto D01–D03 usado en este piloto.

El {download}`control de integridad <../_static/e3_evidence_close_20261120/pilot/generated/archive_verification.json>` comprueba **19 archivos originales**, cinco hashes de artefactos y los enlaces del plan, admisión, launcher y estado. De las predicciones guardadas se recalculan **17 trayectorias y 935 filas**, con los mismos objetivos y fechas. Las 935 filas no son 935 observaciones independientes. Los 30 recibos de fase registran estados entrenados y terminados, sin exceso de plazo declarado en este piloto; ese dato no se extrapola a los recursos de la auditoría histórica. No se descargaron ni reprodujeron checkpoints, ni los 259 archivos completos del snapshot.

### Resultado de las siete recetas y decisión original

La métrica principal permanece fijada como **media de las RMSE individuales**, no RMSE de un ensamble medio ni una métrica elegida después del resultado. La {download}`recomputación completa <../_static/e3_evidence_close_20261120/pilot/generated/metrics_recomputed.csv>` y las {download}`comparaciones de ablación <../_static/e3_evidence_close_20261120/pilot/generated/ablation_comparisons.csv>` conservan todas las recetas:

| Receta | Trayectorias | RMSE media individual | DE entre semillas | MAE media |
|---|---:|---:|---:|---:|
| `atom_mse_only` | 3 | 0,002073833 | 0,000023659 | 0,001644275 |
| `fixed_probs` | 3 | 0,002083649 | 0,000040853 | 0,001557355 |
| `atom_plugin` | 3 | 0,002094286 | 0,000031909 | 0,001549696 |
| `floor_plus_mean_excess` | 1 analítica | 0,002114231 | No aplica | 0,001813483 |
| `atom_crps` —V3 completa— | 3 | 0,002129916 | 0,000013719 | 0,001591685 |
| Aditiva | 3 | 0,002157281 | 0,000022490 | 0,001811966 |
| `floor_only` | 1 analítica | 0,002790341 | No aplica | 0,001831766 |

La V3 completa tiene **2,70 % más RMSE que `atom_mse_only`**, el mejor control neuronal y global, y aproximadamente 0,74 % más que el control de piso más exceso. Mejora un 1,27 % frente a la aditiva, pero eso no satisface la regla frente al mejor de los controles completos. Para superar la heurística del 2 %, su RMSE debía ser como máximo **0,002032356**. El veredicto permanece **`continuation_heuristic_passed=false`**. La completa pierde en RMSE frente a las tres ablaciones neuronales en las tres semillas emparejadas; estos conteos no son una prueba estadística ni establecen un efecto causal general de cada componente.

Retirar CRPS, fijar masas y sustituir la lectura exacta ya tienen evidencia empírica **de desarrollo**. La variante de masas fijas tiene 1236 parámetros y las de masas aprendidas 1262; no se disimula esa diferencia de capacidad. La menor MAE de otro control y los CRPS_U guardados se conservan como resultados secundarios, sin reemplazar RMSE para rescatar la decisión. **La variante sin CRPS no se promueve automáticamente a modelo final.** El resultado adverso no constituye una sanción de integridad ni demuestra por sí mismo ausencia de novedad; muestra que este piloto no aporta la utilidad exigida por su regla original.

```{figure} ../_static/e3_evidence_close_20261120/pilot/generated/pilot_rmse_and_dependence.png
:name: fig-e3-v3-pilot-saved-results
:width: 100%

Resultados de desarrollo guardados. Los puntos muestran semillas, las barras su media de RMSE individuales y los rombos controles analíticos. Las correlaciones de residuos son promedios descriptivos entre semillas, sin intervalos ni pruebas. La dispersión entre semillas no mide incertidumbre entre mercados o calendarios. [SVG descargable](../_static/e3_evidence_close_20261120/pilot/generated/pilot_rmse_and_dependence.svg).
```

### Adecuación posible con las predicciones disponibles

El {download}`reanálisis reproducible <../_static/e3_evidence_close_20261120/pilot/analyze_saved_pilot.py>` añade {download}`errores y correlaciones por trayectoria <../_static/e3_evidence_close_20261120/pilot/generated/residual_diagnostics.csv>` y {download}`cortes mensuales completos <../_static/e3_evidence_close_20261120/pilot/generated/monthly_diagnostics.csv>`. Son diagnósticos posteriores al resultado, sin remuestreos ni valores p. En la completa, el sesgo medio `predicción − objetivo` es −0,00026448 y la correlación descriptiva media del residuo a siete días es −0,320. Una correlación pequeña en otro desfase no prueba independencia de ventanas solapadas.

Los dos meses se conservan juntos: en septiembre —24 fechas—, las RMSE medias son 0,002012253 para la completa y 0,002022917 sin CRPS; en octubre —31—, 0,002216532 y 0,002112206. No se selecciona septiembre para declarar éxito: la decisión usa las 55 fechas y la regla previa. Las cinco recetas seleccionaron la tasa máxima de la cuadrícula, 0,003; esto delimita la sensibilidad estudiada, sin autorizar una búsqueda nueva guiada por esta evaluación.

La {download}`matriz de alcance de adecuación <../_static/e3_evidence_close_20261120/pilot/generated/adequacy_scope_matrix.csv>` identifica límites que no pueden cerrarse con estos archivos. Faltan predicciones de ajuste y riesgos comparables para completar las Tablas A/B y la brecha de sobreajuste. Los CRPS_U originales y su agregación sí están disponibles, pero faltan leyes atómicas por observación e intervalos guardados para recalcular independientemente esa puntuación o certificar calibración. Tampoco se reconstruyen retrospectivamente margen formal, potencia prospectiva, tres pruebas apropiadas por contraste o una familia confirmatoria. Los tres conjuntos adicionales de 8.18 se mantienen como otra auditoría; no fueron evaluados por V3 aquí.

El {download}`verificador aritmético independiente <../_static/e3_evidence_close_20261120/pilot/verify_saved_pilot.py>` y su {download}`recibo <../_static/e3_evidence_close_20261120/pilot/generated/independent_arithmetic_verification.json>` usan una segunda formulación con la biblioteca estándar. Esta actualización declara asistencia de IA y ejecuta **cero entrenamientos, inferencias o remuestreos nuevos**. Se mantienen **`R1_accredited=false` y `R3_accredited=false`**; cualquier modificación posterior deberá registrarse como diseño nuevo, preservando este resultado, su protocolo y el conjunto de ensayos.

(e3-existing-adequacy-close-20261120)=
## 8.20 Adecuación retrospectiva con predicciones guardadas

Este cierre completa evidencia que podía calcularse a partir de archivos ya
guardados. Verifica las doce tablas de la auditoría de 8.18 y cuatro paneles
inmutables del candidato; de estos últimos conserva sólo 18.080 predicciones
de prueba limpia, correspondientes a diez semillas en cada conjunto. No entrena
modelos, realiza inferencia ni repite los 9.999 remuestreos anteriores. El
{download}`manifiesto de entradas <../_static/e3_evidence_close_20261120/adequacy/archive/input_manifest.json>`
y el {download}`recibo del reanálisis <../_static/e3_evidence_close_20261120/adequacy/results/adequacy_receipt.json>`
fijan el origen, los filtros y los hashes; se declara asistencia de IA.

### Brecha, controles triviales y cobertura

Las métricas de ajuste y de las particiones posteriores ya estaban guardadas.
Las {download}`doce comparaciones de brecha <../_static/e3_evidence_close_20261120/adequacy/results/descriptive_fit_heldout_gaps.csv>`
emparejan semillas dentro de cada conjunto. La tabla resume prueba limpia:

| Conjunto | RMSE prueba / RMSE ajuste¹ | Cobertura media, nominal 90 % | Menor cobertura trimestral media² |
|---|---:|---:|---:|
| D00 | 0,9364 | 94,51 % | 86,44 % —2026Q1— |
| D01 | 0,2945 | 91,50 % | 82,10 % —2012Q4— |
| D02 | 0,7802 | 91,88 % | 90,11 % —2021Q2— |
| D03 | 0,6716 | 89,29 % | 85,71 % —2024Q1— |

¹ Cociente de RMSE medias individuales, sin ensamblar predicciones. Las
particiones corresponden a periodos y regímenes distintos; una razón menor
que uno **no demuestra ausencia de sobreajuste**, en particular en D01. Se
conservan también la dispersión y la brecha emparejada entre semillas sobre
el calendario fijo. No se reconstruye un margen de brecha ni potencia previos.

² Mínimo descriptivo entre todos los trimestres guardados, no una selección
previa para contrastar. La {download}`tabla de cobertura y deriva <../_static/e3_evidence_close_20261120/adequacy/results/coverage_drift_readout.csv>`
conserva fechas, rangos entre semillas y ámbito de cada valor.

La {download}`comparación con cinco controles triviales <../_static/e3_evidence_close_20261120/adequacy/results/trivial_baseline_comparison.csv>`
incluye los dos controles de piso, persistencia, estacional y ARIMA. Aunque el
candidato tiene menor RMSE, su **MAE supera a la de `floor_only` en 5,63 % en D00
y 26,59 % en D01**. Una ventaja en la métrica de entrenamiento no se interpreta
como superioridad en todas las pérdidas o escenarios.

### Diagnósticos de residuos: dos ejecuciones, dos familias

Se calcularon Ljung–Box a rezagos 7/14, ARCH-LM a 7 y la hipótesis conjunta de
Mincer–Zarnowitz —intercepto cero y pendiente uno— con HAC Bartlett de rezago
14. Son rezagos en observaciones, que representan sesiones en D01. Se usa una
regresión auxiliar sobre predicciones ya guardadas; no se ajusta nuevamente el
modelo de pronóstico. Las ventanas del objetivo comparten retornos y las diez
semillas no constituyen diez mercados independientes.

La {download}`familia de la auditoría ampliada <../_static/e3_evidence_close_20261120/adequacy/results/extended_candidate_residual_diagnostics.csv>`
contiene **160 valores p diagnósticos**, ajustados conjuntamente por Holm:
cuatro conjuntos × diez semillas × cuatro procedimientos/rezagos. Entre los
resultados adversos se encuentran los siguientes rangos entre semillas:

| Conjunto y diagnóstico | Rango de p ajustado, Holm160 | Lectura delimitada |
|---|---:|---|
| D00, ARCH-LM(7) | 8,15 × 10⁻¹³ a 0,02882 | Dependencia de residuos cuadrados compatible con efecto ARCH, bajo los supuestos del diagnóstico. |
| D01, Mincer–Zarnowitz con HAC(14) | 1,40 × 10⁻¹⁸ a 4,63 × 10⁻¹⁰ | Desviación de la calibración lineal de la media. Las pendientes se sitúan entre 0,9452 y 0,9702. |

La tabla completa conserva los resultados restantes. Los valores grandes o una
falta de rechazo no certifican adecuación. La normalidad no es requisito general
para que un pronóstico de regresión sea útil: Jarque–Bera se aporta como
estadístico de forma, con asimetría y curtosis, **sin interpretar su p iid** bajo
dependencia serial. La asimetría positiva y las colas pronunciadas se conservan.

El {download}`diagnóstico histórico separado <../_static/e3_evidence_close_20261120/adequacy/results/historical_residual_diagnostics.csv>`
utiliza las diez trayectorias de 608 fechas del commit `510c3bdd`, con pesos
anteriores y una familia propia Holm40. No se mezcla con Holm160 ni se atribuye
a V3. Las dos familias son retrospectivas y auxiliares: **no son las tres pruebas
de superioridad por rival de R3**, no justifican potencia prospectiva y no
completan automáticamente las Tablas A/B.

Los valores p están condicionados a los supuestos de dependencia, momentos y
aproximación asintótica. Holm no corrige un p marginal inválido. Los datos
expuestos no recuperan cegamiento y una decisión posterior no se convierte en
preregistro. Tampoco se fabrican PIT o CRPS para un punto acompañado sólo por
un intervalo; esa salida no define una distribución predictiva completa.

### Recursos, reproducción y requisitos abiertos

El {download}`inventario de campos de recursos <../_static/e3_evidence_close_20261120/adequacy/results/resource_field_inventory.csv>`
distingue dato suministrado, ausente o no aplicable. **900 filas de fases
representan 796 trabajos**, con fases adicionales del autoencoder. Tres
segmentos mantienen cómputo activo desconocido tras una interrupción. El máximo
exceso de plazo, 35.283,63 segundos, incluye desconexión y **no es una medida de
9,8 horas de cómputo GPU**. No se completa energía o FLOPs usando parámetros,
ni se certifica igualdad de cómputo total real por compartir un límite nominal.

La {download}`metodología y comandos <../_static/e3_evidence_close_20261120/adequacy/README.md>`,
el {download}`script de reanálisis <../_static/e3_evidence_close_20261120/adequacy/audit_existing_predictions.py>`
y la {download}`verificación independiente <../_static/e3_evidence_close_20261120/adequacy/results/reference_verification.json>`
permiten reproducirlo sin red ni modelos: se comprobaron 250 resultados numéricos
contra statsmodels y 200 ajustes de Holm. La concordancia informática no
demuestra los supuestos científicos. La {download}`matriz de cierre <../_static/e3_evidence_close_20261120/adequacy/closure_matrix.csv>`
separa lo calculado de lo pendiente: margen/criterios de adecuación y potencia
previos, interpretación de estabilidad y distribución, datos realmente reservados,
familia confirmatoria pertinente, recursos completos y reproducción independiente.
**Este cierre aporta diagnósticos; no acredita R1 o R3 ni supera formalmente
las Tablas A/B por retrospectiva.**


(e3-v3-frozen-transfer-20261120)=
## 8.21 Diagnóstico V3 de transferencia con pesos y escalas fijos

Después del piloto se registró una **decisión nueva y exploratoria** en el
[commit `6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe`](https://github.com/jassan-bit/DeepLearning_202630/commit/6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe),
antes de las primeras predicciones de este diagnóstico. La
{download}`decisión fijada <../_static/e3_completion_20261120/protocol_power/decision_zero_shot.json>`
y el {download}`recibo previo al primer forward <../_static/e3_completion_20261120/reproduction/diagnostic_transfer/before_first_forward.json>`
vinculan sus hashes y el código ejecutado. No es un prerregistro
confirmatorio ni recupera el cegamiento de los calendarios.

Se aplicaron las **cinco recetas neuronales** del piloto con sus **tres
semillas finales originales** —20261118, 20261119 y 20261120— y los dos
controles analíticos a las particiones `clean_test` originales D01–D03.
Se conservaron los quince checkpoints y las escalas BTC, incluido el
escalador de volumen y el exceso medio del control físico. A se calcula
causalmente en cada entrada; no se ajusta con los nuevos dominios. No hubo
HPO, entrenamiento, calibración, corrección de escalas, proyección o
eliminación de filas. El diagnóstico completó **51 trayectorias y 20.400
filas finitas**, con **cero ajustes y cero fallos**. Los dos controles
deterministas siguen siendo trayectorias únicas, no réplicas entrenadas.

La {download}`geometría y las fechas de prueba <../_static/e3_completion_20261120/reproduction/reference/additional_clean_test_metadata.json>`
conservan las purgas y los hashes de la admisión anterior: D01 contiene
633 objetivos del 25 de agosto de 2010 al 1 de marzo de 2013; D02, 271
del 5 de abril al 31 de diciembre de 2021; D03, 296 del 11 de marzo al
31 de diciembre de 2024. Esos calendarios ya estuvieron expuestos durante
la auditoría de la aditiva. Aquí se evalúan pesos BTC fijos: **no se
reutilizan los ajustes por dominio de los 796 trabajos como si fueran V3**.

La métrica primaria registrada es la media de las RMSE individuales por
receta, en unidades originales de σ₇. Cada receta neuronal aporta tres
trayectorias y cada control analítico una. No es la RMSE de un ensamble.
La tabla presenta las siete recetas, sin retirar controles desfavorables:

| Receta o control | D01 AAPL RAW | D02 ETHUSDT | D03 BTCUSD_PERP |
|---|---:|---:|---:|
| `additive_positive_head` | 0,003632257 | 0,006572682 | 0,003682024 |
| `atom_crps` | 0,003314508 | 0,007161615 | 0,003444276 |
| `atom_mse_only` | 0,003353165 | 0,007198866 | 0,003436307 |
| `fixed_probs` | 0,002998234 | 0,007262353 | 0,003403212 |
| `atom_plugin` | 0,003307610 | 0,007151505 | 0,003399681 |
| `floor_only` | 0,002931884 | 0,007795489 | 0,004057838 |
| `floor_plus_mean_excess` | 0,002629224 | 0,006956665 | 0,003356707 |

La candidata completa **no obtiene la menor RMSE en ninguno de los tres
dominios**. El menor error descriptivo corresponde a
`floor_plus_mean_excess` en D01/D03 y a la aditiva en D02. El
{download}`resumen numérico <../_static/e3_completion_20261120/reproduction/diagnostic_transfer/summary.csv>`
conserva también MAE y CRPS_U cuando la receta proporciona la distribución
necesaria. El {download}`recibo del diagnóstico <../_static/e3_completion_20261120/reproduction/diagnostic_transfer/diagnostic_receipt.json>`
contiene las trayectorias, hashes, presupuesto observado y ausencia de
p-valores. No se ajustaron intervalos ni se heredó el calibrador de la
aditiva.

### Qué permite y qué impide esta comparación

Los pesos y la selección BTC proceden de **2023–2024** y se aplican a AAPL
hasta 2013 y ETH hasta 2021. Por tanto, **no es un backtest financiero
cronológicamente disponible**: describe transferencia retrospectiva con
parámetros posteriores. D03 comparte el subyacente BTC y parte del
calendario con el desarrollo, por lo que tampoco constituye un mercado
independiente. El volumen conserva unidades distintas —acciones RAW por
sesión, ETH o contratos por día UTC— bajo el mismo escalador BTC; no se
corrige ese cambio después de observar resultados. Sesiones y días UTC
son relojes homólogos, no duraciones idénticas. No se promedian RMSE de
distintas escalas para construir un ganador global.

El alcance es un diagnóstico acotado de inferencia, ejecutado con IA
declarada. No prueba la batería de R3, adecuación completa o potencia,
ni cambia la heurística del piloto original. Sus treinta ajustes y el
**2,70 % de deterioro de `atom_crps` frente a `atom_mse_only`** siguen
intactos; no se promociona automáticamente λ=0 ni otra receta ganadora.
El {ref}`dictamen documental de R1 para V3 <e3-r1-substantive-verdict-20261120>` conserva su alcance propio.

(e3-v3-prospective-preparation)=
## 8.22 Decisión científica y preparación de potencia para V3

**La confirmación sigue en NO-GO.** La
{download}`decisión científica <../_static/e3_completion_20261120/protocol_power/scientific_decision.json>`
conserva la regla del piloto y su resultado adverso. El diagnóstico de
transferencia posterior aporta información limitada, pero no transforma
los calendarios expuestos en confirmación ni autoriza otra campaña grande.
El autor puede defender la contribución delimitada sin afirmar superioridad
predictiva, o formular y fechar por qué merece estudiar una pregunta nueva.
Cambiar una regla o candidata sería otra decisión posterior a los datos.

Se preparó un {download}`protocolo DRAFT <../_static/e3_completion_20261120/protocol_power/protocol.DRAFT.json>`
y su {download}`explicación metodológica <../_static/e3_completion_20261120/protocol_power/protocol.DRAFT.md>`,
sin tag final, fechas de congelamiento inventadas ni número de semillas
declarado suficiente. La familia propuesta reúne **20 sistemas, 19
rivales y 57 componentes**: Top 10 y referencias E2 completos, aditiva y
dos controles físicos frente a `atom_crps`. Las tres ablaciones forman
otra familia. Se propone Holm global sobre los 57 p-valores y, para cada
rival, el máximo de sus tres p ajustados; los diecinueve contrastes deben
pasar. La batería temporal propuesta requiere todavía revisión de
supuestos, dirección y alineación con la media individual de RMSE.

### Cálculo ejecutado y límites

La guía §8.7.2 exige al menos cinco semillas piloto para estimar variabilidad
por rival; V3 tiene **tres**. De los seis controles disponibles, tres son
ablaciones: sólo aditiva y dos analíticos pertenecen a la familia principal.
Faltan las varianzas de los **dieciséis rivales E2**. No se transfieren a
V3 las varianzas o los resultados de los 796 trabajos de la aditiva.

El {download}`registro de escenarios <../_static/e3_completion_20261120/protocol_power/planning_registry.json>`
precede al cálculo CPU sobre pronósticos ya guardados. Se ejecutaron
**486 evaluaciones analíticas**, 162 de sensibilidad a semillas y 324
temporales, con **cero simulaciones Monte Carlo**, ajustes o inferencias
de modelos en esa etapa. El
{download}`recibo reproducible <../_static/e3_completion_20261120/protocol_power/results/planning_receipt.json>`
y las {download}`entradas de varianza <../_static/e3_completion_20261120/protocol_power/results/variance_inputs.csv>`
identifican el origen de los datos y mantienen **R final y calendario
final sin determinar**. No son cálculos de potencia observada posterior.

Las alternativas hipotéticas son `g1=1,25δ`, `1,5δ` y `2δ`; la separación
para `H0: Δ≤δ` es **g1−δ**, nunca la ganancia observada. El margen BTC
heredado se usa sólo como referencia pendiente de justificación operativa
para V3; no se elige desde los p-valores ni desde el umbral heurístico del
2 %. Los demás dominios requieren márgenes en sus escalas. Se usa
`.05/57` como nivel conservador de planificación, separado del ajuste Holm
que aplicaría el análisis final.

La {download}`sensibilidad por semillas <../_static/e3_completion_20261120/protocol_power/results/seed_sensitivity.csv>`
representa inicialización sobre un mismo calendario; la
{download}`sensibilidad temporal <../_static/e3_completion_20261120/protocol_power/results/calendar_sensitivity.csv>`
usa influencias/HAC con rezagos 7/14/28. No se suman ambas varianzas sin
identificar un modelo jerárquico ni se convierten 365 fechas hipotéticas
en observaciones independientes adquiridas. Las 55 fechas de desarrollo
no estabilizan por sí mismas la incertidumbre temporal de largo plazo.
La {download}`sensibilidad de potencia conjunta <../_static/e3_completion_20261120/protocol_power/results/joint_sensitivity.json>`
expone supuestos de dependencia y cotas; no demuestra la potencia de
tres procedimientos F2 distintos. La guía exige marginal mínima de 0,90
por contraste y **reportar** la conjunta; no impone conjunta 0,90.

### Acciones concretas antes de una confirmación

Se necesitan una nueva justificación científica del autor; admisión común
de datos genuinamente no expuestos y objetivos aún desconocidos; márgenes
operativos; batería F2 validada; potencia de la familia completa con
R≥10 o más si el cálculo lo exige; presupuesto comparable autorizado;
motor V3 revisado y tag anotado real antes de los resultados. El calendario
de confirmación no se rellena con tramos ya conocidos; AAPL futuro exige
admitir otra fuente porque WIKI termina en 2013.

La {download}`matriz de pendientes <../_static/e3_completion_20261120/protocol_power/requirements_matrix.csv>`
y el {download}`preflight CPU reanudable <../_static/e3_completion_20261120/protocol_power/preflight.py>`
dejan esos requisitos preparados para revisión. Las ocho pruebas de
fronteras matemáticas, bloqueo y reanudación pasaron; el preflight actual
devuelve **`BLOCKED_PREPARATION_ONLY`** y no tiene una ruta que entrene.
Preparar el documento no congela el protocolo ni acredita R3. Se conserva
el uso de IA declarado y la separación entre dictamen documental de R1,
defensa académica de R2 y validación predictiva de R3.

(e3-followup-scientific-decision)=
## 8.23 Cierre de la decisión sobre V3, posterior a la publicación

La auditoría documental de ese corte **cierra la decisión de continuar con NO-GO para la confirmación de la receta actual**. No cierra R3 favorablemente. Cotejó 65 entradas por hash y recalculó desde los resultados guardados el piloto adverso, las 51 trayectorias de transferencia, sus 20.400 filas y 21 medias descriptivas. No ejecutó ajustes, inferencias, remuestreos ni nuevas pruebas de significación; consumió aproximadamente 0,12 s CPU y cero GPU. La revisión posterior de {ref}`8.24 <e3-resolution-science-20261120>` audita los escenarios de planificación y prepara sus contratos sin cambiar ese dictamen.

`atom_crps` conserva **2,7043119888 % más RMSE que `atom_mse_only`**, sin superar la heurística original del 2 %. En la transferencia posterior tampoco fue mejor que el control observado en ninguno de los tres paneles: el exceso descriptivo de error frente al mejor control fue 26,06 % en D01, 8,96 % en D02 y 2,61 % en D03. Esos resultados no prueban inferioridad universal o equivalencia. La pregunta de transferencia ya fue contestada con calendarios expuestos; repetirla no proporcionaría confirmación independiente. No se promueve la ablación ni se cambia el umbral, la pérdida o el rival para declarar éxito.

Se revisó la guía original: el Cuadro 4 denomina la ruta **F2. Temporal**; «F2a» es una denominación interna de documentos anteriores. El apartado 8.7.2 requiere un piloto de al menos cinco semillas y al menos diez finales, ampliables según potencia. Los tres ajustes finales por receta del piloto actual no cumplen aquel mínimo de planificación. La guía exige potencia **≥0,90 por contraste y potencia conjunta reportada**; no impone automáticamente ≥0,90 a la conjunta. Los 486 escenarios existentes son sensibilidades analíticas condicionales; faltan las varianzas pertinentes de dieciséis rivales E2 y no existe un R o calendario final justificado para los diecinueve rivales y 57 componentes propuestos.

Una pregunta futura sobre calidad distributiva en regímenes definidos antes de ver datos nuevos podría motivar otra investigación, pero no autoriza ahora una campaña. Requeriría justificación científica, datos admisibles no expuestos, batería temporal validada, márgenes operativos, potencia y presupuesto suficientes, y registro real previo. Si cambia el algoritmo, será otra versión y deberá revisarse qué parte de R1 conserva. La preparación Colab anterior mantiene sus guardas y no habilita entrenamiento. **R3 permanece no acreditada**; el dictamen de R1 de V3 sigue siendo técnico, acotado y distinto de R2.

Evidencia nueva: {download}`decisión razonada <../_static/e3_science_followup_20261120/SCIENTIFIC_DECISION.md>`, {download}`decisión estructurada <../_static/e3_science_followup_20261120/decision.json>`, {download}`recibo de los recálculos <../_static/e3_science_followup_20261120/execution_receipt.json>`, {download}`matriz científica <../_static/e3_science_followup_20261120/requirements_matrix.csv>` y {download}`condiciones concretas de reapertura <../_static/e3_science_followup_20261120/REOPENING_CHECKLIST.md>`. La asistencia de IA está declarada y no se simula una aprobación institucional.


(e3-resolution-science-20261120)=
## 8.24 Auditoría de potencia y diseño futuro bloqueado

Este corte conservó las tres propuestas como DRAFT. La implementación y validación controlada posteriores de dos extensiones direccionales y un MCS global se registran en {ref}`8.25 <e3-next-statistics-data-20261121>`, con sus resultados adversos y sin dar por completadas tres pruebas por contraste.

**Estado científico: NO-GO para confirmar la receta actual y R3 no acreditada.** La revisión parte del commit `8d03a73d529eb3f9b2e89992b4488cfeb88a99e6`; conserva los **796 trabajos de la aditiva, 30 ajustes V3 y 51 trayectorias/20.400 predicciones retrospectivas**. `atom_crps` mantiene **2,7043119888 % más RMSE que `atom_mse_only`**, sin superar su heurística ni al mejor control en ninguno de los tres dominios de transferencia expuestos. Se cierra una auditoría y preparación verificables, no una justificación nueva para abrir confirmación. No se cambia la regla ni se promueve una ablación.

La {download}`auditoría ejecutada de los escenarios <../_static/e3_resolution_20261120/science/audit_486.json>` concilia **486 registros guardados: 162 de sensibilidad entre semillas y 324 de calendario**. Se verificaron **71 archivos de entrada por hash**, según el {download}`recibo de ejecución <../_static/e3_resolution_20261120/science/execution_receipt.json>` y la {download}`procedencia de insumos <../_static/e3_resolution_20261120/science/input_provenance.json>`. No se repitieron los escenarios ni se generaron nuevas trayectorias, p-valores científicos, remuestreos de mercado o datos de confirmación. Los rangos condicionales de 10–1032 pares y 4,73–38.800,24 fechas mezclan supuestos ilustrativos: **no recomiendan un R o calendario final**.

### Familia completa, variabilidad ausente y estimando

La {download}`tabla nominal de la familia <../_static/e3_resolution_20261120/science/main_family_variance_inputs.csv>` conserva **19 rivales y 57 componentes** frente a V3: los dieciséis sistemas E2, la aditiva y dos controles físicos. Sólo **tres rivales principales** tienen referencias de varianza del piloto: aditiva y los dos controles. Las otras tres referencias de los seis controles son ablaciones; no cubren a los **dieciséis rivales E2 faltantes**. Además, las referencias disponibles proceden del desarrollo BTC expuesto. La {download}`plantilla futura de varianza <../_static/e3_resolution_20261120/science/future_variance_template.csv>` deja **76 celdas —cuatro paneles por diecinueve rivales—** pendientes, sin sustituir ausencias por cero.

La propuesta de estimando poblacional futuro distingue inicialización y proceso temporal:

$$
\theta_{j,p}=\mathbb{E}_{S}\!\left[
\sqrt{\mathbb{E}_{P_p}(e_{j,S,t}^{2})}
-\sqrt{\mathbb{E}_{P_p}(e_{\mathrm{V3},S,t}^{2})}
\right],\qquad H_{0,j,p}:\theta_{j,p}\leq\delta_{j,p}.
$$ (eq-e3-resolution-population-target)

Aquí $j$ identifica al rival, $p$ al panel, $S$ la inicialización y $P_p$ el proceso temporal por especificar; los errores conservan la unidad de σ₇ poblacional sin anualizar. Es una **propuesta pendiente de revisión**, no una población recuperada de las 55 fechas conocidas. El estimador es la media de diferencias de **RMSE individuales**, no RMSE de un ensamble ni raíz de MSE agrupado entre semillas. Las celdas semilla×fecha no son observaciones independientes. Remuestrear sólo fechas no cubre por sí solo la inicialización; comparar sólo semillas condiciona al calendario. La covarianza temporal, entre semillas, su interacción y la dependencia entre rivales requieren identificación.

El margen práctico $\delta_{j,p}$ y la alternativa de planificación $\Delta_{\mathrm{alt},j,p}>\delta_{j,p}$ son decisiones diferentes. La separación contra el nulo es $\Delta_{\mathrm{alt},j,p}-\delta_{j,p}$; simular exactamente en $\delta$ es estudiar la frontera nula, no una alternativa con potencia 0,90. El margen BTC heredado sigue sin justificación operativa para V3; el 2 % del piloto y la ganancia observada no lo fijan. **Márgenes por panel, MDE, R y T finales permanecen abiertos.**

### Tres procedimientos: propuestas DRAFT, sin batería validada

La {download}`revisión de métodos <../_static/e3_resolution_20261120/science/METHODS_REVIEW.md>` prepara opciones para el mismo nulo unilateral:

| Propuesta futura | Validación que falta |
|---|---|
| P1: bootstrap pareado por bloques del funcional RMSE | Recalcular el funcional completo, justificar la variación cruzada semilla/fecha, los bloques, la studentización y la calibración nula. |
| P2: expansión delta del RMSE con HAC | Derivar componentes entre semillas, calendario e interacción, con momentos, denominadores y dependencia válidos. DM sobre MSE no prueba automáticamente la diferencia de RMSE; una corrección HLN requiere revisión específica. |
| P3: subsampling pareado del mismo funcional | Justificar unidades, tamaños, tasas y distribución límite para semillas y calendario, además de su distinción respecto a P1. **Subsampling no figura con ese nombre en el Cuadro 4**: su admisibilidad o equivalencia debe justificarse antes de congelar el protocolo. |

P3 es una alternativa metodológica posterior del borrador futuro; no reemplaza retrospectivamente las reglas del piloto ni se declara aprobada. GW puede conservarse como diagnóstico condicional auxiliar, pero un rechazo omnibus no establece automáticamente dirección favorable y margen unilateral de RMSE. El horizonte de un día no elimina el solapamiento de ocho cierres de σ₇. Ninguna de estas tres pruebas está implementada y validada por el lector de potencia entregado.

La familia principal propuesta ajusta conjuntamente **57 p-valores mediante Holm a 0,05**. Cada rival requiere la intersección de los tres procedimientos ajustados y la dirección práctica prefijada. Antes de dimensionar potencia faltan validaciones de tamaño/FWER bajo frontera, nulos interiores y parcialmente nulos, dependencia temporal, shocks comunes, heterogeneidad entre semillas y colas. El diseño de multiplicidad entre los cuatro paneles y de la decisión final entre dominios también debe aprobarse antes de congelar.

### Preparación ejecutable y puertas que siguen cerradas

El {download}`contrato CPU <../_static/e3_resolution_20261120/science/power_contract.py>` deja implementada la agregación **futura** de p-valores de simulación: un panel y presupuesto por archivo, familia completa por réplica, Holm, intersección por rival y conjunta en la misma réplica. No multiplica potencias marginales ni aplica pruebas nuevas al mercado. Reporta límites binomiales exactos Clopper–Pearson para la precisión Monte Carlo condicional; esto no valida el generador ni P1/P2/P3. Las **doce pruebas de software** pasaron. La {download}`QA independiente declarada con IA <../_static/e3_resolution_20261120/qa/SCIENCE_QA.json>` contrastó además dieciséis valores de referencia exactos con SciPy: discrepancia máxima **1,38×10⁻¹⁴**, dentro de tolerancia 10⁻⁹. Los fixtures no son evidencia de potencia científica.

Desde la carpeta científica extraída, Python estándar permite inspeccionar el bloqueo:

```bash
python power_contract.py preflight \
  --config prospective_design.DRAFT.json \
  --output /tmp/e3-v3-preflight.json --prepare-only
```

El {download}`recibo del preflight ejecutado <../_static/e3_resolution_20261120/science/preflight_receipt.json>` informa **`BLOCKED_PREPARATION_ONLY`**, ocho puertas pendientes y `confirmation_enabled=false`. Sin `--prepare-only` sale con código 2; con la opción sólo prepara el reporte. **El programa no puede entrenar** y ningún booleano de un JSON sustituye la validez científica. El {download}`diseño futuro DRAFT <../_static/e3_resolution_20261120/science/prospective_design.DRAFT.json>` y el {download}`piloto separado de planificación <../_static/e3_resolution_20261120/science/PLANNING_PILOT.md>` conservan un mínimo de **cinco semillas piloto por receta estocástica y diez finales**, ampliables por dimensionamiento. La guía exige **≥0,90 por contraste y reportar la potencia conjunta**, sin imponer ≥0,90 a la conjunta. Como referencia aritmética, cuatro paneles×quince redes×cinco semillas serían 300 ajustes antes de HPO nuevo; 120 s por ajuste sumarían diez horas nominales, no consumo medido ni presupuesto final. Esos ajustes no se lanzaron.

La {download}`auditoría de elegibilidad <../_static/e3_resolution_20261120/science/data_eligibility.json>` conserva D00–D03 como **admitidos para sus usos exploratorios**, con exposición ya conocida; ninguno se certifica como confirmación no expuesta. F00–F03 sólo describen clases de dominio por evaluar —BTC futuro, renta variable, materias primas y tipos—: fuente, instrumento, licencia, fechas y hashes siguen nulos. No son cuatro conjuntos nuevos seleccionados o admitidos. Para tres adicionales se requieren ejes de diferencia documentados, disponibilidad histórica, dependencia económica/calendario, soporte de ocho cierres, particiones absolutas, purga y custodia/exposición. Cambiar ticker o fuente no recupera independencia.

La {download}`decisión científica de este corte <../_static/e3_resolution_20261120/science/scientific_decision.json>` mantiene **NO-GO**: faltan justificación renovada de la candidata, datos no expuestos, tratamiento/capacidad/presupuesto, márgenes, covarianzas, batería válida y potencia final. **No se generaron Colab confirmatorio ni tag de protocolo**. La preparación es posterior a resultados, con asistencia de IA declarada; no constituye preregistro, aprobación institucional, reproducción humana o acreditación de R3.


(e3-next-statistics-data-20261121)=
## 8.25 Batería implementada, calibración controlada y admisión todavía bloqueada

La preparación de 8.24 avanzó a código y **validación estadística controlada realmente ejecutada**. Se mantienen el NO-GO de V3 y los resultados de mercado anteriores. Este nuevo trabajo distingue las guardas de software, las simulaciones con verdad conocida y una confirmación que sigue sin estar habilitada; no atribuye la calibración del diseño V3 automáticamente a la revisión V4.

### Lectura literal de la guía y alcance de las extensiones

La guía **§8.4, pp. 23–24**, pide tres pruebas distintas del Cuadro 4 que rechacen igualdad de desempeño para cada contraste, métricas favorables, diferencia superior al margen práctico δ y tamaño de efecto no despreciable. El límite inferior del IC95 debe ser mayor que cero e idealmente mayor que δ. **No exige literalmente que los tres p-valores correspondan al mismo nulo unilateral `theta≤delta` ni que cubran idénticas fuentes de incertidumbre.** Ese nulo más fuerte y la intersección homogénea son una elección propuesta del DRAFT anterior, que debe justificarse y cuya potencia debe planificarse; no se presentan como cita literal de la guía.

El Cuadro 4 incluye bootstrap estacionario, DM-HLN y MCS. MCS puede contribuir bajo una traducción explícita de su decisión global a la familia y al criterio de superioridad de la guía. **En esta implementación concreta** no hay esa traducción contrastiva justificada y la calibración presenta un resultado adverso; no se lo cuenta como tercer procedimiento aprobado. Tampoco subsampling se considera automáticamente una opción nombrada por el cuadro. La {download}`nota metodológica de las implementaciones <../_static/e3_next_evidence_20261121/statistics/METHODS_IMPLEMENTED.md>` separa esas condiciones.

El {download}`módulo estadístico ejecutable <../_static/e3_next_evidence_20261121/statistics/temporal_battery.py>` recibe errores cuadrados pareados `semilla×fecha×sistema` y estima la media de diferencias de RMSE individuales. Implementa estas **extensiones propuestas**, sin atribuirles la validez de un método clásico sólo por usar su nombre:

| Componente implementado | Alcance y límite |
|---|---|
| P1: bootstrap estacionario cruzado del funcional RMSE | Remuestrea semillas completas y un calendario de bloques común a todos los sistemas, recalculando cada raíz antes de promediar. Usa raíces centradas e IC básicos; no afirma una studentización anidada ausente. La interacción puede sobrecontarse en muestra finita. |
| P2: expansión delta cruzada, HAC y factor HLN | Separa estimaciones entre semillas/calendario y reporta interacción. La suma de componentes se conserva como aproximación superior bajo el modelo aditivo propuesto; no demuestra una identificación exacta para cualquier funcional. La referencia t, HAC y factor de horizonte necesitan revisión específica del diseño. |
| MCS global del funcional | Implementa eliminación por rango máximo con raíces bootstrap centradas y p de exclusión monótono. Su nulo es igualdad global de riesgos; aquí no produce por sí solo un p práctico pareado ni completa las tres pruebas requeridas. |

Las guardas computacionales de al menos diez semillas y 32 fechas no certifican suficiencia, asintótica o potencia. La unidad fija y la tolerancia numérica pueden clasificar escalas extremadamente pequeñas como no soportadas; no se reajustan después del resultado para generar rechazos.

### Simulación fijada antes de observar sus resultados

El [registro público `550b6d26cbd1b82f40a347c24b180320776e5ca3`](https://github.com/jassan-bit/DeepLearning_202630/commit/550b6d26cbd1b82f40a347c24b180320776e5ca3), el {download}`plan controlado <../_static/e3_next_evidence_20261121/statistics/CONTROLLED_VALIDATION_PLAN.json>` y la {download}`identidad de implementación congelada <../_static/e3_next_evidence_20261121/statistics/IMPLEMENTATION_FREEZE.json>` preceden a los resultados. Se completaron **1.024 réplicas en once casos**, con **2.047 réplicas bootstrap por réplica de DGP: 2.096.128 sorteos controlados en total**. El {download}`recibo ejecutado <../_static/e3_next_evidence_20261121/statistics/controlled_validation_run/execution_receipt.json>` registra aproximadamente **40,86 s de pared en CPU**, desde las 19:50:22 hasta las 19:51:03 de Bogotá del 20 de noviembre —UTC del día 21—. Estos sí son nuevos remuestreos de DGP controlados, distintos de las diecinueve guardas de software y de cualquier remuestreo de mercado.

Se conservaron todos los casos previstos: igualdad global, frontera práctica, nulo interior, nulos parciales, alternativa, colas pesadas, variación sólo entre semillas, sólo de calendario, dependencia fuerte y escalados separados de R/T. Los shocks compartidos, heterogeneidad de semillas y dependencia temporal no se trataron como observaciones IID. El caso de colas t5 tiene segundo momento unitario conocido; su distribución estacionaria completa se aproxima con burn-in y no se declara exacta.

**Resultado adverso principal.** Bajo igualdad de riesgos del caso `EPA_equal`, el MCS extendido rechazó inicialmente **27 de 128 réplicas: 21,09 %**, frente al nominal del **10 %**, con intervalo binomial exacto individual **[14,38 %; 29,19 %]**. Retuvo todos los sistemas verdaderamente mejores en 101/128 réplicas, **78,91 %**. P1/P2 también mostraron señales puntuales de tamaño o cobertura insuficientes en algunos escenarios. Son límites Monte Carlo **individuales, no simultáneos sobre escenarios/rivales**; se conservan sus denominadores y no se seleccionan sólo casos favorables. La conclusión concierne a estas extensiones/configuración y no refuta la teoría clásica de MCS o DM.

El {download}`resumen de todos los casos <../_static/e3_next_evidence_20261121/statistics/validation_compact.csv>`, los {download}`resultados por rival <../_static/e3_next_evidence_20261121/statistics/per_rival_validation.csv>` y la {download}`decisión posterior <../_static/e3_next_evidence_20261121/statistics/VALIDATION_DECISION.json>` mantienen **`method_configuration_approved_for_confirmation=false`**. No se retocaron métodos después de ver este resultado. La {download}`QA posterior declarada con IA <../_static/e3_next_evidence_20261121/qa/STATISTICAL_POSTRUN_QA.json>` conserva límites y archivos congelados; no se presenta como aprobación institucional.

### Familias y potencia que todavía no quedan acreditadas

La validación controlada fue fijada para **veinte sistemas y diecinueve rivales**. Su diagnóstico Holm38 ajusta los dos procedimientos direccionales por rival; también informa una reserva conservadora Holm57 con diecinueve casillas `p=1`. **Esas casillas no son una tercera prueba**, y ni esa reserva ni MCS completan la decisión de tres procedimientos del protocolo futuro. Holm38 identifica el alcance controlado previamente declarado; no reduce después del resultado la familia confirmatoria antigua para declarar éxito.

Los conteos **19 rivales/57 componentes/76 celdas** de 8.24 pertenecen al diseño histórico V3. El {download}`registro futuro V4 condicional <../_static/e3_next_evidence_20261121/planning/V4_FAMILY_AND_POWER.DRAFT.json>` conserva esa historia y propone **21 rivales/63 casillas/84 celdas panel-rival**, añadiendo V3 original y su control MSE-only; no los retira por sus rankings de desarrollo. La {download}`plantilla V4 de varianzas pareadas <../_static/e3_next_evidence_20261121/planning/V4_future_paired_variance_template.csv>` mantiene las **84 celdas sin estimaciones prospectivas**: las tres referencias antiguas de V3 no son diferencias pareadas frente a V4. La validación de veinte sistemas no se transfiere a los veintidós propuestos. Un mínimo ilustrativo de cuatro paneles×diecisiete recetas estocásticas×cinco semillas sería **340 ajustes**, no presupuesto final o trabajo ejecutado. El piloto V4 acotado de cuatro recetas y cinco semillas no llena esa familia ni los cuatro paneles; métodos, multiplicidad, R/T y decisión de continuar deben fijarse antes de cualquier confirmación.

Faltan revisión del tercer procedimiento pertinente y de la traducción global, calibración adecuada de la regla completa, varianzas y covarianzas reales admitidas, márgenes/alternativas operativos, MDE y R/T finales. **No hay potencia ≥0,90 acreditada por contraste de la familia completa.** La guía exige reportar la conjunta, sin imponerle universalmente 0,90. Los 486 escenarios anteriores siguen siendo antecedentes condicionales; estas 1.024 réplicas no son su validación retrospectiva ni potencia de mercado.

### Piloto real V4 de desarrollo completado y receta abandonada

El {download}`resultado del nuevo piloto CPU <../_static/e3_next_evidence_20261121/development/cpu_pilot_results/pilot_result.json>` completa **veinte ajustes reales** de cuatro recetas con cinco semillas y evalúa dos controles analíticos una sola vez. Son **22 trayectorias sobre 55 fechas expuestas y 1.210 filas nuevas de evaluación de desarrollo**, no 1.210 observaciones independientes. El [anclaje público previo de plan y fuentes](https://github.com/jassan-bit/DeepLearning_202630/commit/5ee55723e7d1ce63dbe4a94dca8941cb85a45fee) precedió al primer ajuste; los veinte checkpoints se validaron antes de producir las evaluaciones. No hubo HPO, confirmación o uso de pesos GPU antiguos para comparar recetas nuevas CPU.

| Receta/control CPU | RMSE media individual | DE muestral entre semillas | Trayectorias |
|---|---:|---:|---:|
| V4 `sigma_crps_v4` | 0,002087524306 | 0,000017087831 | 5 |
| `u_crps_control` | 0,002071168327 | 0,000022601117 | 5 |
| `mse_only_control` | **0,002057824526** | 0,000026740769 | 5 |
| `additive_positive_head` | 0,002190798772 | 0,000024041513 | 5 |
| `floor_only` | 0,002790340619 | No aplica | 1 |
| `floor_plus_mean_excess` | 0,002114230941 | No aplica | 1 |

La {download}`tabla completa con MAE y parámetros <../_static/e3_next_evidence_20261121/development/cpu_pilot_results/recipe_summary.csv>` y las {download}`métricas por trayectoria <../_static/e3_next_evidence_20261121/development/cpu_pilot_results/per_trajectory_metrics.csv>` conservan todas las recetas. V4 tuvo **1,443261 % más RMSE que el mejor control**, MSE-only, frente a la mejora mínima previa del 2 % tanto sobre el mejor control neuronal como sobre el mejor de todos los cinco. **Falló ambas puertas: `ABANDON_CURRENT_V4_RECIPE`.** No se promociona MSE-only, no se cambia el criterio después del resultado ni se abre otra versión para buscar significancia. La geometría sintética de 144 casos no se convierte en una mejora predictiva por seguir siendo matemáticamente distinta.

Son resultados nuevos, separados del piloto original GPU de treinta ajustes y su **2,7043119888 % adverso**. El U-CRPS de esta tabla es un control CPU recién ajustado, no una sustitución de aquellas tres semillas. Las cinco semillas de cuatro recetas sobre un panel expuesto no completan las 84 celdas prospectivas V4, ni la familia completa, potencia, calibración, nuevas admisiones o R1/R3. El registro futuro V4 se conserva como DRAFT condicionado; el abandono de esta receta no habilita su confirmación. Los 796 trabajos, treinta ajustes y 20.400 predicciones anteriores permanecen intactos.

La etapa consumió **90,096 s de extremo a extremo y 54,926 s de pared de las fases**, frente a un máximo nominal permitido de 2.400 s de fase. Se conservan recursos, curvas e identidades de checkpoints; no se confunde ese presupuesto con horas GPU, FLOPs o energía medidos. Los veinte ajustes se cuentan como **nuevos ajustes de desarrollo**, no como cero ni como una reproducción humana de los masters. La {ref}`reproducción automática de 11.12 <e3-next-human-book-20261121>` conserva la edición f5c anterior y no afirma haber reproducido estos veinte ajustes.

### Cuatro propuestas de datos concretas, cero admisiones nuevas

El {download}`registro de admisión DRAFT <../_static/e3_next_evidence_20261121/data_admission/concrete_panels.DRAFT.json>` sustituye clases genéricas por instrumentos y proveedores concretos, sin adquirir precios ni observar desempeño nuevo:

| Panel propuesto | Instrumento y fuente | Reloj y límite material |
|---|---|---|
| F00 principal | BTCUSDT Spot, [Binance Vision 1d](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/) | Barras diarias UTC, volumen BTC; mismo subyacente que D00. La ventana futura aún no está materializada. |
| F01 adicional | SPY, serie diaria de [Alpha Vantage](https://www.alphavantage.co/documentation/#daily), identidad de State Street | Participaciones de renta variable estadounidense; sesiones NYSE Arca. |
| F02 adicional | USO, serie diaria Alpha Vantage, identidad de USCF | Fondo expuesto a futuros de petróleo; no es petróleo spot ni un futuro empalmado por el investigador. |
| F03 adicional | TLT, serie diaria Alpha Vantage, identidad de iShares | Fondo de deuda del Tesoro de largo vencimiento; calendario Nasdaq, sin heredar el conteo NYSE. |

La {download}`comprobación ejecutable de metadatos <../_static/e3_next_evidence_20261121/data_admission/validation_receipt.json>` conserva **52 puertas bloqueadas y cero datasets admitidos**. Los tres fondos comparten proveedor, moneda, infraestructura, gran parte del calendario y factores macroeconómicos: dominios distintos no implican independencia. El dominio y un tamaño programado distinto son ejes propuestos, todavía sujetos a justificar la misma tarea, completitud y política T. Ticker, proveedor o reloj no se suman como ejes independientes para rellenar casillas.

La fórmula conserva siete retornos RAW, ocho cierres, `ddof=0` y una observación futura; **una sesión de bolsa no equivale a 24 horas**. Una próxima sesión puede quedar entre uno y cuatro días de calendario después. El guard histórico de días UTC sigue intacto; no se rellenan fines de semana ni se autoriza por declaración una equivalencia no revisada. Las ventanas propuestas de planificación —objetivos 2026-12-14 a 2027-02-26— y eventual confirmación —2027-03-15 a 2028-03-14— son **DRAFT**, sin R/T suficientes ni congelamiento real. Éste deberá preceder también al soporte RAW confirmatorio, no sólo al primer objetivo.

El {download}`informe concreto de disponibilidad, diversidad y derechos <../_static/e3_next_evidence_20261121/data_admission/README.md>` conserva fuentes primarias, condiciones de Binance y restricciones de acceso/redistribución de Alpha Vantage. No se verificó la respuesta de precios de SPY/USO/TLT, derechos para el uso concreto, disponibilidad al origen o el calendario Nasdaq futuro. Versión, tamaño y hashes de archivos futuros siguen pendientes. El intento de metadatos Binance bloqueado en el proxy no se interpreta como respuesta de su servidor. D00–D03 mantienen su admisión exploratoria previa y exposición; no se renombran como confirmación nueva.

El {download}`validador sin acceso al mercado <../_static/e3_next_evidence_20261121/data_admission/admission_validator.py>` devuelve bloqueo con la declaración real aún vacía. Inspeccionar documentación no admite datasets ni habilita modelos. **La confirmación continúa cerrada** por ciencia, métodos, datos y potencia; no se genera tag o Colab confirmatorio para eludir esas puertas. La preparación y los resultados adversos se declaran con asistencia de IA, sin atribuir una ejecución humana independiente.


(e3-statistical-remediation-20261121)=
## 8.26 Remediación estadística: estimando, calibración y adecuación retrospectiva

Este corte desarrolla el componente de pruebas de hipótesis y adecuación a partir de
las fuentes de `d6cf7f1`. Conserva completos 8.24–8.25, las 1.024 réplicas previas,
su rechazo MCS de 27/128 bajo igualdad y los resultados de desarrollo desfavorables.
No vuelve a entrenar ni a predecir. El NO-GO de V3 y el abandono de V4 continúan;
una mejora de la herramienta de inferencia no modifica esas decisiones.

### Qué exige la guía y qué procedimiento responde a cada pregunta

La {download}`auditoría de la guía original <../_static/e3_statistics_remediation_20261121/guide_power/primary/PRIMARY_GUIDE_FACTS.json>` vuelve al PDF: §8.4, pp. 23–24; Cuadro 4, p. 26; §8.7,
pp. 27–28; y checklist 14.5.3, pp. 44–45. Para R3 se requieren al menos tres
procedimientos distintos de la tabla que rechacen igualdad de rendimiento para
cada contraste, dirección favorable, margen práctico y efectos/intervalos. Al
menos uno debe ser global o controlar multiplicidad. La guía no impone literalmente
tres nulos unilaterales idénticos `theta <= delta`, ni permite contar variantes de MCS como tres procedimientos o convertir
exclusiones automáticamente en p-valores prácticos pareados.

La pregunta primaria sigue siendo la **media de RMSE individuales por semilla**,
con incertidumbre de inicialización y calendario. No se sustituye por la raíz de
la MSE promediada, el RMSE de un ensemble o una pérdida temporal condicionada a
una cohorte fija. Esa distinción puede cambiar incluso un ranking: dos semillas
con MSE `0,01` y `8,41` tienen RMSE media `1,5`, menor que `sqrt(3)=1,73205` de un
rival con MSE `3` en ambas; su MSE media `4,21`, en cambio, es mayor que `3`.
Es un contraejemplo determinista del estimando, no un resultado de mercado.

| Procedimiento nombrado en la guía | Pregunta y condición de uso | Límite para el primario actual |
|---|---|---|
| Bootstrap estacionario | Muestreo temporal débilmente dependiente del funcional declarado, con calendario común a rivales. | Para inferencia aleatoria entre semillas hay que incorporar esa variabilidad y justificar el diseño cruzado. |
| DM-HLN | Diferencia media de pérdidas temporales pareadas, horizonte declarado y LRV apropiada. | Horizonte de un día no elimina el solapamiento de σ₇; la media de RMSE por semilla necesita derivación adicional. |
| MCS | Igualdad global de capacidad predictiva y conjunto de confianza, con bootstrap conjunto válido. | La exclusión de un rival no identifica por sí sola su contraste direccional práctico frente a la candidata. |
| Giacomini–White | Capacidad predictiva condicional con instrumentos y marco de estimación/muestreo justificados. | El nulo condicional y la cohorte fija no equivalen automáticamente al primario aleatorio de RMSE. |
| Clark–West | Modelos realmente anidados y ajuste de pérdida de esa teoría. | Tener distinto número de parámetros no acredita anidamiento entre redes heterogéneas. |
| Friedman por series/orígenes y prueba entre conjuntos | Unidades defendibles y post hoc/multiplicidad especificados. | Un calendario diario correlacionado no se convierte en conjuntos u orígenes independientes. |

MCS no queda excluido universalmente del tercer procedimiento. Si se justifican
la cobertura simultánea del conjunto de mejores y su correspondencia con la
familia, un conjunto que conserve únicamente la candidata puede respaldar una
decisión global frente a los rivales, junto con dirección, efectos y margen
práctico. Las exclusiones de la extensión funcional aquí estudiada no reciben
esa interpretación automáticamente ni se convierten en p-valores pareados.

Las guardas de diez semillas y 32 fechas siguen siendo límites de soporte
computacional, no evidencia de potencia o de suficiencia asintótica. Las cinco
semillas del piloto V4 no se usan para eludirlas. Los métodos que cambien el
estimando o condicionen la cohorte se identifican por separado.

### Diagnóstico conservado y versión metodológica nueva

La revisión de los registros guardados no identificó un error algebraico evidente de signo, centrado o regla de eliminación. La dispersión bootstrap fija también se usa en el MCS estándar y no se etiqueta como bug por sí sola. Bajo EPA los veinte modelos sintéticos son intercambiables: el sesgo finito de la raíz es común y **no explica por sí solo** el rechazo 27/128. Pocas semillas, la dimensión del máximo —190 pares con doce semillas—, dependencia temporal y la extensión no lineal/cruzada son mecanismos plausibles, no una causa única identificada.

El {download}`módulo v2 del MCS funcional <../_static/e3_statistics_remediation_20261121/calibration/crossed_rmse_v2.py>` mantiene el riesgo $\widehat R_m=S^{-1}\sum_s\sqrt{\bar q_{sm}}$ y usa una covarianza completa de riesgos: covarianza entre RMSE de semillas dividida por S más HAC de la influencia temporal media dividida por T. Para el par j/k, $\widehat\sigma_{jk}^2=\widehat\Sigma_{jj}+\widehat\Sigma_{kk}-2\widehat\Sigma_{jk}$. En cada réplica se recalculan riesgos y covarianza, de modo que el pivote propuesto es

$$
Z^*_{b,jk}=\frac{(\widehat R^*_{b,j}-\widehat R^*_{b,k})-(\widehat R_j-\widehat R_k)}{\widehat\sigma^*_{b,jk}}.
$$ (eq-e3-crossed-mcs-bootstrap-t-20261121)

La suma de componentes puede contar dos veces interacción no negativa **dentro de la linealización cruzada aditiva**; con ancho de banda finito, raíz no lineal o dependencia arbitraria no es una cota garantizada. Se conservan bloque medio **12** y HAC **8**, incluso en el estrés con $\phi=0,85$. La guarda numérica nueva es relativa a la escala; muestras degeneradas se declaran no soportadas, sin intervalos de certeza fabricados. El factor HLN de horizonte uno y la referencia t con S−1 se conservan para P2 como extensión propuesta, sin atribuirle un teorema clásico de DM para este diseño.

La comparación incluye tres variantes sobre **los mismos sorteos bootstrap**: v1 congelado con su dispersión original; covarianza nueva con studentizador fijo; y covarianza nueva con re-studentización bootstrap-t. La segunda frente a la tercera aísla la re-studentización dentro de la nueva covarianza; v1 frente a v2 contiene dos intervenciones y no identifica una causa única. Son variantes de un complemento global, **no tres procedimientos primarios distintos**. Los dos contrastes por rival reportan ahora igualdad bilateral $\theta=0$; dirección, IC y margen práctico se examinan por separado, sin reetiquetar el nulo práctico más fuerte de la versión previa.

El [commit público `7d304faa0dd583c6109a688ff179ec7e937f6561`](https://github.com/jassan-bit/DeepLearning_202630/commit/7d304faa0dd583c6109a688ff179ec7e937f6561), el {download}`plan nuevo <../_static/e3_statistics_remediation_20261121/calibration/VALIDATION_PLAN.json>` y el {download}`manifiesto de ocho fuentes de ejecución/pruebas <../_static/e3_statistics_remediation_20261121/calibration/CODE_MANIFEST.json>` preceden al Monte Carlo nuevo. Se fijaron cuatro casos, **640 réplicas** y **999 remuestreos por réplica** —639.360 previstos—, con semilla distinta de la validación anterior, dos hilos y máximo de 300 s. Las **30 guardas de software** se distinguen de esa validación estadística.

El criterio previo de desarrollo exige en cada caso EPA un **límite superior del IC95 binomial individual ≤0,13**, para nominal 0,10 y tolerancia exploratoria de 0,03. No certifica nivel exacto 0,10 ni una garantía simultánea. El plan también conserva FWER del subconjunto Holm38, IC puntuales de P1/P2, seis diagnósticos pareados de mecanismo con Holm6 y un escenario alternativo; no es una batería completa de tres métodos o potencia final de mercado. Un caso de estrés desfavorable restringe el alcance, sin retocar bloque o lag después de observarlo.

### Validación controlada ejecutada y decisión de calibración

Se completaron **640/640 réplicas y 639.360 remuestreos cruzados**, con cero errores y las ocho fuentes congeladas intactas. El {download}`recibo de ejecución <../_static/e3_statistics_remediation_20261121/calibration/validation_fresh/execution_receipt.json>` registra **187,684 s de pared**, del 20 de noviembre a las **22:21:52–22:24:59 de Bogotá** —UTC del 21—, dentro del límite previo de 300 s. El control v1 reutilizó los riesgos de los mismos sorteos y no añadió remuestreos; no son tres experimentos independientes.

| Caso nulo EPA nuevo | MCS v1 sobre sorteos nuevos | Covarianza nueva, escala fija | Propuesta bootstrap-t | IC95 % individual de la propuesta |
|---|---:|---:|---:|---:|
| Ordinario, φ=0,35 | 62/256 —24,22 %— | 56/256 —21,88 %— | **21/256 —8,20 %—** | **5,15–12,27 %** |
| Dependencia fuerte, φ=0,85 | 36/128 —28,13 %— | 33/128 —25,78 %— | **4/128 —3,13 %—** | **0,86–7,81 %** |
| Variación sólo entre semillas | 90/128 —70,31 %— | 92/128 —71,88 %— | **2/128 —1,56 %—** | **0,19–5,53 %** |

La propuesta pasó los **tres cribados exploratorios individuales** previamente fijados: sus límites superiores no superan 0,13. El control de escala fija nueva no consiguió ese resultado. Esto respalda investigar la re-studentización dentro de la nueva covarianza y delimita una mejora de tamaño observada en estos DGP; **no certifica nivel exacto 0,10, validez universal o una batería confirmatoria completa**. Se conservaron las seis comparaciones pareadas predeclaradas de mecanismos con Holm6, sin escoger variantes después por sus resultados de mercado. Los límites binomiales son individuales y los casos no son una garantía simultánea.

```{figure} ../_static/e3_statistics_remediation_20261121/editorial/figures/rendered/mcs_fresh_global_rates.png
:name: fig-e3-mcs-fresh-global-20261121
:width: 100%

Tasas globales de los cuatro casos declarados y tres variantes sobre los mismos sorteos, con IC95 % Monte Carlo individuales. Las líneas nominal/cribado sólo aparecen en los nulos EPA; el rechazo del caso alternativo no es tamaño ni potencia de tres procedimientos. Figura posterior a resultados, sin recalcular pruebas o remuestreos. Se conservan el {download}`SVG <../_static/e3_statistics_remediation_20261121/editorial/figures/rendered/mcs_fresh_global_rates.svg>`, el {download}`programa <../_static/e3_statistics_remediation_20261121/editorial/figures/plot_validation_rates.py>` y el {download}`recibo de datos/figura <../_static/e3_statistics_remediation_20261121/editorial/figures/rendered/FIGURE_RECEIPT.json>`.
```

**Resultado adverso que mantiene bloqueada la batería.** La familia de **38 contrastes**, P1/P2 frente a diecinueve rivales, no quedó calibrada en todos los casos. Bajo variación sólo entre semillas rechazó alguna nula verdadera en **14/128 réplicas, 10,94 %**, con IC95 % **[6,11 %; 17,67 %]**, frente al nominal del **5 %**: el límite inferior supera el nominal. Con dependencia fuerte registró **9/128, 7,03 %**, IC95 % **[3,27 %; 12,93 %]**; su cribado de límite superior ≤0,08 fue **inconcluso**, no aprobado. El caso ordinario registró 4/256, 1,56 %, IC95 % [0,43 %; 3,95 %]. Holm no corrige la falta de validez marginal de los p que recibe; estas aproximaciones finitas no se certifican por pasar guardas de software.

En la alternativa sintética prefijada, el MCS nuevo rechazó igualdad global en **112/128 —87,50 %—**, pero sólo conservó un conjunto exclusivamente candidato en **39/128 —30,47 %—**, IC95 % [22,65 %; 39,22 %]. Retuvo al verdadero mejor en 128/128. La intersección conjunta del **subconjunto de dos métodos frente a diecinueve rivales** ocurrió en **18/128 —14,06 %—**, IC95 % [8,55 %; 21,31 %]. Esa intersección cuenta sólo p ajustados y no incorpora la regla completa de margen, efecto y adecuación. Los ámbitos son distintos: ni el rechazo global ni esa intersección son potencia de una batería completa de tres procedimientos o dimensionamiento de mercado.

El {download}`resumen íntegro de cuatro casos <../_static/e3_statistics_remediation_20261121/calibration/validation_fresh/validation_summary.json>` y la {download}`decisión posterior conservada <../_static/e3_statistics_remediation_20261121/calibration/RESULTS_DECISION.md>` mantienen la conclusión: mejora exploratoria del complemento global en los casos ensayados; **familia primaria incompleta/no calibrada y confirmación cerrada**. No se cambiaron bandas, bloques, hipótesis, escenarios o semillas después del resultado, ni se abrió otra validación para buscar una aprobación.

La comprobación de software responde a errores de código y manejo de entradas;
la simulación controlada examina tamaño, cobertura y comportamiento bajo los DGP
especificados. Ninguna de las dos certifica universalmente el método. Las tasas
Monte Carlo conservan denominador, precisión e interpretación individual o
simultánea declarada. Que un intervalo contenga el nominal no prueba calibración.

### Adecuación retrospectiva sobre V4 guardada

El [anclaje público `e8485812241e492f683f0e6ea77459ca3423f30c`](https://github.com/jassan-bit/DeepLearning_202630/commit/e8485812241e492f683f0e6ea77459ca3423f30c) precede al nuevo cálculo, pero sucede después de conocer el calendario y el resultado del piloto. Fija la {download}`familia retrospectiva <../_static/e3_statistics_remediation_20261121/retrospective/analysis_plan.json>`, las fuentes y **46 entradas verificadas**; no recupera independencia o prospectividad confirmatoria. El {download}`recibo realmente ejecutado <../_static/e3_statistics_remediation_20261121/retrospective/results/execution_receipt.json>` registra el cálculo a las **22:15:20 de Bogotá del 20 de noviembre** —`2026-11-21T03:15:20Z`—, con los archivos anteriores intactos.

Se calcularon **88 p asintóticos**, cuatro diagnósticos por cada una de las 22 trayectorias, y se aplicó Holm a las **88 posiciones predefinidas**, sin posiciones no soportadas en esta ejecución:

| Diagnóstico auxiliar | Contrastes calculados | Rechazos después de Holm88 |
|---|---:|---:|
| Ljung–Box a 7 rezagos | 22 | 0 |
| Ljung–Box a 14 rezagos | 22 | 0 |
| ARCH-LM a 7 rezagos | 22 | 0 |
| Mincer–Zarnowitz conjunto, HAC Bartlett/HC1 a 14 | 22 | **13** |

El {download}`resumen con alcances y decisión <../_static/e3_statistics_remediation_20261121/retrospective/results/analysis_summary.json>` y los {download}`88 diagnósticos con p brutos y ajustados <../_static/e3_statistics_remediation_20261121/retrospective/results/diagnostics_Holm88.csv>` conservan esos resultados. Los trece rechazos de MZ corresponden a cinco de cinco trayectorias aditivas, ambos controles analíticos y dos de cinco de cada una de MSE-only, V4 sigma y U-CRPS. MZ contrasta intercepto cero y pendiente uno en una regresión auxiliar de objetivo sobre pronóstico, con transformación coherente de unidades y covarianza; **no reentrena el predictor**. La ejecución realizó 44 regresiones OLS auxiliares para ARCH/MZ y cero ajustes o forwards de modelos de pronóstico.

Los rechazos son evidencia retrospectiva contra esas restricciones lineales **bajo los supuestos del procedimiento**, no una declaración universal de inutilidad predictiva. La falta de rechazos de Ljung–Box/ARCH después de Holm no acredita adecuación. Son diagnósticos clásicos asintóticos con sólo **55 fechas**; Ljung–Box no es robusto por sí mismo a heterocedasticidad y no se validó aquí su tamaño finito. El solapamiento de σ₇ motiva examinar dependencia, pero no obliga a que un error óptimo de predicción a un paso sea serialmente correlacionado: bajo la esperanza condicional apropiada sería una diferencia de martingala. No se explica automáticamente un rechazo por compartir retornos.

Los {download}`44 IC de sesgo residual medio <../_static/e3_statistics_remediation_20261121/retrospective/results/conditional_mean_error_CIs.csv>` son puntuales, asintóticos y condicionados a cada trayectoria, con HAC a 7/14; no son IC de RMSE de la población de semillas ni simultáneos. Los {download}`estimandos separados por receta <../_static/e3_statistics_remediation_20261121/retrospective/results/recipe_estimands.csv>` muestran media de RMSE individuales, DE entre semillas y RMSE del pronóstico promedio con sus nombres distintos. La {download}`sensibilidad mensual <../_static/e3_statistics_remediation_20261121/retrospective/results/monthly_sensitivity.csv>` informa septiembre 7–30 —24 fechas— y octubre 1–31 —31—, sin tratarlos como repeticiones independientes.

La {download}`diferencia pareada descriptiva entre semillas <../_static/e3_statistics_remediation_20261121/retrospective/results/paired_seed_effects_descriptive.csv>` conserva V4 menos MSE-only: media **2,969978×10⁻⁵ σ₇**, DE **2,340272×10⁻⁵** y Cohen dz **1,269** en las cinco semillas, con las cinco diferencias positivas. Ese dz describe cinco repeticiones sobre un único calendario expuesto; no es un p de superioridad, un IC poblacional, potencia o requisito confirmatorio acreditado. La revisión **no revierte el abandono de V4**.

El análisis conserva las 55 fechas y las 22 trayectorias del desarrollo expuesto:
cuatro recetas neuronales con cinco semillas y dos controles analíticos. Esas
trayectorias corresponden a 1.210 filas guardadas, no a 1.210 unidades independientes.
Las referencias descriptivas entre cinco semillas no son potencia prospectiva.
El {download}`inventario de curvas <../_static/e3_statistics_remediation_20261121/retrospective/results/learning_curve_inventory.csv>` no divide el objetivo compuesto de entrenamiento CRPS por la MSE de parada: no son la misma pérdida. La razón RMSE de evaluación/parada compara calendarios distintos y una época elegida con parada; no demuestra ausencia de sobreajuste. Sin predicciones guardadas de ajuste/parada ni calibrador V4 no se inventa una brecha completa
de predicción, PIT o cobertura de intervalos. La calibración lineal MZ es otro objeto. Los meses se describen con sus fechas y
conteos efectivamente presentes; no se presume un mes calendario completo.

### Potencia y requisito confirmatorio todavía pendiente

La {download}`auditoría ejecutada de insumos <../_static/e3_statistics_remediation_20261121/guide_power/EXECUTED_POWER_INPUT_AUDIT.json>` cotejó **71 fuentes distintas y los 486 escenarios guardados**, con sus 162 filas de semillas y 324 de calendario. Las tres referencias de varianza V3 disponibles usan **tres pares de semillas**, por debajo de las cinco del piloto de planificación; faltan dieciséis de los diecinueve rivales. Las **76 celdas futuras V3 y 84 V4 siguen sin insumos compatibles**. Las cinco referencias D00 y 25 entradas de covarianza de V4 pertenecen al desarrollo expuesto de una receta abandonada; no se convierten en información de cuatro paneles futuros.

La {download}`matriz de requisitos y brechas del corte d6 <../_static/e3_statistics_remediation_20261121/guide_power/REQUIREMENT_EVIDENCE_GAP_MATRIX.csv>` conserva doce requisitos y separa lo ejecutable de lo prospectivo. No incorpora retroactivamente los resultados nuevos de calibración. El {download}`contrato estructural <../_static/e3_statistics_remediation_20261121/guide_power/POWER_INPUT_CONTRACT.template.json>` dio **dieciocho puertas bloqueadas** y dejó `final_R` y `final_T` nulos, sin imputar ceros o admitir paneles por declaración. El {download}`programa de compatibilidad de insumos <../_static/e3_statistics_remediation_20261121/guide_power/power_inputs_audit.py>` no puede entrenar, simular o acreditar potencia; la referencia normal que incluye varianza de semilla y un suelo temporal LRV/T es un diagnóstico condicional, no la regla completa de tres métodos. Pasaron **21 guardas de software**, separadas de la validación estadística.

Las instrucciones y comandos de esa auditoría están en el {download}`README de guía y potencia <../_static/e3_statistics_remediation_20261121/guide_power/README.md>`.

La potencia mínima de la guía es **0,90 por contraste**, con la conjunta reportada
y su precisión Monte Carlo; no se le atribuye un umbral conjunto universal de
0,90. Margen práctico, alternativa relevante y MDE tienen papeles diferentes.
Aumentar semillas no repara por sí solo falta de información temporal. Las 486
sensibilidades anteriores son antecedentes de planificación, no una certificación
final; tampoco se rellenan las 84 celdas futuras de V4 con referencias de otro
modelo. La receta fue abandonada y no se ejecutan sus 340 ajustes ilustrativos.

La confirmación permanece condicionada a una candidata viable, paneles admitidos,
métodos pertinentes/calibrados, dimensionamiento y protocolo reales anteriores a
sus resultados. Este corte no registra una campaña confirmatoria ni acredita R3.
La asistencia de IA se declara; no sustituye a la persona autora o externa.


(e3-r3-route-statistics-20261121)=
## 8.27 Reparación primaria y examen de una ruta global R3

Este corte del **21 de noviembre de 2026, hora de Bogotá**, conserva las validaciones anteriores y desarrolla una versión nueva después del fallo Holm38 de 14/128 frente al 5 %. No repite entrenamiento, forwards o predicciones de mercado. Los 796 trabajos aditivos, 30 ajustes V3, 20.400 filas retrospectivas y 20 ajustes/1.210 predicciones V4 mantienen sus identidades y decisiones.

### Qué cambia y qué pregunta sigue pendiente

El {download}`análisis de admisibilidad <../_static/e3_r3_route_20261121/methods_admissibility/README.md>` identifica P1 bootstrap del funcional y P2 delta-HAC como dos familias pareadas propuestas. Studentizar P1 revisa uno de esos procedimientos; no crea una tercera familia ni acredita su validez. P2 y el MCS v2 se conservan como controles idénticos. GW con instrumentos no constantes prueba un nulo condicional diferente; Clark–West exige anidamiento real; rangos o cohortes fijas no sustituyen sin justificación la igualdad de media RMSE poblacional.

Las {download}`fuentes primarias y los límites de consulta <../_static/e3_r3_route_20261121/methods_admissibility/PRIMARY_SOURCES.json>` se identifican por apartado; no se declara lectura integral o prueba de cobertura por citar esos trabajos.

Una ruta global concreta usa un MCS que cubra **todos** los óptimos y retenga sólo a la candidata. Bajo el nulo «existe un rival al menos tan bueno», ese singleton omitiría algún óptimo; su falsa decisión queda acotada por el error de cobertura **si esa cobertura es válida para el mismo funcional y muestreo**. Es una deducción condicional propia, no una prueba de cobertura de la extensión actual ni un p por rival. Tampoco prueba δ. La {download}`ruta del tercer procedimiento <../_static/e3_r3_route_20261121/methods_admissibility/THIRD_PROCEDURE_PLAN.json>` requiere justificación de la correspondencia con tres procedimientos de la guía, nulos parciales/empates y control de la regla completa. El nivel prospectivo MCS sigue sin fijar; un cribado a 0,10 no demuestra control a 0,05.

La {download}`derivación estadística <../_static/e3_r3_route_20261121/statistical_revision/STATISTICAL_DERIVATION.json>` conserva ambos ejes de incertidumbre y los límites de la covarianza semilla+HAC. La raíz puede tener sesgo de calendario finito distinto entre modelos con el mismo riesgo poblacional si difiere su dependencia temporal; aumentar semillas no elimina ese término. El diagnóstico gaussiano determinista estima una diferencia aproximada **0,0093251** entre IID y AR(1) φ=0,85, con riesgo1 y T=128; no es una nueva validación aleatoria ni prueba universal de fallo.

P1 nuevo usa la cola absoluta bilateral de raíces centradas divididas por su SE cruzado recalculado. Con B=999, la corrección +1, los empates y el orden951 dan un IC cerrado coherente con rechazo estricto p<0,05, sin cuantiles interpolados. El bloque12 y HAC8 permanecen fijados; no se prueba consistencia general o exactitud finita por llamarlo bootstrap-t. La guarda de degeneración conserva las posiciones de la familia y reporta falta de soporte, sin borrar sorteos o fabricar intervalos de certeza.

### Validación fijada antes de los resultados

El [anclaje público `374fc00f50da4454522c04a7973d0ef590d5e192`](https://github.com/jassan-bit/DeepLearning_202630/commit/374fc00f50da4454522c04a7973d0ef590d5e192) precede a la validación nueva. El {download}`plan <../_static/e3_r3_route_20261121/statistical_revision/VALIDATION_PLAN.json>` fija **siete casos, 1.664 réplicas y 999 remuestreos por réplica** —1.662.336 previstos—, semilla nueva, dos hilos y máximo600s. El registro corresponde a desarrollo estadístico, no a preregistro de mercado.

El {download}`recibo cerrado <../_static/e3_r3_route_20261121/statistical_revision/validation_new/execution_receipt.json>` completó **1.664/1.664 réplicas y 1.662.336 remuestreos** en **479,823 s**, de **07:16:37 a 07:24:37 de Bogotá** —12:16:37–12:24:37 UTC—, con cero errores y las ocho fuentes intactas. Las 23 guardas deterministas anteriores no son réplicas de calibración.

| Caso de validación nuevo | n | FWER del control P1 básico/P2 | FWER del nuevo P1/P2 con Holm38 | Dictamen del cribado nuevo |
|---|---:|---:|---:|---|
| Sólo semillas | 256 | 31/256 | **6/256** | Pasa tolerancia exploratoria |
| AR moderado | 256 | 13/256 | **9/256** | Pasa tolerancia exploratoria |
| AR fuerte | 256 | 7/256 | **12/256** | **Inconcluso** |
| Igualdad parcial: nueve rivales nulos | 256 | 10/256 | **7/256** | Pasa tolerancia exploratoria |
| Escalas de semilla heterogéneas | 256 | 8/256 | **6/256** | Pasa tolerancia exploratoria |
| Dependencia temporal heterogénea entre modelos | 256 | 3/256 | **2/256** | Pasa tolerancia exploratoria |
| Alternativa frente a los diecinueve rivales | 128 | No aplica como FWER nulo | No aplica como FWER nulo | Potencia parcial, no regla completa |

Son eventos de unión sobre nulas verdaderas dentro de la misma familia38, incluidos los nulos parciales; las comparaciones de control usan los mismos sorteos nuevos. El cribado individual prefijado exige límite superior CP95≤0,08 para nominal0,05 y tolerancia0,03. **AR fuerte da 12/256=4,6875 %, IC95[2,44523 %; 8,044903 %]: 0,08044903>0,08. No se redondea para aprobarlo.** El caso sólo semillas mejora respecto del control31/256, pero ese resultado no compensa un caso inconcluso ni demuestra nivel exacto. En AR fuerte el nuevo conteo12/256 es mayor que7/256 del control; no se afirma una mejora uniforme. No se retocan bandas, escenarios, semillas o criterios después.

La {download}`interpretación posterior y recuento <../_static/e3_r3_route_20261121/statistical_revision/RESULTS_INTERPRETATION.json>` mantiene **`new_development_family_inconclusive_not_approved`**. Hay nueve señales descriptivas puntuales de infracobertura —ocho P1 y una P2—; no son una nueva prueba conjunta corregida. Los IC puntuales no se declaran simultáneos y el control de FWER de p no acredita por sí solo cobertura.

El MCS congelado quedó también inconcluso en el nulo AR moderado:26/256 rechazos EPA, IC95[6,74 %;14,53 %], frente al nominal0,10 y techo exploratorio0,13. En estos ensayos ese0,10 es una elección de desarrollo, no un nivel prospectivo ya aprobado. Excluyó algún óptimo verdadero en **30/256** casos de igualdad parcial: cobertura observada de todos los mejores **226/256=88,28 %**, IC95[83,69 %;91,95 %], sin certificar una garantía nominal del90 %. El rechazo EPA inicial256/256 de ese caso es esperable porque contiene alternativas; **no es un error bajo igualdad global verdadera**. Ninguna exclusión se copia como tercer p por rival.

En la alternativa, los dos métodos rechazaron conjuntamente sus p ajustados frente a todos los rivales en **20/128=15,625 %**, IC95[9,81 %;23,09 %]; al añadir dirección e IC por encima de δ quedó el **subconjunto práctico16/128=12,5 %**. El singleton MCS ocurrió en **35/128=27,34 %**. Son endpoints distintos; ninguno incluye tres métodos más toda la adecuación ni dimensiona potencia de mercado. El {download}`resumen íntegro de siete casos <../_static/e3_r3_route_20261121/statistical_revision/validation_new/validation_summary.json>`, la {download}`tabla familiar/MCS <../_static/e3_r3_route_20261121/statistical_revision/exports/family_and_MCS.csv>` y la {download}`potencia parcial separada <../_static/e3_r3_route_20261121/statistical_revision/exports/two_method_subset_power.csv>` conservan todos los resultados.

La puerta confirmatoria permanece cerrada por la {ref}`ruta de diseño <e3-r3-route-design-20261121>`. Las muestras sintéticas no admiten paneles, rehabilitan candidatas abandonadas o fijan R/T de mercado. Las cotas Monte Carlo y las reglas de igualdad, dirección, margen y adecuación conservan su alcance separado. La asistencia de IA está declarada; no sustituye los actos científicos o académicos pendientes.

(e3-input-only-audit-20261121)=
## 8.28 Auditoría del piloto V5 e input-only: señal de desarrollo, no confirmación

Se verificó el {download}`ZIP público original <../_static/e3_input_only_candidate_20261121/inputs/public_results.zip>` contra SHA-256 `beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c`, sus siete hashes internos y las identidades/recibos disponibles. Fuentes inmutables: [código `c4ca86f1`](https://github.com/jassan-bit/DeepLearning_202630/commit/c4ca86f1431afeaa958ff0b544b453ab17a4d7b8) y [resultados `05c1a637`](https://github.com/jassan-bit/DeepLearning_202630/commit/05c1a637f13356dd1df6b8c5aa8f4676dda61c12), `runs/e3_seoi_v5/run-01`. El {download}`script de auditoría sin dependencias de modelos <../_static/e3_input_only_candidate_20261121/analysis/audit_saved_results.py>` recalcula RMSE y MAE desde 1.760 predicciones: 1.650 neurales y 110 de controles analíticos. No hubo entrenamientos, forwards, predicciones nuevas ni remuestreos en esta tarea.

| Receta o control | RMSE medio individual | MAE medio |
|---|---:|---:|
| `input_only_mse` | 0,0020515608242855487 | 0,0015484962302801841 |
| `global_mse` | 0,0021097806965662083 | 0,0016536738452131044 |
| `floor_plus_mean_excess` | 0,0021142309406569867 | 0,0018134828077995932 |
| `global_crps` | 0,002136141247569356 | 0,0015644943808417393 |
| V5 completo `seoi_mse` | 0,0021609604200311396 | 0,001702061588126237 |
| `additive_mse` del piloto | 0,0021713942553732656 | 0,0018378139638598589 |
| `output_only_mse` | 0,002242368698424727 | 0,0017718685032641983 |
| `floor_only` | 0,002790340619362471 | 0,0018317657439326028 |

Se conserva la media de RMSE por semilla; no se sustituye por RMSE combinado ni por un ensamble. Input-only mejora **2,759522 %** frente a global y gana **4/5** pares de semillas; frente a V5 completo gana los cinco. V5 completo tiene **5,332506 % más RMSE** que el mejor control y conserva **`ABANDON_CURRENT_SEOI_RECIPE`**. Su regla fijada para `seoi_mse` no autoriza promover automáticamente la ablación.

La {download}`auditoría íntegra <../_static/e3_input_only_candidate_20261121/analysis/EMPIRICAL_AUDIT.md>` incluye {download}`métricas por semilla <../_static/e3_input_only_candidate_20261121/analysis/metrics_recomputed.csv>`, {download}`las 56 comparaciones dirigidas <../_static/e3_input_only_candidate_20261121/analysis/all_pairwise_summary.csv>` y {download}`las diferencias emparejadas <../_static/e3_input_only_candidate_20261121/analysis/all_pairwise_seed_differences.csv>`. La SD de RMSE entre semillas es **0,0000724269** para input-only frente a **0,00000995977** para global, unas **7,27 veces** mayor. Input-only ocupa el **último rango de selección** (RMSE medio 0,00430490508, 54 fechas julio–agosto) y el primero de assessment (55 fechas septiembre–octubre). El cambio de dificultad entre periodos impide interpretar la reducción de error como estabilidad demostrada.

La sensibilidad descriptiva frente a global conserva ganancias **4,45694 % / 1,15452 %** en primeras 27/últimas 28 fechas; en cuartiles del objetivo realizado obtiene **−0,591 % / 1,275 % / 6,896 % / 3,210 %**. Son particiones retrospectivas pequeñas, no regímenes causales ni réplicas independientes. Las mismas 55 fechas y etiquetas se comparten entre semillas: **275 filas semilla-fecha no son 275 observaciones independientes**. La selección posterior a resultados y la exposición previa persisten; no se añaden p-valores, intervalos o potencia confirmatorios.

Los recibos documentan 30 ajustes, **965 épocas y 16.405 pasos**, cero fallos/interrupciones registrados, **76,02349 s activos y 10,00463 s de IO de checkpoint**; estos tiempos no incluyen todo el coste de VM, instalación o respaldo. La recuperación declara sesión anterior cerrada y cierre final `assessment_finished`. Se verificó consistencia del inventario, pero el ZIP no contiene pesos o arrays originales para verificar materialmente sus bytes ni distribuciones atómicas para acreditar calibración. El {download}`resumen verificable <../_static/e3_input_only_candidate_20261121/analysis/AUDIT_SUMMARY.json>` separa esas limitaciones de las comprobaciones ejecutadas.

El {download}`protocolo confirmatorio DRAFT <../_static/e3_input_only_candidate_20261121/confirmation/CONFIRMATION_PROTOCOL_DRAFT.json>` es una propuesta nueva con **20 rivales** y estimando condicionado al calendario. La ruta de tres procedimientos sobre semillas requiere justificar supuestos y correspondencia con F2; **no hay tres métodos aprobados ni una puerta F2 satisfecha**. Siguen pendientes cuatro admisiones, reserva verificable, margen práctico, covarianza de toda la familia, calibración de métodos, potencia ≥0,90 por contraste y conjunta reportada, R/T/B y presupuesto finales. Ni esta auditoría ni el notebook de preparación ejecutan o programan una confirmación. Los 796 trabajos aditivos, el 2,7043 % adverso de V3 y el abandono V4 permanecen sin cambios; no son resultados de input-only.

(e3-temporal-engine-statistics-20261121)=
## 8.29 Motor temporal: estimando, calibración y evaluación separada

La guía original y su Tabla 4/F2 motivan tres procedimientos distintos: **DM-HLN/HAC**, **bootstrap estacionario studentizado** y **Giacomini–White condicional**. No se cuenta Clark–West sin anidamiento ni se transforma un MCS global en p direccionales por rival. La ruta anterior de t/bootstrap/Wilcoxon sobre semillas queda conservada como propuesta anterior, sin renombrarla como batería temporal admitida.

(e3-data-selection-review-20261121)=
### Bitcoin principal y revisión de los conjuntos adicionales

**El problema principal sigue siendo el pronóstico de volatilidad de Bitcoin.** La guía, **p. 23, §8.3**, exige tres conjuntos adicionales de la misma tarea y naturaleza distinta. Cada uno debe diferir del principal en al menos dos ejes —tamaño, dominio, ruido, desbalance o dimensionalidad— y seleccionarse antes de conocer sus resultados. La guía permite justificar esos ejes sin obligar a usar acciones o Nasdaq.

El código entregado propone BTCUSDT para F00 y **IBM/MSFT/XOM de WIKI/PRICES** para F01–F03. Su finalidad propuesta era contrastar la misma tarea en otro dominio y con mayor tamaño; el umbral de **1,5×** es una elección del protocolo propuesto, no una exigencia del PDF. Esas propuestas **no son paneles admitidos ni evidencia confirmatoria**. La última salida aportada del Colab informa que faltan las cuatro admisiones.

**La adquisición asistida de esta propuesta queda en pausa mientras se revisa su pertinencia con el autor.** Una cuenta y clave de Nasdaq son dependencias del adaptador elegido, no requisitos académicos. WIKI terminó en 2018 y no aporta una reserva futura. Los adicionales pueden revisarse manteniendo el enfoque cripto si se justifican dos diferencias pertinentes por conjunto; cambiar ticker o trocear el histórico no basta. Cualquier sustitución necesita una versión documentada de fuentes, derechos, tarea y reglas antes de entrenar. Esta aclaración no modifica el código, las identidades congeladas ni los resultados anteriores.

### Estimando y supuestos

La enmienda prospectiva usa la media de errores cuadrados sobre fechas y el conjunto fijo de modelos entrenados. Reporta su raíz como RMSE agregado, conserva además la media de RMSE individuales y exige que ésta sea favorable. Ninguna equivale al RMSE de promediar predicciones como ensamble. Las semillas no multiplican artificialmente las fechas independientes; la inferencia temporal condiciona los modelos y semillas fijados.

DM y bootstrap estudian la media del diferencial de pérdida. GW contrasta momentos condicionales con instrumentos disponibles en el origen, por lo que su rechazo es no direccional y no prueba por sí solo inferioridad de la media. Se informan estadísticos, p crudos/ajustados, direcciones, contadores Monte Carlo y supuestos. La familia principal contiene 120 casillas: 20 rivales × tres procedimientos × igualdad/margen. La regla operacional se fija en el protocolo vigente; no cambia después de resultados de mercado.

Las referencias fixed-b de DM y GW retienen la dependencia entre covarianza HAC y estadístico en el tamaño finito. El bootstrap-t recalcula HAC por réplica. Se conservan intervalos del 95 % y guardas conservadoras del 99 %, denominadas con su nivel real. Su cobertura no se describe como 95 % por conveniencia. Las pruebas incluyen equivalencia de la suma móvil HAC/Bartlett, escalas extremas, singularidades y equivalencia del núcleo nativo con NumPy.

### Validación sintética y alcance

**Dictamen vigente: `PASSED_PREDECLARED_VALIDATION`.** El {download}`resumen de entrega <../_static/e3_confirmatory_engine_20261121/DELIVERY_STATUS.json>`, la {download}`batería y sus hipótesis <../_static/e3_confirmatory_engine_20261121/docs/TEMPORAL_BATTERY.md>` y el {download}`informe vigente <../_static/e3_confirmatory_engine_20261121/qa/statistical_validation/v4_precision/report.json>` permiten comprobar plan, código, semillas, presupuesto, réplicas y cotas exactas. El estudio no se presenta como validación universal ni acredita potencia de una campaña de mercado. Las versiones fallidas y las inconclusas permanecen archivadas; no se ensancharon tolerancias para aprobarlas.

La **batería temporal, versión estadística 4**, usa α operacional 0,025, R=10, T=768, B=8191, HAC48 y bloques32. El expediente compuesto conserva cuatro escenarios de 512 réplicas y añade **2048 réplicas independientes** del nulo AR(0,5). El criterio de aceptación conservó FWER CP95 superior≤0,05 y cobertura de los intervalos **99 %** con CP95 inferior≥0,95.

| Escenario | Réplicas | FWER / CP95 superior | Cobertura CI99 MSE / CP95 inferior | Cobertura CI99 ganancia / CP95 inferior |
|---|---:|---:|---:|---:|
| complete_null_ar05 | 2048 | 50/2048 / 0.032061 | 2019/2048 / 0.979727 | 2019/2048 / 0.979727 |
| partial_null_ar05 | 512 | 7/512 / 0.027965 | 507/512 / 0.977358 | 507/512 / 0.977358 |
| complete_null_overlap5 | 512 | 6/512 / 0.025331 | 503/512 / 0.966894 | 503/512 / 0.966894 |
| practical_boundary_ar05 | 512 | 15/512 / 0.047861 | 507/512 / 0.977358 | 507/512 / 0.977358 |
| strong_alternative_ar05 | 512 | — | 503/512 / 0.966894 | 503/512 / 0.966894 |

La auditoría inicial de esta versión conservó **16/512 rechazos** en ese nulo, con CP95 superior **0,05025287955875999**: su dictamen original sigue **inconcluso**. La elección del escenario a replicar se hizo **después** de conocerlo. El mapa del expediente se publicó antes de las réplicas nuevas en `ace1095452ca6e35af083f92822dad374049b5f6`: fija los cuatro componentes previos y el único componente nuevo, sin agrupar los 512 antiguos con los 2048 nuevos ni permitir elegir otro mapa tras ver el resultado. No es un estudio nuevo de cinco escenarios independientes, un preregistro de los resultados anteriores o una garantía de confianza simultánea ni bajo revisiones opcionales.

El código y el plan iniciales se publicaron antes de simular en `70f0838cdf8809a469ea1ca837e5011faff551e2`. Las versiones **estadísticas** 1/2 fallaron; la 3 quedó inconclusa y sus 960 réplicas completas se recuperaron sin repetirlas. Estas versiones del validador son distintas de las arquitecturas y pilotos V3/V4/V5. Se conservan las coberturas diagnósticas del 95 % y sus fallos, sin presentarlas como calibradas. La admisión obtenida es condicional a los DGP y tamaños ensayados por panel; no es una simulación conjunta de cuatro mercados. Las cotas CP son puntuales para los diagnósticos declarados. La multiplicidad adicional se justifica por Holm y sus supuestos; la potencia conjunta requiere el estudio separado con planificación real.

El planificador de potencia ejecuta la regla efectiva con covarianzas de desarrollo guardadas; no sustituye esos insumos por el piloto de 55 fechas. La alternativa de planificación del 3 % excede el margen práctico del 2 %, con sensibilidad del 5 %; no promete potencia ante mejoras arbitrariamente cercanas al margen. Los límites de Monte Carlo, tiempo, memoria y presupuesto forman parte de la decisión, no se ocultan como resultados suficientes.

### Test único y comprobación de las seis condiciones

Antes de test se verifican todos los checkpoints seleccionados y se congelan calibradores con predicciones de calibración. Los intervalos absolutos al 90 % usan el estadístico de orden finito y el soporte √A; no se atribuye garantía intercambiable a una serie dependiente ni calibración distributiva a átomos aprendidos con MSE. El protocolo vincula los umbrales prospectivos de adecuación con su justificación y hash.

La evaluación registra una consulta por identidad y escenario; publica el marcador durable antes del acceso. Una cache completa se reutiliza. Un marcador pendiente sin resultado detiene la recuperación automática para no repetir una consulta de estado incierto. Los escenarios prefijados de ruido de medición y faltantes preservan los cierres/etiquetas y se guardan dentro del mismo sello, sin retocar la candidata.

El análisis consume exclusivamente CSV guardados y conserva resultados adversos. Comprueba superioridad frente a los veinte rivales, margen/efectos/intervalos, consistencia en al menos dos de tres adicionales y ausencia de inferioridad significativa, adecuación y robustez. Conserva GW en la multiplicidad, pero el veto de inferioridad media usa DM/bootstrap direccionales. La potencia conjunta se reporta; no se le inventa un mínimo universal del 90 %. No se convierten pruebas de software, simulaciones o un JSON favorable en aceptación académica de R3.
