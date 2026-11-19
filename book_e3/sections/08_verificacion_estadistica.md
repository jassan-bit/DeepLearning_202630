# 8 Bloque D Verificación estadística y evaluación ampliada

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
| Tres conjuntos adicionales | Estado actual: **1 de 3 archivos admitidos, 0 evaluados por modelos y 2 pendientes**. D01 se admite para σ₇ RAW entre sesiones según 8.13; el inventario inicial de 8.12 y el reporte original conservan su cero histórico. Completar los otros dos archivos, registrar diferencias entre dominios y fijar el protocolo común antes de evaluar desempeño. |
| Particiones, ejecuciones, semillas y determinismo | Calendario y diez inicializaciones registrados para la comparación histórica; cerrar separación confirmatoria, número final dimensionado y políticas/tolerancias por backend para la campaña futura. |
| Familia, pruebas, ajustes y nivel | Los 16 rivales permanecen completos. El análisis posterior tiene Holm global48, α=0,05 y MCS auxiliar al 90 %; justificar las pruebas y familias definitivas antes de confirmar. |
| Optimización, cómputo y potencia | Registrar número de configuraciones, épocas/parada y tiempo real por modelo, con capacidad y hardware. La política comparable y la potencia por contraste/conjunta siguen abiertas. |
| Nd y desarrollo frente a confirmación | [Bitácora consolidada parcial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md); ocho recetas de desarrollo no son el Nd global, y diez semillas no son diez diseños. |
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

**Pendiente.** Este análisis sólo usa una cronología BTC conocida. No se han evaluado tres conjuntos adicionales ni se ha demostrado consistencia entre dominios. Las semillas no son conjuntos de datos independientes; por eso no se ejecuta Friedman, Iman–Davenport o un diagrama de diferencia crítica usando las diez inicializaciones como si fueran diez mercados.

La elección y evaluación de los conjuntos adicionales, sus márgenes por escala, licencias, auditorías y comparación completa siguen requiriendo un diseño previo. No se promedian RMSE de escalas distintas ni se denomina confirmación independiente a otro tramo ya expuesto.

El inventario preparatorio inicial del 18 de noviembre, resumido en **8.12**, identificó tres candidatos con **cero conjuntos admitidos o evaluados en ese corte**. La revisión posterior de calendario y política de **8.13** admite D01 WIKI/AAPL para una tarea RAW entre sesiones: el estado actual es **1 de 3 archivos admitidos, 0 evaluados por modelos y 2 pendientes**. La admisión de un archivo no completa la evaluación entre dominios ni el protocolo común de modelos.

Cada conjunto debe mantener la **misma tarea y métrica primaria** y diferir del principal en al menos dos ejes pertinentes de la guía: tamaño, dominio, ruido, dimensionalidad o desbalance cuando sea aplicable. En esta regresión no se inventa un desbalance de clases; la distribución del objetivo puede describirse como contexto sin convertirla automáticamente en otro eje acreditado. Su ficha debe identificar fuente pública, licencia/condiciones, versión y hash, periodo/unidad independiente, ejes de diferencia, particiones, auditoría de fuga, presupuesto y margen con unidades. Los 17 sistemas se evalúan en cada conjunto; una limitación de memoria del dashboard no autoriza excluir un rival científico.

La consolidación podrá usar Friedman con corrección de Iman–Davenport y comparaciones por pares con Holm, o una alternativa bayesiana de rangos con signo y ROPE previamente justificada. La elección se fija antes de ver los resultados, con rangos y márgenes pertinentes; con sólo tres conjuntos la potencia global es limitada. Se informarán también diferencias, IC y efectos por conjunto y la consistencia del resultado, sin mezclar RMSE de escalas distintas en un promedio bruto. Un conjunto usado para rediseñar el modelo pasa a desarrollo y deja de aportar confirmación independiente.

## 8.4 Definición operativa de superioridad R3

La guía exige **seis condiciones simultáneas**; ninguna se sustituye por otra:

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
:file: ../_static/e3_revision_bloque_d_20261117/metrics_extended_table.csv
:header-rows: 1
```

La tabla incorpora la CNN-LSTM y las diez semillas de cada red, sin elegir una réplica por rendimiento. MAE, sMAPE y R² se calculan por predictor y después se promedian. sMAPE se expresa en porcentaje; RMSE, MAE y anchura usan unidades originales de σ7, sin anualización. Para una trayectoria,

$$
\operatorname{sMAPE}=\frac{200}{n}\sum_t\frac{|y_t-\widehat y_t|}{|y_t|+|\widehat y_t|},\qquad
R^2=1-\frac{\sum_t(y_t-\widehat y_t)^2}{\sum_t(y_t-\overline y)^2},
$$ (eq-e3-smape-rsquared)

con contribución cero cuando ambos valores de un término sMAPE son cero. La media de R² no es el R² de un ensamble. E3 obtiene sMAPE **11,0977 %** y R² medio **0,911779** en este calendario. La [tabla abreviada del corte estadístico](../_static/e3_estadistica_posterior_20261117/ranking_table.csv) conserva sus bytes; el suplemento amplía métricas descriptivas y no cambia sus p-valores.

La cobertura y anchura de E2 se reconstruyen exclusivamente con los cuantiles residuales **ya congelados** de cada predictor, mediante $[\max(0,\widehat y-q),\widehat y+q]$; no se ajustó un nuevo calibrador con prueba. Son medias de las coberturas y anchuras individuales, no de un intervalo de ensamble. **La campaña original E3 no produjo un cuantil ni intervalo calibrado**, por lo que sus dos campos permanecen ausentes en esta tabla histórica y en el servicio. El suplemento separado **8.11** sí calcula diez cuantiles posteriores; no reescribe estos campos o el protocolo anterior. La DE entre semillas no rellena esas casillas y el nominal 90 % no constituye garantía individual bajo dependencia o cambio temporal.

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

La guía exige al menos tres pruebas distintas y apropiadas y al menos un procedimiento global o con control de comparaciones múltiples; no prescribe literalmente idéntico ámbito de incertidumbre. El análisis posterior aplica Holm global48, pero los procedimientos y sus alcances deben justificarse para la hipótesis F2a: sus rechazos no acreditan automáticamente toda la Tabla 4 ni R3. No se aplican correcciones Nadeau–Bengio a días o semillas de esta única cronología, ni Clark–West sin justificar anidamiento.

### Relación con las pruebas admisibles de la Tabla 4

La ruta es **temporal, F2a**. La comparación actual no se renombra para hacer coincidir métodos diferentes con la tabla de la guía:

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
| Dependencia de residuos y de pérdidas | Ljung–Box, ARCH-LM y BDS cuando sus supuestos sean pertinentes; examinar autocorrelación, heterocedasticidad y dependencia no lineal. No fueron ejecutados para cerrar adecuación de E3 en esta etapa. |
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

**Encabezado y expediente del corte preparatorio inicial.** Se conserva su ancla y su estado histórico; la admisión derivada posterior de D01 se documenta en 8.13. El estado actual es un archivo de tres admitido, cero evaluados por modelos y dos pendientes.

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

El {download}`alcance de la revisión <../_static/e3_gpu_colab_20261118/README.md>` distingue salida compartida de archivo auditado. **El ZIP del antecedente L4 no se recuperó**: para esa sesión no hay SHA-256 del paquete ni validación de casos individuales. Esa limitación histórica no describe el nuevo paquete A100, que sí se guardó y auditó por separado. Ese corte conservaba cero archivos adicionales admitidos; la decisión posterior de 8.13 lleva el estado actual a uno de tres, cero evaluados por modelos y dos pendientes. La campaña real sigue bloqueada por las admisiones restantes y por protocolo/tag, tratamiento común, presupuesto y potencia finales. Pasar los contratos sintéticos no completa R1/R3 ni la reproducción humana independiente.


### Guardado automático de Colab en GitHub

Tras la pérdida de los archivos temporales de la sesión L4, el usuario solicitó guardar los resultados en GitHub. El cuaderno actual incorpora ese destino: recibos JSON, logs, ZIP y manifiesto de hashes quedan en `runs/colab_preflight/<RUN_ID>/`, dentro de una rama nueva `e3-colab-receipts-<RUN_ID>` del repositorio. El guardado usa un commit atómico, comprueba los bytes remotos y reconoce un reintento idéntico. El código y las evidencias anteriores conservan sus versiones.

Para escribir desde Colab se necesita **GITHUB_TOKEN**, con **Contents: Read and write** limitado a `jassan-bit/DeepLearning_202630`, guardado en Secretos o introducido en un campo oculto. La consulta previa comprueba acceso al repositorio; no certifica por sí sola el alcance de escritura de un token fino. El token se mantiene en memoria y se excluye del entorno científico y los archivos publicados.

El {download}`cuaderno con guardado en GitHub <../_static/e3_github_persistence_20261118/e3_remediacion_preflight_colab.ipynb>`, el {download}`módulo de persistencia <../_static/e3_github_persistence_20261118/colab_github.py>`, sus {download}`pruebas de contrato <../_static/e3_github_persistence_20261118/test_colab_github.py>` y las {download}`instrucciones <../_static/e3_github_persistence_20261118/README.md>` describen el flujo. Al terminar se imprime una URL GitHub para revisar la ejecución. Si falla la escritura, se corrige el secreto y se repite sólo el guardado, con los mismos resultados locales.

**La primera ejecución con este flujo ya está guardada y auditada: A100, run `20261118T172624131741Z`.** El ZIP de la sesión L4 anterior no se recuperó; su consola se mantiene como un antecedente distinto, sin reetiquetarla como la ejecución nueva. Guardar recibos no habilita la campaña real: siguen pendientes D02/D03 y el protocolo/tag común previo, el tratamiento, el presupuesto y la potencia finales; D01 sólo tiene cerrada la admisión de archivo para su contrato RAW entre sesiones.


### Primera ejecución A100 guardada y auditada

La ejecución `20261118T172624131741Z` completó **17/17 perfiles sintéticos** y guardó sus [resultados en un commit inmutable](https://github.com/jassan-bit/DeepLearning_202630/tree/47d2f316a50d9268cb6484a547e8a2f0cc0413eb/runs/colab_preflight/20261118T172624131741Z), `47d2f316a50d9268cb6484a547e8a2f0cc0413eb`, cuyo padre es el código ejecutado `1ff136f71c0ffcd1dbd918efbc078a8d9653ed21`. Los 14 recibos neuronales registran CUDA, backward y recarga idéntica; declaran 456 tensores de gradiente finitos. Los tres comparadores clásicos se ejecutaron en CPU, incluido un único ajuste ARIMA artificial. El resultado global es `passed_synthetic_only`; ajustes, predicciones y contrastes nuevos de mercado permanecen en cero.

La auditoría verificó el ZIP de **43.860 bytes**, SHA-256 `44b7fb93a3f9f667bbe3cd7b6a8d4b1a190c0742ac62e78c0dbd127dfa62fd31`: sus **40 miembros** coinciden exactamente con los archivos sueltos. El manifiesto enumera **41 artefactos**, incluido el ZIP; con el propio `storage_manifest.json` son **42 archivos persistidos**, sin discrepancias de bytes. También se cotejaron las siete fuentes del contrato, el preparador de entorno, el lock de 43 paquetes y el ejecutor. El {download}`informe de auditoría <../_static/e3_a100_receipts_20261118/audit.json>`, la {download}`tabla por perfil <../_static/e3_a100_receipts_20261118/per_model.csv>` y la {download}`auditoría de fuentes <../_static/e3_a100_receipts_20261118/source_audit.json>` delimitan esas comprobaciones.

El {download}`ZIP preservado <../_static/e3_a100_receipts_20261118/receipts.zip>`, el {download}`manifiesto de guardado <../_static/e3_a100_receipts_20261118/storage_manifest.json>`, el {download}`verificador <../_static/e3_a100_receipts_20261118/verify.py>` y la {download}`documentación del alcance <../_static/e3_a100_receipts_20261118/README.md>` permiten repetir la auditoría de archivos. **El ZIP no contiene los pesos ni los tensores originales**: no se recalcularon gradientes o salidas GPU ni se reprodujo numéricamente la recarga; se verificaron los recibos y su coherencia con la fuente hasheada. Esta revisión no es reproducción humana independiente ni acredita R1/R3. En el corte A100 había cero archivos adicionales admitidos. Tras la revisión posterior de D01, el estado actual es uno de tres admitido, cero evaluados por modelos y dos pendientes; el bloqueo de la campaña real por protocolo/tag común, tratamiento, presupuesto y potencia finales permanece.


(d01-admission-session-raw)=
## 8.13 D01 admitido para σ₇ RAW entre sesiones

**Estado derivado actual: 1 de 3 archivos admitidos, 0 evaluados por modelos y 2 pendientes.** D01 es el extracto WIKI/AAPL de Bokeh, 3270 sesiones del 1 de marzo de 2000 al 1 de marzo de 2013, admitido para una tarea definida de precio cotizado bruto: desviación poblacional —divisor siete— de siete retornos logarítmicos entre sesiones, con horizonte próxima sesión. La [ejecución CPU del usuario](https://github.com/jassan-bit/DeepLearning_202630/commit/648f2b91b477b5f4226cad3af5928088accd74ed), run `20261118T181551067867Z`, conserva como código ejecutado [`568aee0445a6eaba0f13c11deb2d6d64aa4f3fbd`](https://github.com/jassan-bit/DeepLearning_202630/commit/568aee0445a6eaba0f13c11deb2d6d64aa4f3fbd). Su JSON original registró cero admitidos; **no se modifica ni se presenta como si el usuario hubiera reportado uno**. El cambio a uno es una decisión derivada posterior, después de verificar calendario y fijar la política de datos; tampoco altera los cortes inicial, CPU, L4 o A100.

La fuente queda fijada al commit Bokeh [`3e831d2779ca74c087c7eee05d538015c97b1e52`](https://github.com/bokeh/bokeh_sampledata/tree/3e831d2779ca74c087c7eee05d538015c97b1e52), archivo `stocks.csv.gz`, SHA-256 `5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd`. Su inventario declara dominio público para los datos WIKI y documenta la procedencia; la licencia BSD del software no se usa para relicenciar los datos. El calendario se contrastó con `exchange-calendars==4.11.3`, fuente [`79a4c2cbeed5a16adee5d013edd76f5c8c82d656`](https://github.com/gerrymanoim/exchange_calendars/tree/79a4c2cbeed5a16adee5d013edd76f5c8c82d656): `NASDAQ`/`XNAS` son alias publicados de `XNYS`, no comprobaciones independientes. Coinciden **3270 de 3270 fechas en orden, sin sesiones faltantes ni adicionales**, incluidos los cierres de 9/11, Reagan, Ford y Sandy. Son etiquetas de sesión, no marcas horarias UTC de ejecución; no se rellena ni inventa una vela para días sin negociación.

La política **`RAW_KEEP_ALL`** conserva todas las filas y los OHLCV sin ajustar y excluye `Adj Close` tanto de entradas como de etiquetas. No se corrigen precios por splits, no se eliminan saltos ni se seleccionan etiquetas después de observar resultados de modelos. Los efectos mecánicos de acciones corporativas y los movimientos de mercado permanecen en la serie: **σ₇ describe volatilidad del precio cotizado bruto, no volatilidad económica ajustada por splits/dividendos**. El contrato de sesiones es una adaptación explícita y no se presenta como el reloj diario UTC continuo de BTCUSDT. La {download}`decisión de admisión <../_static/e3_d01_admission_20261118/admission_decision.json>`, la {download}`política RAW <../_static/e3_d01_admission_20261118/policy.md>` y la {download}`documentación del alcance <../_static/e3_d01_admission_20261118/README.md>` fijan esa definición.

La {download}`comprobación de calendario <../_static/e3_d01_admission_20261118/calendar_check.json>`, las {download}`3270 sesiones de referencia <../_static/e3_d01_admission_20261118/NASDAQ_sessions_20000301_20130301.csv>`, la {download}`auditoría de bytes del registro <../_static/e3_d01_admission_20261118/byte_audit.json>` y el {download}`verificador <../_static/e3_d01_admission_20261118/verify.py>` permiten revisar el cierre del archivo. **Admitir D01 no requiere declarar ya cerrado el protocolo MODEL ni habilita entrenamiento.** Los entrenamientos históricos existen y se conservan; esta etapa no añade ajustes, predicciones de mercado ni contrastes de modelos. La campaña futura continúa bloqueada hasta resolver D02/D03 y fijar el protocolo común, las particiones, el tratamiento, el presupuesto y la potencia. Ese protocolo podrá justificar ajustes futuros; no se reentrena por rutina. La admisión de un archivo no acredita evaluación ampliada, reproducción humana independiente, R1/R3, DOI o aceptación pública de Render.

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
