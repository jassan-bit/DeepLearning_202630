# 8 Bloque D Verificación estadística y evaluación ampliada

**Corte del 17 de noviembre de 2026: análisis retrospectivo ejecutado sobre predicciones congeladas.** La CNN-LSTM con atención de contenido y salida aditiva positiva tiene diez semillas; cada una de las trece redes rivales también tiene diez. Persistence, SeasonalNaive y ARIMA conservan una trayectoria determinista cada uno. La comparación incluye los diecisiete sistemas sobre las mismas 608 fechas, sin eliminar rivales, semillas ni observaciones.

Los **48 contrastes principales** —dieciséis rivales por tres procedimientos— rechazaron sus nulas frente al margen heredado, con **Holm sobre los 48 p-valores** y máximo p ajustado **0,0019**. Las sensibilidades temporales conservaron esa decisión. Este resultado está condicionado al calendario y a los supuestos de cada método: **no acredita por sí solo R3**, un diseño confirmatorio previo o la originalidad del mecanismo. El MCS auxiliar conserva diez sistemas, incluido E3; no identifica un ganador único.

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
| Tres conjuntos adicionales | Aún sin elegir: fijar tarea, fuentes, licencia, versión/hash y diferencias en al menos dos ejes antes de conocer su desempeño. |
| Particiones, ejecuciones, semillas y determinismo | Calendario y diez inicializaciones registrados para la comparación histórica; cerrar separación confirmatoria, número final dimensionado y políticas/tolerancias por backend para la campaña futura. |
| Familia, pruebas, ajustes y nivel | Los 16 rivales permanecen completos. El análisis posterior tiene Holm global48, α=0,05 y MCS auxiliar al 90 %; justificar las pruebas y familias definitivas antes de confirmar. |
| Optimización, cómputo y potencia | Registrar número de configuraciones, épocas/parada y tiempo real por modelo, con capacidad y hardware. La política comparable y la potencia por contraste/conjunta siguen abiertas. |
| Nd y desarrollo frente a confirmación | [Bitácora consolidada parcial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md); ocho recetas de desarrollo no son el Nd global, y diez semillas no son diez diseños. |
| Fecha/tag y desviaciones | Registrar la versión final antes de sus experimentos y mantener una bitácora fechada de desviaciones, con motivo e impacto. Un análisis no previsto conserva alcance exploratorio y no sustenta por sí solo R3. |

La métrica primaria es la **media de las RMSE individuales**, en unidades originales de sigma7 poblacional, sin anualización:

$$
M_j=\frac1{R_j}\sum_{r=1}^{R_j}\sqrt{\frac1{608}\sum_{t=1}^{608}(y_t-\widehat y_{j,r,t})^2},\qquad
\widehat\Delta_j=M_j-M_{\mathrm{E3}}.
$$

Positivo favorece a E3. No se sustituye esta media por la RMSE de un ensemble o por la raíz del MSE agregado entre semillas. Se mantiene el margen del borrador:

$$
H_{0,j}:\Delta_j\le\delta,\qquad H_{1,j}:\Delta_j>\delta,
\qquad\delta=0{,}00025772842664981106.
$$

El parámetro y la incertidumbre representada dependen de cada procedimiento. Las semillas son **271801–271810**. La semilla histórica 20261116 queda fuera de la comparación. Las fechas comunes van del **1 de enero de 2025 al 31 de agosto de 2026**; este calendario ya era conocido.

## 8.2 Equidad de la comparación

La auditoría verificó los once archivos fijados, las fechas y objetivos exactos, las 608 máscaras válidas, las coordenadas de semilla y la conservación de los 81 predictores históricos de E2 dentro de su ampliación a **133 predictores**. Las métricas guardadas coinciden con las recalculadas. Se conserva el {download}`informe de admisión <../_static/e3_estadistica_posterior_20261117/input_validation.json>`.

| Tratamiento | E2: trece redes | CNN-LSTM E3 |
|---|---|---|
| Entrenamiento | 655 ejemplos; hasta 2024-10-31 | 532 ejemplos; hasta 2024-06-30 |
| Purga entre ajuste y validación | Receta final de E2 conservada | 7 ejemplos |
| Validación para selección/parada | Épocas finales fijadas por la receta previa | 116 ejemplos; 2024-07-08 a 2024-10-31 |
| Calibración posterior | 53 ejemplos; 2024-11-08 a 2024-12-30 | No utilizada en esta receta; ese bloque no se abrió |
| Prueba común | 608 fechas | Las mismas 608 fechas y objetivos |
| Repeticiones estocásticas | 10 por red | 10 |

La ampliación de E2 añadió cuatro semillas por red, **52 ajustes**, sin retunear las recetas. Los tres clásicos siguen almacenados una sola vez. Los ajustes y calibradores nuevos quedaron confirmados antes de producir sus nuevas predicciones de prueba. La CNN-LSTM se reutilizó desde su réplica de diez semillas.

**Tener diez semillas por red no iguala las particiones, la selección ni el presupuesto de entrenamiento.** Estos contrastes comparan las recetas ejecutadas; no aíslan un efecto causal de arquitectura. Se conservan separados los 5130,970719594005 segundos registrados de la ampliación E2 y los 948,439625042 segundos cerrados de E3. Esta etapa estadística ejecutó cero ajustes y cero inferencias nuevas, en CPU.

La equidad futura requiere fijar la misma herramienta y número de intentos de HPO, máximo de épocas, criterio de parada y espacio relativo de búsqueda; el mismo número de configuraciones no iguala tiempo cuando su coste difiere. Debe declararse si el presupuesto se mide en configuraciones, cómputo o ambos, e igualar el tiempo total o reportar ambos análisis cuando corresponda. El esfuerzo de diseño se registra aparte del HPO. No se cambia la receta congelada de un rival para favorecer retrospectivamente la comparación.

El preprocesamiento, las variables disponibles, aumentos aplicables, particiones, fuente de pesos preentrenados y métrica deben ser comunes en el protocolo pertinente. Se reportarán parámetros y FLOPs de todos los modelos; si el original resulta sustancialmente mayor que el mejor benchmark, se incluirá una comparación de capacidad comparable o una discusión explícita del efecto del tamaño. Las medidas existentes de GPU, eager CPU y TorchScript tienen condiciones distintas: faltan una contabilidad completa por modelo de entrenamiento/inferencia, hardware, memoria pico y energía estimada con método. Una trayectoria determinista se conserva como tal: duplicarla diez veces no produce incertidumbre de inicialización.

## 8.3 Evaluación en conjuntos de datos adicionales

**Pendiente.** Este análisis sólo usa una cronología BTC conocida. No se han evaluado tres conjuntos adicionales ni se ha demostrado consistencia entre dominios. Las semillas no son conjuntos de datos independientes; por eso no se ejecuta Friedman, Iman–Davenport o un diagrama de diferencia crítica usando las diez inicializaciones como si fueran diez mercados.

La elección y evaluación de los conjuntos adicionales, sus márgenes por escala, licencias, auditorías y comparación completa siguen requiriendo un diseño previo. No se promedian RMSE de escalas distintas ni se denomina confirmación independiente a otro tramo ya expuesto.

Cada conjunto debe mantener la **misma tarea y métrica primaria** y diferir del principal en al menos dos ejes pertinentes de la guía: tamaño, dominio, ruido, dimensionalidad o desbalance cuando sea aplicable. En esta regresión no se inventa un desbalance de clases; la distribución del objetivo puede describirse como contexto sin convertirla automáticamente en otro eje acreditado. Su ficha debe identificar fuente pública, licencia/condiciones, versión y hash, periodo/unidad independiente, ejes de diferencia, particiones, auditoría de fuga, presupuesto y margen con unidades. Los 17 sistemas se evalúan en cada conjunto; una limitación de memoria del dashboard no autoriza excluir un rival científico.

La consolidación podrá usar Friedman con corrección de Iman–Davenport y comparaciones por pares con Holm, o una alternativa bayesiana de rangos con signo y ROPE previamente justificada. La elección se fija antes de ver los resultados, con rangos y márgenes pertinentes; con sólo tres conjuntos la potencia global es limitada. Se informarán también diferencias, IC y efectos por conjunto y la consistencia del resultado, sin mezclar RMSE de escalas distintas en un promedio bruto. Un conjunto usado para rediseñar el modelo pasa a desarrollo y deja de aportar confirmación independiente.

## 8.4 Definición operativa de superioridad R3

La guía exige **seis condiciones simultáneas**; ninguna se sustituye por otra:

| Condición de la sección 8.4 | Evidencia de este corte | Cierre necesario |
|---|---|---|
| 1. Mejor media primaria frente a cada rival | E3 tiene la menor RMSE media individual de los 17 sistemas; MAE también favorable. | Observado en el calendario histórico, con tratamientos de entrenamiento distintos. |
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
$$

con contribución cero cuando ambos valores de un término sMAPE son cero. La media de R² no es el R² de un ensamble. E3 obtiene sMAPE **11,0977 %** y R² medio **0,911779** en este calendario. La [tabla abreviada del corte estadístico](../_static/e3_estadistica_posterior_20261117/ranking_table.csv) conserva sus bytes; el suplemento amplía métricas descriptivas y no cambia sus p-valores.

La cobertura y anchura de E2 se reconstruyen exclusivamente con los cuantiles residuales **ya congelados** de cada predictor, mediante $[\max(0,\widehat y-q),\widehat y+q]$; no se ajustó un nuevo calibrador con prueba. Son medias de las coberturas y anchuras individuales, no de un intervalo de ensamble. **E3 no tiene un cuantil ni intervalo calibrado congelado**, por lo que sus dos campos continúan pendientes. La DE entre semillas no rellena esas casillas y el nominal 90 % no constituye garantía individual bajo dependencia o cambio temporal.

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
$$

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
$$

En este corte $m=48$. Se informa por rival el **máximo** de sus tres p ajustados: todos deben rechazar. Holm controla el error familiar bajo p válidos sin exigir independencia entre pruebas; no corrige una prueba inválida o el uso previo del test para diseñar.

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

**No se estimó potencia prospectiva, potencia conjunta ni MDE a partir de este test.** Diez semillas cumplen una cantidad de repeticiones; no demuestran el objetivo de potencia de la guía. Calcular potencia observada usando el efecto de este mismo resultado no reemplazaría un dimensionamiento previo con un piloto adecuado.

La guía exige un piloto con **al menos cinco semillas** para estimar la dispersión de diferencias pareadas frente a cada rival y dimensionar la campaña antes de congelarla: potencia de al menos 0,90 por contraste, potencia conjunta reportada y diez semillas como mínimo o más si hace falta. Deben incluirse las decisiones de las tres pruebas y Holm; semillas y calendario no son fuentes de incertidumbre intercambiables. El piloto y la alternativa planificada no se eligen retrospectivamente para declarar suficiente el tamaño ya usado.

La aproximación normal bilateral ilustrada en la guía, con $\alpha'=0,05/m$, es

$$
R\simeq\frac{(z_{1-\alpha'/2}+z_{1-\beta})^2\,\widehat\sigma_d^{\,2}}{g^2},
$$

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
$$

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

Continúan pendientes la sensibilidad a hiperparámetros propios, las ablaciones comparables, las perturbaciones y curvas de tamaño de muestra, la calibración y cobertura del candidato, diagnósticos completos de adecuación, explicaciones verificadas, costes comparables y los tres conjuntos adicionales. Las sensibilidades de bloques y semillas aquí reportadas no sustituyen esos estudios.

### Estudios exigidos por la Tabla 5

| Estudio | Evidencia actual y mínimo que falta |
|---|---|
| Sensibilidad a hiperparámetros | Falta variar hiperparámetros propios con superficies/curvas, región de ventaja y sensibilidad frente a benchmarks. Replicar la receta aditiva no constituye HPO ni este estudio. |
| Estabilidad por semilla y partición | Hay diez inicializaciones de una receta sobre un mismo calendario; se reportan dispersión y subgrupos. Falta variación pertinente de particiones, IC/coeficiente de variación con unidades claras y la comparación ANOVA/Kruskal–Wallis sólo cuando su diseño la justifique. Las semillas no son mercados independientes. |
| Perturbaciones controladas | Falta aplicar a toda la familia al menos tres severidades y curvas de degradación. En OHLCV, ruido, datos faltantes o atípicos deben respetar o declarar la coherencia temporal/física; no se traslada sin cautela el estrés artificial de E2. |
| Tamaño de muestra y escalabilidad | Faltan curvas de aprendizaje —por ejemplo 10 %, 25 %, 50 % y 100 %— con varias semillas, más tiempo y memoria al variar tamaño de entrada. Esos porcentajes ilustran el estudio, no corridas realizadas. |
| Cambio de distribución | Falta un periodo, dominio o fuente fuera de toda decisión de diseño. Los conjuntos adicionales y sus derechos/particiones se eligen antes de ver desempeño; reutilizar un calendario conocido no lo vuelve independiente. |
| Calibración y reglas de puntuación | Falta calibración de E3 con datos autorizados y evaluación separada. Intercepto/pendiente y residuos son pertinentes a la salida puntual; pinball, CRPS o log-score requieren cuantiles/distribución definidos. No se inventan ECE o probabilidades de clasificación para este objetivo. |
| Cuantificación de incertidumbre | E2 conserva cuantiles residuales por predictor; E3 no tiene intervalos calibrados congelados. Faltan cobertura empírica, IC temporal, subgrupos y anchura de E3. El nominal 90 % no es garantía individual y la dependencia limita intercambiabilidad. La DE de semillas no reemplaza un intervalo predictivo. |
| Eficiencia y frontera de Pareto | Parámetros y algunos costes están inventariados; faltan FLOPs completos, latencias CPU/GPU, memoria pico, entrenamiento y energía con método común. Las tres referencias TorchScript no representan el catálogo completo ni una medición de Render. |
| Análisis de errores | El piloto conserva diagnósticos posteriores por régimen: en 18 fechas de objetivo realizado alto la aditiva tuvo RMSE 22,50 % mayor que M00 proyectado. Ese grupo no está disponible al pronosticar y no identifica una causa. Faltan escenarios prefijados, errores compartidos con rivales y discusión de fallos sin ocultar desenlaces adversos. |

Las intervenciones y criterios se fijarán antes de medir la evaluación correspondiente, con el mismo protocolo para todos los rivales. Una explicación por atención requiere además fidelidad y estabilidad comparadas con el mejor benchmark; no se convierte en evidencia causal sólo por visualizar sus pesos.

## 8.9 Reporte obligatorio de resultados

| Producto | Evidencia de este corte | Límite |
|---|---|---|
| Tabla de 17 sistemas | RMSE, DE, MAE, sMAPE y R²; cobertura/anchura E2 reconstruidas de cuantiles congelados e IC bootstrap entre semillas disponibles por separado | Mismas fechas y tratamientos de entrenamiento distintos; E3 sin intervalos calibrados y recursos completos comparables pendientes |
| Familia completa | 48 contrastes, p crudos/Holm, estadísticos, efectos, IC, diagnósticos y conclusión | Retrospectivos; distintos ámbitos; los supuestos y la admisibilidad no se certifican sólo por rechazar |
| Sensibilidad y MAE | Bloques 7/14/28, subgrupos de semillas y Holm16 secundario | No perturbaciones o nuevos conjuntos |
| MCS con original | Diez sistemas retenidos al 90% | MSE auxiliar; no ganador único o equivalencia |
| Potencia y MDE | Declarados como no estimados | Requieren un diseño/piloto pertinente |
| Tablas A/B del original | Criterios y diagnósticos necesarios identificados en 8.6 | Pendientes de E3 con riesgos comparables, interpretación y fuentes; no se heredan de otras redes |
| Tabla por conjunto adicional y prueba consolidada | Especificación de selección y análisis en 8.3 | Tres conjuntos sin elegir/evaluar, con familia completa y controles aplicados a cada uno |
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
