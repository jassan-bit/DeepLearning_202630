# 8 Bloque D Verificación estadística y evaluación ampliada

**Corte del 17 de noviembre de 2026: análisis retrospectivo ejecutado sobre predicciones congeladas.** La CNN-LSTM con atención de contenido y salida aditiva positiva tiene diez semillas; cada una de las trece redes rivales también tiene diez. Persistence, SeasonalNaive y ARIMA conservan una trayectoria determinista cada uno. La comparación incluye los diecisiete sistemas sobre las mismas 608 fechas, sin eliminar rivales, semillas ni observaciones.

Los **48 contrastes principales** —dieciséis rivales por tres procedimientos— rechazaron sus nulas frente al margen heredado, con **Holm sobre los 48 p-valores** y máximo p ajustado **0,0019**. Las sensibilidades temporales conservaron esa decisión. Este resultado está condicionado al calendario y a los supuestos de cada método: **no acredita por sí solo R3**, una batería confirmatoria homogénea o la originalidad del mecanismo. El MCS auxiliar conserva diez sistemas, incluido E3; no identifica un ganador único.

## 8.1 Protocolo experimental pre-registrado

**No existe un preregistro confirmatorio de esta comparación.** El [borrador de Bloque D](https://github.com/jassan-bit/DeepLearning_202630/blob/c4a4a1e68e9844cdfb81de0075177ced0cd96c07/experiments/e3_bloque_d/protocolo_borrador.md) continúa identificado como borrador. El plan de esta etapa se fijó en el commit [4a21b032](https://github.com/jassan-bit/DeepLearning_202630/commit/4a21b0329b99e864ec64fc2df49260aa874141bf), después de conocer las predicciones y sus métricas medias y antes de calcular estos p-valores. El código probado se fijó en [02132664](https://github.com/jassan-bit/DeepLearning_202630/commit/0213266404a0aa3af985d7306dce6b13892c71ec) antes de ejecutar el análisis.

La cronología permite reproducir las decisiones de esta etapa; no restaura el cegamiento del test. Se conservan el {download}`plan con once insumos y SHA256 <../_static/e3_estadistica_posterior_20261117/plan.json>` y el {download}`protocolo de análisis posterior <../_static/e3_estadistica_posterior_20261117/protocol.md>`.

El {download}`borrador histórico de protocolo <../_static/e3_bloque_d/protocolo_borrador.md>` conserva su archivo y su enlace de descarga anteriores. Su estado de borrador no cambia por esta evaluación posterior.

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

## 8.3 Evaluación en conjuntos de datos adicionales

**Pendiente.** Este análisis sólo usa una cronología BTC conocida. No se han evaluado tres conjuntos adicionales ni se ha demostrado consistencia entre dominios. Las semillas no son conjuntos de datos independientes; por eso no se ejecuta Friedman, Iman–Davenport o un diagrama de diferencia crítica usando las diez inicializaciones como si fueran diez mercados.

La elección y evaluación de los conjuntos adicionales, sus márgenes por escala, licencias, auditorías y comparación completa siguen requiriendo un diseño previo. No se promedian RMSE de escalas distintas ni se denomina confirmación independiente a otro tramo ya expuesto.

## 8.4 Definición operativa de superioridad R3

| Requisito | Evidencia de este corte | Estado |
|---|---|---|
| Mejor media que cada rival | E3 tiene el menor RMSE y MAE medios de los 17 sistemas | Observado en este calendario |
| Ganancia mayor que delta con control familiar | 48/48 rechazos bajo los procedimientos posteriores y sus supuestos | Evidencia exploratoria condicionada |
| Batería confirmatoria F2a, tres pruebas admisibles y coherentes | Bootstrap temporal, HAC de influencia RMSE y t entre semillas tienen ámbitos diferentes | No acreditada por su mera combinación |
| Efecto y estabilidad | Ganancias, intervalos, Cohen dz entre semillas y sensibilidades disponibles | Condicionados a estas trayectorias/calendario |
| Potencia previa y conjunta | No se dimensionaron con un piloto independiente para esta comparación | Pendiente |
| Comparación justa y conjuntos adicionales | Entrenamientos 655 frente a 532; tres conjuntos sin evaluar | Pendiente |
| Adecuación, calibración, robustez y mecanismo | Esta etapa no ejecuta esas baterías ni verifica novedad o defensa humana | Pendiente |

Por tanto, **R3 permanece sin acreditar**. La inferioridad no rechazada, la pertenencia al MCS o un p pequeño no prueban equivalencia, originalidad o cumplimiento del entregable completo.

## 8.5 Familia de contrastes

Se conservaron los dieciséis rivales fijados en el borrador:

| Grupo | Modelos |
|---|---|
| Top 10 de E2 | FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST, iTransformer |
| Referencias neuronales | MLP16, LSTM32, GRU32 |
| Clásicos deterministas | Persistence, SeasonalNaive, ARIMA |

```{csv-table} Métricas sobre las 608 fechas: media individual y dispersión entre semillas
:file: ../_static/e3_estadistica_posterior_20261117/ranking_table.csv
:header-rows: 1
```

```{figure} ../_static/e3_estadistica_posterior_20261117/ranking_rmse.svg
:alt: Los diecisiete sistemas ordenados por RMSE medio; las redes incluyen su desviación estándar entre diez semillas.
:width: 100%

RMSE medio y DE muestral entre inicializaciones. Los clásicos tienen R=1 y no reciben una barra de dispersión artificial. La DE entre semillas no es un intervalo predictivo ni temporal.
```

La CNN-LSTM obtuvo **RMSE medio 0,0029394324** y **MAE medio 0,0018928273**. El mejor rival por RMSE fue LSTM-autoencoder, con **0,0038728895**: la reducción relativa observada de E3 fue **24,10 %**. Frente a ese rival, la ganancia absoluta fue **0,0009334571**, con IC95% temporal marginal **[0,0006329359; 0,0012131746]**.

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

Los tres procedimientos no se presentan como tres confirmaciones temporales homogéneas ni como acreditación automática de la Tabla 4. No se aplican correcciones Nadeau–Bengio a días o semillas de esta única cronología, ni Clark–West sin justificar anidamiento.

## 8.7 Estrategia estadística: familias, potencia y agregación

### 8.7.1 Control del error por familia

Se fijó **una familia principal de 48 hipótesis**, con Holm a alfa=0,05. Este control es más amplio que las tres familias de dieciséis propuestas en el borrador. No se eligió la prueba de menor p ni se excluyeron rivales después de ver el resultado. Para la descripción conjunta por rival se exige que los tres p ajustados sean menores que 0,05.

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

Los consumos históricos se mantienen cerrados y separados. El {download}`registro de ejecución CPU <../_static/e3_estadistica_posterior_20261117/execution.json>` identifica versiones, hashes y tiempo numérico; no se añaden ajustes o inferencias del modelo.

### 8.7.3 Agregación jerárquica y múltiples conjuntos de datos

Las unidades se mantienen diferenciadas: una serie compartida, bloques temporales, diecisiete sistemas y sus inicializaciones. Los 6080 pares semilla-fecha de una red **no se tratan como 6080 casos iid**. La t entre semillas y la inferencia temporal condicionan aspectos distintos; no forman una estimación conjunta de incertidumbre de entrenamiento, selección y mercado.

La {download}`tabla de efectos <../_static/e3_estadistica_posterior_20261117/effects.csv>` incluye Cohen dz de ganancias pareadas entre semillas e intervalos bootstrap nominales, con su alcance condicionado al calendario. No se interpreta como efecto entre mercados independientes.

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

## 8.9 Reporte obligatorio de resultados

| Producto | Evidencia de este corte | Límite |
|---|---|---|
| Tabla de 17 sistemas | Métricas medias, DE entre semillas y figuras con unidades | Mismas fechas; tratamientos de entrenamiento distintos |
| Familia completa | 48 contrastes, p crudos/Holm, efectos e intervalos | Retrospectivos; ámbitos heterogéneos; no batería confirmatoria acreditada |
| Sensibilidad y MAE | Bloques 7/14/28, subgrupos de semillas y Holm16 secundario | No perturbaciones o nuevos conjuntos |
| MCS con original | Diez sistemas retenidos al 90% | MSE auxiliar; no ganador único o equivalencia |
| Potencia y MDE | Declarados como no estimados | Requieren un diseño/piloto pertinente |
| Adecuación y evaluación ampliada | Pendientes identificados | No se rellenan con ceros ni resultados heredados de otros modelos |

Las **42 pruebas de métodos e insumos aprobaron en integración continua**, con los archivos inmutables verificados y casos artificiales para los procedimientos numéricos. El {download}`registro de verificación <../_static/e3_estadistica_posterior_20261117/verification.json>` enlaza la ejecución de CI y distingue esa comprobación de una reproducción humana independiente.

Los {download}`resultados numéricos completos <../_static/e3_estadistica_posterior_20261117/result.json>`, el {download}`manifiesto de artefactos <../_static/e3_estadistica_posterior_20261117/manifest.json>` y el {download}`inventario de publicación <../_static/e3_estadistica_posterior_20261117/publication_manifest.json>` permiten comprobar las tablas. El código y sus pruebas se conservan en [experiments/e3_estadistica_posterior_20261117](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_estadistica_posterior_20261117).

Para reproducir, se instalan las versiones de `requirements.txt` y se ejecuta `python experiments/e3_estadistica_posterior_20261117/run.py --download --output work/e3_estadistica_reproduccion`; `report.py` exporta las tablas de lectura desde esa salida. Los datos se descargan desde commits inmutables y se verifican por SHA256. No se reentrena el modelo ni se requieren GPU o Google Drive.

La conclusión admisible es **evidencia retrospectiva favorable a la receta E3 bajo los procedimientos y supuestos declarados**. No se retrofecha el protocolo, no se oculta la diferencia de entrenamiento y no se transforma este análisis en certificación R1, R2 o R3.
