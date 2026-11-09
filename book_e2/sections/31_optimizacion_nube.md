# 31. Optimización de CPU y presupuesto de cómputo en la nube

**Resultado:** se eligieron el recorte de CPU de TimesNet y dos hilos. Completaron 85/86 perfiles ampliados y los treinta ajustes del nuevo piloto. La nula del generador pasó sus dieciséis validaciones, pero ningún escenario de R respalda el objetivo de 0,80. El R final y el presupuesto primario siguen abiertos; las 7.375,994 horas de la sensibilidad con microperfil son trabajo nominal condicionado.

El trabajo se ejecuta en el servidor remoto de **Codex Cloud**, con una cuota de **cuatro vCPU, 16 GiB de RAM y sin GPU visible**. Los ajustes son secuenciales. El entrenamiento de esta sección no utiliza la CPU del equipo del usuario.

## Registro previo y continuidad del piloto

El [registro de esta medición](https://github.com/jassan-bit/DeepLearning_202630/blob/d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2/experiments/e2_compute_optimization_20261109/registration.json) y el código se publicaron en el commit `d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2`, antes de nuevos ajustes de mercado. El SHA256 del registro es `3b54ac44f5aed956ac98fbc31c95dfcb70c1116b6cca65a8269ff11cd843ac91`. La fecha del registro es 09/11/2026, 13:24:55, hora de Bogotá; los diarios conservan además sus marcas UTC.

La [sección 30](30_piloto_nube.md) conserva su resultado principal incompleto, sus cinco selecciones fallidas de TimeMixer, la censura del extremo TimesNet y sus costes. Los nuevos intentos tienen semillas y recibos propios. No se reemplaza un fallo anterior por un ajuste más rápido ni se reescribe su estado.

Este trabajo mide ejecución, no selecciona una arquitectura por su RMSE. Se mantiene la receta fija anterior al piloto: AdamW, LR 0,001, WD 0,0001, batch 32 y warmup 0. Las variantes NAS conservan sus bits de dropout registrados; TimesNet mantiene `top_k=4` y seis kernels. Los resultados de HPO conocidos no se reutilizan como recetas ganadoras de esta medición.

## Datos y cortes temporales

Se usan exclusivamente los dos derivados de desarrollo 2023–2024 fijados por hash en el registro. El objetivo sigue siendo **σ₇ poblacional de siete retornos logarítmicos diarios, ddof=0 y sin anualización**. La barra incompleta del 24/03/2023 conserva su posición en el calendario. No se interpolan los días ni se comprime la serie.

| Cohorte | Origen de la selección | Entrenamiento | Validación | Uso |
|---|---|---|---|---|
| Pequeña | 01/10/2023 | 161 ventanas, hasta 25/06/2023 | 83 fechas, 03/07…23/09/2023 | Comparación de kernels e hilos y referencias de escalado |
| Grande | 01/01/2025, sólo marcador de calendario | 619 ventanas, hasta 25/09/2024 | 83 fechas, 03/10…24/12/2024 | Perfil de arquitecturas y cuerpos fijos |
| Selección de épocas del nuevo piloto de variación | 01/10/2024 | 527 ventanas, hasta 25/06/2024 | 83 fechas, 03/07…23/09/2024 | Cinco ajustes frescos por modelo |
| Reajuste fresco del nuevo piloto | 08/11/2024 | 655 ventanas, hasta 31/10/2024 | Sin etiquetas de evaluación para parar | Predicciones en 53 fechas, 08/11…30/12/2024 |

El origen de enero de 2025 permite construir el corte de desarrollo mediante posiciones futuras no disponibles; **no incorpora valores de mercado de 2025**. No se abren archivos de 2025–2026 ni las 608 fechas externas. Los datos de desarrollo tienen exposición histórica y no se presentan como una muestra confirmatoria intacta.

## Optimización de TimesNet y elección de hilos

La optimización de CPU aprovecha que los pequeños grids de TimesNet sólo utilizan una parte central de los kernels con padding de ceros. Recorta ese cálculo antes de la convolución y conserva los objetos de parámetros, sus formas, las claves de estado y los metadatos arquitectónicos. Los **225.075.755 parámetros** del extremo siguen presentes; el procedimiento no reduce la arquitectura ni convierte sus pesos en un modelo menor.

La [prueba de equivalencia sintética](https://github.com/jassan-bit/DeepLearning_202630/blob/d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2/experiments/e2_compute_optimization_20261109/cpu_conv_parity.json) pasó once casos, incluidos salidas, gradientes, estado de AdamW, identidad de parámetros y conservación del RNG. Se comprobaron 36 geometrías directas en FP64 y casos FP32. Es una identidad algebraica para pesos y entradas finitos; el redondeo puede producir diferencias dentro de las tolerancias verificadas. Esa prueba acredita corrección, no una aceleración observada.

La decisión de ejecución sigue tres fases registradas:

| Fase | Intentos previstos | Trabajo fijo | Regla de decisión |
|---|---:|---|---|
| Kernels | 8 | TimesNet de referencia y extremo, versión original y recortada, dos semillas; una época sobre 32 ventanas y evaluación de 83 fechas | Activar el recorte sólo si completan los cuatro intentos recortados y, en ambos tamaños, su mediana del tiempo de proceso mejora al menos 5% |
| Hilos | 12 | TimeMixer y TimesNet de referencia, 1/2/4 hilos, dos semillas; cinco épocas sobre la cohorte pequeña | Comparar medianas de construcción+ajuste+evaluación por modelo, normalizadas por cuatro hilos; entre opciones a 5% del mínimo de la suma, elegir la de menos hilos |
| Escalado | 86 | 72 casos NAS grandes, seis referencias NAS pequeñas y ocho cuerpos fijos grandes | Dos épocas por etapa; conservar cada resultado y descomponer el coste para la proyección |

Una censura por tiempo de la versión original de TimesNet sólo aporta un límite inferior de duración a la comparación; otros fallos bloquean la activación del recorte. Si la cuadrícula de doce intentos de hilos queda incompleta, se conserva la política anterior de cuatro hilos y se registra esa falta de evidencia. Los errores predictivos no eligen kernels ni hilos.

Los ocho cuerpos fijos son las cinco recetas sin NAS y las tres referencias neuronales E1. Los 72 casos NAS cubren seis familias por los doce patrones de §27: tres anclas y una muestra uniforme de cada uno de nueve estratos. El autoencoder conserva dos fases auxiliares y una supervisada, con dos épocas en cada fase.

### Resultado observado del recorte

Completaron **los ocho intentos**, con ocho pasos AdamW conocidos y sin censuras. Se invirtió el orden de las versiones entre semillas para reducir el efecto del orden de ejecución. Los tiempos siguientes incluyen el proceso completo y no se limitan a la convolución.

| Patrón TimesNet | Versión CPU | Semilla 272301 (s) | Semilla 272302 (s) | Mediana (s) |
|---|---|---:|---:|---:|
| Referencia 64/64/64, bits 010 | Original | 5,222 | 5,135 | 5,179 |
| Referencia 64/64/64, bits 010 | Recorte | 5,014 | 4,645 | 4,830 |
| Extremo 256/256/256, bits 111 | Original | 106,770 | 50,116 | 78,443 |
| Extremo 256/256/256, bits 111 | Recorte | 19,741 | 19,523 | 19,632 |

La reducción de la mediana fue **6,74% en la referencia y 74,97% en el extremo**. Completaron los cuatro procesos recortados y ambas razones recorte/original quedaron por debajo de 0,95; por tanto, la regla registrada eligió el **recorte de CPU** para las fases siguientes. No se utilizaron puntuaciones predictivas en esa decisión.

Son dos repeticiones de un lote de 32 ventanas y una época. La dispersión observada de la versión original del extremo impide interpretar el porcentaje como una aceleración precisa y universal. Este resultado no demuestra la duración del entrenamiento completo de esa arquitectura.

### Resultado observado de hilos

Los doce ajustes completaron, con **360 pasos AdamW** conocidos y 215,467 segundos de procesos completos sumados. La comparación usó la versión recortada de TimesNet y mantuvo la misma configuración de cada familia.

La política de hilos compara el **tiempo de trabajo de construcción, ajuste y evaluación**, medido por el código registrado. Ese intervalo excluye arranque, importaciones y carga de datos. Es distinto del tiempo del proceso completo empleado en la decisión de kernels y en la suma de 215,467 segundos.

| Hilos | Mediana de trabajo TimeMixer (s) | Mediana de trabajo TimesNet (s) | Suma normalizada respecto de 4 hilos |
|---:|---:|---:|---:|
| 1 | 3,089 | 24,842 | 1,6140 |
| 2 | 3,154 | 17,086 | 1,3866 |
| 4 | 3,724 | 31,663 | 2,0000 |

La regla registrada eligió **dos hilos globales**. Aunque TimeMixer fue ligeramente más rápido con uno, la suma normalizada fue menor con dos; la opción de un hilo quedó fuera del margen de empate del 5%. Las fases posteriores usan esta política común. Dos semillas de dos familias en este servidor no garantizan la misma mejora para cada arquitectura ni para futuras condiciones de carga.

### Resultado observado del escalado

Se intentaron **los 86 perfiles primarios: 85 completos y un fallo por tiempo**. Los seis perfiles NAS de referencia pequeños y los ocho cuerpos fijos grandes completaron. En los 72 perfiles NAS grandes, completaron 71.

| Grupo primario | Intentos | Completos | Fallos |
|---|---:|---:|---:|
| Seis familias NAS × doce patrones, cohorte grande | 72 | 71 | 1 |
| Referencias de las seis familias NAS, cohorte pequeña | 6 | 6 | 0 |
| Cinco cuerpos fijos y tres referencias neuronales E1, cohorte grande | 8 | 8 | 0 |
| Total primario | 86 | 85 | 1 |

El fallo fue TimesNet 256/256/256, bits 111, aun usando recorte y dos hilos. Alcanzó el límite interno de 300 segundos comprobado en fronteras del entrenamiento: dejó **38 pasos conocidos**, una época completa y 309,537 segundos de proceso consumidos. No fue un OOM observado. No se completó su segunda época ni la evaluación final.

El microperfil separado de esa misma arquitectura completó un paso sobre 32 ventanas en **29,263 segundos**, con su semilla registrada y vínculo al fallo. No cambia el estado del perfil grande. Entre los 86 perfiles y ese microperfil se registraron **4.255 pasos conocidos, ninguno desconocido, 1.471,760 segundos de procesos sumados —24,53 minutos— y un pico RSS muestreado de 4,685 GiB**. El coste del fallo y del microperfil están incluidos en esos totales.

La auditoría de huellas de las fuentes no encontró discrepancias; el resumen interno original de escalado tiene SHA256 `865abe0e8dec2dc41f57f33ea69c0f2eb04ef1ab903cbbc6483f98ad09292b4e`. Completar perfiles de dos épocas no constituye una búsqueda NAS/HPO ni demuestra que cada arquitectura complete el presupuesto científico posterior.

### Incidencia al guardar el resumen de hilos

Los doce ajustes de la comparación de hilos completaron y conservaron sus recibos individuales. Al guardar el resumen de la fase, el serializador estricto rechazó las claves enteras `1`, `2` y `4` del diccionario de puntuaciones. Es un fallo administrativo del reporte, separado de un fallo de entrenamiento: no invalida ni convierte en incompletos los ajustes ya registrados.

La [enmienda administrativa](https://github.com/jassan-bit/DeepLearning_202630/blob/c768b51fdf0077d3cf0d52e79e3db824d3453c61/experiments/e2_compute_optimization_20261109/serialization_amendment.json) se publicó antes de continuar con los 86 perfiles de escalado. Cambió únicamente el tipo de las claves para permitir guardar el JSON. Diez pruebas, incluidas serialización, lectura y carga de la política de ejecución, pasaron antes de continuar.

El resumen se reconstruyó a partir del plan, los doce resultados y sus cadenas de eventos inmutables. No se repitió ningún ajuste ni se cambió un tiempo. El resumen interno reconstruido tiene SHA256 `5d8b0325be3394bb0227b1e5e9ea8b8c0a4536fb70b1901f6263d03cbdc2ac4f`. Se conserva la versión original del código junto con los registros previos; la corrección no afecta los cálculos de entrenamiento ni la decisión de dos hilos.

### Costes, censuras y microperfiles

El supervisor registra tiempo de proceso, CPU y pico RSS muestreado. El perfil separa construcción, preparación, bucle de entrenamiento, evaluaciones y cierre. Los límites internos/externos son 210/240 segundos para kernels e hilos y 300/360 segundos para escalado. Se muestrea un límite de 12 GiB por grupo de procesos; el muestreo no es una barrera de asignación del kernel.

Los límites cortos son instrumentos para medir recursos y no sustituyen las asignaciones científicas completa/corta de §27. Si termina externamente un proceso y no se conocen sus pasos, éstos permanecen `null`. Los costes consumidos se incluyen aunque el intento no complete.

Sólo un fallo de recurso o tiempo del perfil de escalado permite un **microperfil separado**, con semilla 272501, una época y 32 ventanas. Se conserva su vínculo al intento original. Un microperfil mide rendimiento de ese lote; no acredita que complete la cohorte grande ni reemplaza una observación censurada. Los fallos de implementación o numéricos bloquean la proyección afectada.

## Presupuesto condicionado de horas y almacenamiento

La estimación expande los costes de los doce perfiles de cada familia con sus probabilidades de inclusión. Las tres anclas se cuentan una vez; cada muestra aleatoria se multiplica por el tamaño de su estrato. Los pesos suman mil arquitecturas. Promediar los doce perfiles por igual distorsionaría el espacio.

Se proyectan por separado el coste por paso de optimización, la evaluación de entrenamiento y la validación de cada época de selección. La preparación y el cierre se añaden una vez por etapa. Se utilizan los tamaños de entrenamiento de cada origen, los lotes de las 18 asignaciones HPO y las tres fases SAE cuando corresponden. Los escenarios conservan **k=40** y muestran **R=6, 10, 20 y 30** como asignaciones de recursos.

La unidad de las proyecciones es **horas de ejecución serial con dos hilos**: se expande tiempo de pared medido. No representa horas por núcleo ni una suma de tiempo CPU de procesos. En los JSON de costes, `core` identifica el subtotal de HPO, NAS y reajustes; no indica una cantidad de núcleos.

El fallo del extremo TimesNet impide completar el estimador primario de todas las familias: **el total primario permanece `null` en los 16 escenarios**, sin eliminar la muestra lenta. Para k=40, R=30, máximos 50/200 épocas y coste constante por paso, el **subtotal proyectado** de los trece modelos restantes es **1.428,121 horas**. No es tiempo observado. Ese subtotal omite TimesNet y no es una cota inferior ni una estimación total de campaña. El tiempo observado de los perfiles y su microperfil fue 24,53 minutos de procesos sumados.

### Sensibilidad de trabajo nominal sin truncar que utiliza el microperfil

La siguiente tabla transfiere la tasa del microperfil al caso grande no completado. Todos sus valores son **horas de ejecución serial con dos hilos del trabajo nominal HPO+NAS+reajustes**, bajo supuestos adicionales sin validar. No subsanan el fallo primario ni certifican que el caso grande complete el ajuste o que quepa en memoria durante toda su trayectoria.

**La proyección no aplica los límites científicos de 1.800 segundos por etapa corta y 7.200 por etapa completa.** Suma el trabajo nominal de las épocas previstas sin truncarlo por tiempo. La demanda puede incluir trayectorias que excedan esos límites; el informe de horas no conserva máximos por trayectoria ni acredita que cada ajuste pueda completarse. Si una etapa futura alcanza el límite, queda registrada como fallo o censura con su coste consumido, sin garantizar las épocas previstas. Los límites permanecen sin cambios. Esta tabla no es una predicción de duración de una campaña completada; el fallo observado a 300 segundos tampoco es un fallo ya observado a 7.200 segundos.

| R como escenario de recursos | Máximos 50/200 épocas, coste constante por paso | Máximos 50/200, coste proporcional al batch | Sensibilidad 25/100 épocas, coste constante por paso | Sensibilidad 25/100, coste proporcional al batch |
|---:|---:|---:|---:|---:|
| 6 | 6.900,803 | 6.855,093 | 3.464,219 | 3.441,364 |
| 10 | 6.980,002 | 6.934,291 | 3.503,922 | 3.481,067 |
| 20 | 7.177,998 | 7.132,287 | 3.603,178 | 3.580,323 |
| 30 | 7.375,994 | 7.330,283 | 3.702,435 | 3.679,579 |

El primer escenario mantiene el coste medido por paso cuando cambia el batch; el segundo lo multiplica por batch/32. Son supuestos de transferencia, no límites demostrados. Ambos suponen que los finalistas NAS y las arquitecturas reajustadas cuestan como la media uniforme del espacio; no se conoce aún el coste de las recetas ganadoras. La sensibilidad 25/100 es ilustrativa: no representa una parada temprana observada ni cambia los presupuestos registrados de 50/200 épocas.

La proyección HPO transfiere el perfil de referencia de cada receta; no mide cada combinación de sus 18 asignaciones. Los doce patrones perfilados corresponden al espacio de arquitectura NAS. No se verificó el mismo coste para los lotes HPO 16/32/64 ni para sus alternativas estructurales: top-k y número de kernels de TimesNet, media móvil de TimeMixer o kernel del híbrido. La variación de warmup y otros hiperparámetros tampoco garantiza una tasa de ejecución idéntica.

El caso principal de sensibilidad, k=40/R=30/50–200/coste constante, suma **7.375,994 horas de trabajo nominal sin truncar, equivalentes a 307,3 días sin interrupciones**. Ese número dimensiona el subtotal modelado con el proxy; no es una cota superior ni un pronóstico firme. **La duración de la campaña completa permanece desconocida**.

La comprobación de transferencia entre cohortes de referencia tampoco valida una extrapolación exacta: las razones tiempo pequeño observado/proyectado desde el grande fueron 1,073 para SAE, 1,332 TimesNet, 1,132 TiDE, 2,617 TimeMixer, 1,295 PatchTST y 2,164 híbrido. Esos seis cocientes describen el error de esa comprobación corta; no forman una banda de error de toda la campaña.

Los techos contables de asignación que incluyen pilotos y perfiles globales tienen un alcance distinto de este subtotal HPO+NAS+reajustes. No validan por comparación la viabilidad ni la precisión de las horas proyectadas.

Estas horas son una proyección condicionada a las tasas medidas y al modelo de escalado; no son duración observada de la campaña ni un intervalo de confianza. Una sola muestra por estrato no identifica su varianza interna. Tampoco permite declarar insesgada la duración real cuando las predicciones de coste por arquitectura tienen error. La transferencia de lotes, cohortes y épocas debe mantenerse visible.

Quedan fuera de la proyección el arranque de procesos, las importaciones, la admisión de datos y la preparación de escaladores que no estén cubiertas por los temporizadores. También quedan fuera los ajustes clásicos no perfilados, escritura de checkpoints a disco, controles, robustez, calibración adicional de modelos, interpretabilidad, inferencia estadística y reproducción independiente. No se convierte esta duración en dinero sin una tarifa del servicio.

### Almacenamiento de checkpoints

El inventario de parámetros y estados se obtiene mediante construcciones `meta`, sin crear pesos CPU ni entrenar. Para la estructura fija de perfilado, conservar un checkpoint de cada una de las mil arquitecturas TimesNet en once orígenes suma **2.637.798.146.720 bytes, aproximadamente 2,64 TB decimales**, sólo en esa fase del cribado NAS. No incluye toda la campaña.

Las proyecciones de retener todos los checkpoints y de conservar únicamente los reajustes seleccionados son escenarios distintos. Los pesos de arquitecturas ganadoras aún se desconocen, por lo que el segundo escenario debe mostrar su rango condicionado. Los tamaños son payload de estados bajo las opciones inventariadas: no incluyen cabeceras de serialización, logs, optimizadores para reanudar, salidas históricas ni todos los posibles ejes HPO. No son una cota universal del almacenamiento.

Con **k=40** operativo, el inventario produce las siguientes asignaciones de payload. Los rangos abarcan las arquitecturas NAS posibles bajo las opciones estructurales inventariadas; las recetas ganadoras aún no se conocen.

| R como escenario de recursos | Retener checkpoints de todas las etapas (TiB) | Retener sólo reajustes seleccionados (GiB) |
|---:|---:|---:|
| 6 | 2,5268…4,0453 | 0,7486…57,9495 |
| 10 | 2,5273…4,0830 | 1,2476…96,5826 |
| 20 | 2,5285…4,1773 | 2,4952…193,1652 |
| 30 | 2,5297…4,2716 | 3,7428…289,7477 |

Se reserva conservadoramente la referencia además de k finalistas distintos; su inclusión efectiva entre los finalistas puede reducir la asignación. Un TiB equivale a 2⁴⁰ bytes y un GiB a 2³⁰ bytes. Estos escenarios no cambian el R científico.

El inventario permite decidir una política de retención antes de buscar. Esta sección no elimina checkpoints existentes ni acredita una optimización de almacenamiento ya aplicada al ejecutor.

## Nuevo piloto de variación antes de seleccionar recetas

Se registraron **TimeMixer, CNN-1D + LSTM + atención temporal y GRU32**, con cinco semillas nuevas por modelo: 273001…273005. Los quince ajustes para seleccionar épocas usan hasta 200 épocas por etapa, paciencia 20 y la política de CPU elegida por los perfiles independientes. Cada proceso tiene un límite interno/externo de 600/660 segundos y el mismo límite RSS muestreado de 12 GiB.

La campaña ejecuta primero las quince selecciones de los tres modelos y después los reajustes. Un modelo sólo recibe reajustes si completan sus cinco selecciones. Su cantidad fija de épocas es la mediana superior de las cinco mejores épocas. Los cinco reajustes reinician pesos, AdamW y escalador de entrenamiento, y no reciben etiquetas de evaluación para decidir la parada.

Las predicciones de las 53 fechas se registran antes de observar cada resultado. Se verifican los hashes de las fuentes, pesos y escaladores, y la carga segura de checkpoints de inferencia. Una réplica incompleta conserva su coste y no se sustituye. El análisis primario de tres modelos y cualquier análisis de grupos completos deben identificarse por separado.

La comparación con el piloto anterior cambia **tres elementos**: semillas nuevas, política de ejecución de CPU y límite de recursos. El límite interno/externo pasa de 150/180 a 600/660 segundos; se mantienen el máximo de 200 épocas y la paciencia de 20. Si TimeMixer completa ahora, ese cambio no puede atribuirse exclusivamente a la optimización. Las cinco selecciones fallidas de §30 siguen siendo resultados del presupuesto anterior.

| Modelo | Selecciones completas | Mejores épocas de las cinco semillas | Épocas fijas para reajuste | Reajustes completos |
|---|---:|---|---:|---:|
| TimeMixer | 5/5 | 64, 55, 70, 79, 56 | 64 | 5/5 |
| CNN-1D + LSTM + atención temporal | 5/5 | 57, 39, 23, 54, 43 | 43 | 5/5 |
| GRU32 | 5/5 | 164, 136, 98, 187, 109 | 136 | 5/5 |

Completaron **los treinta ajustes: quince selecciones y quince reajustes**, sin fallos, pasos desconocidos, bloqueos ni reemplazos. La cuarta semilla de GRU32 completó normalmente el máximo de 200 épocas durante la selección; ese cierre por máximo de épocas no se clasifica como un fallo de tiempo.

| Modelo | Pasos de selección | Pasos de reajuste | Segundos de procesos completos, ambas fases |
|---|---:|---:|---:|
| TimeMixer | 7.208 | 6.720 | 1.879,332 |
| CNN-1D + LSTM + atención temporal | 5.372 | 4.515 | 189,340 |
| GRU32 | 13.379 | 14.280 | 207,974 |
| Total | 25.959 | 25.515 | 2.276,646 |

Los recibos suman **51.474 pasos de optimización y 37,944 minutos de procesos**. Es tiempo observado del nuevo piloto, separado de las proyecciones de campaña. Se verificaron **30 checkpoints con carga segura y quince replays completos de 53 fechas**; las 51 fuentes permanecieron invariantes. El panel de tres modelos por cinco semillas está completo. El resumen original conservado en el entorno tiene SHA256 `22c531744ef5224e9db34c70470e86ba6d49f25993e5679452565bee42feb474`.

Este piloto nuevo sigue usando arquitecturas de referencia. Su variación no acredita la de las futuras recetas ganadoras de NAS/HPO ni cierra el dimensionamiento posterior a la selección.

## Calibración Monte Carlo de la planificación

La tasa nula de la simulación histórica de §30 no cumplía el nivel de referencia α/153. Se conserva ese recibo. El análisis nuevo usa una calibración explícita y una validación independiente; no interpreta la tasa histórica como potencia válida.

Para cada R de 6, 10, 20 y 30 y cada par observado, se emplean tres streams distintos de **200.000 realizaciones cada uno**: calibración nula, validación nula y alternativas. Los pools son 272401, 272402 y 272403. Se conservan bloques circulares de siete fechas, emparejamiento de semillas y fechas, y una sensibilidad superior de la desviación entre semillas bajo supuestos normales. La variación temporal mantiene un componente que no se divide por la raíz de R.

Todos los streams se centran respecto de la diferencia media observada original. La validación y las alternativas no se recentran con sus propias medias. Se aplica un crítico único a la diferencia absoluta y una regla de rechazo estricta `>`.

| Paso | Criterio registrado |
|---|---|
| Calibrar el crítico | Cola absoluta de diseño α_par/2 y tolerancia binomial para el estadístico de orden, con α_par=0,05/153 |
| Validar la nula | Límite superior unilateral Clopper–Pearson de una muestra independiente ≤α_par; no reajustar el crítico si falla |
| Evaluar alternativas | Desplazamientos tanto +δ como −δ; límite inferior Clopper–Pearson ≥0,80 en ambos signos y todos los pares observados |
| Recomendar un R condicionado | Sólo si pasa la validación nula y ambas alternativas de todos los pares observados |

Para P pares observados y cuatro valores de R, las tolerancias son γ_cal=γ_val=0,05/(4P) y γ_alt=0,05/(8P); la última incluye los dos signos del desplazamiento. La cobertura Monte Carlo se controla por separado para las familias de calibración, validación nula y alternativas. **No se afirma cobertura conjunta del 95%** entre las tres. El análisis del par histórico y el grupo nuevo son también resultados separados.

### Resultado de la calibración condicional

Ambos grupos coincidieron con el diseño registrado y conservaron exactamente las 53 fechas y sus cinco semillas completas. El grupo histórico se analizó por separado con híbrido y GRU32; el nuevo grupo usa TimeMixer, híbrido y GRU32. Se ejecutaron **2,4 millones y 7,2 millones de realizaciones**, respectivamente, sin entrenar modelos adicionales.

La validación nula pasó en **4/4 celdas históricas y 12/12 celdas nuevas**: todos los límites superiores CP quedaron por debajo de α_par≈0,000326797. **Ningún R de 6, 10, 20 o 30 respaldó el objetivo condicional de 0,80 en ambos signos y todos los pares.** Tanto `conditionally_supported_R` como `final_R` permanecen `null`.

Las tablas muestran proporciones, no porcentajes. Cada entrada de nula contiene tasa de rechazo y límite superior CP; cada alternativa contiene frecuencia de rechazo condicional y límite inferior CP. Los límites usan las tolerancias por familia descritas arriba y no son IC de potencia sobre el mercado real. Los números se redondean a ocho decimales; los recibos conservan su precisión original.

#### Par histórico, semillas 271901…271905

| Par | R | Crítico absoluto | Nula: tasa; LS CP | +δ: frecuencia; LI CP | −δ: frecuencia; LI CP |
|---|---:|---:|---|---|---|
| Híbrido−GRU32 | 6 | 0,00281096 | 0,00009000; 0,00015038 | 0,00003500; 0,00001063 | 0,00016500; 0,00010201 |
| Híbrido−GRU32 | 10 | 0,00240861 | 0,00007500; 0,00013134 | 0,00004500; 0,00001623 | 0,00025500; 0,00017457 |
| Híbrido−GRU32 | 20 | 0,00180032 | 0,00018500; 0,00026567 | 0,00025500; 0,00017457 | 0,00048500; 0,00037077 |
| Híbrido−GRU32 | 30 | 0,00168370 | 0,00010500; 0,00016911 | 0,00026500; 0,00018284 | 0,00032000; 0,00022885 |

#### Tres modelos nuevos, semillas 273001…273005

| Par | R | Crítico absoluto | Nula: tasa; LS CP | +δ: frecuencia; LI CP | −δ: frecuencia; LI CP |
|---|---:|---:|---|---|---|
| TimeMixer−Híbrido | 6 | 0,00490785 | 0,00012500; 0,00020715 | 0,00011500; 0,00005814 | 0,00005000; 0,00001638 |
| TimeMixer−Híbrido | 10 | 0,00400364 | 0,00005500; 0,00011554 | 0,00015000; 0,00008341 | 0,00005500; 0,00001919 |
| TimeMixer−Híbrido | 20 | 0,00301571 | 0,00007000; 0,00013595 | 0,00014500; 0,00007972 | 0,00005000; 0,00001638 |
| TimeMixer−Híbrido | 30 | 0,00257422 | 0,00006500; 0,00012921 | 0,00021000; 0,00012906 | 0,00002000; 0,00000262 |
| TimeMixer−GRU32 | 6 | 0,00427454 | 0,00009000; 0,00016237 | 0,00011000; 0,00005464 | 0,00001500; 0,00000123 |
| TimeMixer−GRU32 | 10 | 0,00335855 | 0,00004500; 0,00010155 | 0,00014500; 0,00007972 | 0,00002000; 0,00000262 |
| TimeMixer−GRU32 | 20 | 0,00252572 | 0,00013000; 0,00021343 | 0,00022000; 0,00013688 | 0,00001500; 0,00000123 |
| TimeMixer−GRU32 | 30 | 0,00209137 | 0,00010000; 0,00017533 | 0,00051000; 0,00037729 | 0,00007000; 0,00002816 |
| Híbrido−GRU32 | 6 | 0,00181117 | 0,00010500; 0,00018175 | 0,00057000; 0,00042902 | 0,00017000; 0,00009835 |
| Híbrido−GRU32 | 10 | 0,00158724 | 0,00008000; 0,00014926 | 0,00032000; 0,00021733 | 0,00004000; 0,00001109 |
| Híbrido−GRU32 | 20 | 0,00118257 | 0,00009500; 0,00016887 | 0,00067500; 0,00052055 | 0,00013000; 0,00006881 |
| Híbrido−GRU32 | 30 | 0,00102079 | 0,00007500; 0,00014263 | 0,00134500; 0,00112213 | 0,00015500; 0,00008711 |

#### MDE descriptiva bajo el generador, R=30

| Grupo y par | Efecto positivo descriptivo | Magnitud del efecto negativo descriptivo |
|---|---:|---:|
| Histórico: Híbrido−GRU32 | 0,00198200 | 0,00195598 |
| Nuevo: TimeMixer−Híbrido | 0,00302166 | 0,00299627 |
| Nuevo: TimeMixer−GRU32 | 0,00248325 | 0,00245884 |
| Nuevo: Híbrido−GRU32 | 0,00118559 | 0,00117826 |

Estas MDE se obtienen de cuantiles del generador y son sensibilidades descriptivas; no se validaron con una nueva muestra después de elegir cada efecto. No tienen la garantía CP de las alternativas prefijadas ±δ.

El efecto δ es 0,00025772842664981106. La MDE basada en cuantiles se conserva como diagnóstico descriptivo del generador simulado, sin una garantía posterior a su selección. Calibrar este generador empírico no valida automáticamente la inferencia sobre el mercado real. Tampoco estima las varianzas de los pares que no se observaron.

Las probabilidades obtenidas son sensibilidades condicionales de planificación sobre datos de desarrollo. No demuestran potencia final ni superioridad del candidato 11. **`final_R` permanece sin fijar**; R=30 conserva su papel de techo de recursos hasta resolver las recetas seleccionadas y el procedimiento temporal final.

## Evidencia y reproducción

La auditoría de ejecución verificó **1.170 diarios y recibos, 156 cadenas de eventos y 11.492 observaciones de hashes de fuentes**, sin discrepancias ni colas de eventos incompletas. Los originales permanecen en el entorno de ejecución. La publicación ofrece sus contadores y hashes junto con los resultados científicos mediante campos admitidos explícitamente, sin rutas internas, diarios completos ni pesos.

- [Resumen científico](../_static/compute_public_20261109/summary.json) y [manifiesto de los JSON públicos](../_static/compute_public_20261109/manifest.json).
- [Registro científico y decisiones previas](../_static/compute_public_20261109/registration_scientific.json).
- [Perfiles por intento, tiempos, pasos, fallo y microperfil](../_static/compute_public_20261109/profiles.json).
- [Treinta ajustes, épocas y predicciones del nuevo piloto](../_static/compute_public_20261109/variance.json).
- [Calibración matemática de los grupos separados](../_static/compute_public_20261109/calibration.json).
- [Dieciséis escenarios de presupuesto y supuestos](../_static/compute_public_20261109/budget.json).

El manifiesto fija los hashes de los archivos científicos públicos para verificar sus bytes. Los recibos privados conservan los hashes y la comprobación de carga segura de los treinta checkpoints; esa comprobación no requiere publicar los tensores. Los resultados científicos se pueden examinar en los resúmenes públicos; la auditoría de los diarios originales requiere el entorno de ejecución.

La exportación pública pasó sus siete pruebas. El paquete de auditoría interno pasó dieciocho casos, incluidos controles sobre los resultados reales; son verificaciones separadas de la publicación de los JSON científicos.

La reproducción usa Python 3.12 y el lock CPU de §29, desde una versión del repositorio que incluya la enmienda de serialización. El helper de instantáneas prepara los bytes exactos del protocolo usados por §30, comprobando hashes y conservando el respaldo del archivo histórico. Los directorios de salida deben ser nuevos; no se deben eliminar fallos para repetir un comando.

```bash
python -m pip install -r requirements-e2-training-cpu.lock.txt
python -m pip check
python scripts/prepare_e2_pilot_snapshot.py --root .
python scripts/fetch_e2_itransformer_sources.py --output work/author_sources/iTransformer

python scripts/run_e2_cpu_scaling.py --phase kernels \
  --data-root . \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --registration-commit d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2 \
  --parity-evidence experiments/e2_compute_optimization_20261109/cpu_conv_parity.json \
  --output work/kernels_reproduction

python scripts/run_e2_cpu_scaling.py --phase threads \
  --data-root . \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --registration-commit d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2 \
  --kernels-summary work/kernels_reproduction/summary.json \
  --output work/threads_reproduction

python scripts/run_e2_cpu_scaling.py --phase scaling \
  --data-root . \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --registration-commit d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2 \
  --kernels-summary work/kernels_reproduction/summary.json \
  --threads-summary work/threads_reproduction/summary.json \
  --source-root-itransformer work/author_sources/iTransformer \
  --output work/scaling_reproduction

python scripts/run_e2_optimized_variance.py \
  --data-root . \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --expected-registration-sha256 3b54ac44f5aed956ac98fbc31c95dfcb70c1116b6cca65a8269ff11cd843ac91 \
  --registration-commit d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2 \
  --kernels-summary work/kernels_reproduction/summary.json \
  --threads-summary work/threads_reproduction/summary.json \
  --output work/variance_reproduction

E2_VARIANCE_SHA256=$(sha256sum work/variance_reproduction/summary.json | cut -d ' ' -f 1)
python scripts/run_e2_calibration_report.py \
  --old-variance-gzip book_e2/_static/development_pilot_20261109/variance_full.json.gz \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --new-summary work/variance_reproduction/summary.json \
  --expected-new-summary-sha256 "$E2_VARIANCE_SHA256" \
  --output work/calibration_reproduction
```

Los comandos de perfilado y variación ejecutan entrenamiento real en el entorno donde se lancen. La calibración simula únicamente después de validar fechas, semillas, procedencia y replays; no ajusta modelos. Se deben completar los perfiles secuenciales antes de ejecutar simulaciones o trabajos que compitan por CPU.

Para reproducir la aritmética del presupuesto, las once cohortes se toman del JSON científico público y el inventario de parámetros se reconstruye desde el código público mediante dispositivos `meta`, sin crear pesos numéricos ni entrenar:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.e2_cpu_budget import meta_parameter_inventory

source = Path("book_e2/_static/compute_public_20261109/budget.json")
report = json.loads(source.read_text())
protocol = json.loads(Path("book_e2/_static/search_protocol_20261109/protocol.json").read_text())
output = Path("work/budget_inputs_reproduction")
output.mkdir(parents=True)
inputs = {"panels.json": {"origins": report["origins"]},
          "parameter_inventory.json": meta_parameter_inventory(protocol)}
for name, value in inputs.items():
    with (output / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
PY

python scripts/run_e2_budget_scenarios.py \
  --scaling-summary work/scaling_reproduction/summary.json \
  --panels work/budget_inputs_reproduction/panels.json \
  --parameter-inventory work/budget_inputs_reproduction/parameter_inventory.json \
  --registration experiments/e2_compute_optimization_20261109/registration.json \
  --output work/budget_reproduction
```

Esta última reproducción recalcula escenarios con los perfiles generados por la nueva ejecución, las cohortes publicadas y el inventario meta reconstruido. No publica ni reproduce por sí sola los diarios privados de la ejecución original, ni acredita una reconstrucción independiente de la adquisición y derivación de las cohortes. La construcción editorial consume los resultados guardados y no lanza ajustes ni simulaciones. El benchmark completo, su reproducción independiente, las recetas ganadoras y el R final permanecen pendientes.
