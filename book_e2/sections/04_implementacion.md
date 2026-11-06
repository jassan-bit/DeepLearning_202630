# 4. Implementación de Modelos Benchmark

**Estado:** diseño propuesto del benchmark **Bitcoin D7** y auditoría documental ejecutable; sus diez adaptadores, HPO y piloto de dimensionamiento **no se han ejecutado**. La [sección 5](05_paper_guia.md) sí presenta una réplica TimeMixer y un piloto externo de tres modelos en ETTh1, que no sustituyen ese benchmark. Este capítulo desarrolla las cinco capturas recibidas de §4.1–4.6: páginas 12–15 y un fragmento final sin folio visible. Distingue requisitos de la guía, decisiones propuestas y evidencias efectivamente disponibles.

La implementación debe seguir también §7 y §11. Las remisiones a potencia (§9.2), auditoría formal de fuga (§10), agregación (§13.2) y búsqueda de arquitectura (§14) no se sustituyen por supuestos sobre páginas todavía no recibidas. El [registro de requisitos](../_static/guia_e2_recibida.json) conserva ese alcance. La propuesta y su auditoría pueden descargarse en §4.7.

## 4.1 Diseño experimental

### 4.1.1 Variables, tarea y modelos

Se conserva el [contrato científico de E1](../_static/e1_reference.json): regresión temporal de BTCUSDT, siete días de cinco canales OHLCV conocidos en el origen, objetivo σ₇ poblacional de siete retornos diarios, horizonte de un día y sin anualización. No se añaden rezagos explícitos de Y, calendario, variables estáticas ni covariables futuras. Las transformaciones de precios y volumen y la inversión de escala de Y son las de §1.

| Papel | Configuraciones | Alcance de comparación |
|---|---|---|
| Diez candidatos nuevos | TSMixer, TiDE, TimeMixer, TimesNet, PatchTST, Informer, Autoformer, iTransformer, Crossformer, S-Mamba | Cuatro familias; HPO común propuesto y adaptaciones D7 por validar |
| Tres referencias heredadas | GRU/D7-c03, LSTM/D7-c02, TCN/D7-c02 | Se conservan sus recetas y la identidad del baseline GRU-c03 |
| Referencia no profunda | Ridge | Ajuste final pendiente; el diagnóstico Ridge del EDA no cuenta como benchmark ejecutado |
| Referencia trivial | Persistencia σ₇(t−1) | Sin hiperparámetros aprendidos; necesita ocho cierres pasados, como declara E1 |

La guía §3.2 exige **mínimo quince configuraciones**, no quince semillas. La igualdad de información de las redes no convierte la representación de persistencia en D7×5.

**Herencia y equidad del HPO:** las tres referencias E1 tienen recetas fijas. Optimizar los nuevos modelos ahora no permite afirmar que las quince configuraciones recibieron el mismo HPO. La igualdad propuesta de intentos y presupuesto se refiere al Top 10; debe resolverse con §11.1 cómo comparar también las bases ajustadas en E2. Si se incorporan GRU-E2, LSTM-E2 y TCN-E2 con el mismo HPO, serán **tres variantes adicionales**, sin reemplazar las referencias heredadas: mínimo dieciocho configuraciones. Esa ampliación no está ejecutada ni congelada. Ridge requiere su espacio propio; persistencia no recibe una búsqueda ficticia. Mientras esto no se cierre, no se atribuirá una ventaja frente a las referencias exclusivamente a la arquitectura.

### 4.1.2 Partición cronológica y límite del test conocido

| Periodo UTC heredado | Uso en la propuesta E2 |
|---|---|
| 01/01–31/12/2023 | Historia inicial de ajuste y transformaciones causales |
| Enero–octubre de 2024 | Diez bloques mensuales externos de forward chaining; HPO interno anterior a cada bloque |
| 01/11–30/12/2024 | Piloto E2 con recetas previamente congeladas; pesos ajustados antes de su inicio |
| Desarrollo 2023–2024 | Ajuste final nominal al 01/01/2025; etiquetas de entrenamiento hasta el 24/12/2024 y embargo de siete días |
| 01/09–30/10/2026 | Evaluación heredada de sesenta días, **ya observada en E1**; nuevas comparaciones exploratorias |
| Enero de 2025–agosto de 2026 | Archivo histórico conocido de E1; no se reasigna a entrenamiento ni a un test desconocido |

La guía pide reservar el test y tocarlo una vez al final. **La parte histórica de ese requisito no se puede recuperar:** publicar hashes o escribir un protocolo ahora no elimina el conocimiento anterior. Sí puede registrarse una única fase de scoring del benchmark E2, después de cerrar configuraciones y pesos, seguida del análisis registrado sobre sus predicciones guardadas. Esa fase deberá incluir de antemano las condiciones limpia, perturbadas y de ablación; sus recetas y checkpoints se cerrarán antes de consultar sus resultados. No se reentrena o elige una receta por su resultado en esa fase. Su interpretación seguirá siendo exploratoria; aquí no se adquiere ni asigna un nuevo periodo confirmatorio.

La separación se verifica por soportes temporales de entradas y etiquetas, no solo por nombres de archivos. Cada transformación se ajusta dentro del entrenamiento permitido de su fold. Los índices diarios forward chaining heredados proceden de `timeseries-cv`, proyecto de DidierRLopes con Filipe R. Ramos; agruparlos en meses no convierte la librería en una API nativa de folds mensuales. Los índices de la nueva anidación todavía deben generarse y auditarse.

### 4.1.3 Selección interna y validación externa

El procedimiento E1 usa una validación interna para elegir épocas y diez meses para seleccionar recetas. **Eso no es una evaluación externa anidada independiente del HPO.** Para E2 se propone una anidación temporal explícita:

1. En cada origen mensual externo O, solo se permiten etiquetas de entrenamiento hasta O−8 días: horizonte uno más embargo siete.
2. Dentro de ese pasado se forman tres bloques internos consecutivos de treinta días. El último termina en O−8; los otros dos lo preceden. Cada origen interno q vuelve a imponer su propio corte q−8.
3. Para cada candidato y semilla se seleccionan épocas con la regla interna heredada: inicio nominal q−90, final permitido q−8 y entrenamiento temprano hasta q−98. El intervalo de noventa días es nominal respecto al origen; tras el embargo quedan **83 fechas de calendario potenciales**, antes de aplicar elegibilidad. No se cuentan noventa etiquetas válidas por defecto.
4. Se reajusta desde una inicialización registrada sobre todo el entrenamiento permitido durante las épocas seleccionadas, y se evalúa el bloque interno. Escaladores internos y de reajuste se aprenden por separado.
5. El candidato se elige por RMSE en escala original, concatenando las fechas internas por trayectoria y promediando después entre las dos semillas de HPO. Se reajusta la receta seleccionada para el origen externo y se puntúa ese mes sin usarlo para su propia selección.

Los resultados externos evalúan el **procedimiento de selección**, cuyas recetas pueden variar entre meses. No se atribuyen automáticamente a una única receta final. Esta última se escogerá mediante una búsqueda forward adicional en enero–octubre, se congelará antes del piloto y se distinguirá de la evaluación anidada. Ese HPO final es selección en desarrollo, no una evaluación independiente.

### 4.1.4 Espacio y presupuesto propuestos

Se propone una cuadrícula de ocho candidatos por familia nueva: learning rate {0,001; 0,003} × dropout {0; 0,1} × dimensión latente {16; 32}. Es un punto de partida pequeño, orientado a D7 y comparable en número de intentos; **no reproduce las configuraciones originales de horizontes largos**. Profundidad, cabezas, parches, escalas, kernels y estados se especificarán por adaptador, justificando las elecciones con §3. Cada combinación debe pasar restricciones de forma antes del cierre; no se corrige silenciosamente un candidato inválido durante la búsqueda.

| Control | Propuesta; todavía no congelada |
|---|---|
| Optimización | AdamW, weight decay 0,0001; pérdida cuadrática heredada |
| Batch, épocas, parada | Batch 32; máximo 100 épocas; paciencia 12; épocas seleccionadas solo en validación permitida |
| Estabilidad | Gradient clipping de norma 1; FP32; determinismo explícito, sin fallback silencioso |
| Dispositivo | CPU con un thread como propuesta inicial comparable; compatibilidad y perfilado pendientes |
| Intentos | Ocho candidatos registrados por familia y las mismas dos semillas HPO |
| Presupuesto de tiempo y memoria | **Pendiente de perfilado en entrenamiento**; aún no existe un límite común B validado |

Mismo presupuesto significa la misma **asignación y límites**, además del mismo número de intentos; no que las arquitecturas consuman exactamente los mismos segundos. Se contará selección de épocas, reajuste, folds, compilación y fallos. Los timeouts no se reponen solo a una familia ni se conceden intentos extras a la más rápida. Si una familia no completa los intentos previstos dentro del presupuesto, se informa como búsqueda incompleta; no se afirma igualdad de optimización realizada.

El coste previsto se expresa en dos unidades para no ocultar reajustes. Con dos ajustes de optimizador por evaluación —selección de épocas y reajuste—:

| Etapa por familia nueva | Evaluaciones entrenadas | Ajustes de optimizador |
|---|---:|---:|
| HPO interno: 8 × 2 semillas × 3 bloques × 10 orígenes | 480 | 960 |
| Recetas seleccionadas de los diez meses: 2 semillas × 10 | 20 | 40 |
| HPO para receta final: 8 × 2 semillas × 10 bloques | 160 | 320 |
| Total previo al piloto | **660** | **1.320** |

Para el Top 10 son **13.200 ajustes propuestos**, no ocho por modelo. La cuenta no incluye perfilado, NAS condicional, Ridge, referencias adicionales, piloto, R ejecuciones definitivas, robustez o ablaciones. El piloto y el ajuste definitivo añaden dos ajustes por semilla y modelo bajo la misma regla. El cómputo total debe estimarse antes de cerrar una configuración ejecutable; estas cuentas no son trabajo realizado ni horas medidas.

**Arquitectura no predefinida:** §4 remite obligatoriamente a §14. El fragmento recibido describe tres capas, cinco anchos posibles y dropout 0,2 en todas las posiciones: 5³ × 2³ = **1.000 arquitecturas en fase 1** por modelo afectado. No se activa esa regla solo por un nombre de familia, ni se exime a todos los modelos porque tengan un DOI. Se debe localizar la arquitectura original, fijar la adaptación D7 y decidir el caso con la sección completa, incluida la segunda fase y su presupuesto. La cuadrícula de ocho hiperparámetros no sustituye un NAS que resulte obligatorio.

## 4.2 Implementación de modelos

### 4.2.1 Pipeline modular y contratos verificables

La interfaz prevista separa adquisición/elegibilidad, índices, transformaciones, backbone, lectura escalar, ajuste y scoring. El futuro adaptador recibirá un tensor `[batch, 7, 5]` y devolverá una predicción escalar por ventana. La salida debe corresponder a σ₇ y a su escala de entrenamiento; no se selecciona Close de una salida multivariante ni se desnormaliza como precio o volumen.

| Módulo previsto | Comprobación anterior al entrenamiento |
|---|---|
| Índices y elegibilidad | Calendario y embargo heredados; soportes reales de entrada/objetivo por fold; ningún cruce con fechas de evaluación |
| Transformaciones | Logaritmos y ancla causal heredados; scaler y media de Y ajustados únicamente en train; sin imputación principal ni compresión de huecos |
| Adaptadores | Forma 7×5, valor finito/no negativo tras invertir escala, gradientes y checkpoint recuperable; campos futuros ausentes |
| Ajuste y reajuste | Inicializaciones, épocas y transformaciones de cada etapa registradas; validación sin gradientes |
| Scoring | Solo checkpoints cerrados y cohorte común; persistencia con su soporte causal declarado; métricas en unidades originales |

Los controles particulares de §3 se convierten en pruebas de cada adaptador: TiDE covariate-only y sus skips; tendencia de Autoformer en unidades del objetivo; decoders sin verdad futura; parches/escalas válidos con siete días; exclusión de frecuencia cero y selección de periodos por origen en TimesNet; tokens de variables observadas en iTransformer/S-Mamba. Para TimesNet se propone inferencia batch 1 para impedir que distintos orígenes compartan la selección de periodos. Estas pruebas **todavía no están implementadas**; la auditoría documental no las reemplaza.

### 4.2.2 Dependencias, hardware y trazas

Las fuentes de autores están fijadas por commit en §3. El lock editorial del libro y el lock histórico de entrenamiento E1 **no constituyen un entorno E2 validado para los diez backbones**. Se debe resolver y fijar Python, framework, bibliotecas de modelos, kernel Mamba si aplica y dispositivo; registrar versiones, código y hash de cada adaptador propio. La disponibilidad de un repositorio no prueba compatibilidad ni permisos de redistribución. No se ha documentado un entrenamiento GPU de E2.

Si S-Mamba requiere kernels CUDA, no se le concede otro hardware sin justificar la comparación. Debe verificarse una implementación compatible y determinista en el dispositivo común, o cerrar un cambio de entorno para todos los modelos afectados. No se cambia precisión o determinismo automáticamente para superar una prueba de forma.

Se propone tracking local mediante configuraciones JSON versionadas, eventos JSONL y tablas CSV por corrida; no requiere un servicio externo. Cada registro tendrá modelo, fase, candidato, semilla, fold, etapa, datos/índices/código por hash, parámetros, optimizer/scheduler —incluido «sin scheduler»—, batch, dropout, weight decay, pérdida, épocas, clipping, precisión y dispositivo. Los estados completado, fallido, timeout y no estimable se conservarán junto al error y los recursos consumidos.

| Evidencia obligatoria por ajuste | Regla de registro propuesta |
|---|---|
| Curvas train/validación | Pérdida por época, mejor época y checkpoint; separar selección y reajuste |
| Entrenamiento e inferencia | Tiempo total y por etapa; inferencia batch 1 por muestra y throughput separado, con calentamiento, sincronización y orden intercalado |
| Memoria, parámetros y FLOPs | Pico CPU/GPU, parámetros registrados/activos y método exacto de conteo; distinguir FLOPs de MACs y operaciones no soportadas |
| Hardware y energía | CPU/GPU, RAM, threads, precisión y versiones; energía solo si hay medición o estimación trazable, sin convertir TDP por tiempo en una medición real |

Se usarán datos de desarrollo o tensores sintéticos para perfilado y latencia; no hace falta puntuar el test para medir coste. Un valor no medido se reportará como tal, nunca como cero. Robustez, explicaciones y ablaciones tienen presupuestos adicionales identificados.

## 4.3 Requisitos experimentales y piloto

Se registrarán las semillas de Python, NumPy y el framework y las opciones de determinismo, orden de datos y threads. Dos semillas con el mismo entero en arquitecturas distintas no producen pesos idénticos; el emparejamiento conserva la realización del procedimiento, las fechas y la configuración de aleatoriedad. La clave de cada corrida permitirá reanudar solo artefactos con hashes coincidentes, sin reemplazar ejecuciones desfavorables.

La guía exige al menos cinco ejecuciones independientes o particiones CV justificadas. **Cinco folds temporales solapados no son cinco series de mercado independientes.** Para contrastes Wilcoxon bilaterales exactos, con cinco pares no nulos y sin empates el mínimo p es 0,0625; seis permiten 0,03125, sin que ello demuestre potencia suficiente. R debe dimensionarse con §9.2; el R=465 de E1 no se transfiere.

Se propone un piloto de dimensionamiento Bitcoin D7 de diez semillas por modelo con recetas previamente congeladas y las fechas heredadas 01/11–30/12/2024. Los pesos se ajustarán antes del 01/11 y se conservarán durante esas sesenta fechas. El piloto estima dispersión del **procedimiento E2** antes del ajuste definitivo; no precede al benchmark histórico E1 ni crea nuevas fechas desconocidas. Las sesenta fechas continúan temporalmente dependientes. Solo después se fijarán R, potencia objetivo, multiplicidad, estimador de variabilidad y sus supuestos. Diez semillas no acreditan por sí solas potencia temporal o conjunta. El piloto externo de tres modelos y cinco semillas de §5 ya se ejecutó en ETTh1 para estudiar limitaciones; no estima la dispersión del futuro procedimiento Bitcoin D7 ni fija su R.

La cronología prevista separa: cierre de adaptadores/entorno/presupuesto → HPO en desarrollo → recetas congeladas → piloto → dimensionamiento → ajustes definitivos → cierre de pesos/escaladores → scoring único E2 → análisis registrado. Un cambio posterior se documentará como enmienda con fecha y motivo; no se le atribuye carácter previo a resultados ya conocidos.

## 4.4 Prevención de fuga y auditoría obligatoria

La auditoría formal de §10 debe ejecutarse sobre **cada pipeline real**, no solo sobre la tabla de fechas. El EDA ya aporta una auditoría de soportes de los folds heredados; no certifica los nuevos índices internos ni adaptadores.

| Riesgo específico | Evidencia que se exigirá antes de cerrar la ejecución |
|---|---|
| Solapamiento temporal indebido | Manifiestos de soportes de etiquetas/entradas; cortes q−8 y comprobación de todos los folds internos/externos |
| Scaler aprendido fuera de train | Índices y hashes de su ajuste en selección/reajuste; prueba de invariancia al modificar datos posteriores |
| Información futura en decoder o canales | Firma de entrada y prueba de que una modificación del futuro no cambia la predicción del origen |
| Contaminación entre muestras | Inferencia por origen o prueba de independencia del batch, especialmente selección de periodos TimesNet |
| Selección con test | Configuraciones, cohortes, presupuestos y pesos cerrados antes del scoring; sin consultas de métricas durante HPO |
| Duplicados o índices desalineados | Unicidad de timestamp/semilla/modelo, calendario sin compresión y misma cohorte para las comparaciones previstas |
| Adaptación silenciosa de contratos | Comparación exacta con E1 de objetivo, margen, unidad, partición y recetas protegidas |

Las comprobaciones espaciales y de duplicados semánticos de imágenes/texto se declararán no aplicables al dataset OHLCV, sin usarlas como evidencia de haber auditado el riesgo temporal. Una auditoría sin incidencias detectadas no acredita ausencia universal de fuga; deben publicarse alcance, resultados e incidencias. El conocimiento previo del test sigue siendo una limitación aunque todos los nuevos controles pasen.

## 4.5 Evaluación y análisis de errores

**Estado:** diseño propuesto a partir de las páginas recibidas; no hay métricas del benchmark Bitcoin D7. Las métricas externas de réplica y piloto ETTh1 están en §5. Las pruebas específicas de §7.3, la agregación de §13.2 y el dimensionamiento de §9.2 se incorporarán al recibir esos apartados. No se atribuye al pasado una decisión tomada ahora ni se presenta el test conocido de E1 como prueba sellada nueva.

### 4.5.1 Métricas, estimando y cohorte

Se conserva la primaria heredada: para cada semilla se calcula RMSE en todas las fechas elegibles comunes y después se promedian las RMSE entre semillas. No se sustituye por RMSE de un ensamble, raíz de los errores mezclados entre semillas o media no ponderada de RMSE mensuales.

$$
\widehat\mu_m=\frac{1}{R}\sum_{s=1}^{R}\sqrt{\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2}.
$$

Se reportarán MAE, MSE, R², MAPE y MASE en escala original. MAPE usa únicamente objetivos positivos y publica numerador de casos definidos, denominador de elegibles y cobertura; no se añade épsilon para ocultar ceros. R² se marca no definido si la variación del objetivo es cero. MASE divide MAE por la escala ingenua de lag 1 ajustada exclusivamente en pares de días consecutivos elegibles del entrenamiento correspondiente; no se comprime el calendario, no se usa test y no se recalcula la escala en cada subconjunto de evaluación. Si esa escala es cero, MASE queda no definida y se publica su cobertura.

El margen absoluto sigue siendo **δ = 0,00025772842664981106**. Su origen es el 5 % de una RMSE CV histórica específica; no equivale al 5 % de cualquier nueva RMSE. Se conserva aparte la reducción relativa académica frente a GRU-c03 y persistencia. Una mejora puntual del 5 % no acredita por sí sola una mejora práctica con incertidumbre controlada.

Las tablas incluirán modelo/configuración, fase, semilla, fechas, número elegible/definido, métricas, coste y estado de ejecución. Se publicarán resultados por semilla y fold además de media, DE muestral e IC95 %, con un agregado que concatene predicciones de las mismas fechas. No se descartan semillas desfavorables ni se completan métricas ausentes con cero.

### 4.5.2 Dos fuentes distintas de incertidumbre

| Fuente | Qué estima y cómo se tratará | Qué no acredita |
|---|---|---|
| Semillas de entrenamiento | Variabilidad condicionada a los datos fijos; trayectorias emparejadas entre modelos, media/DE/IC95 % sobre las ejecuciones previstas | Nuevas series independientes, incertidumbre del mercado o potencia temporal por aumentar R |
| Calendario | Dependencia de ventanas y etiquetas solapadas; bootstrap temporal pareado, remuestreando las mismas fechas/bloques para ambos modelos y recalculando la RMSE de cada trayectoria | Independencia iid de días o validez exacta bajo cambios de régimen |
| Perturbaciones sintéticas | Monte Carlo de corrupción condicionado a pesos/datos; idénticas realizaciones entre modelos | Nuevos ajustes independientes ni aumento del número de semillas R |

El número de ejecuciones se justificará con el piloto de E2 y §9.2; el mínimo de cinco es una cota de la guía, no prueba de potencia suficiente. Con cinco pares no nulos y sin empates, el p-valor bilateral mínimo del Wilcoxon exacto es 0,0625; no se promete significación a α=0,05 mediante cinco semillas ni se selecciona una prueba favorable tras inspeccionar resultados.

Como propuesta para la incertidumbre temporal, se conservará un bloque principal de siete fechas y sensibilidades de catorce y veintiocho, con 4.999 remuestreos y semilla 20261105, coherentes con el solapamiento de σ₇ y la referencia de E1. **Son parámetros por ratificar con el procedimiento completo de E2**, sin elegir luego la longitud que produzca el menor p-valor. La propuesta posterior de §6 eleva el número de remuestreos a 49.999 para resolver mejor las colas de una familia de treinta contrastes; conserva el carácter de borrador previo al benchmark y no cambia análisis históricos. El tamaño y número efectivo de bloques, las fechas y las hipótesis de estacionariedad deben figurar junto al intervalo. Si no hay soporte temporal suficiente, el resultado se declara inconcluso.

Se calcularán ganancias absolutas y relativas respecto a los comparadores registrados, intervalos y p-valores ajustados y sin ajustar. La familia principal y el ajuste por multiplicidad se fijarán antes del scoring E2; no se heredan automáticamente las tres comparaciones de E1 ni se convierten todos los 105 pares de quince configuraciones en hipótesis principales. La inferencia de mejora práctica contrasta la frontera δ, no solo la diferencia cero. No rechazar no equivale a demostrar equivalencia.

La descomposición de varianza entre modelos, semillas y calendario queda como producto pendiente de §13.2. No se modelan las 60 fechas de la reserva conocida o las semillas como unidades iid intercambiables para obtener artificialmente grados de libertad. La [sección 6](06_benchmark_final.md) desarrolla la tabla, los rangos, la eficiencia y el plan de inferencia a partir de las nuevas capturas; mantiene el estado pendiente de las ejecuciones Bitcoin.

### 4.5.3 Sobreajuste, casos difíciles y estratos

Las curvas de entrenamiento/validación registrarán pérdida, épocas y checkpoint seleccionado. Las brechas se calcularán con las mismas métricas, unidades y regla de predicción en train y validación; el train score se reportará fuera del modo entrenamiento. Las pruebas formales serán las de §7.3 cuando se reciba esa sección; una brecha descriptiva no se etiqueta como diagnóstico estadístico de sobreajuste o subajuste.

Se congelan estratos con información disponible en el origen. Se conserva la propuesta de §1: alta volatilidad cuando σ₇(t−1) supera el cuantil 0,75 de entrenamiento 2023, **0,024247163097617024**, sin introducirla como sexto canal de entrada. Se reportan por estrato fechas, bloques útiles, métricas e incertidumbre; un subconjunto pequeño no recibe automáticamente un IC fiable. El análisis de la reserva ya conocida describe desplazamiento retrospectivo; no selecciona umbrales o arquitecturas.

El reporte de errores combinará: trayectoria de predicción/objetivo en fechas seleccionadas por una regla fijada en desarrollo; máximos errores absolutos de cada modelo con la misma regla; errores en aumentos/disminuciones de volatilidad; sesgo firmado y distribución de residuos por régimen. Los casos extremos se etiquetan como ejemplos seleccionados por error, sin presentarlos como muestra representativa. La taxonomía permitirá distinguir retraso de respuesta, sobrepredicción, infrapredicción y fallos de datos, con recuentos y ejemplos adversos. Los motivos no observables se mantendrán como hipótesis.

## 4.6 Robustez, interpretabilidad y ablación

**Todo lo siguiente es un plan pendiente de implementación y ejecución.** Niveles, máscaras, transformaciones, semillas, modelos elegidos y familias de contrastes se registrarán antes de sus resultados. El diseño se comprobará primero en desarrollo; no se ajustarán severidades para que favorezcan a un modelo en test.

### 4.6.1 Perturbaciones pertinentes para OHLCV

Se evaluarán los pesos entrenados con datos limpios y congelados; esta prueba mide sensibilidad de inferencia, no robust training. La cohorte elegible y las etiquetas originales permanecen iguales para nivel cero y niveles perturbados. No se reordena el calendario, no se cambia Y y no se añaden rezagos de Y, indicadores nuevos o covariables futuras.

| Perturbación | Tres niveles propuestos, además de cero | Aplicación causal y controles |
|---|---|---|
| Ruido de medición en OHLC | α = 0,05; 0,10; 0,20 de la DE de retornos logarítmicos estimada solo en train | Para cada barra pasada, multiplicar O,H,L,C por el mismo factor positivo exp(ασᵣξₜ). Se preserva L≤min(O,C)≤max(O,C)≤H y positividad. Recalcular las transformaciones heredadas, incluido el ancla del último cierre, desde la entrada perturbada. Volumen y objetivo no se modifican |
| Ausencia del canal volumen | Tasas p = 0,10; 0,20; 0,30 | Máscaras sintéticas por fecha con umbrales anidados sobre uniformes comunes. Sustituir únicamente V ausente por la mediana de log1p(V) aprendida en train y transformada de vuelta. Se mantienen OHLC, fechas y cinco canales. Publicar la tasa realmente obtenida y ventanas afectadas |

Las innovaciones y uniformes se generarán por timestamp y realización, de forma común a modelos y semillas, para que una barra compartida por ventanas solapadas reciba la misma corrupción. Se proponen cinco realizaciones de perturbación, semillas 30101–30105, sin contarlas como cinco entrenamientos. Se reutilizan innovaciones base entre niveles para obtener comparaciones anidadas. La transformación principal y el imputador de diagnóstico se ajustan sin consultar validación o test; su estado se versionará.

La ausencia sintética de V es una **rama diagnóstica separada**. El benchmark principal conserva su regla de exclusión de días inválidos; esta prueba no autoriza imputar silenciosamente huecos originales ni interpolar con observaciones futuras. La imputación no añade canales de máscara, ni historial fuera de D7. No se simulan precios negativos, velas desordenadas o una reordenación arbitraria de fechas como si fueran perturbaciones financieras plausibles.

Persistencia usa ocho cierres pasados para construir σ₇(t−1). Si se incluye en la prueba de ruido OHLC, se aplica la misma realización a los cierres de su soporte pasado permitido y se recalcula la referencia, sin concederle datos futuros. Se documenta que su representación de entrada difiere de las siete barras de las redes. La ausencia de V no modifica persistencia; eso es una propiedad de la referencia y no una ventaja que deba ocultarse.

Para cada nivel se reportará RMSE absoluta y normalizada por el nivel cero de cada modelo, con las métricas secundarias y el coste de inferencia. Se propone

$$
D_m(u)=\widehat\mu_m(u)/\widehat\mu_m(0),\qquad
A_m=\int_0^1D_m(u)\,du,
$$

con severidad u normalizada por el máximo de cada perturbación y aproximación trapezoidal sobre los cuatro puntos registrados. Menor área indica menor degradación relativa; A=1 representa ausencia de cambio y A−1 puede ser negativo si el ruido mejora el error en esa muestra. No se recorta esa mejora ni se asume monotonicidad. Si la RMSE de nivel cero es cero, la normalización se declara no definida y se publica la curva absoluta, sin épsilon. Se acompañará siempre de RMSE absoluta: un modelo impreciso puede parecer estable en términos relativos. Las unidades de severidad entre ruido y ausencia no tienen la misma interpretación física; no se funden en un único ranking sin una ponderación justificada previamente.

La incertidumbre usará comparaciones pareadas con calendario y corrupción comunes; R×5 realizaciones no son R×5 series independientes. Se mostrará la curva completa y las sensibilidades de bloques. No rechazar un aumento del error no demuestra robustez o equivalencia. Los estratos de volatilidad de §4.5.3 aportan cambio de distribución temporal, con su límite retrospectivo; no acreditan generalización a otra criptomoneda o proveedor.

### 4.6.2 Interpretabilidad de tres modelos y pruebas de explicación

Los tres modelos profundos a explicar se seleccionarán por la RMSE media de validación forward chaining, con desempate por coste de inferencia y luego identificador de configuración. La regla se congelará antes del scoring; no se elegirán después tres modelos que produzcan las explicaciones más convincentes. Se distinguirá esta selección de cualquier orden observado en el test conocido. Para explicar orígenes de validación se usarán sus checkpoints forward, entrenados solo con el pasado permitido; no se presentarán como fuera de muestra explicaciones de un ajuste final que ya incorporó esas etiquetas.

Se propone Integrated Gradients sobre la predicción escalar σ₇ y las 35 entradas D7×5, usando un tensor de referencia derivado de medianas de entrenamiento y manteniendo las transformaciones congeladas. Se registrarán implementación, número inicial de pasos —64 propuestos—, normalización y error de completitud. Un control numérico a 128 pasos solo responderá a un criterio de convergencia fijado previamente, sin seleccionar el resultado más atractivo. Las atenciones no se interpretarán automáticamente como importancia causal.

La elección de IG tiene una condición técnica: TimesNet puede seleccionar periodos mediante top-k discreto. Los gradientes de una rama no acreditan por sí solos la explicación del cambio de periodo. Se registrará antes de la evaluación una alternativa de oclusión por grupos para modelos con operaciones no diferenciables, y se reportará el alcance del método en vez de atribuirle una fidelidad completa no probada.

La fidelidad se medirá con borrado e inserción progresiva de **bloques de un día completos**, cinco canales juntos, reemplazados por una referencia de train coherente; se evita borrar solo H o L y generar una vela inválida. Se ordenan los siete días por atribución absoluta agregada y se comparan la respuesta de la predicción y el error con órdenes aleatorios comunes y orden inverso. Se publicarán las siete fracciones, áreas y controles. El cambio de pérdida y el cambio de predicción son medidas distintas: borrar una característica influyente no tiene por qué empeorar el error si la predicción inicial era mala.

La estabilidad entre semillas se reportará mediante correlación de rangos y solapamiento top-k de los mismos grupos de días, junto a dispersión de atribuciones normalizadas cuando su norma sea positiva. Correlaciones o normalizaciones indefinidas se marcarán como tales con su cobertura. Las muestras explicadas se fijarán por calendario y régimen de desarrollo, con todas las semillas previstas y una regla de selección reproducible. Una explicación inestable, infiel o limitada por discontinuidades se reporta como resultado adverso; plausibilidad visual y acuerdo con intuiciones financieras no sustituyen fidelidad ni demuestran causalidad económica.

### 4.6.3 Ablación de al menos tres componentes

El modelo profundo a ablacionar se escogerá exclusivamente en desarrollo con la misma regla de desempeño/coste anterior, antes de sus resultados de test. Se seleccionarán tres componentes reales de su implementación congelada y se retirará o sustituirá **uno a la vez**, conservando el resto y la cabeza de σ₇. La biblioteca de ablaciones se cerrará al validar los adaptadores; no se elegirá después qué cambios producen diferencias significativas.

| Modelo que resulte elegido | Ejemplos de componentes que deben mapearse al código exacto antes de ejecutar |
|---|---|
| TSMixer | Mezcla temporal, mezcla de variables, conexiones residuales |
| TiDE | Bloques residuales del encoder, skip global, decoder de características; revisar proyección al objetivo adaptado |
| TimeMixer | Descomposición estacional/tendencia, mezcla estacional ascendente, mezcla de tendencia descendente |
| TimesNet | Agregación de varios periodos frente a uno, convolución intra/interperiodo, ponderación de amplitudes frente a fusión uniforme |
| PatchTST | Parches frente a tokens de un paso, codificación posicional, normalización reversible |
| Informer | Selección ProbSparse frente a atención completa, distilling, codificación posicional interna |
| Autoformer | Agregación de retardos, rama de tendencia, un bloque de procesamiento; auditar dependencias entre mecanismos al retirarlos |
| iTransformer | Atención entre variables, bloque FFN, normalización interna |
| Crossformer | Atención temporal, atención dimensional con routers, jerarquía de segmentos |
| S-Mamba | Rama inversa de variables, selectividad de entrada frente a parámetros no dependientes de ella, bloque FFN |

Esta tabla es una lista de candidatos, no recetas implementadas. Si la mejor red es una base heredada, se definirán los tres componentes de esa arquitectura antes de su evaluación, sin fingir que posee un mecanismo del Top 10. Si Ridge o persistencia ganan el benchmark completo, se conservará ese resultado: la ablación arquitectónica de la mejor red profunda no la convierte en el mejor modelo global. La condición de una ablación de tres componentes para una referencia trivial se declarará no aplicable y se contrastará con el requisito completo de la guía.

Cada ablación usará las mismas semillas y fechas, el mismo preprocesamiento permitido y el mismo número y presupuesto de intentos. La variante completa y las ablaciones tendrán búsquedas comparables cuando el objetivo sea medir capacidad tras ajuste; un estudio secundario sin reajuste podrá medir dependencia de la receta original, pero no se confundirá con el primero. Se registrarán parámetros, FLOPs, latencia y memoria para separar mecanismo y capacidad. Un control con capacidad aproximadamente emparejada se justificará cuando la retirada reduzca parámetros sustancialmente; si no es posible, esa confusión será una limitación explícita.

Para cada retirada se reportarán diferencia completa–ablación, IC95 %, tamaño de efecto y p-valor ajustado y sin ajustar, con inferencia temporal pareada y control de la familia de al menos tres contrastes. Las semillas de las ablaciones serán las mismas y el número R estará dimensionado con el procedimiento final, no elegido por el primer resultado. Un intervalo que incluya cero no prueba que el componente sea inútil. Las comparaciones sobre la reserva conocida conservan carácter exploratorio.



## 4.7 Productos verificables y requisitos de cierre

La herramienta `scripts/audit_e2_benchmark_readiness.py` se ejecuta con biblioteca estándar de Python. Comprueba el contrato científico, la lista de quince configuraciones, los seis hashes públicos de referencia de E1 y las cuentas/calendarios propuestos. No carga archivos de mercado ni predicciones individuales; los agregados históricos conocidos solo se usan como artefactos de procedencia. No importa librerías DL ni ajusta modelos.

- [Descargar diseño, auditor y pruebas](../_static/benchmark_design_download/Diseno_E2_propuesta_auditoria.zip) y [manifiesto de integridad](../_static/benchmark_design_download/MANIFIESTO.json).
- [Propuesta de configuración](../_static/benchmark_design/proposal.json) e [informe de preparación](../_static/benchmark_design/report.json).
- [Estado de los quince modelos](../_static/benchmark_design/model_readiness.csv) y [trazabilidad de los 26 requisitos recibidos](../_static/benchmark_design/requirements.csv).
- [Calendario anidado propuesto: treinta bloques internos](../_static/benchmark_design/nested_calendar.csv) y [hashes de procedencia](../_static/benchmark_design/provenance.json).

El calendario CSV contiene fechas, **no índices supervisados ni una auditoría de elegibilidad ejecutada**. El estado de preparación del benchmark Bitcoin D7 es `proposal_valid=true`, `ready_to_train=false`: una propuesta válida puede tener prerrequisitos incompletos. La réplica externa ejecutada de §5 no convierte ese estado en aprobación del benchmark. El auditor valida documentos y no convierte por sí mismo un archivo existente en una implementación científicamente certificada.

```bash
python scripts/audit_e2_benchmark_readiness.py
python tests/test_e2_benchmark_readiness.py
```

El modo `--require-ready` devuelve código 2 mientras existan esos pendientes; no debe interpretarse como fallo del experimento o como ausencia de resultados significativos. Un contrato inválido devuelve 1. Este auditor no invoca el ejecutor de la réplica ETTh1 ni entrena un modelo Bitcoin D7.

| Producto de §4 | Evidencia actual | Falta para cumplir su ejecución |
|---|---|---|
| Variables, herencia y roles | Contrato, catálogo y propuesta versionados | Cerrar la equidad de optimización con las referencias |
| CV/HPO y presupuesto | Fechas anidadas, cuadrícula y cuentas propuestas | Índices reales, perfilado y presupuesto común; §11.1 y §14 completos |
| Implementación de diez modelos | Originales/código identificados en §3 | Adaptadores D7, lock y pruebas de forma, causalidad y recuperación |
| Registro de experimentos y costes BTC D7 | Esquema de trazas definido; la ejecución externa ETTh1 tiene sus propios recibos en §5 | Tracker y mediciones del benchmark Bitcoin |
| Repeticiones y piloto BTC D7 | Cronología y mínimo de guía documentados; piloto ETTh1 separado en §5 | Piloto de dimensionamiento Bitcoin, potencia y R con §9.2 |
| Prevención de fuga | Auditoría EDA previa y controles previstos | Auditoría formal de cada pipeline conforme a §10 |
| Evaluación e inferencia | Estimando, métricas y reglas de reporte propuestos | Resultados; rutas §7 y agregación §13.2 completas |
| Robustez, explicaciones y ablación | Dos perturbaciones de tres niveles; selección, fidelidad, estabilidad y tres componentes previstos | Implementación y ejecución con decisiones registradas |
| Test nuevo sellado | Limitación identificada | No disponible en la partición heredada; análisis E2 exploratorio |

El diseño no se presenta como cumplimiento experimental terminado. La siguiente etapa del benchmark Bitcoin D7 requiere cerrar sus adaptadores, entorno, presupuesto, auditoría y procedimiento estadístico con las páginas restantes. Las curvas, tiempos, tablas y contrastes deberán proceder de las corridas efectivamente registradas.
