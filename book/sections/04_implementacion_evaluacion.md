# 4. Implementación y evaluación: diseño y estado de ejecución

El **4 de noviembre de 2026 se implementaron y entrenaron MLP, LSTM y CNN1D poco profunda** sobre desarrollo de BTCUSDT. La nueva autorización para ejecutar y actualizar E1 sustituye el aplazamiento anterior a E2. El experimento completó **96 corridas de búsqueda, 15 de piloto y 90 de benchmark de desarrollo: 201 corridas y 402 ajustes**. Cada corrida comprende selección interna de épocas y un reajuste nuevo sobre entrenamiento permitido de 2023.

**El test del 01/01/2025 al 31/08/2026 permanece cerrado.** La búsqueda usó enero–junio de 2024; piloto y benchmark comparten julio–diciembre de 2024. Este último periodo es desarrollo/calibración expuesto al dimensionamiento, incluso con semillas nuevas; sus resultados no constituyen un test independiente. Los perfiles LSTM/TCN/encoder de §3.3 son antecedentes documentales. La ejecución propia no resuelve por sí sola las fichas aplicadas pendientes de MLP/CNN1D en la revisión bibliográfica.

La evidencia vigente está en el [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json), la [configuración seleccionada](../_static/dl_benchmark_20261104/selected_configuration.json), el [piloto](../_static/dl_benchmark_20261104/pilot_summary.json), el [protocolo de desarrollo congelado](../_static/dl_benchmark_20261104/protocol_final.json) y el [resumen de ejecución](../_static/dl_benchmark_20261104/execution_summary.json). La {download}`revisión documental del 03/11 <../_static/e1/revision_implementacion_evaluacion_20261103.md>` y su [especificación](../_static/experiments/protocolo_e2_documental_20261103.json) conservan valor histórico; sus estados anteriores al entrenamiento no describen esta ejecución.

## 4.1. Diseño Experimental

### 4.1.1. Definición clara de:

#### 4.1.1.1. Variables de entrada: representación y dimensiones

La unidad es un **día UTC de BTCUSDT**, agregado de las velas oficiales de un minuto. Las cinco covariables son apertura, máximo, mínimo, cierre y volumen (OHLCV), sin noticias ni atributos adicionales.

La búsqueda evaluó $D\in\{7,14,21,28\}$ con tensores $X\in\mathbb R^{N\times D\times5}$ y seleccionó **D=7 común a las tres redes**. La MLP aplana los mismos $5D$ valores; LSTM y CNN1D mantienen el eje temporal. Las ventanas respetan el calendario y no atraviesan días incompletos. El día 24/03/2023 conserva su exclusión; no se imputa ni se forma un retorno que salte ese hueco.

#### 4.1.1.2. Variable objetivo y horizonte de predicción

El objetivo principal es la volatilidad poblacional de **siete retornos logarítmicos diarios**, $\sigma_7$, con `ddof=0`, sin anualizar; el horizonte es **h=1**. Las sensibilidades de objetivo $n=14,21,28$ **no se entrenaron**. Son distintas de las cuatro historias D exploradas con n=7.

Para el día de resultado $d$, se pronostica al inicio de ese día con $X_d=(z_{d-D},\ldots,z_{d-1})$, $z=(O,H,L,C,V)$, y $y_d=\sigma_7(d)$, conocida al terminar $d$. Incluye el retorno desconocido de ese día; no representa siete días enteramente futuros. Se conservan fecha, origen y disponibilidad de etiqueta.

La salida escalar utiliza Softplus para mantener positividad. Se entrenó con **MSE**, tras dividir Y por su media positiva de entrenamiento sin centrarla; todas las métricas se calcularon después de invertir esa escala. **RMSE es primaria**; MAE, MSE, R², MAPE y MASE son complementarias. Las políticas de ceros, escala y cobertura se mantienen en §4.4.2.

### 4.1.2. División del dataset

#### 4.1.2.1. Entrenamiento, validación y prueba

Se conservan 2023 para ajuste y 2024 para desarrollo, con este reparto efectivo:

| Etapa | Calendario UTC | Muestras ejecutadas | Uso |
|---|---|---:|---|
| Entrenamiento estático, D=7 | Resultados elegibles de 2023 hasta 24/12 | 343 | Reajuste completo de cada corrida |
| Subentrenamiento interno, D=7 | Resultados elegibles hasta 23/09/2023 | 251 | Ajuste de pesos y escaladores para elegir duración |
| Validación interna, D=7 | 01/10–24/12/2023 | 85 | Parada temprana; sin seleccionar con 2024 por época |
| Selección | 01/01–30/06/2024 | 182 | Búsqueda de D, anchura y learning rate |
| Calibración y benchmark de desarrollo | 01/07–31/12/2024 | 184 | Piloto y evaluación con semillas nuevas |
| Test reservado | 01/01/2025–31/08/2026 | No construidas ni evaluadas aquí | 608 días de calendario; cerrado |

Los 365 días de 2023, 366 de 2024 y 608 de test suman 1.339 días (27,26 %, 27,33 % y 45,41 %); estas proporciones de calendario no son proporciones de muestras supervisadas. El reparto interno no añade días al calendario. Septiembre–octubre de 2026 queda fuera del experimento.

La [reserva de archivos brutos de test](../_static/data/test_raw_reserve_20250101_20260831.json) identifica 20 ZIP por metadatos y hashes de adquisición, sin abrirlos para este experimento. No equivale a congelar sus muestras, máscaras o transformaciones. La [auditoría de datos](../_static/dl_benchmark_20261104/data_audit.json) registra cero archivos de test abiertos.

#### 4.1.2.2. Esquema de validación y selección de hiperparámetros

La ejecución adoptó **entrenamiento estático de 2023, sin reajustes diarios en 2024**. El manifiesto previo de forward chaining conserva 366 orígenes por cada combinación D–n y controles de disponibilidad; sus 5.856 filas no significan 5.856 entrenamientos realizados. El runner verifica el hash del manifiesto previo y reconstruye los conjuntos del ajuste estático, cuyos índices efectivos y hashes quedan registrados.

El embargo excluye D días de resultado anteriores al primer origen externo $t$: solo se admiten etiquetas de entrenamiento $u\le t-D-1$. Se aplica también antes del comienzo interno $v=01/10/2023$: $u\le v-D-1$. Para D=7 los límites son 24/12 y 23/09/2023, respectivamente. Para D=14,21,28 el entrenamiento completo tiene 322,301,280 muestras; los límites y conjuntos internos están en la auditoría. El embargo no demuestra independencia de los días ni de las etiquetas móviles.

Enero–junio de 2024 selecciona configuraciones. Julio–diciembre se utiliza después para estimar dispersión en el piloto y para el benchmark con semillas distintas: **es calibración de desarrollo, no evaluación externa de todo el procedimiento de selección**. Ni los resultados de ese semestre ni los de test cambian la duración dentro de una corrida. No se ejecutó una evaluación temporal anidada que autorice presentar 2024 como test independiente.

### 4.1.3. Definición de pipeline

#### 4.1.3.1. Preprocesamiento y construcción de datos

El pipeline verifica hashes, calendario, continuidad y alineación; utiliza exclusivamente los derivados diarios de 2023–2024. Las covariables se transforman con logaritmo de OHLC positivo y `log1p` del volumen, y luego se estandarizan por canal. Media y desviación se ajustan sobre los **días de entrada únicos cubiertos por ventanas de entrenamiento**, sin ponderar repetidamente días presentes en varias ventanas. Una desviación nula se sustituye por uno y se registra.

Y se divide por su media positiva de entrenamiento, sin centrarla. El subajuste interno y el reajuste completo calculan sus propios escaladores con sus respectivos entrenamientos. No se usan estadísticas de 2024 para ajustar transformaciones, ni imputación, PCA o selección exploratoria de atributos.

#### 4.1.3.2. Entrenamiento y parada temprana

Cada corrida comprende **dos ajustes**. Primero se entrena con el subentrenamiento de 2023 y se elige el número de épocas por la MSE escalada de su validación interna pasada. Ese modelo se descarta; después se inicializa **otro modelo desde cero** y se ajusta sobre todo el entrenamiento permitido de 2023 durante exactamente ese número de épocas, con escaladores recalculados. No se transfieren pesos ni se hace warm start.

El modelo reajustado emite las predicciones de su fase de 2024 sin nuevas actualizaciones. La etiqueta del día pronosticado no decide la parada. Los detalles comunes y las semillas por etapa figuran en §4.2.

#### 4.1.3.3. Evaluación y agregación de métricas

Se conservan predicciones, objetivos, fechas, modelo, fase, configuración y semilla. Para cada corrida se calcula la RMSE agrupando las fechas de su fase:

$$
\operatorname{RMSE}_{m,s}=\sqrt{\frac1{N_{\mathrm{fase}}}\sum_{d\in\mathcal V}(Y_d-\widehat Y_{m,s,d})^2}.
$$

**No se promedian RMSE unitarias**, cuyo promedio sería MAE. Después se resumen las métricas entre semillas; esa media tampoco equivale a la RMSE de un ensemble. La búsqueda agrega 182 fechas y piloto/benchmark 184. La evaluación definitiva de test requiere su propio cierre previo y no fue ejecutada.

### 4.1.4. Piloto

Se completaron **cinco semillas por modelo**, `11, 23, 37, 53, 71`, después de seleccionar D e hiperparámetros y sin volver a seleccionarlos. Son 15 corridas/30 ajustes sobre las mismas 184 fechas de julio–diciembre de 2024. El [piloto registrado](../_static/dl_benchmark_20261104/pilot_summary.json) contiene:

| Modelo | R piloto | RMSE media | DE muestral | CV |
|---|---:|---:|---:|---:|
| MLP | 5 | 0,01458767 | 0,00232148 | 15,91 % |
| LSTM | 5 | 0,00915657 | 0,00087432 | 9,55 % |
| CNN1D | 5 | 0,02083167 | 0,00179438 | 8,61 % |

Para métricas $m_s$, $\bar m=R^{-1}\sum_s m_s$, $s_m=\sqrt{\sum_s(m_s-\bar m)^2/(R-1)}$ y $\mathrm{CV}=s_m/\bar m$. Las cinco semillas estiman variación del entrenamiento bajo datos fijados, no cinco mercados independientes.

Las semillas del piloto **no se incorporaron** a las 30 del benchmark. El calendario sí se reutilizó, por lo que permanece expuesto al dimensionamiento. El límite de recursos y la regla de R estaban registrados antes del piloto; no se añadieron semillas hasta obtener significación.

| Fase completada | Corridas | Ajustes por corrida | Total de ajustes |
|---|---:|---:|---:|
| Búsqueda | 96 | 2 | 192 |
| Piloto | 15 | 2 | 30 |
| Benchmark de desarrollo | 90 | 2 | 180 |
| **Total** | **201** | **2** | **402** |

Los folds, las configuraciones, los dos ajustes de una corrida y las fechas no se cuentan como semillas independientes.

### 4.1.5. Diferencia mínima de interés

El margen operativo es una **reducción relativa del 5 % de RMSE frente a persistencia** sobre las mismas fechas. Si $b>0$ es su RMSE y $m_{A,s}$ la de una red:

$$
G_s=b-m_{A,s},\qquad Q_s=1-\frac{m_{A,s}}b.
$$

En calibración, $b=0{,}0048224787524133545$ y $\delta_{\mathrm{plan}}=0{,}05b=0{,}00024112393762066773$. La persistencia es determinista: restar el mismo b deja para este cálculo la misma dispersión entre semillas que la RMSE de la red. Esto no hace conocido el error de persistencia en otros periodos.

**Detectar una diferencia de magnitud 5 % respecto de cero no demuestra que la ganancia supere el 5 %.** Para probar $H_0:\mu_Q\le0{,}05$ se necesita inferencia respecto a esa frontera; para dimensionarla haría falta una alternativa $q_{\mathrm{alt}}>0{,}05$, que no se fijó en este ciclo. El bootstrap temporal recalcula ambas RMSE y su razón sobre los mismos bloques.

Los pares de redes usan $d_s=m_{A,s}-m_{B,s}$ y su DE pareada. El 5 % frente a persistencia no se transfiere a equivalencia, a todos los pares ni a una red avanzada frente al baseline DL; los márgenes específicos para esas afirmaciones siguen requiriendo justificación.

### 4.1.6. Tamaño de muestra

La aproximación registrada para detectar una diferencia frente a **cero** es:

$$
R\approx\left\lceil\frac{(z_{1-\alpha_{\mathrm{plan}}/2}+z_{0{,}80})^2\widehat\sigma_d^2}{\delta_{\mathrm{plan}}^2}\right\rceil.
$$

Se usó $\alpha_{\mathrm{plan}}=0{,}05/9$ como nivel conservador de planificación de la familia de nueve hipótesis; el análisis usa Holm. El factor 7,85 correspondería a α=0,05 bilateral sin multiplicidad y **no es el factor usado aquí**. Esta aproximación normal y su dispersión de solo cinco semillas tienen incertidumbre y condicionan el resultado al calendario observado.

| Contraste de planificación | R requerido aproximado | R ejecutado por modelo | Decisión |
|---|---:|---:|---|
| MLP frente a persistencia | 1.212 | 30 | Limitado por recursos |
| LSTM frente a persistencia | 172 | 30 | Limitado por recursos |
| CNN1D frente a persistencia | 724 | 30 | Limitado por recursos |

El protocolo inicial fijó mínimo 10 y **máximo 30 semillas** para este primer ciclo CPU. El mayor requerimiento fue 1.212; se aplicó el máximo de 30 y se registró `cap_limits_power=true` antes del benchmark. **No se alcanzó el tamaño dimensionado ni se acredita potencia del 80 %.** Cumplir los mínimos de cinco/seis corridas no subsana esa limitación.

La potencia objetivo se refiere a una diferencia de magnitud $\delta_{\mathrm{plan}}$ contra cero; no a superar la frontera del 5 %, ni a potencia conjunta de toda la familia. La planificación de superación exigiría usar $(q_{\mathrm{alt}}-0{,}05)^2$ en el denominador con una alternativa predefinida. Tampoco se dimensionaron así nuevos periodos de mercado.

### 4.1.7. Diferencia mínima detectable

Para R semillas, la MDE contra cero se aproxima por:

$$
\mathrm{MDE}_0(R)=\frac{(z_{1-\alpha_{\mathrm{plan}}/2}+z_{0{,}80})\widehat\sigma_d}{\sqrt R}.
$$

Con la dispersión del piloto y **R=30**, se registraron:

| Modelo | MDE₀ en RMSE | δ de planificación | Potencia aproximada al δ planificado |
|---|---:|---:|---:|
| MLP | 0,00153200 | 0,00024112 | 1,42 % |
| LSTM | 0,00057698 | 0,00024112 | 10,34 % |
| CNN1D | 0,00118415 | 0,00024112 | 2,11 % |

En los tres casos MDE₀ supera δ. Son resultados de planificación calculados **antes** del benchmark con el piloto, no potencia retrospectiva a partir de su efecto observado. No certifican independencia temporal ni potencia suficiente para afirmar superación del 5 %. Los valores completos están en el resumen piloto.

### 4.1.8. Relación con el Entregable 2

La autorización de ejecutar ahora permitió completar búsqueda, piloto y benchmark de **desarrollo en E1**. Su aplazamiento anterior a E2 queda superado para esas actividades. La dispersión y el dimensionamiento observados son insumos para el siguiente diseño: muestran que el límite de 30 semillas queda muy por debajo del requerimiento aproximado.

E2 podrá justificar otra asignación de recursos, ampliar evidencia bibliográfica y registrar el protocolo de evaluación reservado. No se da por satisfecho el requisito de potencia ni se abre test automáticamente por haber completado este ciclo. Cualquier cambio posterior identificará qué resultados de desarrollo eran ya conocidos.

### 4.1.9. Registro

#### 4.1.9.1. Preregistro fechado y trazabilidad

El [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json) quedó registrado el **04/11/2026 a las 14:28:17 UTC (09:28:17 Bogotá)** y en el commit **`3c3c481`**, antes del primer ajuste DL del proyecto. Fija datos/hashes, modelos, búsqueda, semillas, dos etapas de ajuste, exclusión estricta, embargo, métricas, potencia y límite de 30.

Después de la búsqueda se guardó la [selección](../_static/dl_benchmark_20261104/selected_configuration.json); después del piloto se congeló el [protocolo de benchmark de desarrollo](../_static/dl_benchmark_20261104/protocol_final.json), el **04/11 a las 14:37:41 UTC**, antes de sus 90 corridas. Este registra D=7, configuraciones, 30 semillas y hashes de los documentos anteriores, sin autorizar test. El código de entrenamiento se identifica por la firma `74dc4b3f4e43ab88c10e6e7d66331b82cc1e0f41287162504cf692f97790a9fc`. La ejecución terminó a las **14:38:23 UTC**.

La especificación del 03/11 y el borrador del 02/11 se conservan como antecedentes, no como configuración vigente. El registro actual es posterior al EDA y a observar datos históricos de desarrollo; preceder a los ajustes no equivale a registrarse antes de toda observación del estudio.

#### 4.1.9.2. Estado del diseño y del piloto

| Producto | Estado comprobable |
|---|---|
| Datos, calendario y exclusiones de desarrollo | Verificados; hashes e índices en auditoría |
| Arquitecturas propias MLP–LSTM–CNN1D | Implementadas; fichas bibliográficas aplicadas aún incompletas según §3 |
| Búsqueda y configuraciones | 96 corridas completadas; D=7 común y tres configuraciones congeladas |
| Preregistro | Protocolo inicial previo a ajustes; versión final previa al benchmark de desarrollo |
| Piloto | Cinco semillas por modelo; 15 corridas completadas |
| Media, DE, CV y dimensionamiento | Calculados; R máximo requerido=1.212; se ejecutó límite 30 |
| Benchmark de desarrollo | 30 semillas por modelo; 90 corridas y métricas disponibles |
| Sensibilidades de objetivo n=14/21/28 | No entrenadas |
| Test 2025–agosto de 2026 | Sin abrir ni evaluar; protocolo final de prueba pendiente |

## 4.2. Implementación de los Tres Modelos Base

Se implementaron **MLP, LSTM unidireccional y CNN1D causal de dos capas**, con configuraciones mínimas propias y comparables. GRU permanece como alternativa documental, sin corrida en este ciclo; TCN y encoder de las fuentes no sustituyen estas redes. Las decisiones propias se declaran a continuación, sin atribuirlas a una réplica exacta de artículos ni presentar como cerrada la revisión aplicada de §3.

### 4.2.1. Implementación de los tres modelos base

Las tres redes reciben D días de las mismas cinco covariables y producen un escalar positivo. Las configuraciones elegidas por la búsqueda son:

| Modelo | Arquitectura implementada | H seleccionado |
|---|---|---:|
| MLP | Flatten → Linear(5D,H) → ReLU → Dropout(0,1) → Linear(H,1) → Softplus | 16 |
| LSTM | Una capa unidireccional → último estado oculto → Dropout(0,1) → Linear(H,1) → Softplus | 32 |
| CNN1D | Conv causal k=3, 5→H → ReLU/Dropout → Conv causal k=3, H→H → ReLU/Dropout → media temporal → Linear(H,1) → Softplus | 16 |

CNN1D usa padding solo a la izquierda; carece de dilatación, conexiones residuales y atención. El dropout es 0,1 después de cada convolución. LSTM usa los dos vectores de sesgo por puerta de PyTorch. Son implementaciones del proyecto; las ecuaciones y configuraciones de artículos de §3.3 se conservan con su atribución original.

### 4.2.2. Entrenamiento con el mismo pipeline, partición y semillas

#### 4.2.2.1. Pipeline, partición e información comunes

| Componente | Aplicación común ejecutada |
|---|---|
| Datos | Mismos dos derivados diarios de 2023–2024 y hashes |
| Entrada y objetivo | OHLCV de D días; σ7 del día siguiente; D=7 en piloto/benchmark |
| Representación | Tensor temporal o aplanado, con los mismos valores |
| Entrenamiento | Cohorte estática 2023 purgada; dos ajustes nuevos por corrida |
| Validación interna | Corte 01/10/2023, con embargo D y escaladores propios |
| Preprocesamiento | Log OHLC, log1p volumen, estandarización train; Y/media train |
| Semillas | Listas comunes dentro de cada fase y separadas entre fases |
| Selección y métricas | Mismo grid, presupuesto, fechas, RMSE y reglas de cobertura |

Para D=7 todas las redes comparten 343 muestras de reajuste y 184 fechas de calibración. La auditoría conserva identificadores y hashes. La referencia de persistencia tiene el alcance informativo declarado en §4.4.3.

#### 4.2.2.2. Semillas, reinicios y reproducibilidad

Se usaron **101 y 202 en búsqueda**; **11, 23, 37, 53 y 71 en piloto**; y **1001–1030 en benchmark de desarrollo**, iguales entre modelos dentro de cada fase. El ajuste interno usa `runseed`; el reajuste completo usa `runseed + 10000000`; el generador de lotes usa la semilla de su etapa más 17.

Cada etapa inicializa pesos nuevos y ajusta sus propios escaladores, sin conservar checkpoints anteriores como warm start. Los lotes de entrenamiento se barajan con generador fijado; las ventanas y cortes siguen siendo cronológicos. Las semillas distintas describen aleatoriedad de entrenamiento sobre el mismo mercado, no nuevas realizaciones independientes de ese mercado. El registro de corridas conserva fase, configuración, semilla, firma de código y resultados.

### 4.2.3. Definición de hiperparámetros

El grid registrado exploró **H∈{16,32}** y **learning rate∈{0,001;0,003}**, con cuatro candidatos por familia y D. Los demás valores son decisiones comunes del protocolo, no hiperparámetros ajustados después del piloto.

| Campo | Valor ejecutado |
|---|---|
| Historia común seleccionada | D=7; objetivo n=7, h=1 |
| MLP | H=16; learning rate 0,001 |
| LSTM | H=32; learning rate 0,003 |
| CNN1D | H=16; learning rate 0,001 |
| Planificador | Ninguno |
| Batch size | 32 |
| Optimizador | AdamW |
| Regularización | Dropout 0,1; weight decay 0,0001 |
| Pérdida | MSE en Y dividida por media positiva train |
| Salida | Softplus |
| Gradiente | Clipping de norma 1 |
| Épocas máximas | 100 en ajuste interno |
| Parada | Mínima MSE de validación interna, patience=12, min_delta=0,00001 |

#### 4.2.3.1. Learning rate y planificador

Se evaluaron learning rates 0,001 y 0,003 con AdamW, **sin planificador**. Se seleccionaron 0,001 para MLP/CNN1D y 0,003 para LSTM usando exclusivamente enero–junio de 2024.

#### 4.2.3.2. Número de capas y unidades

Se compararon anchuras 16 y 32. La MLP tiene una capa oculta (H=16), LSTM una capa recurrente unidireccional (H=32) y CNN1D dos capas convolucionales k=3 (H=16). La profundidad permaneció fija por familia. Son variantes mínimas del proyecto; no se adjudican estos números a las fuentes documentales.

#### 4.2.3.3. Batch size

Se utilizó **batch size=32** para los tres modelos en ambas etapas de ajuste. No se buscó otro tamaño de lote. La medición de inferencia usa un máximo de 256 ejemplos por lote y se distingue del entrenamiento.

#### 4.2.3.4. Optimizador, regularización y función de pérdida

Se usó AdamW con weight decay=0,0001, dropout=0,1 y clipping de norma de gradiente=1. La ubicación del dropout se especifica en §4.2.1. La pérdida es MSE de $Y/\bar Y_{\mathrm{train}}$, con $\bar Y_{\mathrm{train}}>0$ y sin centrar: su término de datos equivale a la MSE original dividida por una constante positiva dentro de cada ajuste. No se entrenó con log(Y).

Los escaladores de X e Y se estiman de nuevo con el entrenamiento admitido de cada etapa; no con su validación. Softplus es la decisión propia para positividad. La inversión de la escala de Y precede a toda métrica publicada.

### 4.2.4. Estrategia de entrenamiento

#### 4.2.4.1. Número máximo de épocas

El ajuste interno permite **hasta 100 épocas**, en CPU y float32. Cada etapa usa pesos nuevos y el orden reproducible de lotes de §4.2.2.2. El reajuste completo ejecuta exactamente la mejor época elegida internamente; no vuelve a seleccionar duración sobre 2024 ni actualiza pesos durante ese año.

#### 4.2.4.2. Early stopping y validación interna

Se evalúa la **MSE escalada de la validación interna cada época**, minimizándola con `patience=12` y `min_delta=1e-5`. Se conserva el número de la mejor época. El ajuste interno se descarta; por tanto, no se usa una restauración de pesos como modelo final: se inicia otro modelo y se reajusta por ese número de épocas.

Para D=7, el subentrenamiento termina el 23/09/2023 y la validación interna abarca 01/10–24/12/2023. Las transformaciones de esta etapa usan solo sus 251 muestras de subentrenamiento. El reajuste completo usa 343 muestras elegibles hasta 24/12, con escaladores recalculados. Los hashes de los tres conjuntos y los cortes para las otras D están en la auditoría. Ninguna etiqueta de 2024 selecciona épocas dentro de una corrida.

### 4.2.5. Registro

Una **corrida** identifica fase, modelo, configuración y semilla, y reúne sus pronósticos. Contiene un ajuste interno para elegir épocas y otro desde cero sobre entrenamiento completo permitido. El total observado es **201 corridas y 402 ajustes**, no 402 semillas independientes.

El [registro de corridas](../_static/dl_benchmark_20261104/runs.csv) conserva identificadores, firma de código, tamaños de muestra, mejor época, épocas de reajuste, parámetros, tiempos, métricas y cobertura. Los artefactos asociados guardan escaladores, pesos, pérdidas por época y predicciones; el protocolo y la auditoría vinculan datos, índices y entorno.

#### 4.2.5.1. Curvas de entrenamiento y validación

Las curvas se generaron desde pérdidas realmente registradas. La figura muestra entrenamiento y validación **interna** de la semilla designada 1001, no pérdidas de test ni curvas sintéticas:

![Curvas reales de los tres modelos, semilla 1001](../_static/dl_benchmark_20261104/analysis/figures/learning_curves_designated_seed.png)

Cada corrida conserva su log; la gráfica representativa no sustituye las 30 métricas individuales del benchmark.

#### 4.2.5.2. Número de parámetros

Los conteos medidos coinciden con estas expresiones propias; todos los parámetros son entrenables y no hay parámetros no entrenables:

| Modelo | Expresión para cinco canales | D y H seleccionados | Parámetros |
|---|---|---|---:|
| MLP | $(5D+1)H+(H+1)$ | D=7, H=16 | 593 |
| LSTM | $4H(5+H+2)+(H+1)$ | D=7, H=32 | 5.025 |
| CNN1D | $(15H+H)+(3H^2+H)+(H+1)$ | D=7, H=16 | 1.057 |

Los dos sesgos por puerta explican el +2 de LSTM. CNN incluye dos convoluciones k=3 y su cabeza lineal; el pooling y Softplus no añaden parámetros. Estos son conteos de código ejecutado, distintos de los conteos documentales condicionados de §3.

#### 4.2.5.3. Tiempo computacional y hardware empleado

Las mediciones siguientes son media ± DE de las 30 corridas de desarrollo, **serializadas en CPU**:

| Modelo | Entrenamiento: dos ajustes (s) | Inferencia de 184 casos (ms) |
|---|---:|---:|
| MLP | 0,3448 ± 0,1384 | 0,03979 ± 0,01082 |
| LSTM | 0,2270 ± 0,0576 | 0,46344 ± 0,08246 |
| CNN1D | 0,6349 ± 0,4926 | 0,35376 ± 0,05189 |

Entrenamiento mide tiempo de pared del ajuste interno más reajuste, incluida su evaluación de parada. Inferencia mide la ruta forward por lotes, incluida conversión y concatenación de salidas, con cinco calentamientos y media de 20 pasadas, batch máximo 256; excluye preprocesamiento, inversión de escala y escritura de artefactos. Los números son locales a este tamaño y entorno, no estimaciones universales de costo. La búsqueda usó tres workers y sus tiempos concurrentes no se mezclan con esta comparación.

El [runtime](../_static/dl_benchmark_20261104/runtime.json) registra CPU, float32, un hilo PyTorch por worker, cinco CPU visibles y cuota cgroup equivalente a cuatro CPU; no se utilizó GPU.

### 4.2.6. Equidad en la optimización

Las tres redes recibieron el mismo grid H×learning rate, las cuatro D y dos semillas de búsqueda: **4 candidatos × 4 historias × 2 semillas =32 corridas por familia**, 96 en total, con dos ajustes por corrida. La profundidad y el resto del entrenamiento se mantuvieron fijos. Igual número de evaluaciones no implica igual tiempo ni capacidad: los parámetros y costos se reportan por separado.

La selección calcula RMSE original sobre las 182 fechas de enero–junio. Para cada familia y D gana la menor media de las dos semillas; los empates exactos favorecen menos parámetros y luego identificador. Se escoge la D común con menor media de los tres ganadores por familia, con empate a menor D. El resultado fue **D=7**.

| Modelo | RMSE media de búsqueda | DE, dos semillas | Configuración elegida |
|---|---:|---:|---|
| MLP | 0,01052383 | 0,00263939 | H=16, lr=0,001 |
| LSTM | 0,01077513 | 0,00014039 | H=32, lr=0,003 |
| CNN1D | 0,01348078 | 0,00079256 | H=16, lr=0,001 |

Por la regla previa, **MLP quedó como identidad del baseline operativo de desarrollo** antes del piloto. Su identidad no se cambia por el ranking posterior de julio–diciembre. Se conservan todos los candidatos en `runs.csv`; no se amplió el grid tras observar el piloto. Este es un presupuesto finito y reducido, no evidencia de optimización exhaustiva de cada familia.

### 4.2.7. Implementación desde cero o uso de librerías justificadas

#### 4.2.7.1. Implementación y entorno de software

La implementación propia usa módulos de **PyTorch 2.15.0+cpu**, Python 3.12.14 y float32. `scripts/dl_data.py` construye datos y escaladores; `scripts/dl_models.py` define las tres redes; `scripts/train_dl_benchmark.py` ejecuta búsqueda, piloto y benchmark; `scripts/analyze_dl_benchmark.py` calcula tablas y figuras. Las librerías se justifican para tensores, diferenciación automática, optimización y análisis, manteniendo explícitas capas y reglas.

El entorno ejecutado queda fijado en `requirements-training.lock.txt` y [runtime.json](../_static/dl_benchmark_20261104/runtime.json): NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, statsmodels 0.15.0 y Matplotlib 3.11.2. Los entornos de EDA y `timeseries-cv` se conservan separados; este último genera índices y no entrena las redes.

#### 4.2.7.2. Estado de implementación

| Producto | MLP | LSTM | CNN1D |
|---|---|---|---|
| Código y configuración propia | Ejecutados, H=16 | Ejecutados, H=32 | Ejecutados, H=16, dos convoluciones |
| Búsqueda comparable | 32 corridas | 32 corridas | 32 corridas |
| Piloto | 5 corridas | 5 corridas | 5 corridas |
| Benchmark de desarrollo | 30 corridas | 30 corridas | 30 corridas |
| Curvas, parámetros y tiempos | Registrados | Registrados | Registrados |
| Test reservado | No ejecutado | No ejecutado | No ejecutado |

La existencia de código entrenado no completa la trazabilidad bibliográfica de MLP/CNN1D ni demuestra reproducción exacta de los artículos. Esas limitaciones de §3 permanecen identificadas.

## 4.3. Requisitos Experimentales

Los controles siguientes acompañan la ejecución real de desarrollo del 04/11/2026. Se distingue lo verificado en esas corridas de las sensibilidades, pruebas de robustez adicionales y evaluación reservada que siguen pendientes.

### 4.3.1. Control de semillas aleatorias

El runner fija Python, NumPy y PyTorch por etapa; utiliza las semillas y offsets de §4.2.2.2, incluido un generador explícito para lotes. Se activó **`torch.use_deterministic_algorithms(True)`** y un hilo de cómputo/interoperabilidad por worker. Runtime, precisión, backend y firma de código quedan registrados; no se presume identidad bit a bit entre versiones o hardware distintos.

La misma etiqueta de semilla entre arquitecturas conserva el emparejamiento operativo; no produce pesos iguales ni garantiza reducción de varianza. Una repetición técnica con igual semilla no se cuenta como otra corrida independiente de entrenamiento.

### 4.3.2. Repetición de experimentos

Se completaron **30 semillas nuevas por modelo** en el benchmark: 90 corridas, además de las 15 piloto y 96 de búsqueda. Comparten las 184 fechas y D=7. La cifra supera los mínimos de cinco/seis, pero **no el requerimiento máximo de 1.212 calculado con el piloto**: se aplicó el límite de recursos predeclarado, con potencia insuficiente registrada.

| Unidad | Interpretación correcta |
|---|---|
| Semilla | Aleatoriedad de entrenamiento bajo datos y procedimiento fijados |
| Ajuste | Una de las dos etapas de una corrida; no otra semilla |
| Fecha | Resultado temporal dependiente; no repetición independiente |
| Configuración | Otro candidato; no réplica de una misma hipótesis |

No se barajó la serie para crear particiones independientes ni se contaron 366 folds/5.856 filas previas como semillas. La dependencia de fechas se aborda separadamente; más semillas no crean más periodos independientes de mercado.

### 4.3.3. Reporte de media, desviación estándar e intervalos de confianza

El [resumen de métricas](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv) contiene media, DE muestral y **IC marginal del 95 % t de Student con 29 grados de libertad** sobre las 30 semillas, condicionado a las mismas fechas de desarrollo. Los valores primarios aparecen en §4.4.9.1; no son intervalos de generalización a mercados futuros ni intervalos simultáneos de toda la familia.

Las [comparaciones pareadas](../_static/dl_benchmark_20261104/analysis/paired_tests.csv) usan las mismas semillas y calendario con Holm en nueve hipótesis predefinidas. Los supuestos se diagnostican sin cambiar a la prueba que resulte favorable. Para fechas dependientes se ejecutó [bootstrap circular pareado por bloques](../_static/dl_benchmark_20261104/analysis/block_bootstrap.csv), longitud 14, sensibilidad 7/28, 2.000 réplicas y semilla 20261104: recalcula el funcional de RMSE y la referencia sin reentrenar. Ambos análisis son de desarrollo y condicionan en los modelos ya seleccionados.

### 4.3.4. Uso de GPU

**Se entrenó en CPU; no se usó GPU.** El [runtime](../_static/dl_benchmark_20261104/runtime.json) registra `cuda_available=false`, Linux x86_64, precisión float32, versiones y límites cgroup. La memoria máxima permitida fue 32 GiB; ese límite no es una medición de pico consumido. No se inventan driver o memoria GPU inexistentes.

Piloto y benchmark se ejecutaron con un worker, un hilo PyTorch y un hilo de interoperabilidad. La búsqueda tuvo tres workers; sus tiempos no se usan como medida comparable serial. El alcance y calentamiento de la medición de inferencia están en §4.2.5.3.

### 4.3.5. Empaquetado reproducible

El experimento de desarrollo tiene código, versiones, manifiestos y artefactos para repetir sus fases. Reproducirlo requiere los derivados verificados de 2023–2024; no necesita abrir test.

#### 4.3.5.1. Archivo de dependencias

`requirements-training.lock.txt` fija las versiones efectivamente usadas en entrenamiento/análisis; los entornos de EDA y partición conservan sus propios archivos. El runtime identifica las versiones importadas, además de la lista de dependencias.

#### 4.3.5.2. Scripts o cuadernos ejecutables de extremo a extremo

Los scripts `dl_data.py`, `dl_models.py`, `train_dl_benchmark.py`, `analyze_dl_benchmark.py` y `package_dl_benchmark.py` conservan construcción de muestras, ajuste, análisis y exportación. El protocolo registra argumentos, fuentes y firmas; la auditoría detalla tamaños e índices de cada D. Los brutos oficiales conservan URLs y hashes de adquisición y no se redistribuyen dentro del sitio.

#### 4.3.5.3. Registro de configuración y artefactos de cada corrida

Se conservan configuración por corrida, semilla, dos etapas de ajuste, escaladores, logs, predicciones, métricas y tiempos. Los pesos finales guardados corresponden a las 105 corridas piloto y benchmark; la búsqueda no conserva checkpoints. El [registro CSV](../_static/dl_benchmark_20261104/runs.csv), los protocolos, runtime, auditoría y resumen de ejecución enlazan esas piezas; el análisis conserva hashes de predicciones y del script. Se puede descargar el [paquete de artefactos de corrida](../_static/dl_benchmark_20261104/run_artifacts.tar.gz), verificar su [manifiesto](../_static/dl_benchmark_20261104/run_artifacts_manifest.json) y consultar las {download}`instrucciones de reproducción <../_static/dl_benchmark_20261104/README.md>`.

La reproducción de búsqueda/piloto/benchmark de **desarrollo ya dispone de implementaciones ejecutadas y resultados**. Esto no acredita todavía reproducción de un experimento de test, que no se ha realizado. Los registros históricos vacíos se mantienen diferenciados de los artefactos reales del 04/11.

### 4.3.6. Buenas Prácticas Obligatorias

#### 4.3.6.1. Prevención de data leakage (crítico)

| Control | Aplicación ejecutada | Evidencia |
|---|---|---|
| Separación temporal | Ajuste 2023; búsqueda enero–junio; calibración julio–diciembre 2024 | Protocolo y auditoría de índices |
| Transformaciones solo train | Escaladores propios de cada una de las dos etapas | Código firmado y artefactos de corrida |
| Continuidad | Sin imputar ni formar ventanas/retornos atravesando huecos | Calendario y exclusiones |
| Embargo D | Antes del primer origen externo y del corte interno | Límites e índices auditados |
| Mismas condiciones entre redes | Información, fechas, semillas, grid y métrica comunes | 201 corridas registradas |
| Parada sin etiqueta externa | Solo validación interna 2023; reajuste nuevo por épocas fijas | Logs y mejor época por corrida |
| Test cerrado | Cero archivos test abiertos | Auditoría y resumen de ejecución |

Los incisos por modalidad se conservan a continuación, con su aplicación o motivo de no aplicación.

##### 4.3.6.1.1. Separación de entrenamiento, validación y prueba

Aplica: las etiquetas de 2024 no entraron en el ajuste de pesos ni escaladores de estas corridas estáticas. Entradas de validación reutilizan historia pasada observable. El uso de julio–diciembre para dimensionar y evaluar está declarado como calibración; no se denomina test independiente.

##### 4.3.6.1.2. Transformaciones ajustadas solo en train

Aplica: log/log1p son transformaciones puntuales; medias/desviaciones de X y media positiva de Y se ajustan solo en el entrenamiento admitido de cada etapa. No se usaron estadísticas de validación para escalado ni imputación. Codificación categórica y vocabulario no aplican a OHLCV numérico.

##### 4.3.6.1.3. Prevención de mezcla temporal

Aplica: se respetaron disponibilidad de covariables/etiquetas, ventanas completas y embargo D. Barajar lotes del entrenamiento permitido no altera sus límites. La validación interna permanece en 2023 y no se actualizan pesos durante 2024. Las volatilidades móviles siguen siendo temporalmente dependientes.

##### 4.3.6.1.4. Prevención de fuga espacial

El control exige validar por bloques con separación mayor al rango de autocorrelación espacial. **No aplica al proyecto actual:** se modela una sola serie financiera sin coordenadas, unidades geográficas ni una estructura de vecindad espacial. No se presentan bloques o rangos espaciales como controles ejecutados.

##### 4.3.6.1.5. Prevención de fuga semántica en NLP

El control exige evitar duplicados y casi duplicados entre particiones y respetar el agrupamiento por autor o documento cuando exista. **No aplica al proyecto actual:** las entradas son OHLCV y no incluyen corpus textual, autores o documentos. Los datos de Twitter de una fuente bibliográfica no forman parte del dataset propio.

##### 4.3.6.1.6. Prevención de fuga por agrupamiento en imágenes

El control exige mantener las imágenes del mismo sujeto, sitio o escena en una sola partición. **No aplica al proyecto actual:** no hay imágenes de entrada ni unidades de agrupamiento de esa modalidad. Los gráficos del EDA son salidas de análisis, no observaciones visuales usadas para entrenar los modelos.

#### 4.3.6.2. Validación consistente entre los tres modelos base

Aplica y se ejecutó en desarrollo: las redes comparten muestras elegibles, horizonte, cinco covariables, semillas por fase, métricas, cobertura y presupuesto. Los conteos e índices están auditados. La diferencia informativa de persistencia se declara aparte; ninguna condición común convierte la calibración en evaluación final de test.

#### 4.3.6.3. Evaluación única del test por corrida

El test de 2025–agosto de 2026 **no se evaluó** y el protocolo del 04/11 no autoriza abrirlo. Antes de su ejecución se fijarán manifiesto supervisado, arquitectura, hiperparámetros, semillas, actualización de pesos y regla de parada. Se realizará una sola pasada cronológica por corrida, con predicciones guardadas para métricas/figuras sin reselección.

La política estática ejecutada en desarrollo no obliga a una política futura: cualquier actualización durante test deberá registrarse previamente, emitir cada pronóstico antes de observar su etiqueta y respetar el embargo. No se escogerá semilla ni cadencia a partir de test. Se reportarán 2025 y 2026 por separado, sin incorporar automáticamente archivos posteriores al calendario reservado.

#### 4.3.6.4. Estado de cumplimiento

| Requisito | Estado |
|---|---|
| Datos, hashes y exclusiones de desarrollo | Verificados |
| Embargo, cortes y disponibilidad | Auditados para las cuatro D |
| Semillas y determinismo CPU | Registrados y ejecutados |
| Implementaciones, piloto y benchmark de desarrollo | Completados: 201 corridas, 402 ajustes |
| Media, DE, IC, parámetros y tiempos | Calculados y exportados |
| Potencia objetivo | No satisfecha al δ planificado; límite R=30 explícito |
| Código y dependencias de entrenamiento | Disponibles; firma y runtime conservados |
| Robustez con perturbaciones y objetivos n≠7 | No ejecutada |
| Identidad bruta de test | Reserva de 20 ZIP por hashes/metadatos |
| Test supervisado y evaluación final | Pendientes; sin archivos abiertos |

## 4.4. Evaluación

El proyecto corresponde a **regresión temporal con objetivo continuo, ruta 6.4 de la guía**, porque las entradas OHLCV no incluyen rezagos explícitos del objetivo. Se aplican las métricas de regresión y los diagnósticos temporales pertinentes. Las métricas de clasificación, segmentación, texto o localización espacial no se incorporan a este problema de volatilidad de Bitcoin.

### 4.4.1. Métrica principal, métricas secundarias y clasificación

La métrica primaria ejecutada es **RMSE**, coherente con MSE de entrenamiento; se reportaron **MAE, MSE, R², MAPE y MASE**. No se calculó sMAPE en este ciclo; adoptarla después exigiría declarar su carácter adicional.

**Clasificación tabular, de imágenes y de texto no aplica:** se predice una volatilidad continua y no clases o probabilidades de clase. Se conservan todos los incisos de la guía para documentar ese alcance; no se presentan como métricas calculadas del proyecto.

#### 4.4.1.1. Accuracy

Proporción de observaciones correctamente clasificadas. Debe acompañarse del soporte por clase, pues el desbalance puede hacer que una exactitud elevada oculte errores en clases minoritarias.

#### 4.4.1.2. Precision y Recall

Reportar precisión y exhaustividad por clase, identificando la clase positiva, el umbral de decisión y el criterio de agregación cuando corresponda. Los umbrales se fijan con desarrollo y no se seleccionan mediante test.

#### 4.4.1.3. F1-score macro, ponderado y por clase

Informar F1 por clase, promedio macro y promedio ponderado por soporte. Declarar el tratamiento de clases ausentes o denominadores nulos; los promedios no sustituyen el desglose por clase.

#### 4.4.1.4. AUC-ROC

Calcular el área bajo la curva ROC a partir de puntuaciones o probabilidades, no solo de etiquetas finales. En multiclase se declararán la estrategia uno contra el resto o uno contra uno y el promedio utilizado.

#### 4.4.1.5. AUC-PR

Reportar el área bajo la curva precisión–recall, recomendada especialmente con desbalance, junto con la prevalencia de la clase positiva. Declarar la convención de cálculo y distinguir el área de PR de average precision cuando la implementación no las haga equivalentes.

#### 4.4.1.6. Coeficiente de correlación de Matthews

Reportar MCC, especificando su variante binaria o multiclase y el tratamiento de los casos no definidos. Complementa las métricas anteriores; no reemplaza el análisis de errores por clase.

#### 4.4.1.7. Calibración

Evaluar el error de calibración esperado (ECE) y presentar un diagrama de fiabilidad. Declarar el número y la regla de los intervalos de probabilidad, sus soportes y el procedimiento multiclase utilizado. Discriminación y calibración responden a propiedades distintas de las probabilidades.

### 4.4.2. Regresión y series de tiempo: métricas de error

Este bloque **aplica**. Las métricas se calcularon en escala original por corrida y fase. En el benchmark D=7,n=7,h=1 todas las redes tienen 30 corridas y 184 fechas por corrida; MAPE y MASE tienen cobertura del 100 %.

#### 4.4.2.1. MSE, RMSE y MAE

Para objetivos $Y_t$, predicciones $\widehat Y_t$ y $N$ pares elegibles del mismo modelo, configuración, semilla y horizonte, se define $e_t=Y_t-\widehat Y_t$:

$$
\mathrm{MSE}=\frac1N\sum_t e_t^2,\qquad
\mathrm{RMSE}=\sqrt{\frac1N\sum_t e_t^2},\qquad
\mathrm{MAE}=\frac1N\sum_t|e_t|.
$$

#### 4.4.2.2. Coeficiente de determinación R²

$$
R^2=1-\frac{\sum_t e_t^2}{\sum_t(Y_t-\bar Y)^2}.
$$

R² puede ser negativo. Si el objetivo evaluado es constante, su denominador es cero y se reportará como no definido, junto con el contexto; no se sustituirá silenciosamente por un valor favorable. Con un único resultado por fold, R² por fold no está definido: se calculará sobre la colección de predicciones fuera de entrenamiento, no promediando R² diarios.

La media de Y del periodo evaluado aparece solo en la **definición de R²**. No autoriza usarla para entrenar ni como pronóstico disponible en el origen. Un R² frente a media train, si se informa, tendrá nombre distinto del R² convencional.

#### 4.4.2.3. Escala original y transformaciones del objetivo

Las métricas publicadas se calcularon en la **escala original de la volatilidad**, después de invertir la escala de la etiqueta: unidades de retorno logarítmico sin anualizar, no puntos porcentuales. Multiplicar la etiqueta por 100 multiplica RMSE y MAE por 100 y MSE por 10.000; los resultados de distintas escalas no se mezclarán.

#### 4.4.2.4. MAPE y cobertura de objetivos positivos

La política de MAPE ya está fijada: se utiliza exclusivamente el subconjunto de objetivos positivos, $\mathcal I_+=\{t:Y_t>0\}$, con $N_+=|\mathcal I_+|$:

$$
\mathrm{MAPE}_+=\frac{100}{N_+}\sum_{t\in\mathcal I_+}\left|\frac{e_t}{Y_t}\right|,\qquad N_+>0.
$$

Se informarán el número de ceros, los casos incluidos y la cobertura $N_+/N$. Si no hay objetivos positivos, MAPE se marca no definida. No se añadirán epsilons arbitrarios ni se ocultarán casos excluidos. Valores positivos muy pequeños pueden dominar MAPE y requieren interpretación conjunta con MAE y RMSE.

#### 4.4.2.5. sMAPE: convención y valores cercanos a cero

sMAPE no fue adoptada ni calculada en el protocolo ejecutado. Si se incorpora en otro ciclo, se registrarán convención, factor de escala y tratamiento de Y=predicción=0 antes de evaluarlo; su simetría no elimina los problemas cerca de cero.

#### 4.4.2.6. MASE y escala de entrenamiento

Para MASE se fijó el rezago de referencia $s=1$ día de calendario. En el entrenamiento permitido se calcula:

$$
a_{\mathrm{train},s}
=\frac1{|\mathcal J_s|}\sum_{j\in\mathcal J_s}|Y_j-Y_{j-s}|,
\qquad |\mathcal J_s|>0.
$$

$\mathcal J_s$ contiene pares de etiquetas válidas separados por **$s$ días de calendario**, ambos pertenecientes a la historia de entrenamiento permitida. No se comprimen huecos. La escala se calcula en unidades originales del objetivo y es común a los métodos comparados en ese origen, con identificador de su conjunto de referencia.

Se define la máscara de escalas admisibles y su tamaño:

$$
\mathcal I_M=\{t\in\mathcal V:a_{\mathrm{train}(t),1}>0\text{ y escala definida}\},
\qquad N_M=|\mathcal I_M|,
$$

$$
\mathrm{MASE}=\frac1{N_M}\sum_{t\in\mathcal I_M}
\frac{|Y_t-\widehat Y_t|}{a_{\mathrm{train}(t),1}},\qquad N_M>0.
$$

Si no hay pares válidos o la escala es cero, se conserva el caso como no definido y su motivo. Se reportan $N_M$, el total esperado y cobertura $N_M/N$; si $N_M=0$, la métrica completa es indefinida. La máscara proviene del entrenamiento y es común entre métodos, sin excluir fechas por su error. Una predicción no finita sigue la política de incidentes, no se oculta como exclusión de MASE.

En este ajuste estático D=7, la escala registrada fue **0,002482618143400422**, común a las redes durante 2024. Si en otro protocolo cambia con entrenamiento creciente, se promedian **errores escalados por origen**; no se divide la MAE global por una escala elegida después. MASE<1 compara esos errores con la variación ingenua estimada en entrenamiento, no demuestra por sí sola que el modelo supere a persistencia durante la evaluación. El rezago s=1 permanece fijado sin mirar test.

### 4.4.3. Series de tiempo: error por horizonte y referencias ingenuas

Se evaluó **h=1, n=7**: búsqueda con D=7/14/21/28 y piloto/benchmark con D=7 seleccionado. No se entrenaron objetivos n=14/21/28 ni horizontes adicionales. Estos últimos requerirían reporte separado; variar D no significa cambiar horizonte.

La referencia calculada es $\widehat\sigma_7(d)=\sigma_7(d-1)$, **persistencia de la volatilidad disponible**, con RMSE 0,00482248 y MAE 0,00287267 en las mismas 184 fechas. Es una sola trayectoria determinista, no 30 entrenamientos. Necesita ocho cierres históricos para siete retornos; la red D=7 recibe siete días OHLCV. Esta diferencia informativa está declarada y limita la comparación como contraste de arquitecturas con información idéntica, aunque ambos pronósticos respeten disponibilidad.

La regresión lineal regularizada de §2.2.3.2.3 sigue siendo diagnóstico previo de EDA; no se reetiqueta como corrida del nuevo benchmark. El ingenuo estacional **no se ejecutó**: falta justificar/fijar su periodo. Ninguna referencia clásica ocupa un puesto del trío DL.

Las referencias y redes se comparan en calendario, escala y etiqueta comunes. RMSE se calcula sobre todos los errores de cada corrida, sin promediar RMSE de días unitarios.

### 4.4.4. Regresión espacial y espacio-temporal

**No aplica al proyecto actual:** existe una única serie financiera temporal sin ubicaciones, regiones ni una estructura geográfica de vecindad. Los requisitos siguientes se documentan sin presentar mapas o diagnósticos espaciales ejecutados.

#### 4.4.4.1. Errores globales y por ubicación o región

Reportar métricas globales y desglosadas por ubicación o región, junto con cobertura y número de observaciones. Declarar cómo se agregan ubicaciones y periodos para que las unidades con más registros no determinen inadvertidamente todo el resultado.

#### 4.4.4.2. Autocorrelación espacial de los residuos

Aplicar el I de Moran sobre **residuos del modelo**, con matriz de pesos espaciales, criterio de vecindad y procedimiento de inferencia documentados. La autocorrelación del objetivo no sustituye este diagnóstico residual. Este control es distinto de la autocorrelación temporal examinada para BTCUSDT.

### 4.4.5. Etiquetado secuencial

**No aplica al proyecto actual:** se predice una volatilidad escalar por origen y no etiquetas por token ni entidades delimitadas en una secuencia.

#### 4.4.5.1. Precisión, exhaustividad y F1 a nivel de entidad

Reportar precisión, exhaustividad y F1 con coincidencia exacta de spans y tipo de entidad. Documentar el esquema de etiquetas y el tratamiento de secuencias inválidas; un acierto parcial de un span no cuenta como una entidad exacta.

#### 4.4.5.2. Exactitud a nivel de token

Reportar adicionalmente la proporción de tokens correctamente etiquetados, excluyendo padding y posiciones ignoradas según una regla declarada. Esta exactitud complementa las métricas de entidad y no las sustituye.

### 4.4.6. Aprendizaje secuencia a secuencia

**No aplica al proyecto actual:** la salida es una volatilidad continua escalar, no una secuencia generada ni una referencia textual.

#### 4.4.6.1. Métricas de generación acordes a la tarea

Seleccionar BLEU, ROUGE, chrF o exactitud de secuencia completa según el objetivo de generación y declarar la métrica principal. Estas métricas no se trasladan a errores de volatilidad.

#### 4.4.6.2. Configuración exacta del cálculo

Registrar tokenización y normalización, versión de la implementación, referencias utilizadas, parámetros de la métrica y agregación por ejemplo o corpus. Documentar también la decodificación empleada para generar las secuencias; cambios de configuración no se presentan como mejoras del modelo bajo una evaluación idéntica.

### 4.4.7. Segmentación de imágenes

**No aplica al proyecto actual:** no existen imágenes de entrada, máscaras ni etiquetas por píxel. Las expresiones siguientes documentan la guía y no representan resultados de segmentación del proyecto.

#### 4.4.7.1. Intersection over Union (IoU)

Para una clase, TP, FP y FN son verdaderos positivos, falsos positivos y falsos negativos a nivel de píxel:

$$
\mathrm{IoU}=\frac{TP}{TP+FP+FN}.
$$

Se declarará el tratamiento de clases o imágenes con denominador cero, sin asignarles silenciosamente un valor favorable.

#### 4.4.7.2. Dice Coefficient

El coeficiente Dice, equivalente al F1 binario de píxeles para la clase evaluada, se define como:

$$
\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}.
$$

Se mantendrán las mismas convenciones de clases, máscaras y casos no definidos al comparar modelos.

#### 4.4.7.3. Pixel Accuracy

Proporción de píxeles evaluables correctamente clasificados. Declarar las posiciones ignoradas y el tratamiento del fondo; un fondo dominante puede ocultar errores en objetos pequeños.

#### 4.4.7.4. Mean IoU (mIoU)

Promedio de IoU sobre las clases incluidas en la evaluación. Especificar si se incluye el fondo, cómo se tratan las clases ausentes y si se agregan primero conteos del corpus o métricas por imagen. Esas convenciones producen estimandos distintos y deben mantenerse comunes entre modelos.

#### 4.4.7.5. Métricas de borde (opcionales)

Cuando la precisión espacial de los contornos sea crítica, documentar la métrica de borde, tolerancia, unidades y resolución espacial. La unidad de análisis y remuestreo será la imagen o el grupo independiente correspondiente; los píxeles de una imagen no se tratarán como réplicas independientes.

### 4.4.8. Comparación justa entre modelos base

#### 4.4.8.1. Mismo dataset y partición

La auditoría verifica para las tres redes D=7 los mismos 343 ejemplos de reajuste y 184 fechas de calibración, con etiquetas, información y semillas compartidas. La MLP reorganiza los mismos valores. Persistencia conserva la diferencia de un cierre histórico documentada en §4.4.3; su costo o información no se confunden con los de una cuarta red.

#### 4.4.8.2. Mismo pipeline de preprocesamiento

Las redes aplican idéntico contrato causal y transformaciones, con escaladores propios de cada etapa ajustados a sus mismos conjuntos train. Los cambios de forma temporal/aplanada no añaden atributos. Se mantiene exclusión estricta, sin imputación ni selección de días por error.

#### 4.4.8.3. Misma configuración de evaluación

Se ejecutaron las mismas definiciones de métricas, inversión de escala y reglas de cobertura. MAPE/MASE cubren todas las fechas; el registro conserva los conteos. La búsqueda tuvo igual presupuesto por familia y se reportan medias de todas las semillas de la fase, sin escoger la mejor semilla para una red.

Para $b>0$, $Q=1-\mathrm{RMSE}_{\mathrm{red}}/b$. Un valor puntual $Q\ge0{,}05$ alcanzaría el criterio puntual; demostrar superación requiere intervalo/contraste contra 0,05. En estas corridas **ninguna red supera a persistencia en RMSE media**. El bootstrap pareado recalcula numerador y denominador conjuntamente; su incertidumbre no transforma el periodo de calibración en test.

### 4.4.9. Análisis de resultados

#### 4.4.9.1. Media, desviación estándar e intervalo de confianza

Resultados de **julio–diciembre de 2024**, no de test, con D=7,n=7,h=1 y 30 semillas nuevas por red:

| Modelo | RMSE media ± DE | IC marginal 95 % de la media | MAE media | R² medio |
|---|---:|---:|---:|---:|
| MLP | 0,01599031 ± 0,00285446 | [0,01492443; 0,01705618] | 0,01381739 | −2,73998 |
| LSTM | 0,00890346 ± 0,00048544 | [0,00872220; 0,00908473] | 0,00688318 | −0,12809 |
| CNN1D | 0,01758669 ± 0,00519212 | [0,01564792; 0,01952546] | 0,01522349 | −3,75860 |
| Persistencia | 0,00482248; determinista | No tiene IC entre semillas | 0,00287267 | Véase tabla de métricas |

Los IC son t de Student entre semillas, condicionados a estas fechas; no resumen toda la incertidumbre temporal ni corrigen simultáneamente todas las métricas. MSE, MAPE, MASE y sus dispersiones están en el [CSV completo](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv). R² negativo significa error mayor que la referencia de media del periodo en su definición; no cambia el criterio causal de entrenamiento.

LSTM tiene el menor error descriptivo entre redes en este semestre, pero **MLP conserva la identidad operativa seleccionada en enero–junio**. El cambio de ranking advierte sobre selección y estabilidad; no se reemplaza el baseline después de ver calibración. Ninguna media neuronal supera a persistencia.

#### 4.4.9.2. Variabilidad por semilla frente a diferencias entre modelos

Se conservaron las métricas individuales, DE y diferencias pareadas de las 30 semillas. La figura muestra variación de entrenamiento condicionada al mismo semestre:

![Distribución de RMSE por semilla](../_static/dl_benchmark_20261104/analysis/figures/rmse_seed_variability.png)

No se ajustó un modelo formal de componentes de varianza. Las tres arquitecturas son niveles elegidos, no una muestra aleatoria de todas las redes. La dispersión de sus medias no estima una varianza universal entre familias. Tampoco se concatenan semillas o fechas como réplicas i.i.d.; la incertidumbre de fechas se examina con bloques y diferencias temporales de pérdidas, con el alcance y supuestos del capítulo 5.

#### 4.4.9.3. Robustez por subgrupos, ruido y perturbaciones

**No se ejecutaron perturbaciones controladas de ruido o faltantes, ni un estudio de robustez por subgrupos.** El bootstrap de bloques estudia incertidumbre de las predicciones guardadas; no es un entrenamiento con datos perturbados. Las sensibilidades de objetivo n=14/21/28 siguen sin entrenar.

Para un estudio futuro, las reglas de segmentos y umbrales deberán fijarse con entrenamiento. Si se perturba OHLCV, se conservarán precios positivos, volumen no negativo y coherencia H/L/O/C; la ley, intensidad, semilla y cobertura perdida quedarán registradas. Alterar Y y recalcular volatilidad sería otra simulación. Se mantendría la exclusión estricta de huecos, sin elegir máscaras que retiren días difíciles.

#### 4.4.9.4. Diagnóstico de overfitting y underfitting

Las curvas reales de §4.2.5.1 permiten examinar brechas entre subentrenamiento y validación interna para la semilla designada. No se interpretan como curvas de test. La duración elegida y la de reajuste se conservan para todas las corridas; no se escogió una curva especialmente favorable.

Las métricas publicadas muestran error alto respecto a persistencia, por lo que una brecha pequeña por sí sola no acreditaría un ajuste adecuado. No se atribuye todo el error a overfitting o underfitting sin separar capacidad, escala, selección y cambios temporales. La revisión de curvas es diagnóstica y no reabre hiperparámetros después del benchmark.

#### 4.4.9.5. Análisis de errores por segmentos relevantes

La búsqueda conserva resultados por D=7/14/21/28 con n=7 y h=1; las métricas de enero–junio son de selección. El nuevo benchmark solo evalúa D=7. **No se completó un desglose cuantitativo por rangos de volatilidad o subperiodos adicionales**; las predicciones fechadas permiten hacerlo como análisis exploratorio posterior claramente identificado.

Los rangos del objetivo real serían diagnósticos retrospectivos, no covariables disponibles al emitir el pronóstico. Cualquier segmento deberá informar regla, tamaño y cobertura; no se escogerán solo periodos favorables. Clase, ubicación y longitud textual no aplican porque la tarea no tiene esas modalidades. Las sensibilidades n≠7 no se inventan a partir del estudio de D.

### 4.4.10. Análisis cualitativo

#### 4.4.10.1. Visualización de predicciones frente a valores reales

La figura utiliza valores reales y predicciones de las tres redes para la semilla **1001 designada en el protocolo**, D=7,n=7,h=1, sobre las 184 fechas de calibración:

![Predicciones reales de desarrollo, semilla 1001](../_static/dl_benchmark_20261104/analysis/figures/predictions_designated_seed.png)

No es una selección de la mejor semilla ni una evaluación de test. La comparación con máscaras ground truth no aplica por ausencia de segmentación.

#### 4.4.10.2. Errores sistemáticos y casos difíciles

La visualización fechada y los residuos permiten localizar desviaciones persistentes y episodios difíciles. El hallazgo agregado verificable es que ninguna red mejora la RMSE media de persistencia en calibración. No se completó una clasificación causal de casos difíciles ni se atribuyen errores a causas económicas no comprobadas.

#### 4.4.10.3. Residuos: estructura, sesgo y heterocedasticidad

Los residuos conservan fecha y signo $e_d=Y_d-\widehat Y_d$; un valor medio positivo indica subestimación. Se calculó ACF de residuos y residuos al cuadrado sin concatenar semillas ni tramos separados. La figura corresponde a la semilla 1001:

![ACF de residuos, semilla designada](../_static/dl_benchmark_20261104/analysis/figures/residual_acf_designated_seed.png)

Se ejecutaron **Ljung–Box a rezagos 7/14/28**, con `model_df=0` (sin restar pesos neuronales), y **BDS con dimensiones 2/3 y radio 1,5 veces la DE muestral residual**, sobre el tramo continuo de 184 fechas. La familia de 15 diagnósticos de las tres redes usa Holm; parámetros, advertencias, estadísticas y p-valores quedan en [diagnostics.csv](../_static/dl_benchmark_20261104/analysis/diagnostics.csv) y la ACF en [acf.csv](../_static/dl_benchmark_20261104/analysis/acf.csv).

Son diagnósticos aproximados de residuos de desarrollo. Un rechazo de BDS cuestiona su hipótesis i.i.d.; no identifica por sí solo una causa no lineal ni demuestra mejora predictiva alcanzable. La volatilidad móvil consecutiva comparte seis retornos, por lo que h=1 no asegura errores independientes. No se ejecutó un análisis formal adicional de heterocedasticidad que permita darlo por completado.

#### 4.4.10.4. Productos y estado de ejecución

| Producto de evaluación | Estado real |
|---|---|
| Métricas y agregación | Calculadas en escala original, con cobertura MAPE/MASE del 100 % |
| Predicciones fechadas de las tres redes | Generadas en búsqueda, piloto y benchmark de desarrollo |
| Media, DE e IC entre semillas | Disponibles para 30 corridas por red |
| Curvas, parámetros y tiempos | Registrados; figuras de semilla designada disponibles |
| Comparaciones pareadas y análisis temporal | Ejecutados sobre predicciones de desarrollo; detalle en §5 |
| ACF, Ljung–Box y BDS de residuos | Ejecutados con parámetros y multiplicidad registrados |
| Robustez por ruido/subgrupos y sensibilidades n=14/21/28 | No ejecutadas |
| Evaluación definitiva de test | No ejecutada; archivos reservados cerrados |

Estos productos proceden de corridas y predicciones verificables del 04/11/2026. El benchmark documenta un resultado de desarrollo desfavorable frente a persistencia, un cambio de ranking entre semestres y una limitación de potencia; no demuestra superioridad final ni generalización a 2025–2026.
