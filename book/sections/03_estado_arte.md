# 3. Revisión del Estado del Arte y Extracción de los Tres Modelos Base

La revisión debe identificar exactamente tres modelos DL básicos usados como referencias comparativas en la literatura de volatilidad de Bitcoin. Este capítulo reúne **35 referencias bibliográficas y un corpus exploratorio de cinco textos completos**; son unidades distintas. Las fichas de LSTM, TCN y encoder Transformer siguen siendo candidatas de trabajo: la selección final por recurrencia, mínima complejidad y no redundancia todavía no está acreditada.

La revisión del **03/11/2026** corrigió motivos de cribado, recuperó dos textos adicionales y completó la extracción de los hiperparámetros ganadores de S1, que sí aparecen en su apéndice. Se conservan los cortes anteriores y se explicitan fuentes grises, parámetros desconocidos y límites de las métricas. No se entrenaron redes ni se consultaron datos de test del proyecto.

El problema propio es **regresión temporal, tipo 4, ruta 6.4**: OHLCV rezagadas sin historia explícita de la volatilidad objetivo. Las tareas de las fuentes no coinciden automáticamente con ese contrato. La revisión punto por punto está en el {download}`informe del capítulo 3 <../_static/e1/revision_estado_arte_20261103.md>` y su [matriz de requisitos](../_static/e1/revision_estado_arte_20261103.json).

La actualización de 3 y 3.1 incorpora cadenas nativas para WoS, Scopus, IEEE Xplore y arXiv, junto con cuatro intentos de acceso del 03/11/2026. Los cuatro fueron rechazados por el proxy antes de alcanzar las plataformas: **las búsquedas académicas y sus exportaciones siguen pendientes**. El corpus, sus frecuencias y las 35 referencias se conservan; un intento de acceso no añade estudios. La revisión se orienta a literatura reciente y a referencias comparativas sencillas; la preferencia Q1 solo se acreditará mediante índice, revista, año y categoría verificables.

## 3.1. Estrategia de Búsqueda (Obligatoria)

### 3.1.1. Cadena de búsqueda

La revisión pregunta qué arquitecturas DL sencillas se usan como referencias comparativas para pronosticar volatilidad de Bitcoin. Se separan tres unidades: **registro de descubrimiento**, **documento científico** y **familia de arquitectura**. Un repositorio puede localizar un documento, pero no acredita por sí solo publicación, revisión por pares o uso comparativo de todas sus clases de código.

El protocolo actual distingue **B1, búsqueda general DL**, **B2, sensibilidad sin exigir DL en los metadatos**, y cuatro búsquedas F opcionales por familias: feedforward, recurrentes, convolucionales y atención. Así se contemplan Bitcoin/BTC/BTCUSDT, variantes realized/realised y pronóstico/predicción sin decidir previamente las tres familias finales. No se excluyen automáticamente trabajos que también mencionen precio, GARCH o modelos híbridos.

Las cadenas B1 preparadas para cada fuente son las siguientes. **No se acredita su ejecución ni aceptación por el motor**: los intentos actuales se bloquearon antes de llegar a las plataformas. Campos, expansión de términos y filtros requieren confirmación en la respuesta real.

**Web of Science — TS: Topic (database-defined title, abstract and keyword fields)**

```text
TS=(Bitcoin OR BTC OR BTCUSDT)
AND TS=(volatil* OR "realized variance" OR "realised variance")
AND TS=(forecast* OR predict*)
AND TS=("deep learning" OR "neural network" OR "neural networks")
AND PY=(2019-2026)
```

**Scopus — TITLE-ABS-KEY**

```text
TITLE-ABS-KEY(Bitcoin OR BTC OR BTCUSDT)
AND TITLE-ABS-KEY(volatil* OR "realized variance" OR "realised variance")
AND TITLE-ABS-KEY(forecast* OR predict*)
AND TITLE-ABS-KEY("deep learning" OR "neural network" OR "neural networks")
AND PUBYEAR > 2018
AND PUBYEAR < 2027
```

**IEEE Xplore — All Metadata, Command Search**

```text
("All Metadata":Bitcoin OR "All Metadata":BTC OR "All Metadata":BTCUSDT)
AND ("All Metadata":volatil* OR "All Metadata":"realized variance" OR "All Metadata":"realised variance")
AND ("All Metadata":forecast* OR "All Metadata":predict*)
AND ("All Metadata":"deep learning" OR "All Metadata":"neural network" OR "All Metadata":"neural networks")
```

**arXiv — ti and abs, API/advanced search**

```text
(ti:Bitcoin OR abs:Bitcoin OR ti:BTC OR abs:BTC OR ti:BTCUSDT OR abs:BTCUSDT)
AND (ti:volatility OR abs:volatility OR ti:"realized volatility" OR abs:"realized volatility" OR ti:"realised volatility" OR abs:"realised volatility" OR ti:"realized variance" OR abs:"realized variance" OR ti:"realised variance" OR abs:"realised variance")
AND (ti:forecast OR abs:forecast OR ti:forecasting OR abs:forecasting OR ti:prediction OR abs:prediction OR ti:predicting OR abs:predicting OR ti:predictive OR abs:predictive)
AND (ti:"deep learning" OR abs:"deep learning" OR ti:"neural network" OR abs:"neural network" OR ti:"neural networks" OR abs:"neural networks")
AND submittedDate:[201901010000 TO 202611032359]
```

En IEEE Xplore se solicita además el filtro Publication Year 2019–2026; su aplicación no está verificada. En arXiv, submittedDate corresponde a envío, no a año editorial. Los comodines y campos no se presumen equivalentes entre motores.

Las **24 cadenas completas**, incluidas B2 y las dieciséis dirigidas opcionales, están en el {download}`protocolo de búsqueda <../_static/literature_e1/academic_search_20261103/protocol.md>` y el [plan JSON](../_static/literature_e1/academic_search_20261103/query_plan.json). Son consultas preparadas, no 24 búsquedas ejecutadas. El identificador B1 de consulta se distingue del cuartil Q1 de una revista.

(tabla-referencias-e1)=
### 3.1.2. Bases de datos consultadas

Las cuatro fuentes solicitadas tuvieron un intento de acceso actual. La tabla distingue ese intento de una consulta bibliográfica realizada; sus resultados permanecen desconocidos. El [registro de acceso](../_static/literature_e1/academic_search_20261103/access_attempts.json) y el {download}`informe legible <../_static/literature_e1/academic_search_20261103/access_report.md>` conservan fecha, ruta, cadena candidata y capa del fallo.

| Fuente | Estado del intento del 03/11/2026 | Conteo de resultados de búsqueda |
|---|---|---|
| Web of Science, Core Collection prevista | Portal de entrada bloqueado por el proxy antes de poder enviar la cadena | Desconocido (`null`), no cero |
| Scopus | Solicitud con cadena candidata bloqueada antes de alcanzar la plataforma | Desconocido (`null`), no cero |
| IEEE Xplore | Solicitud con cadena candidata bloqueada antes de alcanzar la plataforma | Desconocido (`null`), no cero |
| arXiv, complemento reciente | Solicitud a la API Atom con cadena candidata bloqueada antes de alcanzar la plataforma | Desconocido (`null`), no cero |

Los cuatro rechazos fueron **403 al establecer el túnel del proxy**; no son respuestas HTTP de los buscadores. No permiten atribuir el impedimento a una suscripción, CAPTCHA, clave API o autenticación. La configuración de red requerida quedó guardada como borrador pendiente de aplicar y publicar; eso no acredita acceso en la instancia activa.

El historial de acceso a portales se conserva como antecedente en las [comprobaciones registradas](../_static/literature_e1/access_checks_20261102_update.json), las [rutas preparadas](../_static/literature_e1/access_routes.csv) y el [intento institucional mediante Uninorte](../_static/literature_e1/access_uninorte_wos_20261102.json).

**Catálogo bibliográfico conservado.** El siguiente catálogo identifica las fuentes previas; esta revisión de acceso no acredita nuevas lecturas, resolución de DOI, indexación o cuartiles.

La tabla reúne las **35 referencias del documento** y organiza sus enlaces, la evidencia disponible y el aporte de cada una al proyecto. Comprende once aplicaciones a Bitcoin, ocho fundamentos de redes neuronales y dieciséis referencias de volatilidad, diagnóstico y evaluación. Se conservan las treinta referencias iniciales, las tres incorporadas para las fichas y los dos textos recuperados en esta revisión. El catálogo anterior de 33 entradas permanece como corte histórico.

El **enlace bibliográfico** identifica el trabajo, su DOI, su preprint o el registro donde se documentó la referencia. La columna **Fuente y evidencia consultada** indica qué material se utilizó realmente: texto completo de autores, código, documentación o una referencia localizada en otra bibliografía. Un DOI registrado no acredita su resolución directa ni la lectura del artículo original. Los manuscritos de Zeri y Gao conservan las precisiones sobre revisión por pares y año editorial recogidas en sus fichas.

**Aplicaciones a Bitcoin (11 referencias).**

| Referencia citada | Documento y enlace bibliográfico | Fuente y evidencia consultada | Aporte al proyecto |
|---|---|---|---|
| {cite:p}`huang2024` | *Forecasting Bitcoin volatility using machine learning techniques* · [DOI registrado](https://doi.org/10.1016/j.intfin.2024.102064) | [Antecedente del proyecto](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/246755355/1-s2.0-S1042443124001306-main.pdf): Antecedente bibliográfico previo; nueva lectura no acreditada. | Antecedente de aprendizaje automático para volatilidad de Bitcoin. |
| {cite:p}`akbiyik2023` | *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data* · [DOI registrado](https://doi.org/10.1145/3539597.3570387) | [Artículo y código de autores](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf): Texto completo y código de autores consultados. | Comparadores LSTM, GRU y TCN para volatilidad del día siguiente. |
| {cite:p}`guo2018` | *Bitcoin Volatility Forecasting with a Glimpse into Buy and Sell Orders* · [DOI registrado](https://doi.org/10.1109/ICDM.2018.00123) | [Repositorio de autores](https://github.com/weilai0980/mixture_models_crypto/blob/7aeaf66bb4438d022ae0cf2af2ffec317a8d068f/README.md): README y código inspeccionados; original no leído. | Información de órdenes de compra y venta como covariables alternativas. |
| {cite:p}`barjasic2021` | *Time-Varying Volatility in Bitcoin Market and Information Flow at Minute-Level Frequency* · [Registro en fuente consultada](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf) | [Bibliografía de Akbiyik et al.](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf): Referencia localizada en otra bibliografía; original no leído. | Relación entre volatilidad e información a frecuencia de minutos. |
| {cite:p}`katsiampa2017` | *Volatility estimation for Bitcoin: A comparison of GARCH models* · [Registro en fuente consultada](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf) | [Bibliografía de Akbiyik et al.](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf): Referencia localizada en otra bibliografía; original no leído. | Contexto de comparación econométrica mediante variantes GARCH. |
| {cite:p}`dyhrberg2016` | *Bitcoin, gold and the dollar – A GARCH volatility analysis* · [Registro en fuente consultada](https://raw.githubusercontent.com/ga0m1/bitcoin-volatility-models/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/M2259.docx) | [Bibliografía de Gao et al.](https://raw.githubusercontent.com/ga0m1/bitcoin-volatility-models/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/M2259.docx): Referencia localizada en otra bibliografía; original no leído. | Contexto de varianza condicional de Bitcoin, oro y dólar. |
| {cite:p}`herremans2025` | *Forecasting Bitcoin Volatility Spikes From Whale Transactions and CryptoQuant Data Using Synthesizer Transformer Models* · [Enlace bibliográfico](https://ieeexplore.ieee.org/document/11058926/) | [README y código de autores](https://raw.githubusercontent.com/dorienh/bitcoin_synthesizer/8570f4e8bdcfe1dd609e5f094d3d5b0059a810dc/README.md): Resumen, README y código de autores; original no leído. | Antecedente de atención con datos de ballenas y actividad on-chain. |
| {cite:p}`zeri2026` | *Forecasting High-Frequency Bitcoin Volatility: Neural Networks, Market Microstructure and On-Chain Data* · [Manuscrito](https://github.com/tommasozeri/Forecasting-High-Frequency-Bitcoin-Volatility-Neural-Networks/blob/d69f951f2ced7273bdc4c3e5a8a4cb220c2ee5ce/paper_en/manuscript.tex) | [Manuscrito y notebook del autor](https://github.com/tommasozeri/Forecasting-High-Frequency-Bitcoin-Volatility-Neural-Networks/blob/d69f951f2ced7273bdc4c3e5a8a4cb220c2ee5ce/paper_en/manuscript.tex): Texto completo y código; manuscrito sin revisión por pares acreditada. | Fuente aplicada de encoder Transformer; evidencia de literatura gris. |
| {cite:p}`gao2023` | *Comparing the Predictive Power of Five Models on Bitcoin Volatility* · [Manuscrito](https://github.com/ga0m1/bitcoin-volatility-models/blob/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/M2259.docx) | [Manuscrito y notebook de autores](https://github.com/ga0m1/bitcoin-volatility-models/blob/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/M2259.docx): Manuscrito y código consultados; año editorial pendiente de corroborar. | Fuente de una LSTM pequeña; objetivo y protocolo distintos del proyecto. |
| {cite:p}`wangnd` | *A comparative analysis of volatility predictions via EGARCH, Transformer and hybrid EGARCH-Transformer models: changes in forecasting ability after introducing sentiment factors* · [Documento de autores](https://github.com/cathyw111/bitcoin-volatility-forecasting/blob/61feeec81d695e19eac9c819be41d704e6279547/research-draft.pdf) | [Texto completo y procedencia](https://github.com/cathyw111/bitcoin-volatility-forecasting/blob/61feeec81d695e19eac9c819be41d704e6279547/research-draft.pdf): Texto completo de literatura gris; sin fecha editorial, DOI o revisión por pares acreditados. | Transformer puro con OHLCV diario como comparador de EGARCH e híbridos; arquitectura y resultados publicados en el borrador, con inconsistencias de evaluación explícitas. |
| {cite:p}`alakhouch2026` | *Bitcoin Volatility Forecasting: Econometric and Machine Learning Approaches Applied to Daily Price Data (July 2010–February 2024)* · [Documento de autores](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Repport/Asset-volatility-Forecasting.pdf) | [Texto completo y procedencia](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Repport/Asset-volatility-Forecasting.pdf): Informe académico y código consultados; sin DOI, revista o revisión por pares acreditados. | LSTM apilada como comparador de GARCH, RF y XGBoost para volatilidad diaria; aporta evidencia de uso con riesgos de fuga y discrepancias entre informe y código. |

**Fundamentos de redes neuronales (8 referencias).**

| Referencia citada | Documento y enlace bibliográfico | Fuente y evidencia consultada | Aporte al proyecto |
|---|---|---|---|
| {cite:p}`cybenko1989` | *Approximation by superpositions of a sigmoidal function* · [DOI registrado](https://doi.org/10.1007/BF02551274) | [D2L: bibliografía y MLP](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Capacidad de aproximación de redes; sin garantía de generalización. |
| {cite:p}`goodfellow2016` | *Deep learning* · [Sitio del libro](https://www.deeplearningbook.org/) | [Bibliografía de D2L](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Fundamentos de arquitecturas, regularización y optimización. |
| {cite:p}`hochreiter1997` | *Long short-term memory* · [DOI registrado](https://doi.org/10.1162/neco.1997.9.8.1735) | [D2L: bibliografía y LSTM](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Memoria recurrente y compuertas de la arquitectura LSTM. |
| {cite:p}`cho2014` | *Learning phrase representations using RNN encoder–decoder for statistical machine translation* · [Ficha del preprint](https://arxiv.org/abs/1406.1078) | [D2L: bibliografía y GRU](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Antecedente de GRU y codificador–decodificador recurrente. |
| {cite:p}`lecun1998` | *Gradient-based learning applied to document recognition* · [DOI registrado](https://doi.org/10.1109/5.726791) | [D2L y bibliografía académica](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Convolución y pesos compartidos como fundamento arquitectónico. |
| {cite:p}`bai2018` | *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling* · [Ficha del preprint](https://arxiv.org/abs/1803.01271) | [README de autores de TCN](https://raw.githubusercontent.com/locuslab/TCN/master/README.md): README y cita de autores; original no leído. | TCN y comparación general de convolución frente a recurrencia. |
| {cite:p}`kingma2014` | *Adam: A method for stochastic optimization* · [Ficha del preprint](https://arxiv.org/abs/1412.6980) | [D2L: bibliografía y Adam](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos y explicación en fuente secundaria; lectura del original no acreditada. | Optimización adaptativa mediante estimaciones de momentos. |
| {cite:p}`vaswani2017` | *Attention is all you need* · [Ficha del preprint](https://arxiv.org/abs/1706.03762) | [Bibliografía de D2L](https://raw.githubusercontent.com/d2l-ai/d2l-en/master/d2l.bib): Metadatos en bibliografía de D2L; original no leído. | Fundamento de atención y de la arquitectura Transformer. |

**Fundamentos de volatilidad, diagnóstico y evaluación (16 referencias).**

| Referencia citada | Documento y enlace bibliográfico | Fuente y evidencia consultada | Aporte al proyecto |
|---|---|---|---|
| {cite:p}`engle1982` | *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/tseries/0.10-54/man/garch.Rd) | [Documentación de tseries](https://raw.githubusercontent.com/cran/tseries/0.10-54/man/garch.Rd): Metadatos y documentación metodológica; original no leído. | Varianza condicional cambiante en el tiempo mediante ARCH. |
| {cite:p}`bollerslev1986` | *Generalized Autoregressive Conditional Heteroskedasticity* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/tseries/0.10-54/man/garch.Rd) | [Documentación de tseries](https://raw.githubusercontent.com/cran/tseries/0.10-54/man/garch.Rd): Metadatos y documentación metodológica; original no leído. | Referencia econométrica de heterocedasticidad condicional GARCH. |
| {cite:p}`andersen2003` | *Modeling and Forecasting Realized Volatility* · [Registro en fuente consultada](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf) | [Bibliografía de Akbiyik et al.](https://raw.githubusercontent.com/meakbiyik/ask-who-not-what/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf): Metadatos y documentación metodológica; original no leído. | Definición de volatilidad realizada y diferencia frente al objetivo diario. |
| {cite:p}`corsi2009` | *A Simple Approximate Long-Memory Model of Realized Volatility* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/highfrequency/master/man/HARmodel.Rd) | [Documentación de highfrequency](https://raw.githubusercontent.com/cran/highfrequency/master/man/HARmodel.Rd): Metadatos y documentación metodológica; original no leído. | Componentes históricos de volatilidad a distintas escalas (HAR-RV). |
| {cite:p}`diebold1995` | *Comparing Predictive Accuracy* · [Registro en fuente consultada](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/dm.test.Rd) | [Documentación de forecast](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/dm.test.Rd): Metadatos y documentación metodológica; original no leído. | Comparación predictiva mediante diferencias de pérdidas. |
| {cite:p}`harvey1997` | *Testing the Equality of Prediction Mean Squared Errors* · [Registro en fuente consultada](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/dm.test.Rd) | [Documentación de forecast](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/dm.test.Rd): Metadatos y documentación metodológica; original no leído. | Corrección de muestra pequeña para comparación predictiva. |
| {cite:p}`hyndman2006` | *Another Look at Measures of Forecast Accuracy* · [Registro en fuente consultada](https://raw.githubusercontent.com/robjhyndman/forecast/master/R/errors.R) | [Código y documentación de forecast](https://raw.githubusercontent.com/robjhyndman/forecast/master/R/errors.R): Metadatos y documentación metodológica; original no leído. | MASE y cautelas de las métricas porcentuales de error. |
| {cite:p}`bergmeir2018` | *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction* · [Registro en fuente consultada](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/CVar.Rd) | [Documentación de forecast](https://raw.githubusercontent.com/robjhyndman/forecast/master/man/CVar.Rd): Metadatos y documentación metodológica; original no leído. | Condiciones de validez de la validación cruzada temporal. |
| {cite:p}`holm1979` | *A Simple Sequentially Rejective Multiple Test Procedure* · [Registro en fuente consultada](https://raw.githubusercontent.com/wch/r-source/R-4-4-branch/src/library/stats/man/p.adjust.Rd) | [Documentación de R: p.adjust](https://raw.githubusercontent.com/wch/r-source/R-4-4-branch/src/library/stats/man/p.adjust.Rd): Metadatos y documentación metodológica; original no leído. | Corrección de comparaciones múltiples para una familia de contrastes. |
| {cite:p}`kunsch1989` | *The Jackknife and the Bootstrap for General Stationary Observations* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/boot/master/man/tsboot.Rd) | [Documentación de boot: tsboot](https://raw.githubusercontent.com/cran/boot/master/man/tsboot.Rd): Metadatos y documentación metodológica; original no leído. | Remuestreo de observaciones dependientes mediante bloques. |
| {cite:p}`politis1994` | *The Stationary Bootstrap* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/boot/master/man/tsboot.Rd) | [Documentación de boot: tsboot](https://raw.githubusercontent.com/cran/boot/master/man/tsboot.Rd): Metadatos y documentación metodológica; original no leído. | Bootstrap estacionario con longitud aleatoria de bloques. |
| {cite:p}`haddaway2022` | *PRISMA2020: An R Package and Shiny App for Producing PRISMA 2020-Compliant Flow Diagrams, with Interactivity for Optimised Digital Transparency and Open Synthesis* · [DOI registrado](https://doi.org/10.1002/cl2.1230) | [README de PRISMA2020](https://raw.githubusercontent.com/prisma-flowdiagram/PRISMA2020/master/README.md): README y cita de autores; original no leído. | Trazabilidad del flujo de búsqueda y selección bibliográfica. |
| {cite:p}`dickey1979` | *Distribution of the estimators for autoregressive time series with a unit root* · [Registro en fuente consultada](https://raw.githubusercontent.com/JimVaranelli/DFGLS/main/README.md) | [Documentación de implementación](https://raw.githubusercontent.com/JimVaranelli/DFGLS/main/README.md): Metadatos y documentación metodológica; original no leído. | Diagnóstico de raíz unitaria en series temporales. |
| {cite:p}`kwiatkowski1992` | *Testing the null hypothesis of stationarity against the alternative of a unit root* · [Registro en fuente consultada](https://raw.githubusercontent.com/cran/tseries/0.10-58/man/kpss.test.Rd) | [Documentación de tseries: KPSS](https://raw.githubusercontent.com/cran/tseries/0.10-58/man/kpss.test.Rd): Metadatos y documentación metodológica; original no leído. | Contraste de estacionariedad complementario al de raíz unitaria. |
| {cite:p}`ljung1978` | *On a measure of lack of fit in time series models* · [DOI registrado](https://doi.org/10.2307/2335207) | [Documentación de R: Box.test](https://raw.githubusercontent.com/wch/r-source/R-4-4-branch/src/library/stats/man/box.test.Rd): Metadatos y documentación metodológica; original no leído. | Diagnóstico de autocorrelación residual a rezagos declarados. |
| {cite:p}`broock1996` | *A test for independence based on the correlation dimension* · [DOI registrado](https://doi.org/10.1080/07474939608800353) | [Código de statsmodels: BDS](https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/tsa/_bds.py): Metadatos y documentación metodológica; original no leído. | Contraste i.i.d.; el rechazo no identifica por sí solo no linealidad. |

La tabla se puede descargar en [CSV](../_static/literature_e1/references_access_35.csv) y [JSON](../_static/literature_e1/references_access_35.json), con las fuentes y limitaciones de cada entrada. Las referencias completas se presentan en la [bibliografía del libro](referencias.md).

### 3.1.3. Periodo de búsqueda

El intervalo preferente preparado es **01/01/2019–03/11/2026**, ocho años calendario hasta la fecha de esta revisión. Es coherente con la preferencia de cinco a ocho años de la guía, sin afirmar ocho años completos de tiempo transcurrido. Los trabajos anteriores se justifican como antecedentes o fundamentos.

WoS/Scopus usan campos de publicación; IEEE requiere el filtro de años correspondiente y arXiv la fecha de envío indicada. Se registrarán fecha editorial, primera disponibilidad y versiones cuando existan. Los filtros solicitados siguen **sin aplicación verificada**, y una fecha de commit no sustituye al año de un documento.

**Periodo y alcance.** La búsqueda inicial del 02/11/2026 usa 2019–2026 como intervalo preferente, sin un filtro de fecha de publicación verificable en GitHub. La recuperación dirigida del 03/11/2026 añade S6 y S7; S6 no declara fecha editorial y no acredita recencia. Se informa la sensibilidad sin ese estudio. Los trabajos anteriores a 2019 son fundamentos justificados. El catálogo no equivale a 35 originales leídos ni a 35 estudios aplicados: el denominador de extracción ampliado es cinco.

### 3.1.4. Criterios de inclusión y exclusión

Los criterios siguientes se hicieron explícitos al revisar el corpus; no se presentan como prerregistrados antes de las búsquedas del 02/11/2026.

| Dimensión | Evidencia necesaria para extraer un estudio | Tratamiento cuando falta o no corresponde |
|---|---|---|
| Pertinencia | Bitcoin y objetivo de volatilidad o varianza; identificar mercado, frecuencia y horizonte | Precio, dirección o trading sin pronóstico de volatilidad y otros activos quedan fuera del alcance aplicado; una descripción ambigua permanece pendiente |
| Deep Learning comparativo | Modelo DL empleado como comparador independiente en un experimento identificable | Un modelo clásico, una capa de salida o una clase no invocada no acreditan una referencia DL publicada |
| Resultados verificables | Métrica, valor, escala y tabla/sección de origen; dispersión si se reporta | Sin resultados recuperados no se añaden cifras ni frecuencias verificadas |
| Arquitectura y protocolo | Información suficiente para identificar el tipo de referencia y su papel independiente; partición, objetivo y evaluación identificables | Distinguir falta de evidencia del comparador de hiperparámetros no reportados. No completar una configuración por preferencia personal; registrar lo no reportado y la búsqueda de arquitectura requerida por E2 |
| Identidad y publicación | Autoría, título y documento localizable; fecha declarada o desconocida de forma explícita; DOI si existe | Código o README sin texto suficiente es contexto. Un informe o borrador no se excluye solo por su género; la literatura gris se identifica y no acredita revisión por pares |
| Periodo | Preferencia 01/01/2019–03/11/2026 en el plan actual; filtros sin ejecución verificada | Los anteriores se justifican como antecedentes/fundamentos; los sin fecha no acreditan recencia y requieren sensibilidad. La actualización de un repositorio no sustituye al año del documento |

Se distinguen pertinencia, recuperabilidad y calidad. Que un texto esté incluido para extracción exploratoria no certifica su validez ni basta para sostener que respalda uno de los tres modelos básicos definitivos. La preferencia Q1 requiere verificar índice, revista, año y categoría; no se deduce de que exista un DOI o un repositorio, ni se traslada automáticamente a congresos.

Un estudio cuenta como máximo una vez por familia. Preprint, artículo y código del mismo estudio no deben duplicarse; la deduplicación bibliográfica entre bases deberá usar DOI y, cuando falte, título/autores/año. La deduplicación del flujo actual solo acredita repositorios repetidos en la búsqueda complementaria. El catálogo de trabajos citados y los cinco textos aplicados con extracción completa cumplen funciones distintas.

Para el nuevo flujo académico se normalizarán DOI; se agruparán identificadores arXiv sin versión y vínculos preprint–artículo comprobados; cuando falten identificadores se revisarán título, autores y año. Las coincidencias ambiguas se resolverán conservando el motivo. Esta deduplicación documental no se presenta como ejecutada mientras no existan exportaciones. La literatura gris se identifica y requiere sensibilidad; no se transforma en evidencia revisada por pares.

### 3.1.5. Trazabilidad del proceso

**Intento académico actual.** Se registran cuatro GET y ninguna consulta bibliográfica verificada. Los totales de coincidencias y registros de cada base son `null`; los accesos bloqueados no constituyen artículos identificados, excluidos ni una búsqueda válida con cero resultados. No se añade un diagrama académico con cifras inventadas. La bitácora actual permanece separada del flujo complementario histórico. El [manifiesto de los archivos de registro](../_static/literature_e1/academic_search_20261103/manifest.json) conserva sus SHA-256; no son hashes de exportaciones bibliográficas inexistentes.

Cuando exista acceso, se conservarán las exportaciones originales RIS/BibTeX/CSV/Atom y sus hashes antes de deduplicar. Se registrarán cadena literal, colección, filtros realmente aplicados, fecha/hora y zona, orden, páginas, tamaño de página, regla de parada y recuperación parcial. El flujo distinguirá registros identificados por fuente, duplicados, documentos cribados, textos evaluados, incluidos y motivos de exclusión.

**Consultas complementarias históricas de GitHub.**

El [registro histórico de consultas complementarias](../_static/literature_e1/search_log.json) conserva URL, campo, página, fecha con zona horaria, respuesta y hash. La tabla siguiente se reconstruyó desde ese registro; no es una nueva búsqueda. Todos sus intentos corresponden al **02/11/2026**; las horas de cada intento se conservan en el [resumen por cadena](../_static/literature_e1/query_summary_20261103.csv).

| ID | Consulta literal | Intentos / páginas recuperadas | Ocurrencias recuperadas | Repositorios distintos dentro de la consulta | Estimación del índice |
|---|---|---:|---:|---:|---:|
| G1 | `bitcoin volatility forecasting` | 3 / 3 | 30 | 30 | 148 |
| G2 | `cryptocurrency volatility deep learning` | 2 / 2 | 20 | 20 | 33 |
| G3 | `bitcoin realized volatility neural` | 2 / 2 | 1 | 1 | 1 |
| G4 | `bitcoin volatility MLP in:readme` | 1 / 1 | 10 | 10 | 241 |
| G5 | `bitcoin volatility feedforward in:readme` | 1 / 1 | 10 | 10 | 103 |
| G6 | `bitcoin volatility paper in:description` | 1 / 1 | 7 | 7 | 7 |
| G7 | `"bitcoin volatility" "neural network" in:readme` | 2 / 1 | 10 | 10 | 45 |
| G8 | `"realized volatility" "MLP" "Bitcoin" in:readme` | 2 / 1 | 10 | 10 | 24 |
| G9 | `"volatility" "ANN" "bitcoin" in:readme` | 2 / 1 | 10 | 10 | 192 |
| G10 | `"bitcoin volatility" "CNN" in:readme` | 2 / 1 | 10 | 10 | 21 |

La estimación del índice es la cifra mostrada por GitHub en las páginas recuperadas. **No se suman esas estimaciones**: las consultas se solapan. Las 118 ocurrencias efectivamente recuperadas se deduplican a 106 repositorios. Hubo cuatro intentos limitados por el servidor y sus reintentos están en el registro; no son cuatro cadenas adicionales. G3 incluye una segunda página vacía: cuenta como respuesta recuperada, no como un registro nuevo.

El orden fue la relevancia predeterminada y los límites variaron entre una y tres páginas por cadena. No se aplicó un filtro verificable por fecha de publicación, revisión por pares o cuartil. Las consultas dirigidas a MLP/ANN/CNN y las búsquedas truncadas por página introducen sesgo de descubrimiento; por ello, sus resultados no estiman prevalencia de arquitecturas en la literatura.

La [procedencia](../_static/literature_e1/provenance.json) registra la importación de comprobantes previos y distingue la fecha original de búsqueda de la revisión del 03/11/2026. Las comprobaciones de Guo y Herremans se documentan aparte. La recuperación dirigida de seis registros del 03/11/2026 amplía las inspecciones y los textos evaluados, pero no las cadenas ni las ocurrencias iniciales. Ninguna clase de código se convierte automáticamente en comparador publicado.

**Flujo histórico y recuperación dirigida.**

La identificación original corresponde al **02/11/2026**; la ampliación del **03/11/2026** recupera seis repositorios ya presentes en esos registros, sin ejecutar cadenas nuevas.

![Flujo acumulado de búsqueda complementaria y recuperación dirigida](../_static/literature_e1/flow.svg)

| Etapa | Corte 02/11/2026 | Corte acumulado 03/11/2026 | Unidad y límite |
|---|---:|---:|---|
| Ocurrencias recuperadas | 118 | 118 | Registros de repositorios en diez cadenas; no artículos |
| Ocurrencias duplicadas retiradas | 12 | 12 | Mismo repositorio recuperado varias veces |
| Repositorios distintos | 106 | 106 | No se añadieron registros de búsqueda |
| Repositorios inspeccionados | 14 | 20 | Seis recuperaciones dirigidas documentadas |
| Repositorios sin inspección | 92 | 86 | No equivalen a artículos completos excluidos |
| Textos completos evaluados | 5 | 7 | Dos documentos adicionales: S6 y S7 |
| Textos completos excluidos | 2 | 2 | Identidad/fecha insuficientes y estudio de NASDAQ100; el género «curso» no excluye por sí mismo |
| Textos completos incluidos | 3 | 5 | S1, S2, S5, S6 y S7; corpus exploratorio ampliado |
| Fuentes temáticas con texto completo pendiente | 2 | 2 | S3/S4 como contexto, sin frecuencia adicional |

Se reconcilian **118−12=106**, **20+86=106** y **7=5+2**. En la revisión local se corrigieron los motivos de los 92 registros inicialmente no inspeccionados: algunos sí mencionaban DL y volatilidad, por lo que no podían descartarse como ausencia de DL o como precio únicamente. Diecisiete quedaron marcados para recuperación; seis se inspeccionaron después y **once continúan pendientes sin inspección**. Los cuatro repositorios nuevos que no aportaron un documento identificable se conservan como contexto de software, sin inventar artículos ni frecuencias.

Se preservan el [cribado original](../_static/literature_e1/screening_20261102_original.csv), el [cribado vigente](../_static/literature_e1/screening.csv), la [corrección de motivos](../_static/literature_e1/screening_revision_20261103.json) y la [etapa de recuperación](../_static/literature_e1/screening_recovery_stage_20261103.json). Los [manifiestos de recuperación](../_static/literature_e1/directed_recovery_20261103/source_manifest.json) identifican URL fijada, commit, fecha y hash. El corte anterior de matrices y flujo permanece en `history_20261102/`. No se suman diapositivas, código y borrador del mismo trabajo como estudios distintos.

**Inclusión exploratoria posterior a la búsqueda.** S6 —Wang y colaboradores— es un borrador de cinco autores sin fecha de publicación acreditada; su commit de 2026 solo fecha la versión disponible. S7 —Alakhouch Idriss— es un informe académico fechado el 02/06/2026. Se retienen bajo los mismos criterios básicos que permitieron incluir literatura gris en S2 y evidencia con limitaciones en S5: objetivo, identidad, comparador, protocolo y resultados extraíbles. No se excluyen por ser borrador o informe ni se presentan como evidencia indexada, Q1 o revisada por pares. Sus riesgos severos exigen mostrar sensibilidad **sin S6/S7**, con el corpus previo S1/S2/S5, y **sin documentos de fecha no acreditada**. Incorporarlos amplía el corpus, pero no cierra la selección definitiva de los tres modelos básicos.

El cribado fue asistido por Codex para el autor responsable, Jassan Arteta. No se acredita segundo revisor humano, consenso previo ni acuerdo entre revisores; las decisiones y dudas quedan registradas con su evidencia y fecha. El flujo comunica una revisión complementaria, no exhaustividad PRISMA ni consultas en las bases que permanecen sin búsqueda acreditada.

La frase histórica «sin ejecutar cadenas nuevas» describe aquella recuperación de seis repositorios. El intento académico de esta actualización es otra fase y no altera 118 ocurrencias, 106 repositorios, 20 inspeccionados, siete textos evaluados, cinco incluidos ni 81 cifras publicadas por sus autores.

### 3.1.6. Evaluación de la calidad de los estudios

La evaluación usa dimensiones y ubicaciones de evidencia, sin una puntuación agregada arbitraria. Distingue lo declarado por el texto, lo comprobado en el código de autor y lo no recuperado; no certifica ausencia de fuga ni reproducción de resultados.

| Estudio | Arquitectura y protocolo | Repetición e inferencia | Riesgos y alcance |
|---|---|---|---|
| S1. Akbiyik et al. | Comparadores, entradas, tuning y código documentados | Veinte corridas y dispersión en Tabla 2 | Test de 48 días; verificar familia de contrastes y multiplicidad; inicializaciones no son mercados independientes |
| S2. Zeri | Declara escalado solo con entrenamiento, purga y embargo; redes y referencias no reciben igual historia | Tres semillas con predicciones promediadas y DM | Literatura gris; no confundir promedio de predicciones con dispersión entre corridas ni atribuir toda ganancia a arquitectura |
| S5. Gao et al. | LSTM pequeña y densa lineal descritas en notebook; año editorial externo pendiente | No se acreditan varias semillas ni intervalos | Fechas y objetivo desplazado requieren crítica; posible cruce de cortes sin purga explícita; Dense(1) sin capa oculta no es MLP no lineal |
| S6. Wang et al. (s. f.) | Transformer OHLCV independiente y nueve variantes; dos capas, d_model=32, cuatro cabezas; ancho FFN/readout no reportados | Una semilla declarada; DM expresamente no ejecutado | Fuente bruta/proxy y cohortes sin reconciliar; afirmación de mejor MAE contradice Tabla4; RMSE/R² no concuerdan bajo una cohorte común convencional; sin año de publicación ni código de entrenamiento |
| S7. Alakhouch Idriss (2026) | LSTM64→LSTM32→Dense16→Dense1 declarada; doce atributos en PDF frente a ocho en código | Sin varias semillas o dispersión acreditadas | Aparente fuga al ajustar escaladores antes del split; objetivo30 días/anualizado distinto; recuentos y early stopping del informe/código no reconciliados |

Se conserva la [valoración por dimensión](../_static/literature_e1/quality_assessment.csv) y su [JSON](../_static/literature_e1/quality_assessment.json). Ausencia de descripción no equivale a ausencia del defecto; una prueba publicada tampoco garantiza calibración, comparabilidad o control de multiplicidad. Los detalles y la repercusión sobre la selección se desarrollan en §§3.2–3.4.

La consulta de una base devuelve registros; la inclusión exige recuperar y valorar documentos. Para la nueva búsqueda se mantendrá la evaluación de arquitectura/comparador, protocolo, partición y fuga aparente, repeticiones, dispersión e inferencia. Indexación y cuartil se comprobarán por fuente, revista, año y categoría: no se deducen de un DOI ni se atribuyen automáticamente a congresos o preprints.

## 3.2. Extracción, frecuencia y selección de tres familias

### 3.2.1. Matriz de extracción de los estudios incluidos

El corpus exploratorio ampliado contiene **cinco textos completos**, S1, S2, S5, S6 y S7. Incluirlos permite extraer lo que sus autores describen; no certifica la validez de sus resultados. S6 y S7 son incorporaciones de la revisión dirigida del 03/11/2026 y conservan sus limitaciones editoriales, temporales y metodológicas.

| Estudio y referencia | Dataset, problema y objetivo | Referencias DL verificadas | Métricas y valores de autores | Límite de interpretación |
|---|---|---|---|---|
| S1. Akbiyik et al. (2023), Tabla 2 {cite:p}`akbiyik2023` | Bitcoin a 15 minutos y Twitter; RV del día siguiente agregada de 96 retornos | LSTM, GRU y TCN; D-TCN es la propuesta avanzada | LSTM RMSE 2,37±0,19; GRU 4,23±5,61; TCN 2,58±0,07; veinte corridas | RV×100; información, frecuencia y objetivo diferentes de E1 |
| S2. Zeri (2026), Run B {cite:p}`zeri2026` | BTCUSDT perpetuo, Crypto Lake; log-RV futura de 30 minutos | GRU, BiLSTM y encoder Transformer | SELF-ATT R² frente a media train 0,600/QLIKE 0,562; BiLSTM 0,586/0,582; GRU 0,582/0,564 | Tres semillas promediadas en predicciones; literatura gris; no es el R² convencional ni el objetivo diario Spot |
| S5. Gao et al. (manuscrito 2023), Tabla 4 {cite:p}`gao2023` | Yahoo BTC-USD diario; medida móvil de 30 días desplazada siete días | LSTM de una capa y H=20; la densa lineal es contexto | LSTM RMSE 0,074071 y RMSPE 0,406084 | Evaluación de validación sobre valores escalados; sin dispersión entre semillas |
| S6. Wang et al. (s. f.), Tabla 4 {cite:p}`wangnd` | Bitcoin diario; volatilidad diaria denominada RV, sin fórmula suficiente para equipararla a σ7 | Transformer OHLCV independiente; EGARCH–Transformer es híbrido | Modelo 1: RMSE 1,4738; MAE 0,4369; QLIKE 0,4369; R² −0,7347, transcritos tal como aparecen | Borrador sin fecha editorial; inconsistencias de MAE y particiones no reconciliadas; cifras no validadas por esta revisión |
| S7. Alakhouch Idriss (2026), Tabla 4.5 {cite:p}`alakhouch2026` | Bitcoin diario; medida futura de 30 días anualizada en el código | LSTM 64→32 con cabeza Dense16→1; no una MLP independiente | RMSE 0,0358; MAE 0,0254; MAPE 13,5 % | Informe académico; PDF declara 12 entradas y código 8; escalado global anterior al corte y recuentos inconsistentes |

La [matriz completa](../_static/literature_e1/evidence_matrix.csv) conserva problema, dataset, arquitectura, resultados y localización. El [registro de métricas](../_static/literature_e1/reported_metrics.csv) distingue los 42 valores del corte original y los 39 transcritos de S6/S7: **81 valores de autores**, no 81 resultados independientes ni experimentos propios. No se corrigen silenciosamente cifras impresas inconsistentes ni se trasladan a la escala de E1.

S3 —Herremans y Low— y S4 —Huang et al.— siguen como fuentes identificadas con texto completo pendiente. El código de autores de S3 no aumenta la frecuencia hasta verificar en el artículo qué redes se compararon. Las nuevas inspecciones de repositorios sin documento identificable también permanecen fuera del denominador.

### 3.2.2. Frecuencia, variantes y sensibilidad del corpus

La unidad es **un estudio por familia utilizada como referencia independiente**. No se cuentan la propuesta novedosa, una capa interna, una variante repetida del mismo experimento o el promedio de varias semillas como estudios adicionales.

| Familia | Estudios del corpus ampliado | Frecuencia ampliada | Corte original | Distinción necesaria |
|---|---|---:|---:|---|
| LSTM, incluyendo BiLSTM | S1, S2, S5, S7 | 4/5 | 3/3 | La variante unidireccional aparece en S1/S5/S7: 3/5 |
| GRU | S1, S2 | 2/5 | 2/3 | No es idéntica a LSTM aunque comparte el mecanismo recurrente con compuertas |
| Encoder Transformer | S2, S6 | 2/5 | 1/3 | Ambos apoyos son literatura gris; S6 no acredita fecha editorial |
| TCN | S1 | 1/5 | 1/3 | La TCN de una rama no equivale a D-TCN ni a toda CNN1D poco profunda |
| MLP independiente no lineal | Ninguno acreditado | 0/5 | 0/3 | La densa lineal de S5 y la cabeza Dense16 de S7 no son estudios de MLP independiente |
| CNN1D simple independiente | Ninguno acreditado | 0/5 | 0/3 | No se fusiona con TCN ni con una CNN–LSTM por compartir convoluciones |

Las [frecuencias](../_static/literature_e1/family_frequencies.csv), la [extracción por modelo](../_static/literature_e1/model_extraction_review_20261103.csv) y la [reconstrucción con sensibilidades](../_static/literature_e1/model_frequency_review_20261103.json) permiten revisar el denominador. Al retirar solo S6, cuya fecha es desconocida, quedan cuatro textos: LSTM 4/4, GRU 2/4, TCN 1/4 y encoder Transformer 1/4. Al retirar las dos incorporaciones se recupera exactamente el corte original. Estas proporciones describen el corpus accesible y sesgado por recuperación; no estiman prevalencia en WoS o Scopus.

La inspección previa de Guo encontró una clase MLP con ReLU, pero no acreditó su uso publicado como comparador. En Herremans, los nombres `dense` forman parte de Synthesizer Transformer. Ninguno de esos hallazgos incrementa el conteo de MLP autónoma.

### 3.2.3. Regla reproducible y estado de la selección

La guía exige recurrencia, sencillez, reproducibilidad y tres modelos no redundantes. Se aplica la siguiente regla de revisión documental, registrada **después** de inspeccionar las fuentes:

1. Exigir un texto identificable que muestre el uso independiente del comparador DL; separar propuestas avanzadas, híbridos, componentes internos y modelos lineales.
2. Registrar por variante entrada, salida, capas, ancho, regularización, pérdida, optimización y localización. Un campo no reportado permanece desconocido; no se completa por preferencia.
3. Ordenar por número de estudios y conservar las variantes: la recurrencia de una familia no acredita el uso de una configuración concreta en todos esos estudios.
4. Comparar profundidad y parámetros con sus hipótesis. Mostrar tanto el tamaño de origen como una representación común p=5, q=1; no utilizar RMSE de tareas distintas como desempate.
5. Revisar sesgo inductivo y suficiencia de fuentes. Si los tres primeros no satisfacen conjuntamente sencillez y no redundancia, declarar la selección pendiente y ampliar evidencia, en vez de omitir un candidato sin explicación.
6. Antes de implementar, cerrar exactamente tres arquitecturas con su respaldo. Cuando falta una arquitectura concreta, declarar en E2 su espacio de búsqueda antes de entrenar, conforme a la guía; esto no convierte un espacio supuesto en configuración publicada.

La corrección del apéndice C de S1 cambia la valoración de complejidad: sus ganadores **sí están publicados**. La tabla muestra cálculos de capas descritas, sin instanciar redes:

| Configuración de autor usada como ejemplo | Profundidad/ancho identificados | Parámetros con la tarea de origen | Parámetros derivados con p=5, q=1 |
|---|---|---:|---:|
| LSTM S5 | Una capa, H=20 y cabeza lineal | 1.781 | 2.101, un sesgo por compuerta |
| LSTM S1 | Una capa, H=261; Tabla 5 | 299.724 | 279.010, un sesgo por compuerta |
| GRU S1 | Una capa, H=43; Tabla 5 | 10.158 con dos sesgos | 6.494 con dos sesgos |
| TCN S1 | Seis niveles, F=287, k=5, base de dilatación 4; Tabla 6 | 4.563.396, condicionado al bloque de dos convoluciones y un stack | 4.542.924 bajo las mismas hipótesis |
| SELF-ATT S2 | Dos encoder, d=48, FF=96, cuatro cabezas y M=96 posiciones | 49.249 | 47.521, conservando M=96 |

Los [escenarios analíticos](../_static/literature_e1/model_complexity_review_20261103.csv) documentan fórmulas, sesgos, proyecciones y supuestos. El script `scripts/audit_literature_models.py` reproduce frecuencias y conteos leyendo la extracción; no ejecuta frameworks, entrenamiento ni datos de mercado. S6 no permite un total exacto porque faltan ancho feedforward y detalles de la cabeza/posiciones. En S7 se conservan por separado las hipótesis de 12 atributos del PDF y 8 del código.

Los conteos comunes controlan dimensiones de entrada/salida, pero no igualan capacidad, datos o cómputo. GRU tiene menos pesos que la LSTM H261 de S1, mientras la LSTM H20 de S5 es menor que esa GRU. La TCN ganadora no es una CNN1D pequeña: su campo receptivo condicional es 10.921 frente a 96 pasos observados. Reducirla a otra red sería una adaptación que debe justificarse, no una réplica del ganador.

| Estado | Familia | Justificación actual | Condición pendiente |
|---|---|---|---|
| M1, candidata de trabajo | LSTM unidireccional | Mayor recurrencia familiar y arquitectura pequeña identificable en S5 | Justificar variante y adaptación a OHLCV sin transferir resultados |
| M2, candidata de trabajo | TCN causal dilatada | Comparador independiente con filtros compartidos y dilatación | Justificar una configuración básica; verificar versión de biblioteca y distinguir réplica de simplificación |
| M3, candidata de trabajo | Encoder Transformer | Mecanismo de atención distinto, documentado en S2 y S6 | Corroborar suficiencia académica de fuentes grises y sencillez de la variante propia |
| Candidata que debe seguir evaluándose | GRU | 2/5 y menor complejidad que algunas variantes recurrentes | No descartarla solo porque LSTM también sea recurrente; justificar no redundancia y criterios del trío final |

El [registro de selección](../_static/literature_e1/selection_evidence.json) mantiene las tres fichas como **candidatas provisionales**, no como los tres básicos definitivamente extraídos. La frecuencia por sí sola ordena LSTM, luego GRU y Transformer empatados, y después TCN. La elección anterior de LSTM–TCN–Transformer priorizaba diversidad; no satisface por sí sola la regla conjunta de la guía. La revisión no presenta ese compromiso como una selección cerrada ni fija modelos nuevos sin respaldo.

### 3.2.4. Equivalencia funcional con referencias clásicas

En este problema continuo, el ejemplo clásico pertinente de la guía es **SVR lineal**. El modelo logístico corresponde a clasificación. La equivalencia se refiere al propósito de establecer una referencia de capacidad controlada; no a igualdad de funciones, supuestos o error:

| Referencia | Capacidad y supuestos | Papel comparativo |
|---|---|---|
| SVR lineal | Función afín sobre covariables vectorizadas, regularización y pérdida ε-insensible | Referencia clásica; no ocupa un puesto DL |
| LSTM | Estado recurrente, memoria y transformaciones no lineales con compuertas | Referencia DL secuencial si se controla su capacidad y presupuesto |
| TCN | Filtros causales compartidos y campo receptivo definido | Referencia DL convolucional si se justifica una variante básica |
| Encoder Transformer | Interacciones por contenido y posiciones en una ventana observable | Candidata DL de atención; su carácter mínimo no se presupone |

Su comparación exige información, partición, objetivo y esfuerzo de ajuste comunes. Persistencia, Ridge y GARCH aportan contexto y no sustituyen los tres modelos DL. Los cuatro Ridge del bloque C son diagnósticos de EDA; el benchmark propio de redes continúa reservado para E2.

## 3.3. Fichas documentales de las tres familias candidatas

Las tres familias siguientes representan memoria recurrente, convolución causal y atención. Su candidatura es **documental y acotada al corpus recuperado**; todavía no acredita la selección definitiva por recurrencia y mínima complejidad exigida por la guía. Se distinguen: arquitectura reportada por la fuente, cálculo analítico derivado de su descripción y adaptación que necesitaría el proyecto. La elaboración de estas fichas no ejecutó código de Deep Learning ni generó predicciones o resultados nuevos con datos de mercado; las referencias Ridge del EDA se documentan por separado en el capítulo 2.

Tras la recuperación dirigida, el corpus exploratorio contiene **cinco textos completos**: S1, S2, S5, S6 y S7. La **familia LSTM aparece en 4/5**, incluida la BiLSTM de S2; sus variantes unidireccionales están en S1/S5/S7, **3/5**. GRU aparece en **2/5**, TCN en **1/5** y encoder Transformer en **2/5**, S2/S6. S6 y S7 se incorporan como literatura gris con riesgos explícitos; una mayor cantidad de textos no acredita revisión por pares, mínima complejidad o comparabilidad de sus resultados. Las fichas concretas se mantienen apoyadas en LSTM de S5, TCN de S1 y SELF-ATT de S2. S6/S7 amplían evidencia de las familias, no definen réplicas idénticas de esas configuraciones. Una referencia general de arquitectura no cuenta como estudio aplicado adicional.

El contrato del proyecto conserva cinco covariables OHLCV diarias, entradas de D=7/14/21/28 días, volatilidad histórica de n=7 retornos diarios y horizonte h=1. Al inicio de t se conoce información hasta t−1 y se pronostica Y_t=σ_(t+1), que incorpora r_t. Este objetivo no es volatilidad realizada de siete días enteramente futuros. Ninguno de los cinco textos utiliza exactamente este contrato.

### 3.3.1. LSTM unidireccional

#### Referencias y papel comparativo

**S1.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023). *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*. WSDM 2023. DOI: [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387). El artículo compara LSTM, GRU y TCN antes de proponer D-TCN. La LSTM es un comparador y no el modelo novedoso del trabajo. Evidencia: PDF pp. 4–6, §§5.2–5.4, Fig. 1 y Tabla 2; código `scripts/hpo.py`, líneas 151–235.

**S5.** Gao, M., Li, Q. y Tao, Z. *Comparing the Predictive Power of Five Models on Bitcoin Volatility*. Manuscrito de 2023 y código del primer autor; referencia editorial declarada a Springer, *Learning and Analytics in Intelligent Systems*, vol. 32, pp. 473–486, DOI declarado [10.1007/978-3-031-38074-7_41](https://doi.org/10.1007/978-3-031-38074-7_41). Año editorial externo pendiente de corroboración; no se infiere de la fecha del DOCX. El texto compara media, random walk, GARCH, una red densa lineal y LSTM. La arquitectura sencilla de una capa y veinte unidades está descrita en Tabla 4 y en la celda 111 del [notebook fijado](https://github.com/ga0m1/bitcoin-volatility-models/blob/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/BTC_USD.ipynb).

#### Arquitecturas efectivamente descritas

S5 utiliza catorce observaciones de volatilidad como entrada, una LSTM de veinte unidades y una salida Dense(1). No incorpora una segunda capa recurrente. El notebook compila con MSE y Adam, batch 64, máximo 200 épocas, paciencia 20 y restauración de pesos elegidos según `val_rmspe`. El callback de checkpoint monitoriza `loss`, un criterio distinto: esta diferencia debe conservarse al interpretar el procedimiento y no se sustituye por un protocolo supuesto. No se reportan varias semillas ni dispersión entre corridas.

S1 utiliza un día de 96 retornos de quince minutos como entrada y produce 96 retornos del día siguiente mediante una cabeza densa. Su búsqueda varía unidades recurrentes entre 32 y 512, dropout entre 0 y 0,5, learning rate entre 10⁻⁷ y 10⁻², weight decay entre 10⁻⁹ y 10⁻² y ε entre 10⁻² y 10⁻¹. Usa AdamW y pérdida cuadrática ε-insensible, 250 configuraciones por red, treinta épocas y veinte corridas para la comparación. El apéndice C, Tabla 5, sí reporta la configuración ganadora de S1: H=261, dropout 0,0237, ε=0,0364, learning rate 0,00521 y weight decay 9,25×10⁻⁷. La revisión anterior omitía esta tabla; se corrige esa extracción. Estos valores de autores no son una configuración del proyecto.

#### Apoyo adicional de la familia LSTM: S7

**S7.** Alakhouch Idriss (2026). *Bitcoin Volatility Forecasting: Econometric and Machine Learning Approaches Applied to Daily Price Data (July 2010–February 2024)*. Informe fechado 02/06/2026, sin DOI ni revisión por pares acreditados. [Informe y versión de autor](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Repport/Asset-volatility-Forecasting.pdf), §3.5.3 y Tabla 4.5, p. 34 impresa; [código](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Python/Python.py), líneas 366–415.

El informe compara una LSTM independiente con GARCH, Random Forest y XGBoost, y presenta una propuesta EGARCH+XGBoost. Declara ventana de 60 días, dos capas LSTM de 64/32 unidades, dropout 0,2, Dense(16, ReLU) y salida lineal escalar; usa Adam 10⁻³, batch 32, máximo 200 épocas y paciencia 15. Es una variante más profunda que la ficha concreta de S5. El PDF declara doce atributos, mientras el código enumera ocho, incluida varianza condicional EGARCH; no se trata de las cinco OHLCV del proyecto. El objetivo del código es desviación móvil futura de treinta retornos, anualizada, no σ7 diario.

Tabla 4.5 reporta LSTM RMSE 0,0358, MAE 0,0254 y MAPE 13,5 %, sin dispersión entre semillas; son valores del autor, no verificados mediante reproducción. El código ajusta MinMaxScaler de entradas y objetivo sobre todo `df_garch` antes del corte temporal, riesgo aparente de fuga de información. También monitoriza `val_loss`, frente a MAE descrita en el informe, y los recuentos declarados 3.849+1.006=4.855 no reconcilian el total 4.955. Estas discrepancias impiden usar la fuente para acreditar superioridad o una réplica exacta y no se corrigen inventando resultados o dimensiones. Se cuenta una sola vez como apoyo exploratorio de familia.

#### Ecuaciones, flujo y sesgo inductivo

Para x_j∈R^p, h_j,c_j∈R^H y activación logística s:

$$
i_j=s(W_ix_j+U_ih_{j-1}+b_i),\quad
f_j=s(W_fx_j+U_fh_{j-1}+b_f),\quad
o_j=s(W_ox_j+U_oh_{j-1}+b_o),
$$

$$
g_j=\tanh(W_gx_j+U_gh_{j-1}+b_g),\quad
c_j=f_j\odot c_{j-1}+i_j\odot g_j,\quad
h_j=o_j\odot\tanh(c_j),\quad
\widehat y=W_yh_D+b_y.
$$

```text
Ventana histórica D × p
       ↓ lectura secuencial con parámetros compartidos
LSTM: compuertas i/f/o + memoria c + estado h
       ↓ último estado H
Cabeza densa H → q
```

Las compuertas controlan el almacenamiento y olvido del estado; su memoria ofrece un mecanismo diferente al de filtros locales o atención directa. Esto no garantiza aprender dependencias largas en una muestra pequeña ni acredita superioridad predictiva. El orden de los datos y la disponibilidad de las entradas siguen siendo parte del contrato causal.

#### Parámetros y costo analítico

Una LSTM con un vector de sesgo por compuerta y cabeza de q salidas tiene:

$$
P_{\mathrm{LSTM}}=4H(p+H+1)+Hq+q.
$$

En una implementación con dos vectores de sesgo por compuerta, como la parametrización de `torch.nn.LSTM`, la expresión es:

$$
P_{\mathrm{LSTM,2bias}}=4H(p+H+2)+Hq+q.
$$

Para la configuración de S5, p=1,H=20,q=1, la primera expresión da **1.781 parámetros**, cálculo analítico de las capas descritas y no un total reportado por el artículo. Una adaptación con p=5,H=20,q=1 daría **2.101** con un sesgo o **2.181** con dos. Estos últimos conteos son hipótesis de adaptación, no tamaños de un modelo del proyecto ya implementado. Cambiar D altera la longitud del cómputo, pero no el número de pesos de una LSTM con salida tomada del último estado. Con p=1, H=261 y q=96 de S1, la fórmula de un sesgo produce **299.724 parámetros**; es un cálculo derivado de la configuración ganadora de Tabla 5, no un total textual ni una medición ejecutada. La diferencia frente a S5 refleja configuraciones y tareas distintas.

Por ventana, el costo dominante de inferencia es O(D·H(p+H)+Hq). Un minibatch de B ventanas multiplica ese costo por B; la retropropagación mantiene el mismo orden, con mayor constante y almacenamiento de estados O(BDH). La recurrencia impone dependencia secuencial entre pasos. No se infiere tiempo real ni consumo de GPU a partir de esta expresión.

#### Resultados publicados y límites de comparabilidad

| Fuente y evaluación | Resultado publicado | Dispersión y escala |
|---|---|---|
| S1, Tabla 2, test de 48 días | LSTM RMSE 2,37; MAE 1,70; MAPE 0,30; MSLE 0,09 | DE 0,19 / 0,26 / 0,08 / 0,01, respectivamente, entre veinte corridas; RV = 100·sqrt(sum(r_15min²)); MAPE como fracción |
| S5, Tabla 4, comparación de validación | LSTM(20) RMSE 0,074071; RMSPE 0,406084 | Sin dispersión entre semillas; comparación sobre valores escalados y objetivo de treinta días desplazado siete días |

No se ordenan estos RMSE como si midieran el mismo problema. S1 predice retornos intradiarios y después agrega volatilidad; S5 emplea volatilidad pasada como entrada y no centra los retornos al construir su medida. El contrato del proyecto conserva OHLCV diario y desviación poblacional centrada de siete retornos. La pérdida MSE fijada para el diseño del proyecto coincide nominalmente con S5, pero no con la pérdida ε-insensible de S1; que una pérdida comparta nombre tampoco iguala escalas y objetivos.

**Fortalezas:** comparador recurrente documentado en tres fuentes unidireccionales, con variantes distintas, arquitectura pequeña completamente identificable en S5 y número de parámetros independiente de D. **Límites:** S5 contiene fechas inconsistentes y carece de repeticiones e inferencia; S1 dispone de veinte corridas, pero solo 48 días de test. La variante, transformación, activación de salida y configuración propia deben distinguirse de la réplica de cada fuente. No se atribuye a las fuentes una salida `softplus`: sus cabezas descritas son lineales.

### 3.3.2. TCN: convolución temporal causal y dilatada

#### Referencias y papel comparativo

S1 documenta una **TCN única** como comparador de LSTM, GRU, AR-RV y GARCH. Luego propone una **D-TCN de dos ramas**, una para retornos y otra para señales de Twitter. La ficha selecciona la TCN comparativa, no la D-TCN novedosa. La base conceptual de convoluciones causales y dilatadas está descrita por Bai, Kolter y Koltun (2018), referencia de fundamentos del libro; ese trabajo genérico no se cuenta como otro estudio de Bitcoin.

Fuentes: S1, Fig. 1, §§5.2–5.4 y Tabla 2; [hpo.py por commit](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/scripts/hpo.py), líneas 151–235. `dtcn.py` corresponde a la extensión propuesta y no debe utilizarse como si describiera la base de una sola rama.

#### Arquitectura y entrenamiento publicados

El código invoca `TCN` de `keras-tcn` con seis niveles de dilatación `(a⁰,a¹,a²,a³,a⁴,a⁵)`, a∈{2,3,4}, kernel k∈{2,…,6}, canales F∈{32,…,512}, dropout en [0,0,5], skip connections activadas o no y alternativas de normalización batch/weight/layer/ninguna. Después se aplica Dense(96) y reshape a 96×1 retornos futuros. Entrada fuente: 96×1 retornos observados. El artículo compara modelos con información financiera de precios antes de introducir las ramas de Twitter.

La selección utiliza 250 configuraciones, treinta épocas por configuración, AdamW y pérdida cuadrática ε-insensible sobre retornos. Los rangos de learning rate, weight decay y ε son los mismos de la ficha LSTM. Los veinte reinicios de la comparación final se evalúan después de agregar los 96 retornos predichos a RV diaria. El apéndice C, Tabla 6 del artículo, sí publica los hiperparámetros ganadores: F=287 filtros, kernel k=5, base de dilatación a=4, dropout 0,217, ε=0,0913, learning rate 6,49×10⁻⁵, weight decay 5,93×10⁻⁶, skip connections activadas y ninguna normalización. Se corrige la omisión de esa tabla en la extracción anterior. Sigue sin estar fijada la versión de `keras-tcn`, por lo que los detalles internos de sus bloques deben verificarse antes de una réplica.

#### Ecuaciones y diagrama

Para una convolución causal con dilatación d_b:

$$
u_{j,f}=b_f+\sum_{i=0}^{k-1}\sum_{c=1}^{C_{in}}W_{i,c,f}\,x_{j-d_b i,c}.
$$

Solo intervienen índices anteriores o iguales a j; el padding es a la izquierda. En un bloque residual representativo de dos convoluciones:

$$
H^{(b)}=\phi\!\left(F_b(H^{(b-1)})+S_b(H^{(b-1)})\right),
$$

donde F_b representa convoluciones, activaciones, dropout y la normalización que se declare, y S_b es identidad o proyección 1×1 para igualar canales. La representación temporal final alimenta la cabeza densa.

```text
Ventana pasada D × p
       ↓ convoluciones 1D causales
Bloques residuales con dilataciones crecientes
       ↓ representación temporal F
Cabeza densa F → q
```

El sesgo inductivo es localidad temporal con pesos compartidos y contexto ampliado mediante dilataciones. Se diferencia de almacenar una memoria recurrente y de relacionar todas las posiciones mediante atención. Atender una ventana pasada no elimina el requisito de que las variables originales y sus transformaciones fueran observables en cada origen.

#### Parámetros, profundidad y complejidad: hipótesis explícitas

La evidencia directa fija **seis niveles de dilatación**. El número de convoluciones por bloque depende de la implementación residual de la biblioteca. El esquema habitual de dos convoluciones por bloque daría doce convoluciones, pero este conteo **no aparece como total textual del artículo ni sustituye la versión ausente de la dependencia**.

Una convolución con sesgo aporta k·C_in·C_out+C_out parámetros. Para J bloques de dos convoluciones, ancho constante F, p≠F, una sola proyección residual inicial p→F, sin normalización entrenable y una cabeza F→q, se deriva:

$$
P_{\mathrm{TCN}}=(k+1)pF+(2J-1)kF^2+(2J+1)F+Fq+q.
$$

El campo receptivo de ese esquema es:

$$
R=1+2(k-1)\sum_{b=0}^{J-1}d_b.
$$

Con los hiperparámetros ganadores F=287, k=5, base a=4, J=6, p=1 y q=96, la fórmula bajo esas hipótesis da **4.563.396 parámetros** y un campo receptivo formal **R=10.921**. Son cálculos condicionados a dos convoluciones por bloque y un stack, no totales publicados ni mediciones de una implementación. R excede los 96 retornos observados: no crea historia disponible adicional.

Ejemplo **analítico dentro del espacio descrito**, no configuración ganadora: J=6,F=32,k=2,p=1,q=96, sin normalización entrenable, da 26.208 parámetros y R=127 para dilataciones 1/2/4/8/16/32. Una adaptación p=5,q=1 bajo esas mismas hipótesis daría 23.457 parámetros. La normalización, otros stacks o una estructura residual distinta cambian el conteo. No se presentan estos valores como observados en un entrenamiento.

El costo dominante por ventana es O(D·Σ_l k_l C_(l−1)C_l+Fq), aproximadamente O(DJkF²) en la parte de ancho constante. Para minibatch B, se multiplica por B. El entrenamiento añade almacenamiento de activaciones y retropropagación del mismo orden. Las operaciones pueden paralelizarse dentro de la ventana; esto no demuestra menor latencia real que LSTM. Si D es de 7–28 días, un campo receptivo formal mayor que D no crea historia observada adicional: parte del contexto correspondería a padding.

#### Resultados publicados, fortalezas y limitaciones

S1, Tabla 2, informa **TCN RMSE 2,58±0,07; MAE 1,63±0,06; MAPE 0,23±0,01; MSLE 0,11±0,01**, veinte corridas, test de 48 días y la misma escala RV de la ficha LSTM. En ese estudio, TCN tiene mejor MAPE/MAE que LSTM, mientras LSTM tiene menor RMSE/MSLE. El orden depende de la métrica; no se comunica un ganador universal. Los contrastes del artículo son unilaterales respecto a AR-RV y no implican automáticamente un contraste directo entre las redes ni control de toda la multiplicidad.

**Fortalezas:** comparador separado de la propuesta híbrida, filtros compartidos y contexto multiescala con cálculo paralelo. **Límites:** frecuencia 1/5 en el corpus exploratorio, detalles internos dependientes de una biblioteca sin versión fijada y test corto. La fuente respalda TCN y no acredita una CNN1D de una sola capa como réplica exacta. «Base comparativa» no significa mínima profundidad absoluta.

Para el proyecto, pasar de retornos intradiarios a cinco OHLCV diarios y de 96 salidas a una volatilidad escalar sería una adaptación. Su profundidad, dilataciones, normalización y salida deben registrarse antes de implementar en E2. Si falta detalle de arquitectura, corresponde consultar el protocolo de E2 §14 citado por la guía; esta ficha no inventa sus reglas.

### 3.3.3. Transformer encoder compacto, comparador SELF-ATT

#### Referencia, estado y papel experimental

**S2.** Zeri, T. (octubre de 2026). *Forecasting High-Frequency Bitcoin Volatility: Neural Networks, Market Microstructure and On-Chain Data*. Manuscrito público de investigador independiente, sin DOI ni revisión por pares acreditados. [Fuente TeX fijada](https://github.com/tommasozeri/Forecasting-High-Frequency-Bitcoin-Volatility-Neural-Networks/blob/d69f951f2ced7273bdc4c3e5a8a4cb220c2ee5ce/paper_en/manuscript.tex), §§Models/Training, líneas 179–189, tabla de resultados y apéndice, líneas 395–409. Arquitectura leída en celda 21 del notebook `modelli_btc_hf.ipynb`, mismo commit.

SELF-ATT es uno de los tres comparadores neuronales genéricos —junto a GRU y BiLSTM— del estudio; la contribución se centra en pipeline, comparación y ablación de atributos, no en declarar una nueva arquitectura de atención. La ficha concreta describe S2. El encoder aparece además en S6, con una configuración distinta: **2/5 textos del corpus exploratorio**, ambos sin revisión por pares acreditada; esto no prueba recurrencia en toda la literatura. Es compacto frente a grandes Transformers, pero tiene más parámetros que la LSTM simple de S5; no se le atribuye mínima complejidad absoluta.

#### Arquitectura y protocolo documentados

Run B utiliza 48 barras de cinco minutos y 41 atributos de Binance Futures BTCUSDT perpetuo mediante Crypto Lake. El objetivo es log-RV de los siguientes treinta minutos, seis barras, a partir del precio medio del libro de órdenes; no es desviación móvil diaria de siete retornos. Hay 31.524 predicciones de test, correspondientes al 13/09–31/12/2025. La tabla publica métricas de predicciones promediadas entre tres semillas, no media y desviación de tres métricas.

La arquitectura contiene proyección lineal F→d, posiciones aprendidas de longitud máxima M, dos bloques encoder pre-LayerNorm y cuatro cabezas. Cada bloque incluye atención multihead, dos conexiones residuales y feedforward de ancho 2d. El código no especifica la activación del encoder, por lo que usa ReLU por defecto; la cabeza sí usa GELU. La cabeza concatena último token y media de tokens y aplica dropout→Dense(2d,d)→GELU→Dense(d,1).

La configuración elegida reportada para SELF-ATT en Run B es d=48, dropout 0,21, learning rate 3·10⁻⁴, weight decay 1,4·10⁻³, ruido de entrada 0,12 y ventana L=48. El código conserva M=96 posiciones, aunque la ventana elegida sea menor. La pérdida es MSE sobre el objetivo log-RV estandarizado, según la línea 119 del manuscrito y `loss_fn` en la celda 23. Se declara AdamW, batch 512, máximo cuarenta épocas, paciencia seis, clipping 1 y semillas 42/43/44; cada red recibe veinte trials de búsqueda. Las predicciones finales se promedian entre semillas antes del cálculo de métricas.

#### Apoyo adicional de encoder Transformer: S6

**S6.** Wang, Y., Yu, Q., Peng, J., Yang, B. y Chen, Z. (s.f.). *A comparative analysis of volatility predictions via EGARCH, Transformer and hybrid EGARCH-Transformer models: changes in forecasting ability after introducing sentiment factors*. [Borrador de autores](https://github.com/cathyw111/bitcoin-volatility-forecasting/blob/61feeec81d695e19eac9c819be41d704e6279547/research-draft.pdf), §3.4 y Tablas 1–2, pp. 6–7, y Tabla 4, p. 10. La versión del repositorio está disponible en 2026; la fecha de commit no se presenta como año editorial. No se acreditan DOI ni revisión por pares.

Su comparador `model1` es un encoder Transformer sobre OHLCV diario de Bitcoin, con historia de 20 días, d=32, cuatro cabezas, dos capas, dropout 0,1 y salida Softplus. Declara MSE, Adam 0,001, batch 32, máximo 100 épocas, paciencia 10 y semilla 42. El periodo es 25/03/2021–30/12/2024 y el objetivo declarado RV1D, cuya construcción exacta no quedó suficientemente especificada. Faltan ancho feedforward, pooling/lectura, orden de normalización y versión de implementación; por ello no se inventa un total de parámetros ni se traslada el conteo 49.249 de S2 a S6.

Tabla 4 informa para `model1` RMSE 1,4738, R² −0,7347, QLIKE 0,4369 y MAE 0,4369, sin dispersión. Se conservan literalmente, sin tratarlos como cifras validadas: el texto llama mejor al MAE 0,7399 de `model9` aunque 0,4369 es menor; la combinación RMSE/R² no permite reconciliar una cohorte común con la definición convencional, y el corte 70/30 de Transformer no está reconciliado con los 203 días de test EGARCH. No se ejecutó DM por costo computacional, según §6.2. Sin datos o código de entrenamiento no puede verificarse disponibilidad, escalado ni el origen de esas diferencias. La fuente respalda la presencia de un comparador de atención con información OHLCV, pero no acredita una mejora reproducida ni cierra su especificación.

#### Ecuaciones, flujo y causalidad

Las ecuaciones siguientes describen **SELF-ATT de S2**, con d=48 y la cabeza GELU documentada. La configuración parcial de S6, d=32 y salida Softplus, permanece diferenciada y no se completa con componentes supuestos de S2.

$$
H^{(0)}=XW_{in}+b_{in}+P,\qquad
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Con d=48 y cuatro cabezas, d_k=12 por cabeza. Esta notación desagrega el espacio de atención; el manuscrito usa un símbolo d genérico. Los bloques son pre-LN con suma residual tras atención y tras feedforward. La salida usa:

$$
z=\left[H^{(2)}_L;\,\frac1L\sum_{j=1}^{L}H^{(2)}_j\right],\qquad
\widehat y=w_2^\top\operatorname{GELU}\!\left(W_1\operatorname{Dropout}(z)+b_1\right)+b_2.
$$

```text
Ventana histórica L × F
       ↓ proyección a d + posiciones aprendidas
Encoder pre-LN ×2: atención 4 cabezas → residual → FF(2d) → residual
       ↓ concatenar último token y media
Dropout → Dense(2d,d) → GELU → Dense(d,1)
```

![Flujo documental del encoder Transformer sobre una ventana pasada](../_static/literature_e1/diagrama_transformer.svg)

El sesgo inductivo permite relacionar posiciones distantes mediante atención dependiente del contenido, sin recurrencia ni una vecindad local fija. El encoder se llama sin máscara causal: es compatible con pronóstico si toda su ventana está disponible antes del origen. Atender posiciones posteriores dentro de una ventana enteramente pasada no implica por sí solo utilizar el futuro. La construcción causal de cada atributo debe verificarse por separado.

#### Complejidad y número de parámetros derivado

Para feedforward m=2d, F atributos y M posiciones aprendidas, cada bloque encoder aporta:

$$
P_{encoder}=4d^2+2dm+9d+m=8d^2+11d.
$$

La proyección aporta (F+1)d; posiciones, Md; y cabeza, 2d²+2d+1. Para dos bloques:

$$
P_{\mathrm{SELF-ATT}}=18d^2+(F+M+25)d+1.
$$

F=41,M=96,d=48 da **49.249 parámetros**: 2.016 de proyección, 4.608 de posiciones, 37.920 de encoder y 4.705 de cabeza. Es un cálculo analítico de capas leídas, **no un total reportado por el manuscrito ni una medición por instanciación**. Cambiar F a cinco o modificar el máximo de posiciones cambia el conteo; no se presenta 49.249 como el tamaño del futuro modelo OHLCV del proyecto.

El costo por bloque y minibatch B es O(B(L²d+Ld²)), más proyección O(BLFd) y cabeza O(Bd²). El almacenamiento de atención es cuadrático en L por cabeza. La retropropagación conserva el orden con mayor constante. Estas expresiones no prueban eficiencia ni escalabilidad medida frente a otras redes.

#### Valores publicados y crítica

| Modelo de S2, Run B | R² respecto a media train | R² respecto a media test | QLIKE | Spearman |
|---|---:|---:|---:|---:|
| SELF-ATT | 0,600 | 0,566 | 0,562 | 0,749 |
| BiLSTM | 0,586 | 0,550 | 0,582 | 0,742 |
| GRU | 0,582 | 0,546 | 0,564 | 0,743 |
| HAR | 0,508 | 0,466 | 0,773 | 0,676 |

El manuscrito reporta DM de SELF-ATT frente a HAR de 19,7, p unilateral <10⁻¹⁵ y HAC de cinco rezagos por solapamiento. No es un contraste directo SELF-ATT contra BiLSTM ni una prueba ejecutada por este proyecto. R² respecto a media train no equivale al R² convencional respecto a media test. No hay en esa tabla desviación ni IC entre las tres semillas: se evalúa su promedio de predicciones.

**Fortalezas:** mecanismo distinto, encoder/cabeza descritos y configuración acotada; relaciones directas entre posiciones de la ventana. **Límites:** los dos apoyos aplicados de encoder son literatura gris; S6 tiene especificación parcial y métricas inconsistentes. La ficha concreta de S2 utiliza datos licenciados, una ventana de evaluación y costo cuadrático, con resultados no reproducidos aquí. Además, las redes utilizan 24–48 barras mientras los referentes lineal y LightGBM usan solo la última: la comparación no aísla completamente arquitectura de historia informativa. El desglose de atributos de libro tiene una discrepancia de enumeración; se conserva F=41 como dimensión declarada, con esa salvedad.

La fuente tiene salida lineal para log-RV estandarizado. Adaptarla a OHLCV diario y volatilidad no negativa requiere un contrato propio; añadir `softplus` o reducir capas no sería una descripción de la arquitectura publicada. No se fijan esas modificaciones a partir de cifras de otro objetivo.

### 3.3.4. Comparación documental y cobertura de las fichas

| Dimensión | LSTM | TCN | Transformer encoder |
|---|---|---|---|
| Mecanismo | Estado recurrente y memoria con compuertas | Filtros causales compartidos y dilatados | Atención por contenido entre tokens de la ventana |
| Fuente de arquitectura | S5 concreta; S1 comparativa y ganadora en Tabla 5; S7 apoyo distinto | S1 comparativa y ganadora en Tabla 6 | S2 concreta; S6 apoyo parcial distinto, ambos grises |
| Frecuencia del corpus exploratorio | Familia 4/5; unidireccional 3/5 | 1/5 | 2/5, ambas fuentes grises |
| Profundidad identificada | S5 una capa, H20; S1 H261 | Seis niveles, F287, k5 y base4; convoluciones por bloque dependen de biblioteca no fijada | Dos encoder, cuatro cabezas |
| Parámetros fuente | 1.781 derivados para S5; 299.724 para S1 con un sesgo | 4.563.396 condicionados al bloque residual de dos convoluciones y un stack | 49.249 derivados para Run B |
| Datos/objetivo idénticos a E1 | No | No | No |
| Configuración OHLCV propia congelada | No; adaptación distinguida | No; falta variante y protocolo aplicable | No; cambio de frecuencia, dimensión y salida distinguido |

Las fichas aportan evidencia, ecuaciones, diagramas, resultados publicados y análisis de complejidad, con los límites y detalles no verificados identificados. No completan una búsqueda de hiperparámetros propia ni convierten una fuente en treinta estudios. La diversidad de mecanismos no acredita por sí sola mayor recurrencia y mínima complejidad; la selección definitiva sigue pendiente. Si el criterio académico exige exclusivamente estudios revisados por pares, S2/S6 no acreditan todavía el respaldo del encoder y S7 tampoco cumple ese filtro. La extracción definitiva de tres familias permanece abierta.

## 3.4. Comparación crítica de las familias de trabajo

### 3.4.1. Desempeño publicado y comparabilidad

S1 permite comparar LSTM, GRU y TCN dentro de un protocolo: LSTM tiene menor RMSE puntual que TCN, mientras TCN tiene menor MAE. El tuning selecciona por MAPE; ningún orden aislado acredita superioridad en todas las métricas. S5 aporta validación en una escala diferente. S2 evalúa log-RV intradiaria y promedia predicciones entre semillas. Sus métricas no forman un ranking común con las de S1/S5.

S6 y S7 amplían la evidencia de uso, pero **no mejoran automáticamente su calidad**. S6 publica una afirmación sobre el mejor MAE que contradice su propia Tabla 4; además, no se reconciliaron las particiones ni las definiciones de las métricas. S7 presenta discrepancias de atributos, muestras y selección de pesos entre informe y código, además de un escalado global previo al corte. Sus cifras se conservan como resultados declarados, sin validarlas ni usarlas para decidir el trío final.

**Brecha cuantificada entre una propuesta avanzada y su referencia.** S1, Tabla 4 (PDF p. 8), permite examinar las cuatro variantes D-TCN con un único grupo de información social, manteniendo también las que empeoran:

| Modelo de S1 | RMSE media ± DE | Reducción relativa puntual de RMSE frente a TCN | MAE | MAPE |
|---|---:|---:|---:|---:|
| TCN de una rama | 2,58 ± 0,07 | Referencia | 1,63 | 0,23 |
| D-TCN User | 2,36 ± 0,05 | 8,53 % | 1,44 | 0,20 |
| D-TCN Tweet | 2,45 ± 0,09 | 5,04 % | 1,59 | 0,21 |
| D-TCN VADER | 2,70 ± 0,09 | −4,65 % | 1,72 | 0,26 |
| D-TCN Count | 3,27 ± 0,47 | −26,74 % | 1,97 | 0,32 |

La reducción se deriva como `100·(RMSE_TCN−RMSE_variante)/RMSE_TCN` a partir de medias redondeadas de autores; un valor negativo indica empeoramiento. Es una transformación descriptiva, sin nuevo experimento, intervalo ni contraste pareado. D-TCN usa cuarenta reinicios frente a los veinte del comparador; la fila TCN de Tabla 4 repite los resultados base. Cambian conjuntamente arquitectura e información social y no se repite una búsqueda independiente para cada variante. Las brechas no aíslan el efecto de complejidad ni calibran el margen del 5 % adoptado para E1 frente a persistencia.

El proyecto conserva OHLCV diario de Binance Spot, desviación poblacional centrada de siete retornos y horizonte de un día. S1 agrega retornos intradiarios; S2 usa un perpetuo y log-RV de treinta minutos; S5 desplaza siete días una medida de treinta; S6 no permite equiparar su RV a nuestro objetivo y S7 construye una medida futura de treinta días anualizada. Ninguna cifra anterior es un resultado propio con ese contrato.

### 3.4.2. Robustez, calidad y fuentes de sesgo

| Dimensión | LSTM | TCN | Encoder Transformer |
|---|---|---|---|
| Uso documentado | Familia en S1/S2/S5/S7; S2 es BiLSTM | S1 | S2 y S6, literatura gris |
| Repeticiones | S1: veinte corridas; S5/S7 sin dispersión entre semillas acreditada | S1: veinte corridas | S2: promedio de predicciones de tres semillas; S6 una semilla declarada |
| Riesgo de partición | S5: posible cruce de etiquetas futuras; S7: escalado global anterior al corte | Solo 48 días de test en S1; no prueban robustez entre mercados | S2: información histórica distinta de referencias lineales; S6: particiones y selección de EGARCH no reconciliadas |
| Inferencia publicada | No confundir pruebas de otros modelos de S7 con evidencia sobre LSTM | Contrastes unilaterales contra AR-RV, no automáticamente entre redes | S2: DM contra HAR, no contra todas las redes; S6 declara DM no ejecutado |
| Conclusión permitida | Referencia recurrente documentada con calidad heterogénea | Comparador convolucional documentado, sin configuración mínima acreditada | Comparador de atención documentado; respaldo editorial y métricas requieren cautela |

Las veinte inicializaciones de S1 no son veinte mercados independientes y su test de 48 días limita la generalización. La selección por una métrica y los contrastes unilaterales requieren una familia de hipótesis explícita. En S5, el objetivo desplazado usa información posterior a la fila sin una purga identificada en el corte: se registra riesgo aparente, sin afirmar una reproducción que no se ejecutó.

En S2, las redes usan ventanas de 24/48 barras mientras ciertas referencias ven solo la última barra; la ganancia no aísla exclusivamente no linealidad. Sus tres predicciones promediadas tampoco aportan una desviación entre métricas de semillas. En S6, la fecha de una versión del repositorio no prueba año editorial; las incongruencias de la tabla no deben corregirse suponiendo una fórmula alternativa. En S7, la función DM del código compara otros modelos y no trata adecuadamente la dependencia temporal del objetivo móvil; no valida por sí sola la LSTM ni las cifras del informe.

Se conserva la evaluación por dimensión, con evidencia y desconocidos, sin una puntuación inventada. Disponer de código no certifica ausencia de fuga, y un artículo incluido con riesgo alto no constituye respaldo suficiente para adoptar su configuración o su desempeño como baseline.

### 3.4.3. Escalabilidad y simplicidad

| Aspecto | LSTM | TCN | Encoder Transformer |
|---|---|---|---|
| Cómputo dominante por ventana | O(D·H(p+H)) para una capa | O(D·Σ k_l C_(l−1)C_l) | O(D²d+Dd²) por encoder |
| Ejecución temporal | Dependencia secuencial entre estados | Filtros paralelizables en la ventana | Interacciones entre pares de posiciones |
| Memoria de activaciones en entrenamiento | Crece con D y H | Crece con D, ancho y profundidad | Atención completa añade almacenamiento cuadrático en D |
| Evidencia de sencillez | S5 identifica una LSTM pequeña; S1 y S7 son mayores | El ganador S1 es ancho y de seis niveles; no equivale a CNN1D poco profunda | Dos encoder documentados; menos pesos que la TCN bajo las hipótesis, sin garantizar menor tiempo |
| Medición comparable propia de tiempo/GPU/RAM | No ejecutada | No ejecutada | No ejecutada |

Las fórmulas y los conteos de §3.2.3/§3.3 controlan sus hipótesis, pero no miden eficiencia. Entrenamiento e inferencia dependen también de implementación, hardware, batch, precisión y presupuesto. Las épocas descritas por S2 no equivalen a tiempos de reloj comparables. Los tamaños de ventanas y salidas de las fuentes son distintos; el conteo con p=5,q=1 no valida esos modelos bajo OHLCV diario.

### 3.4.4. Brecha verificable y proyección

**Brecha limitada a los cinco textos incluidos:** ninguno compara conjuntamente LSTM, TCN y encoder Transformer con el mismo OHLCV diario de Binance Spot, desviación poblacional de siete retornos, horizonte de un día, protocolo cronológico común y tratamiento explícito de incertidumbre y multiplicidad. La afirmación se contrasta en la matriz; no demuestra una ausencia universal ni originalidad mundial. Tampoco convierte esas tres candidatas en la selección definitiva que sigue pendiente.

La contribución evaluable sería establecer referencias DL de capacidad controlada, bajo la misma información y presupuesto, y cuantificar qué mejora debe superar un modelo avanzado. Antes de ejecutarla hace falta cerrar el trío con evidencia suficiente y registrar sus configuraciones. Se mantienen RMSE, el margen operativo del 5 % frente a persistencia y la protección del test. No se promete que una red supere esa referencia.

### 3.4.5. Correspondencia y estado de cierre

| Requisito de la guía | Evidencia de esta revisión | Estado y límite |
|---|---|---|
| Cadenas completas, fechas y resultados | Diez consultas literales, intentos, páginas, ocurrencias y hashes | Trazable para la vía complementaria; no equivale a consulta de índices científicos |
| WoS, Scopus, IEEE y arXiv complementario | Rutas e intentos diferenciados de consultas ejecutadas | Cobertura bibliográfica obligatoria pendiente; no se asignan conteos cero ni cuartiles no verificados |
| Inclusión, exclusión, flujo y responsable | Criterios retrospectivos, cribado histórico y revisado; 20 repositorios inspeccionados, siete textos evaluados y cinco incluidos | Revisión exploratoria asistida; no exhaustividad ni segundo revisor humano acreditados |
| Matriz y evaluación de calidad | Cinco textos, resultados con ubicación y riesgos por dimensión | La inclusión no certifica validez; dos incorporaciones y S2 son literatura gris |
| Frecuencia y complejidad | Variantes, frecuencias ampliadas/sensibilidades, ganadores de S1 y conteos analíticos reproducibles | Corregido; no demuestra mínima complejidad global ni uso frecuente en todos los índices |
| Exactamente tres modelos básicos no redundantes | Tres fichas candidatas y GRU mantenida en evaluación | Selección final pendiente; no se descarta GRU por preferencia ni se adopta TCN grande como básica automáticamente |
| Fichas técnicas completas | Referencias, ecuaciones, diagramas, objetivos, pérdidas, entrenamiento, parámetros y costos | Ganadores fuente recuperados; versión de biblioteca y adaptación propia aún requieren cierre |
| Comparación crítica y brecha | Desempeño separado por protocolo, robustez, escalabilidad y brecha avanzada/base cuantificada | Documentada sin ranking entre tareas, inferencia propia ni afirmación de novedad universal |
| Experimentos y baseline medido | Diseño futuro y separación del diagnóstico Ridge de EDA | Tres redes, piloto, benchmark y test siguen reservados para E2 |

## 3.5. Fundamentos bibliográficos y contexto de la revisión

### 3.5.1. Búsqueda documental efectuada y alcance de las citas

La consulta del **2 de noviembre de 2026** incorporó treinta referencias: siete sobre Bitcoin, siete de fundamentos de redes neuronales y dieciséis de volatilidad, diagnóstico y evaluación. Se consultaron repositorios públicos de autores, documentación de software académico y bibliografías de artículos y libros. Para cada entrada se conservan el enlace consultado, los campos corroborados y el nivel de acceso en el [registro de verificación](../_static/bibliografia/verification_30.csv) y el [catálogo detallado](../_static/bibliografia/references_30.json).

El catálogo vigente conserva las treinta referencias iniciales y las tres fuentes de las fichas, y añade S6/S7 recuperadas el 03/11/2026: 35 referencias en la tabla de §3.1.2 y su [versión CSV](../_static/literature_e1/references_access_35.csv). La nueva etapa es recuperación dirigida de registros ya localizados, no una búsqueda ejecutada en WoS o Scopus.

La red del entorno bloqueó el acceso directo a Crossref y arXiv. Por ello no se afirma haber consultado sus interfaces ni leído treinta originales completos. La entrada de Huang y colaboradores conserva su verificación previa; su acceso directo actual también está limitado. Las verificaciones nuevas de las otras referencias proceden de las fuentes accesibles indicadas en el registro. Una referencia identificada en otra bibliografía se distingue de un artículo original leído.

Los trabajos anteriores a 2019 se incluyen como fundamentos y antecedentes expresamente justificados. No se cuentan automáticamente como estudios recientes elegibles para calcular frecuencia de modelos base. Los registros de acceso a metadatos tampoco sustituyen el flujo complementario documentado en §3.1.

### 3.5.2. Antecedentes de volatilidad de Bitcoin

Katsiampa estudia la estimación de volatilidad de Bitcoin mediante especificaciones GARCH, mientras Dyhrberg relaciona su análisis de volatilidad con oro y dólar {cite:p}`katsiampa2017,dyhrberg2016`. Estos antecedentes justifican contrastar las redes con referencias financieras y delimitar la variable que se estima; no demuestran por sí solos qué arquitectura neuronal debe seleccionarse.

Los estudios asociados a órdenes de compra y venta, información de Twitter y flujos de información a frecuencia de minuto plantean conjuntos de información diferentes {cite:p}`guo2018,akbiyik2023,barjasic2021`. Sus títulos, datos documentados y referencias permiten reconocer líneas de investigación pertinentes, pero no autorizan atribuir a OHLCV diario mejoras obtenidas con otras señales. El artículo de Akbiyik y colaboradores se recuperó como copia de autores; para parte de los antecedentes se corroboraron metadatos en sus bibliografías y no se declara lectura del original.

Huang y colaboradores aportan un antecedente de pronóstico de volatilidad con técnicas de aprendizaje automático. Herremans y Low se centran en picos de volatilidad e información de transacciones de grandes tenedores y CryptoQuant mediante modelos Synthesizer Transformer {cite:p}`huang2024,herremans2025`. Pronosticar picos, una medida realizada intradiaria o el siguiente valor de una desviación móvil diaria constituye tareas diferentes. La pertinencia temática no acredita igualdad de objetivo, datos, horizonte o protocolo.

### 3.5.3. Definiciones de volatilidad y referencias financieras

ARCH formaliza varianza condicional cambiante y GARCH amplía esa estructura incorporando dinámica de la varianza {cite:p}`engle1982,bollerslev1986`. Son fundamentos econométricos y referencias de contexto; no ocupan los puestos reservados a modelos de Deep Learning.

La literatura de volatilidad realizada y de modelos HAR estudia construcciones y dependencias a distintas escalas {cite:p}`andersen2003,corsi2009`. Es necesario conservar la distinción entre varianza condicional, volatilidad realizada construida con alta frecuencia y desviación estándar móvil de retornos diarios. La medida definida en este proyecto no se convierte en cualquiera de las anteriores por compartir la palabra «volatilidad». Cambiar ventana, frecuencia o anualización también cambia la escala de las métricas.

### 3.5.4. Fundamentos de las familias neuronales candidatas

El resultado de aproximación de redes feedforward y la exposición general de aprendizaje profundo fundamentan el uso de MLP como familia candidata {cite:p}`cybenko1989,goodfellow2016`. La capacidad de aproximar funciones no garantiza generalización ni acredita que una MLP sea una referencia frecuente para Bitcoin. Esa frecuencia requiere evidencia en estudios aplicados incluidos.

LSTM introduce un mecanismo de memoria con compuertas; el trabajo de Cho y colaboradores documenta una arquitectura recurrente con compuertas en un codificador–decodificador {cite:p}`hochreiter1997,cho2014`. Estos fundamentos permiten describir LSTM y GRU y discutir sus diferencias de parametrización. El artículo de Cho procede de traducción automática: no se presenta como un experimento de volatilidad financiera.

Las redes convolucionales de reconocimiento documental y la comparación genérica de modelos convolucionales y recurrentes para secuencias sustentan mecanismos de filtros locales y dependencias temporales {cite:p}`lecun1998,bai2018`. Una CNN de imágenes, una CNN1D simple y una TCN no son arquitecturas idénticas. Su parentesco permite formular candidatos, pero no afirmar una réplica exacta ni trasladar un ranking a BTCUSDT.

Adam aporta un procedimiento de optimización que debe separarse de la arquitectura y de su evidencia de desempeño {cite:p}`kingma2014`. Se cita aquí la prepublicación identificada de 2014; no se cambia silenciosamente al año de una versión de conferencia. El optimizador, la capacidad y el presupuesto de búsqueda deben describirse al comparar estudios, sin deducir superioridad de una familia solo por emplearlos.

### 3.5.5. Diagnóstico temporal e interpretación del EDA

Dickey–Fuller y KPSS utilizan hipótesis nulas diferentes: raíz unitaria y estacionariedad, respectivamente {cite:p}`dickey1979,kwiatkowski1992`. Esto respalda su lectura conjunta. Un resultado no significativo no confirma su hipótesis nula, y el conflicto entre diagnósticos requiere revisar especificación, muestra y posibles cambios de régimen.

Ljung–Box examina autocorrelación y BDS evalúa una hipótesis de independencia e idéntica distribución mediante la dimensión de correlación {cite:p}`ljung1978,broock1996`. Rechazar BDS no identifica de forma exclusiva no linealidad aprovechable. Tampoco un diagnóstico aplicado a retornos sustituye un análisis de residuos de modelos. Estas referencias fundamentan la interpretación metodológica del EDA y la evaluación prevista, sin aportar resultados experimentales propios.

### 3.5.6. Métricas, validación e incertidumbre

Hyndman y Koehler analizan medidas de precisión y fundamentan el uso de errores escalados como MASE {cite:p}`hyndman2006`. Una comparación reproducible debe declarar unidades, agregación, referencia ingenua y tratamiento de denominadores nulos. MAPE cerca de cero, RMSE y métricas escaladas no responden exactamente a la misma pregunta.

La validez de la validación cruzada para predicción autorregresiva depende de los supuestos y de la estructura de dependencia {cite:p}`bergmeir2018`. Esta referencia no se interpreta como autorización general para barajar series financieras. La comparación de estudios debe comprobar el reloj de disponibilidad de entradas, etiquetas y transformaciones.

Diebold–Mariano estudia diferencias de precisión predictiva mediante pérdidas, y Harvey, Leybourne y Newbold proponen una modificación para muestras pequeñas {cite:p}`diebold1995,harvey1997`. La corrección no resuelve automáticamente selección adaptativa, no estacionariedad o cualquier dependencia. Estas condiciones deben conservarse al interpretar resultados publicados y al formular el protocolo posterior.

El bootstrap por bloques para observaciones estacionarias y el stationary bootstrap proporcionan procedimientos de remuestreo para datos dependientes {cite:p}`kunsch1989,politis1994`. Son métodos diferentes: bloques de longitud fija y bloques de longitud aleatoria no deben presentarse como una implementación idéntica. Se necesita justificar longitud, unidad y supuestos; disponer de muchas fechas no basta para tratarlas como réplicas independientes.

Holm aporta un procedimiento secuencial para comparaciones múltiples {cite:p}`holm1979`. La familia de hipótesis debe declararse con las métricas, ventanas y contrastes que sustenten afirmaciones; añadir referencias no elimina ese requisito. Este punto forma parte de la revisión del diseño y no constituye una ejecución de contrastes de modelos.

### 3.5.7. Trazabilidad de búsqueda y conclusión de la revisión

La herramienta descrita por Haddaway y colaboradores facilita producir diagramas de flujo PRISMA 2020 {cite:p}`haddaway2022`. Se cita como apoyo para documentar el proceso, sin atribuirle la declaración original ni afirmar cumplimiento de una revisión sistemática solo por dibujar un flujo. Los conteos de búsquedas, duplicados y exclusiones deben corresponder a registros reales.

Las 35 referencias amplían el soporte documental del Entregable 1 y permiten distinguir problema financiero, arquitectura y método de evaluación. No equivalen a 35 experimentos ni a 35 estudios elegibles de modelos base de Bitcoin. Los textos de fundamentos no se suman a una frecuencia de uso en el problema aplicado. Las cinco fuentes completas se distinguen por calidad y procedencia. Las fichas y la comparación de las tres candidatas están documentadas en §§3.3–3.4. Permanecen abiertos la cobertura bibliográfica, el orden de los candidatos por frecuencia y complejidad y el respaldo de las configuraciones propias de E2. La cantidad de citas no acredita por sí sola la identificación definitiva exigida por la guía.
