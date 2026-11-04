# 3. Revisión del Estado del Arte y Extracción de los Tres Modelos Base

La revisión debe identificar exactamente tres modelos DL básicos usados como referencias comparativas en la literatura de volatilidad de Bitcoin. Este capítulo reúne **35 referencias bibliográficas y un corpus exploratorio de cinco textos completos**; son unidades distintas. La propuesta del proyecto sigue la guía 6.4: **MLP, LSTM y CNN1D poco profunda**, con GRU como alternativa recurrente. Las fichas de LSTM, TCN y encoder Transformer se conservan como perfiles documentales de las fuentes; no son el trío propuesto. La extracción aplicada y las configuraciones finales de MLP/CNN1D todavía requieren respaldo.

La revisión del **03/11/2026** corrigió motivos de cribado, recuperó dos textos adicionales y completó la extracción de los hiperparámetros ganadores de S1, que sí aparecen en su apéndice. Se conservan los cortes anteriores y se explicitan fuentes grises, parámetros desconocidos y límites de las métricas. No se entrenaron redes ni se consultaron datos de test del proyecto.

El problema propio es **regresión temporal, tipo 4, ruta 6.4**: OHLCV rezagadas sin historia explícita de la volatilidad objetivo. Las tareas de las fuentes no coinciden automáticamente con ese contrato. La revisión punto por punto está en el {download}`informe del capítulo 3 <../_static/e1/revision_estado_arte_20261103.md>` y su [matriz de requisitos](../_static/e1/revision_estado_arte_20261103.json).

La estrategia actual sigue la solicitud del usuario: **búsqueda web pública y verificación posterior de procedencia e indexación**. Se ejecutaron doce consultas públicas, se recuperaron fuentes de autores y se contrastaron cinco revistas por título e ISSN en copias públicas secundarias de listas WoS/Scopus. Los identificadores corroborados, los enlaces declarados y la indexación individual pendiente se distinguen expresamente. Esta fase conserva las 35 referencias y el corpus de cinco textos; no acredita búsquedas internas en esas bases ni completa por sí sola la extracción de los tres modelos.

La numeración mantiene los seis incisos de búsqueda (§3.1.1–3.1.6). La identificación de modelos (§3.2) separa procedimiento, criterios, síntesis, equivalencia, arquitecturas incompletas y familias orientativas. Cada ficha de §3.3 presenta los nueve campos de la guía; §3.4 distingue comparación, validez, brecha de desempeño y brecha de investigación.

## 3.1. Estrategia de Búsqueda (Obligatoria)

### 3.1.1. Cadena de búsqueda

#### 3.1.1.1. Pregunta y unidades de búsqueda

La pregunta es qué arquitecturas DL sencillas se utilizan como comparadores independientes para pronosticar volatilidad de Bitcoin. Primero se localizan documentos, repositorios de autores y recursos públicos; después se comprueba identidad, publicación y posible indexación. **Descubrimiento web, documento científico y familia de arquitectura son unidades distintas.**

#### 3.1.1.2. Cadenas de descubrimiento ejecutadas

Las consultas se ejecutaron en el buscador web público de GitHub, con `in:readme`, primera página y relevancia predeterminada. No se utilizó Google ni se presenta este recorrido acotado como una búsqueda exhaustiva de Internet o una consulta interna en WoS/Scopus. Las cuatro cadenas de descubrimiento fueron:

| ID | Cadena literal ejecutada | Total mostrado por GitHub | Registros recuperados |
|---|---|---:|---:|
| W1 | `bitcoin volatility forecasting in:readme` | 4577 | 10 |
| W2 | `bitcoin realized volatility neural in:readme` | 245 | 10 |
| W3 | `bitcoin realised volatility neural in:readme` | 45 | 10 |
| W4 | `"bitcoin volatility" arxiv in:readme` | 10 | 10 |

#### 3.1.1.3. Consultas dirigidas y antecedentes

Hubo además cuatro consultas de títulos/identificadores y cuatro de recursos de indexación. Las **doce cadenas**, URLs, fecha/hora, página y hashes de respuesta están en la [bitácora pública](../_static/literature_e1/public_web_verification_20261103/search_log.json). Se recuperaron 40 ocurrencias de repositorios y 36 repositorios distintos en el descubrimiento; esas cifras no son artículos, estudios incluidos ni frecuencias de arquitecturas. No se suman los totales del índice de consultas solapadas.

Las 24 cadenas nativas preparadas anteriormente se conservan como [antecedente no ejecutado](../_static/literature_e1/academic_search_20261103/query_plan.json). El método aplicado en esta fase es recuperación pública más comprobación posterior; no se cambia retrospectivamente el nombre del motor o sus campos para simular una búsqueda dentro de las bases.

### 3.1.2. Fuentes consultadas y verificación de procedencia e indexación

**WoS y Scopus son índices**, IEEE Xplore es una biblioteca digital y arXiv un repositorio de preprints. Un documento puede estar asociado a varias fuentes a la vez. Se verifica por separado: identidad del trabajo; publicación o identificador; presencia de la revista en un listado; registro individual del artículo; y cuartil del año/categoría. Un DOI, el nombre de una editorial o un resultado web no prueban todos esos niveles.

#### 3.1.2.1. Documentos y enlaces públicos comprobados

| Trabajo | Fuente pública comprobada | Hallazgo | Alcance de la comprobación |
|---|---|---|---|
| Akbiyik et al. (2023) | [PDF de autores](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf), portada | arXiv **2110.14317v2**, DOI 10.1145/3539597.3570387 y ACM WSDM 2023 | Identidad e identificador corroborados en copia pública de autores; registro arXiv en vivo no consultado. |
| Herremans y Low (2025) | [README de autores](https://github.com/dorienh/bitcoin_synthesizer/blob/8570f4e8bdcfe1dd609e5f094d3d5b0059a810dc/README.md#L3-L7) | IEEE Access 13, 117788–117807; [registro Xplore 11058926](https://ieeexplore.ieee.org/document/11058926/) | Enlace de registro declarado por autores; la página de Xplore no se abrió. |
| Guo et al. (2018) | [README de autores](https://github.com/weilai0980/mixture_models_crypto/blob/7aeaf66bb4438d022ae0cf2af2ffec317a8d068f/README.md#L7-L15) | IEEE ICDM 2018, pp. 989–994 | Publicación declarada por autores; registro Xplore específico no corroborado en el material nuevo. |

#### 3.1.2.2. Revistas: coincidencias en copias públicas de listas WoS/Scopus

Se inspeccionó [wosaide/wosaide-journal-lists](https://github.com/wosaide/wosaide-journal-lists/tree/4d0b1a32915b04180c37e6f697a7da04a6f05de6), repositorio independiente que declara copias WoS de agosto de 2026 y Scopus de julio de 2026. No está avalado por sus proveedores. Los títulos e ISSN de estas cinco revistas coinciden en ambas listas:

| Referencia | Revista declarada e ISSN | Colección WoS en la copia de 2026 | SourceID y cobertura Scopus declarada |
|---|---|---|---|
| Huang et al. (2024) | Journal of International Financial Markets, Institutions and Money · ISSN 1042-4431 | SSCI | 17531 · 1997-2026 |
| Barjašić y Antulov-Fantulin (2021) | Frontiers in Physics · ISSN 2296-424X | SCIE | 21100831025 · 2013-2026 |
| Katsiampa (2017) | Economics Letters · ISSN 0165-1765 | SSCI | 28490 · 1978-2026 |
| Dyhrberg (2016) | Finance Research Letters · ISSN 1544-6123 | SSCI | 15519 · 2004-2026 |
| Herremans y Low (2025) | IEEE Access · ISSN 2169-3536 | SCIE | 21100374601 · 2013-2026 |

Esta es **evidencia secundaria de revista**. No confirma el registro individual de cada artículo, la cobertura histórica WoS del año editorial ni un cuartil Q1. Las fuentes de tipo congreso de Akbiyik/Guo no se evalúan con esta lista limitada a revistas; su ausencia allí no significa que no estén indexadas. Las fuentes grises tampoco se declaran no indexadas por falta de prueba.

La [matriz de las once referencias aplicadas](../_static/literature_e1/public_web_verification_20261103/document_platform_matrix.json) conserva estados por documento; los [extractos de filas, ISSN, cobertura y fuentes](../_static/literature_e1/public_web_verification_20261103/secondary_journal_evidence.json) identifican URLs fijadas, commits y hashes. La [procedencia documental](../_static/literature_e1/public_web_verification_20261103/publication_provenance.json) recoge los identificadores y las asociaciones rechazadas. El {download}`informe de verificación <../_static/literature_e1/public_web_verification_20261103/report.md>` permite revisar el recorrido completo.

#### 3.1.2.3. Control de identidad

Un mirror atribuía a Herremans un PDF y el ID arXiv:2211.00759; al leer la portada se comprobó que era otro trabajo, de Reijnen y colaboradores. Se descartó esa asociación. La pista secundaria 2211.08281 permanece sin corroboración oficial. La concordancia de título, autores, DOI y portada prevalece sobre el nombre de un archivo o una etiqueta del buscador.

(tabla-referencias-e1)=
#### 3.1.2.4. Catálogo bibliográfico conservado: 35 referencias

Las cuatro columnas siguientes mantienen el contenido de las 35 referencias y sus comprobantes originales. La comprobación pública posterior está en las tablas y matrices anteriores; no se sustituyen silenciosamente sus fechas o niveles de lectura.

La tabla reúne las **35 referencias del documento** y organiza sus enlaces, la evidencia disponible y el aporte de cada una al proyecto. Comprende once aplicaciones a Bitcoin, ocho fundamentos de redes neuronales y dieciséis referencias de volatilidad, diagnóstico y evaluación. Se conservan las treinta referencias iniciales, las tres incorporadas para las fichas y los dos textos recuperados en esta revisión. El catálogo anterior de 33 entradas permanece como corte histórico.

El **enlace bibliográfico** identifica el trabajo, su DOI, su preprint o el registro donde se documentó la referencia. La columna **Fuente y evidencia consultada** indica qué material se utilizó realmente: texto completo de autores, código, documentación o una referencia localizada en otra bibliografía. Un DOI registrado no acredita su resolución directa ni la lectura del artículo original. Los manuscritos de Zeri y Gao conservan las precisiones sobre revisión por pares y año editorial recogidas en sus fichas.

##### 3.1.2.4.1. Aplicaciones a Bitcoin (11 referencias)

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

##### 3.1.2.4.2. Fundamentos de redes neuronales (8 referencias)

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

##### 3.1.2.4.3. Fundamentos de volatilidad, diagnóstico y evaluación (16 referencias)

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

El intervalo preferente de publicación es **01/01/2019–03/11/2026**, ocho años calendario. Las consultas web actuales no aplicaron un filtro verificable de año editorial: ese criterio se revisa durante la identificación documental. La fecha de actualización de un repositorio no se utiliza como fecha de publicación. Los trabajos anteriores, como Guo (2018), Katsiampa (2017) y Dyhrberg (2016), conservan su papel de antecedentes/contexto.

El snapshot de un listado de revistas de 2026 es distinto del año de un artículo y de su cuartil. La fecha impresa de una versión arXiv tampoco sustituye al año de la publicación posterior; por ejemplo, Akbiyik tiene una versión pública de 2022 y referencia ACM 2023. Los manuscritos sin fecha editorial acreditada siguen identificados como tales y requieren sensibilidad.

### 3.1.4. Criterios de inclusión y exclusión

Los criterios siguientes se hicieron explícitos al revisar el corpus; no se presentan como prerregistrados antes de las búsquedas del 02/11/2026.

| Dimensión | Evidencia necesaria para extraer un estudio | Tratamiento cuando falta o no corresponde |
|---|---|---|
| Pertinencia | Bitcoin y objetivo de volatilidad o varianza; identificar mercado, frecuencia y horizonte | Precio, dirección o trading sin pronóstico de volatilidad y otros activos quedan fuera del alcance aplicado; una descripción ambigua permanece pendiente |
| Deep Learning comparativo | Modelo DL empleado como comparador independiente en un experimento identificable | Un modelo clásico, una capa de salida o una clase no invocada no acreditan una referencia DL publicada |
| Resultados verificables | Métrica, valor, escala y tabla/sección de origen; dispersión si se reporta | Sin resultados recuperados no se añaden cifras ni frecuencias verificadas |
| Arquitectura y protocolo | Información suficiente para identificar el tipo de referencia y su papel independiente; partición, objetivo y evaluación identificables | Distinguir falta de evidencia del comparador de hiperparámetros no reportados. No completar una configuración por preferencia personal; registrar lo no reportado y la búsqueda de arquitectura requerida por E2 |
| Identidad y publicación | Autoría, título y documento localizable; fecha declarada o desconocida de forma explícita; DOI si existe | Código o README sin texto suficiente es contexto. Un informe o borrador no se excluye solo por su género; la literatura gris se identifica y no acredita revisión por pares |
| Periodo | Preferencia 01/01/2019–03/11/2026; comprobar fecha editorial durante el cribado, sin atribuir un filtro de publicación al buscador web | Los anteriores se justifican como antecedentes/fundamentos; los sin fecha no acreditan recencia y requieren sensibilidad. La actualización de un repositorio no sustituye al año del documento |

Se distinguen pertinencia, recuperabilidad y calidad. Que un texto esté incluido para extracción exploratoria no certifica su validez ni basta para sostener que respalda uno de los tres modelos básicos definitivos. La preferencia Q1 requiere verificar índice, revista, año y categoría; no se deduce de que exista un DOI o un repositorio, ni se traslada automáticamente a congresos.

Un estudio cuenta como máximo una vez por familia. Preprint, artículo y código del mismo estudio no deben duplicarse; la deduplicación bibliográfica entre bases deberá usar DOI y, cuando falte, título/autores/año. La deduplicación del flujo actual solo acredita repositorios repetidos en la búsqueda complementaria. El catálogo de trabajos citados y los cinco textos aplicados con extracción completa cumplen funciones distintas.

Para el nuevo flujo académico se normalizarán DOI; se agruparán identificadores arXiv sin versión y vínculos preprint–artículo comprobados; cuando falten identificadores se revisarán título, autores y año. Las coincidencias ambiguas se resolverán conservando el motivo. Esta deduplicación documental no se presenta como ejecutada mientras no existan exportaciones. La literatura gris se identifica y requiere sensibilidad; no se transforma en evidencia revisada por pares.

Para la verificación posterior se registra evidencia primaria de autores/documento, evidencia secundaria de recursos públicos y estado pendiente por separado. Una correspondencia de revista/ISSN no basta para incluir el artículo en un corpus DL: todavía deben comprobarse objetivo, comparador, arquitectura, protocolo y resultados. Los repositorios nuevos de código o tesis se mantienen como pistas de recuperación hasta resolver identidad y posibles duplicados; no se excluyen por su género ni se incluyen por su descripción.

### 3.1.5. Trazabilidad del proceso

#### 3.1.5.1. Fase pública actual

Las doce consultas devolvieron respuestas de búsqueda verificables. Sus unidades y funciones se conservan separadas:

| Función | Consultas | Ocurrencias recuperadas | Repositorios distintos dentro de esa función |
|---|---:|---:|---:|
| Descubrimiento de documentos | 4 | 40 | 36 |
| Recuperación dirigida de títulos/identificadores | 4 | 7 | 7 |
| Localización de recursos de indexación | 4 | 20 | 20 |

Las 67 ocurrencias de las tres funciones contienen repeticiones y no son 67 artículos. Se recuperó una sola página por cadena; la relevancia del motor y los límites de recuperación sesgan el descubrimiento. El [registro de resultados](../_static/literature_e1/public_web_verification_20261103/discovery_records.json) y el [resumen de la fase](../_static/literature_e1/public_web_verification_20261103/flow_summary.json) permiten reconstruir los conteos sin mezclarlos con el historial anterior.

Se inspeccionaron además cinco README de candidatos localizados en el descubrimiento, con [URLs fijadas, hashes y líneas pertinentes](../_static/literature_e1/public_web_verification_20261103/candidate_sources.json). Incluyen una tesis sobre sentimiento y volatilidad y un informe híbrido GARCH–LSTM: son pistas pendientes de recuperación/identidad y no nuevas inclusiones automáticas. El corpus continúa en cinco textos y 81 cifras publicadas por autores. No se modifica el denominador de frecuencias hasta completar cribado y extracción.

Los intentos directos anteriores y sus cuatro fallos de acceso se conservan únicamente como [antecedente](../_static/literature_e1/academic_search_20261103/access_attempts.json); no se confunden con esta búsqueda pública ejecutada. El recorrido actual se realizó con recursos abiertos y sin sesiones de acceso a las cuatro plataformas.

#### 3.1.5.2. Consultas complementarias históricas de GitHub

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

#### 3.1.5.3. Flujo histórico y recuperación dirigida

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

La frase histórica «sin ejecutar cadenas nuevas» describe aquella recuperación de seis repositorios. La búsqueda pública actual es otra fase: no se suman sus repositorios a este flujo ni se añaden estudios sin deduplicación documental y extracción verificables.

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

La verificación de procedencia añadida en esta fase no avala la calidad experimental. Un identificador correcto o una revista presente en una lista no sustituyen revisión de fugas, particiones, repeticiones, dispersión y comparabilidad. El cuartil Q1 permanece pendiente de evidencia específica por índice, año y categoría; no se deduce de las coincidencias secundarias de 2026.

## 3.2. Identificación de los tres modelos base de Deep Learning

El corpus exploratorio ampliado contiene **cinco textos completos**, S1, S2, S5, S6 y S7. Incluirlos permite extraer lo que sus autores describen; no certifica la validez de sus resultados. S6 y S7 son incorporaciones de la revisión dirigida del 03/11/2026 y conservan sus limitaciones editoriales, temporales y metodológicas.

### 3.2.1. Procedimiento de extracción

La guía exige recurrencia, sencillez, reproducibilidad y tres modelos no redundantes. La siguiente regla de revisión documental se registró **después** de inspeccionar las fuentes; no se presenta como un protocolo preregistrado:

1. **Revisar los comparadores de cada estudio incluido.** Exigir un texto identificable que muestre el uso independiente del modelo DL; separar propuestas avanzadas, híbridos, componentes internos y modelos lineales.
2. **Registrar la configuración por variante.** Extraer entrada, salida, tipo de arquitectura, profundidad, ancho, regularización, pérdida, optimización, entrenamiento y localización. Un campo no reportado permanece desconocido; no se completa por preferencia.
3. **Construir la tabla de frecuencias.** Contar un estudio por familia utilizada como referencia independiente, conservando sus variantes. La recurrencia de una familia no acredita el uso de una configuración concreta en todos esos estudios.
4. **Ordenar los candidatos por frecuencia y complejidad.** Comparar profundidad y parámetros con sus hipótesis, mostrando tanto el tamaño de origen como una representación común p=5, q=1. No utilizar RMSE de tareas distintas como desempate.
5. **Seleccionar exactamente tres modelos cuando la evidencia lo permita.** Justificar conjuntamente recurrencia, sencillez, reproducibilidad y suficiencia de fuentes. Antes de implementar deben cerrarse tres arquitecturas con respaldo; si falta evidencia, mantener la selección pendiente sin omitir un candidato por preferencia.
6. **Verificar sesgo inductivo y no redundancia del trío.** Explicar qué diferencias de representación son relevantes para el problema. Si los tres candidatos no satisfacen conjuntamente los criterios, ampliar la evidencia y conservar el cierre pendiente. Cuando falta una arquitectura concreta, aplica el procedimiento de §3.2.5 y la búsqueda del Entregable 2 antes de entrenar; un espacio supuesto no es una configuración publicada.

#### 3.2.1.1. Frecuencias, variantes y sensibilidad del corpus

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

#### 3.2.1.2. Complejidad de las configuraciones documentadas

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

#### 3.2.1.3. Estado de la selección

**Propuesta de diseño alineada con la guía 6.4:**

| Puesto propuesto | Familia | Entrada y papel | Evidencia pendiente |
|---|---|---|---|
| P1 | MLP sencilla | Las mismas covariables OHLCV rezagadas, aplanadas en 5D componentes; referencia no lineal sin estado recurrente | No hay MLP independiente no lineal acreditada en el corpus; faltan referencias aplicadas y ficha propia. |
| P2 | LSTM unidireccional | Secuencia D×5 de las mismas covariables; referencia con memoria recurrente | La variante unidireccional está documentada en 3/5 textos, con tareas distintas. Falta justificar la adaptación y cerrar la configuración; GRU es alternativa para este puesto. |
| P3 | CNN1D poco profunda | La misma secuencia de covariables; referencia convolucional temporal | No hay CNN1D simple independiente acreditada en el corpus. La TCN ganadora de S1 no satisface por sí sola esta especificación. |

El [registro de propuesta](../_static/literature_e1/guide_aligned_proposal_20261104.json) identifica la procedencia curricular y las lagunas. Elegir LSTM para el puesto recurrente se apoya en su mayor recurrencia observada y en una variante pequeña descrita; no afirma que prediga mejor que GRU. Ninguna arquitectura, ancho o configuración experimental queda congelada.

**Perfiles documentales conservados de la revisión previa:**

| Estado | Familia | Justificación actual | Condición pendiente |
|---|---|---|---|
| Perfil documental recurrente | LSTM unidireccional | Mayor recurrencia familiar y arquitectura pequeña identificable en S5 | Justificar variante y adaptación a OHLCV sin transferir resultados |
| Perfil documental convolucional | TCN causal dilatada | Comparador independiente con filtros compartidos y dilatación | Verificar versión de biblioteca y supuestos de la fuente; no es la CNN1D simple propuesta |
| Perfil documental de atención | Encoder Transformer | Mecanismo de atención distinto, documentado en S2 y S6 | Corroborar procedencia y detalles de fuentes grises como antecedente suplementario |
| Alternativa del puesto recurrente | GRU | 2/5 y menor complejidad que algunas variantes recurrentes | No descartarla solo porque LSTM también sea recurrente; justificar no redundancia y criterios del trío final |

El [registro de selección](../_static/literature_e1/selection_evidence.json) conserva los perfiles como evidencia documental y registra por separado la propuesta MLP–LSTM–CNN1D; la selección experimental final sigue sin congelarse. La frecuencia por sí sola ordena LSTM, luego GRU y Transformer empatados, y después TCN. La elección anterior de LSTM–TCN–Transformer priorizaba diversidad; no satisface por sí sola la regla conjunta de la guía. La propuesta actual se alinea con las tres familias de la guía; completar su respaldo aplicado y las configuraciones reproducibles sigue siendo necesario para satisfacer la extracción de tres básicos.

### 3.2.2. Criterios que debe cumplir cada modelo base

Los cuatro criterios de la guía se evalúan por modelo, junto con la suficiencia de las fuentes. Disponer de una ficha no implica haberlos satisfecho:

| Criterio | Evidencia disponible | Estado y límite |
|---|---|---|
| Arquitectura simple, reproducible y ampliamente documentada | S5 identifica una LSTM pequeña; S1 y S2 describen configuraciones y las fichas separan fuente, cálculo y adaptación | La TCN ganadora de S1 es grande; falta su versión de biblioteca y S6 no permite reconstruir un total exacto. La configuración propia sigue pendiente. |
| Uso frecuente como referencia en la literatura del problema | Frecuencias por estudio y sensibilidades de §3.2.1.1 | LSTM 4/5, GRU 2/5, encoder 2/5 y TCN 1/5 describen este corpus, sin demostrar recurrencia global. GRU continúa elegible. |
| Capacidad de entrenarse bajo el mismo pipeline y partición que los otros modelos base | El contrato propio fija información OHLCV, objetivo, horizonte y evaluación cronológica | Deben justificarse representaciones compatibles con cada arquitectura y el esfuerzo de ajuste común. Implementación, presupuesto y configuraciones de las tres redes permanecen reservados para E2. |
| Ser un modelo de Deep Learning independiente | Se identifican comparadores LSTM, GRU, TCN y encoder; se separan propuestas e híbridos | Una Dense1 lineal, una cabeza interna, persistencia, SVR, Ridge o GARCH no ocupan un puesto de los tres modelos DL. |

El mismo pipeline exige las mismas reglas de información disponible, partición, preprocesamiento y evaluación, con transformaciones ajustadas solo sobre entrenamiento. No obliga a que una representación vectorizada y una secuencial tengan idéntica forma de tensor. Ninguna de estas condiciones equivale a una validación experimental ya ejecutada.

### 3.2.3. Producto obligatorio: síntesis de los estudios incluidos

| Referencia | Problema | Dataset | Modelos DL | Métricas | Valores |
|---|---|---|---|---|---|
| S1. Akbiyik et al. (2023), Tabla 2 {cite:p}`akbiyik2023` | RV del día siguiente agregada de 96 retornos intradiarios | Bitcoin a 15 minutos y Twitter | LSTM, GRU y TCN; D-TCN es la propuesta avanzada | RMSE; media ± DE de veinte corridas | LSTM 2,37±0,19; GRU 4,23±5,61; TCN 2,58±0,07 |
| S2. Zeri (2026), Run B {cite:p}`zeri2026` | Log-RV futura de 30 minutos | BTCUSDT perpetuo, Crypto Lake | GRU, BiLSTM y encoder Transformer SELF-ATT | R² frente a media train y QLIKE | SELF-ATT 0,600/0,562; BiLSTM 0,586/0,582; GRU 0,582/0,564, respectivamente |
| S5. Gao et al. (manuscrito 2023), Tabla 4 {cite:p}`gao2023` | Medida móvil de 30 días desplazada siete días | Yahoo BTC-USD diario | LSTM de una capa y H=20; la densa lineal es contexto | RMSE y RMSPE | LSTM 0,074071 y 0,406084, respectivamente |
| S6. Wang et al. (s. f.), Tabla 4 {cite:p}`wangnd` | Volatilidad diaria denominada RV, sin fórmula suficiente para equipararla a σ7 | Bitcoin diario | Transformer OHLCV independiente; EGARCH–Transformer es híbrido | RMSE, MAE, QLIKE y R² del Modelo 1 | 1,4738; 0,4369; 0,4369; −0,7347, respectivamente, transcritos tal como aparecen |
| S7. Alakhouch Idriss (2026), Tabla 4.5 {cite:p}`alakhouch2026` | Medida futura de 30 días anualizada en el código | Bitcoin diario | LSTM 64→32 con cabeza Dense16→1; no una MLP independiente | RMSE, MAE y MAPE | 0,0358; 0,0254; 13,5 %, respectivamente |

**Límites de interpretación de las cinco filas.** S1 usa RV×100, información, frecuencia y objetivo diferentes de E1. S2 es literatura gris: sus tres semillas se promedian en predicciones; no se informa aquí una media y dispersión de métricas, el R² usa la media train y el objetivo no es diario Spot. S5 evalúa validación sobre valores escalados y no acredita dispersión entre semillas. S6 es un borrador sin fecha editorial, con inconsistencias de MAE y particiones no reconciliadas; esta revisión no valida sus cifras. S7 es un informe académico: PDF declara 12 entradas y código 8; hay escalado global anterior al corte y recuentos inconsistentes. Estas diferencias impiden construir un ranking común de los valores.

La síntesis de seis columnas se puede descargar en [CSV](../_static/literature_e1/study_synthesis_six_columns_20261103.csv); el catálogo bibliográfico de 35 referencias conserva sus cuatro columnas y permanece separado de este producto.

La [matriz completa](../_static/literature_e1/evidence_matrix.csv) conserva problema, dataset, arquitectura, resultados y localización. El [registro de métricas](../_static/literature_e1/reported_metrics.csv) distingue los 42 valores del corte original y los 39 transcritos de S6/S7: **81 valores de autores**, no 81 resultados independientes ni experimentos propios. No se corrigen silenciosamente cifras impresas inconsistentes ni se trasladan a la escala de E1.

S3 —Herremans y Low— y S4 —Huang et al.— siguen como fuentes identificadas con texto completo pendiente. El código de autores de S3 no aumenta la frecuencia hasta verificar en el artículo qué redes se compararon. Las nuevas inspecciones de repositorios sin documento identificable también permanecen fuera del denominador.

### 3.2.4. Equivalencia funcional con referencias clásicas

En este problema continuo, el ejemplo clásico pertinente de la guía es **SVR lineal**. El modelo logístico corresponde a clasificación. La equivalencia se refiere al propósito de establecer una referencia de capacidad controlada; no a igualdad de funciones, supuestos o error:

| Referencia | Capacidad y supuestos | Papel comparativo |
|---|---|---|
| SVR lineal | Función afín sobre covariables vectorizadas, regularización y pérdida ε-insensible | Referencia clásica; no ocupa un puesto DL |
| MLP propuesta | Función no lineal sobre covariables con rezagos aplanadas | Referencia feedforward de capacidad controlada; respaldo aplicado pendiente |
| LSTM | Estado recurrente, memoria y transformaciones no lineales con compuertas | Referencia DL secuencial si se controla su capacidad y presupuesto |
| CNN1D propuesta | Filtros temporales compartidos sobre la historia observable | Referencia convolucional poco profunda; respaldo aplicado pendiente |
| TCN | Filtros causales compartidos y campo receptivo definido | Antecedente documental; no se equipara a la CNN1D poco profunda propuesta |
| Encoder Transformer | Interacciones por contenido y posiciones en una ventana observable | Antecedente documental de atención, fuera de la propuesta actual de tres básicos |

Su comparación exige información, partición, objetivo y esfuerzo de ajuste comunes. Persistencia, Ridge y GARCH aportan contexto y no sustituyen los tres modelos DL. Los cuatro Ridge del bloque C son diagnósticos de EDA; el benchmark propio de redes continúa reservado para E2.

### 3.2.5. Arquitecturas no especificadas y búsqueda posterior

La guía exige que, si la literatura no define una arquitectura concreta —por ejemplo, informa solo «un MLP»—, no se elija por criterio personal: se debe aplicar la búsqueda de arquitectura del **Entregable 2, sección 14**, declarando el espacio antes de entrenar. Esa referencia corresponde a la guía suministrada; no afirma que este capítulo haya ejecutado la búsqueda ni fijado ya un espacio propio.

En la extracción se identifica primero qué componente no está reportado y qué fuente o código lo respalda. Si continúa desconocido, se conserva como tal y se documenta la decisión pendiente. S6 no especifica lo necesario para un conteo exacto; S7 presenta discrepancias entre PDF y código. Estas lagunas no se resuelven imputando valores por preferencia.

Los ganadores de S1 **sí están publicados**: no procede tratar su ancho o profundidad como una arquitectura ausente. La versión de biblioteca pendiente y los supuestos del cálculo permanecen explícitos en §3.2.1.2 y en las fichas. Reducir la TCN de seis niveles y F=287 sería una adaptación que exige justificación, no una réplica del ganador ni una forma de completar un dato omitido. No se inventa en esta revisión un espacio de búsqueda actual; el trío y sus configuraciones siguen pendientes.

### 3.2.6. Familias de arquitecturas orientativas por tipo de problema

Las siguientes familias son una orientación de la guía, no resultados de frecuencia ni una selección automática. La elección final debe proceder de la literatura del problema y cumplir los criterios anteriores. El contrato actual usa covariables OHLCV temporales y una salida continua escalar:

| Tipo de problema | Familias orientativas | Aplicabilidad al proyecto y motivo |
|---|---|---|
| Clasificación y regresión tabular | MLP superficial, MLP profundo regularizado y red residual tabular | No es la ruta principal: aunque los atributos son numéricos, la disponibilidad y dependencia temporal condicionan la partición. No se acredita una MLP no lineal independiente en el corpus. |
| Series de tiempo | MLP sobre rezagos, RNN simple, LSTM o GRU de una capa, CNN 1D/TCN poco profunda | Sus diagnósticos temporales son pertinentes. No se adopta como ruta principal porque el contrato actual construye rezagos de covariables, sin usar rezagos explícitos de la variable objetivo. |
| Regresión temporal | MLP sobre covariables rezagadas, LSTM o GRU sobre sus secuencias y CNN 1D poco profunda | Es la ruta aplicable. La lista orienta la revisión, sin cerrar el trío ni convertir la TCN ganadora de S1 en una red pequeña. |
| Regresión espacial | MLP con coordenadas y variables, MLP con embeddings posicionales y CNN sobre rejilla de vecindad | No aplica: no hay posiciones geográficas ni unidades espaciales que definan vecindad. |
| Regresión espacio-temporal | LSTM o GRU global sobre ventanas, ConvLSTM y convolución sobre grafo o rejilla poco profunda | No aplica: existe orden temporal, pero no múltiples ubicaciones ni una estructura espacial del objetivo. |
| Clasificación de imágenes | CNN poco profunda estilo LeNet/VGG reducida, CNN pequeña con normalización por lotes y ResNet de baja profundidad | No aplica: el contrato no recibe imágenes ni predice clases visuales. |
| Segmentación de imágenes | FCN sencilla, codificador–decodificador convolucional básico y U-Net reducida | No aplica: no hay máscaras ni etiquetas por píxel. |
| Clasificación de texto | MLP sobre embeddings promediados (tipo fastText con red), CNN de texto y LSTM o GRU uni/bidireccional de una capa | No aplica: el contrato propio no incluye un corpus textual de entrada. El uso de Twitter por S1 es contexto bibliográfico. |
| Etiquetado secuencial | LSTM o GRU bidireccional con clasificación por token, la misma con CRF y CNN dilatada | No aplica: no se asigna una etiqueta a cada token o elemento de entrada. |
| Aprendizaje secuencia a secuencia | Codificador–decodificador GRU o LSTM, la misma arquitectura con atención aditiva y codificador–decodificador convolucional sencillo | No aplica: la salida del proyecto es una volatilidad escalar, no una secuencia destino. |

Un encoder no convierte por sí solo el problema en secuencia a secuencia; una convolución temporal tampoco implica imágenes. Las orientaciones no aplicables se conservan con su justificación y no añaden experimentos ni puestos al trío.

## 3.3. Análisis documental de las arquitecturas recuperadas

Las fichas siguientes documentan LSTM, TCN y encoder Transformer tal como aparecen en las fuentes recuperadas. **No corresponden al trío propuesto MLP–LSTM–CNN1D poco profunda.** La ficha LSTM aporta un antecedente para el puesto recurrente; las fichas TCN y encoder se conservan para trazabilidad y comparación de literatura. Faltan las fichas aplicadas de MLP y CNN1D simple, y las configuraciones propias todavía no están cerradas. Se distinguen: arquitectura reportada por la fuente, cálculo analítico derivado de su descripción y adaptación que necesitaría el proyecto. La elaboración de estas fichas no ejecutó código de Deep Learning ni generó predicciones o resultados nuevos con datos de mercado; las referencias Ridge del EDA se documentan por separado en el capítulo 2.

Tras la recuperación dirigida, el corpus exploratorio contiene **cinco textos completos**: S1, S2, S5, S6 y S7. La **familia LSTM aparece en 4/5**, incluida la BiLSTM de S2; sus variantes unidireccionales están en S1/S5/S7, **3/5**. GRU aparece en **2/5**, TCN en **1/5** y encoder Transformer en **2/5**, S2/S6. S6 y S7 se incorporan como literatura gris con riesgos explícitos; una mayor cantidad de textos no acredita revisión por pares, mínima complejidad o comparabilidad de sus resultados. Las fichas concretas se mantienen apoyadas en LSTM de S5, TCN de S1 y SELF-ATT de S2. S6/S7 amplían evidencia de las familias, no definen réplicas idénticas de esas configuraciones. Una referencia general de arquitectura no cuenta como estudio aplicado adicional.

El contrato del proyecto conserva cinco covariables OHLCV diarias, entradas de D=7/14/21/28 días, volatilidad histórica de n=7 retornos diarios y horizonte h=1. Al inicio de t se conoce información hasta t−1 y se pronostica Y_t=σ_(t+1), que incorpora r_t. Este objetivo no es volatilidad realizada de siete días enteramente futuros. Ninguno de los cinco textos utiliza exactamente este contrato.

### 3.3.1. LSTM unidireccional

#### 3.3.1.1. Referencias completas y papel comparativo

**S1.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023). *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*. WSDM 2023. DOI: [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387). El artículo compara LSTM, GRU y TCN antes de proponer D-TCN. La LSTM es un comparador y no el modelo novedoso del trabajo. Evidencia: PDF pp. 4–6, §§5.2–5.4, Fig. 1 y Tabla 2; código `scripts/hpo.py`, líneas 151–235.

**S5.** Gao, M., Li, Q. y Tao, Z. *Comparing the Predictive Power of Five Models on Bitcoin Volatility*. Manuscrito de 2023 y código del primer autor; referencia editorial declarada a Springer, *Learning and Analytics in Intelligent Systems*, vol. 32, pp. 473–486, DOI declarado [10.1007/978-3-031-38074-7_41](https://doi.org/10.1007/978-3-031-38074-7_41). Año editorial externo pendiente de corroboración; no se infiere de la fecha del DOCX. El texto compara media, random walk, GARCH, una red densa lineal y LSTM. La arquitectura sencilla de una capa y veinte unidades está descrita en Tabla 4 y en la celda 111 del [notebook fijado](https://github.com/ga0m1/bitcoin-volatility-models/blob/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/BTC_USD.ipynb).

**S7.** Alakhouch Idriss (2026). *Bitcoin Volatility Forecasting: Econometric and Machine Learning Approaches Applied to Daily Price Data (July 2010–February 2024)*. Informe fechado 02/06/2026, sin DOI ni revisión por pares acreditados. [Informe y versión de autor](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Repport/Asset-volatility-Forecasting.pdf), §3.5.3 y Tabla 4.5, p. 34 impresa; [código](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Python/Python.py), líneas 366–415.

#### 3.3.1.2. Tipo de arquitectura

Red recurrente LSTM con memoria y compuertas, seguida de una cabeza densa. La ficha concreta toma la variante unidireccional de una capa y H=20 de S5. S1 y S7 aportan comparadores de la misma familia con configuraciones y tareas diferentes; no son réplicas de S5 ni tres estudios de una arquitectura idéntica.

#### 3.3.1.3. Descripción técnica

**Arquitecturas y entrenamiento efectivamente descritos.**

S5 utiliza catorce observaciones de volatilidad como entrada, una LSTM de veinte unidades y una salida Dense(1). No incorpora una segunda capa recurrente. El notebook compila con MSE y Adam, batch 64, máximo 200 épocas, paciencia 20 y restauración de pesos elegidos según `val_rmspe`. El callback de checkpoint monitoriza `loss`, un criterio distinto: esta diferencia debe conservarse al interpretar el procedimiento y no se sustituye por un protocolo supuesto. No se reportan varias semillas ni dispersión entre corridas.

S1 utiliza un día de 96 retornos de quince minutos como entrada y produce 96 retornos del día siguiente mediante una cabeza densa. Su búsqueda varía unidades recurrentes entre 32 y 512, dropout entre 0 y 0,5, learning rate entre 10⁻⁷ y 10⁻², weight decay entre 10⁻⁹ y 10⁻² y ε entre 10⁻² y 10⁻¹. Usa AdamW y pérdida cuadrática ε-insensible, 250 configuraciones por red, treinta épocas y veinte corridas para la comparación. El apéndice C, Tabla 5, sí reporta la configuración ganadora de S1: H=261, dropout 0,0237, ε=0,0364, learning rate 0,00521 y weight decay 9,25×10⁻⁷. La revisión anterior omitía esta tabla; se corrige esa extracción. Estos valores de autores no son una configuración del proyecto.

**Variante adicional de S7.**

El informe compara una LSTM independiente con GARCH, Random Forest y XGBoost, y presenta una propuesta EGARCH+XGBoost. Declara ventana de 60 días, dos capas LSTM de 64/32 unidades, dropout 0,2, Dense(16, ReLU) y salida lineal escalar; usa Adam 10⁻³, batch 32, máximo 200 épocas y paciencia 15. Es una variante más profunda que la ficha concreta de S5. El PDF declara doce atributos, mientras el código enumera ocho, incluida varianza condicional EGARCH; no se trata de las cinco OHLCV del proyecto. El objetivo del código es desviación móvil futura de treinta retornos, anualizada, no σ7 diario.

La extracción documental de S7 identifica MSE como pérdida y Adam como optimizador. Es información de esa fuente; no fija la configuración de las redes del proyecto.

**Componentes, ecuaciones, flujo y sesgo inductivo.**

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

#### 3.3.1.4. Tipo de problema que aborda

Las fuentes tratan pronósticos continuos de Bitcoin con definiciones distintas: S1 predice una secuencia de 96 retornos del día siguiente y después agrega RV diaria; S5 pronostica una medida móvil de treinta días desplazada siete días; el código de S7 utiliza una medida futura de treinta retornos anualizada. La variante del proyecto se define por su propio contrato OHLCV diario, n=7 y h=1, sin trasladar esas tareas o resultados como si fueran equivalentes.

#### 3.3.1.5. Datasets utilizados en las fuentes

| Fuente | Dataset documentado | Unidad y separación relevantes |
|---|---|---|
| S1 | Bitcoin, precios cada quince minutos; 144 días declarados | 96 días de desarrollo, separados en 72/24, y 48 días de test. El estudio también incorpora Twitter en su propuesta D-TCN; la LSTM comparativa descrita usa retornos. |
| S5 | Yahoo Finance BTC-USD diario; 2.607 registros declarados | Las fechas finales no se reconcilian entre las partes de la fuente; el notebook distingue validación de 365 días y test final de treinta. |
| S7 | BTC/USD diario de Investing.com/Kaggle, según el informe; 18/07/2010–09/02/2024 | 4.955 observaciones declaradas; número de atributos, recuentos y particiones del informe/código requieren conciliación. |

La procedencia de estas fuentes se toma de la extracción documental conservada en `studies.json`; no se han adquirido o reconstruido sus datasets en esta revisión.

#### 3.3.1.6. Métricas y valores publicados

| Fuente y evaluación | Resultado publicado | Dispersión y escala |
|---|---|---|
| S1, Tabla 2, test de 48 días | LSTM RMSE 2,37; MAE 1,70; MAPE 0,30; MSLE 0,09 | DE 0,19 / 0,26 / 0,08 / 0,01, respectivamente, entre veinte corridas; RV = 100·sqrt(sum(r_15min²)); MAPE como fracción |
| S5, Tabla 4, comparación de validación | LSTM(20) RMSE 0,074071; RMSPE 0,406084 | Sin dispersión entre semillas; comparación sobre valores escalados y objetivo de treinta días desplazado siete días |

No se ordenan estos RMSE como si midieran el mismo problema. S1 predice retornos intradiarios y después agrega volatilidad; S5 emplea volatilidad pasada como entrada y no centra los retornos al construir su medida. El contrato del proyecto conserva OHLCV diario y desviación poblacional centrada de siete retornos. La pérdida MSE fijada para el diseño del proyecto coincide nominalmente con S5, pero no con la pérdida ε-insensible de S1; que una pérdida comparta nombre tampoco iguala escalas y objetivos.

**S7: valores publicados y condiciones de interpretación.**

Tabla 4.5 reporta LSTM RMSE 0,0358, MAE 0,0254 y MAPE 13,5 %, sin dispersión entre semillas; son valores del autor, no verificados mediante reproducción. El código ajusta MinMaxScaler de entradas y objetivo sobre todo `df_garch` antes del corte temporal, riesgo aparente de fuga de información. También monitoriza `val_loss`, frente a MAE descrita en el informe, y los recuentos declarados 3.849+1.006=4.855 no reconcilian el total 4.955. Estas discrepancias impiden usar la fuente para acreditar superioridad o una réplica exacta y no se corrigen inventando resultados o dimensiones. Se cuenta una sola vez como apoyo exploratorio de familia.

#### 3.3.1.7. Fortalezas

**Fortalezas:** comparador recurrente documentado en tres fuentes unidireccionales, con variantes distintas, arquitectura pequeña completamente identificable en S5 y número de parámetros independiente de D.

#### 3.3.1.8. Limitaciones

**Límites:** S5 contiene fechas inconsistentes y carece de repeticiones e inferencia; S1 dispone de veinte corridas, pero solo 48 días de test. La variante, transformación, activación de salida y configuración propia deben distinguirse de la réplica de cada fuente. No se atribuye a las fuentes una salida `softplus`: sus cabezas descritas son lineales.

Las discrepancias de entrada, etiquetado y escalado de S7 se conservan junto a sus cifras en §3.3.1.6. Su inclusión exploratoria no acredita una réplica, ausencia de fuga o una mejora transferible al proyecto.

#### 3.3.1.9. Complejidad computacional

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

**Tiempo de entrenamiento e inferencia:** esta ficha no dispone de un tiempo de reloj recuperado y comparable entre S1/S5/S7. Épocas, paciencia y batch describen presupuesto de entrenamiento, no duración. No se ha medido un tiempo propio ni se deduce uno a partir del conteo de parámetros.

### 3.3.2. TCN: convolución temporal causal y dilatada

#### 3.3.2.1. Referencias completas y papel comparativo

**S1.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023). *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*. WSDM 2023. DOI: [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387).

S1 documenta una **TCN única** como comparador de LSTM, GRU, AR-RV y GARCH. Luego propone una **D-TCN de dos ramas**, una para retornos y otra para señales de Twitter. La ficha selecciona la TCN comparativa, no la D-TCN novedosa. La base conceptual de convoluciones causales y dilatadas está descrita por Bai, Kolter y Koltun (2018), referencia de fundamentos del libro; ese trabajo genérico no se cuenta como otro estudio de Bitcoin.

Fuentes: S1, Fig. 1, §§5.2–5.4 y Tabla 2; [hpo.py por commit](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/scripts/hpo.py), líneas 151–235. `dtcn.py` corresponde a la extensión propuesta y no debe utilizarse como si describiera la base de una sola rama.

**Fundamento arquitectónico.** Bai, S., Kolter, J. Z. y Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. Prepublicación [arXiv:1803.01271](https://arxiv.org/abs/1803.01271); sin DOI identificado en la referencia local. Sustenta el mecanismo general, no una nueva observación de uso en Bitcoin.

#### 3.3.2.2. Tipo de arquitectura

Red convolucional temporal 1D con causalidad, dilataciones y conexiones residuales. S1 usa una TCN comparativa de una rama y distingue la D-TCN propuesta de dos ramas. La relación con una CNN1D no permite identificar automáticamente esta configuración con una CNN poco profunda.

#### 3.3.2.3. Descripción técnica

**Arquitectura y entrenamiento publicados.**

El código invoca `TCN` de `keras-tcn` con seis niveles de dilatación `(a⁰,a¹,a²,a³,a⁴,a⁵)`, a∈{2,3,4}, kernel k∈{2,…,6}, canales F∈{32,…,512}, dropout en [0,0,5], skip connections activadas o no y alternativas de normalización batch/weight/layer/ninguna. Después se aplica Dense(96) y reshape a 96×1 retornos futuros. Entrada fuente: 96×1 retornos observados. El artículo compara modelos con información financiera de precios antes de introducir las ramas de Twitter.

La selección utiliza 250 configuraciones, treinta épocas por configuración, AdamW y pérdida cuadrática ε-insensible sobre retornos. Los rangos de learning rate, weight decay y ε son los mismos de la ficha LSTM. Los veinte reinicios de la comparación final se evalúan después de agregar los 96 retornos predichos a RV diaria. El apéndice C, Tabla 6 del artículo, sí publica los hiperparámetros ganadores: F=287 filtros, kernel k=5, base de dilatación a=4, dropout 0,217, ε=0,0913, learning rate 6,49×10⁻⁵, weight decay 5,93×10⁻⁶, skip connections activadas y ninguna normalización. Se corrige la omisión de esa tabla en la extracción anterior. Sigue sin estar fijada la versión de `keras-tcn`, por lo que los detalles internos de sus bloques deben verificarse antes de una réplica.

**Componentes, ecuaciones, flujo y sesgo inductivo.**

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

#### 3.3.2.4. Tipo de problema que aborda

En S1, la TCN predice 96 retornos intradiarios del día siguiente y su evaluación agrega una medida de volatilidad realizada diaria. Es un pronóstico continuo de secuencia; no es la salida escalar σ7 del contrato OHLCV diario del proyecto. La ficha corresponde al comparador financiero de una rama.

#### 3.3.2.5. Dataset utilizado en la fuente

S1 documenta precios de Bitcoin cada quince minutos durante 144 días: 96 de desarrollo, separados en 72/24, y 48 de test. La entrada de la TCN comparativa es una ventana de 96 retornos; las señales de Twitter se introducen en la propuesta D-TCN, cuya segunda rama no se atribuye a esta referencia. No se presenta ese dataset como idéntico a Binance Spot diario del proyecto.

#### 3.3.2.6. Métricas y valores publicados

S1, Tabla 2, informa **TCN RMSE 2,58±0,07; MAE 1,63±0,06; MAPE 0,23±0,01; MSLE 0,11±0,01**, veinte corridas, test de 48 días y la misma escala RV de la ficha LSTM. En ese estudio, TCN tiene mejor MAPE/MAE que LSTM, mientras LSTM tiene menor RMSE/MSLE. El orden depende de la métrica; no se comunica un ganador universal. Los contrastes del artículo son unilaterales respecto a AR-RV y no implican automáticamente un contraste directo entre las redes ni control de toda la multiplicidad.

#### 3.3.2.7. Fortalezas

**Fortalezas:** comparador separado de la propuesta híbrida, filtros compartidos y contexto multiescala con cálculo paralelo.

#### 3.3.2.8. Limitaciones

**Límites:** frecuencia 1/5 en el corpus exploratorio, detalles internos dependientes de una biblioteca sin versión fijada y test corto. La fuente respalda TCN y no acredita una CNN1D de una sola capa como réplica exacta. «Base comparativa» no significa mínima profundidad absoluta.

Para el proyecto, pasar de retornos intradiarios a cinco OHLCV diarios y de 96 salidas a una volatilidad escalar sería una adaptación. Su profundidad, dilataciones, normalización y salida deben registrarse antes de implementar en E2. Si falta detalle de arquitectura, corresponde consultar el protocolo de E2 §14 citado por la guía; esta ficha no inventa sus reglas.

#### 3.3.2.9. Complejidad computacional

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

**Tiempo de entrenamiento e inferencia:** no se dispone en la extracción de una medición de reloj comparable para esta ficha. Treinta épocas y veinte reinicios son cantidades del procedimiento, no una duración. La paralelización descrita no acredita velocidad medida, GPU o memoria observadas.

### 3.3.3. Transformer encoder compacto, comparador SELF-ATT

#### 3.3.3.1. Referencias completas y papel comparativo

**S2.** Zeri, T. (octubre de 2026). *Forecasting High-Frequency Bitcoin Volatility: Neural Networks, Market Microstructure and On-Chain Data*. Manuscrito público de investigador independiente, sin DOI ni revisión por pares acreditados. [Fuente TeX fijada](https://github.com/tommasozeri/Forecasting-High-Frequency-Bitcoin-Volatility-Neural-Networks/blob/d69f951f2ced7273bdc4c3e5a8a4cb220c2ee5ce/paper_en/manuscript.tex), §§Models/Training, líneas 179–189, tabla de resultados y apéndice, líneas 395–409. Arquitectura leída en celda 21 del notebook `modelli_btc_hf.ipynb`, mismo commit.

SELF-ATT es uno de los tres comparadores neuronales genéricos —junto a GRU y BiLSTM— del estudio; la contribución se centra en pipeline, comparación y ablación de atributos, no en declarar una nueva arquitectura de atención. La ficha concreta describe S2. El encoder aparece además en S6, con una configuración distinta: **2/5 textos del corpus exploratorio**, ambos sin revisión por pares acreditada; esto no prueba recurrencia en toda la literatura. Es compacto frente a grandes Transformers, pero tiene más parámetros que la LSTM simple de S5; no se le atribuye mínima complejidad absoluta.

**S6.** Wang, Y., Yu, Q., Peng, J., Yang, B. y Chen, Z. (s.f.). *A comparative analysis of volatility predictions via EGARCH, Transformer and hybrid EGARCH-Transformer models: changes in forecasting ability after introducing sentiment factors*. [Borrador de autores](https://github.com/cathyw111/bitcoin-volatility-forecasting/blob/61feeec81d695e19eac9c819be41d704e6279547/research-draft.pdf), §3.4 y Tablas 1–2, pp. 6–7, y Tabla 4, p. 10. La versión del repositorio está disponible en 2026; la fecha de commit no se presenta como año editorial. No se acreditan DOI ni revisión por pares.

#### 3.3.3.2. Tipo de arquitectura

Encoder Transformer de atención multihead sobre una ventana histórica, con cabeza de regresión. La ficha concreta es SELF-ATT de S2, con dos bloques pre-LayerNorm, d=48 y cuatro cabezas. S6 documenta otro encoder con d=32 y salida Softplus, cuya especificación parcial permanece separada. Recibir una secuencia y producir un escalar no convierte por sí solo esta tarea en aprendizaje secuencia a secuencia.

#### 3.3.3.3. Descripción técnica

**Arquitectura y entrenamiento de S2.**

La arquitectura contiene proyección lineal F→d, posiciones aprendidas de longitud máxima M, dos bloques encoder pre-LayerNorm y cuatro cabezas. Cada bloque incluye atención multihead, dos conexiones residuales y feedforward de ancho 2d. El código no especifica la activación del encoder, por lo que usa ReLU por defecto; la cabeza sí usa GELU. La cabeza concatena último token y media de tokens y aplica dropout→Dense(2d,d)→GELU→Dense(d,1).

La configuración elegida reportada para SELF-ATT en Run B es d=48, dropout 0,21, learning rate 3·10⁻⁴, weight decay 1,4·10⁻³, ruido de entrada 0,12 y ventana L=48. El código conserva M=96 posiciones, aunque la ventana elegida sea menor. La pérdida es MSE sobre el objetivo log-RV estandarizado, según la línea 119 del manuscrito y `loss_fn` en la celda 23. Se declara AdamW, batch 512, máximo cuarenta épocas, paciencia seis, clipping 1 y semillas 42/43/44; cada red recibe veinte trials de búsqueda. Las predicciones finales se promedian entre semillas antes del cálculo de métricas.

**Variante adicional de S6.**

Su comparador `model1` es un encoder Transformer sobre OHLCV diario de Bitcoin, con historia de 20 días, d=32, cuatro cabezas, dos capas, dropout 0,1 y salida Softplus. Declara MSE, Adam 0,001, batch 32, máximo 100 épocas, paciencia 10 y semilla 42. El periodo es 25/03/2021–30/12/2024 y el objetivo declarado RV1D, cuya construcción exacta no quedó suficientemente especificada. Faltan ancho feedforward, pooling/lectura, orden de normalización y versión de implementación; por ello no se inventa un total de parámetros ni se traslada el conteo 49.249 de S2 a S6.

**Componentes, ecuaciones, flujo y causalidad.**

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

#### 3.3.3.4. Tipo de problema que aborda

SELF-ATT pronostica el logaritmo de RV de los próximos treinta minutos. S6 declara pronóstico de RV1D del día siguiente, pero la fórmula del proxy no quedó suficientemente especificada. Ambas salidas continuas difieren de la desviación poblacional centrada de siete retornos diarios que define el proyecto.

#### 3.3.3.5. Datasets utilizados en las fuentes

Run B utiliza 48 barras de cinco minutos y 41 atributos de Binance Futures BTCUSDT perpetuo mediante Crypto Lake. El objetivo es log-RV de los siguientes treinta minutos, seis barras, a partir del precio medio del libro de órdenes; no es desviación móvil diaria de siete retornos. Hay 31.524 predicciones de test, correspondientes al 13/09–31/12/2025. La tabla publica métricas de predicciones promediadas entre tres semillas, no media y desviación de tres métricas.

S2 utiliza Binance Futures BTCUSDT perpetuo de Crypto Lake, con datos de operaciones, libro, financiación, interés abierto, liquidaciones y atributos on-chain de 2024–2025. Es una fuente licenciada y multivariante, con 41 atributos declarados en Run B. S6 declara Bitcoin diario del 25/03/2021 al 30/12/2024 y 1.376 retornos, pero la procedencia bruta no quedó suficientemente identificada. Ninguna de estas fuentes se presenta como el mismo dataset del proyecto.

#### 3.3.3.6. Métricas y valores publicados

| Modelo de S2, Run B | R² respecto a media train | R² respecto a media test | QLIKE | Spearman |
|---|---:|---:|---:|---:|
| SELF-ATT | 0,600 | 0,566 | 0,562 | 0,749 |
| BiLSTM | 0,586 | 0,550 | 0,582 | 0,742 |
| GRU | 0,582 | 0,546 | 0,564 | 0,743 |
| HAR | 0,508 | 0,466 | 0,773 | 0,676 |

El manuscrito reporta DM de SELF-ATT frente a HAR de 19,7, p unilateral <10⁻¹⁵ y HAC de cinco rezagos por solapamiento. No es un contraste directo SELF-ATT contra BiLSTM ni una prueba ejecutada por este proyecto. R² respecto a media train no equivale al R² convencional respecto a media test. No hay en esa tabla desviación ni IC entre las tres semillas: se evalúa su promedio de predicciones.

**S6: cifras literales y condiciones de interpretación.**

Tabla 4 informa para `model1` RMSE 1,4738, R² −0,7347, QLIKE 0,4369 y MAE 0,4369, sin dispersión. Se conservan literalmente, sin tratarlos como cifras validadas: el texto llama mejor al MAE 0,7399 de `model9` aunque 0,4369 es menor; la combinación RMSE/R² no permite reconciliar una cohorte común con la definición convencional, y el corte 70/30 de Transformer no está reconciliado con los 203 días de test EGARCH. No se ejecutó DM por costo computacional, según §6.2. Sin datos o código de entrenamiento no puede verificarse disponibilidad, escalado ni el origen de esas diferencias. La fuente respalda la presencia de un comparador de atención con información OHLCV, pero no acredita una mejora reproducida ni cierra su especificación.

#### 3.3.3.7. Fortalezas

**Fortalezas:** mecanismo distinto, encoder/cabeza descritos y configuración acotada; relaciones directas entre posiciones de la ventana.

#### 3.3.3.8. Limitaciones

**Límites:** los dos apoyos aplicados de encoder son literatura gris; S6 tiene especificación parcial y métricas inconsistentes. La ficha concreta de S2 utiliza datos licenciados, una ventana de evaluación y costo cuadrático, con resultados no reproducidos aquí. Además, las redes utilizan 24–48 barras mientras los referentes lineal y LightGBM usan solo la última: la comparación no aísla completamente arquitectura de historia informativa. El desglose de atributos de libro tiene una discrepancia de enumeración; se conserva F=41 como dimensión declarada, con esa salvedad.

La fuente tiene salida lineal para log-RV estandarizado. Adaptarla a OHLCV diario y volatilidad no negativa requiere un contrato propio; añadir `softplus` o reducir capas no sería una descripción de la arquitectura publicada. No se fijan esas modificaciones a partir de cifras de otro objetivo.

#### 3.3.3.9. Complejidad computacional

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

**Tiempo de entrenamiento e inferencia:** no se dispone en esta ficha de una duración recuperada y comparable con las otras familias. El máximo de cuarenta épocas o los veinte trials de S2 no se convierten en tiempo de reloj; no hay medición propia de entrenamiento, inferencia o memoria.

### 3.3.4. Cobertura y comparación documental de las tres fichas

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

La numeración de cada ficha sigue los nueve incisos de la guía: referencias; tipo de arquitectura; descripción técnica; tipo de problema; datasets; métricas/valores; fortalezas; limitaciones; y complejidad. Un DOI ausente o un tiempo no recuperado se declara, y un conteo analítico condicionado no se presenta como tamaño medido por una implementación.

## 3.4. Comparación crítica de la evidencia recuperada

Las tablas siguientes preservan resultados y configuraciones de las fuentes LSTM/TCN/encoder. Su función es documental; no sustituyen la comparación futura de MLP, LSTM y CNN1D poco profunda ni acreditan las fichas que aún faltan.

### 3.4.1. Comparación estructurada entre las arquitecturas documentadas

#### 3.4.1.1. Desempeño publicado y comparabilidad

S1 permite comparar LSTM, GRU y TCN dentro de un protocolo: LSTM tiene menor RMSE puntual que TCN, mientras TCN tiene menor MAE. El tuning selecciona por MAPE; ningún orden aislado acredita superioridad en todas las métricas. S5 aporta validación en una escala diferente. S2 evalúa log-RV intradiaria y promedia predicciones entre semillas. Sus métricas no forman un ranking común con las de S1/S5.

S6 y S7 amplían la evidencia de uso, pero **no mejoran automáticamente su calidad**. S6 publica una afirmación sobre el mejor MAE que contradice su propia Tabla 4; además, no se reconciliaron las particiones ni las definiciones de las métricas. S7 presenta discrepancias de atributos, muestras y selección de pesos entre informe y código, además de un escalado global previo al corte. Sus cifras se conservan como resultados declarados, sin validarlas ni usarlas para decidir el trío final.

El proyecto conserva OHLCV diario de Binance Spot, desviación poblacional centrada de siete retornos y horizonte de un día. S1 agrega retornos intradiarios; S2 usa un perpetuo y log-RV de treinta minutos; S5 desplaza siete días una medida de treinta; S6 no permite equiparar su RV a nuestro objetivo y S7 construye una medida futura de treinta días anualizada. Ninguna cifra anterior es un resultado propio con ese contrato.

#### 3.4.1.2. Robustez y evidencia disponible

La comparación documental siguiente no constituye una medición propia de robustez bajo el contrato de E1. Se conservan las tres fichas de trabajo y sus límites, sin retirar a GRU de la evaluación para el trío final.

| Dimensión | LSTM | TCN | Encoder Transformer |
|---|---|---|---|
| Uso documentado | Familia en S1/S2/S5/S7; S2 es BiLSTM | S1 | S2 y S6, literatura gris |
| Repeticiones | S1: veinte corridas; S5/S7 sin dispersión entre semillas acreditada | S1: veinte corridas | S2: promedio de predicciones de tres semillas; S6 una semilla declarada |
| Riesgo de partición | S5: posible cruce de etiquetas futuras; S7: escalado global anterior al corte | Solo 48 días de test en S1; no prueban robustez entre mercados | S2: información histórica distinta de referencias lineales; S6: particiones y selección de EGARCH no reconciliadas |
| Inferencia publicada | No confundir pruebas de otros modelos de S7 con evidencia sobre LSTM | Contrastes unilaterales contra AR-RV, no automáticamente entre redes | S2: DM contra HAR, no contra todas las redes; S6 declara DM no ejecutado |
| Conclusión permitida | Referencia recurrente documentada con calidad heterogénea | Comparador convolucional documentado, sin configuración mínima acreditada | Comparador de atención documentado; respaldo editorial y métricas requieren cautela |

#### 3.4.1.3. Escalabilidad y simplicidad

| Aspecto | LSTM | TCN | Encoder Transformer |
|---|---|---|---|
| Cómputo dominante por ventana | O(D·H(p+H)) para una capa | O(D·Σ k_l C_(l−1)C_l) | O(D²d+Dd²) por encoder, con ancho feedforward proporcional a d |
| Ejecución temporal | Dependencia secuencial entre estados | Filtros paralelizables en la ventana | Interacciones entre pares de posiciones |
| Memoria de activaciones en entrenamiento | Crece con D y H | Crece con D, ancho y profundidad | Atención completa añade almacenamiento cuadrático en D |
| Evidencia de sencillez | S5 identifica una LSTM pequeña; S1 y S7 son mayores | El ganador S1 es ancho y de seis niveles; no equivale a CNN1D poco profunda | Dos encoder documentados; menos pesos que la TCN bajo las hipótesis, sin garantizar menor tiempo |
| Medición comparable propia de tiempo/GPU/RAM | No ejecutada | No ejecutada | No ejecutada |

Las fórmulas y los conteos de §3.2.1.2 y §3.3 controlan sus hipótesis, pero no miden eficiencia. Entrenamiento e inferencia dependen también de implementación, hardware, batch, precisión y presupuesto. Las épocas descritas por S2 no equivalen a tiempos de reloj comparables. Los tamaños de ventanas y salidas de las fuentes son distintos; el conteo con p=5,q=1 no valida esos modelos bajo OHLCV diario.

### 3.4.2. Análisis de validez de los estudios revisados

La evaluación distingue amenazas internas, externas, de constructo y de conclusión estadística. Son riesgos documentales con evidencia y desconocidos; no resultados de una reproducción ejecutada:

| Dimensión de validez | Evidencia o riesgo identificado | Consecuencia para la interpretación |
|---|---|---|
| Interna | Riesgos de fuga y selección: objetivo futuro sin purga identificada en S5, escalado global en S7 y cortes/selección de EGARCH no reconciliados en S6 | No atribuir diferencias solo a la arquitectura ni asumir ausencia de fuga por disponer de código. |
| Externa | Un activo, fuentes y frecuencias distintas; S1 limita su test a 48 días | No está demostrada la generalización a Binance Spot diario ni a otros regímenes o mercados. |
| De constructo | Distintas definiciones de RV, log-RV, desplazamiento, anualización, escala e información disponible | RMSE, MAE y R² de estudios diferentes no forman un ranking común ni miden necesariamente el objetivo de E1. |
| De conclusión estadística | Inicializaciones con observaciones de test compartidas, promedios de predicciones, una semilla o ausencia de dispersión, según la fuente | Diferenciar variabilidad del entrenamiento, muestreo temporal, tamaño del efecto, precisión y familia de contrastes. |
| Procedencia editorial | Literatura gris, año editorial desconocido y coincidencias secundarias de revista en listas de 2026 | La procedencia no certifica el protocolo experimental ni el registro individual, el cuartil o la cobertura del año de publicación. |

Las veinte inicializaciones de S1 no son veinte mercados independientes y su test de 48 días limita la generalización. La selección por una métrica y los contrastes unilaterales requieren una familia de hipótesis explícita. En S5, el objetivo desplazado usa información posterior a la fila sin una purga identificada en el corte: se registra riesgo aparente, sin afirmar una reproducción que no se ejecutó.

En S2, las redes usan ventanas de 24/48 barras mientras ciertas referencias ven solo la última barra; la ganancia no aísla exclusivamente no linealidad. Sus tres predicciones promediadas tampoco aportan una desviación entre métricas de semillas. En S6, la fecha de una versión del repositorio no prueba año editorial; las incongruencias de la tabla no deben corregirse suponiendo una fórmula alternativa. En S7, la función DM del código compara otros modelos y no trata adecuadamente la dependencia temporal del objetivo móvil; no valida por sí sola la LSTM ni las cifras del informe.

Se conserva la evaluación por dimensión, con evidencia y desconocidos, sin una puntuación inventada. Disponer de código no certifica ausencia de fuga, y un artículo incluido con riesgo alto no constituye respaldo suficiente para adoptar su configuración o su desempeño como baseline.

### 3.4.3. Brecha entre propuestas avanzadas y modelos base

**Brecha cuantificada entre una propuesta avanzada y su referencia.** S1, Tabla 4 (PDF p. 8), permite examinar las cuatro variantes D-TCN con un único grupo de información social, manteniendo también las que empeoran:

| Modelo de S1 | RMSE media ± DE | Reducción relativa puntual de RMSE frente a TCN | MAE | MAPE |
|---|---:|---:|---:|---:|
| TCN de una rama | 2,58 ± 0,07 | Referencia | 1,63 | 0,23 |
| D-TCN User | 2,36 ± 0,05 | 8,53 % | 1,44 | 0,20 |
| D-TCN Tweet | 2,45 ± 0,09 | 5,04 % | 1,59 | 0,21 |
| D-TCN VADER | 2,70 ± 0,09 | −4,65 % | 1,72 | 0,26 |
| D-TCN Count | 3,27 ± 0,47 | −26,74 % | 1,97 | 0,32 |

La reducción se deriva como `100·(RMSE_TCN−RMSE_variante)/RMSE_TCN` a partir de medias redondeadas de autores; un valor negativo indica empeoramiento. Es una transformación descriptiva, sin nuevo experimento, intervalo ni contraste pareado. D-TCN usa cuarenta reinicios frente a los veinte del comparador; la fila TCN de Tabla 4 repite los resultados base. Cambian conjuntamente arquitectura e información social y no se repite una búsqueda independiente para cada variante. Las brechas no aíslan el efecto de complejidad ni calibran el margen del 5 % adoptado para E1 frente a persistencia.

### 3.4.4. Brecha verificable y proyección

**Límite del corpus respecto a la propuesta actual:** ninguno de los cinco textos incluidos acredita una comparación conjunta de MLP, LSTM y CNN1D poco profunda con el contrato OHLCV diario de Binance Spot, n=7 y h=1. En particular, MLP independiente y CNN1D simple tienen 0/5 apoyos verificados en este corpus. Esta es una laguna de cobertura de la revisión accesible, no prueba de ausencia mundial ni de originalidad. Debe completarse la evidencia aplicada antes de presentar el trío como extraído y justificado por la literatura.

La contribución evaluable sería establecer referencias DL de capacidad controlada, bajo la misma información y presupuesto, y cuantificar qué mejora debe superar un modelo avanzado. Antes de ejecutarla hace falta cerrar el trío con evidencia suficiente y registrar sus configuraciones. Se mantienen RMSE, el margen operativo del 5 % frente a persistencia y la protección del test. No se promete que una red supere esa referencia.

### 3.4.5. Correspondencia y estado de cierre

| Requisito de la guía | Evidencia de esta revisión | Estado y límite |
|---|---|---|
| Cadenas completas, fechas y resultados actuales | Doce búsquedas web públicas: cuatro de descubrimiento, cuatro dirigidas a títulos/identificadores y cuatro de recursos de indexación; 67 ocurrencias con repeticiones, registro de respuestas y hashes | Son búsquedas ejecutadas sobre recursos públicos, con una página por cadena; no son 67 artículos ni consultas nativas WoS/Scopus |
| Historial complementario conservado | Diez consultas G1–G10 del 02/11/2026, intentos, páginas, 118 ocurrencias y 106 repositorios distintos; comprobantes de acceso posteriores separados | Se preserva como corte histórico, sin sumarlo al flujo público actual ni tratar los cuatro fallos directos como cobertura científica |
| Procedencia WoS/Scopus y publicaciones IEEE/arXiv | Cinco revistas con coincidencias secundarias de título e ISSN en copias públicas WoS/Scopus de 2026; arXiv 2110.14317v2 corroborado en portada de autores; enlace Xplore 11058926 declarado por autores | Evidencia de revista e identidad documental, no verificación individual de artículos en los índices, consulta nativa, cobertura histórica WoS ni Q1; páginas arXiv/Xplore en vivo no consultadas |
| Inclusión, exclusión, flujo y responsable | Criterios retrospectivos, cribado histórico y revisado: 20 repositorios inspeccionados, siete textos evaluados y cinco incluidos; cinco README adicionales del descubrimiento actual como pistas | El flujo histórico y las nuevas pistas se distinguen; corpus sin nuevas inclusiones automáticas, sin exhaustividad ni segundo revisor humano acreditados |
| Matriz y evaluación de calidad | Cinco textos, resultados con ubicación y riesgos por dimensión | La inclusión no certifica validez; dos incorporaciones y S2 son literatura gris |
| Frecuencia y complejidad | Variantes, frecuencias ampliadas/sensibilidades, ganadores de S1 y conteos analíticos reproducibles | Corregido; no demuestra mínima complejidad global ni uso frecuente en todos los índices |
| Exactamente tres modelos básicos no redundantes | Propuesta MLP–LSTM–CNN1D poco profunda según guía 6.4; GRU alternativa recurrente | Extracción aplicada y configuraciones pendientes; no se presenta la TCN grande como CNN simple |
| Fichas técnicas del trío propuesto | Perfiles de fuente LSTM/TCN/encoder con referencias y análisis conservados | Faltan fichas aplicadas de MLP/CNN1D simple y configuraciones propias; las fichas fuente no las sustituyen |
| Comparación crítica y brecha | Desempeño separado por protocolo, robustez, escalabilidad y brecha avanzada/base cuantificada | Documentada sin ranking entre tareas, inferencia propia ni afirmación de novedad universal |
| Experimentos y baseline medido | Diseño futuro y separación del diagnóstico Ridge de EDA | Tres redes, piloto, benchmark y test siguen reservados para E2 |

La fase actual sigue la búsqueda pública y la comprobación posterior de procedencia descritas en §3.1. Las cinco coincidencias secundarias corresponden a *Journal of International Financial Markets, Institutions and Money*, *Frontiers in Physics*, *Economics Letters*, *Finance Research Letters* e *IEEE Access*. No sustituyen la verificación individual de los artículos ni certifican Q1. Las listas inspeccionadas no permiten juzgar actas de congresos: la ausencia de WSDM/ICDM en ellas no prueba falta de indexación.

Los comprobantes se conservan en el [registro público de resultados](../_static/literature_e1/public_web_verification_20261103/discovery_records.json), el [flujo de la fase actual](../_static/literature_e1/public_web_verification_20261103/flow_summary.json), la [evidencia secundaria de revistas](../_static/literature_e1/public_web_verification_20261103/secondary_journal_evidence.json) y la [procedencia documental](../_static/literature_e1/public_web_verification_20261103/publication_provenance.json). Las asociaciones rechazadas y las pistas pendientes no se convierten en textos incluidos. Permanecen cinco estudios, 81 valores de autores y el trío final pendiente.

## 3.5. Fundamentos bibliográficos y contexto de la revisión

### 3.5.1. Búsqueda documental efectuada y alcance de las citas

La consulta del **2 de noviembre de 2026** incorporó treinta referencias: siete sobre Bitcoin, siete de fundamentos de redes neuronales y dieciséis de volatilidad, diagnóstico y evaluación. Se consultaron repositorios públicos de autores, documentación de software académico y bibliografías de artículos y libros. Para cada entrada se conservan el enlace consultado, los campos corroborados y el nivel de acceso en el [registro de verificación](../_static/bibliografia/verification_30.csv) y el [catálogo detallado](../_static/bibliografia/references_30.json).

El catálogo vigente conserva las treinta referencias iniciales y las tres fuentes de las fichas, y añade S6/S7 recuperadas el 03/11/2026: 35 referencias en la tabla de §3.1.2.4 y su [versión CSV](../_static/literature_e1/references_access_35.csv). La recuperación dirigida de registros previos amplió el corpus a cinco textos. La fase pública posterior de §3.1 ejecutó doce consultas web y comprobaciones de procedencia e indexación secundaria, sin incorporar otros estudios ni consultar internamente WoS o Scopus.

La red del entorno bloqueó el acceso directo a Crossref y arXiv. Por ello no se afirma haber consultado sus interfaces ni leído treinta originales completos. La entrada de Huang y colaboradores conserva su verificación previa; su acceso directo actual también está limitado. Las verificaciones nuevas de las otras referencias proceden de las fuentes accesibles indicadas en el registro. Una referencia identificada en otra bibliografía se distingue de un artículo original leído.

Los trabajos anteriores a 2019 se incluyen como fundamentos y antecedentes expresamente justificados. No se cuentan automáticamente como estudios recientes elegibles para calcular frecuencia de modelos base. Los registros de acceso a metadatos tampoco sustituyen el flujo complementario documentado en §3.1.

### 3.5.2. Antecedentes de volatilidad de Bitcoin

Katsiampa estudia la estimación de volatilidad de Bitcoin mediante especificaciones GARCH, mientras Dyhrberg relaciona su análisis de volatilidad con oro y dólar {cite:p}`katsiampa2017,dyhrberg2016`. Estos antecedentes justifican contrastar las redes con referencias financieras y delimitar la variable que se estima; no demuestran por sí solos qué arquitectura neuronal debe seleccionarse.

Los estudios asociados a órdenes de compra y venta, información de Twitter y flujos de información a frecuencia de minuto plantean conjuntos de información diferentes {cite:p}`guo2018,akbiyik2023,barjasic2021`. Sus títulos, datos documentados y referencias permiten reconocer líneas de investigación pertinentes, pero no autorizan atribuir a OHLCV diario mejoras obtenidas con otras señales. El artículo de Akbiyik y colaboradores se recuperó como copia de autores; para parte de los antecedentes se corroboraron metadatos en sus bibliografías y no se declara lectura del original.

Huang y colaboradores aportan un antecedente de pronóstico de volatilidad con técnicas de aprendizaje automático. Herremans y Low se centran en picos de volatilidad e información de transacciones de grandes tenedores y CryptoQuant mediante modelos Synthesizer Transformer {cite:p}`huang2024,herremans2025`. Pronosticar picos, una medida realizada intradiaria o el siguiente valor de una desviación móvil diaria constituye tareas diferentes. La pertinencia temática no acredita igualdad de objetivo, datos, horizonte o protocolo.

### 3.5.3. Definiciones de volatilidad y referencias financieras

ARCH formaliza varianza condicional cambiante y GARCH amplía esa estructura incorporando dinámica de la varianza {cite:p}`engle1982,bollerslev1986`. Son fundamentos econométricos y referencias de contexto; no ocupan los puestos reservados a modelos de Deep Learning.

La literatura de volatilidad realizada y de modelos HAR estudia construcciones y dependencias a distintas escalas {cite:p}`andersen2003,corsi2009`. Es necesario conservar la distinción entre varianza condicional, volatilidad realizada construida con alta frecuencia y desviación estándar móvil de retornos diarios. La medida definida en este proyecto no se convierte en cualquiera de las anteriores por compartir la palabra «volatilidad». Cambiar ventana, frecuencia o anualización también cambia la escala de las métricas.

### 3.5.4. Fundamentos de las familias neuronales y perfiles fuente

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

Las 35 referencias amplían el soporte documental del Entregable 1 y permiten distinguir problema financiero, arquitectura y método de evaluación. No equivalen a 35 experimentos ni a 35 estudios elegibles de modelos base de Bitcoin. Los textos de fundamentos no se suman a una frecuencia de uso en el problema aplicado. Las cinco fuentes completas se distinguen por calidad y procedencia. Los perfiles fuente y su comparación están documentados en §§3.3–3.4. Las fichas aplicadas de MLP y CNN1D simple de la propuesta actual aún deben completarse. Permanecen abiertos la cobertura bibliográfica, el orden de los candidatos por frecuencia y complejidad y el respaldo de las configuraciones propias de E2. La cantidad de citas no acredita por sí sola la identificación definitiva exigida por la guía.
