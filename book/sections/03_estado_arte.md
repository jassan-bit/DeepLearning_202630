# 3. Revisión del Estado del Arte y Extracción de los Tres Modelos Base

La revisión debe identificar exactamente tres modelos DL básicos usados como referencias comparativas en la literatura de volatilidad de Bitcoin. La bibliografía vigente reúne **39 referencias** y mantiene **seis textos completos en el corpus aplicado de Bitcoin**; son unidades distintas. S8, Miura, Pichl y Kaizoji (2019), compara MLP, CNN, LSTM y GRU independientes {cite:p}`miura2019`. La ampliación metodológica añade **T1, un artículo editorial completo de Oxford sobre redes feedforward**, **T2, código primario de autores que verifica convoluciones temporales 1D**, y **T3, una publicación de Quantitative Finance identificada por el CV del autor, con original pendiente**. Sus diferentes niveles de acceso se conservan; no son tres textos completos nuevos ni nueve estudios de Bitcoin.

El trío implementado **MLP–LSTM–CNN1D poco profunda**, orientado por la guía 6.4, permanece como adaptación propia. Bucci (2020) refuerza el fundamento de la MLP como referencia sencilla de volatilidad financiera {cite:p}`bucci2020realized`. El código de HARNet verifica una CNN1D causal en otro mercado {cite:p}`reisenhofer2022harnet`; no revela la dimensionalidad de la CNN de S8 ni convierte la red del proyecto en una réplica. La identificación editorial de DeepVol {cite:p}`morenopino2024deepvol` no se usa para atribuirle parámetros o resultados del original no leído.

La revisión del **03/11/2026** corrigió motivos de cribado, recuperó dos textos adicionales y completó la extracción de los hiperparámetros ganadores de S1, que sí aparecen en su apéndice. Se conservan los cortes anteriores y se explicitan fuentes grises, parámetros desconocidos y límites de las métricas. Aquella revisión documental del 03/11/2026 no entrenó redes ni consultó test; la ejecución posterior de desarrollo se registra por separado.

La nueva lectura se realizó el **04/11/2026, hora de Bogotá**, con registros UTC del 05/11, después del benchmark ya examinado. La referencia es anterior al experimento por su publicación de 2019, pero su recuperación en esta revisión es posterior: **no se reescribe como fundamento de una selección que ya estaba cerrada**. Se actualiza la evidencia documental sin nuevos entrenamientos, pronósticos o selección por test.

El problema propio es **regresión temporal, tipo 4, ruta 6.4**: OHLCV rezagadas sin historia explícita de la volatilidad objetivo. Las tareas de las fuentes no coinciden automáticamente con ese contrato. El corte documental del 03/11/2026 está en el {download}`informe del capítulo 3 <../_static/e1/revision_estado_arte_20261103.md>` y su [matriz histórica de requisitos](../_static/e1/revision_estado_arte_20261103.json); el estado propio de las tres fichas implementadas se actualiza en §§3.3–3.4.

La estrategia mantiene la solicitud del usuario: **búsqueda web pública y verificación posterior de procedencia e indexación**. La fase de doce consultas y comprobación secundaria de cinco revistas conserva sus registros. Las recuperaciones posteriores verifican el capítulo S8 y el artículo editorial T1, identifican código primario T2 y la publicación T3. La [bitácora de esta ampliación](../_static/literature_e1/journal_update_20261105/search_and_access_log.json) distingue recuperación primaria, candidatos pendientes y consultas bloqueadas. No acredita búsquedas nativas WoS/Scopus/IEEE Xplore/arXiv, registros individuales en los índices o Q1. Los catálogos de 35 y 36 referencias son cortes históricos; el actual contiene 39.

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

La nueva etapa de recuperación está documentada en el {download}`informe complementario <../_static/literature_e1/recovery_20261105/README.md>` y la [procedencia primaria de S8](../_static/literature_e1/recovery_20261105/primary_source.json). Incluye las cadenas y fuentes efectivamente recuperadas, sus fechas y hashes. Sus resultados de descubrimiento no se suman a las consultas anteriores como si fueran artículos distintos; solo S8 pasa a esta actualización del corpus científico.

### 3.1.2. Fuentes consultadas y verificación de procedencia e indexación

**WoS y Scopus son índices**, IEEE Xplore es una biblioteca digital y arXiv un repositorio de preprints. Un documento puede estar asociado a varias fuentes a la vez. Se verifica por separado: identidad del trabajo; publicación o identificador; presencia de la revista en un listado; registro individual del artículo; y cuartil del año/categoría. Un DOI, el nombre de una editorial o un resultado web no prueban todos esos niveles.

#### 3.1.2.1. Documentos y enlaces públicos comprobados

| Trabajo | Fuente pública comprobada | Hallazgo | Alcance de la comprobación |
|---|---|---|---|
| Akbiyik et al. (2023) | [PDF de autores](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/paper.pdf), portada | arXiv **2110.14317v2**, DOI 10.1145/3539597.3570387 y ACM WSDM 2023 | Identidad e identificador corroborados en copia pública de autores; registro arXiv en vivo no consultado. |
| Herremans y Low (2025) | [README de autores](https://github.com/dorienh/bitcoin_synthesizer/blob/8570f4e8bdcfe1dd609e5f094d3d5b0059a810dc/README.md#L3-L7) | IEEE Access 13, 117788–117807; [registro Xplore 11058926](https://ieeexplore.ieee.org/document/11058926/) | Enlace de registro declarado por autores; la página de Xplore no se abrió. |
| Guo et al. (2018) | [README de autores](https://github.com/weilai0980/mixture_models_crypto/blob/7aeaf66bb4438d022ae0cf2af2ffec317a8d068f/README.md#L7-L15) | IEEE ICDM 2018, pp. 989–994 | Publicación declarada por autores; registro Xplore específico no corroborado en el material nuevo. |
| Miura, Pichl y Kaizoji (2019) | [Capítulo maquetado de Springer, copia conservada en un repositorio académico](https://raw.githubusercontent.com/Anton21a/Thesis/8bc3aa3c96fd214b34b914d2057f4339f2c8ed1e/ML%20articles/Artificial%20Neural%20Networks.pdf) | *Advances in Neural Networks—ISNN 2019*, LNCS 11554, pp. 165–172; DOI impreso **10.1007/978-3-030-22796-8_18** | Texto completo y Tabla 1 leídos y verificados; identidad de la publicación acreditada por el PDF. No se abrió el registro oficial del artículo en índices ni se comprobó Q1. |

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

La ampliación incorpora dos revistas adicionales, cotejadas por **título e ISSN en las mismas copias fijadas**. En T1 la portada editorial acredita Oxford University Press, revista y DOI; en T3 el CV primario del autor acredita la referencia y su enlace DOI. El cotejo de los índices sigue siendo secundario:

| Revista nueva | Editorial e ISSN | WoS en la copia de 2026 | Scopus en la copia de 2026 |
|---|---|---|---|
| Journal of Financial Econometrics | Oxford University Press · 1479-8409 | SSCI | 145695 · 2005–2026 |
| Quantitative Finance | Taylor & Francis · 1469-7688 | SSCI y SCIE | 16062 · 2001–2026 |

Las [filas exactas y su procedencia](../_static/literature_e1/journal_update_20261105/journal_index_matches.json) y su [CSV](../_static/literature_e1/journal_update_20261105/journal_index_matches.csv) permiten repetir el cotejo de las siete revistas. **Publicación, texto leído, código de autores, indexación de revista e indexación individual no son equivalentes.** Elsevier es una editorial; WoS y Scopus son índices. La ampliación tampoco asigna cuartiles.

Los [candidatos editoriales pendientes](../_static/literature_e1/journal_update_20261105/editorial_candidates.json) conservan Huang et al. (2024), D’Amato et al. (2022) y Wang et al. (2023), vinculados bibliográficamente a revistas Elsevier. Sus originales y comparadores no se dan por leídos. PreBit, de Expert Systems with Applications, tiene evidencia primaria de autores de CNN para movimientos extremos del precio: permanece fuera de la regresión de volatilidad continua. No se cambia el denominador de estudios por encontrar un DOI o un título editorial.

La [matriz de las once referencias aplicadas](../_static/literature_e1/public_web_verification_20261103/document_platform_matrix.json) conserva estados por documento; los [extractos de filas, ISSN, cobertura y fuentes](../_static/literature_e1/public_web_verification_20261103/secondary_journal_evidence.json) identifican URLs fijadas, commits y hashes. La [procedencia documental](../_static/literature_e1/public_web_verification_20261103/publication_provenance.json) recoge los identificadores y las asociaciones rechazadas. El {download}`informe de verificación <../_static/literature_e1/public_web_verification_20261103/report.md>` permite revisar el recorrido completo.

#### 3.1.2.3. Control de identidad

Un mirror atribuía a Herremans un PDF y el ID arXiv:2211.00759; al leer la portada se comprobó que era otro trabajo, de Reijnen y colaboradores. Se descartó esa asociación. La pista secundaria 2211.08281 permanece sin corroboración oficial. La concordancia de título, autores, DOI y portada prevalece sobre el nombre de un archivo o una etiqueta del buscador.

(tabla-referencias-e1)=
#### 3.1.2.4. Catálogo bibliográfico vigente: 39 referencias

Las tablas conservan las 36 entradas previas y añaden tres referencias metodológicas con su propio nivel de evidencia. La [versión vigente de 39 referencias](../_static/literature_e1/journal_update_20261105/references_access_39.json) no modifica las fechas ni atribuye nuevas lecturas a las entradas anteriores. El [catálogo de 36](../_static/literature_e1/recovery_20261105/references_access_36.json) permanece como antecedente.

El catálogo reúne **39 referencias: doce aplicaciones a Bitcoin, ocho fundamentos de redes neuronales, dieciséis referencias de volatilidad, diagnóstico y evaluación, y tres respaldos metodológicos adicionales**. Los catálogos de 33, 35 y 36 entradas siguen como cortes históricos. Los candidatos de recuperación pendientes se presentan como registros de búsqueda, fuera de este aumento de citas y de las frecuencias del corpus BTC.

El **enlace bibliográfico** identifica el trabajo, su DOI, su preprint o el registro donde se documentó la referencia. La columna **Fuente y evidencia consultada** indica qué material se utilizó realmente: texto completo de autores, código, documentación o una referencia localizada en otra bibliografía. Un DOI registrado no acredita su resolución directa ni la lectura del artículo original. Los manuscritos de Zeri y Gao conservan las precisiones sobre revisión por pares y año editorial en los perfiles fuente archivados y la matriz de estudios.

##### 3.1.2.4.1. Aplicaciones a Bitcoin (12 referencias)

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
| {cite:p}`miura2019` | *Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series* · [DOI impreso](https://doi.org/10.1007/978-3-030-22796-8_18) | [Capítulo completo, pp. 165–172](https://raw.githubusercontent.com/Anton21a/Thesis/8bc3aa3c96fd214b34b914d2057f4339f2c8ed1e/ML%20articles/Artificial%20Neural%20Networks.pdf): portada y Tabla 1 verificadas en la copia del capítulo publicado; no se acredita consulta del índice o resolución en vivo del DOI. | S8: MLP de cuatro capas y CNN de dos capas más Dense como comparadores independientes de RV de Bitcoin; parámetros y dimensión CNN parcialmente no reportados. |

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

**Respaldo metodológico adicional, separado del corpus Bitcoin:**

| Referencia | Evidencia consultada | Aporte y límite |
|---|---|---|
| {cite:p}`bucci2020realized` · Journal of Financial Econometrics · [DOI](https://doi.org/10.1093/jjfinec/nbaa008) | Artículo editorial completo, pp. 502–531; estructura y Tablas 2–5 verificadas | FNN/FNNX independientes de una capa oculta para volatilidad mensual del S&P 500. Respalda el papel de una MLP básica, sin replicar el objetivo o la receta de E1. |
| {cite:p}`reisenhofer2022harnet` · HARNet | Código primario de autores, versión fijada; original y revista pendientes | Verifica Conv1D causal, kernels, dilataciones y un preset financiero. No acredita métricas del artículo ni una CNN independiente en Bitcoin. |
| {cite:p}`morenopino2024deepvol` · Quantitative Finance · [DOI](https://doi.org/10.1080/14697688.2024.2387222) | CV primario del autor con referencia y enlace DOI | Acredita identidad de una publicación de revista; el original, sus parámetros y métricas permanecen pendientes. |

El catálogo vigente se descarga en [CSV](../_static/literature_e1/journal_update_20261105/references_access_39.csv), [JSON](../_static/literature_e1/journal_update_20261105/references_access_39.json) y [BibTeX](../_static/literature_e1/journal_update_20261105/references_access_39.bib). Los cortes anteriores de [36](../_static/literature_e1/recovery_20261105/references_access_36.csv) y [35](../_static/literature_e1/references_access_35.csv) se conservan. Las citas completas están en la [bibliografía del libro](referencias.md).

**Recuperación dirigida posterior a la auditoría del 04/11/2026.** Se ejecutó otra etapa de descubrimiento público, separada del flujo original: **13 cadenas literales, 18 páginas y 113 ocurrencias en 92 repositorios únicos**; se inspeccionaron nueve raíces/README y siete archivos de código/notebook sin ejecutarlos. El [registro de consultas](../_static/e1_audit_improvements_20261104/literature/search_log.json) y la [evaluación de candidatos](../_static/e1_audit_improvements_20261104/literature/candidate_assessment.csv) documentan este recorrido. No se suman estas ocurrencias al PRISMA original sin una deduplicación conjunta: son dos etapas con denominadores propios.

En aquella recuperación se verificaron pistas de una CNN1D independiente y una MLP TS-only, pero **no se obtuvo un manuscrito científico completo nuevo**: su resultado histórico fue 35 referencias, cinco textos y 0/5 para MLP/CNN1D simple. La [matriz de siete fuentes](../_static/e1_audit_improvements_20261104/literature/dataset_comparability.csv) y el {download}`informe de esa etapa <../_static/e1_audit_improvements_20261104/literature/report.md>` conservan ese alcance. La inclusión posterior de S8 es otra recuperación primaria, no una reinterpretación de esos README ni prueba de que sus fuentes tengan el mismo dataset/σ7 del proyecto.

### 3.1.3. Periodo de búsqueda

El intervalo preferente de publicación se actualiza a **01/01/2019–04/11/2026**, ocho años calendario; el corte previo terminaba el 03/11. S8, publicado en 2019, entra en ese intervalo. Las consultas web no aplicaron un filtro verificable de año editorial: se comprueba durante la identificación documental. La actualización de un repositorio no sustituye al año del documento. Guo (2018), Katsiampa (2017) y Dyhrberg (2016) conservan su papel de antecedentes/contexto.

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
| Periodo | Preferencia 01/01/2019–04/11/2026, conservando el corte previo | Los anteriores se justifican como antecedentes/fundamentos; los sin fecha no acreditan recencia. S8 tiene fecha editorial impresa de 2019; la fecha de su recuperación no la modifica. |

Se distinguen pertinencia, recuperabilidad y calidad. Que un texto esté incluido para extracción exploratoria no certifica su validez ni basta para sostener que respalda uno de los tres modelos básicos definitivos. La preferencia Q1 requiere verificar índice, revista, año y categoría; no se deduce de que exista un DOI o un repositorio, ni se traslada automáticamente a congresos.

Un estudio cuenta como máximo una vez por familia. Preprint, artículo y código del mismo trabajo se deduplican por identidad documental; las dos variantes MLP de S8 cuentan como **un estudio MLP**, y su CNN no se suma a CNN1D explícita. El flujo anterior deduplicaba repositorios, no registros nativos de bases. El catálogo de 39 citas y los seis textos aplicados a Bitcoin cumplen funciones distintas. T1/T2/T3 fundamentan transferencia metodológica con niveles distintos y no entran en el denominador BTC.

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

En esa fase se inspeccionaron además cinco README de candidatos, con [URLs fijadas, hashes y líneas pertinentes](../_static/literature_e1/public_web_verification_20261103/candidate_sources.json). Se mantuvo entonces el corte de cinco textos y 81 cifras de autores. **La nueva incorporación S8 exige lectura completa y extracción verificable**, documentadas aparte; no se cuentan como estudios los repositorios adicionales sin ese respaldo.

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

La incorporación S8 se muestra en un flujo complementario, separado del anterior:

![Recuperación complementaria de S8 y actualización de cinco a seis textos incluidos](../_static/literature_e1/recovery_20261105/retrieval_flow.svg)

**Recuperación editorial y metodológica posterior.** El [registro nuevo](../_static/literature_e1/journal_update_20261105/search_and_access_log.json) conserva las búsquedas focalizadas, fuentes, fechas y hashes. Su resultado verificable es **un artículo completo nuevo sobre volatilidad financiera (T1), un código primario de autores (T2) y una identidad editorial acreditada en CV (T3)**. No se suman como tres artículos leídos ni como aplicaciones adicionales a Bitcoin. Los originales Elsevier que no pudieron recuperarse siguen pendientes; el fallo de acceso no significa que no existan ni que carezcan de indexación. No se suman repositorios, citas y textos completos como si fueran registros independientes de PRISMA.

La [procedencia primaria](../_static/literature_e1/recovery_20261105/primary_source.json) permite identificar la pista, la copia completa, sus hashes y la decisión. El paso de cinco a seis es una actualización del corpus incluido; **no convierte los 106 repositorios anteriores ni los nuevos candidatos de software en artículos científicos ni construye un PRISMA acumulado sin deduplicación conjunta**.

### 3.1.6. Evaluación de la calidad de los estudios

La evaluación usa dimensiones y ubicaciones de evidencia, sin una puntuación agregada arbitraria. Distingue lo declarado por el texto, lo comprobado en el código de autor y lo no recuperado; no certifica ausencia de fuga ni reproducción de resultados.

| Estudio | Arquitectura y protocolo | Repetición e inferencia | Riesgos y alcance |
|---|---|---|---|
| S1. Akbiyik et al. | Comparadores, entradas, tuning y código documentados | Veinte corridas y dispersión en Tabla 2 | Test de 48 días; verificar familia de contrastes y multiplicidad; inicializaciones no son mercados independientes |
| S2. Zeri | Declara escalado solo con entrenamiento, purga y embargo; redes y referencias no reciben igual historia | Tres semillas con predicciones promediadas y DM | Literatura gris; no confundir promedio de predicciones con dispersión entre corridas ni atribuir toda ganancia a arquitectura |
| S5. Gao et al. | LSTM pequeña y densa lineal descritas en notebook; año editorial externo pendiente | No se acreditan varias semillas ni intervalos | Fechas y objetivo desplazado requieren crítica; posible cruce de cortes sin purga explícita; Dense(1) sin capa oculta no es MLP no lineal |
| S6. Wang et al. (s. f.) | Transformer OHLCV independiente y nueve variantes; dos capas, d_model=32, cuatro cabezas; ancho FFN/readout no reportados | Una semilla declarada; DM expresamente no ejecutado | Fuente bruta/proxy y cohortes sin reconciliar; afirmación de mejor MAE contradice Tabla4; RMSE/R² no concuerdan bajo una cohorte común convencional; sin año de publicación ni código de entrenamiento |
| S7. Alakhouch Idriss (2026) | LSTM64→LSTM32→Dense16→Dense1 declarada; doce atributos en PDF frente a ocho en código | Sin varias semillas o dispersión acreditadas | Aparente fuga al ajustar escaladores antes del split; objetivo30 días/anualizado distinto; recuentos y early stopping del informe/código no reconciliados |
| S8. Miura, Pichl y Kaizoji (2019) | Capítulo publicado identificable; MLP/CNN independientes, pero sin anchos, kernels, optimizador o dimensión CNN suficientes para réplica | Declara CV temporal de diez folds/100 corridas y test con 30 inicializaciones; Tabla 1 informa varianzas | No publica fechas de cortes, tamaños train/test o purga; historias de entrada diferentes entre modelos. RV de tres horas y precio Bitstamp difieren de σ7/OHLCV diario. No se acredita indexación o Q1. |

Se conserva la [valoración por dimensión](../_static/literature_e1/quality_assessment.csv) y su [JSON](../_static/literature_e1/quality_assessment.json). Ausencia de descripción no equivale a ausencia del defecto; una prueba publicada tampoco garantiza calibración, comparabilidad o control de multiplicidad. Los detalles y la repercusión sobre la selección se desarrollan en §§3.2–3.4.

La [evaluación complementaria de S8](../_static/literature_e1/recovery_20261105/quality_assessment.json) añade estos desconocidos sin sobrescribir la valoración histórica de S1–S7. Se distingue existencia de un comparador independiente, que sí se verifica, de reproducibilidad exacta, que sigue incompleta.

La consulta de una base devuelve registros; la inclusión exige recuperar y valorar documentos. Para la nueva búsqueda se mantendrá la evaluación de arquitectura/comparador, protocolo, partición y fuga aparente, repeticiones, dispersión e inferencia. Indexación y cuartil se comprobarán por fuente, revista, año y categoría: no se deducen de un DOI ni se atribuyen automáticamente a congresos o preprints.

La verificación de procedencia añadida en esta fase no avala la calidad experimental. Un identificador correcto o una revista presente en una lista no sustituyen revisión de fugas, particiones, repeticiones, dispersión y comparabilidad. El cuartil Q1 permanece pendiente de evidencia específica por índice, año y categoría; no se deduce de las coincidencias secundarias de 2026.

## 3.2. Identificación de los tres modelos base de Deep Learning

El corpus vigente contiene **seis textos completos: S1, S2, S5, S6, S7 y S8**. S6/S7 mantienen sus límites de literatura gris; S8 aporta un capítulo de Springer publicado en 2019 y recuperado después del benchmark. La inclusión permite extraer lo reportado por autores, sin certificar su validez, indexación o equivalencia con el contrato propio.

### 3.2.1. Procedimiento de extracción

La guía exige recurrencia, sencillez, reproducibilidad y tres modelos no redundantes. La siguiente regla de revisión documental se registró **después** de inspeccionar las fuentes; no se presenta como un protocolo preregistrado:

1. **Revisar los comparadores de cada estudio incluido.** Exigir un texto identificable que muestre el uso independiente del modelo DL; separar propuestas avanzadas, híbridos, componentes internos y modelos lineales.
2. **Registrar la configuración por variante.** Extraer entrada, salida, tipo de arquitectura, profundidad, ancho, regularización, pérdida, optimización, entrenamiento y localización. Un campo no reportado permanece desconocido; no se completa por preferencia.
3. **Construir la tabla de frecuencias.** Contar un estudio por familia utilizada como referencia independiente, conservando sus variantes. La recurrencia de una familia no acredita el uso de una configuración concreta en todos esos estudios.
4. **Ordenar los candidatos por frecuencia y complejidad.** Comparar profundidad y parámetros con sus hipótesis, mostrando tanto el tamaño de origen como una representación común p=5, q=1. No utilizar RMSE de tareas distintas como desempate.
5. **Seleccionar exactamente tres modelos cuando la evidencia lo permita.** Justificar conjuntamente recurrencia, sencillez, reproducibilidad y suficiencia de fuentes. Para acreditar el cierre bibliográfico deben justificarse tres arquitecturas con respaldo; si falta evidencia, mantener esa acreditación pendiente sin omitir un candidato por preferencia. La ejecución propia autorizada no completa por sí sola ese requisito.
6. **Verificar sesgo inductivo y no redundancia del trío.** Explicar qué diferencias de representación son relevantes para el problema. Si los tres candidatos no satisfacen conjuntamente los criterios, ampliar la evidencia y conservar el cierre pendiente. Cuando falta una arquitectura concreta, la guía remite al procedimiento de §3.2.5 y a la búsqueda del Entregable 2; una adaptación propia registrada no se convierte en una configuración publicada.

#### 3.2.1.1. Frecuencias, variantes y sensibilidad del corpus

La unidad es **un estudio por familia empleada como referencia independiente**. Dos variantes de una MLP, una cabeza densa interna o varias inicializaciones no crean estudios nuevos. La frecuencia es descriptiva del corpus recuperado, sin estimar prevalencia en todos los índices.

| Familia | Estudios vigentes | Corte actual | Corte anterior de cinco textos | Corte original de tres textos | Distinción necesaria |
|---|---|---:|---:|---:|---|
| LSTM, incluyendo BiLSTM | S1, S2, S5, S7, S8 | 5/6 | 4/5 | 3/3 | Unidireccional verificada en S1/S5/S7: 3/6; S8 no precisa direccionalidad |
| GRU | S1, S2, S8 | 3/6 | 2/5 | 2/3 | Alternativa recurrente elegible, no idéntica a LSTM |
| Encoder Transformer | S2, S6 | 2/6 | 2/5 | 1/3 | Dos fuentes grises, sin convertirlas en configuraciones mínimas |
| TCN | S1 | 1/6 | 1/5 | 1/3 | No se fusiona con CNN genérica o CNN1D simple |
| MLP independiente no lineal | S8 | 1/6 | 0/5 | 0/3 | Las variantes dropout/BN de S8 cuentan una vez; no se suman Dense1 de S5 o cabezas internas |
| CNN independiente, dimensión no reportada | S8 | 1/6 | 0/5 | 0/3 | Dos capas más Dense verificadas; no se infiere Conv1D |
| CNN1D simple identificada explícitamente | Ninguno | 0/6 | 0/5 | 0/3 | La adaptación propia 1D no modifica el reporte del artículo |

Las [frecuencias vigentes](../_static/literature_e1/recovery_20261105/family_frequencies.csv) y la [decisión actualizada](../_static/literature_e1/recovery_20261105/selection_update.json) conservan esa separación. Las [frecuencias anteriores](../_static/literature_e1/family_frequencies.csv) y la [reconstrucción histórica](../_static/literature_e1/model_frequency_review_20261103.json) siguen accesibles. Si se retira S6, cuya fecha editorial es desconocida, quedan cinco textos: LSTM 5/5, GRU 3/5, MLP y CNN genérica 1/5 cada una; ello no demuestra uso amplio ni cierra la dimensión CNN. Retirar S8 recupera exactamente el corte anterior de cinco.

La clase MLP localizada en el código de Guo y los componentes `dense` de Synthesizer no aumentan frecuencias sin comprobar su papel en un texto completo. La nueva frecuencia MLP procede de la **Tabla 1 de S8**, no de aquella clase ni de los experimentos propios.

#### 3.2.1.2. Complejidad de las configuraciones documentadas

La corrección del apéndice C de S1 cambia la valoración de complejidad: sus ganadores **sí están publicados**. La tabla muestra cálculos de capas descritas, sin instanciar redes:

| Configuración de autor usada como ejemplo | Profundidad/ancho identificados | Parámetros con la tarea de origen | Parámetros derivados con p=5, q=1 |
|---|---|---:|---:|
| LSTM S5 | Una capa, H=20 y cabeza lineal | 1.781 | 2.101, un sesgo por compuerta |
| LSTM S1 | Una capa, H=261; Tabla 5 | 299.724 | 279.010, un sesgo por compuerta |
| GRU S1 | Una capa, H=43; Tabla 5 | 10.158 con dos sesgos | 6.494 con dos sesgos |
| TCN S1 | Seis niveles, F=287, k=5, base de dilatación 4; Tabla 6 | 4.563.396, condicionado al bloque de dos convoluciones y un stack | 4.542.924 bajo las mismas hipótesis |
| SELF-ATT S2 | Dos encoder, d=48, FF=96, cuatro cabezas y M=96 posiciones | 49.249 | 47.521, conservando M=96 |
| MLP S8 | Cuatro capas, variantes dropout/BN; anchos y activaciones no reportados | No calculable de forma exacta | No calculable sin inventar anchos o interpretación de capas |
| CNN S8 | Dos capas más Dense; dimensión, canales y kernel no reportados | No calculable de forma exacta | La CNN1D H16/k3 propia es una adaptación, no el conteo de S8 |

Los [escenarios analíticos](../_static/literature_e1/model_complexity_review_20261103.csv) documentan fórmulas, sesgos, proyecciones y supuestos. El script `scripts/audit_literature_models.py` reproduce las frecuencias y conteos del corte histórico de cinco textos leyendo aquella extracción; no ejecuta frameworks, entrenamiento ni datos de mercado. La actualización de seis textos conserva su propia tabla y manifiesto. S6 no permite un total exacto porque faltan ancho feedforward y detalles de la cabeza/posiciones. En S7 se conservan por separado las hipótesis de 12 atributos del PDF y 8 del código.

Los conteos comunes controlan dimensiones de entrada/salida, pero no igualan capacidad, datos o cómputo. GRU tiene menos pesos que la LSTM H261 de S1, mientras la LSTM H20 de S5 es menor que esa GRU. La TCN ganadora no es una CNN1D pequeña: su campo receptivo condicional es 10.921 frente a 96 pasos observados. Reducirla a otra red sería una adaptación que debe justificarse, no una réplica del ganador.

#### 3.2.1.3. Estado de la selección

**Trío ejecutado en desarrollo, alineado con la guía 6.4:**

| Puesto | Familia | Entrada y papel | Evidencia y límite |
|---|---|---|---|
| P1 | MLP sencilla | OHLCV rezagadas aplanadas; referencia feedforward no lineal sin estado recurrente | S8 acredita la familia MLP independiente, 1/6. Su red de cuatro capas no es la MLP propia de una capa oculta H16. |
| P2 | LSTM unidireccional | Secuencia D×5; referencia con memoria recurrente | Familia 5/6; unidireccional verificada 3/6. La receta propia se documenta en §4.2; GRU 3/6 sigue como alternativa no ejecutada. |
| P3 | CNN1D poco profunda | Misma secuencia de covariables; filtros temporales compartidos | S8 acredita CNN independiente poco profunda a nivel de capas, 1/6, pero no su dimensión. La elección 1D causal es propia, orientada por el eje temporal y la guía 6.4. |

El [registro histórico de propuesta](../_static/literature_e1/guide_aligned_proposal_20261104.json) conserva la procedencia curricular y las lagunas existentes cuando se entrenó. La [actualización documental](../_static/literature_e1/recovery_20261105/selection_update.json) añade S8 después del benchmark, sin cambiar modelos, pesos, hiperparámetros o la selección previa de MLP/D7. LSTM tiene mayor recurrencia familiar y antecedentes pequeños; no se afirma que prediga mejor que GRU. La CV de 2024 ya expuesta no se vuelve anidada por añadir literatura.

Las configuraciones ejecutadas usan D=7/14/21/28 días y cinco covariables, objetivo fijo σ7, horizonte de un día y salida Softplus:

| Modelo propio | Arquitectura ejecutada | Learning rate | Parámetros entrenables |
|---|---|---:|---:|
| MLP | Aplanado D×5 → Dense16/ReLU/dropout 0,1 → Dense1 | 0,001 | 593/1.153/1.713/2.273 para D7/14/21/28 |
| LSTM | Una capa unidireccional H=32 → último estado/dropout 0,1 → Dense1; dos sesgos por compuerta PyTorch | 0,003 | 5.025 |
| CNN1D poco profunda | Dos convoluciones causales de 16 canales, kernel 3, sin dilatación ni residuales → media temporal → Dense1 | 0,001 | 1.057 |

La CNN1D propia tiene campo receptivo local de cinco posiciones antes de la media; no reproduce la TCN de S1 ni se atribuye a la CNN dimensionalmente no especificada de S8. El [método y ejecución](../_static/filipe_multihistory_20261104/method.json) conserva las 120 trayectorias, 1.440 segmentos y 2.880 ajustes; la revisión documental no añade corridas ni transforma sus resultados en evidencia bibliográfica.

**Perfiles documentales conservados de la revisión previa:**

| Estado | Familia | Justificación actual | Condición pendiente |
|---|---|---|---|
| Perfil documental recurrente | LSTM unidireccional | Mayor recurrencia familiar y arquitectura pequeña identificable en S5 | Justificar variante y adaptación a OHLCV sin transferir resultados |
| Perfil documental convolucional | TCN causal dilatada | Comparador independiente con filtros compartidos y dilatación | Verificar versión de biblioteca y supuestos de la fuente; no es la CNN1D simple propuesta |
| Perfil documental de atención | Encoder Transformer | Mecanismo de atención distinto, documentado en S2 y S6 | Corroborar procedencia y detalles de fuentes grises como antecedente suplementario |
| Alternativa del puesto recurrente | GRU | 3/6 y menor complejidad que algunas LSTM de autor | No descartarla solo por ser recurrente; tiene más apoyos que MLP/CNN y debe considerarse en una selección nueva. |

El [registro de selección previo](../_static/literature_e1/selection_evidence.json) se conserva y la [actualización](../_static/literature_e1/recovery_20261105/selection_update.json) distingue respaldo posterior y decisiones históricas. **La frecuencia sola ordena LSTM, GRU, encoder y luego MLP/CNN/TCN**: el trío ejecutado no se presenta como los tres ganadores de ese orden. Su justificación conjunta combina antecedentes aplicados, arquitecturas de capacidad controlada y tres mecanismos diferentes —interacciones por posición, memoria recurrente y filtros locales compartidos—, siguiendo la ruta 6.4. GRU es una alternativa legítima con mayor frecuencia que MLP/CNN; compartir compuertas recurrentes no basta para descartarla ni prueba redundancia. Compararla en una versión futura requeriría el mismo presupuesto y una evaluación nueva. Se conserva el benchmark actual para no reseleccionar tras observar test. S8 respalda las familias, pero no certifica mínima complejidad global ni una réplica de sus configuraciones.

La validación principal utiliza índices reales de `timeseries-cv`, doce bloques mensuales de 2024 y diez semillas por modelo e historia. Los [resultados de cuatro entradas](../_static/filipe_multihistory_20261104/analysis_cv/metrics_summary.csv) son evidencia propia de desarrollo, separada de cifras de autores y de test. El control relativo–absoluto inicial D7 conserva su alcance: mejorar esa representación no cierra las lagunas bibliográficas ni demuestra generalización universal.

(seleccion-diversidad-d7)=
**Selección justificada para el nuevo estudio D7.** La [decisión fechada](../_static/literature_e1/selection_20261105/decision.json) combina los criterios de frecuencia, sencillez y diversidad de §3.2 de la guía mediante una selección condicionada: una referencia no lineal sobre covariables aplanadas, una recurrente y una convolucional local. No afirma que el trío ocupe los tres primeros puestos del conteo global.

Dentro del puesto recurrente se elige **LSTM, presente en 5/6 estudios**, frente a GRU en 3/6. GRU sigue siendo una alternativa válida: reservar un solo puesto recurrente evita que la comparación se concentre en dos variantes con estado y compuertas, sin afirmar que sus funciones sean idénticas. **MLP** aporta interacciones no lineales por posición sobre la ventana aplanada; **CNN1D** aporta filtros locales compartidos y un campo receptivo acotado. La CNN independiente y la MLP tienen respaldo primario de familia en S8, una ocurrencia cada una; esa cobertura limitada se conserva.

La sencillez de las implementaciones se comprueba con **593 parámetros MLP, 1.057 CNN1D y 5.025 LSTM para D7**, no atribuyendo esos conteos a los artículos. La dimensión temporal 1D se justifica por el contrato de entrada $7\times5$, sin ejes espaciales, y la receta convolucional permanece como adaptación declarada. El antecedente TCN causal de S1 no se fusiona con la CNN de S8 para inventar una especificación de autor. Encoder y TCN son candidatos documentados; su exclusión del trío no prueba que sean siempre más complejos o inferiores.

La [matriz de candidatos](../_static/literature_e1/selection_20261105/candidate_matrix.csv) y la {download}`justificación completa <../_static/literature_e1/selection_20261105/justification.md>` separan evidencia primaria, decisiones propias y alternativas. Esta decisión es **posterior al benchmark histórico y anterior a las nuevas corridas del estudio D7**. Las recetas se mantienen fijas, con su búsqueda histórica igualada y su transferencia documentadas en §3.2.5; no se seleccionan otra vez mirando test. Aquella decisión conserva su corte de seis textos BTC y 36 referencias. La ampliación metodológica T1/T2/T3 se documenta después y eleva a 39 la bibliografía citada, sin alterar las frecuencias BTC ni reescribir la selección histórica.

### 3.2.2. Criterios que debe cumplir cada modelo base

Los cuatro criterios de la guía se evalúan por modelo, junto con la suficiencia de las fuentes. Disponer de una ficha no implica haberlos satisfecho:

| Criterio | Evidencia disponible | Estado y límite |
|---|---|---|
| Arquitectura simple, reproducible y ampliamente documentada | S5 identifica una LSTM pequeña; S1 y S2 describen configuraciones y las fichas separan fuente, cálculo y adaptación | La TCN ganadora de S1 es grande; falta su versión de biblioteca y S6 no permite reconstruir un total exacto. Las configuraciones propias de desarrollo están implementadas y contadas en §4.2; no reproducen automáticamente los ganadores de estas fuentes. |
| Uso frecuente como referencia en la literatura del problema | Frecuencias por estudio y cortes de §3.2.1.1 | LSTM5/6 y GRU3/6 superan MLP1/6/CNNgenérica1/6; una fuente no demuestra uso amplio. Conv1D explícita permanece 0/6. |
| Capacidad de entrenarse bajo el mismo pipeline y partición que los otros modelos base | El contrato propio fija información OHLCV, objetivo, horizonte y evaluación cronológica | Las cuatro historias D=7/14/21/28 comparten ventanas OHLCV, cortes, reglas causales, representación principal `relative_window` y diez semillas por familia. Solo el ciclo inicial D7 incluyó el control `absolute_log`; no se ejecutaron dos representaciones para las cuatro historias. Las configuraciones se registran en §4.2. |
| Ser un modelo de Deep Learning independiente | Se identifican MLP, CNN, LSTM, GRU, TCN y encoder; se separan propuestas e híbridos | La MLP y CNN de S8 tienen filas independientes. Dense1 lineal, cabeza interna, persistencia, Ridge o GARCH no ocupan un puesto DL. |

El mismo pipeline exige igualdad de información disponible, partición, preprocesamiento y evaluación, con transformaciones ajustadas solo al entrenamiento. No exige que vectorización y secuencia tengan idéntica forma. La implementación propia acredita compatibilidad técnica; la literatura aporta ahora existencia de familias, con cobertura y arquitectura incompletas, sin certificar generalización ni cerrar todos los requisitos de extracción.

### 3.2.3. Producto obligatorio: síntesis de los estudios incluidos

| Referencia | Problema | Dataset | Modelos DL | Métricas | Valores |
|---|---|---|---|---|---|
| S1. Akbiyik et al. (2023), Tabla 2 {cite:p}`akbiyik2023` | RV del día siguiente agregada de 96 retornos intradiarios | Bitcoin a 15 minutos y Twitter | LSTM, GRU y TCN; D-TCN es la propuesta avanzada | RMSE; media ± DE de veinte corridas | LSTM 2,37±0,19; GRU 4,23±5,61; TCN 2,58±0,07 |
| S2. Zeri (2026), Run B {cite:p}`zeri2026` | Log-RV futura de 30 minutos | BTCUSDT perpetuo, Crypto Lake | GRU, BiLSTM y encoder Transformer SELF-ATT | R² frente a media train y QLIKE | SELF-ATT 0,600/0,562; BiLSTM 0,586/0,582; GRU 0,582/0,564, respectivamente |
| S5. Gao et al. (manuscrito 2023), Tabla 4 {cite:p}`gao2023` | Medida móvil de 30 días desplazada siete días | Yahoo BTC-USD diario | LSTM de una capa y H=20; la densa lineal es contexto | RMSE y RMSPE | LSTM 0,074071 y 0,406084, respectivamente |
| S6. Wang et al. (s. f.), Tabla 4 {cite:p}`wangnd` | Volatilidad diaria denominada RV, sin fórmula suficiente para equipararla a σ7 | Bitcoin diario | Transformer OHLCV independiente; EGARCH–Transformer es híbrido | RMSE, MAE, QLIKE y R² del Modelo 1 | 1,4738; 0,4369; 0,4369; −0,7347, respectivamente, transcritos tal como aparecen |
| S7. Alakhouch Idriss (2026), Tabla 4.5 {cite:p}`alakhouch2026` | Medida futura de 30 días anualizada en el código | Bitcoin diario | LSTM 64→32 con cabeza Dense16→1; no una MLP independiente | RMSE, MAE y MAPE | 0,0358; 0,0254; 13,5 %, respectivamente |
| S8. Miura, Pichl y Kaizoji (2019), Tabla 1, p. 171 {cite:p}`miura2019` | Predicción de RV a partir de RV pasada; agregación de tres horas, horizonte numérico no precisado | Bitstamp BTCUSD minuto vía Kaggle; corpus 2012–2018 | MLP de cuatro capas (dropout/BN), CNN de dos capas + Dense, LSTM y GRU independientes | RMSE de test; varianzas reportadas por separado | MLP-dropout 0,0022067; MLP-BN 0,0021952; CNN 0,0021817; LSTM 0,0021960; GRU 0,0021778 |

**Límites de interpretación.** Las cinco filas anteriores mantienen sus advertencias: S1 usa RV×100 y otro objetivo; S2 promedia predicciones y usa R² frente a media train; S5 informa validación escalada; S6 conserva inconsistencias de métricas/particiones; S7 presenta escalado global y discrepancias entre informe/código. **S8 añade otro objetivo e historia de información:** RV es la suma de retornos logarítmicos minuto al cuadrado en intervalos de 180 minutos, sin raíz, frente a la desviación centrada de siete retornos diarios de E1. Las entradas son rezagos de RV, no OHLCV diario. Sus métricas no se ordenan con las de los otros estudios ni con RMSE de nuestro benchmark.

La [síntesis vigente de seis columnas](../_static/literature_e1/recovery_20261105/synthesis_six_columns.csv) contiene seis estudios; la [versión anterior](../_static/literature_e1/study_synthesis_six_columns_20261103.csv) conserva cinco. El [catálogo vigente de 39 referencias](../_static/literature_e1/journal_update_20261105/references_access_39.csv) es un producto distinto, con fundamentos, antecedentes y fuentes metodológicas además de los seis estudios BTC.

La [matriz completa](../_static/literature_e1/evidence_matrix.csv) conserva problema, dataset, arquitectura, resultados y localización. El [registro de métricas](../_static/literature_e1/reported_metrics.csv) distingue los 42 valores del corte original y los 39 transcritos de S6/S7: **81 valores de autores**, no 81 resultados independientes ni experimentos propios. No se corrigen silenciosamente cifras impresas inconsistentes ni se trasladan a la escala de E1.

**Transcripción verificable de S8, Tabla 1, p. 171.** El artículo declara diez folds temporales/100 corridas y 30 inicializaciones en test; no proporciona fechas train/test, purga o tamaños suficientes para reconstruir exactamente el protocolo. Las varianzas impresas no se presentan como DE:

| Modelo de S8 | Longitud de entrada | MSE de test | Varianza reportada de MSE | RMSE de test | Varianza reportada de RMSE |
|---|---:|---:|---:|---:|---:|
| MLP cuatro capas, dropout | 10 | 4,8704×10⁻⁶ | 1,3311×10⁻¹⁴ | 2,2067×10⁻³ | 6,8202×10⁻¹⁰ |
| MLP cuatro capas, BN | 10 | 4,8212×10⁻⁶ | 4,9997×10⁻¹⁴ | 2,1952×10⁻³ | 2,5325×10⁻⁹ |
| LSTM dos capas + Dense | 12 | 4,8233×10⁻⁶ | 2,0082×10⁻¹⁴ | 2,1960×10⁻³ | 1,0293×10⁻⁹ |
| GRU una capa + dos Dense | 5 | 4,7433×10⁻⁶ | 8,8148×10⁻¹⁵ | 2,1778×10⁻³ | 4,5781×10⁻¹⁰ |
| CNN dos capas + Dense | 6 | 4,7605×10⁻⁶ | 1,2904×10⁻¹⁴ | 2,1817×10⁻³ | 6,7603×10⁻¹⁰ |
| Ridge, referencia clásica | 6 | 4,6667×10⁻⁶ | No reportada | 2,1603×10⁻³ | No reportada |

La [Tabla 1 completa](../_static/literature_e1/recovery_20261105/miura_table1.csv) conserva también HARRV, SVM y los valores de CV. Ridge obtiene menor RMSE puntual de test que las redes; GRU tiene la menor MSE de CV entre los modelos comparados. No se deduce superioridad estadística de esas medias ni se convierte la longitud de entrada en horizonte. La [procedencia](../_static/literature_e1/recovery_20261105/primary_source.json) identifica el PDF y su hash; aquí se publican extracción y enlace, no se redistribuye el capítulo completo.

S3 —Herremans y Low— y S4 —Huang et al.— siguen como fuentes identificadas con texto completo pendiente. El código de autores de S3 no aumenta la frecuencia hasta verificar en el artículo qué redes se compararon. Las nuevas inspecciones de repositorios sin documento identificable también permanecen fuera del denominador.

### 3.2.4. Equivalencia funcional con referencias clásicas

En este problema continuo, el ejemplo clásico pertinente de la guía es **SVR lineal**. El modelo logístico corresponde a clasificación. La equivalencia se refiere al propósito de establecer una referencia de capacidad controlada; no a igualdad de funciones, supuestos o error:

| Referencia | Capacidad y supuestos | Papel comparativo |
|---|---|---|
| SVR lineal | Función afín sobre covariables vectorizadas, regularización y pérdida ε-insensible | Referencia clásica; no ocupa un puesto DL |
| MLP propuesta | Función no lineal sobre covariables rezagadas aplanadas | Referencia feedforward de capacidad controlada; S8 aporta apoyo aplicado de familia, no la receta propia H16 |
| LSTM | Estado recurrente, memoria y transformaciones no lineales con compuertas | Referencia DL secuencial si se controla su capacidad y presupuesto |
| CNN1D propuesta | Filtros temporales compartidos sobre la historia observable | Adaptación convolucional propia; S8 respalda CNN independiente, con dimensión no reportada |
| TCN | Filtros causales compartidos y campo receptivo definido | Antecedente documental; no se equipara a la CNN1D poco profunda propuesta |
| Encoder Transformer | Interacciones por contenido y posiciones en una ventana observable | Antecedente documental de atención, fuera de la propuesta actual de tres básicos |

Su comparación exige información, partición, objetivo y esfuerzo de ajuste comunes. Persistencia, Ridge y GARCH aportan contexto y no sustituyen los tres modelos DL. Los cuatro Ridge del bloque C son diagnósticos de EDA; el benchmark propio de redes y su evaluación final se informan en el capítulo 5. Los resultados de test no se usaron para elegir la configuración.

### 3.2.5. Arquitecturas no especificadas y búsqueda posterior

La guía exige que, si la literatura no define una arquitectura concreta —por ejemplo, informa solo «un MLP»—, no se elija por criterio personal: se debe aplicar la búsqueda de arquitectura del **Entregable 2, sección 14**, declarando el espacio antes de entrenar. Esa referencia corresponde a la guía suministrada. El experimento propio posterior declaró y ejecutó un espacio acotado, documentado en §4.2; no se presenta como cumplimiento del procedimiento completo de E2 §14, cuyo documento no se ha aportado.

En la extracción se identifica primero qué componente no está reportado y qué fuente o código lo respalda. Si continúa desconocido, se conserva como tal y se documenta la decisión pendiente. S6 no especifica lo necesario para un conteo exacto; S7 presenta discrepancias entre PDF y código. Estas lagunas no se resuelven imputando valores por preferencia.

S8 informa cuatro capas MLP y dos capas CNN más Dense, pero deja sin reportar anchos, activaciones, kernels y dimensión CNN. **No se rellenan con H16, ReLU o Conv1D de nuestra implementación**. Las recetas actuales ya habían sido entrenadas cuando se recuperó S8; la revisión fortalece el vínculo de familia sin simular una búsqueda anterior motivada por ese capítulo. La búsqueda exigida para completar una arquitectura futura debe declarar sus espacios antes de entrenar y mantener un conjunto de evaluación nuevo cuando corresponda.

Los ganadores de S1 **sí están publicados**: no procede tratar su ancho o profundidad como una arquitectura ausente. La versión de biblioteca pendiente y los supuestos del cálculo permanecen explícitos en §3.2.1.2 y en los perfiles fuente archivados. Reducir la TCN de seis niveles y F=287 sería una adaptación que exige justificación, no una réplica del ganador ni una forma de completar un dato omitido. La configuración propia se recoge en el [método de CV](../_static/dl_r2_cv_20261104/method.json): ancho 16 para MLP/CNN1D y 32 para LSTM. Son decisiones del proyecto, sin atribuirlas a S1 ni presentarlas como prueba de mínima complejidad global.

**Trazabilidad del ajuste de hiperparámetros.** La [búsqueda histórica registrada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/dl_benchmark_20261104/protocol_initial.json) recorrió dos anchos (16/32), dos learning rates (0,001/0,003), cuatro D y dos semillas (101/202) en cada una de las tres familias: **3×4×2×2×2=96 corridas**, con el mismo número de evaluaciones por familia. Su [tabla de selección](https://github.com/jassan-bit/DeepLearning_202630/blob/main/results/dl_benchmark_20261104/search_configurations.csv) contiene 48 configuraciones agregadas sobre esas dos semillas. Ese recuento no describe las diez repeticiones de la CV vigente.

| Familia | Receta histórica D7 transferida | Decisión vigente antes de test | Alcance de la transferencia |
|---|---|---|---|
| MLP | Ancho 16; learning rate 0,001 | Mismos H/LR para las cuatro D; MLP D7 elegida como baseline neuronal con CV | La representación relativa y el entrenamiento creciente modifican el procedimiento respecto a la búsqueda histórica. |
| LSTM | Ancho 32; learning rate 0,003 | Mismos H/LR para las cuatro D; historia D7 seleccionada con CV | No se atribuye un espacio HPO nuevo a cada mes ni una validación anidada que no se ejecutó. |
| CNN1D | Canales 16; learning rate 0,001 | Mismos H/LR para las cuatro D; historia D7 seleccionada con CV | El control absoluto es solo D7; la ampliación conserva la receta relativa fijada. |

La [extensión registrada](../_static/filipe_multihistory_20261104/protocol.json) mantuvo las recetas, diez semillas 4101–4110 y reglas de parada. No hubo una nueva búsqueda H/LR en D14/21/28; se comparó longitud D. El presupuesto histórico igual acredita esa etapa, no optimización bajo otra representación ni capacidades equivalentes. La revisión posterior de S8 no modifica esa historia: completa apoyo de familias, pero no atribuye retrospectivamente a la literatura configuraciones, resultados o decisiones del proyecto.

Los metadatos se leen por su identificador de experimento: `dl_benchmark_20261104` conserva la búsqueda/piloto históricos, `dl_r2_cv_20261104` el primer ciclo de Filipe D7 y `filipe_multihistory_20261104` la extensión de cuatro entradas y test final. El nuevo `e1_filipe_pilot_20261105` registra cinco semillas nuevas por modelo e historia para dimensionar repeticiones futuras sobre el calendario conocido; no sustituye esos registros ni vuelve prospectivo el benchmark anterior. Los campos históricos de dos semillas no se actualizan para fingir diez; cada manifiesto enumera sus semillas y resultados reales.

### 3.2.6. Familias de arquitecturas orientativas por tipo de problema

Las siguientes familias son una orientación de la guía, no resultados de frecuencia ni una selección automática. La elección final debe proceder de la literatura del problema y cumplir los criterios anteriores. El contrato actual usa covariables OHLCV temporales y una salida continua escalar:

| Tipo de problema | Familias orientativas | Aplicabilidad al proyecto y motivo |
|---|---|---|
| Clasificación y regresión tabular | MLP superficial, MLP profundo regularizado y red residual tabular | No es la ruta principal; la cronología condiciona la partición. S8 acredita MLP aplicada a RV autorregresiva, no convierte el contrato propio en tabular i.i.d. |
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

## 3.3. Fichas técnicas de los tres modelos base implementados

Las fichas corresponden a **MLP, LSTM y CNN1D poco profunda**, las tres redes ejecutadas con CV forward chaining de Filipe. Cada una desarrolla los nueve incisos de la guía: referencias, arquitectura, descripción técnica, problema, dataset, métricas, fortalezas, limitaciones y complejidad. Se distingue el fundamento general de la familia, su evidencia aplicada en el corpus y la configuración propia realmente implementada.

El contrato común contempla **D∈{7,14,21,28} días × p=5 canales OHLCV**, con objetivo σ7 poblacional de retornos logarítmicos, `ddof=0`, salida escalar y horizonte diario h=1. Cambiar D modifica la historia de entrada, no la definición del objetivo n=7. La representación principal resta al logaritmo de cada OHLC el logaritmo del último cierre observable de la ventana; aplica `log1p` al volumen y escala cada canal usando solo el entrenamiento. Y se divide por su media positiva de entrenamiento y se invierte antes de evaluar. La variante absoluta se conserva como control del ciclo inicial D7 (§4.2).

Los doce cortes mensuales usan pasado permitido de 2023 y 2024, embargo de D días y parada interna cronológica. En cada segmento se descarta el modelo interno y se reajusta una red nueva durante el número de épocas seleccionado. Las tres redes utilizan **AdamW, dropout 0,1, weight decay 0,0001, batch 32, clipping de norma 1, máximo interno de 100 épocas y paciencia 12**, sin planificador. Las configuraciones no se atribuyen a los autores de las referencias fundacionales.

Los perfiles históricos de LSTM/TCN/encoder permanecen en el [archivo del repositorio](https://github.com/jassan-bit/DeepLearning_202630/blob/main/references/literature_e1/source_profiles_archive_20261104.md). S8 aporta ahora MLP independiente y CNN genérica, una fuente por familia. **LSTM familiar alcanza 5/6, la unidireccional explícita 3/6 y CNN1D explícita permanece 0/6**. Las fichas separan el antecedente aplicado de las recetas propias; no se presentan como réplicas exactas ni como prueba de amplia recurrencia.

### 3.3.1. Perceptrón multicapa — MLP

#### 3.3.1.1. Referencias completas y papel comparativo

**Fundamento de arquitectura y entrenamiento.** Goodfellow, I., Bengio, Y. y Courville, A. (2016). *Deep Learning*. MIT Press, capítulos 6 y 8; [libro abierto](https://www.deeplearningbook.org/). Describe redes feedforward, retropropagación y optimización {cite:p}`goodfellow2016`. Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function*. *Mathematics of Control, Signals and Systems*, 2(4), 303–314. DOI: [10.1007/BF02551274](https://doi.org/10.1007/BF02551274) {cite:p}`cybenko1989`. El teorema de Cybenko corresponde a su clase de funciones y activaciones; no garantiza generalización de esta MLP ReLU.

**Antecedente aplicado y papel en el proyecto.** Miura, R., Pichl, L. y Kaizoji, T. (2019). *Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series*. En *Advances in Neural Networks—ISNN 2019*, LNCS 11554, pp. 165–172. DOI [10.1007/978-3-030-22796-8_18](https://doi.org/10.1007/978-3-030-22796-8_18), [texto completo consultado](https://raw.githubusercontent.com/Anton21a/Thesis/8bc3aa3c96fd214b34b914d2057f4339f2c8ed1e/ML%20articles/Artificial%20Neural%20Networks.pdf) {cite:p}`miura2019`. Su Tabla 1 compara dos MLP independientes de cuatro capas, con dropout o BN, en RV de Bitcoin. Esa evidencia aporta 1/6 apoyos de familia; la Dense lineal de S5 y las cabezas de LSTM siguen sin contar. La MLP propia es una adaptación de una capa oculta H16 sobre OHLCV, no una réplica del modelo de autor ni una anchura extraída del capítulo.

**Respaldo de revista para una referencia feedforward sencilla.** Andrea Bucci (2020), *Realized Volatility Forecasting with Neural Networks*, Journal of Financial Econometrics 18(3), 502–531, Oxford University Press, DOI [10.1093/jjfinec/nbaa008](https://doi.org/10.1093/jjfinec/nbaa008) {cite:p}`bucci2020realized`. El original editorial consultado define una FNN independiente con **una capa oculta**, ecuaciones 2–3 y Figura 3, pp. 507–508. La Tabla 2 selecciona cinco neuronas para FNN y una para FNNX; la Tabla 3 usa tres en ambas durante la submuestra de crisis. Esa estructura acredita la familia MLP sin confundirla con una cabeza densa o un híbrido GARCH.

La [extracción primaria](../_static/literature_e1/journal_update_20261105/bucci2020_primary_extraction.json) identifica páginas, selección de variables, arquitectura y evaluación. El modelo usa datos **mensuales del S&P 500**, con $\log\sqrt{\sum_i r_{i,t}^{2}}$ como objetivo y horizontes de uno y cinco meses; no es σ7 diaria ni utiliza las mismas covariables que E1. El trabajo informa 300 inicializaciones y selección del ancho por MSE de entrenamiento, no un procedimiento idéntico a nuestra validación. Los controles de escalado, versiones y purga no se completan por inferencia. El apoyo es metodológico y añade cero estudios a las frecuencias BTC.

**Resultados de autores, separados del benchmark propio.** Las Tablas 4–5 de Bucci, pp. 518 y 523, reportan:

| Muestra | Horizonte, meses | MSE FNN | MSE HAR | QLIKE FNN | QLIKE HAR |
|---|---:|---:|---:|---:|---:|
| Completa, 245 observaciones de test | 1 | 0,130 | 0,115 | 3,311 | 3,308 |
| Completa, 245 observaciones de test | 5 | 0,176 | 0,102 | 3,325 | 3,306 |
| Crisis, 22 observaciones de test | 1 | 0,138 | 0,247 | 2,849 | 2,878 |
| Crisis, 22 observaciones de test | 5 | 0,234 | 0,263 | 2,874 | 2,885 |

El [CSV de autores transcrito](../_static/literature_e1/journal_update_20261105/bucci2020_metrics.csv) conserva las filas y ubicaciones. La FNN no supera a HAR en MSE de la muestra completa; el patrón cambia en crisis. No se compara esta escala logarítmica mensual con la RMSE diaria de E1 ni se atribuye a una MLP superioridad general. La receta H16/ReLU/Softplus del proyecto sigue siendo propia: el artículo refuerza su función comparativa, no demuestra que nuestro ancho sea mínimo global ni que se haya elegido antes de entrenar por esa publicación.

#### 3.3.1.2. Tipo de arquitectura

MLP con **una capa oculta de 16 unidades ReLU**, dropout 0,1 y cabeza escalar Softplus. Aplana los mismos D×5 valores que reciben las otras redes, conservando el orden de días y canales. No incluye estado recurrente, convoluciones ni entradas adicionales.

#### 3.3.1.3. Descripción técnica

![Arquitectura propia MLP](../_static/e1_architectures_20261104/mlp.svg)

MLP: aplanado de D×5, capa oculta de 16 unidades y cabeza Softplus; el número de parámetros crece con D. El diagrama corresponde a la implementación del proyecto y conserva n=7/h=1.

Para X∈R^(D×5), x=vec(X)∈R^(5D), W₁∈R^(16×5D) y w₂∈R^16:

$$
h=\operatorname{ReLU}(W_1x+b_1),\qquad
\widehat y^{\mathrm{esc}}=\operatorname{Softplus}\!\left(w_2^\top\operatorname{Dropout}_{0,1}(h)+b_2\right).
$$

$$
\operatorname{Softplus}(a)=\log(1+e^a),\qquad
\widehat y=\bar y_{\mathrm{train}}\,\widehat y^{\mathrm{esc}}.
$$

Flujo: **(B,D,5) → aplanado (B,5D) → Dense16/ReLU → dropout → Dense1/Softplus → (B)**. Cada posición temporal tiene pesos propios; la red puede aprender relaciones no lineales de la ventana, pero no impone filtros compartidos o memoria entre pasos. Se entrena con MSE de Y escalada y **learning rate 0,001**, con los controles comunes descritos arriba. El código ejecutado es `SmallMLP` en [dl_models.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models.py).

#### 3.3.1.4. Tipo de problema que aborda

Regresión temporal supervisada del siguiente valor de σ7 a partir de OHLCV anterior al origen. El objetivo es continuo y no negativo. Usar una MLP no convierte la partición en una regresión tabular i.i.d.; se conserva el tipo 4 y la cronología de la ruta 6.4.

#### 3.3.1.5. Dataset y unidad de observación


Binance Spot BTCUSDT, agregado diario UTC desde velas de un minuto. Las entradas D=7/14/21/28 conservan cinco canales y el mismo objetivo sigma7/h1. Los doce bloques de validación de 2024 usan cohortes crecientes de 2023 y pasado permitido de 2024; las diez semillas son comunes. Tras congelar decisiones se ajustó una red final por historia y semilla, con desarrollo permitido y embargo D, y se evaluó la cohorte común del 01/01/2025 al 31/08/2026. No se comprimen huecos ni se imputa el 24/03/2023. Los [reglas, fechas y hashes públicos](../_static/filipe_multihistory_20261104/method.json) distinguen ambos periodos y las fuentes documentales no se presentan como el dataset propio.

#### 3.3.1.6. Métricas y valores propios de validación


Resultados propios en escala original para las cuatro historias, con n=7 y h=1. Cada valor anual reúne primero las 366 predicciones de una trayectoria; la media y la DE describen diez semillas del mismo calendario, no diez mercados independientes.

```{include} ../_static/filipe_multihistory_20261104/tables/cv_MLP.md
```

La evaluación final de estas mismas cuatro entradas se conserva en §5 y en [las tablas de test](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv). La configuración de referencia se eligió con desarrollo antes de abrir test; sus métricas posteriores no se usan para reseleccionar. Las cifras son propias, no resultados de los autores de las referencias fundacionales.

#### 3.3.1.7. Fortalezas

Tiene un flujo sencillo de verificar y **593/1.153/1.713/2.273 parámetros para D=7/14/21/28**. Es la menor del trío solo para D7; CNN1D tiene menos parámetros con las historias mayores. Representa interacciones no lineales entre covariables y posiciones sin mantener estado oculto. Su desempeño por historia está documentado en las tablas de CV; la capacidad feedforward no garantiza superioridad predictiva frente a recurrencia, convolución o persistencia.

#### 3.3.1.8. Limitaciones

El aplanado exige pesos distintos por posición y parámetros crecientes con D. No impone filtros compartidos ni memoria recurrente. Su resultado depende de representación, regularización y configuración; el desarrollo expuesto no es una validación anidada independiente. S8 acredita MLP aplicada, pero una fuente no demuestra uso amplio ni respalda la receta propia exacta; sus cuatro capas y RV intradiaria no coinciden con nuestra capa H16 y σ7 diaria. La comparación con persistencia conserva su interpretación por periodo e historia.

#### 3.3.1.9. Complejidad computacional

Con p=5 y H=16, contando los sesgos de ambas capas:

$$
P_{\mathrm{MLP}}=(pD)H+H+H+1=pDH+2H+1=80D+33.
$$

Para D=7/14/21/28, el total es **593/1.153/1.713/2.273 parámetros**, respectivamente. El ancho oculto se mantiene en 16; aumentar la historia incrementa los pesos de la primera capa.

El costo dominante por ejemplo es O(pDH+H); el aplanado no introduce parámetros. Las activaciones de la capa oculta crecen como O(H), además de la entrada O(Dp). No se confunde ese orden con tiempo de reloj.

**Medición del ciclo inicial D7.** La suma de tiempos de los dos ajustes por mes, sobre los doce meses, es **21,352 ± 1,234 s por trayectoria**; la suma de tiempos medios de inferencia de los doce bloques es **0,623 ± 0,041 ms**. Son medias ± DE entre diez semillas, medidas en CPU bajo tres trabajadores concurrentes. Excluyen preprocesamiento y escritura de artefactos según §4.2.5; no son latencias individuales de producción ni picos de memoria.

### 3.3.2. LSTM unidireccional

#### 3.3.2.1. Referencias completas y papel comparativo

**Fundamento.** Hochreiter, S. y Schmidhuber, J. (1997). *Long Short-Term Memory*. *Neural Computation*, 9(8), 1735–1780. DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) {cite:p}`hochreiter1997`.

**Antecedentes aplicados incluidos.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023). *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*. WSDM. DOI: [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387) {cite:p}`akbiyik2023`. Gao, M., Li, Q. y Tao, Z. (manuscrito de 2023). *Comparing the Predictive Power of Five Models on Bitcoin Volatility*; referencia editorial declarada a Springer, *Learning and Analytics in Intelligent Systems*, vol. 32, pp. 473–486, DOI declarado [10.1007/978-3-031-38074-7_41](https://doi.org/10.1007/978-3-031-38074-7_41), cuyo año editorial externo sigue sin corroborarse {cite:p}`gao2023`. Alakhouch Idriss (2026). *Bitcoin Volatility Forecasting: Econometric and Machine Learning Approaches Applied to Daily Price Data (July 2010–February 2024)*, informe fechado 02/06/2026, sin DOI ni revisión por pares acreditados {cite:p}`alakhouch2026`.

S1/S5/S7 aportan variantes unidireccionales verificadas, **3/6 textos**. La familia alcanza **5/6** al añadir BiLSTM de S2 y LSTM de S8, cuya direccionalidad no se precisa. S8 informa dos capas LSTM más Dense, longitud 12 y RMSE de test 0,0021960 con varianza 1,0293×10⁻⁹ {cite:p}`miura2019`. Estas fuentes difieren de σ7/OHLCV y no determinan H32 del proyecto. El [archivo de perfiles](https://github.com/jassan-bit/DeepLearning_202630/blob/main/references/literature_e1/source_profiles_archive_20261104.md) conserva las fichas anteriores; la implementación siguiente es propia.

#### 3.3.2.2. Tipo de arquitectura

Una **capa LSTM unidireccional de 32 unidades**, con estado inicial nulo, dropout 0,1 sobre el último estado oculto y cabeza Dense1/Softplus. No utiliza una segunda capa recurrente, bidireccionalidad, atención ni rezagos explícitos de σ7.

#### 3.3.2.3. Descripción técnica

![Arquitectura propia LSTM](../_static/e1_architectures_20261104/lstm.svg)

LSTM: una capa recurrente unidireccional de 32 unidades; la salida usa el último estado de la ventana observable. El diagrama corresponde a la implementación del proyecto y conserva n=7/h=1.

Para xⱼ∈R^5, hⱼ,cⱼ∈R^32 y s la función logística, la celda calcula:

$$
(i_j,f_j,o_j)=s(W_{i,f,o}x_j+U_{i,f,o}h_{j-1}+b_{i,f,o}),\qquad
 g_j=\tanh(W_gx_j+U_gh_{j-1}+b_g),
$$

$$
c_j=f_j\odot c_{j-1}+i_j\odot g_j,\qquad
h_j=o_j\odot\tanh(c_j),\qquad
\widehat y^{\mathrm{esc}}=\operatorname{Softplus}\!\left(w^\top\operatorname{Dropout}_{0,1}(h_D)+b\right).
$$

Cada b resume los **dos vectores de sesgo por compuerta de PyTorch**, uno de entrada y otro recurrente; ambos se cuentan por separado en los parámetros. Flujo: **(B,D,5) → LSTM32 → último estado (B,32) → dropout → Dense1/Softplus → (B)**. El estado recurrente representa dependencias ordenadas dentro de la ventana. Se optimiza la MSE del objetivo escalado con **learning rate 0,003**. Código ejecutado: `SingleLayerLSTM` en [dl_models.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models.py).

#### 3.3.2.4. Tipo de problema que aborda

Regresión temporal de una volatilidad escalar futura a partir de secuencias OHLCV ya observables. La memoria está dentro de cada ventana de entrada; no se conserva un estado entre las ventanas de ejemplos ni se actualizan pesos con la etiqueta del día pronosticado. Se mantiene tipo 4 y horizonte h=1.

#### 3.3.2.5. Dataset y unidad de observación


Binance Spot BTCUSDT, agregado diario UTC desde velas de un minuto. Las entradas D=7/14/21/28 conservan cinco canales y el mismo objetivo sigma7/h1. Los doce bloques de validación de 2024 usan cohortes crecientes de 2023 y pasado permitido de 2024; las diez semillas son comunes. Tras congelar decisiones se ajustó una red final por historia y semilla, con desarrollo permitido y embargo D, y se evaluó la cohorte común del 01/01/2025 al 31/08/2026. No se comprimen huecos ni se imputa el 24/03/2023. Los [reglas, fechas y hashes públicos](../_static/filipe_multihistory_20261104/method.json) distinguen ambos periodos y las fuentes documentales no se presentan como el dataset propio.

#### 3.3.2.6. Métricas y valores propios de validación


Resultados propios en escala original para las cuatro historias, con n=7 y h=1. Cada valor anual reúne primero las 366 predicciones de una trayectoria; la media y la DE describen diez semillas del mismo calendario, no diez mercados independientes.

```{include} ../_static/filipe_multihistory_20261104/tables/cv_LSTM.md
```

La evaluación final de estas mismas cuatro entradas se conserva en §5 y en [las tablas de test](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv). La configuración de referencia se eligió con desarrollo antes de abrir test; sus métricas posteriores no se usan para reseleccionar. Las cifras son propias, no resultados de los autores de las referencias fundacionales.

#### 3.3.2.7. Fortalezas

La recurrencia y las compuertas representan relaciones ordenadas sin asignar una matriz de entrada distinta a cada posición. Para H fijo, **el número de pesos no depende de D**, aunque el costo de procesar la ventana sí. Dispone de antecedentes aplicados recurrentes dentro del corpus, con heterogeneidad y calidad editorial explícitas.

#### 3.3.2.8. Limitaciones

Tiene más parámetros que las otras dos redes y ejecución recurrente secuencial. Memoria con compuertas no garantiza un aprendizaje útil de volatilidad; los resultados dependen del preprocesamiento y de la parada. Su posición en RMSE depende de la historia y del periodo; se conserva la comparación explícita de §5. Los antecedentes de S1/S5/S7 no comparten exactamente σ7/h1/OHLCV diario, y sus riesgos documentales no desaparecen por disponer de código. La evaluación propia sigue siendo exploratoria sobre desarrollo expuesto.

#### 3.3.2.9. Complejidad computacional

Con cuatro compuertas, dos vectores de sesgo por compuerta y cabeza escalar:

$$
P_{\mathrm{LSTM}}=4H(p+H+2)+(H+1)=4\cdot32(5+32+2)+33=5\,025.
$$

El costo dominante por ejemplo es O(4DH(p+H)); las activaciones de entrenamiento crecen como O(DH), además de la entrada. El cómputo de estados depende del orden temporal, aunque las operaciones dentro de cada paso se vectorizan.

**Medición del ciclo inicial D7.** Los doce procedimientos mensuales requieren **18,505 ± 2,305 s por trayectoria** y la suma de tiempos medios de inferencia de los doce bloques es **3,212 ± 0,362 ms**, medias ± DE entre diez semillas bajo la concurrencia registrada. Su menor tiempo de entrenamiento observado no acredita eficiencia intrínseca frente a MLP: el número de épocas seleccionado también cambia. El alcance de medición está en §4.2.5.

### 3.3.3. CNN1D temporal poco profunda

#### 3.3.3.1. Referencias completas y papel comparativo

**Fundamento de filtros compartidos.** LeCun, Y., Bottou, L., Bengio, Y. y Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. *Proceedings of the IEEE*, 86(11), 2278–2324. DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791) {cite:p}`lecun1998`.

**Contexto de secuencias.** Bai, S., Kolter, J. Z. y Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. Prepublicación arXiv:1803.01271 {cite:p}`bai2018`. Describe familias temporales más generales y no equivale a la CNN concreta de dos capas aquí ejecutada. LeCun estudia reconocimiento documental, no volatilidad de Bitcoin.

**Antecedente aplicado.** Miura, R., Pichl, L. y Kaizoji, T. (2019), *Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series*, ISNN 2019, LNCS 11554, pp. 165–172, DOI [10.1007/978-3-030-22796-8_18](https://doi.org/10.1007/978-3-030-22796-8_18) {cite:p}`miura2019`, incluye una **CNN independiente de dos capas más Dense** (Tabla 1, p. 171), [verificada en el texto completo](https://raw.githubusercontent.com/Anton21a/Thesis/8bc3aa3c96fd214b34b914d2057f4339f2c8ed1e/ML%20articles/Artificial%20Neural%20Networks.pdf). La dimensión 1D/2D, filtros, kernels y padding no se reportan: acredita la familia CNN, 1/6, sin acreditar una Conv1D específica. La TCN S1 tampoco se reetiqueta como la CNN ejecutada. La elección temporal 1D, dos capas H16/k3 y causalidad es una adaptación propia orientada por la guía 6.4, con receta y resultados propios explícitos.

**Dimensionalidad temporal verificada en una fuente primaria de autores.** HARNet, de Reisenhofer, Bayer y Hautsch {cite:p}`reisenhofer2022harnet`, aporta [código de autores fijado y extractos localizados](../_static/literature_e1/journal_update_20261105/harnet2022_primary_code.json). En su preset con rezagos [1,5,20], `Conv1D` opera sobre el eje temporal con padding causal, un filtro por capa, kernels 5 y 4 y dilataciones 1 y 5; usa ReLU, ausencia de sesgo en esas convoluciones, inicialización inspirada en HAR y Dense(1) lineal. El preset SPX identifica entrenamiento 2012–2015 y evaluación 2016. Estos hechos proceden del código, no de asumir la dimensionalidad a partir de «CNN».

HARNet constituye **respaldo metodológico de convolución 1D para volatilidad financiera**, con manuscrito original y publicación en revista pendientes de comprobar en esta revisión. El cargador utiliza la columna `rv5_ss` de Oxford-Man; su definición completa, frecuencia y horizonte económico no quedaron corroborados en esta extracción de código. No se equiparan al objetivo σ7/h1 de E1. Tampoco acredita una CNN1D independiente sobre Bitcoin ni valores de MAE/MSE/QLIKE: el repositorio contiene funciones, no una tabla de resultados verificada. Un minibatch de etiquetas de forma [4,5,1] no identifica por sí solo cinco horizontes. Las dos capas H16/k3, dilatación 1, promedio temporal y Softplus de E1 no se atribuyen a HARNet, cuya parametrización e información son distintas.

La revisión de calidad detecta además que `main.py`, líneas 72–80 de la versión fijada, aplica `scaler.fit_transform` a la serie cargada antes de seleccionar los intervalos train/test. Por ello, el código aporta evidencia de arquitectura, **sin acreditar un protocolo temporal libre de fuga o sus resultados fuera de muestra**. La separación causal del pipeline de E1 se evalúa con sus propios controles.

**Conexión editorial separada.** El CV primario de Fernando Moreno-Pino identifica con Stefan Zohren *DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions*, Quantitative Finance (2024), DOI [10.1080/14697688.2024.2387222](https://doi.org/10.1080/14697688.2024.2387222) {cite:p}`morenopino2024deepvol`. La [ficha de identidad editorial](../_static/literature_e1/journal_update_20261105/deepvol2024_publication.json) conserva el enlace extraído del CV y su ubicación. El artículo y su código no se recuperaron: no se le imputan kernels, resultados o datos desde HARNet. La presencia de convoluciones en su título no convierte ambos trabajos en la misma evidencia.

#### 3.3.3.2. Tipo de arquitectura

Dos convoluciones temporales 1D ordinarias, **16 canales por capa, kernel 3, stride 1 y dilatación 1**, con padding exclusivamente izquierdo, ReLU y dropout 0,1 después de cada capa. La cabeza aplica promedio temporal y Dense1/Softplus. No contiene bloques residuales ni atención.

#### 3.3.3.3. Descripción técnica

![Arquitectura propia CNN1D](../_static/e1_architectures_20261104/cnn1d.svg)

CNN1D: dos convoluciones causales ordinarias de 16 canales y media temporal; campo receptivo local de cinco días. El diagrama corresponde a la implementación del proyecto y conserva n=7/h=1.

Para a⁽⁰⁾=X, cada capa l∈{1,2} y posición j calcula, con ceros en posiciones anteriores al comienzo de la ventana:

$$
a^{(l)}_{j,c}=\operatorname{Dropout}_{0,1}\!\left[\operatorname{ReLU}\!\left(b_c^{(l)}+
\sum_{q}\sum_{r=0}^{2} K^{(l)}_{c,q,r}\,a^{(l-1)}_{j-2+r,q}\right)\right].
$$

$$
\bar a_c=\frac1D\sum_{j=1}^{D}a^{(2)}_{j,c},\qquad
\widehat y^{\mathrm{esc}}=\operatorname{Softplus}(w^\top\bar a+b),\qquad
R=1+2(3-1)=5.
$$

Flujo: **(B,D,5) → transposición (B,5,D) → Conv1D16/k3/ReLU/dropout → Conv1D16/k3/ReLU/dropout → media temporal (B,16) → Dense1/Softplus → (B)**. El campo receptivo **local** de cada posición antes del promedio es cinco; el promedio reúne todas las posiciones de la ventana observable. La normalización puede usar su último cierre porque ya se conoce al origen, sin utilizar datos posteriores al pronóstico.

Los filtros comparten pesos a lo largo del tiempo y capturan patrones locales repetidos; no representan el mecanismo recurrente de LSTM ni pesos por posición de MLP. Se entrena con MSE del objetivo escalado y **learning rate 0,001**. Código: `ShallowCausalCNN1D` en [dl_models.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models.py).

#### 3.3.3.4. Tipo de problema que aborda

Regresión temporal sobre secuencias numéricas OHLCV. La convolución opera sobre días, no sobre píxeles; el modelo no convierte el proyecto en clasificación de imágenes ni en generación secuencia a secuencia. Su única salida es σ7 del día de resultado, con horizonte h=1.

#### 3.3.3.5. Dataset y unidad de observación


Binance Spot BTCUSDT, agregado diario UTC desde velas de un minuto. Las entradas D=7/14/21/28 conservan cinco canales y el mismo objetivo sigma7/h1. Los doce bloques de validación de 2024 usan cohortes crecientes de 2023 y pasado permitido de 2024; las diez semillas son comunes. Tras congelar decisiones se ajustó una red final por historia y semilla, con desarrollo permitido y embargo D, y se evaluó la cohorte común del 01/01/2025 al 31/08/2026. No se comprimen huecos ni se imputa el 24/03/2023. Los [reglas, fechas y hashes públicos](../_static/filipe_multihistory_20261104/method.json) distinguen ambos periodos y las fuentes documentales no se presentan como el dataset propio.

#### 3.3.3.6. Métricas y valores propios de validación


Resultados propios en escala original para las cuatro historias, con n=7 y h=1. Cada valor anual reúne primero las 366 predicciones de una trayectoria; la media y la DE describen diez semillas del mismo calendario, no diez mercados independientes.

```{include} ../_static/filipe_multihistory_20261104/tables/cv_CNN1D.md
```

La evaluación final de estas mismas cuatro entradas se conserva en §5 y en [las tablas de test](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv). La configuración de referencia se eligió con desarrollo antes de abrir test; sus métricas posteriores no se usan para reseleccionar. Las cifras son propias, no resultados de los autores de las referencias fundacionales.

#### 3.3.3.7. Fortalezas

Comparte filtros entre posiciones con **1.057 parámetros**, independientes de D; tiene menos pesos que LSTM y que MLP con D≥14. Permite cómputo paralelo sobre la ventana, representa patrones locales y mantiene una arquitectura pequeña con conteo exacto. La comparación observada se reporta por D y periodo, sin atribuir superioridad a la familia únicamente por compartir filtros.

#### 3.3.3.8. Limitaciones

El campo local es corto y la media puede perder información de posición; padding y preprocesamiento afectan la representación. Causalidad convolucional no elimina por sí sola la fuga de datos. S8 respalda una CNN independiente, pero su dimensión y componentes no identificados impiden llamarla réplica de nuestra CNN1D; tampoco se replica la TCN de S1. El único apoyo CNN genérico no prueba recurrencia amplia ni que la receta actual sea mínima u óptima. Los resultados iniciales D7 no se extrapolan a otras historias o a una nueva evaluación.

#### 3.3.3.9. Complejidad computacional

Con p=5, H=16 y k=3:

$$
P_{\mathrm{CNN}}=(kpH+H)+(kH^2+H)+(H+1)
=3H^2+18H+1=1\,057.
$$

Las convoluciones requieren O(D(kpH+kH²)) por ejemplo y la media/cabeza O(DH+H). Las activaciones crecen como O(DH) para profundidad fija. No se infiere menor tiempo solo por paralelizar posiciones o reducir parámetros.

**Medición del ciclo inicial D7.** Los doce procedimientos mensuales requieren **44,350 ± 2,425 s por trayectoria** y la suma de tiempos medios de inferencia de los doce bloques es **3,691 ± 0,235 ms**, medias ± DE entre diez semillas en la CPU registrada. Estos valores describen la implementación y no los tiempos de LeCun/Bai/S1; la memoria pico no fue medida.

### 3.3.4. Cobertura y comparación de las tres fichas

| Campo | MLP | LSTM | CNN1D poco profunda |
|---|---|---|---|
| Ficha de implementación con nueve incisos | Completa | Completa | Completa |
| Entrada / salida propias | (D,5) → escalar, D=7/14/21/28 | Mismas cuatro entradas y salida | Mismas cuatro entradas y salida |
| Sesgo inductivo principal | Pesos por posición e interacciones feedforward | Memoria recurrente con compuertas | Filtros locales compartidos y promedio temporal |
| Configuración propia | Una capa H16 | Una capa unidireccional H32 | Dos capas H16/k3, sin dilataciones |
| Parámetros medidos por código | 593/1.153/1.713/2.273 para D7/14/21/28 | 5.025 | 1.057 |
| Apoyo aplicado independiente en el corpus | MLP 1/6, S8 | Unidireccional explícita 3/6; familia 5/6 | CNN genérica 1/6, S8; CNN1D explícita 0/6 |
| Datos/objetivo de fuentes idénticos al proyecto | No acreditado | No acreditado | No acreditado |
| Cumplimiento del respaldo bibliográfico del trío | Familia aplicada identificada; adaptación propia | Antecedentes identificados, con límites | Familia aplicada identificada; dimensión fuente no reportada |

Las **fichas propias están completas** y S8 añade el apoyo aplicado que faltaba para MLP y CNN genérica. Los seis textos BTC se distinguen de las 39 referencias citadas y de T1/T2/T3, que tienen diferentes niveles de lectura;  los **81 valores del corte anterior** y la nueva Tabla 1 se conservan por separado, sin sumar corridas como estudios. Permanecen la cobertura BTC limitada, la dimensionalidad no reportada de la CNN de S8 y la falta de especificación para replicar todas las recetas. El código T2 sí permite verificar Conv1D causal en otro mercado; T1 añade una referencia feedforward de revista leída íntegramente. La selección conjunta de tres básicos debe valorar frecuencia, simplicidad y diversidad; no se declara cumplida por el número de fichas o citas.

## 3.4. Comparación crítica de los modelos y de la evidencia

La comparación propia enfrenta **MLP, LSTM y CNN1D** bajo el contrato y CV comunes. Los antecedentes LSTM/TCN/encoder permanecen en el [archivo de fuentes](https://github.com/jassan-bit/DeepLearning_202630/blob/main/references/literature_e1/source_profiles_archive_20261104.md); sus configuraciones y resultados no se mezclan con las cifras de las tres redes implementadas. La validez de los estudios y la brecha avanzada/base de literatura se analizan por separado.

### 3.4.1. Comparación estructurada entre los tres modelos base

#### 3.4.1.1. Desempeño de CV y comparabilidad


La comparación principal reúne las cuatro historias D=7/14/21/28, n=7/h=1, cinco covariables y diez semillas. Cada puntuación anual reúne primero las 366 fechas de una trayectoria; no es la RMSE de un ensemble ni una media de R² mensuales.

```{include} ../_static/filipe_multihistory_20261104/tables/cv_models.md
```

```{include} ../_static/filipe_multihistory_20261104/tables/selected_configuration.md
```

El test final de §5 conserva esa selección previa. Persistencia resume ocho cierres para sigma7 previa: es más historia de cierre que D7 y menos días que las entradas D14/21/28; la información no se presenta como idéntica. Las fuentes aplicadas usan otros objetivos y escalas y no forman un ranking cuantitativo común con estas métricas propias. La CV de desarrollo expuesto no es selección anidada independiente.

#### 3.4.1.2. Robustez y evidencia disponible


Las tres redes usan las mismas cuatro historias y diez semillas, con 366 fechas de CV y una cohorte común de test. El control absoluto se ejecutó exclusivamente en el ciclo inicial D7. Los veinte meses de test permiten describir estabilidad temporal de los modelos finales, sin reajustarlos ni contarlos como réplicas de mercados.

El respaldo aplicado vigente es MLP 1/6, LSTM unidireccional explícita 3/6 y CNN genérica 1/6; CNN1D explícita permanece 0/6. No se ejecutaron pruebas de estrés con ruido controlado, otros activos o proveedores. Un orden mensual o R² positivo no demuestra robustez universal. La referencia BDS mensual es aproximada y sus promedios son descriptivos, no una prueba combinada.

#### 3.4.1.3. Escalabilidad y simplicidad


El costo dominante por ventana es O(DpH) para MLP, O(4DH(p+H)) para LSTM y O(D(kpH+kH²)) para CNN1D. MLP asigna pesos por posición; LSTM comparte las compuertas y CNN comparte filtros. El conteo de pesos por entrada es:

```{include} ../_static/filipe_multihistory_20261104/tables/parameters.md
```

Los costos medidos de CV se conservan por historia, modelo, semilla y fold; el [ajuste final](../_static/filipe_multihistory_20261104/final_fit_summary.csv) registra sus tiempos por corrida. El aumento de D modifica la entrada y el costo, pero no los pesos recurrentes o convolucionales con ancho fijo. Los tiempos dependen de épocas y concurrencia CPU; no son latencias garantizadas de producción. Los tiempos iniciales D7 siguen disponibles en sus registros y no se extrapolan a D14/21/28. No se midió memoria pico.

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

S8 utiliza entradas de longitudes distintas —MLP10, LSTM12, GRU5 y CNN6—, por lo que su comparación mezcla información y arquitectura. Declara diez folds temporales y 30 inicializaciones de test, pero no fechas, purga o un presupuesto comparable de búsqueda. Los pesos iniciales no crean nuevas trayectorias de mercado. Sus varianzas reportadas no completan por sí solas una comparación pareada, intervalos de confianza o multiplicidad. La copia publicada permite identificar el estudio, sin certificar su protocolo ni su transferibilidad a OHLCV diario.

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

**Contrapunto de S8.** Su Tabla 1 no presenta una propuesta avanzada que supere a todos los básicos: Ridge obtiene RMSE de test 0,0021603, frente a 0,0021778–0,0022067 de las redes, aunque GRU es la mejor por MSE de CV. El artículo es evidencia de uso comparativo de MLP/CNN/LSTM/GRU, no de superioridad universal DL ni de un margen avanzado transferible a E1. Estas cifras pertenecen exclusivamente a su tarea RV de tres horas.

### 3.4.4. Brecha verificable y proyección

**Brecha verificable del contrato propio:** S8 sí compara conjuntamente MLP, LSTM y CNN para RV de Bitcoin; por ello ya no corresponde afirmar ausencia de esas familias. Sin embargo, ninguno de los seis textos identifica la comparación conjunta exacta **MLP–LSTM–CNN1D causal con OHLCV diario de Binance Spot, n=7 y h=1**. La dimensión CNN de S8 es desconocida y su objetivo usa sumas de retornos minuto al cuadrado e historia de RV. Es una limitación de cobertura del corpus, no prueba de inexistencia mundial ni de novedad del proyecto.

La CV de Filipe establece referencias DL propias bajo reglas comunes y cuatro historias. El baseline neuronal cerrado antes de test sigue siendo **MLP D7**, según el [registro de selección](../_static/filipe_multihistory_20261104/freeze.json). La lectura posterior de S8 no reabre esa decisión, no valida retroactivamente la selección de recetas y no transforma las brechas de otros autores en un margen de mejora propio demostrado. Los capítulos 4–5 conservan resultados, cronología y límites inferenciales.

### 3.4.5. Correspondencia y estado de cierre

| Requisito de la guía | Evidencia de esta revisión | Estado y límite |
|---|---|---|
| Cadenas completas, fechas y resultados actuales | Doce búsquedas de la fase pública previa con respuestas/hashes, más recuperación complementaria registrada y capítulo S8 verificado | Los conteos de páginas/repositorios son descubrimiento, no artículos; las etapas no se suman como un PRISMA acumulado ni consultas nativas de bases. |
| Historial complementario conservado | Diez consultas G1–G10 del 02/11/2026, intentos, páginas, 118 ocurrencias y 106 repositorios distintos; comprobantes de acceso posteriores separados | Se preserva como corte histórico, sin sumarlo al flujo público actual ni tratar los cuatro fallos directos como cobertura científica |
| Procedencia WoS/Scopus y publicaciones IEEE/arXiv | Siete revistas con coincidencias secundarias de título e ISSN en copias públicas WoS/Scopus de 2026;  arXiv 2110.14317v2 corroborado en portada de autores; enlace Xplore 11058926 declarado por autores | Evidencia de revista e identidad documental, no verificación individual de artículos en los índices, consulta nativa, cobertura histórica WoS ni Q1; páginas arXiv/Xplore en vivo no consultadas |
| Inclusión, exclusión, flujo y responsable | El flujo histórico conserva siete textos evaluados/cinco incluidos; el [flujo complementario S8](../_static/literature_e1/recovery_20261105/retrieval_flow.svg) y su extracción actualizan a seis | Una inclusión nueva identificada, no inclusión automática de software; sin exhaustividad ni segundo revisor humano acreditados. |
| Matriz y evaluación de calidad | Seis textos, síntesis, ubicación de resultados y valoración de S8 añadida | S8 es un capítulo publicado; S2/S6/S7 conservan procedencia gris. La inclusión no acredita validez o reproducibilidad exacta. |
| Frecuencia y complejidad | Variantes, frecuencias ampliadas/sensibilidades, ganadores de S1 y conteos analíticos reproducibles | Corregido; no demuestra mínima complejidad global ni uso frecuente en todos los índices |
| Exactamente tres modelos básicos no redundantes | MLP–LSTM–CNN1D ejecutados, con apoyo aplicado de familias MLP/CNN añadido por S8; GRU3/6 alternativa | No son los tres primeros por frecuencia; dimensión CNN1D y adaptación mínima requieren sus propios límites. S8 no se presenta como motivo anterior al entrenamiento. |
| Fichas técnicas del trío implementado | Nueve campos propios y referencias aplicadas por familia; se incorpora S8 en MLP/LSTM/CNN | Parámetros propios no atribuidos a autores; CNN de S8 no permite probar dimensión o réplica de Conv1D. |
| Comparación crítica y brecha | Comparación propia anual MLP/LSTM/CNN1D; validez de estudios y brecha avanzada/base de S1 separadas | Ranking propio descriptivo; sin mezclar tareas, atribuir causalidad única o declarar novedad universal |
| Experimentos y baseline medido | CV principal de cuatro historias: 120 trayectorias, 1.440 segmentos modelo–fold y 2.880 ajustes, con D7 reutilizado; ajuste final y test separados del diagnóstico Ridge | Diez semillas por red/historia; test posterior a congelación, un solo activo y sin selección mediante sus métricas |

La fase actual sigue la búsqueda pública y la comprobación posterior de procedencia descritas en §3.1. Las siete coincidencias secundarias corresponden a *Journal of International Financial Markets, Institutions and Money*, *Frontiers in Physics*, *Economics Letters*, *Finance Research Letters*, *IEEE Access*, *Journal of Financial Econometrics* y *Quantitative Finance*. No sustituyen la verificación individual de los artículos ni certifican Q1. Las listas inspeccionadas no permiten juzgar actas de congresos: la ausencia de WSDM/ICDM en ellas no prueba falta de indexación.

Los comprobantes históricos permanecen en el [registro público](../_static/literature_e1/public_web_verification_20261103/discovery_records.json), el [flujo anterior](../_static/literature_e1/public_web_verification_20261103/flow_summary.json), la [evidencia de revistas](../_static/literature_e1/public_web_verification_20261103/secondary_journal_evidence.json) y la [procedencia](../_static/literature_e1/public_web_verification_20261103/publication_provenance.json). El [registro primario S8](../_static/literature_e1/recovery_20261105/primary_source.json), la [selección actualizada](../_static/literature_e1/recovery_20261105/selection_update.json) y la [síntesis vigente](../_static/literature_e1/recovery_20261105/synthesis_six_columns.csv) añaden un sexto texto. Los 81 valores anteriores no se reetiquetan como el total nuevo; la Tabla 1 de S8 se enlaza separadamente. Sigue sin acreditarse cobertura exhaustiva, indexación individual o réplica exacta de las configuraciones de autor.

## 3.5. Fundamentos bibliográficos y contexto de la revisión

### 3.5.1. Búsqueda documental efectuada y alcance de las citas

La consulta del **2 de noviembre de 2026** incorporó treinta referencias: siete sobre Bitcoin, siete de fundamentos de redes neuronales y dieciséis de volatilidad, diagnóstico y evaluación. Se consultaron repositorios públicos de autores, documentación de software académico y bibliografías de artículos y libros. Para cada entrada se conservan el enlace consultado, los campos corroborados y el nivel de acceso en el [registro de verificación](../_static/bibliografia/verification_30.csv) y el [catálogo detallado](../_static/bibliografia/references_30.json).

El catálogo vigente conserva las 36 referencias del corte que añadió S8 y agrega T1/T2/T3: **39 referencias** en §3.1.2.4 y su [CSV actualizado](../_static/literature_e1/journal_update_20261105/references_access_39.csv). Los cortes de 35 citas/cinco textos y las doce consultas de la fase pública anterior permanecen fechados. La recuperación complementaria actual aporta lectura primaria y no acredita consulta nativa de índices.

La red del entorno bloqueó el acceso directo a Crossref y arXiv. Por ello no se afirma haber consultado sus interfaces ni leído treinta originales completos. La entrada de Huang y colaboradores conserva su verificación previa; su acceso directo actual también está limitado. Las verificaciones nuevas de las otras referencias proceden de las fuentes accesibles indicadas en el registro. Una referencia identificada en otra bibliografía se distingue de un artículo original leído.

Los trabajos anteriores a 2019 se incluyen como fundamentos y antecedentes expresamente justificados. No se cuentan automáticamente como estudios recientes elegibles para calcular frecuencia de modelos base. Los registros de acceso a metadatos tampoco sustituyen el flujo complementario documentado en §3.1.

### 3.5.2. Antecedentes de volatilidad de Bitcoin

Katsiampa estudia la estimación de volatilidad de Bitcoin mediante especificaciones GARCH, mientras Dyhrberg relaciona su análisis de volatilidad con oro y dólar {cite:p}`katsiampa2017,dyhrberg2016`. Estos antecedentes justifican contrastar las redes con referencias financieras y delimitar la variable que se estima; no demuestran por sí solos qué arquitectura neuronal debe seleccionarse.

Los estudios asociados a órdenes de compra y venta, información de Twitter y flujos de información a frecuencia de minuto plantean conjuntos de información diferentes {cite:p}`guo2018,akbiyik2023,barjasic2021`. Sus títulos, datos documentados y referencias permiten reconocer líneas de investigación pertinentes, pero no autorizan atribuir a OHLCV diario mejoras obtenidas con otras señales. El artículo de Akbiyik y colaboradores se recuperó como copia de autores; para parte de los antecedentes se corroboraron metadatos en sus bibliografías y no se declara lectura del original.

Miura, Pichl y Kaizoji aportan ahora un **benchmark publicado de MLP, CNN, LSTM y GRU independientes para RV de Bitcoin** {cite:p}`miura2019`. La comparación de su Tabla 1 es directamente pertinente a familias neuronales para volatilidad; el cambio de proveedor, agregación, objetivo e información impide trasladar sus valores al experimento propio. Su CNN no tiene dimensión reportada, y sus cuatro capas MLP no definen la receta H16 del proyecto.

Huang y colaboradores aportan un antecedente de pronóstico de volatilidad con técnicas de aprendizaje automático. Herremans y Low se centran en picos de volatilidad e información de transacciones de grandes tenedores y CryptoQuant mediante modelos Synthesizer Transformer {cite:p}`huang2024,herremans2025`. Pronosticar picos, una medida realizada intradiaria o el siguiente valor de una desviación móvil diaria constituye tareas diferentes. La pertinencia temática no acredita igualdad de objetivo, datos, horizonte o protocolo.

### 3.5.3. Definiciones de volatilidad y referencias financieras

ARCH formaliza varianza condicional cambiante y GARCH amplía esa estructura incorporando dinámica de la varianza {cite:p}`engle1982,bollerslev1986`. Son fundamentos econométricos y referencias de contexto; no ocupan los puestos reservados a modelos de Deep Learning.

La literatura de volatilidad realizada y de modelos HAR estudia construcciones y dependencias a distintas escalas {cite:p}`andersen2003,corsi2009`. Es necesario conservar la distinción entre varianza condicional, volatilidad realizada construida con alta frecuencia y desviación estándar móvil de retornos diarios. La medida definida en este proyecto no se convierte en cualquiera de las anteriores por compartir la palabra «volatilidad». Cambiar ventana, frecuencia o anualización también cambia la escala de las métricas.

### 3.5.4. Fundamentos de las fichas implementadas y sus antecedentes

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

Diebold–Mariano estudia diferencias de precisión predictiva mediante pérdidas, y Harvey, Leybourne y Newbold proponen una modificación para muestras pequeñas {cite:p}`diebold1995,harvey1997`. La corrección no resuelve automáticamente selección adaptativa, no estacionariedad o cualquier dependencia. Estas condiciones deben conservarse al interpretar resultados publicados y la evaluación propia de §5.

El bootstrap por bloques para observaciones estacionarias y el stationary bootstrap proporcionan procedimientos de remuestreo para datos dependientes {cite:p}`kunsch1989,politis1994`. Son métodos diferentes: bloques de longitud fija y bloques de longitud aleatoria no deben presentarse como una implementación idéntica. Se necesita justificar longitud, unidad y supuestos; disponer de muchas fechas no basta para tratarlas como réplicas independientes.

Holm aporta un procedimiento secuencial para comparaciones múltiples {cite:p}`holm1979`. La familia de hipótesis debe declararse con las métricas, ventanas y contrastes que sustenten afirmaciones; añadir referencias no elimina ese requisito. Citar el procedimiento no sustituye los resultados propios ni las familias de contrastes documentadas en §5.

### 3.5.7. Trazabilidad de búsqueda y conclusión de la revisión

La herramienta descrita por Haddaway y colaboradores facilita producir diagramas de flujo PRISMA 2020 {cite:p}`haddaway2022`. Se cita como apoyo para documentar el proceso, sin atribuirle la declaración original ni afirmar cumplimiento de una revisión sistemática solo por dibujar un flujo. Los conteos de búsquedas, duplicados y exclusiones deben corresponder a registros reales.

Las **39 referencias**, los **seis textos completos del corpus BTC** y los tres respaldos metodológicos distinguen problema, arquitectura, lectura e identidad editorial. La lectura de S8 resuelve la ausencia de evidencia aplicada de MLP y de CNN genérica en el corte previo; no demuestra uso amplio de esas familias, no identifica Conv1D en el artículo ni convierte el benchmark propio en réplica. Las fichas y resultados implementados conservan su origen, mientras el [registro de selección actualizado](../_static/literature_e1/recovery_20261105/selection_update.json) explica el aporte posterior y la alternativa GRU. Permanecen la cobertura bibliográfica acotada, los parámetros de autor no reportados, la verificación individual en índices y la justificación conjunta de frecuencia, simplicidad y diversidad. T1 aporta un original de revista con FNN independiente; T2 acredita código Conv1D causal; T3 acredita una publicación de revista sin lectura del original. La cantidad de citas o corridas no sustituye esos niveles de evidencia ni elimina los límites de transferencia, recurrencia BTC e indexación individual.
