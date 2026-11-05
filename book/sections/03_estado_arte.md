# 3. Revisión del Estado del Arte y Extracción de los Tres Modelos Base

La revisión debe identificar exactamente tres modelos DL básicos usados como referencias comparativas en la literatura del problema. El alcance documental actual abarca **todas las criptomonedas**, mientras el experimento propio sigue en Bitcoin. La bibliografía vigente reúne **40 referencias** y el corpus aplicado contiene **siete textos completos**: seis estudios previos de BTC y S9, un estudio de veinte criptomonedas que también incluye BTC. Por tanto, el subconjunto que incluye BTC tiene siete estudios y uno de ellos incluye otras criptomonedas; son categorías superpuestas, no dos grupos que se suman. S8 verifica MLP, CNN, LSTM y GRU como comparadores independientes {cite:p}`miura2019`; S9 incorpora LSTM como propuesta, con su papel separado {cite:p}`djanga2025commonality`. T1/T2/T3 permanecen como transferencia financiera: un original editorial completo, código primario y una identidad de publicación, respectivamente; no aumentan el denominador cripto.

La terna principal vigente es **GRU–LSTM–TCN causal dilatada residual**. Akbiyik et al. (2023), Tabla 2, comparan las tres familias como referencias independientes para volatilidad de Bitcoin {cite:p}`akbiyik2023`; Miura, Pichl y Kaizoji (2019), Tabla 1, aportan otra comparación publicada de GRU y LSTM {cite:p}`miura2019`. En el corpus recuperado, las frecuencias estrictas son **GRU 3/7, LSTM 5/7 y TCN 1/7**. La TCN tiene respaldo directo, pero su recurrencia amplia no está acreditada. Las redes pequeñas del proyecto son adaptaciones declaradas, con 3.777, 5.025 y 4.289 parámetros, respectivamente; no son las configuraciones ganadoras de esos artículos.

La decisión de publicar esta terna se tomó **después del benchmark histórico y de revisar el experimento privado GRU–LSTM–TCN**. No se presenta como una selección bibliográfica anterior a esas ejecuciones ni como confirmación sobre un calendario nuevo. El protocolo privado había fijado las recetas antes de sus propios ajustes; esa congelación y la decisión posterior de publicación son hechos distintos. El [expediente de selección vigente](../_static/literature_e1/gru_lstm_tcn_selection_20261105/selection.json), su [tabla](../_static/literature_e1/gru_lstm_tcn_selection_20261105/selection.csv) y la {download}`justificación <../_static/literature_e1/gru_lstm_tcn_selection_20261105/README.md>` conservan evidencia, criterios y límites.

MLP y CNN permanecen en las extracciones originales cuando los autores las usaron; la terna experimental principal contiene únicamente GRU, LSTM y TCN. Bucci, HARNet y DeepVol siguen como transferencia financiera, con los niveles de acceso ya documentados {cite:p}`bucci2020realized,reisenhofer2022harnet,morenopino2024deepvol`; no añaden estudios al denominador cripto ni se necesitan para completar la evidencia TCN de S1.

La revisión del **03/11/2026** corrigió motivos de cribado, recuperó dos textos adicionales y completó la extracción de los hiperparámetros ganadores de S1, que sí aparecen en su apéndice. Se conservan los cortes anteriores y se explicitan fuentes grises, parámetros desconocidos y límites de las métricas. Aquella revisión documental del 03/11/2026 no entrenó redes ni consultó test; la ejecución posterior de desarrollo se registra por separado.

La lectura previa de S8 se realizó el **04/11/2026, hora de Bogotá**, con registros UTC del 05/11, después del benchmark histórico ya examinado. Su publicación de 2019 precede a esos experimentos, pero su recuperación no se reescribe como fundamento de decisiones experimentales previas. La terna vigente se registra como una decisión posterior y diferente; su respaldo se distingue de las fechas de publicación, recuperación y ejecución.

El problema propio es **regresión temporal, tipo 4, ruta 6.4**: OHLCV rezagadas sin historia explícita de la volatilidad objetivo. Las tareas de las fuentes no coinciden automáticamente con ese contrato. El expediente de selección vigente y las tres fichas implementadas se desarrollan en §§3.2–3.4; los registros de etapas anteriores permanecen en el historial versionado del repositorio.

La estrategia mantiene la solicitud del usuario: **búsqueda web pública y verificación posterior de procedencia e indexación**. Los registros de las fases previas se conservan. La [ampliación cripto del 05/11/2026](../_static/literature_e1/crypto_extension_20261105/search_log.json) documenta consultas, fuentes, accesos y reevaluación de S9; el manuscrito era una pista conocida y no se presenta como descubrimiento enteramente nuevo. Su original se leyó para incorporarlo bajo el alcance ampliado. La [ampliación financiera previa](../_static/literature_e1/journal_update_20261105/search_and_access_log.json) sigue distinguiendo T1/T2/T3. Ninguna etapa acredita por sí sola búsquedas nativas WoS/Scopus/IEEE Xplore/arXiv, registros individuales o Q1. Los catálogos de 35, 36 y 39 referencias son cortes anteriores; el vigente contiene 40.

La numeración mantiene los seis incisos de búsqueda (§3.1.1–3.1.6). La identificación de modelos (§3.2) separa procedimiento, criterios, síntesis, equivalencia, arquitecturas incompletas y familias orientativas. Cada ficha de §3.3 presenta los nueve campos de la guía; §3.4 distingue comparación, validez, brecha de desempeño y brecha de investigación.

## 3.1. Estrategia de Búsqueda (Obligatoria)

### 3.1.1. Cadena de búsqueda

#### 3.1.1.1. Pregunta y unidades de búsqueda

La pregunta inicial fue qué arquitecturas DL sencillas se utilizan como comparadores independientes para pronosticar volatilidad de Bitcoin. La ampliación documentada extiende esa pregunta a cualquier criptomoneda, conservando el experimento propio de BTC. Primero se localizan documentos, repositorios de autores y recursos públicos; después se comprueba identidad, publicación y posible indexación. **Descubrimiento web, documento científico y familia de arquitectura son unidades distintas.**

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

La etapa previa de recuperación de S8 está documentada en el {download}`informe complementario <../_static/literature_e1/recovery_20261105/README.md>` y la [procedencia primaria de S8](../_static/literature_e1/recovery_20261105/primary_source.json). Incluye las cadenas y fuentes efectivamente recuperadas, sus fechas y hashes. Sus resultados de descubrimiento no se suman a las consultas anteriores como si fueran artículos distintos; aquella etapa incorporó solo S8 al corpus científico.

**Ampliación cripto del 05/11/2026.** Se conserva esta cadena semilla conceptual, separada de las consultas nativas de cada plataforma:

```text
(bitcoin OR ethereum OR litecoin OR ripple OR solana OR dogecoin OR cryptocurrency) AND (volatility OR realized variance OR realized volatility) AND (deep learning OR MLP OR feedforward neural network OR CNN OR Conv1D OR LSTM OR GRU)
```

Las consultas públicas ejecutadas incluyeron, literalmente, `cryptocurrency volatility forecasting in:readme neural` y `ethereum volatility forecasting in:readme`. La [bitácora actual](../_static/literature_e1/crypto_extension_20261105/search_log.json) conserva todas las variantes, URL, página, fecha, respuesta y hash; la cadena semilla no se presenta como búsqueda enviada a WoS/Scopus. Se registraron **60 cadenas, 64 solicitudes de páginas, 59 respuestas exitosas, 501 ocurrencias y 379 repositorios únicos devueltos** en tres líneas de búsqueda. Son repositorios descubiertos, no artículos incluidos; no se suman los totales estimados por el índice ni los conteos de etapas anteriores.

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
#### 3.1.2.4. Catálogo bibliográfico vigente: 40 referencias

Las tablas conservan las 39 entradas del corte anterior y añaden una referencia cripto, S9. El [catálogo vigente de 40 referencias](../_static/literature_e1/crypto_extension_20261105/references_access_40.json) conserva procedencia y nivel de acceso. Los catálogos de [39](../_static/literature_e1/journal_update_20261105/references_access_39.json) y [36](../_static/literature_e1/recovery_20261105/references_access_36.json) permanecen como antecedentes; no se modifican sus fechas ni se atribuyen lecturas nuevas a entradas anteriores.

El catálogo reúne **40 referencias: trece aplicaciones a Bitcoin u otras criptomonedas, ocho fundamentos de redes neuronales, dieciséis referencias de volatilidad, diagnóstico y evaluación, y tres respaldos metodológicos financieros**. Siete son estudios cripto completos incluidos; la cantidad de citas no equivale a ese denominador. Las pistas sin original y la evidencia solo de código o identidad se conservan por separado.

El **enlace bibliográfico** identifica el trabajo, su DOI, su preprint o el registro donde se documentó la referencia. La columna **Fuente y evidencia consultada** indica qué material se utilizó realmente: texto completo de autores, código, documentación o una referencia localizada en otra bibliografía. Un DOI registrado no acredita su resolución directa ni la lectura del artículo original. Los manuscritos de Zeri y Gao conservan las precisiones sobre revisión por pares y año editorial en los perfiles fuente archivados y la matriz de estudios.

##### 3.1.2.4.1. Aplicaciones a Bitcoin y otras criptomonedas (13 referencias)

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
| {cite:p}`djanga2025commonality` | *Cryptocurrency Volatility Forecasting with Applications in Trading* · manuscrito fechado 26/01/2025, sin DOI/revista verificados | [Original y extracción S9](../_static/literature_e1/crypto_extension_20261105/S9_extraction.json): veinte pares cripto, incluido BTC; LSTM propuesta y cabeza MLP interna, sin comparador MLP/CNN independiente verificado. |

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

El catálogo vigente se descarga en [CSV](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv), [JSON](../_static/literature_e1/crypto_extension_20261105/references_access_40.json) y [BibTeX](../_static/literature_e1/crypto_extension_20261105/references_access_40.bib). Los cortes anteriores de [39](../_static/literature_e1/journal_update_20261105/references_access_39.csv), [36](../_static/literature_e1/recovery_20261105/references_access_36.csv) y [35](../_static/literature_e1/references_access_35.csv) se conservan. Las citas completas están en la [bibliografía del libro](referencias.md).

**Recuperación dirigida posterior a la auditoría del 04/11/2026.** Se ejecutó otra etapa de descubrimiento público, separada del flujo original: **13 cadenas literales, 18 páginas y 113 ocurrencias en 92 repositorios únicos**; se inspeccionaron nueve raíces/README y siete archivos de código/notebook sin ejecutarlos. El [registro de consultas](../_static/literature_e1/recovery_audit_20261104/search_log.json) y la [evaluación de candidatos](../_static/literature_e1/recovery_audit_20261104/candidate_assessment.csv) documentan este recorrido. No se suman estas ocurrencias al PRISMA original sin una deduplicación conjunta: son dos etapas con denominadores propios.

En aquella recuperación se verificaron pistas de una CNN1D independiente y una MLP TS-only, pero **no se obtuvo un manuscrito científico completo nuevo**: su resultado histórico fue 35 referencias, cinco textos y 0/5 para MLP/CNN1D simple. La [matriz de siete fuentes](../_static/literature_e1/recovery_audit_20261104/dataset_comparability.csv) y el {download}`informe de esa etapa <../_static/literature_e1/recovery_audit_20261104/report.md>` conservan ese alcance. La inclusión posterior de S8 es otra recuperación primaria, no una reinterpretación de esos README ni prueba de que sus fuentes tengan el mismo dataset/σ7 del proyecto.

### 3.1.3. Periodo de búsqueda

El intervalo preferente se actualiza a **01/01/2019–05/11/2026**, conservando los cierres previos del 03/11 y 04/11. S8 tiene fecha editorial impresa de 2019 y S9 fecha de manuscrito de 2025; sus fechas de recuperación no cambian esas identidades. Las consultas web no aplicaron un filtro verificable de año editorial: se comprueba durante la identificación. Un commit no sustituye el año del documento. Guo (2018), Katsiampa (2017) y Dyhrberg (2016) conservan su papel de antecedentes/contexto.

El snapshot de un listado de revistas de 2026 es distinto del año de un artículo y de su cuartil. La fecha impresa de una versión arXiv tampoco sustituye al año de la publicación posterior; por ejemplo, Akbiyik tiene una versión pública de 2022 y referencia ACM 2023. Los manuscritos sin fecha editorial acreditada siguen identificados como tales y requieren sensibilidad.

### 3.1.4. Criterios de inclusión y exclusión

Los criterios siguientes se hicieron explícitos al revisar el corpus; no se presentan como prerregistrados antes de las búsquedas del 02/11/2026.

| Dimensión | Evidencia necesaria para extraer un estudio | Tratamiento cuando falta o no corresponde |
|---|---|---|
| Pertinencia | Pronóstico continuo de volatilidad o varianza de cualquier criptomoneda; identificar activos, mercado, frecuencia y horizonte | Precio, dirección o trading sin pronóstico de volatilidad quedan fuera del corpus aplicado. Otros mercados financieros se separan como transferencia metodológica; una descripción ambigua permanece pendiente. |
| Uso empírico y papel del modelo DL | Texto completo con DL usado para el pronóstico y papel identificable: referencia, propuesta o componente interno | La inclusión amplia y la frecuencia de referencias son decisiones distintas. Solo un comparador DL independiente verificado suma a la frecuencia estricta; una cabeza o clase no invocada no acredita una referencia independiente. |
| Resultados verificables | Métrica, valor, escala y tabla/sección de origen; identificar a qué modelo pertenece cada cifra | Si no se recupera una cifra DL identificable no se la inventa ni se reasigna desde un ganador clásico. La presencia DL puede documentarse con su papel, sin incrementar automáticamente la frecuencia estricta de referencias. |
| Arquitectura y protocolo | Tipo de arquitectura, papel, objetivo, partición y evaluación identificables con límites explícitos | Separar falta de evidencia del comparador, hiperparámetros no reportados y configuración ganadora desconocida. No completar una receta por preferencia; conservar lo desconocido y la búsqueda requerida por E2. |
| Identidad y publicación | Autoría, título y documento localizable; fecha declarada o desconocida de forma explícita; DOI si existe | Código o README sin texto suficiente es contexto. Un informe o borrador no se excluye solo por su género; la literatura gris se identifica y no acredita revisión por pares |
| Periodo | Preferencia 01/01/2019–05/11/2026, conservando los cortes previos | Los anteriores se justifican como antecedentes; los sin fecha no acreditan recencia. El año del manuscrito de S9 no acredita publicación editorial o revisión por pares. |

Se distinguen pertinencia, recuperabilidad y calidad. Que un texto esté incluido para extracción exploratoria no certifica su validez ni basta para sostener que respalda uno de los tres modelos básicos definitivos. La preferencia Q1 requiere verificar índice, revista, año y categoría; no se deduce de que exista un DOI o un repositorio, ni se traslada automáticamente a congresos.

Un estudio cuenta como máximo una vez por familia y papel verificados. Preprint, artículo, código y variantes del mismo trabajo se deduplican por identidad. Las dos MLP de S8 cuentan una vez; su CNN genérica no se convierte en Conv1D. S9 cuenta como **un estudio de veinte activos**, con LSTM propuesta y cabeza MLP interna: no aporta una referencia MLP/CNN/LSTM simple adicional al conteo estricto. Las 40 citas, los siete textos cripto y los tres respaldos financieros son unidades distintas. T1/T2/T3 no entran en el denominador cripto.

Para el nuevo flujo académico se normalizarán DOI; se agruparán identificadores arXiv sin versión y vínculos preprint–artículo comprobados; cuando falten identificadores se revisarán título, autores y año. Las coincidencias ambiguas se resolverán conservando el motivo. Esta deduplicación documental no se presenta como ejecutada mientras no existan exportaciones. La literatura gris se identifica y requiere sensibilidad; no se transforma en evidencia revisada por pares.

Para la verificación posterior se registra evidencia primaria de autores/documento, evidencia secundaria de recursos públicos y estado pendiente por separado. Una correspondencia de revista/ISSN no basta para incluir el artículo en un corpus DL: todavía deben comprobarse objetivo, comparador, arquitectura, protocolo y resultados. Los repositorios nuevos de código o tesis se mantienen como pistas de recuperación hasta resolver identidad y posibles duplicados; no se excluyen por su género ni se incluyen por su descripción.

### 3.1.5. Trazabilidad del proceso

#### 3.1.5.1. Fase pública del 03/11/2026, corte histórico

Las doce consultas devolvieron respuestas de búsqueda verificables. Sus unidades y funciones se conservan separadas:

| Función | Consultas | Ocurrencias recuperadas | Repositorios distintos dentro de esa función |
|---|---:|---:|---:|
| Descubrimiento de documentos | 4 | 40 | 36 |
| Recuperación dirigida de títulos/identificadores | 4 | 7 | 7 |
| Localización de recursos de indexación | 4 | 20 | 20 |

Las 67 ocurrencias de las tres funciones contienen repeticiones y no son 67 artículos. Se recuperó una sola página por cadena; la relevancia del motor y los límites de recuperación sesgan el descubrimiento. El [registro de resultados](../_static/literature_e1/public_web_verification_20261103/discovery_records.json) y el [resumen de la fase](../_static/literature_e1/public_web_verification_20261103/flow_summary.json) permiten reconstruir los conteos sin mezclarlos con el historial anterior.

En esa fase se inspeccionaron además cinco README de candidatos, con [URLs fijadas, hashes y líneas pertinentes](../_static/literature_e1/public_web_verification_20261103/candidate_sources.json). Se mantuvo entonces el corte de cinco textos y 81 cifras de autores. **La incorporación posterior de S8 requirió lectura completa y extracción verificable**, documentadas aparte; no se cuentan como estudios los repositorios adicionales sin ese respaldo.

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

Se preservan el [cribado original](../_static/literature_e1/screening_20261102_original.csv), el [cribado de aquel corte](../_static/literature_e1/screening.csv), la [corrección de motivos](../_static/literature_e1/screening_revision_20261103.json) y la [etapa de recuperación](../_static/literature_e1/screening_recovery_stage_20261103.json). Los [manifiestos de recuperación](../_static/literature_e1/directed_recovery_20261103/source_manifest.json) identifican URL fijada, commit, fecha y hash. El corte anterior de matrices y flujo permanece en `history_20261102/`. No se suman diapositivas, código y borrador del mismo trabajo como estudios distintos.

**Inclusión exploratoria posterior a la búsqueda.** S6 —Wang y colaboradores— es un borrador de cinco autores sin fecha de publicación acreditada; su commit de 2026 solo fecha la versión disponible. S7 —Alakhouch Idriss— es un informe académico fechado el 02/06/2026. Se retienen bajo los mismos criterios básicos que permitieron incluir literatura gris en S2 y evidencia con limitaciones en S5: objetivo, identidad, comparador, protocolo y resultados extraíbles. No se excluyen por ser borrador o informe ni se presentan como evidencia indexada, Q1 o revisada por pares. Sus riesgos severos exigen mostrar sensibilidad **sin S6/S7**, con el corpus previo S1/S2/S5, y **sin documentos de fecha no acreditada**. Incorporarlos amplía el corpus, pero no cierra la selección definitiva de los tres modelos básicos.

El cribado fue asistido por Codex para el autor responsable, Jassan Arteta. No se acredita segundo revisor humano, consenso previo ni acuerdo entre revisores; las decisiones y dudas quedan registradas con su evidencia y fecha. El flujo comunica una revisión complementaria, no exhaustividad PRISMA ni consultas en las bases que permanecen sin búsqueda acreditada.

La frase histórica «sin ejecutar cadenas nuevas» describe aquella recuperación de seis repositorios. La búsqueda pública actual es otra fase: no se suman sus repositorios a este flujo ni se añaden estudios sin deduplicación documental y extracción verificables.

La incorporación histórica S8 se muestra en un flujo complementario, separado del anterior:

![Recuperación complementaria de S8 y actualización de cinco a seis textos incluidos](../_static/literature_e1/recovery_20261105/retrieval_flow.svg)

**Recuperación editorial y metodológica posterior.** El [registro nuevo](../_static/literature_e1/journal_update_20261105/search_and_access_log.json) conserva las búsquedas focalizadas, fuentes, fechas y hashes. Su resultado verificable es **un artículo completo nuevo sobre volatilidad financiera (T1), un código primario de autores (T2) y una identidad editorial acreditada en CV (T3)**. No se suman como tres artículos leídos ni como aplicaciones adicionales a Bitcoin. Los originales Elsevier que no pudieron recuperarse siguen pendientes; el fallo de acceso no significa que no existan ni que carezcan de indexación. No se suman repositorios, citas y textos completos como si fueran registros independientes de PRISMA.

La [procedencia primaria](../_static/literature_e1/recovery_20261105/primary_source.json) permite identificar la pista, la copia completa, sus hashes y la decisión. El paso de cinco a seis es una actualización del corpus incluido; **no convierte los 106 repositorios anteriores ni los nuevos candidatos de software en artículos científicos ni construye un PRISMA acumulado sin deduplicación conjunta**.

**Ampliación a todas las criptomonedas, posterior al benchmark.** La [bitácora nueva](../_static/literature_e1/crypto_extension_20261105/search_log.json), el [cribado](../_static/literature_e1/crypto_extension_20261105/screening.csv) y el [resumen](../_static/literature_e1/crypto_extension_20261105/summary.json) documentan la incorporación de S9. La matriz normaliza diez registros: siete estudios cripto más T1/T2/T3 en sus estratos separados. Los seis textos previos mantienen sus identidades; S9 era una pista conocida que se reevaluó bajo el alcance ampliado. No se multiplica el número de estudios por sus veinte activos ni se suman consultas solapadas como un PRISMA acumulado.

![Ampliación documental a siete estudios cripto con estratos separados](../_static/literature_e1/crypto_extension_20261105/flow.svg)

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
| S9. Djanga, Cucuringu y Zhang (2025) | Manuscrito completo sobre veinte criptoactivos; LSTM propuesta en UAM y cabeza MLP opcional, sin configuración ganadora identificada | Reajuste diario, validación interna 80/20 y DM reportado; semillas/DE/IC neuronales no acreditados | RV de 30 minutos en rejilla de cinco minutos y trayectoria del día siguiente; fronteras de winsorización/clustering no plenamente descritas; selección por resultados OOS y etiqueta repetida CAM,1w en la columna 1m. Sin DOI/revista/indexación verificados. |

Se conserva la [valoración por dimensión](../_static/literature_e1/quality_assessment.csv) y su [JSON](../_static/literature_e1/quality_assessment.json). Ausencia de descripción no equivale a ausencia del defecto; una prueba publicada tampoco garantiza calibración, comparabilidad o control de multiplicidad. Los detalles y la repercusión sobre la selección se desarrollan en §§3.2–3.4.

La [evaluación complementaria de S8](../_static/literature_e1/recovery_20261105/quality_assessment.json) añade estos desconocidos sin sobrescribir la valoración histórica de S1–S7. Se distingue existencia de un comparador independiente, que sí se verifica, de reproducibilidad exacta, que sigue incompleta.

La consulta de una base devuelve registros; la inclusión exige recuperar y valorar documentos. Para la nueva búsqueda se mantendrá la evaluación de arquitectura/comparador, protocolo, partición y fuga aparente, repeticiones, dispersión e inferencia. Indexación y cuartil se comprobarán por fuente, revista, año y categoría: no se deducen de un DOI ni se atribuyen automáticamente a congresos o preprints.

La verificación de procedencia añadida en esta fase no avala la calidad experimental. Un identificador correcto o una revista presente en una lista no sustituyen revisión de fugas, particiones, repeticiones, dispersión y comparabilidad. El cuartil Q1 permanece pendiente de evidencia específica por índice, año y categoría; no se deduce de las coincidencias secundarias de 2026.

## 3.2. Identificación de los tres modelos base de Deep Learning

El corpus vigente contiene **siete textos completos cripto: S1, S2, S5, S6, S7, S8 y S9**. Todos incluyen BTC; S9 incluye también diecinueve pares adicionales. S1 y S8 tienen identidad de publicación en sus documentos; S2/S5/S6/S7/S9 mantienen su género gris o publicación editorial pendiente, sin atribuir indexación. La inclusión permite extraer lo reportado y su papel; no certifica validez, equivalencia de contratos o que cada estudio añada una referencia básica independiente.

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

| Familia como referencia independiente | Estudios verificados | Corpus cripto formal, N=7 | Estrato con referencias DL extraíbles, N=6 | Distinción necesaria |
|---|---|---:|---:|---|
| LSTM, incluyendo BiLSTM | S1, S2, S5, S7, S8 | 5/7 | 5/6 | S1/S5/S7 son unidireccionales; S2 es BiLSTM y S8 no precisa dirección. S9 usa LSTM como propuesta. |
| GRU | S1, S2, S8 | 3/7 | 3/6 | Alternativa recurrente; no idéntica a LSTM |
| Encoder Transformer | S2, S6 | 2/7 | 2/6 | Dos fuentes grises; no equivalen a una configuración mínima |
| TCN | S1 | 1/7 | 1/6 | Separada de CNN genérica, CNN1D simple y D-TCN propuesta |
| MLP independiente no lineal | S8 | 1/7 | 1/6 | Dos variantes de S8 cuentan una vez; Dense1 de S5 y la cabeza opcional de S9 no cuentan |
| CNN independiente, dimensión no reportada | S8 | 1/7 | 1/6 | Dos capas más Dense; no se infiere Conv1D |
| CNN1D simple explícitamente identificada | Ninguno verificado | 0/7 | 0/6 | Falta de verificación positiva en este corte, sin afirmar inexistencia universal |

Las [frecuencias actuales](../_static/literature_e1/crypto_extension_20261105/family_frequencies.csv) distinguen **referencias independientes** y **presencia empírica**. Al añadir la LSTM propuesta de S9, la presencia de la familia LSTM es **6/7**, pero su frecuencia estricta como referencia permanece **5/7**. El estrato de seis estudios con referencias DL extraíbles conserva los numeradores previos y 1/6 para MLP/CNN; no se presenta como todo el corpus. Los siete estudios incluyen BTC; solo S9 incorpora otras criptomonedas, con solapamiento entre categorías.

El [corte previo de seis](../_static/literature_e1/recovery_20261105/family_frequencies.csv), su [decisión fechada](../_static/literature_e1/recovery_20261105/selection_update.json) y las [frecuencias de cinco textos](../_static/literature_e1/family_frequencies.csv) conservan sus denominadores originales. S9 se incluye por pertinencia y lectura completa aunque no aumente MLP/CNN: no se condiciona su inclusión a favorecer el trío implementado. Las diferencias de calidad y publicación se mantienen en la matriz y sus estratos; no se usan para borrar silenciosamente un estudio incluido.

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
| CNN S8 | Dos capas más Dense; dimensión, canales y kernel no reportados | No calculable de forma exacta | La dimensión desconocida se conserva como dato de esa fuente; no determina la TCN propia |

Los [escenarios analíticos](../_static/literature_e1/model_complexity_review_20261103.csv) documentan fórmulas, sesgos, proyecciones y supuestos. El script `scripts/audit_literature_models.py` reproduce las frecuencias y conteos del corte histórico de cinco textos leyendo aquella extracción; no ejecuta frameworks, entrenamiento ni datos de mercado. El corte previo de seis textos conserva su propia tabla y manifiesto. S6 no permite un total exacto porque faltan ancho feedforward y detalles de la cabeza/posiciones. En S7 se conservan por separado las hipótesis de 12 atributos del PDF y 8 del código.

Los conteos comunes controlan dimensiones de entrada/salida, pero no igualan capacidad, datos o cómputo. GRU tiene menos pesos que la LSTM H261 de S1, mientras la LSTM H20 de S5 es menor que esa GRU. La TCN ganadora no es una CNN1D pequeña: su campo receptivo condicional es 10.921 frente a 96 pasos observados. Reducirla a otra red sería una adaptación que debe justificarse, no una réplica del ganador.

#### 3.2.1.3. Estado de la selección

(seleccion-diversidad-d7)=
(seleccion-gru-lstm-tcn)=
**Terna principal vigente: GRU, LSTM y TCN.** La selección combina referencia aplicada directa, capacidad controlada, reproducibilidad y diferencias de mecanismo. No corresponde a los tres primeros puestos de un ordenamiento por frecuencia: el encoder tiene 2/7 frente a TCN 1/7. Se elige una TCN compacta por su comparación independiente en el original publicado S1 y por contrastar recurrencia con convolución causal dilatada, sin declarar que un encoder pequeño sea imposible o intrínsecamente inferior.

| Modelo propio | Respaldo directo y frecuencia estricta | Mecanismo y configuración registrada | Parámetros / learning rate |
|---|---|---|---|
| GRU | S1, S2 y S8: **3/7**; S1 y S8 tienen publicación identificada en el original | Una capa unidireccional H=32; dos compuertas, estado oculto integrado y lectura del último estado | **3.777 / 0,003** |
| LSTM | S1, S2, S5, S7 y S8: **5/7**; la variante unidireccional está explícita en S1/S5/S7 | Una capa unidireccional H=32; tres compuertas y estado de celda separado; lectura del último estado | **5.025 / 0,003** |
| TCN | S1: **1/7**, Tabla 2; no se acredita uso ampliamente recurrente | Tres bloques residuales, dos Conv1D causales por bloque, H=16, k=3, dilataciones 1/2/4 y lectura de la última posición | **4.289 / 0,001** |

Las tres reciben **(B,D,5), D∈{7,14,21,28}**, producen una salida escalar Softplus y comparten objetivo σ7/h=1, información, particiones y controles de entrenamiento. Los conteos no dependen de D con estos anchos fijos. La TCN tiene campo receptivo de **29 posiciones**: cubre las cuatro ventanas, incluida la de 28 días; las posiciones anteriores al comienzo de la ventana son relleno izquierdo, no observaciones adicionales. La especificación TCN incluye causalidad, dilatación, conexiones residuales y lectura de la última posición temporal; esos elementos definen la red que se evalúa.

GRU y LSTM comparten recurrencia, pero sus estados y actualizaciones son distintos. GRU usa compuertas de actualización y reinicio; LSTM usa entrada, olvido y salida, además de una memoria de celda separada. Compararlas con H=32 controla ancho, no iguala capacidad: sus parámetros difieren. La TCN aporta filtros temporales compartidos y rutas residuales sin estado recurrente. Estas diferencias justifican un contraste, sin garantizar que produzcan errores distintos o mejoras predictivas.

La configuración ganadora de TCN en S1 tiene seis niveles, 287 filtros y k=5; **no es nuestra TCN compacta**. S8 tampoco define las recetas H32 del proyecto. La arquitectura propia se documenta en las [especificaciones públicas](../_static/gru_lstm_tcn_20261105/model_specs.json). El [resumen de ejecución](../_static/gru_lstm_tcn_20261105/summary.json) y la [selección por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) pertenecen al experimento propio, no a los autores citados.

La decisión vigente sucede al benchmark histórico y a la inspección del experimento privado. El protocolo de ese experimento fijó las recetas antes de sus nuevos ajustes, con diez semillas y las cuatro historias, sobre el calendario de desarrollo 2024 ya observado. No hubo una nueva búsqueda de anchos o learning rates: GRU heredó H/LR recurrentes y TCN H/LR convolucionales. Se informa el rendimiento de estas recetas, no la superioridad de cada familia óptimamente ajustada ni una selección prospectiva independiente.

Los [registros originales de procedencia](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/gru_lstm_tcn_20261105/original_registration) se conservan en el archivo fuente del repositorio, fuera de los recursos experimentales presentados en este capítulo. La selección actual es posterior y no se atribuye retroactivamente a esos registros. El [expediente vigente](../_static/literature_e1/gru_lstm_tcn_selection_20261105/selection.json) documenta GRU–LSTM–TCN sin cambiar la matriz de siete estudios ni sus frecuencias.

### 3.2.2. Criterios que debe cumplir cada modelo base

Los cuatro criterios de la guía se evalúan con evidencia y límites concretos:

| Criterio | GRU y LSTM | TCN y límite de la selección |
|---|---|---|
| Arquitectura simple y reproducible | Una capa H=32, cabeza escalar; 3.777 y 5.025 parámetros, con fórmulas en §3.3 | Tres bloques residuales y 4.289 parámetros; pequeña frente al ganador de S1, sin afirmar mínima complejidad global. La receta exacta es del proyecto. |
| Uso como referencia en literatura del problema | GRU 3/7 y LSTM 5/7; ambas en los originales publicados S1 y S8 | TCN 1/7, referencia independiente en S1, Tabla 2. **Uso ampliamente recurrente no acreditado**; la selección autorizada acepta esa cobertura limitada y la hace explícita. |
| Mismo pipeline, información y evaluación | Cuatro historias OHLCV, σ7/h=1, mismos cortes mensuales 2024, diez semillas y transformaciones ajustadas a entrenamiento | Mismo contrato y presupuesto fijo por configuración. Igualar número de corridas no iguala capacidad ni demuestra optimización equivalente. |
| Modelo DL independiente | GRU y LSTM son redes completas con salida propia | TCN de una rama con salida propia; no D-TCN social, cabeza interna o red híbrida. |

El criterio de recurrencia no queda satisfecho por igual en los tres puestos. La publicación de una terna coherente no convierte TCN 1/7 en evidencia abundante ni todas las siete fuentes en artículos revisados por pares. La suficiencia operativa se justifica mediante el antecedente comparativo directo de S1, recetas explícitas y evaluación común; el alcance bibliográfico limitado permanece visible. Los riesgos de cada fuente siguen en §3.1.6 y §3.4.2.

La terna no es redundante por definición ni se demuestra no redundante solo contando compuertas. Se contrastan dos mecanismos recurrentes diferentes y una representación convolucional temporal; el análisis de sus errores y diferencias se realiza con resultados propios bajo el mismo contrato. Persistencia, Ridge, SVR o GARCH son referencias clásicas de contexto, sin ocupar los tres puestos DL.

### 3.2.3. Producto obligatorio: síntesis de los estudios incluidos

| Referencia | Problema | Dataset | Modelos DL | Métricas | Valores |
|---|---|---|---|---|---|
| S1. Akbiyik et al. (2023), Tabla 2 {cite:p}`akbiyik2023` | RV del día siguiente agregada de 96 retornos intradiarios | Bitcoin a 15 minutos y Twitter | LSTM, GRU y TCN; D-TCN es la propuesta avanzada | RMSE; media ± DE de veinte corridas | LSTM 2,37±0,19; GRU 4,23±5,61; TCN 2,58±0,07 |
| S2. Zeri (2026), Run B {cite:p}`zeri2026` | Log-RV futura de 30 minutos | BTCUSDT perpetuo, Crypto Lake | GRU, BiLSTM y encoder Transformer SELF-ATT | R² frente a media train y QLIKE | SELF-ATT 0,600/0,562; BiLSTM 0,586/0,582; GRU 0,582/0,564, respectivamente |
| S5. Gao et al. (manuscrito 2023), Tabla 4 {cite:p}`gao2023` | Medida móvil de 30 días desplazada siete días | Yahoo BTC-USD diario | LSTM de una capa y H=20; la densa lineal es contexto | RMSE y RMSPE | LSTM 0,074071 y 0,406084, respectivamente |
| S6. Wang et al. (s. f.), Tabla 4 {cite:p}`wangnd` | Volatilidad diaria denominada RV, sin fórmula suficiente para equipararla a σ7 | Bitcoin diario | Transformer OHLCV independiente; EGARCH–Transformer es híbrido | RMSE, MAE, QLIKE y R² del Modelo 1 | 1,4738; 0,4369; 0,4369; −0,7347, respectivamente, transcritos tal como aparecen |
| S7. Alakhouch Idriss (2026), Tabla 4.5 {cite:p}`alakhouch2026` | Medida futura de 30 días anualizada en el código | Bitcoin diario | LSTM 64→32 con cabeza Dense16→1; no una MLP independiente | RMSE, MAE y MAPE | 0,0358; 0,0254; 13,5 %, respectivamente |
| S8. Miura, Pichl y Kaizoji (2019), Tabla 1, p. 171 {cite:p}`miura2019` | Predicción de RV a partir de RV pasada; agregación de tres horas, horizonte numérico no precisado | Bitstamp BTCUSD minuto vía Kaggle; corpus 2012–2018 | MLP de cuatro capas (dropout/BN), CNN de dos capas + Dense, LSTM y GRU independientes | RMSE de test; varianzas reportadas por separado | MLP-dropout 0,0022067; MLP-BN 0,0021952; CNN 0,0021817; LSTM 0,0021960; GRU 0,0021778 |
| S9. Djanga, Cucuringu y Zhang (2025), Tabla 2, p. 19/PDF20 {cite:p}`djanga2025commonality` | RV intradiaria y trayectoria del día siguiente; suma de retornos cuadrados | Veinte pares USDT de Binance vía tardis.dev, enero de 2021–junio de 2023; segmento Spot/futuros no precisado | LSTM propuesta UAM y cabeza MLP opcional; sin MLP/CNN independientes verificados | QLIKE de seis ganadores HAR/LightGBM; sin fila LSTM numérica separada recuperada | 1,11; 0,75; 0,76; 0,57; 0,80; 0,84, cada uno con su segmento/modelo en la extracción; no son métricas LSTM |

**Límites de interpretación.** Las cinco filas anteriores mantienen sus advertencias: S1 usa RV×100 y otro objetivo; S2 promedia predicciones y usa R² frente a media train; S5 informa validación escalada; S6 conserva inconsistencias de métricas/particiones; S7 presenta escalado global y discrepancias entre informe/código. **S8 añade otro objetivo e historia de información:** RV es la suma de retornos logarítmicos minuto al cuadrado en intervalos de 180 minutos, sin raíz, frente a la desviación centrada de siete retornos diarios de E1. Las entradas son rezagos de RV, no OHLCV diario. Sus métricas no se ordenan con las de los otros estudios ni con RMSE de nuestro benchmark.

La [síntesis vigente](../_static/literature_e1/crypto_extension_20261105/study_synthesis.csv) incluye siete estudios cripto y la [matriz normalizada CSV](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.csv), también [en JSON](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.json), separa los siete de T1/T2/T3. El [catálogo de 40 citas](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv) es otro producto. La [síntesis de seis](../_static/literature_e1/recovery_20261105/synthesis_six_columns.csv), la de cinco y el catálogo de 39 quedan como cortes históricos.

La [matriz histórica de aquel corte](../_static/literature_e1/evidence_matrix.csv) conserva problema, dataset, arquitectura, resultados y localización. El [registro de métricas](../_static/literature_e1/reported_metrics.csv) distingue los 42 valores del corte original y los 39 transcritos de S6/S7: **81 valores de autores**, no 81 resultados independientes ni experimentos propios. No se corrigen silenciosamente cifras impresas inconsistentes ni se trasladan a la escala de E1.

**S9: papel y valores recuperados.** La [extracción primaria](../_static/literature_e1/crypto_extension_20261105/S9_extraction.json) identifica el manuscrito fechado el 26/01/2025, sus veinte pares y el RV de 30 minutos representado en rejilla de cinco minutos. Los autores describen una trayectoria intradiaria del día siguiente; las fórmulas y el horizonte deben reconciliarse antes de replicar. La LSTM se usa como técnica propuesta UAM. Su búsqueda permite H=8/16/32 y una a cinco capas, con cabeza densa opcional; la arquitectura ganadora no quedó identificada. Esa cabeza no es un MLP independiente. El código de autor no se ejecutó.

| Segmento de S9 | Ganador rotulado por autores | QLIKE, Tabla 2 |
|---|---|---:|
| L_train=1w | AR(CAM,1w,LightGBM) | 1,11 |
| L_train=1m | AR(CAM,1w,LightGBM), etiqueta repetida tal como está impresa | 0,75 |
| L_train=6m | HAR_EQ(CAM,6m,VIXM) | 0,76 |
| Régimen de volatilidad baja | AR(UAM,1m,LightGBM) | 0,57 |
| Régimen normal | HAR(UAM,1m,LightGBM) | 0,80 |
| Régimen alto | HAR_EQ(CAM,6m,VIXM) | 0,84 |

Son cifras de **ganadores clásicos/árboles, no de LSTM**, transcritas en el [registro actual de métricas](../_static/literature_e1/crypto_extension_20261105/reported_metrics.csv). Se conserva la inconsistencia de la etiqueta 1w bajo la columna 1m. No se asignan valores desde gráficos o bases de resultados no consultadas. La selección por desempeño OOS, la falta de repeticiones/IC neuronales y la descripción incompleta de todas las fronteras de winsorización y clustering limitan la inferencia; no se trasladan mejoras de trading al benchmark académico propio.

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

Para una salida continua, **SVR lineal** es el ejemplo clásico pertinente de la guía. Equivalencia funcional significa establecer una referencia de capacidad controlada; no igualdad de funciones, supuestos o errores:

| Referencia | Representación y capacidad | Papel |
|---|---|---|
| SVR lineal / Ridge | Función afín regularizada sobre covariables vectorizadas | Contexto clásico; no ocupa un puesto DL |
| GRU | Recurrencia no lineal con reinicio y actualización de un estado oculto | Referencia DL independiente y compacta |
| LSTM | Recurrencia con tres compuertas y memoria de celda separada | Referencia DL independiente con distinta parametrización |
| TCN | Convoluciones causales dilatadas, conexiones residuales y lectura temporal final | Referencia DL independiente sin estado recurrente |

Comparar las redes exige información disponible, objetivo, partición, preprocesamiento y evaluación comunes. No exige que sus mecanismos internos coincidan. Los Ridge del bloque C siguen como diagnósticos del EDA; las métricas, incertidumbre y selección del baseline neuronal vigente se informan en el capítulo 5 y en el [registro propio de selección por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json).

### 3.2.5. Arquitecturas no especificadas y búsqueda posterior

La guía remite a la búsqueda de arquitectura del **Entregable 2, sección 14** cuando el estudio no define una receta concreta. Un campo no reportado se mantiene desconocido; no se rellena con parámetros de este proyecto. El documento completo de E2 no se ha aportado y no se afirma haber ejecutado su procedimiento íntegro.

S1 sí publica ganadores GRU/LSTM/TCN en su apéndice; su complejidad figura en §3.2.1.2. Adoptar H32 en las recurrentes y una TCN H16 de tres bloques es una **adaptación compacta**, no la reconstrucción de un dato ausente ni una réplica de aquellos ganadores. Las especificaciones y fórmulas propias permiten reproducir lo implementado sin atribuir esa receta a los artículos.

S8 deja anchos, activaciones y otros detalles sin reportar; la dirección de su LSTM no se precisa y su CNN no tiene dimensión acreditada. Esas lagunas pertenecen a la extracción de S8 y se conservan. La terna vigente ya no depende de convertir esa CNN en Conv1D. Para TCN, el límite bibliográfico principal es su **única referencia independiente recuperada**, no una supuesta falta de dimensión de la arquitectura de S1.

El experimento vigente no ejecutó HPO nuevo: mantuvo H32/LR0,003 para LSTM, los transfirió a GRU y usó H16/LR0,001 para la TCN compacta, comparando D=7/14/21/28 y diez semillas. Su protocolo se congeló antes de esos ajustes; el desarrollo 2024 y el test histórico ya habían sido observados en etapas previas. Esto permite una comparación exploratoria de recetas fijas, sin presentarla como validación anidada o confirmación en mercado nuevo.

La [documentación original del procedimiento](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/gru_lstm_tcn_20261105/original_registration) se conserva en el archivo fuente; no se presenta como una búsqueda nueva de arquitecturas para esta terna. El [adaptador público](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn.py) se preparó después de la ejecución privada para facilitar reproducción. No es el archivo que produjo retrospectivamente sus pesos; la procedencia del export conserva los hashes de los archivos originales y la diferencia de versiones.

### 3.2.6. Familias de arquitecturas orientativas por tipo de problema

Las siguientes familias son una orientación de la guía, no resultados de frecuencia ni una selección automática. La elección final debe proceder de la literatura del problema y cumplir los criterios anteriores. El contrato actual usa covariables OHLCV temporales y una salida continua escalar:

| Tipo de problema | Familias orientativas | Aplicabilidad al proyecto y motivo |
|---|---|---|
| Clasificación y regresión tabular | MLP superficial, MLP profundo regularizado y red residual tabular | No es la ruta principal; la cronología condiciona la partición. S8 acredita MLP aplicada a RV autorregresiva, no convierte el contrato propio en tabular i.i.d. |
| Series de tiempo | MLP sobre rezagos, RNN simple, LSTM o GRU de una capa, CNN 1D/TCN poco profunda | Sus diagnósticos temporales son pertinentes. No se adopta como ruta principal porque el contrato actual construye rezagos de covariables, sin usar rezagos explícitos de la variable objetivo. |
| Regresión temporal | MLP sobre covariables rezagadas, LSTM o GRU sobre sus secuencias y CNN 1D poco profunda | Es la ruta aplicable. La terna GRU–LSTM–TCN usa secuencias de covariables; la TCN compacta es una adaptación declarada del proyecto y no el ganador de S1. |
| Regresión espacial | MLP con coordenadas y variables, MLP con embeddings posicionales y CNN sobre rejilla de vecindad | No aplica: no hay posiciones geográficas ni unidades espaciales que definan vecindad. |
| Regresión espacio-temporal | LSTM o GRU global sobre ventanas, ConvLSTM y convolución sobre grafo o rejilla poco profunda | No aplica: existe orden temporal, pero no múltiples ubicaciones ni una estructura espacial del objetivo. |
| Clasificación de imágenes | CNN poco profunda estilo LeNet/VGG reducida, CNN pequeña con normalización por lotes y ResNet de baja profundidad | No aplica: el contrato no recibe imágenes ni predice clases visuales. |
| Segmentación de imágenes | FCN sencilla, codificador–decodificador convolucional básico y U-Net reducida | No aplica: no hay máscaras ni etiquetas por píxel. |
| Clasificación de texto | MLP sobre embeddings promediados (tipo fastText con red), CNN de texto y LSTM o GRU uni/bidireccional de una capa | No aplica: el contrato propio no incluye un corpus textual de entrada. El uso de Twitter por S1 es contexto bibliográfico. |
| Etiquetado secuencial | LSTM o GRU bidireccional con clasificación por token, la misma con CRF y CNN dilatada | No aplica: no se asigna una etiqueta a cada token o elemento de entrada. |
| Aprendizaje secuencia a secuencia | Codificador–decodificador GRU o LSTM, la misma arquitectura con atención aditiva y codificador–decodificador convolucional sencillo | No aplica: la salida del proyecto es una volatilidad escalar, no una secuencia destino. |

Un encoder no convierte por sí solo el problema en secuencia a secuencia; una convolución temporal tampoco implica imágenes. Las orientaciones no aplicables se conservan con su justificación y no añaden experimentos ni puestos al trío.

## 3.3. Fichas técnicas de los tres modelos base implementados

Las fichas vigentes corresponden a **GRU, LSTM y TCN**. Cada una conserva los nueve incisos: referencias, arquitectura, descripción técnica, problema, dataset, métricas, fortalezas, limitaciones y complejidad. La evidencia publicada respalda familias y papeles comparativos; las recetas pequeñas y los resultados propios se identifican por separado.

El contrato común usa **D∈{7,14,21,28} días × cinco canales OHLCV**, σ7 poblacional de retornos logarítmicos (`ddof=0`, sin anualización), salida escalar y horizonte h=1. D cambia la historia de entrada, no los siete retornos que definen el objetivo. Las entradas terminan antes del día objetivo y no añaden rezagos explícitos de σ7. El OHLC se expresa respecto al último cierre observable de la ventana; al volumen se aplica `log1p`. Los escaladores X/Y se ajustan solo al entrenamiento permitido y se invierten las salidas antes de evaluar.

La CV comprende doce bloques mensuales de 2024, embargo D, parada interna cronológica y reajuste de una red nueva con el número de épocas seleccionado. Se usan diez semillas 4101–4110 por modelo e historia: **120 trayectorias y 1.440 segmentos modelo–fold**, con dos ajustes por segmento. Las tres redes emplean AdamW, dropout 0,1, weight decay 0,0001, batch 32, clipping de norma 1, máximo interno de 100 épocas y paciencia 12. El [resumen propio](../_static/gru_lstm_tcn_20261105/summary.json) conserva el resultado de la ejecución; los [modelos públicos](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models_gru_lstm_tcn.py) son una adaptación portable preparada después de la ejecución privada, cuya procedencia se registra por separado.

### 3.3.1. GRU unidireccional

#### 3.3.1.1. Referencias completas y papel comparativo

**Aplicaciones directas.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023), *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*, ACM WSDM, DOI [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387), Tabla 2 {cite:p}`akbiyik2023`. Miura, R., Pichl, L. y Kaizoji, T. (2019), *Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series*, ISNN, LNCS 11554, pp. 165–172, DOI [10.1007/978-3-030-22796-8_18](https://doi.org/10.1007/978-3-030-22796-8_18), Tabla 1 {cite:p}`miura2019`. Ambos originales contienen GRU independiente; S2 añade un tercer estudio con el estrato de procedencia ya declarado. Frecuencia estricta: **3/7**.

**Fundamento.** Cho et al. (2014), *Learning Phrase Representations Using RNN Encoder–Decoder for Statistical Machine Translation*, prepublicación arXiv:1406.1078 {cite:p}`cho2014`. Es fundamento del mecanismo, no un experimento financiero ni un artículo original nuevo leído en esta etapa.

#### 3.3.1.2. Tipo de arquitectura

Una capa GRU unidireccional de **32 unidades**, estado inicial nulo, dropout 0,1 sobre el último estado y cabeza lineal escalar con Softplus. No incluye atención, bidireccionalidad, otra capa recurrente o estado persistente entre ventanas.

#### 3.3.1.3. Descripción técnica

GRU tiene **dos compuertas**, actualización z y reinicio r, y una proyección candidata n. Para xⱼ∈R⁵ y hⱼ∈R³², la convención de PyTorch puede escribirse:

$$
r_j=\sigma(W_rx_j+b_{ir}+U_rh_{j-1}+b_{hr}),\quad
z_j=\sigma(W_zx_j+b_{iz}+U_zh_{j-1}+b_{hz}),
$$

$$
n_j=\tanh\!\left(W_nx_j+b_{in}+r_j\odot(U_nh_{j-1}+b_{hn})\right),\quad
h_j=(1-z_j)\odot n_j+z_j\odot h_{j-1}.
$$

Flujo: **(B,D,5) → GRU32 → último estado (B,32) → dropout → Dense1/Softplus → (B)**. La memoria está integrada en h; no existe un estado de celda separado c. Hay tres proyecciones parametrizadas, pero n no es una tercera compuerta. Se minimiza MSE de Y escalada con **learning rate 0,003**.

#### 3.3.1.4. Tipo de problema que aborda

Regresión temporal de la siguiente σ7 a partir de covariables OHLCV observables: tipo 4, ruta 6.4, h=1. La recurrencia modela orden dentro de la ventana, sin añadir historia explícita de volatilidad.

#### 3.3.1.5. Dataset y unidad de observación

Binance Spot BTCUSDT, velas de un minuto agregadas a días UTC. Cada ejemplo es una ventana de D días y una etiqueta diaria; no se rellenan huecos ni se unen días separados por ellos. Los doce bloques de 2024 comparten cohortes crecientes, reglas causales y diez semillas con las otras dos redes. Es el dataset propio, distinto de Twitter/RV intradiaria de S1 y de RV de tres horas de S8.

#### 3.3.1.6. Métricas y valores

**Autores:** S1, Tabla 2, reporta RMSE **4,23 ± 5,61** en veinte corridas y escala RV×100. S8, Tabla 1, reporta RMSE de test **0,0021778**, con entrada GRU de cinco observaciones de su RV. No son métricas del contrato σ7/OHLCV ni un intervalo de confianza común.

**Proyecto:** RMSE anual por trayectoria es la métrica primaria; MAE, MSE, R², MAPE y MASE son secundarias. Las medias/DE entre semillas, resultados por historia y configuración seleccionada están en el [resumen vigente](../_static/gru_lstm_tcn_20261105/summary.json), la [selección por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) y el capítulo 5. Para la historia seleccionada de **D=7**, la RMSE de CV es **0,00515457 ± 0,00016156** (media ± DE muestral de diez semillas), en unidades propias de σ7.

#### 3.3.1.7. Fortalezas

Representa dependencias ordenadas con parámetros compartidos y usa menos pesos que la LSTM propia con H=32. Cuenta con dos originales publicados directos y tres referencias en el corpus total. Su menor parametrización facilita un contraste de capacidad controlada, sin garantizar menor error o menor tiempo de entrenamiento.

#### 3.3.1.8. Limitaciones

El cómputo recurrente depende del paso anterior. Comparte un solo estado para actualización y salida, a diferencia de la celda LSTM; no se deduce que eso sea siempre mejor o peor. La receta H32 no es una réplica de S1/S8. Las diez semillas describen variabilidad de ajuste sobre las mismas fechas y el desarrollo ya observado; no son diez mercados independientes.

#### 3.3.1.9. Complejidad computacional

Con tres proyecciones, dos sesgos por proyección y cabeza escalar:

$$
P_{\mathrm{GRU}}=3H(p+H+2)+(H+1)=3\cdot32(5+32+2)+33=3\,777.
$$

Costo dominante por ventana O(3DH(p+H)); activaciones O(DH). Aumentar D incrementa cómputo, no el número de pesos de esta receta. Los tiempos medidos corresponden al entorno de la ejecución y no se extrapolan a latencia de producción.

### 3.3.2. LSTM unidireccional

#### 3.3.2.1. Referencias completas y papel comparativo

Akbiyik et al. (2023), *Ask “Who”, Not “What”*, ACM WSDM, DOI [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387), Tabla 2, y Miura, Pichl y Kaizoji (2019), *Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series*, ISNN, LNCS 11554, pp. 165–172, DOI [10.1007/978-3-030-22796-8_18](https://doi.org/10.1007/978-3-030-22796-8_18), Tabla 1, usan LSTM independiente {cite:p}`akbiyik2023,miura2019`. La familia alcanza **5/7** al incluir S2/S5/S7; la variante unidireccional está explícita en S1/S5/S7, **3/7**. No se borra la heterogeneidad de esas fuentes.

El fundamento es Hochreiter, S. y Schmidhuber, J. (1997), *Long Short-Term Memory*, *Neural Computation* 9(8), 1735–1780, DOI [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) {cite:p}`hochreiter1997`.

#### 3.3.2.2. Tipo de arquitectura

Una capa LSTM unidireccional de **32 unidades**, estados iniciales nulos, dropout 0,1 sobre el último estado oculto y cabeza lineal escalar con Softplus. No utiliza atención, bidireccionalidad ni otra capa recurrente.

#### 3.3.2.3. Descripción técnica

LSTM tiene **tres compuertas** —entrada i, olvido f y salida o—, una transformación candidata g y un **estado de celda c separado de h**:

$$
(i_j,f_j,o_j)=\sigma(W_{i,f,o}x_j+U_{i,f,o}h_{j-1}+b_{i,f,o}),\quad
g_j=\tanh(W_gx_j+U_gh_{j-1}+b_g),
$$

$$
c_j=f_j\odot c_{j-1}+i_j\odot g_j,\qquad h_j=o_j\odot\tanh(c_j).
$$

Flujo: **(B,D,5) → LSTM32 → último estado h (B,32) → dropout → Dense1/Softplus → (B)**. Cada b en la notación resume dos vectores de sesgo PyTorch, contados por separado. Se minimiza MSE de Y escalada con **learning rate 0,003**. La distinción entre c y h permite contrastar una memoria separada con el estado integrado de GRU; no prueba superioridad funcional.

#### 3.3.2.4. Tipo de problema que aborda

La misma regresión temporal σ7/h=1 con OHLCV rezagadas. El estado solo recorre cada ventana; no se conserva entre ejemplos ni se actualizan pesos con la etiqueta del día pronosticado.

#### 3.3.2.5. Dataset y unidad de observación

Binance Spot BTCUSDT diario UTC, D=7/14/21/28 y cinco canales. Comparte las doce validaciones mensuales de 2024, cohortes, embargo D, transformación causal y diez semillas con GRU/TCN. El calendario ya observado mantiene el carácter exploratorio de esta evaluación.

#### 3.3.2.6. Métricas y valores

**Autores:** S1, Tabla 2, informa RMSE **2,37 ± 0,19** en veinte corridas, en escala RV×100. S8, Tabla 1, informa RMSE de test **0,0021960** con entrada LSTM de doce observaciones de su RV. Se mantienen sus tareas y escalas originales.

**Proyecto:** RMSE es primaria; MAE, MSE, R², MAPE y MASE complementan el análisis en escala original. El [resumen propio](../_static/gru_lstm_tcn_20261105/summary.json) y el capítulo 5 presentan todas las historias y semillas. Los resultados se identifican por ejecución y no se mezclan con cifras de autores. Para la historia seleccionada de **D=7**, la RMSE de CV es **0,00633541 ± 0,00018701** (media ± DE muestral de diez semillas), en unidades propias de σ7.

#### 3.3.2.7. Fortalezas

Tiene la mayor frecuencia familiar recuperada de la terna y antecedentes de referencia independientes. Comparte parámetros entre posiciones y conserva número de pesos al variar D. Su celda separada permite examinar otra forma de memoria recurrente bajo el mismo contrato.

#### 3.3.2.8. Limitaciones

Es la mayor de las tres recetas por parámetros y mantiene dependencia secuencial. Más compuertas no garantizan aprender memoria útil para volatilidad. El antecedente S1 tiene otro objetivo e información y S8 otra historia de entrada; tampoco las fuentes grises añaden automáticamente evidencia de igual calidad. La comparación propia sigue siendo exploratoria en un calendario ya conocido.

#### 3.3.2.9. Complejidad computacional

Con cuatro proyecciones, dos sesgos por proyección y cabeza escalar:

$$
P_{\mathrm{LSTM}}=4H(p+H+2)+(H+1)=4\cdot32(5+32+2)+33=5\,025.
$$

Costo dominante O(4DH(p+H)); activaciones O(DH). El ancho común H32 no iguala los parámetros con GRU ni el tiempo real, que también depende de épocas y ejecución.

### 3.3.3. TCN causal dilatada residual

#### 3.3.3.1. Referencias completas y papel comparativo

Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023), *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*, ACM WSDM, DOI [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387), Tabla 2, presenta **TCN como comparador independiente**; D-TCN es la propuesta avanzada {cite:p}`akbiyik2023`. Es una referencia directa de volatilidad de Bitcoin, **1/7** del corpus, sin amplia recurrencia demostrada.

Bai, S., Kolter, J. Z. y Koltun, V. (2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv:1803.01271, es fundamento general de TCN; el registro conserva acceso a README/cita de autores y no afirma nueva lectura completa ni revisión por pares de ese preprint {cite:p}`bai2018`.

#### 3.3.3.2. Tipo de arquitectura

TCN de **tres bloques residuales**, dos Conv1D causales por bloque, 16 canales, kernel 3, stride 1 y dilataciones de bloque **1, 2 y 4**. Cada convolución lleva ReLU y dropout 0,1; hay proyección residual 1×1 cuando cambian los canales. La salida toma la **última posición temporal**, seguida de dropout, cabeza escalar y Softplus. No usa pooling global, atención o normalización temporal.

#### 3.3.3.3. Descripción técnica

Flujo: **(B,D,5) → transposición (B,5,D) → bloques residuales d=1/2/4 → último paso (B,16) → dropout → Dense1/Softplus → (B)**. Cada convolución de dilatación d rellena solo a la izquierda con (k−1)d ceros y mantiene longitud D. La salida en j depende únicamente de posiciones ≤j de la ventana. Las conexiones residuales suman identidad o proyección de canales a la rama de dos convoluciones y aplican ReLU a esa suma.

Con dos convoluciones por bloque, el campo receptivo máximo es:

$$
R=1+2(k-1)(1+2+4)=1+2\cdot2\cdot7=29.
$$

R=29 cubre D=7/14/21/28; para D28 hay como máximo **28 observaciones reales**, y el soporte restante es padding anterior a la ventana. La lectura final recoge dependencias causales a varias escalas en la última posición temporal. Se minimiza MSE de Y escalada con **learning rate 0,001**. La receta es propia, no la TCN de seis niveles/F287/k5 ganadora de S1.

#### 3.3.3.4. Tipo de problema que aborda

Regresión temporal σ7/h=1 con las mismas covariables rezagadas. Las convoluciones se aplican al eje de días; no implican imágenes, ubicaciones o una tarea de secuencia a secuencia.

#### 3.3.3.5. Dataset y unidad de observación

Binance Spot BTCUSDT diario UTC, ventana D×5 y salida escalar. Comparte ejemplos, cohortes, calendario, diez semillas y reglas de preprocesamiento con GRU/LSTM. La causalidad se establece respecto a las posiciones de la ventana y al reloj de disponibilidad de sus OHLCV; no basta por sí sola para garantizar toda la validez del pipeline.

#### 3.3.3.6. Métricas y valores

**Autores:** S1, Tabla 2, reporta para su TCN independiente RMSE **2,58 ± 0,07** en veinte corridas, en escala RV×100. No se atribuye ese valor a nuestra TCN compacta. Las variantes D-TCN y su comparación se conservan en §3.4.3.

**Proyecto:** RMSE anual por trayectoria es primaria, con MAE, MSE, R², MAPE y MASE secundarias. El [resumen vigente](../_static/gru_lstm_tcn_20261105/summary.json), la [selección por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) y el capítulo 5 presentan resultados propios. Para la historia seleccionada de **D=7**, la RMSE de CV es **0,00581961 ± 0,00025622** (media ± DE muestral de diez semillas), en unidades propias de σ7.

#### 3.3.3.7. Fortalezas

Aporta filtros causales compartidos, conexiones residuales y un campo receptivo multiescala explícito. Permite procesar posiciones de una capa en paralelo sin mantener un estado recurrente. Tiene 4.289 parámetros y cubre las cuatro historias con la misma arquitectura; ello acredita capacidad pequeña y fija, sin demostrar mejor precisión o menor costo real.

#### 3.3.3.8. Limitaciones

Su respaldo estricto es **un estudio recuperado**. La capacidad pequeña exige alejarse de la configuración ganadora de S1 y evaluar esa adaptación. El campo receptivo finito y el padding cambian el soporte efectivo con D. Las conexiones residuales y dilataciones no eliminan errores de preprocesamiento, selección o dependencia estadística. No se afirma que TCN sea ampliamente frecuente o que una CNN dimensionalmente no especificada de S8 constituya otra ocurrencia TCN.

#### 3.3.3.9. Complejidad computacional

Con p=5, H=16 y k=3, las seis convoluciones incluyen sesgos, la primera conexión residual proyecta 5→16 y la cabeza tiene 17 parámetros:

$$
P_{\mathrm{TCN}}=H(pk+1)+5H(Hk+1)+H(p+1)+(H+1)
=256+3\,920+96+17=4\,289.
$$

Costo dominante O(Dk(pH+5H²)), más proyección residual y cabeza; activaciones O(6DH). D modifica cómputo y soporte, no el número de pesos de esta receta. No se deduce eficiencia de producción a partir del paralelismo teórico o solo de los parámetros.

### 3.3.4. Cobertura y comparación de las tres fichas

| Campo | GRU | LSTM | TCN |
|---|---|---|---|
| Comparador directo publicado | S1 y S8 | S1 y S8 | S1 |
| Frecuencia estricta, N=7 | 3/7 | 5/7, incluyendo BiLSTM | 1/7 |
| Memoria / soporte | Estado integrado, dos compuertas | Estado oculto y celda separados, tres compuertas | Filtros causales dilatados y residuales; R=29 |
| Receta propia | Una capa H32 | Una capa H32 | Tres bloques / seis Conv1D H16, k3, d1/2/4 |
| Lectura | Último estado | Último estado oculto | Última posición temporal |
| Parámetros | 3.777 | 5.025 | 4.289 |
| Límite bibliográfico principal | Transferencia de tarea y receta | Familia heterogénea y transferencia de tarea | Solo una referencia independiente recuperada |

Los nueve campos están desarrollados para las tres redes. Las fuentes prueban el uso comparativo de las familias y conservan desconocidos; la ejecución propia acredita estas configuraciones concretas. Ninguno de esos hechos garantiza generalización, validación prospectiva o mínima complejidad global.

## 3.4. Comparación crítica de los modelos y de la evidencia

La comparación principal vigente enfrenta **GRU, LSTM y TCN** bajo el contrato σ7/h=1 y CV comunes. La evidencia aplicada de S1/S8 y el resto del corpus sigue separada de los resultados propios. MLP/CNN permanecen en las tablas de autores, sin ocupar puestos en esta comparación.

### 3.4.1. Comparación estructurada entre los tres modelos base

#### 3.4.1.1. Desempeño de CV y comparabilidad

El estudio propio usa las cuatro historias D=7/14/21/28, cinco canales y diez semillas. Cada RMSE anual se calcula reuniendo las 366 fechas de una trayectoria antes de resumir media/DE entre semillas. No es la RMSE de un ensemble ni una media de R² mensuales. Se publican todas las configuraciones en el [resumen actual](../_static/gru_lstm_tcn_20261105/summary.json) y el baseline seleccionado por CV en el [registro vigente](../_static/gru_lstm_tcn_20261105/selected_by_cv.json); el capítulo 5 presenta métricas, contrastes y evaluación propia.

La mejor historia por familia en ese registro es D7:

| Familia propia | Historia seleccionada por CV | RMSE anual 2024, media ± DE de diez semillas | Parámetros |
|---|---:|---:|---:|
| GRU | 7 | 0,00515457 ± 0,00016156 | 3.777 |
| LSTM | 7 | 0,00633541 ± 0,00018701 | 5.025 |
| TCN | 7 | 0,00581961 ± 0,00025622 | 4.289 |

**GRU D7** es el baseline neuronal vigente elegido por media de RMSE de CV. La tabla resume las recetas seleccionadas; las doce combinaciones modelo–historia siguen en el registro y no se descartan las de mayor error. Estas cifras son propias y no se comparan numéricamente con RV×100 o RV de tres horas de los autores.

Las recetas se fijaron antes de los ajustes privados y la decisión de publicar esta terna fue posterior a sus resultados. La comparación sigue siendo **exploratoria sobre desarrollo ya observado**, sin convertir la congelación interna en una selección global prospectiva. Las tablas propias corresponden únicamente a las tres redes vigentes. Los RMSE de distintos artículos tampoco forman un ranking común con el contrato propio.

#### 3.4.1.2. Robustez y evidencia disponible

Las tres redes comparten historias, fechas, semillas y reglas de entrenamiento. Las diez inicializaciones permiten describir variabilidad de ajuste condicionada al mismo mercado; no son diez muestras independientes de mercados. Los contrastes, correcciones e intervalos deben leerse con la unidad de análisis y el calendario que realmente emplean. La selección pública posterior a revisar el experimento privado impide tratar esas mismas métricas como confirmación independiente de la decisión.

El respaldo estricto es GRU 3/7, LSTM familiar 5/7 y TCN 1/7. S9 eleva la presencia LSTM a 6/7 al incluir una propuesta, sin cambiar su frecuencia como referencia. No se verificaron en este ciclo otros activos, proveedores o perturbaciones controladas. Menor RMSE, cierta estabilidad por mes o un campo receptivo suficiente no acreditan robustez universal.

#### 3.4.1.3. Escalabilidad y simplicidad

| Receta propia | Parámetros | Costo dominante por ventana | Diferencia relevante |
|---|---:|---|---|
| GRU H32 | 3.777 | O(3DH(p+H)) | Dos compuertas y un estado integrado; dependencia secuencial |
| LSTM H32 | 5.025 | O(4DH(p+H)) | Tres compuertas y memoria separada; dependencia secuencial |
| TCN H16, seis Conv1D | 4.289 | O(Dk(pH+5H²)) | Causalidad, dilatación, residuales y paralelismo por posición |

Los parámetros son independientes de D en estas recetas; el costo de procesar ventanas aumenta con D. El ancho común de las recurrentes no iguala capacidad y el mismo número de corridas no optimiza por igual las familias. Los tiempos propios dependen de épocas, CPU y concurrencia; se consultan en el registro vigente y no se toman del benchmark antiguo ni de los artículos. No se atribuyen estos conteos a S1/S8.

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

S1 **ya compara conjuntamente GRU, LSTM y TCN para volatilidad de Bitcoin**. Por tanto, la terna no constituye una combinación bibliográficamente inédita. S8 aporta otra comparación publicada GRU/LSTM. La diferencia verificable del proyecto está en su contrato —OHLCV diario de Binance Spot, σ7 centrada, h=1, cuatro historias y recetas compactas—, no en inventar esas familias o afirmar ausencia universal de trabajos equivalentes.

El corpus de siete textos no identifica la misma combinación exacta de contrato, proveedores, recetas y evaluación. Esta es una **limitación de cobertura**, sin demostrar novedad mundial. La cobertura TCN sigue siendo una referencia independiente y requiere ampliación antes de afirmar uso ampliamente recurrente. Su receta pequeña se evalúa como adaptación, sin convertir la dimensión desconocida de la CNN de S8 en una laguna de la TCN actual.

El baseline neuronal vigente se identifica en el [registro seleccionado por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json). Los [registros originales](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/gru_lstm_tcn_20261105/original_registration) permanecen en el archivo fuente del repositorio. La selección actual se toma tras literatura y revisión del experimento privado, con sus límites de selección exploratoria; no se atribuye retroactivamente a la documentación anterior. Ninguna mejora observada cambia esa cronología ni convierte brechas de autores en una mejora propia demostrada sobre un mercado nuevo.

### 3.4.5. Correspondencia y estado de cierre

| Requisito de la guía | Evidencia de esta revisión | Estado y límite |
|---|---|---|
| Cadenas completas, fechas y resultados actuales | Doce búsquedas de la fase pública previa con respuestas/hashes, más recuperación complementaria registrada y capítulo S8 verificado | Los conteos de páginas/repositorios son descubrimiento, no artículos; las etapas no se suman como un PRISMA acumulado ni consultas nativas de bases. |
| Historial complementario conservado | Diez consultas G1–G10 del 02/11/2026, intentos, páginas, 118 ocurrencias y 106 repositorios distintos; comprobantes de acceso posteriores separados | Se preserva como corte histórico, sin sumarlo al flujo público actual ni tratar los cuatro fallos directos como cobertura científica |
| Procedencia WoS/Scopus y publicaciones IEEE/arXiv | Siete revistas con coincidencias secundarias de título e ISSN en copias públicas WoS/Scopus de 2026;  arXiv 2110.14317v2 corroborado en portada de autores; enlace Xplore 11058926 declarado por autores | Evidencia de revista e identidad documental, no verificación individual de artículos en los índices, consulta nativa, cobertura histórica WoS ni Q1; páginas arXiv/Xplore en vivo no consultadas |
| Inclusión, exclusión, flujo y responsable | Cortes históricos de cinco/seis conservados; [flujo actual](../_static/literature_e1/crypto_extension_20261105/flow.svg), cribado y extracción de S9 completan siete estudios cripto | Un estudio multiactivo cuenta una vez; sin exhaustividad, consulta nativa de índices o segundo revisor humano acreditados. |
| Matriz y evaluación de calidad | Diez registros normalizados: siete cripto más T1/T2/T3; extracción de S9 y errata HARNet | S1/S8 tienen identidad de publicación; S2/S5/S6/S7/S9 mantienen su estrato gris o editorial pendiente. No se acredita validez, artículo indexado o réplica exacta. |
| Frecuencia y complejidad | Variantes, frecuencias ampliadas/sensibilidades, ganadores de S1 y conteos analíticos reproducibles | Corregido; no demuestra mínima complejidad global ni uso frecuente en todos los índices |
| Exactamente tres modelos básicos y diferencias de mecanismo | GRU–LSTM–TCN, con S1 como comparación directa publicada y S8 como apoyo adicional GRU/LSTM | Frecuencias 3/7, 5/7 y 1/7: TCN no acredita amplia recurrencia. Selección pública posterior al benchmark y al experimento privado; no se presenta como prospectiva. |
| Fichas técnicas del trío implementado | Nueve campos para GRU/LSTM/TCN; mecanismos, soporte y conteos 3.777/5.025/4.289 explícitos | Adaptaciones propias, no recetas exactas de S1/S8; R=29 frente a D≤28, con padding izquierdo. |
| Comparación crítica y brecha | Comparación propia GRU/LSTM/TCN y brecha avanzada/base de S1 separadas; las cifras de autores se conservan | No se mezclan tareas ni se declara novedad de la terna: S1 ya la compara. Evaluación exploratoria sobre calendario conocido. |
| Experimentos y baseline medido | Ejecución GRU/LSTM/TCN de cuatro historias: 120 trayectorias, 1.440 segmentos y 2.880 ajustes de CV; baseline en registro vigente | Diez semillas por receta/historia; publicación posterior a revisar el experimento privado. La procedencia distingue código original y adaptador público posterior. |

La fase actual sigue la búsqueda pública y la comprobación posterior de procedencia descritas en §3.1. Las siete coincidencias secundarias corresponden a *Journal of International Financial Markets, Institutions and Money*, *Frontiers in Physics*, *Economics Letters*, *Finance Research Letters*, *IEEE Access*, *Journal of Financial Econometrics* y *Quantitative Finance*. No sustituyen la verificación individual de los artículos ni certifican Q1. Las listas inspeccionadas no permiten juzgar actas de congresos: la ausencia de WSDM/ICDM en ellas no prueba falta de indexación.

Los comprobantes históricos, las [frecuencias previas de seis](../_static/literature_e1/recovery_20261105/family_frequencies.csv) y los catálogos anteriores conservan su contenido. La [matriz actual](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.json), el [flujo](../_static/literature_e1/crypto_extension_20261105/flow.svg) y la [extracción de S9](../_static/literature_e1/crypto_extension_20261105/S9_extraction.json) documentan el paso a siete estudios cripto. Las cifras de cada etapa se mantienen separadas; no se reetiquetan como resultados propios ni como un PRISMA exhaustivo acumulado.

## 3.5. Fundamentos bibliográficos y contexto de la revisión

### 3.5.1. Búsqueda documental efectuada y alcance de las citas

La consulta del **2 de noviembre de 2026** incorporó treinta referencias: siete sobre Bitcoin, siete de fundamentos de redes neuronales y dieciséis de volatilidad, diagnóstico y evaluación. Se consultaron repositorios públicos de autores, documentación de software académico y bibliografías de artículos y libros. Para cada entrada se conservan el enlace consultado, los campos corroborados y el nivel de acceso en el [registro de verificación](../_static/bibliografia/verification_30.csv) y el [catálogo detallado](../_static/bibliografia/references_30.json).

El catálogo vigente conserva las 39 referencias anteriores y añade S9: **40 referencias** en §3.1.2.4 y su [CSV](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv). Los cortes de 35/cinco textos, 36/seis textos y 39/seis textos más transferencia permanecen fechados. La ampliación documental a siete textos fue posterior al benchmark y no ejecutó entrenamientos. La decisión posterior de publicar GRU–LSTM–TCN y su ejecución privada se identifican por separado en §3.2.1.3; no se atribuyen a la ampliación búsquedas nativas de índices.

La red del entorno bloqueó el acceso directo a Crossref y arXiv. Por ello no se afirma haber consultado sus interfaces ni leído treinta originales completos. La entrada de Huang y colaboradores conserva su verificación previa; su acceso directo actual también está limitado. Las verificaciones nuevas de las otras referencias proceden de las fuentes accesibles indicadas en el registro. Una referencia identificada en otra bibliografía se distingue de un artículo original leído.

Los trabajos anteriores a 2019 se incluyen como fundamentos y antecedentes expresamente justificados. No se cuentan automáticamente como estudios recientes elegibles para calcular frecuencia de modelos base. Los registros de acceso a metadatos tampoco sustituyen el flujo complementario documentado en §3.1.

### 3.5.2. Antecedentes de volatilidad de Bitcoin

Katsiampa estudia la estimación de volatilidad de Bitcoin mediante especificaciones GARCH, mientras Dyhrberg relaciona su análisis de volatilidad con oro y dólar {cite:p}`katsiampa2017,dyhrberg2016`. Estos antecedentes justifican contrastar las redes con referencias financieras y delimitar la variable que se estima; no demuestran por sí solos qué arquitectura neuronal debe seleccionarse.

Los estudios asociados a órdenes de compra y venta, información de Twitter y flujos de información a frecuencia de minuto plantean conjuntos de información diferentes {cite:p}`guo2018,akbiyik2023,barjasic2021`. Sus títulos, datos documentados y referencias permiten reconocer líneas de investigación pertinentes, pero no autorizan atribuir a OHLCV diario mejoras obtenidas con otras señales. El artículo de Akbiyik y colaboradores se recuperó como copia de autores; para parte de los antecedentes se corroboraron metadatos en sus bibliografías y no se declara lectura del original.

Miura, Pichl y Kaizoji aportan ahora un **benchmark publicado de MLP, CNN, LSTM y GRU independientes para RV de Bitcoin** {cite:p}`miura2019`. La comparación de su Tabla 1 es directamente pertinente a familias neuronales para volatilidad; el cambio de proveedor, agregación, objetivo e información impide trasladar sus valores al experimento propio. Su CNN no tiene dimensión reportada y sus cuatro capas MLP se conservan como extracción de autores, sin ocupar un puesto en la terna actual. Las recetas GRU/LSTM H32 tampoco se deducen de esos detalles.

Huang y colaboradores aportan un antecedente de pronóstico de volatilidad con técnicas de aprendizaje automático. Herremans y Low se centran en picos de volatilidad e información de transacciones de grandes tenedores y CryptoQuant mediante modelos Synthesizer Transformer {cite:p}`huang2024,herremans2025`. Pronosticar picos, una medida realizada intradiaria o el siguiente valor de una desviación móvil diaria constituye tareas diferentes. La pertinencia temática no acredita igualdad de objetivo, datos, horizonte o protocolo.

### 3.5.3. Definiciones de volatilidad y referencias financieras

ARCH formaliza varianza condicional cambiante y GARCH amplía esa estructura incorporando dinámica de la varianza {cite:p}`engle1982,bollerslev1986`. Son fundamentos econométricos y referencias de contexto; no ocupan los puestos reservados a modelos de Deep Learning.

La literatura de volatilidad realizada y de modelos HAR estudia construcciones y dependencias a distintas escalas {cite:p}`andersen2003,corsi2009`. Es necesario conservar la distinción entre varianza condicional, volatilidad realizada construida con alta frecuencia y desviación estándar móvil de retornos diarios. La medida definida en este proyecto no se convierte en cualquiera de las anteriores por compartir la palabra «volatilidad». Cambiar ventana, frecuencia o anualización también cambia la escala de las métricas.

### 3.5.4. Fundamentos de las fichas implementadas y sus antecedentes

El resultado de aproximación de redes feedforward y la exposición general de aprendizaje profundo fundamentan el uso de MLP como familia candidata {cite:p}`cybenko1989,goodfellow2016`. La capacidad de aproximar funciones no garantiza generalización ni acredita que una MLP sea una referencia frecuente para Bitcoin. MLP permanece como antecedente y no como modelo experimental principal de la terna vigente.

LSTM emplea tres compuertas y un estado de celda separado; GRU usa actualización y reinicio con un estado integrado. El trabajo de Cho y colaboradores documenta una arquitectura recurrente con compuertas en un codificador–decodificador {cite:p}`hochreiter1997,cho2014`. Estos fundamentos permiten describir LSTM y GRU y discutir sus diferencias de parametrización. El artículo de Cho procede de traducción automática: no se presenta como un experimento de volatilidad financiera.

Las redes convolucionales de reconocimiento documental y la comparación genérica de modelos convolucionales y recurrentes para secuencias sustentan mecanismos de filtros locales y dependencias temporales {cite:p}`lecun1998,bai2018`. Una CNN de imágenes, una CNN1D simple y una TCN no son arquitecturas idénticas. La TCN propia añade causalidad, dilataciones y conexiones residuales con lectura final; su apoyo aplicado directo procede de S1. Ese fundamento no acredita una réplica exacta, una amplia recurrencia de TCN o un ranking transferible a BTCUSDT.

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

Las **40 referencias**, los **siete estudios cripto completos** y los tres respaldos metodológicos separan cita, lectura y papel del modelo. Todos los estudios cripto incluyen BTC y uno incluye otras diecinueve criptomonedas; las categorías se solapan. La terna principal es **GRU–LSTM–TCN**, con frecuencias estrictas **3/7, 5/7 y 1/7**. MLP 1/7, CNN genérica 1/7 y CNN1D simple 0/7 permanecen como datos de la extracción, sin alterar la matriz; la presencia LSTM 6/7 incorpora la propuesta S9. TCN tiene respaldo directo de S1, con amplia recurrencia no acreditada. La selección para publicación es posterior al benchmark y a revisar el experimento privado, no una decisión preregistrada de la revisión. No se cuentan cabezas internas, código solo o identidad editorial como nuevas referencias independientes. La ampliación no demuestra recurrencia amplia, indexación individual o superioridad de las recetas propias. S9 añade cobertura y límites explícitos; la errata HARNet corrige una inferencia de fuga sin certificar validez global. T1/T2/T3 siguen separados; los originales pendientes de HARNet y DeepVol no se presentan como leídos.
