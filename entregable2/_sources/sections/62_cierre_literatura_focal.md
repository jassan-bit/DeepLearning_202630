# Elegibilidad, selección documental y brecha sustentada

> **Ubicación:** complemento del [capítulo 3](03_literatura.md). Identificador documental anterior: **§62**; [ver concordancia](concordancia_guia.md).

**Estado documental vigente.** El **flujo tipo PRISMA del corpus recibido queda reconciliado**: los **9.559 grupos** tienen cribado y disposición final, con **42 estudios aplicados únicos**, **120 evaluaciones primarias focales o reutilizadas** y **cero pendientes operativos**. Se documentan **174 registros cuyo informe no se recuperó**, con elegibilidad desconocida. La cobertura de las búsquedas históricas permanece limitada: faltan 374 registros Scopus mostrados y no se conocen todos los totales/rangos de WoS. El cierre del flujo recibido no acredita exhaustividad de las bases.

## Flujo del corpus recibido

La unidad inicial es el **grupo bibliográfico recibido**. El cribado individual asistido por IA distingue título, resumen íntegro y reutilización de una lectura con hash coincidente. Se conservaron las propuestas previas y se añadieron las adjudicaciones finales con su nivel de evidencia. No hubo doble revisión humana independiente ni se calcula κ. El protocolo es retrospectivo y la recuperación bibliográfica no causó la selección experimental anterior.

```{figure} ../_static/e2_prisma_received_v1/prisma_received.svg
:alt: Flujo reconciliado de 9559 grupos recibidos: cribado, recuperación, evaluación primaria e inclusión de 42 estudios únicos.
:width: 100%

Flujo del material recibido. Los alias conservan los registros y evitan duplicar estudios; los contenedores no aportan otro efecto ni una exclusión científica.
```

| Etapa y unidad | Resultado verificable |
|---|---:|
| Ocurrencias recibidas | 11.635 |
| Duplicados dentro de las rondas / reducciones entre rondas | 1.720 / 356 |
| Grupos cribados, una disposición por grupo | **9.559 / 9.559** |
| Resúmenes auténticos almacenados y enlazados | **8.939** |
| Grupos con resumen íntegro leído o lectura previa reutilizada durante el cribado | **2.197** |
| Exclusiones del núcleo por título, resumen o metadatos, incluidas las adjudicaciones tardías | 7.084 |
| Candidatos contextuales conservados en otra función | 2.178 |
| Grupos retenidos para comprobación documental | 297 |
| Registros cuyo informe no se recuperó / contenedores vinculados | **174 / 2** |
| Grupos con evidencia primaria / evaluaciones primarias focales o reutilizadas | **121 / 120** |
| Grupos con inclusión aplicada / estudios aplicados únicos | **44 / 42** |
| Pendientes operativos del flujo recibido | **0** |

Los **121 grupos con evidencia primaria** se distribuyen en 44 con inclusión aplicada, 34 excluidos tras secciones primarias, 41 contextos y dos informes con criterio DL no resuelto. Una evaluación reutilizable sustenta dos grupos del mismo estudio; preprint y publicación también pueden ser dos informes de un único estudio. Por eso no se intercambian los denominadores. «Evaluación primaria» significa examen de secciones localizadas de métodos/resultados o evidencia primaria previa conservada; no afirma lectura íntegra de todos los PDF.

El [registro individual de 9.559 grupos](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/received_groups_9559.csv), el [flujo estructurado](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/received_flow.json), las [decisiones de reporte](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/full_reports_compact.jsonl) y los [intentos de recuperación](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/retrieval_attempts.csv) permiten reconstruir los destinos. Los **174 no recuperados** tienen intentos documentados; no se contabilizan como exclusiones científicas ni como elegibles. Los dos informes recuperados cuya profundidad/rol DL no se pudo acreditar conservan incertidumbre científica y quedan fuera del núcleo aplicado; la operación documental está cerrada.

La [matriz aplicada](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/quality_applied_studies.csv) contiene **504 filas, doce dimensiones para 42 estudios**, con una versión preferida por estudio. La [matriz general de dictámenes](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/quality_12_dimensions.csv) tiene otro denominador: incluye acceso, contextos y exclusiones. «No verificado en las secciones examinadas» no demuestra ausencia global, cero ejecuciones o fuga. Las notas extensas y los cuerpos de artículos/resúmenes quedan fuera del paquete público; se distribuyen metadatos, hashes, localizadores y síntesis propias breves.

### Identidades y fronteras verificadas

Los [vínculos documentales](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/identity_links.csv) enlazan Crypto-Winter con su DOI y el preprint Synthesizer con su publicación editorial. Para este último se prefiere la versión final de 2025, manteniendo separadas las fórmulas y métricas del preprint de 2022. No se importan resultados de una versión como si fueran de otra. Dos volúmenes de actas se vinculan a sus artículos; no se cuentan como estudios empíricos adicionales.

Las adjudicaciones distinguen criterios excluyentes positivos de incertidumbre. El trabajo de sentimiento e IGARCH conserva un lugar contextual: su ecuación tiene orden temporal, pero las secciones evaluadas muestran estimación y significación de la varianza; no acreditan evaluación empírica del pronóstico de dispersión. El híbrido de sentimiento neuronal con HAR/ML que sí evalúa pronósticos de RV permanece incluido. Por ejemplo, una red que predice precio no acredita un pronóstico de dispersión por mencionar mercados volátiles. Una arquitectura SDE neuronal puede ser estructuralmente un SSM aunque sus cabezas tengan una sola capa oculta: su exclusión aquí se apoya en el objetivo evaluado —factores de precios, simulación y cobertura—, no en negar ese carácter neuronal. El [historial de adjudicaciones](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/report_adjudication_history.json) conserva esas correcciones.

### Síntesis que añade la revisión del corpus

La ampliación muestra contratos distintos: desviación realizada, varianza, índices/carteras, proxies de rango y componentes DL que extraen sentimiento para un predictor final econométrico. Esas funciones se identifican por fuente y no equivalen a un predictor autónomo sobre las cinco variables OHLCV de E2.

Crypto-Winter informa diez semillas agregadas y un objetivo intradía; EMGNN utiliza QML-RV y señales entre mercados. Son antecedentes relevantes para repeticiones y transferencia. La extracción distingue semillas, ventanas y configuraciones: miles de variantes experimentales no acreditan por sí mismas miles de inicializaciones independientes. La comparación externa conserva objetivo, escala, horizonte, población y versión; no se realiza pooling numérico con A2 ni se transforma el número de fuentes en superioridad de E2.

## Qué requisito se resuelve y qué evidencia lo acredita

| Producto evaluable | Estado de esta actualización | Evidencia |
|---|---|---|
| Citas uniformes de las fuentes actuales | **9/9 estudios, 10/10 sistemas**; FCN y ResNet comparten Wang | [Recibo](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/receipt.json) y [registros Scopus](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/records.csv) |
| Tabla explícita C/U/P/A/F | **Escenarios documentales conservados y comparación estratificada**; el mérito global no está identificado | [Veinte filas de los dos escenarios](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/conditional_scores.csv) y [fórmulas](../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/scenario_protocol.json) |
| Extracción y calidad de las fuentes del catálogo | **9 estudios / 10 sistemas**, con campos conocidos, NR y NV identificados | [Matriz vigente](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_quality.csv), [puente estudio–sistema](../_static/e2_literature_delivery_closure_20261114_v3/quality/model_study_bridge.csv) y disponibilidad abajo |
| Disponibilidad pública identificada | **Datos 9/9 y código de autor 9/9**; no equivale a reproducción | [Evidencia por estudio](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv) y [verificación oficial](../_static/e2_literature_delivery_closure_20261114_v3/quality/verified_urls.json) |
| Extracción y calidad aplicada | **42 estudios únicos y doce dimensiones**; el corte focal anterior conserva 17 | [Matriz aplicada vigente](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/quality_applied_studies.csv) y [matriz focal conservada](../_static/e2_literature_closure_final/quality/quality_extraction_matrix_17.csv) |
| Elegibilidad focal de treinta objetivos | **17 incluidos, once excluidos y dos contextos adjudicados**; cero revisiones o accesos pendientes | [Registro completo](../_static/e2_literature_closure_final/eligibility.csv) y [flujo focal](../_static/e2_literature_closure_final/flow.json) |
| PRISMA del corpus recibido | **Reconciliado**, 9.559 disposiciones y cero pendientes operativos; cobertura histórica limitada | [Flujo vigente](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/received_flow.json) y [conciliación histórica (§59)](59_auditoria_literatura_final.md) |

## Citas y puntuación reproducible de los diez sistemas

La exportación dirigida del usuario contiene **nueve EID únicos**, todos Scopus, con título, autores, año y número de citas. Sus identidades cubren las nueve fuentes originales de los diez sistemas. Wang identifica FCN y ResNet; TSMixer es *An All-MLP Architecture* de Chen et al., distinto del modelo de KDD. En seis registros no hay DOI editorial en la exportación: se conserva vacío y el enlace se apoya en título, autores, fuente y EID. No se suman citas de versiones.

Todos los conteos del corte bibliométrico proceden de un único archivo identificado en el recibo. El CSV no incorpora hora de consulta; el recibo distingue la fecha de recepción, la captura del usuario y los límites de la consulta nativa visible. Se conservan sin cambios los seis C previos y sus cuatro valores desconocidos en el paquete anterior. La tabla nueva usa:

```text
C_nuevo = clip(citas_Scopus / (2026 − año_inicial_del_estudio + 1) / 487, 0, 1)
S = 100 × (0,15 C + 0,15 U + 0,25 P + 0,25 A + 0,20 F)
```

El año inicial incluye la primera versión conocida del estudio; por ejemplo, Wang 2016 frente a editorial 2017 e Inception 2019 frente a editorial 2020. Se conserva la referencia fija 487, la regla de antigüedad y los pesos anteriores. El clip de PatchTST se declara; no se recalibra el denominador para obtener un resultado favorable.

| Sistema | Citas Scopus | C | U | P contextual | A | F | S documental |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCN-1D | 2248 | 0.4196 | 0.0000 | 1.0000 | 1.0000 | 0.2500 | 61.29 |
| ResNet-1D | 2248 | 0.4196 | 0.0000 | 0.9362 | 1.0000 | 0.2500 | 59.70 |
| Inception-1D | 1761 | 0.4520 | 0.0000 | 0.0000 | 1.0000 | 0.2500 | 36.78 |
| LSTM-autoencoder con predictor | 321 | 0.0824 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 46.24 |
| TimesNet | 1886 | 0.7745 | 0.4286 | 0.9712 | 1.0000 | 0.2500 | 72.33 |
| TSMixer | 247 | 0.1268 | 0.5714 | 0.7209 | 1.0000 | 0.3333 | 60.16 |
| TiDE | 247 | 0.1268 | 0.5714 | 0.6295 | 1.0000 | 0.3333 | 57.88 |
| TimeMixer | 619 | 0.4237 | 0.7143 | 1.0000 | 1.0000 | 0.3333 | 73.74 |
| PatchTST | 2477 | 1.0000 | 0.4286 | 0.8633 | 1.0000 | 0.5000 | 78.01 |
| iTransformer | 1378 | 0.7074 | 0.5714 | 0.9640 | 1.0000 | 0.5000 | 78.28 |

Las filas se muestran en el orden del catálogo, no como un nuevo ranking global. U, A y F conservan el escenario previo; C procede del nuevo corte. **La aritmética de los diez scores está completa; su interpretación sigue siendo condicional a escalas locales de desempeño que no están calibradas entre tareas.** No acredita igualdad de HPO ni eficacia sobre Bitcoin y no convierte este puntaje posterior en la causa de la selección autorizada el 12/11.

### Comparabilidad estricta y alcance de la selección

La guía §3.2, página 11, exige puntaje explícito, reproducible y desempeño en datasets comparables. **No exige un preregistro bibliográfico ni fija que el puntaje deba preceder al benchmark.** La cronología se declara; no se añade una obligación temporal que ese apartado no establece, ni se atribuye al cálculo posterior haber causado la elección pasada.

La tabla puntual anterior se conserva como escenario documental de utilidades locales. El análisis estricto separa **tarea, dataset, horizonte, lookback, métrica y fuente/tabla**. [Los puntajes por estrato](../_static/e2_literature_closure_final/selection/stratified_scores.csv) utilizan todos los comparadores de cada pool; no se unen tablas distintas como un experimento controlado único.

| Panel de pronóstico ETTh1/H96/L96/MSE | Sistemas actuales presentes | Pérdidas publicadas |
|---|---|---|
| iTransformer, tabla 10 | TimesNet, iTransformer, PatchTST, TiDE | 0,384; 0,386; 0,414; 0,479 |
| TimeMixer, tabla 13 | TimeMixer, TimesNet, PatchTST | 0,375; 0,384; 0,460 |
| TimeMixer, tabla 23, configuración unificada | TimeMixer, TSMixer | 0,375; 0,387 |

Ninguna tabla contiene conjuntamente los seis modelos. PatchTST cambia de **0,414 a 0,460** entre fuentes pese al mismo dataset/horizonte/lookback nominal: no se acredita igualdad de implementación, preprocesamiento o HPO por compartir el nombre del dataset. UCR44/MPCE y UCR2018/128 se mantienen separados. En Beijing, separar L25 de L30 deja a SAE sin pool comparable no degenerado; se informa no identificación y no se imputa cero.

Para la prioridad global se publica la [tabla completa con límites algebraicos](../_static/e2_literature_closure_final/selection/tabla_principal.csv): si P global no está identificado, `S_global∈[B;B+25]`, donde `B=100(0,15C+0,15U+0,25A+0,20F)`. Con los pesos originales, **los diez modelos admiten rangos 1–10 y no existe dominancia robusta de puntaje**. Esos límites no son IC ni métricas nuevas. La sensibilidad al peso de P=0,10/0,40 está en [rangos y dominancias](../_static/e2_literature_closure_final/selection/robust_global_P_bounds.csv); no se elige el peso favorable.

La [justificación individual del catálogo](../_static/e2_literature_closure_final/selection/catalogue_selection_rationale.csv) identifica mecanismo, fuente, código y transferencia preliminar a D7, con cuatro familias y diez sistemas. Las [exclusiones históricas identificadas](../_static/e2_literature_closure_final/selection/named_exclusions_preserved.csv) conservan su motivo y citas conocidas, sin atribuir inferioridad demostrada a los omitidos. **El alcance defendible es una selección documental por mecanismos y diversidad con análisis de desempeño estratificado; no una afirmación de Top 10 óptimo global.**

(e2-lit-inception-single)=
### Desempeño de una red Inception, separado del ensemble

Se verificaron los [resultados numéricos del autor](https://github.com/hfawaz/InceptionTime/tree/952d115e83fb3a66d75c5858c702f8dd5eeb18b7): el archivo de redes individuales contiene **640 filas, 128 datasets UCR2018 × cinco ejecuciones**, y el del ensemble de cinco redes contiene 128 filas para los mismos nombres. El promedio uniforme del error de cada ejecución individual es **0,1639239119738714**; el error medio del ensemble es **0,1564056859794546**. Promediar errores de redes individuales no combina sus predicciones.

El nuevo pool local contiene esas dos variantes publicadas y utiliza `P=(máxima pérdida−pérdida)/(máxima pérdida−mínima pérdida)`. Produce **P=0 para la red individual**: sólo significa la mayor pérdida entre esas dos variantes, no eficacia absoluta nula. Es error medio de clasificación, **no MPCE**, no la tabla UCR85 del paper ni una medición del predictor D7 de E2. No se verificaron los bytes de los splits originales ni igualdad de coste. Se conservan los [CSV, comprobaciones y cálculo](../_static/e2_literature_delivery_closure_20261114_v3/inception_single/candidate.json).

La regla P9 anterior permanece intacta. La extensión de Inception se identifica como **nuevo escenario de artefactos de autor UCR2018/128**, formulado después de ver esos datos. Los otros nueve P siguen sus pools anteriores: UCR44 para FCN/ResNet; RMSE de Beijing para SAE; MSE ETTh1/H96/L96 dentro de las tablas elegidas para los seis predictores restantes. Se publica también el escenario de **49/50 componentes**, que conserva P9 sin esa extensión. Las sensibilidades MAE/SMAPE anteriores no se eliminan; SAE sigue teniendo P=0 con RMSE y P=1 con esas métricas alternativas en su pool.

## Revisión primaria de los cuatro originales y cierre focal

El denominador sigue siendo **30 objetivos por DOI**. Los cuatro textos aportados se identifican por sus [SHA256 y extensión](../_static/e2_literature_closure_final/received_originals.json); no se redistribuyen los PDF. La revisión da **tres inclusiones narrativas y una exclusión del núcleo aplicado DL**, según los criterios que ya existían:

| Estudio | Decisión y evidencia primaria | Límites materiales conservados |
|---|---|---|
| Framework AMEM–LSTM, DOI `10.1016/j.physa.2021.126613` | **Incluido.** BTC; residuo multiplicativo modelado con LSTM; proxy Garman–Klass convertido en desviación anualizada porcentual, h=1. Lectura del texto completo y comprobación visual de ecuaciones/tablas | MSE propuesto 1.230,32 frente a 1.921,94 de AMEM en tabla; prosa 1.229,80 discrepante. Veinte reajustes del propuesto no acreditan ese presupuesto en cada comparador; no se atribuye ajuste múltiple de DM no verificado |
| Stacking, DOI `10.1007/s10614-024-10713-2` | **Incluido.** BTC; LSTM empírico y NBEATSx; varianza realizada y proxy AbsVol separados de VaR. Lectura focal de métodos, resultados y tablas del original | La tabla 5 coincide con condición sin intradía de tabla 8: no se la atribuye a intradía. HAR gana los errores individuales de RV; el ganador de stacking cambia con la métrica. Split final, validación y ámbito de PCA/clustering incompletamente reconstruidos |
| CEEMDAN–RF–LSTM, DOI `10.1016/j.future.2024.04.043` | **Incluido.** BTC/ETH/BNB; ecuación 1 define suma diaria de retornos intradía al cuadrado, sin raíz; LSTM y reconstrucción verificadas. Lectura del cuerpo completo y tablas seleccionadas | Figura 4 sitúa descomposición antes de partición 7:2:1: riesgo de anticipación, sin prueba de fuga ejecutada. Horizonte y escala exactos no establecidos; MSE/RMSE de tabla 4 no concilian. RF sin LSTM y otras descomposiciones ganan algunas condiciones |
| Jordan, DOI `10.1016/j.physa.2022.127158` | **Excluido sólo del núcleo DL; conservado como comparador neuronal.** §2.3/figura 2 verifica una capa oculta de JNN, sin LSTM/GRU; profundidad de NLANN no reconstruida. No basta el título para cumplir el rol DL de la regla v2 | Sí pronostica dispersión cripto; diferencia con σ7 no motiva la exclusión. En IQR, JNN mejora MSE pero empeora MAPE frente a ambos comparadores; se conservan discrepancias de calendario |

Las evaluaciones públicas resumidas contienen el dictamen, objetivo, modelo, hallazgos relevantes, límites y localizadores: [Framework](../_static/e2_literature_closure_final/original_reviews/framework_review.json), [Stacking](../_static/e2_literature_closure_final/original_reviews/primary_review.json), [CEEMDAN](../_static/e2_literature_closure_final/original_reviews/extraction.json) y [Jordan](../_static/e2_literature_closure_final/original_reviews/jordan_review.json). La matriz de diecisiete estudios enlaza las catorce fichas públicas anteriores y los tres nuevos estudios aplicados. Se diferencian lectura focal, lectura completa e inspecciones visuales. Las anotaciones privadas y los originales no se redistribuyen. Los resultados permanecen dentro de su fuente y escala; no se convierten en eficacia demostrada sobre E2.

Los tres estudios bursátiles aplicados de TimesNet, TSMixer y Transformers permanecen fuera del núcleo cripto y dentro del contexto de mecanismos. La guía permite transferencia desde otros dominios; esta distinción no los elimina de toda la revisión.

### Dos contextos con revisión cerrada e incertidumbre científica explícita

Zahid (`10.3390/risks10120237`) y CryptoMamba-SSM (`10.1109/OJCS.2026.3651226`) tienen **rol y acceso documental adjudicados**. No falta otra lectura ni un PDF idéntico. Se conservan como contexto metodológico porque sus etiquetas/escala no pueden reconstruirse inequívocamente. En Zahid persiste la alineación diario/intradía/horizonte; en CryptoMamba, la asignación de estimadores a tablas y la separación entre varianza de retornos e incertidumbre predictiva.

El [estado de entrega por fuente](../_static/e2_literature_closure_final/constructs/current_delivery_state.csv) registra `outstanding_access_task=false` y `outstanding_review_task=false`. Conforme al protocolo v2 se conserva `applied_eligibility=pendiente` como **estado de conocimiento científico**, no como tarea pendiente del estudiante. No cuentan entre las 17 inclusiones aplicadas ni aportan efectos. La [adjudicación](../_static/e2_literature_closure_final/constructs/adjudication.json) deja una eventual aclaración del autor como nueva evidencia opcional; no inventa una fórmula para cerrar casillas.

## Calidad, disponibilidad y brecha que justifica el benchmark

La matriz de las **nueve fuentes actuales** valora el reporte de cada condición seleccionada. El escalado ajustado sólo con entrenamiento está reportado en **2/9**, NR en 6/9 y NV en 1/9. Las repeticiones están reportadas en **4/9**, NR en 3/9, parcialmente en una y NV en otra. La igualdad del presupuesto HPO queda NR en 9/9. NR/NV no prueban fuga, cero ejecuciones ni desigualdad, y tampoco certifican la ausencia de esos problemas.

Una comprobación adicional de fuentes oficiales identifica publicación pública del dataset para **9/9 estudios** y repositorio de autor fijado para **9/9**. Wang se enlaza a UCR44 dentro del legado 2015; Inception al UCR85, y su nuevo escenario numérico a UCR2018/128 por separado. SAE se enlaza a UCI Beijing PM2.5; los seis estudios de pronóstico restantes a ETTh1. La disponibilidad identificada no verifica el snapshot, los datos preparados, sus splits o reproducción. Por ejemplo, UCI publica 43.824 observaciones y SAE informa 43.800 preparadas. [Evidencia y límites por fuente](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv).

La matriz focal conservada tiene otro denominador: **17 estudios incluidos narrativamente**, de los cuales **siete declaran conteos de ejecuciones o inicializaciones**, con alcances distintos. No se atribuye ausencia de repeticiones a los otros diez. La {download}`síntesis ampliada <../_static/e2_literature_closure_final/quality/synthesis17.md>` conecta observaciones concretas con decisiones de E2:

| Brecha observada en las fuentes examinadas | Respuesta verificable de E2 | Límite que permanece |
|---|---|---|
| Objetivos heterogéneos: varianza intradía, desviación a treinta días, volatilidad implícita y condicional | σ7 no anualizada, D7×5, h=1 y las mismas 608 fechas | No permite combinar métricas externas en un solo efecto |
| Folds, HPO, activos y nuevas inicializaciones se reportan con alcances distintos | Separar selección, seis semillas finales y evidencia temporal pareada | Potencia condicional del piloto ≈12,3 %, inferior al 80 % objetivo |
| Alcance del escalado/interpolación y esfuerzo de comparadores incompletamente establecido | Transformaciones de entrenamiento, calendario, embargo y artefactos conservados | El reporte de terceros no demuestra fuga; igual número de pruebas no garantiza igual coste |
| El ganador cambia con pérdida, target y versión | Comparadores simples, resultados adversos, Holm e inferencia temporal con 20.000 réplicas | Calendario heredado conocido y prueba formal de δg pendiente |

La afirmación de gap es refutable y acotada: **en las nueve fuentes que sustentan el catálogo no se identificó una evaluación conjunta de esos mecanismos para σ7 de Bitcoin, h=1 y siete días OHLCV, bajo un presupuesto común e inferencia temporal pareada**. Encontrar en cualquiera de esas nueve fuentes un experimento que cumpla ese contrato obliga a corregirla. La síntesis aplicada respalda la necesidad de comparar contratos, no una afirmación de ausencia global ni superioridad de E2. La contribución observada es un contraste retrospectivo auditable.

## Búsquedas recuperadas y límite del flujo global

**Las cadenas originales sí estaban guardadas en GitHub.** El [primer commit de literatura E2](https://github.com/jassan-bit/DeepLearning_202630/blob/f8fbbe6af3e91202f2eb6971d60102624ef38446/book_e2/sections/03_literatura.md) conserva WoS Q1/Q2 completas, los lotes de **302/500/500 registros** y Scopus **7.500 recibidos frente a 7.874 mostrados**. No es necesario pedir esas cadenas otra vez. El [ledger de consultas](../_static/e2_literature_closure_final/search_history/git_query_history_ledger.csv) enlaza consulta, commit, ruta y cobertura conocida; las cuatro consultas arXiv también se recuperaron.

El expediente inicial ya decía que los originales se mantenían **fuera de los archivos públicos**, que la correspondencia consulta–lote/rangos no estaba confirmada y que faltaban **374 registros de Scopus**. La recuperación de la historia y de 52 referencias remotas confirma ese antecedente; la hora de commit no se convierte en hora de ejecución de consulta. La {download}`cadena de evidencia histórica <../_static/e2_literature_closure_final/search_history/GIT_HISTORY.md>` y el inventario de paquetes distinguen lo encontrado de las copias privadas no recuperadas.

El corpus histórico mantiene **11.635 ocurrencias recibidas − 1.720 duplicados internos − 356 enlaces entre rondas = 9.559 grupos recibidos**. Los seis grupos exclusivamente dirigidos se conservan aparte. Los cuatro originales iniciales aportados se identificaron por sus hashes: Scopus de 7.500 registros y los WoS de 302, 500 y 500. También coincide el WoS enfocado de 340. Los WoS adicionales se usan como material enlazable por identidad; no se les atribuyen consultas históricas no verificadas.

La restauración obtiene **8.396 grupos con resumen auténtico**: se excluyeron del conteo de disponibilidad 20 celdas que sólo decían «sin resumen». A continuación se recuperaron 530 resúmenes arXiv y trece enfocados, disjuntos de la base: **8.939 grupos con cuerpo de resumen enlazado**. La lectura efectiva durante el cribado fue de 2.197 grupos, declarada separadamente. El [recibo de restauración](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_material_recovery_v1/input_receipt.json) y la [auditoría](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/audit.json) permiten comprobarlo.

El ZIP histórico aportado contiene derivados públicos y matrices; no incorpora los originales privados con resúmenes ni recupera por sí mismo las exportaciones enfocadas Scopus828/IEEE466. La recuperación pública posterior y los intentos sobre los 31 registros sin resumen disponible tienen disposiciones terminales dentro del flujo. La ausencia de un resumen o el bloqueo de un editor no se convierte en ausencia de DL.

La exportación aplicada de **1.104 EID únicos** mantiene su propio [recibo](../_static/e2_literature_delivery_closure_20261114_v3/identification/receipt.json) y [metadatos](../_static/e2_literature_delivery_closure_20261114_v3/identification/records.csv): 1.047 DOI distintos y 57 sin DOI. Sus coincidencias aportan material para los grupos históricos, conservando sus identificadores. Esta ronda no sustituye el universo recibido ni demuestra recuperar los 374 registros históricos ausentes.

**Queda cerrado el flujo del corpus recibido, con límites de cobertura y acceso explícitos.** No se afirma cobertura completa de las búsquedas nativas, ni elegibilidad conocida para los inaccesibles. Las incertidumbres científicas tras leer una fuente se mantienen como conocimiento no identificado, sin simular otra tarea resuelta o exigir repetir el mismo PDF. Los cortes focales anteriores y su [flujo histórico](../_static/e2_literature_closure_final/flow.json) conservan sus valores originales.

(auditoria-y-tarea-de-recuperacion-restante)=
(e2-lit-coverage-audit)=
## Auditoría y límites de cobertura

El [manifiesto del flujo recibido](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/manifest.json) liga los insumos y derivados por SHA. Los comandos verifican el denominador, la cobertura individual, los dos alias, la selección de versión para los 42 estudios, los doce campos de calidad, la separación de acceso/evidencia y la ausencia de pendientes operativos:

```bash
python scripts/audit_e2_prisma_material_recovery.py
python scripts/verify_e2_prisma_bundle.py references/literature_e2/prisma_received_v1
```

La recuperación de material privado es reproducible con `scripts/restore_e2_literature_materials.py`; la agregación de dictámenes usa `scripts/build_e2_prisma_bundle.py`. La clasificación semántica está registrada como revisión asistida por IA, no como resultado de reglas deterministas ni doble revisión humana. Los cuerpos de las fuentes no se distribuyen en el repositorio público.

Permanecen **374 registros Scopus mostrados pero no recibidos** y la correspondencia incompleta entre consultas, totales y rangos WoS. Son desviaciones de cobertura documentadas. Una recuperación posterior produciría otra versión del flujo; no se modifica el corte histórico para fingir exhaustividad. Tampoco cambian el alcance retrospectivo del calendario heredado, la prueba formal pendiente de δg o la limitación de potencia del benchmark. La recuperación bibliográfica no ejecutó nuevos ajustes de mercado ni alteró A2.

## Antecedentes conservados

Las descargas anteriores mantienen sus URL y bytes. Sus estados y escenarios corresponden al corte indicado en cada recibo y no sustituyen el estado vigente:

- {download}`citation_reconciliation_manifest.json · v2 <../_static/e2_literature_delivery_closure_20261114_v2/citation_score/citation_reconciliation_manifest.json>`
- {download}`conditional_score_by_scenario.csv · v2 <../_static/e2_literature_delivery_closure_20261114_v2/citation_score/conditional_score_by_scenario.csv>`
- {download}`preferred_intrapaper_scenario_rule.json · v2 <../_static/e2_literature_delivery_closure_20261114_v2/citation_score/contextual_p/preferred_intrapaper_scenario_rule.json>`
- {download}`cross_entregable_primary_links.csv · v2 <../_static/e2_literature_delivery_closure_20261114_v2/cross_entregable/cross_entregable_primary_links.csv>`
- {download}`applied_target_decisions.csv · v2 <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/applied_target_decisions.csv>`
- {download}`focused_synthesis.md · v2 <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/focused_synthesis.md>`
- {download}`historical43_source_roles.csv · v2 <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/historical43_source_roles.csv>`
- {download}`mechanism_source_decisions.csv · v2 <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/mechanism_source_decisions.csv>`
- {download}`manifest.json · v2 <../_static/e2_literature_delivery_closure_20261114_v2/manifest.json>`
- {download}`conditional_scores.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/conditional_scores.csv>`
- {download}`receipt.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/receipt.json>`
- {download}`records.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/records.csv>`
- {download}`scenario_protocol.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/citation_snapshot/scenario_protocol.json>`
- {download}`eligibility.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/eligibility.csv>`
- {download}`focal_flow.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/focal_flow.json>`
- {download}`matching_summary.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/identification/matching_summary.json>`
- {download}`receipt.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/identification/receipt.json>`
- {download}`records.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/identification/records.csv>`
- {download}`candidate.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/inception_single/candidate.json>`
- {download}`manifest.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/manifest.json>`
- {download}`pair_c_primary_assessments.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/pair_c_primary_assessments.json>`
- {download}`pair_d_adjudications.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/pair_d_adjudications.json>`
- {download}`two_constructs.md · v3 <../_static/e2_literature_delivery_closure_20261114_v3/primary_reviews/two_constructs.md>`
- {download}`current_nine_data_access.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv>`
- {download}`current_nine_quality.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_quality.csv>`
- {download}`model_study_bridge.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/model_study_bridge.csv>`
- {download}`quality_extraction_matrix_14.csv · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/quality_extraction_matrix_14.csv>`
- {download}`synthesis_updated.md · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/synthesis_updated.md>`
- {download}`verified_urls.json · v3 <../_static/e2_literature_delivery_closure_20261114_v3/quality/verified_urls.json>`

### Descargas históricas adicionales

- {download}`manifest.json · corte conservado <../_static/e2_literature_closure_final/manifest.json>`
- {download}`expected_private_source_inventory.json · corte conservado <../_static/e2_literature_closure_final/search_history/expected_private_source_inventory.json>`
