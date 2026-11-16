# Auditoría final de la literatura recibida

> **Archivo de antecedentes:** documento **§59** del registro anterior. Conserva su fecha, alcance y resultados históricos. [Ver organización del informe](concordancia_guia.md).

**Estado vigente en [§62](62_cierre_literatura_focal.md):** el flujo del corpus recibido está reconciliado: **9.559 grupos cribados y 42 estudios aplicados únicos**, con doce dimensiones de extracción y disposiciones de acceso explícitas. Los puntajes de los diez sistemas son calculables dentro de sus escenarios; la comparabilidad global permanece condicionada. **Esta página conserva el corte histórico y sus derivados**; sus estados parciales se interpretan en ese corte, sin sustituir el registro vigente.

La conciliación del **14/11/2026 UTC (13/11 de Bogotá)** resuelve el solapamiento entre los corpus recibidos y hace explícitos los límites del cribado y de los puntajes. Es una auditoría retrospectiva: no constituye un nuevo prerregistro ni convierte los resultados del benchmark en criterios de selección bibliográfica. Las búsquedas, versiones y lecturas previas se conservan.

## Flujo con denominadores verificables

| Corpus recibido | Ocurrencias | Duplicados dentro de la ronda | Grupos de la ronda |
|---|---:|---:|---:|
| Scopus y WoS, 05/11 | 8.802 | 969 | 7.833 |
| Scopus, WoS e IEEE, 08/11 | 1.634 | 556 | 1.078 |
| arXiv complementario, 08/11 | 1.199 | 195 | 1.004 |
| **Total antes de conciliar rondas** | **11.635** | **1.720** | **9.915** |

La conciliación detecta **356 duplicados adicionales entre rondas** y deja **9.559 grupos bibliográficos recibidos**:

$$
11\,635-1\,720-356=9\,559.
$$

Se enlazan DOI exactos normalizados, identificadores arXiv sin versión y títulos exactos normalizados con el mismo año cuando falta DOI. Tres conjuntos con títulos coincidentes y DOI diferentes se conservan separados. El [registro de enlaces](../_static/e2_literature_final_audit_20261114/cross_round_link_ledger.csv), los [conflictos](../_static/e2_literature_final_audit_20261114/title_year_identifier_conflicts.json) y las [filas con procedencia](../_static/e2_literature_final_audit_20261114/source_nodes.csv) permiten revisar estas decisiones.

Los 19 recibos de fuentes dirigidas —nueve estudios actuales y diez históricos— se contabilizan aparte y añaden seis grupos que no estaban enlazados al corpus recibido. Un grupo bibliográfico no equivale automáticamente a un estudio independiente ni elegible. Tampoco se suman los resultados posteriores de tres consultas nativas (106/78/3) sin una unión verificable de sus registros.

```{warning}
El **flujo del corpus recibido está reconciliado**: 9.559 grupos cribados, 42 estudios aplicados únicos, 174 registros cuyo informe no se recuperó y cero pendientes operativos. Su [flujo vigente](../_static/e2_prisma_received_v1/received_flow.json) y [auditoría](../_static/e2_prisma_received_v1/audit.json) completan esta conciliación inicial sin reescribir sus artefactos. La cobertura global de las bases mantiene límites: los 374 Scopus no recibidos no son exclusiones y no se conocen todos los totales/rangos WoS. El cierre del material recibido no acredita exhaustividad ni elegibilidad conocida para los inaccesibles. El [cierre posterior de alcance (§62)](62_cierre_literatura_focal.md) fija los 7.500 registros Scopus originales y finaliza la recuperación adicional; los límites anteriores se conservan como antecedentes.
```

## Qué se cribó y qué se evaluó

**Estado actual:** cada uno de los 9.559 grupos tiene disposición final y nivel de evidencia; se registran 120 evaluaciones primarias focales o reutilizadas y 42 estudios aplicados únicos. La [matriz aplicada de doce dimensiones](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/references/literature_e2/prisma_received_v1/quality_applied_studies.csv) contiene 504 filas. No hubo doble revisión humana independiente ni se calcula κ. Los dos informes cuyo papel/profundidad DL no pudo determinarse conservan incertidumbre científica, no una tarea operativa de lectura sin realizar.

**Corte inicial conservado:** los 7.833 grupos de la primera ronda tuvieron recomendaciones automáticas, con [motivos y conteos separados](../_static/e2_literature_final_audit_20261114/screening_reasons_not_final_exclusions.csv). No se presentan esas recomendaciones como exclusiones humanas validadas ni como las adjudicaciones finales posteriores.

Las **396 lecturas individuales asistidas por IA** del corte inicial correspondían a la ronda enfocada de 1.078 grupos: 59 prioridades y 337 exclusiones propuestas. Ese contador histórico no acredita lectura individual del corpus completo ni sustituye la cobertura del registro posterior. La armonización y adjudicación actual se documentan en §62; las lecturas asistidas no se convierten en doble revisión humana.

En el corte dirigido inicial de 40 candidatos se habían buscado 22: seis textos editoriales completos, dos preprints y 14 accesos parciales/fallidos; entonces faltaba buscar 18. Ese estado histórico se conserva en sus recibos, pero no describe tareas actuales: el flujo recibido ya registra intentos y disposiciones finales. Un texto recuperado tampoco se equipara automáticamente a elegibilidad.

La [adjudicación inicial de roles](../_static/e2_literature_final_audit_20261114/bounded_source_role_adjudication.csv) conserva 20 fichas: nueve fuentes del catálogo, ocho antecedentes aplicados y tres lecturas focales arXiv. Cada ficha contiene objetivo, versión, localizador, decisión y limitación. La adjudicación posterior de [§62](62_cierre_literatura_focal.md) registra la recepción de cuatro originales, cierra sus revisiones y distingue dos incertidumbres de constructo de una falta de acceso o lectura. FCN y ResNet son dos sistemas de una misma fuente; Wang (2016) conserva una excepción fundacional explícita al periodo preferido.

Los mecanismos pueden conservarse por su relevancia estructural sin atribuirles eficacia directa sobre Bitcoin. La clasificación UCR, PM2.5, ETTh1, volatilidad implícita, varianza realizada intradía y otras ventanas de volatilidad no se convierten en métricas de $\sigma_7$.

## Puntajes: información observada y sensibilidad contextual

**Estado actual:** [§62](62_cierre_literatura_focal.md) publica una exportación uniforme Scopus para las nueve fuentes y diez sistemas, con escenarios aritméticos completos C/U/P/A/F. La [tabla estricta de selección](../_static/e2_literature_closure_final/selection/tabla_principal.csv) mantiene P global no identificado y compara por estratos; los diez modelos admiten rangos 1–10, sin dominancia robusta. Completar la aritmética local no calibra sus tareas o métricas entre sí.

**Corte inicial conservado:** el [puntaje auditado entonces](../_static/e2_literature_final_audit_20261114/current10_score_component_audit.csv) tenía 36 de 50 componentes observados: C en seis sistemas y U/A/F en diez; C desconocido para FCN, ResNet, Inception y SAE. Las solicitudes OpenAlex fallidas pertenecen a ese intento, no invalidan la exportación Scopus posterior. No se mezclan cifras editoriales aisladas ni se reescriben los valores anteriores.

P permanecía desconocido bajo el criterio inicial de eficacia directamente comparable con E2. La guía §3.2 pide desempeño en **datasets comparables**, sin exigir que sea Bitcoin: la sensibilidad contextual del corte siguiente se conserva debajo. La actualización estricta de §62 separa tarea, dataset, horizonte, lookback, métrica y fuente/tabla, sin tratar nombres nominalmente iguales como experimentos controlados idénticos.

| Modelo | Dataset y horizonte del resultado fuente | Lookback | MSE | MAE |
|---|---|---:|---:|---:|
| TimesNet | ETTh1, H=96 | 96 | 0,384 | 0,402 |
| TSMixer | ETTh1, H=96 | 512 | 0,361 | 0,392 |
| TiDE | ETTh1, H=96 | 720 | 0,375 | 0,398 |
| TimeMixer | ETTh1, H=96 | 96 | 0,375 | 0,400 |
| PatchTST | ETTh1, H=96 | 512 | 0,370 | 0,400 |
| iTransformer | ETTh1, H=96 | 96 | 0,386 | 0,405 |

Estas cifras pertenecen a sus artículos, no a los experimentos E2. La [matriz de comparabilidad](../_static/e2_literature_final_audit_20261114/contextual_performance_comparability.csv) conserva sus localizadores y límites: TiDE declara un split 7:1:2, TSMixer declara 12/4/4 meses para ETT, PatchTST no especifica el split en el PDF revisado y varias fuentes no aclaran por completo la escala y el ajuste del escalador. Compartir ETTh1 y horizonte no demuestra equivalencia de protocolo.

La alternativa descriptiva normaliza, dentro de cada estrato:

$$
P_{\mathrm{contexto},i}=\frac{\max_j\mathrm{MSE}_j-\mathrm{MSE}_i}
{\max_j\mathrm{MSE}_j-\min_j\mathrm{MSE}_j}.
$$

Se publican dos escenarios: seis fuentes nominales ETTh1/H96, y estratos adicionales por lookback. El primero aporta P descriptivo para seis sistemas; el segundo para cinco, pues el estrato único de TiDE no permite normalización. No se mezclan clasificación ni PM2.5 en esos estratos. La [tabla completa de sensibilidad](../_static/e2_literature_final_audit_20261114/posthoc_contextual_score_sensitivity.csv) conserva valores desconocidos, límites algebraicos y las diferencias de protocolo. El [protocolo](../_static/e2_literature_final_audit_20261114/posthoc_contextual_score_protocol.json) advierte que es una sensibilidad retrospectiva, no una comparación controlada ni un ranking que haya determinado el catálogo.

El catálogo fue autorizado el 12/11 y los pesos de apoyo se documentaron el 13/11; esa cronología se conserva. La guía §3.2 no exige que el puntaje preceda al benchmark ni un preregistro bibliográfico. Se evita atribuir al cálculo posterior haber causado la elección previa y no se usa A2 como resultado de los artículos. El [puntaje anterior auditado](../_static/e2_literature_final_audit_20261114/current10_score_component_audit.csv) permanece separado, sin imputar sus componentes desconocidos como cero.

## Fuentes excluidas del catálogo y alcance del gap

La [tabla de originales históricos fuera del catálogo](../_static/e2_literature_final_audit_20261114/excluded_named_originals_scope_audit.csv) conserva Informer, Autoformer, Crossformer y S-Mamba como contexto. Crossformer y S-Mamba tienen 1.675 y 313 citas, respectivamente, en el corte recibido; no se rellenan las citas ausentes de los otros dos. La explicación retrospectiva es el alcance autorizado de diez sistemas con cuatro familias, no inferioridad experimental ni dominancia por puntaje. La cobertura disponible no acredita que esta lista contenga todos los modelos más citados excluidos de la literatura global.

El gap defendible sigue siendo **acotado a las fuentes examinadas**: ninguna de las nueve fuentes primarias actuales documenta la combinación exacta de objetivo, información D7×5, horizonte y comparación temporal de E2. Esto justifica contrastar la transferencia; no prueba ausencia global de estudios ni superioridad de un modelo.

El estado vigente resuelve la contabilidad y disposición del corpus recibido, las citas uniformes y los escenarios calculables, y publica calidad y disponibilidad para las nueve fuentes. Datos y código públicos están identificados en 9/9 estudios, sin acreditar sus snapshots ni reproducción. Permanecen límites de cobertura de las bases y de comparabilidad global: la selección se defiende por mecanismos y diversidad con análisis estratificado, no como Top 10 óptimo global. La [lista del punto 3](../_static/e2_literature_current_v1/guide_checklist.csv) enlaza el estado de cada requisito con la evidencia actual.

## Artefactos y reproducción

[Flujo recibido](../_static/e2_literature_final_audit_20261114/flow_received_corpus.json) · [Criterios y cobertura](../_static/e2_literature_final_audit_20261114/coverage_and_criteria.json) · [Manifest de entradas](../_static/e2_literature_final_audit_20261114/source_manifest.json) · [Manifest de salidas](../_static/e2_literature_final_audit_20261114/manifest.json).

Desde el repositorio: `python scripts/e2_literature_final_audit.py --check` reconstruye **este corte histórico** a partir de metadatos y recibos, con once pruebas de deduplicación, contabilidad y preservación de incógnitas. La auditoría actual del corpus recibido se documenta en {ref}`§62 <e2-lit-coverage-audit>`. Ninguno de esos procedimientos entrena, ejecuta inferencia ni utiliza A2 como evidencia bibliográfica.
