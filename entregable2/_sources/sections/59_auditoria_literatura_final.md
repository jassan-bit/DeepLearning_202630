# 59. Auditoría final de la literatura recibida

**Actualización del 14/11/2026, Bogotá:** La actualización focal adjudica 30 objetivos, reutiliza 43 recibos primarios deE 1 con solapamientos explícitos y amplía la sensibilidad contextual a P de 9 modelos. El puntaje anterior permanece intacto; el PRISMA global y cuatro C siguen parciales. [Detalle](62_cierre_literatura_focal.md).

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
El flujo anterior cubre el material recibido; **el PRISMA global sigue incompleto**. Los 374 registros Scopus no recibidos son una carencia de cobertura, no exclusiones. El universo global de WoS y la correspondencia completa entre consultas y exportaciones tampoco están acreditados. Los conteos globales de elegibles, incluidos y excluidos finales permanecen desconocidos.
```

## Qué se cribó y qué se evaluó

Los 7.833 grupos de la primera ronda tienen recomendaciones automáticas, con [motivos y conteos separados](../_static/e2_literature_final_audit_20261114/screening_reasons_not_final_exclusions.csv). No se presentan como exclusiones humanas validadas.

Las **396 lecturas individuales asistidas por IA** de los recibos anteriores corresponden a la ronda enfocada de 1.078 grupos: 59 prioridades y las 337 exclusiones propuestas. No acreditan lectura individual de los 9.559 grupos. Las reglas de conservación como antecedente secundario tampoco quedaron armonizadas entre todas las rondas. No hubo doble revisión humana independiente y no se calcula kappa.

Dentro de los 40 candidatos de recuperación dirigida, se buscaron 22: seis textos completos editoriales, dos preprints con equivalencia editorial pendiente y 14 recuperaciones parciales o fallidas; 18 candidatos no se buscaron. Texto recuperado no significa elegibilidad final.

La nueva [adjudicación de roles por fuente](../_static/e2_literature_final_audit_20261114/bounded_source_role_adjudication.csv) concilia 20 fichas ya documentadas: nueve fuentes actuales, ocho antecedentes aplicados y tres lecturas focales arXiv. Cada ficha contiene objetivo, versión, localizador, decisión y limitación. Se conserva como pendiente lo que depende de resolver una definición de volatilidad o una equivalencia entre versiones. FCN y ResNet son dos sistemas de una misma fuente; Wang (2016) constituye una excepción fundacional explícita al periodo preferido.

Los mecanismos pueden conservarse por su relevancia estructural sin atribuirles eficacia directa sobre Bitcoin. La clasificación UCR, PM2.5, ETTh1, volatilidad implícita, varianza realizada intradía y otras ventanas de volatilidad no se convierten en métricas de $\sigma_7$.

## Puntajes: información observada y sensibilidad contextual

El puntaje de apoyo previamente documentado conserva **36 de 50 componentes observados**: C en seis sistemas, y U, A y F en los diez. C sigue desconocido para FCN, ResNet, Inception y SAE; son tres fuentes porque Wang se comparte. Las nueve solicitudes oficiales OpenAlex para un nuevo corte coherente resultaron bloqueadas por el proxy. Una cifra editorial aislada de otra fuente no se mezcla con las citas Scopus del corte recibido.

P permanece desconocido bajo el criterio local anterior de eficacia directamente comparable con el contrato E2. La guía §3.2 pide desempeño en **datasets comparables**, sin exigir que ese dataset sea Bitcoin: por ello se añade una sensibilidad contextual nueva, separada del puntaje anterior.

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

El catálogo fue autorizado el 12/11 y los pesos de apoyo se documentaron el 13/11. Esta actualización **no modifica esa cronología ni usa A2**. El [puntaje anterior auditado](../_static/e2_literature_final_audit_20261114/current10_score_component_audit.csv) permanece separado; no se imputan componentes desconocidos como cero.

## Fuentes excluidas del catálogo y alcance del gap

La [tabla de originales históricos fuera del catálogo](../_static/e2_literature_final_audit_20261114/excluded_named_originals_scope_audit.csv) conserva Informer, Autoformer, Crossformer y S-Mamba como contexto. Crossformer y S-Mamba tienen 1.675 y 313 citas, respectivamente, en el corte recibido; no se rellenan las citas ausentes de los otros dos. La explicación retrospectiva es el alcance autorizado de diez sistemas con cuatro familias, no inferioridad experimental ni dominancia por puntaje. La cobertura disponible no acredita que esta lista contenga todos los modelos más citados excluidos de la literatura global.

El gap defendible sigue siendo **acotado a las fuentes examinadas**: ninguna de las nueve fuentes primarias actuales documenta la combinación exacta de objetivo, información D7×5, horizonte y comparación temporal de E2. Esto justifica contrastar la transferencia; no prueba ausencia global de estudios ni superioridad de un modelo.

La revisión global y una selección del Top 10 completamente sustentada por todos los componentes siguen siendo limitaciones reales frente a §§3.1–3.2. Lo resuelto en esta actualización es la contabilidad del material recibido, la procedencia de las decisiones y la exposición reproducible de la información que falta.

## Artefactos y reproducción

[Flujo recibido](../_static/e2_literature_final_audit_20261114/flow_received_corpus.json) · [Criterios y cobertura](../_static/e2_literature_final_audit_20261114/coverage_and_criteria.json) · [Manifest de entradas](../_static/e2_literature_final_audit_20261114/source_manifest.json) · [Manifest de salidas](../_static/e2_literature_final_audit_20261114/manifest.json).

Desde el repositorio: `python scripts/e2_literature_final_audit.py --check`. La auditoría usa metadatos normalizados y recibos documentales, sin entrenamiento, inferencia ni acceso a métricas A2. Once pruebas verifican deduplicación conservadora, contabilidad, procedencia y preservación de incógnitas.
