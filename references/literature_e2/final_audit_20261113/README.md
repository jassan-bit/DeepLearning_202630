# Conciliación final de literatura recibida

Auditoría retrospectiva del 2026-11-14. Concilia las rondas recibidas y las lecturas
ya documentadas; no declara una nueva búsqueda, una nueva lectura de todos los
artículos ni revisión independiente humana. Los artefactos del 05–13/11 permanecen
intactos. No se usan métricas del test A2 para seleccionar literatura.

## Flujo y denominadores

Los tres corpus contienen **11,635 ocurrencias recibidas**.
Se conservan 1,720 duplicados resueltos dentro de sus
rondas y se detectan 356 enlaces
adicionales entre rondas: **9,559 grupos bibliográficos**
recibidos. Estos grupos no equivalen a estudios elegibles. Los 19 recibos dirigidos
(9 fuentes actuales y10 históricas, con solapamiento) se contabilizan por separado:
aportan 6 grupos adicionales.
La identidad contable es `11635-1720-356=9559`.

La deduplicación enlaza DOI exacto normalizado, identificador arXiv sin versión o
título exacto normalizado y año cuando falta DOI. Dos DOI distintos nunca se
fusionan sólo por el título. El registro de enlaces y los conflictos preservan
cada decisión; no se deduce equivalencia entre versiones editoriales y preprints.

**No hay PRISMA global cerrado.** Faltan374 registros del total Scopus mostrado en
la captura; no son exclusiones. Las fechas/cadenas/totales WoS globales y la
correspondencia de todos los ficheros con cada consulta no están acreditadas.
Los totales nativos posteriores106/78/3 no se suman sin unión verificable.

La ronda del05/11 tiene7833 recomendaciones automáticas, no exclusiones humanas.
Las396 lecturas individuales AI agregadas (59prioridades y337 exclusiones
propuestas) pertenecen al corpus enfocado de1078 registros del08/11. No acreditan
revisión individual del corpus completo. La clasificación de secundarios entre
rondas tampoco estaba armonizada. Se mantienen `null` los elegibles/incluidos y
las exclusiones globales finales; no se informa kappa sin revisores independientes.

De los40 candidatos de recuperación dirigida,22 fueron buscados:6 textos completos
editoriales,2 preprints con equivalencia editorial pendiente y14 parciales/no
recuperados;18 no buscados. Disponibilidad de texto no equivale a inclusión final.

## Decisiones comprobables sobre fuentes

El registro de20 fichas concilia9 fuentes actuales,8 antecedentes aplicados y3
lecturas focales arXiv. Es una adjudicación retrospectiva de su **rol** apoyada en
las lecturas previas y sus localizadores; no transforma los8 antecedentes ni los
3 preprints en estudios globalmente incluidos. FCN yResNet comparten una fuente;
Wang2016 se conserva como excepción fundacional justificada. Las métricas de
clasificación, PM2.5, ETTh1 y otras definiciones de volatilidad se mantienen fuera
de una comparación numérica directa con sigma7. Las discrepancias de objetivo o
versión que no están resueltas siguen pendientes.

## Puntaje actual y límite de identificación

Se auditan los50 componentes C/U/P/A/F de10 sistemas. Hay
**36/50 observados**: C en6 sistemas y U/A/F en10;
P no es comparable en ninguno. Los4 C ausentes corresponden a3 fuentes (Wang se
comparte). Se conserva el corte Scopus recibido sin mezclar contadores de otras
fechas/proveedores. El archivo de componentes incluye los límites algebraicos,
las tareas/datasets de las fuentes y el motivo de cada P no identificado; no
imputa cero ni usa el punto medio como clasificación.

El catálogo fue autorizado el12/11; los pesos de apoyo son del13/11. Esta auditoría
**no convierte ese catálogo en una selección bibliográfica previa por puntaje**.
Completar C exige un corte coherente verificable; completar P requiere evidencia
en datasets comparables o una modificación explícita y retrospectiva del criterio,
que no sería equivalente al criterio original ni justificaría retrospectivamente
la selección. La revisión global y el puntaje plenamente observado permanecen
como limitaciones reales de cumplimiento de la guía 3.1–3.2.

La guía 3.2 pide datasets comparables y no exige que P proceda de sigma7. Por eso
se añade una **sensibilidad contextual nueva y retrospectiva**, separada del
puntaje previo: seis fuentes reportan MSE/MAE de ETTh1 a horizonte96. Se muestran
sus lookbacks96/512/720, los splits y la cobertura del escalado. Dos alternativas
normalizan descriptivamente el MSE dentro de ese grupo nominal, o dentro de cada
lookback. No se mezclan clasificación ni PM2.5 con ETTh1. La fila única L720 queda
sin P en el segundo escenario. Estas alternativas observan P en6 o5 sistemas,
pero no solucionan splits/escalas/esfuerzo desiguales: son sensibilidad, no una
comparación controlada ni una selección previa justificada retrospectivamente.
El protocolo y las20 filas se publican sin sustituir P original.

Reproducción sin red: `python scripts/e2_literature_final_audit.py --check`.
Los hashes de todas las entradas normalizadas están en `source_manifest.json`.
