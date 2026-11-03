# Auditoría de búsqueda y recuperación dirigida — 03/11/2026

Se conservaron las consultas del 02/11/2026 y se corrigió su cribado de metadatos. El 03/11 se inspeccionaron seis repositorios adicionales ya presentes entre los 106. Se recuperaron y evaluaron dos textos completos, incluidos como S6/S7 solo en el corpus exploratorio ampliado. No se ejecutaron nuevas cadenas, modelos ni datos propios de mercado/test.

| ID | Requisito | Estado | Evidencia | Límite o corrección |
|---|---|---|---|---|
| L01 | Cadena completa, sinónimos y variantes | Parcial | Cadena general y diez consultas GitHub preservadas. Falta ejecutar cadenas adaptadas a cada base, con filtros y exportaciones. | No se presenta la cadena académica propuesta como consulta realizada. |
| L02 | Fecha y número de resultados por cadena | Corregible con registros locales | Agregados de diez consultas recalculados desde las dieciocho entradas del log. | Estimaciones de índice no se suman ni se confunden con registros recuperados. |
| L03 | WoS, Scopus, IEEE Xplore y arXiv complementario | Bloqueado/no acreditado | Los comprobantes existentes registran consultas no ejecutadas y conteos null. | Un enlace, autenticación prevista o código de autores no equivale a buscar la base. |
| L04 | Periodo preferente de cinco a ocho años | Parcial | 2019–2026 es ventana de elegibilidad declarada. GitHub no tiene filtro verificable de año de publicación en las consultas. | Distinguir fecha de publicación científica de fecha de creación/actualización de repositorio; fundamentos clásicos separados. |
| L05 | Inclusión y exclusión a nivel de estudio | Parcial | Volatilidad de Bitcoin, DL comparativo, métricas verificables, identidad y protocolo identificables. S6/S7 se incluyen exploratoriamente bajo la misma política gris que S2; no se excluyen por su género documental. | Cinco textos completos incluidos, tres grises; no se acredita Q1 ni selección final de tres modelos básicos. |
| L06 | Cribado y exclusiones justificadas | Corregido y ampliado con recuperación dirigida | 106 registros;20 repositorios inspeccionados;7 textos completos evaluados;5 incluidos. Se preservaron decisiones previas y se corrigieron motivos de los92 no inspeccionados inicialmente. | Once candidatos de metadatos continúan sin inspección. Cuatro nuevos repositorios inspeccionados no aportaron documento completo identificable; no añaden frecuencias. |
| L07 | Flujo reproducible y deduplicación | Documentado con límite | 118−12=106;20+86=106;7=5+2; corte previo14+92 y5=3+2 conservado. | Se deduplican repositorios, no queda acreditada deduplicación sistemática por DOI/título/autores entre bases no consultadas. |
| L08 | Quién seleccionó y resolvió desacuerdos | Parcial | Cribado asistido por Codex para Jassan Arteta; no se acredita segundo revisor humano. | No inventar consenso o kappa. Dudas y cambios deben quedar en un registro con fecha, motivo y decisión responsable. |
| L09 | Calidad experimental por estudio | Documentado por dimensiones, con límites | Los CSV y JSON de calidad cubren datos, partición, arquitectura, repeticiones, inferencia y reproducibilidad. | No sumar puntuación arbitraria; ausencia de descripción no acredita ausencia de fuga. Veinte corridas de S1 no equivalen a mercados independientes. |
| L10 | Literatura reciente y preferenciaQ1 | No acreditado globalmente | Cinco textos completos aplicados; S2/S6/S7 son grises; S6 sin fecha de publicación acreditada y año editorial de S5 pendiente de corroboración. | Para reclamarQ1 registrar índice, revista, año, categoría y fuente. No equiparar congreso,repo,DOI y cuartil de revista. |

## Flujo acumulado

118 ocurrencias−12 repetidas=106 repositorios. Inspección acumulada:14 previos+6 nuevos=20;86 sin inspección. Documentos:5 previos+2 nuevos=7 evaluados;5 incluidos+2 excluidos. No son20 artículos ni cinco modelos. El corte previo está en history_20261102 y el corte entre corrección local y recuperación en history_20261103_metadata_revision.

## Política de literatura gris

S6 es un borrador de cinco autores sin año de publicación acreditado; una fecha de commit no es año editorial. S7 es un informe académico fechado 02/06/2026. Ambos presentan riesgos severos y se incluyen con los mismos requisitos básicos de identidad, objetivo, comparación, protocolo y resultados usados para el corpus exploratorio anterior. No se excluyen por ser borrador o informe, ni se presentan como artículos indexados/Q1/revisados por pares. Se exige sensibilidad sin estos dos y sin documentos de fecha no acreditada. S3/S4 siguen como contexto sin texto completo recuperado.

## Evidencia y límites

El cribado actual y los registros de revisiones conservan fechas, motivos anteriores/actuales, commits y hashes. El manifiesto de recuperación dirigida registra URLs y hashes de autores; no se redistribuyen aquí sus PDFs. Las 81filas de métricas son transcripciones de autores, no resultados propios; los 36valores nuevos de Tabla4 S6 se mantienen tal como se imprimieron, incluidas las inconsistencias, y S7 aporta tres valores de su LSTM. No se recalcularon resultados.

WoS, Scopus, IEEE Xplore y arXiv siguen sin consulta acreditada en los comprobantes: conteo no disponible, no cero. No se acredita segundo revisor humano, consenso previo o cumplimiento total de la búsqueda/selección.
