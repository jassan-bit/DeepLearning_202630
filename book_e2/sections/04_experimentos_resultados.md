# Página de trabajo · Benchmark y resultados

**Todavía no se ha ejecutado el benchmark Bitcoin D7 del Top 10 y sus referencias.** La [sección 2](02_eda.md) sí contiene diagnósticos propios, incluida una referencia Ridge ajustada en desarrollo con forward chaining. Esa ejecución diagnóstica no pertenece al benchmark definitivo; Ridge se propone como referencia no profunda, sin modificar los tres modelos base heredados.

La sección 4 oficial ya se desarrolla en el [capítulo de implementación](04_implementacion.md), con un diseño propuesto y auditoría documental. Esta página conserva el espacio para resultados del benchmark Bitcoin D7. La [sección 5](05_paper_guia.md) ya presenta la réplica TimeMixer y el piloto externo de tres modelos en ETTh1, con cinco y quince ajustes respectivamente; esos veinte ajustes no son entrenamientos nuevos de Bitcoin. Las métricas de E1 siguen identificadas como antecedentes.

| Evidencia de E2 | Estado |
|---|---|
| Diez modelos y referencias | Propuesta en sección 3; implementación y diseño pendientes |
| Piloto y dimensionamiento Bitcoin D7 | Sin ejecutar; el piloto ETTh1 tiene otro objetivo y calendario |
| Ajuste de hiperparámetros Bitcoin D7 | Sin ejecutar |
| Comparación definitiva Bitcoin D7 | Sin ejecutar |
| Tablas e interpretación Bitcoin D7 | Se elaborarán con resultados reales |
| Réplica TimeMixer ETTh1 | Cinco ajustes ejecutados; criterio previo, resultados y discrepancias en §5 |
| Piloto externo ETTh1 | Quince ajustes ejecutados en TimeMixer, iTransformer y PatchTST; no dimensiona el benchmark Bitcoin |

Cuando exista un protocolo cerrado se conservarán todas las corridas previstas, incidencias y exclusiones justificadas. La comparación reportará incertidumbre, tamaños de efecto, coste y casos no definidos, sin reemplazar semillas desfavorables o escoger el test con un resultado favorable.

Una ausencia de significación no se interpretará como equivalencia. Los resultados adversos o inconclusos también formarán parte del informe.
