# Archivo de trabajo anterior · Benchmark y resultados

**Archivo histórico anterior a la auditoría de la guía completa (06/11/2026).** Las cifras de quince configuraciones y las remisiones entonces pendientes reflejan esa propuesta inicial, hoy sustituida. Consultar la [sección vigente](09_protocolo_registrado.md) y el [protocolo actual](09_protocolo_registrado.md). El texto siguiente se conserva como trazabilidad; no es un segundo protocolo activo.

**Todavía no se ha ejecutado el benchmark Bitcoin D7 del Top 10 y sus referencias.** La [sección 2](02_eda.md) sí contiene diagnósticos propios, incluida una referencia Ridge ajustada en desarrollo con forward chaining. Esa ejecución diagnóstica no pertenece al benchmark definitivo; Ridge se propone como referencia no profunda, sin modificar los tres modelos base heredados.

La sección 4 oficial ya se desarrolla en el [capítulo de implementación](04_implementacion.md), con un diseño propuesto y auditoría documental. La [sección 6](06_benchmark_final.md) reúne ahora el plan de comparación y sus plantillas; esta página de trabajo conserva un resumen del estado del benchmark Bitcoin D7. La [sección 5](05_paper_guia.md) ya presenta la réplica TimeMixer y el piloto externo de tres modelos en ETTh1, con cinco y quince ajustes respectivamente; esos veinte ajustes no son entrenamientos nuevos de Bitcoin. Las métricas de E1 siguen identificadas como antecedentes.

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
