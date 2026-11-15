# Agregación estadística, NAS no aplicable y entrega vigente

Este directorio acompaña los capítulos 13 y 14, la entrega consolidada y los Apéndices A/B/C. Corte de fuente revisado: `2769b25574f4406bf8afcc44778a2a20f04977e5`. Se conservan el benchmark de 16 sistemas/81 predictores/608 fechas, sus resultados originales y el carácter retrospectivo de una única serie.

## Contenido

| Archivo | Alcance |
| --- | --- |
| `guide_checklist.csv` / `.json` | 59 subcondiciones, con evidencia, estado, límite y acción; sus recuentos no constituyen una nota. |
| `decision_traceability.csv` | Vínculos entre evidencia, decisiones e interpretación vigente. |
| `appendix_B_current.csv` / `.json` | Los 21 literales de la guía: nueve completos dentro de su alcance, diez parciales y dos no aplicables. B14/B15 corresponden a NAS. Las versiones anteriores conservan sus bytes. |
| `a3_primary_contrasts_readout.csv` / `.json` | Unión estática de veinte contrastes y 41 campos: refinamiento bootstrap B=20.000, original B=4.999 y diagnósticos t/cambios de signo. No hay nuevos tests, IC o efectos. Los tamaños estandarizados primarios y sus IC ausentes permanecen vacíos. |
| `rubric_and_deliverables.json` | Doce entregables, seis pesos literales 15/15/25/20/15/10 y cuatro descriptores de desempeño. No calcula una calificación ni redistribuye puntos por NAS no aplicable. |
| `current_evidence_index.json` | Fuentes con tamaño y SHA-256, unidades, decisiones y límites. |
| `*_before_update.md` | Copias exactas de capítulos 13/14 y documentos 47/61 antes de esta consolidación. Sus textos, encabezados y descargas siguen en los bloques documentales del libro. |
| `manifest.json` | Inventario y hashes de los productos de este directorio. |

## Decisiones y límites

La ruta metodológica es F2a, series de tiempo. La primaria promedia seis RMSE individuales por red, condicionadas a una cronología común. Ni las semillas ni las fechas correlacionadas constituyen mercados independientes. Se separan familias y estimandos: ganancia práctica RMSE, MSE diaria, rangos condicionados y superioridad diaria descriptiva. Una fracción de fechas favorables o de réplicas bootstrap no se declara probabilidad de ganar en un nuevo mercado. El modelo mixto y la descomposición actual de varianza no están ejecutados.

**La búsqueda de 1.000 arquitecturas y dos fases NAS no aplica al catálogo actual de trece cuerpos definidos, incluida TiDE R0-v2.** No es una campaña pendiente. Se mantienen fuentes, adaptaciones, HPO, costes y rigor como requisitos propios, sin llamar NAS a seis configuraciones de entrenamiento.

El Apéndice B reconoce la incertidumbre cuantificada y cobertura empírica del Top 3 en el alcance literal de B17. Conserva infracobertura y límites temporales sin inventar una garantía exacta como condición de ese literal. Los controles de fuga pendientes, el margen δg no acreditado formalmente, la potencia insuficiente, el coste completo, los efectos con IC ausentes y la falta de equivalencia siguen visibles.

## Operaciones

Esta consolidación transcribe requisitos y une valores existentes. No ejecuta nuevos ajustes, inferencias reales, scores, hipótesis estadísticas, remuestreos, cálculos de potencia o pruebas unitarias. Las cinco guardas sintéticas de la consolidación anterior conservan su recibo; no se atribuyen como nueva ejecución. A2, pesos, calibradores, muestras y archivos científicos originales permanecen intactos. El auditor de entrega verifica 195 archivos y reconstruye A2; no realiza un replay científico completo.
