# Datos, entrenamiento e incertidumbre: evidencia vigente

Este directorio acompaña los capítulos 10–12 del Entregable 2. Reúne requisitos, fuentes y límites del benchmark congelado de 16 sistemas, 81 predictores y 608 fechas. La ruta metodológica es F2a, series de tiempo. El corte de fuente revisado es `5c78f32b763450df583693547e59784cc23ebe7a`.

## Archivos y alcance

| Archivo | Contenido |
| --- | --- |
| `guide_checklist.csv` / `.json` | 65 subcondiciones de la guía, con estado, evidencia, límites y acción pendiente. Los recuentos no son una calificación. |
| `decision_traceability.csv` | Doce vínculos entre decisiones, resultados y sus alcances. |
| `current_evidence_index.json` | Índice con tamaños y SHA-256 de las fuentes, operaciones e interpretación. |
| `targeted_data_guard_checks.json` / `.log` | Cinco pruebas unitarias existentes aprobadas con datos sintéticos y funciones simuladas. CPU, Python 3.12.14/Torch 2.8.0+cpu; ejecución posterior al benchmark. Seis archivos originales permanecen idénticos al commit científico. |
| `model_training_register.csv` | Registro unido de trece redes: ganador, hiperparámetros, épocas por fase, parámetros, recursos y seis RMSE individuales. Los 41 campos copian valores existentes; no recalculan resultados. |
| `model_training_register_provenance.json` | Fuentes y hashes del registro, unidades y exclusiones. Los clásicos mantienen sus métricas originales, sin recursos comparables inventados. |
| `border_cases.csv` | Tres extremos por modelo principal: 30 filas sobre nueve fechas distintas. Orden descendente del error absoluto medio entre semillas ya publicado, con desempate por fecha. |
| `border_case_readout.json` | Regla reproducible, hash de origen, límites y primer caso por modelo. No se trasladan pesos de muestreo a esta selección dirigida. |
| `*_before_update.md` | Copias exactas de los tres capítulos antes de la consolidación. Su texto y encabezados se conservan también en bloques documentales del libro. |
| `manifest.json` | Inventario de los productos de este directorio y sus hashes. |

## Interpretación

La separación de IDs y la causalidad temporal tienen evidencia concreta. La auditoría integral de fuga sigue parcial: no se atribuyen al pipeline final los controles históricos, ni se declaran ejecutados los experimentos adversarial y de etiquetas aleatorias. Un fallo real exige corregir y repetir el experimento afectado según la regla de la guía. Una comprobación pendiente no equivale a aprobarla ni a demostrar un fallo.

Las seis plazas HPO por red y los tres folds están cerrados, con fallos conservados. La cobertura mínima del espacio, su importancia, la varianza de selección y su coste completo mantienen límites. Tiempo final, latencia, throughput, FLOPs core y memoria de proceso tienen ámbitos diferentes. El análisis de coste por mejora con bootstrap y la equivalencia TOST/ROPE no están ejecutados; no se deduce equivalencia del no rechazo.

El proxy empírico de techo tiene un intervalo γ amplio y no identifica riesgo Bayes. La cobertura global no garantiza cobertura por grupos. Las categorías de errores son automáticas y posteriores; los extremos dirigidos ilustran errores, sin convertirse en revisión experta, confianza calibrada o atribución causal. La muestra original de 100 fechas/600 casos por principal permanece intacta; el control específico de atajos sigue pendiente.

## Operaciones de esta consolidación

Se ejecutaron cinco comprobaciones unitarias existentes con fixtures y mocks. No se escribieron tests nuevos ni se efectuaron nuevos ajustes de modelos reales, inferencias reales, scores predictivos, pruebas estadísticas, remuestreos, cálculos de potencia o instalaciones. La tabla A2 y los artefactos científicos originales conservan sus bytes. El registro y los casos extremos son una unión y una selección de valores ya guardados.
