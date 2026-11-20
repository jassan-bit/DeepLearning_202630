(e3-design-inventory-reconciled)=
## 7.14 Conciliación documental de diseños, selecciones e intentos

La [bitácora canónica actualizada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md) incorpora una conciliación retrospectiva posterior a los resultados conocidos. El {download}`inventario de identidades <../_static/e3_evidence_close_20261120/inventory/design_inventory.csv>`, la {download}`tabla de alias <../_static/e3_evidence_close_20261120/inventory/design_aliases.csv>` y el {download}`resumen estructurado <../_static/e3_evidence_close_20261120/inventory/reconciliation.json>` fijan los conteos y sus unidades. Los registros históricos conservan sus bytes y fechas.

| Unidad documental | Conteo reconciliado |
|---|---:|
| Variantes neuronales funcionales implementadas | **15**: cinco iniciales, tres v2, tres por momentos y cuatro atómicas. |
| Variantes neuronales distintas con ajuste de mercado | **12**; las tres intervenciones iniciales restantes sólo tienen evidencia artificial. |
| Controles analíticos o de transformación examinados | **4**, incluido full proyectado, omitido en el subtotal previo de tres. |
| Propuestas neuronales sin implementación acreditada | **3**. |
| Entradas de diseño documentadas con identidad propia | **22**; un bosquejo de referencia Faithful sigue sin especificación suficiente para decidir si sería una entrada adicional. |
| Ensayos HPO conservados | **183**: 168 en la auditoría y 15 en el piloto V3. |
| Trabajos completos en los alcances conciliados | **844**: 18 neuronales históricos, los 796 de auditoría y 30 neuronales V3. |

Los 22 registros son un conteo **del corpus disponible**, no de todas las ideas informales o decisiones sin archivo. El intervalo 22–23 sólo representa la ambigüedad identificada de aquel bosquejo dentro del corpus; no constituye un límite superior de todas las oportunidades de selección. La conciliación cierra las identidades observables y deja una ausencia concreta, en lugar de equiparar ocho recetas, diez semillas o 796 trabajos al Nd global.

La completa inicial y full/M11 se deduplican; M00 y la aditiva conservan sus identidades. Las 93 corridas de la aditiva —una v2, diez de estabilidad, 76 de auditoría y seis del nuevo piloto— son repeticiones de una estructura. Las pérdidas distintas sí definen recetas. El {download}`registro de selección <../_static/e3_evidence_close_20261120/inventory/selection_opportunities.csv>` separa tasas, inicialización y validaciones; las decisiones de cuadratura 64/128/256 o pruebas artificiales no se ocultan ni se cuentan como arquitecturas entrenadas en mercado.

El {download}`ledger comprimido de intentos <../_static/e3_evidence_close_20261120/inventory/attempt_ledger.csv.gz>` contiene 824 ajustes neuronales, cuatro ARIMA y 16 trabajos sin optimizador. Los 900 renglones de recursos históricos corresponden a fases de 796 trabajos únicos; el autoencoder tiene tres fases que comparten presupuesto. El {download}`registro de incidentes <../_static/e3_evidence_close_20261120/inventory/incidents.json>` conserva interrupciones de admisión, transporte y trabajador. Tres fases históricas mantienen cómputo perdido desconocido; los sobrepasos de plazo que incluyen desconexiones no se presentan como horas GPU medidas.

El {download}`verificador reproducible <../_static/e3_evidence_close_20261120/inventory/verify_inventory.py>`, las {download}`fuentes propias congeladas <../_static/e3_evidence_close_20261120/inventory/sources.zip>` y su {download}`manifiesto <../_static/e3_evidence_close_20261120/inventory/source_manifest.json>` permiten reconstruir la conciliación sin PyTorch, NPZ, pesos o entrenamiento. Se declara IA; los 21 blobs fuente se cotejaron con sus commits públicos. El conteo no corrige por sí solo el sesgo de selección, restaura cegamiento o acredita R1/R3. La V3 completa conserva su resultado adverso y la heurística incumplida, sin promocionar retrospectivamente al control ganador.
