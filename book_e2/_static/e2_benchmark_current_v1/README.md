# Evidencia vigente del benchmark final

Este expediente reúne la evidencia ya ejecutada para los apartados 6.1–6.4 de la guía. Conserva la A2 original, los pronósticos congelados, sus familias estadísticas y los registros históricos. Las nuevas tablas y figuras permiten revisar juntos resultados que estaban distribuidos en varios suplementos.

## Comparación y estimandos

El catálogo tiene dieciséis sistemas: diez modelos principales, tres redes de referencia de E1 y tres comparadores deterministas. Las trece redes tienen seis semillas; los clásicos, una trayectoria cada uno. Son 81 predictores evaluados en las mismas 608 fechas, del 01/01/2025 al 31/08/2026.

`comparison_current.csv` copia las cifras publicadas mediante la clave `model`. Sus 38 columnas incluyen métricas A2, IC t de RMSE de las trece redes, rangos condicionados, costes, calibración original, áreas de robustez y pertenencia MCS. `comparison_schema.json` identifica las diez fuentes y los ámbitos de cada columna.

- **Primaria:** media de las RMSE individuales, en unidades originales de σ₇. No es RMSE de un ensemble ni raíz de la media de RMSE².
- **Variación entre semillas:** DE e IC t condicionados a una cronología, no seis mercados. Un clásico R=1 no recibe DE ni IC entre semillas inventados.
- **Cobertura y sMAPE:** fracciones en el CSV; el capítulo declara explícitamente cuándo muestra porcentajes.
- **Calibración:** se copian los intervalos originales de A2. Los suplementos ACI/ventana móvil mantienen sus políticas y alcance retrospectivo; no reemplazan esos intervalos.
- **Robustez:** cuatro áreas adimensionales de degradación relativa, por trece redes y tres severidades por perturbación. Una menor degradación no significa menor error absoluto ni demuestra robustez externa.
- **Ausencias:** las celdas vacías indican magnitudes no medidas o no estimables. Nunca significan cero.

SAE presenta la menor RMSE observada; ARIMA y GRU32 siguen en ese orden. Persistence obtiene el menor MAE y sMAPE. Ninguno de los veinte contrastes primarios demuestra una ganancia mayor que δ tras Holm. El refinamiento de 20.000 remuestreos conserva las mismas hipótesis, estimando y decisiones del original de 4.999; no reescribe su cronología.

## Rangos y costes

`conditional_rank_groups.svg` y `.png` presentan los rangos medios de las trece redes y las conexiones de no rechazo de las 78 pruebas Wilcoxon–Holm ya publicadas. Es un diagrama de rangos con agrupación Holm, en el formato de diagramas de comparación crítica: no calcula una distancia común Nemenyi ni una prueba nueva. Todos los p ajustados son 1. Con seis pares, el mínimo bilateral exacto 0,03125 no permite superar Holm78. La conexión de los trece modelos no demuestra equivalencia o ausencia de diferencias; señala falta de rechazo bajo ese procedimiento condicionado a los datos.

`latency_pareto.svg`, `.png` y `.csv` presentan la dominancia puntual RMSE–latencia de las trece redes a partir de puntos ya medidos. La frontera observada contiene MLP16, GRU32 y SAE. La latencia es la media de seis medianas por semilla, medida con eventos CUDA en A100, entrada residente y batch1, 32 pasadas de calentamiento y 100 repeticiones. Excluye transferencia, preprocesamiento, carga y servicio. Esta reconstrucción descriptiva no incorpora incertidumbre conjunta ni resuelve la prueba específica RQ3.

Los tiempos finales incluyen actualización, evaluación del riesgo y checkpoints; SAE incluye sus dos fases auxiliares y la supervisada. No son coste de selección completo. Los FLOPs core cuentan matrices y convoluciones con 2 FLOPs/MAC; excluyen FFT, activaciones, normalizaciones, pooling, sesgos y movimientos de memoria. Los mínimos/máximos observados en tres sondas no son cotas universales de rutas dependientes de los valores. Los recursos clásicos, el coste completo de búsqueda y las magnitudes excluidas siguen sin medirse.

## Inferencia y discusión

Las familias originales y posteriores responden a preguntas distintas:

| Procedimiento | Ámbito conservado |
|---|---|
| Holm20 primaria | Ganancia RMSE mayor que δ, bootstrap temporal y semillas pareados; cero rechazos. |
| DM–HLN/Holm120 | Igualdad de MSE diaria media de trayectorias fijas; 35 rechazos de 120, sin reinterpretarlos como ganancia práctica primaria. |
| MCS90 % | Procedimiento conjunto sobre esa MSE; permanecen FCN, SAE, GRU32, Persistence y ARIMA. No acredita equivalencia. |
| Rangos/Friedman/Wilcoxon | Diagnóstico de trece redes y seis vectores de semillas; una cronología. Ninguna pareja supera Holm78. |
| GR/constancia | Diagnósticos posteriores con familia propia Holm40; no rechazo no demuestra estabilidad universal. |
| Ablaciones SAE | Tres contrastes de dieciocho ajustes reales; retirar dropout supervisado empeora RMSE con rechazo Holm3. |

La partición, métrica y δ heredados de E1 se aceptan conforme a la guía. El conocimiento previo del periodo limita la confirmación prospectiva. El δg previo está documentado: no se afirma que faltó definirlo. Su comprobación formal para los predictores actuales permanece sin establecerse con riesgos temporalmente comparables. La potencia del piloto fue aproximadamente 12,3 %, frente al objetivo de 80 %.

RQ2/RQ3 conservan sus especificaciones secundarias aclaradas posteriormente, sin declarar ejecutadas sus pruebas específicas. La calibración global no garantiza cobertura por grupos. El ensemble calibrado aporta un proxy empírico de techo, con incertidumbre amplia; no identifica riesgo Bayes ni un porcentaje cierto de mejora pendiente. Cohen/Cliff con IC y las verificaciones Shapiro–Wilk/Levene no se presentan como análisis ejecutados. Los efectos en unidades originales y sus IC sí están disponibles.

## Archivos y reproducción documental

- `guide_checklist.csv` y `.json`: requisitos de la guía, evidencia, límites y tareas de cierre. Sus filas son subcondiciones, no ponderaciones ni calificaciones.
- `decision_traceability.csv`: hallazgos, conclusiones permitidas y límites de la discusión.
- `current_evidence_index.json`: fuentes originales, tamaños, SHA256 y separación de estimandos.
- `comparison_current.csv` y `comparison_schema.json`: unión reproducible de tablas publicadas, con unidades y ausencia de datos explícitas.
- `conditional_rank_groups.*`, `latency_pareto.*` y `summary_plots.json`: nuevas presentaciones de resultados guardados, fuentes y métodos.
- `06_benchmark_final_before_update.md`: copia exacta del capítulo anterior a esta consolidación.
- `manifest.json`: tamaños y SHA256 del expediente.

Desde el checkout, con Python estándar, se puede reconstruir la unión en un directorio nuevo:

```bash
python scripts/assemble_e2_benchmark_current.py --output-dir /tmp/e2-benchmark-comparison
```

El programa comprueba los diez hashes de entrada, identidades y soporte de los modelos, copia los intervalos existentes y rechaza sobrescribir resultados. El script de figuras `scripts/plot_e2_benchmark_summary.py` describe sus argumentos y usa Matplotlib. Ninguno entrena, carga pesos, ejecuta pronósticos, recalcula scores ni realiza nuevos contrastes o remuestreos.

Esta consolidación preserva todas las descargas y anclas anteriores. La reproducción numérica aprobada de los 81 predictores fortalece la trazabilidad; no elimina las limitaciones de una sola cronología ni convierte los análisis posteriores en preregistros históricos.
