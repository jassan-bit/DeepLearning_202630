# Evidencia vigente de la ruta metodológica

Este expediente acompaña los apartados 7.1–7.5 de la guía. Reúne la clasificación del problema, los once pasos comunes, las tablas A/B/C, la ruta temporal y las decisiones sustentadas por los resultados guardados. Conserva la evidencia anterior y no modifica el contrato científico.

## Ruta aplicable y continuidad

Corresponde **F2a, series de tiempo**. La regla RT1 de F2b exige reclasificar cuando una referencia deja autocorrelación residual fuerte. La referencia Ridge D7 de desarrollo, con 305 fechas de diez ajustes mensuales de 2024, mostró ACF(1)=0,632228 y Ljung–Box lag7 Q=266,4062, p≈8,88×10⁻⁵⁴. La aproximación diagnóstica tiene límites por dependencia, solapamiento y cambios de ajuste. Esta evidencia no demuestra que añadir rezagos del objetivo mejore el modelo ni que exista causalidad o no linealidad aprovechable.

`route_decision.json` conserva la decisión histórica y sus fuentes, y distingue el calendario actual de aquellos diseños anteriores. Los antiguos catálogos de veinte sistemas, las asignaciones previas y sus fechas de ajuste no se trasladan al protocolo reducido.

Se mantienen D7×5 OHLCV → σ₇ poblacional, h=1, RMSE media individual por semilla, δ, unidad temporal y partición heredados de E1. Los controles de disponibilidad de covariables de F2b siguen siendo pertinentes. Una entrada matricial no activa automáticamente la ruta tabular. Los tres folds HPO actuales y los tamaños 655/53/608 tienen un papel distinto de los diez folds del diagnóstico histórico. Las redes no reciben un canal nuevo de Y rezagada.

## Batería real y alcance inferencial

Para cada uno de los veinte pares primarios sí se ejecutaron **tres procedimientos originales**: bootstrap pareado de semillas/calendario, t pareada entre semillas y enumeración exhaustiva de 64 cambios de signo. Comparten la ganancia de RMSE y δ; cada lista original de veinte tiene su propio control Holm20. La definición concreta quedó congelada antes del scoring actual. El periodo de E1 ya era conocido: no se atribuye a ese registro un nuevo test ciego.

La guía exige tres pruebas distintas con las categorías y supuestos pertinentes. No basta una sola prueba acompañada de Holm. Tampoco exige independencia entre las tres ni obliga a sumar Friedman/MCS cuando el componente de control múltiple ya está satisfecho. El contador de tres no verifica por sí mismo normalidad de seis ganancias, simetría para cambios de signo o adecuación de bloques temporales. Los métodos condicionados al calendario y el remuestreo temporal deben conservar sus dominios de incertidumbre.

`contrast_battery_map.csv` identifica cada par, sus procedimientos y las fuentes posteriores relacionadas. El bootstrap original de 4.999 y el refinamiento de 20.000 son **un mismo método**, no dos pruebas; las sensibilidades 7/14/28 tampoco crean métodos adicionales. DM, MCS y Friedman/Wilcoxon posteriores conservan sus hipótesis y familias. Los diez pares frente a Persistence no aparecen en Wilcoxon13 porque los clásicos no se repiten artificialmente seis veces. Las comparaciones MSE o contra cero no se presentan como confirmaciones adicionales del margen práctico RMSE.

## Razones frente a los pronósticos ingenuos

F2a permite reportar MASE o la razón de error frente al ingenuo. `relative_error_vs_naive.csv` incluye los dieciséis sistemas y dos cocientes descriptivos:

```text
RMSE_ratio_vs_Persistence = RMSE_A2_del_sistema / RMSE_A2_de_Persistence
RMSE_ratio_vs_SeasonalNaive = RMSE_A2_del_sistema / RMSE_A2_de_SeasonalNaive
```

Se usan las mismas 608 fechas y las RMSE ya publicadas: media de seis RMSE individuales para redes y una trayectoria para cada referencia determinista. Un cociente menor que uno indica menor RMSE puntual. Para SAE frente a Persistence es 0,9309363276, coherente con la reducción observada de 6,91 %. El cociente no es MASE, una nueva prueba de hipótesis, equivalencia ni evidencia de ganancia superior a δ. No cambia la métrica primaria ni incorpora SeasonalNaive a la familia original de veinte comparaciones. `ratio_provenance.json` conserva fórmula, denominadores y hash de A2.

## Adecuación y decisiones actuales

Ningún candidato acredita simultáneamente las cuatro condiciones de adecuación: mejora práctica ajustada frente al trivial, tolerancia de sobreajuste, calibración sin estructura residual y consistencia comparable entre validación y prueba. Esto describe los resultados y límites actuales; no significa que falten los entrenamientos del benchmark ni invalida automáticamente toda utilidad predictiva.

El **δg previo existe**. Para pérdidas, la brecha se orienta como RMSE_validación−RMSE_train: nula de brecha≥δg frente a alternativa menor. La mediana corresponde al procedimiento Wilcoxon y la media a la t bajo sus supuestos. Las brechas train/calibración guardadas no ejecutan una nueva prueba formal temporal comparable. Los IC descriptivos calibración53/prueba608 tampoco equivalen a validación interna de los mismos ajustes. Un no rechazo no demuestra ausencia de sobreajuste, estabilidad o equivalencia.

Las curvas por época ya disponibles no son curvas por tamaño de entrenamiento. El catálogo A/B/C contiene alternativas: no se reclaman como obligatorias simultáneas todas las pruebas de clasificación, espaciales, de optimismo o de validación. PIT necesita una distribución predictiva; dos extremos de un intervalo no la definen. Las perturbaciones sintéticas y la calibración temporal posterior conservan sus limitaciones.

## Archivos y verificación

- `guide_checklist.csv` y `.json`: subcondiciones de la guía con evidencia, estado, límites y tareas concretas; no son ponderaciones ni calificaciones.
- `decision_traceability.csv`: relación entre hallazgos, decisiones permitidas y evidencia.
- `contrast_battery_map.csv`: veinte pares y procedimientos realmente disponibles, sin calcular estadísticos nuevos.
- `route_decision.json`: clasificación, contrato y fuentes históricas/actuales separadas.
- `relative_error_vs_naive.csv` y `ratio_provenance.json`: cocientes descriptivos y procedencia reproducible.
- `current_evidence_index.json`: fuentes y ámbitos con SHA256.
- `07_ruta_temporal_before_update.md`: copia exacta del capítulo anterior.
- `manifest.json`: tamaños y hashes del expediente.

Las cifras se verifican desde las tablas originales y sus hashes. Para reproducir los cocientes basta leer A2, identificar las filas Persistence/SeasonalNaive y dividir cada RMSE publicada por la referencia correspondiente. No se cargaron pesos, datos brutos ni nuevas predicciones; no hubo nuevos ajustes, pruebas, remuestreos o cambios de presupuesto. Los resultados y enlaces históricos se conservan.
