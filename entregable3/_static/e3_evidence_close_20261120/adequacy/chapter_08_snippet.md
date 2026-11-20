(e3-existing-adequacy-close-20261120)=
## 8.20 Adecuación retrospectiva con predicciones guardadas

Este cierre completa evidencia que podía calcularse a partir de archivos ya
guardados. Verifica las doce tablas de la auditoría de 8.18 y cuatro paneles
inmutables del candidato; de estos últimos conserva sólo 18.080 predicciones
de prueba limpia, correspondientes a diez semillas en cada conjunto. No entrena
modelos, realiza inferencia ni repite los 9.999 remuestreos anteriores. El
{download}`manifiesto de entradas <../_static/e3_evidence_close_20261120/adequacy/archive/input_manifest.json>`
y el {download}`recibo del reanálisis <../_static/e3_evidence_close_20261120/adequacy/results/adequacy_receipt.json>`
fijan el origen, los filtros y los hashes; se declara asistencia de IA.

### Brecha, controles triviales y cobertura

Las métricas de ajuste y de las particiones posteriores ya estaban guardadas.
Las {download}`doce comparaciones de brecha <../_static/e3_evidence_close_20261120/adequacy/results/descriptive_fit_heldout_gaps.csv>`
emparejan semillas dentro de cada conjunto. La tabla resume prueba limpia:

| Conjunto | RMSE prueba / RMSE ajuste¹ | Cobertura media, nominal 90 % | Menor cobertura trimestral media² |
|---|---:|---:|---:|
| D00 | 0,9364 | 94,51 % | 86,44 % —2026Q1— |
| D01 | 0,2945 | 91,50 % | 82,10 % —2012Q4— |
| D02 | 0,7802 | 91,88 % | 90,11 % —2021Q2— |
| D03 | 0,6716 | 89,29 % | 85,71 % —2024Q1— |

¹ Cociente de RMSE medias individuales, sin ensamblar predicciones. Las
particiones corresponden a periodos y regímenes distintos; una razón menor
que uno **no demuestra ausencia de sobreajuste**, en particular en D01. Se
conservan también la dispersión y la brecha emparejada entre semillas sobre
el calendario fijo. No se reconstruye un margen de brecha ni potencia previos.

² Mínimo descriptivo entre todos los trimestres guardados, no una selección
previa para contrastar. La {download}`tabla de cobertura y deriva <../_static/e3_evidence_close_20261120/adequacy/results/coverage_drift_readout.csv>`
conserva fechas, rangos entre semillas y ámbito de cada valor.

La {download}`comparación con cinco controles triviales <../_static/e3_evidence_close_20261120/adequacy/results/trivial_baseline_comparison.csv>`
incluye los dos controles de piso, persistencia, estacional y ARIMA. Aunque el
candidato tiene menor RMSE, su **MAE supera a la de `floor_only` en 5,63 % en D00
y 26,59 % en D01**. Una ventaja en la métrica de entrenamiento no se interpreta
como superioridad en todas las pérdidas o escenarios.

### Diagnósticos de residuos: dos ejecuciones, dos familias

Se calcularon Ljung–Box a rezagos 7/14, ARCH-LM a 7 y la hipótesis conjunta de
Mincer–Zarnowitz —intercepto cero y pendiente uno— con HAC Bartlett de rezago
14. Son rezagos en observaciones, que representan sesiones en D01. Se usa una
regresión auxiliar sobre predicciones ya guardadas; no se ajusta nuevamente el
modelo de pronóstico. Las ventanas del objetivo comparten retornos y las diez
semillas no constituyen diez mercados independientes.

La {download}`familia de la auditoría ampliada <../_static/e3_evidence_close_20261120/adequacy/results/extended_candidate_residual_diagnostics.csv>`
contiene **160 valores p diagnósticos**, ajustados conjuntamente por Holm:
cuatro conjuntos × diez semillas × cuatro procedimientos/rezagos. Entre los
resultados adversos se encuentran los siguientes rangos entre semillas:

| Conjunto y diagnóstico | Rango de p ajustado, Holm160 | Lectura delimitada |
|---|---:|---|
| D00, ARCH-LM(7) | 8,15 × 10⁻¹³ a 0,02882 | Evidencia asintótica contra varianza residual constante. |
| D01, Mincer–Zarnowitz con HAC(14) | 1,40 × 10⁻¹⁸ a 4,63 × 10⁻¹⁰ | Desviación de la calibración lineal de la media. Las pendientes se sitúan entre 0,9452 y 0,9702. |

La tabla completa conserva los resultados restantes. Los valores grandes o una
falta de rechazo no certifican adecuación. La normalidad no es requisito general
para que un pronóstico de regresión sea útil: Jarque–Bera se aporta como
estadístico de forma, con asimetría y curtosis, **sin interpretar su p iid** bajo
dependencia serial. La asimetría positiva y las colas pronunciadas se conservan.

El {download}`diagnóstico histórico separado <../_static/e3_evidence_close_20261120/adequacy/results/historical_residual_diagnostics.csv>`
utiliza las diez trayectorias de 608 fechas del commit `510c3bdd`, con pesos
anteriores y una familia propia Holm40. No se mezcla con Holm160 ni se atribuye
a V3. Las dos familias son retrospectivas y auxiliares: **no son las tres pruebas
de superioridad por rival de R3**, no justifican potencia prospectiva y no
completan automáticamente las Tablas A/B.

Los valores p están condicionados a los supuestos de dependencia, momentos y
aproximación asintótica. Holm no corrige un p marginal inválido. Los datos
expuestos no recuperan cegamiento y una decisión posterior no se convierte en
preregistro. Tampoco se fabrican PIT o CRPS para un punto acompañado sólo por
un intervalo; esa salida no define una distribución predictiva completa.

### Recursos, reproducción y requisitos abiertos

El {download}`inventario de campos de recursos <../_static/e3_evidence_close_20261120/adequacy/results/resource_field_inventory.csv>`
distingue dato suministrado, ausente o no aplicable. **900 filas de fases
representan 796 trabajos**, con fases adicionales del autoencoder. Tres
segmentos mantienen cómputo activo desconocido tras una interrupción. El máximo
exceso de plazo, 35.283,63 segundos, incluye desconexión y **no es una medida de
9,8 horas de cómputo GPU**. No se completa energía o FLOPs usando parámetros,
ni se certifica igualdad de cómputo total real por compartir un límite nominal.

La {download}`metodología y comandos <../_static/e3_evidence_close_20261120/adequacy/README.md>`,
el {download}`script de reanálisis <../_static/e3_evidence_close_20261120/adequacy/audit_existing_predictions.py>`
y la {download}`verificación independiente <../_static/e3_evidence_close_20261120/adequacy/results/reference_verification.json>`
permiten reproducirlo sin red ni modelos: se comprobaron 250 resultados numéricos
contra statsmodels y 200 ajustes de Holm. La concordancia informática no
demuestra los supuestos científicos. La {download}`matriz de cierre <../_static/e3_evidence_close_20261120/adequacy/closure_matrix.csv>`
separa lo calculado de lo pendiente: margen/criterios de adecuación y potencia
previos, interpretación de estabilidad y distribución, datos realmente reservados,
familia confirmatoria pertinente, recursos completos y reproducción independiente.
**Este cierre aporta diagnósticos; no acredita R1 o R3 ni supera formalmente
las Tablas A/B por retrospectiva.**
