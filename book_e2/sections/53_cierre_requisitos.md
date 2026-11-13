# 53. Cierre de requisitos y experimentos complementarios

**Actualización del 13/11/2026.** Esta página separa resultados ya obtenidos de las ejecuciones complementarias iniciadas en la A100. Los suplementos son posteriores al conocimiento del periodo de prueba. No reemplazan la A2, sus veinte contrastes primarios, los predictores congelados ni los registros históricos.

## Evidencias que ya pueden revisarse

| Frente | Evidencia ejecutada | Alcance que permanece abierto |
|---|---|---|
| Comparaciones F2a | [§50](50_comparaciones_f2a.md): 120 DM–HLN/HAC14 con Holm, MCS por bloques con 10.000 remuestreos y rangos condicionales entre semillas. | Son comparaciones posteriores con estimandos propios. El diagnóstico entre semillas no representa otros mercados; δg no se había definido para las pruebas originales. |
| Ajuste y calibración | [§49](49_diagnosticos_f2a.md): residuos, calibración, cobertura, evolución temporal y curvas de los ajustes disponibles. | Los rechazos adversos se conservan. No rechazo no acredita estabilidad, equivalencia o ausencia de sobreajuste. Un intervalo del 90 % no define una CDF para PIT. |
| Literatura | [§52](52_sintesis_literatura.md): diez sistemas actuales vinculados a nueve estudios, extracción, calidad, corpus delimitado, soporte cuantificado y sensibilidad de puntuaciones. | El flujo global de todos los registros históricos y las decisiones pendientes no se convierten en una revisión cerrada. Las puntuaciones documentales actuales no se presentan como decisiones previas al benchmark. |
| Eficiencia | [§51](51_eficiencia_costes.md): parámetros y tiempos originales medidos, costes por fases y frentes de Pareto; [§54](54_robustez_interpretabilidad.md): perfil nuevo de inferencia A100 de los 78 predictores neuronales. | FLOPs, energía y dinero siguen sin medir. El forward CUDA con entrada residente excluye preprocesamiento y transferencia. |
| Reproducción y presentación | Se actualizan los estados de los capítulos centrales y se mantienen todas las descargas originales. | Compilar el libro y reproducir la aritmética de los pronósticos no equivalen a entrenar de nuevo todo el benchmark en un entorno limpio. |

## Robustez, explicaciones y perfil A100 completados

El [estado verificado](../_static/e2_posthoc_gpu_20261113/status.json) y su [procedencia](../_static/e2_posthoc_gpu_20261113/upstream_provenance.json) acreditan **79/79 diagnósticos**, el recibo agregado, **468 filas de robustez**, las explicaciones del Top 3 y **78 perfiles de inferencia**. Se verificaron los SHA de los 321 archivos de 80 respaldos. [§54](54_robustez_interpretabilidad.md) presenta resultados y límites. El [corte previo de 68/79](../_static/e2_posthoc_progress_20261113/status.json) permanece como evidencia histórica; fue superado por este cierre. Su [fe de erratas de formato CSV](../_static/e2_posthoc_gpu_20261113/progress68_erratum.json) conserva la descarga publicada y los bytes originalmente vinculados, con filas y valores idénticos. Las tres consultas nativas previas también devolvieron metadatos, documentados en §52.

El código complementario trabaja en un directorio separado y conserva **GitHub como único destino persistente**. Antes de admitir cómputo comprueba los modelos y archivos originales. Usa una contabilidad adicional vinculada al diario original cerrado, con el mismo límite acumulado autorizado de **80 horas**. No reinicia cargos ni añade crédito.

1. **Robustez de los predictores congelados.** Perturbaciones de ruido, outliers, valores faltantes y cambio de distribución de entradas, con tres severidades registradas por tipo. Las referencias de imputación y escala proceden del entrenamiento. Los objetivos y calibradores originales se conservan; se mide degradación e incertidumbre temporal.
2. **Explicaciones del Top 3 observado.** Autoencoder, ARIMA y GRU32 se explican según las entradas que efectivamente consumen. Se contrastan oclusión, borrado/inserción y un orden de control, y se evalúa estabilidad entre las seis semillas de las redes. Las respuestas del predictor no se interpretan como causalidad del mercado.
3. **Perfil de inferencia.** Se registra hardware, batch, sincronización, calentamiento, repeticiones y memoria. Un perfil nuevo no se atribuye a la ejecución original ni completa operaciones FLOPs que el método no mida.
4. **Reproducción en un entorno nuevo.** Se prepara un checkout limpio y un entorno Python separado, se conservan las recetas ganadoras y los datos admitidos, y se ajustan nuevamente los 81 predictores: 78 redes y tres comparadores. Se registran discrepancias de estados y predicciones frente a los originales; no se usa esa comparación para seleccionar otra receta.
5. **Tres ablaciones entrenadas del autoencoder.** Se retiran, por separado, el preentrenamiento auxiliar, el segundo encoder y el dropout supervisado. Se usan las seis semillas y los hiperparámetros y épocas ya fijados. El autoencoder completo reproducido sirve de referencia. Son **18 ajustes de ablación adicionales**, con pesos y curvas propios; se distinguen de retirar entradas únicamente en inferencia.

La reproducción y las ablaciones suman **99 ajustes nuevos planificados**, no resultados completados. Después del cierre de robustez, la preparación del entorno limpio **falló antes de admitir cualquier ajuste nuevo**: el [recibo presupuestario](../_static/e2_posthoc_gpu_20261113/setup_failure_budget.json) conserva su cargo de 13,0199 segundos, con estado `failed`. La causa técnica no figura en ese recibo; requiere los logs locales de checkout e instalación. Los 79 diagnósticos completos se conservan y no necesitan repetirse.

El controlador sólo admite el siguiente trabajo cuando su reserva cabe en el saldo disponible, registra el coste real y respalda los recibos. Una interrupción conserva sus cargos e incertidumbre; no autoriza repetir silenciosamente un ajuste perdido. La validación del software con datos sintéticos no constituye la ejecución de estos experimentos.

## Límites que una ejecución posterior no puede borrar

El periodo de prueba ya se conocía en E1 y se ha analizado en E2. Una ejecución nueva no recupera cegamiento prospectivo. Las hipótesis complementarias deben conservar su fecha real y su condición posterior. Tampoco es posible convertir retrospectivamente un margen δg ausente en un margen registrado antes de los experimentos originales.

Las evidencias nuevas se incorporarán con sus recibos de GitHub, tiempos y resultados efectivamente medidos. El estado actual es **robustez, explicaciones y perfil completados; reproducción y ablaciones pendientes tras el fallo de preparación**. La [matriz de cumplimiento](38_cumplimiento_guia.md) distingue ese estado de un cierre integral del entregable.
