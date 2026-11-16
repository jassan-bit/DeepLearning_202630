# Entrega consolidada con la evidencia disponible

**Resolución final vigente:** [§67](67_resolucion_final.md) sustituye el seguimiento abierto de estas cuatro observaciones. El cierre conserva los requisitos no acreditados y sus límites; no implica aprobación científica ni nuevas tareas correctivas en esta versión.

> **Ubicación:** complemento del [capítulo 8](08_reproducibilidad.md). Identificador documental anterior: **§47**; [ver concordancia](concordancia_guia.md).

**Lectura vigente:** el informe entrega el benchmark de **16 sistemas, 81 predictores y 608 fechas**, junto con sus fuentes, límites y suplementos. La ruta metodológica es **F2a, series de tiempo**. **La búsqueda NAS del capítulo 14 no aplica al catálogo actual de arquitecturas definidas**, incluida TiDE publicada R0-v2. No constituye un experimento pendiente ni se sustituye por las seis configuraciones HPO.

La [síntesis científica del 16/11 (§65)](65_cierre_cientifico_real.md) reúne los controles reales sobre las 608 fechas, la comparación formal de periodos, RQ3 CPU y la cronología recuperada. El [cierre de alcance Scopus](62_cierre_literatura_focal.md) fija los 7.500 registros originales ya analizados y finaliza su recuperación adicional, con limitación de cobertura explícita. El [seguimiento posterior (§66)](66_seguimiento_pendientes.md) añade seis controles negativos, comparación OOF HPO–prueba, descomposición finita de pérdidas, diagnóstico de dependencia de entradas y nuevos archivos independientes. La {download}`repetición CPU de recetas fijas desde cero <../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/README.md>` está completada con sus propios resultados; se distingue de los controles sobre los checkpoints originales.

La [lista verificable](../_static/e2_agregacion_entrega_current_v1/guide_checklist.csv), la [trazabilidad](../_static/e2_agregacion_entrega_current_v1/decision_traceability.csv), el [índice de evidencias](../_static/e2_agregacion_entrega_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_agregacion_entrega_current_v1/manifest.json) relacionan esta entrega con las páginas 58–65 de la guía. El {download}`índice de esta consolidación <../_static/e2_agregacion_entrega_current_v1/README.md>` explica su alcance. El {download}`texto anterior exacto <../_static/e2_agregacion_entrega_current_v1/47_entrega_consolidada_before_update.md>` también permanece íntegro en el archivo final.


## Archivos independientes para descarga y ejecución local

La edición del 16/11 ofrece doce componentes como ZIP separados en {ref}`la portada de descargas <e2-entregables-portada>`. Cada archivo contiene su propio punto de entrada, datos/fuentes e instrucciones; no necesita copiar otro paquete ni clonar el repositorio. El recibo de ejecución identifica el snapshot y las comprobaciones realizadas con los paquetes extraídos en carpetas ajenas al proyecto.

D01 incluye el libro offline completo y su fuente reconstruible; D02 ejecuta las trece arquitecturas en CPU y ofrece refit CUDA opcional; D03 reconstruye las cohortes reales train655/cal53/test608 e incluye el OHLCV diario recuperado; D06/D08 reproducen métricas, cobertura y errores desde las trayectorias incluidas; D10 ofrece reanálisis agregado y entrenamiento CPU opcional; D12 prepara 16 diapositivas y notas. Los demás paquetes generan sus informes y registros localmente. Esta comprobación de portabilidad tiene un alcance distinto del replay integral de toda la campaña científica y no cambia por sí sola los estados del Apéndice B.

[Inventario de la edición 16/11](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/index.json) · [Recibo de ejecución de los ZIP](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/execution_report.json) · [Instrucciones](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/LEEME.md). La [edición anterior del 15/11](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v2_20261115/index.json) y su [recibo propio](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v2_20261115/execution_report.json) permanecen disponibles con sus bytes originales.

(e2-entrega-resumen-vigente)=
## Resumen de la entrega

Se pronostica diariamente la volatilidad de Bitcoin con siete días completos de OHLCV y horizonte de un día. El objetivo es la desviación poblacional de siete retornos logarítmicos, sin anualizar. La ruta es F2a por su dependencia temporal. El benchmark compara diez candidatos, tres referencias neuronales de E1 y tres comparadores clásicos en 608 fechas comunes. Cada red tiene seis semillas; los clásicos deterministas, una trayectoria. SAE registra la menor media de RMSE individuales, 0,003803430, frente a 0,003921625 de ARIMA y 0,004001162 de GRU32. Ninguno de los veinte contrastes primarios de mejora práctica rechaza tras Holm; los resultados son no concluyentes, sin demostrar equivalencia. Se entregan auditorías de datos, calibración temporal, robustez, explicaciones, tres ablaciones y 81 reproducciones en entorno limpio con discrepancia máxima de predicción cero. Las arquitecturas actuales están definidas y la NAS condicional no aplica. La ampliación ejecuta auditoría real de train/cal/test, 13 controles con etiquetas permutadas, un adversarial que sí detecta deriva, pruebas de implementación y RQ3 CPU completa. La brecha train/cal ya tiene trece contrastes comparables; la comparación formal calibración/prueba no constituye validación HPO independiente. Persisten la adecuación conjunta no demostrada, los límites de potencia, cobertura HPO y coste homogéneo completo. El test heredado ya era conocido; la evidencia es retrospectiva. Se conserva la tabla original y se distinguen los análisis posteriores, sin seleccionar otra receta por el ranking final.

Los valores e IC se consultan en el [benchmark vigente](06_benchmark_final.md) y sus fuentes; no se deduce incertidumbre de nuevos mercados a partir de seis inicializaciones. Los capítulos [12](12_incertidumbre_errores.md) y [13](13_agregacion_estadistica.md) separan predictibilidad, incertidumbre, unidades y familias estadísticas.

(e2-entregables-guia-vigentes)=
## Entregables de la guía: ubicación y estado

| Entregable, pp. 58–59 | Evidencia disponible | Estado y límite |
| --- | --- | --- |
| **Informe en Jupyter Book y ejecución reproducible** | Este libro, [organización](concordancia_guia.md), [entorno e instrucciones](08_reproducibilidad.md). | Libro compilable y fuentes versionadas. El {download}`flujo de recetas fijas CPU <../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/README.md>` ya completó entrenamiento, calibración, congelación, admisión, scoring y análisis de 20.000 remuestreos. La auditoría aritmética de entrega tiene otro alcance; no se declara repetida toda la campaña histórica. |
| **Código completo y documentado** | [Repositorio](https://github.com/jassan-bit/DeepLearning_202630), [motor científico fijado](https://github.com/jassan-bit/DeepLearning_202630/tree/1e8745497b484518f766b6da5c19be8362d9add7), capítulos [4](04_implementacion.md), [7](07_ruta_temporal.md) y suplementos. | Hay implementaciones, configuraciones, tests y recibos; controles sólo implementados no se presentan como experimentos ejecutados. |
| **Datos o instrucciones verificables, Data Card y contrato** | [Problema y fuentes](01_punto_partida.md), [EDA](02_eda.md), [contrato y auditoría](10_auditoria_datos.md). | Adquisición oficial, hashes y reglas documentados. Inventarios y hashes no equivalen a publicar bytes de todos los tensores o checkpoints. |
| **Preregistro fechado y reporte de fuga** | [Capítulo 9](09_protocolo_registrado.md), [capítulo 10](10_auditoria_datos.md) y [cierre científico](65_cierre_cientifico_real.md). | Se verificó anterioridad del protocolo/motor reducido para 170 HPO nuevos y 78 finales; 61 cacheados conservan su historia. Procedimientos de auditoría real 655/53/608, 78 ports, 36 controles clásicos, [pantalla de 35 atributos](../_static/e2_scientific_closure_20261116/univariate_attributes/result_v1/summary.json), adversarial y 13 etiquetas permutadas ejecutados. El adversarial significativo se explica y trata; el requisito de rendimiento cercano al azar del control permutado sigue sin establecerse pese a cero alertas Holm13. |
| **Ficha de ruta y pruebas de tablas A/B/C** | [Ruta F2a](07_ruta_temporal.md), [mapa de veinte pares](../_static/e2_route_current_v1/contrast_battery_map.csv), [diagnósticos](49_diagnosticos_f2a.md) e [inferencia](13_agregacion_estadistica.md). | [§64](64_correcciones_metodologicas.md) añade trece pruebas de δg con riesgos train/cal comparables de los 78 predictores CPU: dos tolerancias condicionales. [§65](65_cierre_cientifico_real.md) añade comparación formal de los riesgos originales de calibración/prueba y RQ3 CPU completa. No certifican paridad CUDA, validación HPO independiente ni los cuatro criterios de adecuación conjuntos. |
| **Tabla comparativa según Apéndice A** | [Benchmark](06_benchmark_final.md), [comparación de dieciséis sistemas](../_static/e2_benchmark_current_v1/comparison_current.csv), [Apéndices A.1–A.4](61_reproduccion_consolidada.md). | Métricas, IC, calibración, robustez y costes con sus ámbitos. [§64](64_correcciones_metodologicas.md) añade veinte d_z con IC; los recursos clásicos y el coste integral no medidos permanecen explícitos. |
| **Registro experimental y fichas de modelo** | [Capítulo 11](11_hpo_equidad.md), [registro de trece redes](../_static/e2_data_training_uncertainty_current_v1/model_training_register.csv), [protocolo y recibos](48_protocolo_reducido.md). | Configuraciones, semillas, ganadores y resultados preservados; fallos y reutilizaciones conservan su procedencia. |
| **Techo e incertidumbre** | [Capítulo 12](12_incertidumbre_errores.md), [calibración condicional](60_calibracion_temporal_condicional.md), [proxy empírico](63_ensamble_calibracion.md). | Cuantificación publicada, incluidos resultados adversos. El proxy y su γ incierto no identifican techo óptimo ni ruido Bayes. |
| **Búsqueda de arquitectura para modelos sin diseño predefinido** | [No aplicabilidad del capítulo 14](14_arquitecturas.md), [fuentes de los cuerpos](44_arquitecturas_publicadas.md) y TiDE publicada en [§48](48_protocolo_reducido.md). | **No aplica al catálogo actual.** No se exige listado de 1.000 arquitecturas, dos fases o finalistas NAS a estos cuerpos. HPO y registro de las adaptaciones sí se conservan. |
| **Paper guía e insumo para E3** | [Réplica TimeMixer/ETTh1](05_paper_guia.md), limitación de batch y piloto sobre tres modelos. | Réplica parcial con criterio previo cumplido. El piloto no confirma una limitación predictiva transferible a Bitcoin; no se proclama originalidad para E3 desde ese resultado. |
| **Lista de reproducibilidad completada** | [Apéndice B vigente](61_reproduccion_consolidada.md), [CSV actual de 21 ítems](../_static/e2_scientific_closure_20261116/delivery/appendix_B_current.csv). | El corte actual distingue 14 completos, 5 parciales y 2 no aplicables. Los reportes de pruebas, familias y estimación operativa de techo se reconocen aunque sus resultados sean adversos o inciertos; eso no acredita adecuación conjunta favorable. |
| **Presentación oral con defensa técnica** | Las tablas, fuentes, decisiones y límites del libro sustentan la defensa. | Su ejecución y evaluación oral no se acreditan mediante la publicación del informe. |

(e2-evaluacion-rubrica-vigente)=
## Evaluación: pesos y evidencia para la revisión

| Criterio literal de la guía, p. 59 | Peso | Ubicación principal |
| --- | ---: | --- |
| Definición del problema, hipótesis, preregistro y EDA orientado a decisiones | 15 % | [1](01_punto_partida.md), [2](02_eda.md), [9](09_protocolo_registrado.md) |
| Revisión sistemática de literatura, paper guía y gap | 15 % | [3](03_literatura.md), [5](05_paper_guia.md), [literatura focal](62_cierre_literatura_focal.md) |
| Implementación de los 10 modelos, búsqueda de arquitectura y rigor experimental | 25 % | [4](04_implementacion.md), [11](11_hpo_equidad.md), [14: NAS no aplicable](14_arquitecturas.md) |
| Benchmark e inferencia estadística (diferencias significativas) | 20 % | [6](06_benchmark_final.md), [13](13_agregacion_estadistica.md) |
| Ruta por tipo de problema: bondad de ajuste, sobreajuste, calibración, robustez, incertidumbre e interpretabilidad | 15 % | [7](07_ruta_temporal.md), [12](12_incertidumbre_errores.md), [robustez e interpretabilidad](54_robustez_interpretabilidad.md) |
| Reproducibilidad, redacción científica y defensa | 10 % | [8](08_reproducibilidad.md), [Apéndice B](61_reproduccion_consolidada.md), defensa oral |
| **Total** | **100 %** | Pesos originales, sin redistribución |

**Que la NAS no aplique no elimina el criterio de implementación y rigor del 25 % ni redistribuye sus puntos.** La ponderación corresponde a la evaluación docente; contar modelos, pruebas, casillas o entrenamientos no produce una nota. Tampoco un resultado no significativo implica perder automáticamente el componente de inferencia: debe valorarse su método, reporte e interpretación, sin forzar un rechazo.

### Niveles de desempeño y buenas prácticas

Los niveles **sobresaliente, satisfactorio, en desarrollo e insuficiente** son descriptores de la rúbrica de la página 60, no un diagnóstico automático de esta entrega. El nivel sobresaliente reúne requisitos simultáneos de auditoría, preregistro, ruta completa, potencia, efectos, sensibilidad, gap y reproducción. Las condiciones parciales declaradas no se borran por una compilación o reproducción aprobada; no se asigna aquí un nivel o una calificación.

La evidencia actual conserva seis semillas por red y resultados por ejecución; el piloto de potencia y su objetivo no alcanzado se reportan en [9](09_protocolo_registrado.md). Media, DE, IC y efectos se identifican por estimando y unidad. Las [familias y métodos](13_agregacion_estadistica.md) mantienen corrección múltiple y sensibilidad; el no rechazo no se interpreta como equivalencia. Los capítulos [10](10_auditoria_datos.md) y [11](11_hpo_equidad.md) separan causalidad, auditoría, equidad y presupuesto. El [techo e incertidumbre](12_incertidumbre_errores.md) incluyen sus límites. La regla de dos fases NAS queda expresamente **no aplicable** a los cuerpos actuales.

(e2-entrega-acceso-vigente)=
## Acceso, reproducción y límites del paquete

El [repositorio versionado](https://github.com/jassan-bit/DeepLearning_202630) es la fuente de código y documentación. Las descargas se ofrecen junto a cada capítulo y el [índice de evidencia actual](../_static/e2_agregacion_entrega_current_v1/current_evidence_index.json) fija sus hashes. El ZIP enlazado en el archivo histórico conserva el corte del 12 de noviembre: **no es un paquete actualizado del benchmark final** y no se sobrescribe bajo la misma URL.

Desde la raíz, el siguiente comando comprueba los artefactos científicos publicados y reconstruye A2:

```bash
python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

Verifica 195 archivos de los manifiestos científicos vinculados, dieciséis sistemas y 81 predictores; la diferencia máxima de reconstrucción A2 publicada es **2,6020852139652106×10⁻¹⁸**. Su alcance se detalla en el [Apéndice B](61_reproduccion_consolidada.md). **No es un comando para repetir todos los ajustes o generar de nuevo cada tabla y figura.** Los recibos de reproducción en entorno limpio, con discrepancia máxima de predicción cero, corresponden a su campaña original de [§55](55_reproduccion_ablaciones.md).

La [ampliación del 15/11](64_correcciones_metodologicas.md) ejecutó veinte efectos estandarizados con IC y RQ2. El [cierre científico del 16/11](65_cierre_cientifico_real.md) admite las entradas reales de las 608 fechas y completa los controles adversarial/permutados, las auditorías de implementación, la comparación de periodos y la IUT de RQ3 en CPU. El adversarial significativo revela deriva, descrita y tratada sin reajustar A2. [§66](66_seguimiento_pendientes.md) ejecuta la comparación OOF HPO249–prueba608, una descomposición finita vigente con IC y el diagnóstico específico de dependencia de volumen/OHLC. No identifica varianza poblacional entre mercados, validación independiente posterior a selección ni una señal espuria causalmente demostrada. Los seis controles negativos nuevos tampoco establecen equivalencia con la constante. Permanecen sus límites de aceptación, potencia, amplitud e importancia HPO, recursos clásicos comparables, coste histórico integral y equivalencia práctica. Una única serie y un calendario conocido limitan la extrapolación a otros mercados. La no aplicabilidad de NAS se distingue de estos requisitos pendientes.

Esta consolidación conserva A2 y sus predictores originales, distingue las nuevas inferencias y pruebas realmente ejecutadas y mantiene los cortes anteriores. La repetición CPU desde cero completó 78 ajustes neuronales, tres clásicos y las etapas posteriores hasta sus contrastes de 20.000 remuestreos. Sus [resultados](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/analysis/benchmark_A2.csv), [pronósticos](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/predictions_81.npz) y [recibos](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/export_receipt.json) tienen identidad propia. V4 reutilizó 80 workers exitosos verificados de V3 y ejecutó sólo tres etapas nuevas; no volvió a entrenar las redes. Se conservaron los fallos anteriores y sus cargos. No se repitió HPO ni toda la campaña histórica, y no se afirma igualdad bit a bit con A100.

````{admonition} Archivo documental: contenido anterior de la entrega consolidada
:class: dropdown

Este bloque conserva íntegramente el texto anterior, sus encabezados, fechas y descargas. Las referencias a NAS TiDE, benchmark pendiente o presupuestos anteriores pertenecen a esos cortes. La lectura vigente se encuentra arriba.

> **Ubicación:** complemento del [capítulo 8](08_reproducibilidad.md). Identificador documental anterior: **§47**; [ver concordancia](concordancia_guia.md).

**Consolidación vigente:** [§60](60_calibracion_temporal_condicional.md) añade políticas temporales y cobertura condicional; [§61](61_reproduccion_consolidada.md), auditoría de entrega y Apéndice B; [§62](62_cierre_literatura_focal.md), flujo de 9.559 grupos recibidos y 42 estudios aplicados únicos; [§63](63_ensamble_calibracion.md), referencia empírica de ensemble. Los resultados adversos y límites prospectivos se conservan.

**Actualización de evidencias del 13/11/2026:** [§49](49_diagnosticos_f2a.md) añade los diagnósticos F2a sobre las predicciones originales. Los resultados de §6 siguen siendo el benchmark primario; los suplementos posteriores conservan sus propias hipótesis y límites.

**Actualización del benchmark primario verificada al 13/11/2026, 11:58:36 de Bogotá.** El benchmark común ya está ejecutado: **78 ajustes neuronales finales, 10 pilotos, tres comparadores y 81 predictores congelados** dieron lugar a resultados de los **16 sistemas en las mismas 608 fechas conocidas de E1**. La [tabla principal y el análisis actualizado (§6)](06_benchmark_final.md) contienen las métricas completas, los contrastes, la calibración y las desviaciones registradas. El archivo documental del 12 de noviembre se conserva más abajo con su alcance original.

**Cierre complementario al 13/11/2026, 22:42:55 de Bogotá.** [§55](55_reproduccion_ablaciones.md) documenta **99/99 ajustes nuevos completados: 81 reproducciones en entorno limpio con sus comprobaciones numéricas aprobadas y 18 ablaciones entrenadas del autoencoder**. Las discrepancias máximas registradas en calibración y prueba fueron **0**; esta repetición numérica no aporta una nueva evaluación predictiva independiente. El diario quedó cerrado, sin reservas abiertas, con **80,2561 horas acumuladas consumidas de 84 autorizadas**. La ampliación conserva los cargos y resultados previos. Las conclusiones de esta etapa son posteriores al conocimiento de la prueba; la A2 original y su familia primaria permanecen sin cambios.

**Verificación numérica posterior:** [§56](56_inferencia_estabilidad_final.md) reproduce exactamente los veinte resultados originales de 4.999 remuestreos y recalcula las mismas hipótesis, δ, semillas y Holm con **20.000 réplicas**. Los veinte p ajustados siguen siendo 1,0. Es un refinamiento posterior de precisión, separado del archivo primario original y sin nueva selección.

## Resultado principal del benchmark final

El **LSTM-autoencoder con predictor presenta la menor RMSE media observada: 0,003803430**, con DE entre semillas **0,000245424**. ARIMA obtiene **0,003921625** y GRU32 **0,004001162**. La persistencia obtiene el menor **MAE, 0,002353164**, y la menor **sMAPE, 13,23 %**. Cada posición corresponde a la métrica indicada; no se proclama un ganador universal.

La métrica primaria promedia las RMSE de las seis semillas de cada red, no las predicciones de un ensamble. La evaluación utiliza R=6 para las trece redes y una trayectoria por comparador determinista. El IC95 % t de SAE, **[0,003545873; 0,004060986]**, describe variación entre semillas condicionada a este calendario; no sustituye incertidumbre temporal ni datos independientes.

**Ninguno de los 20 contrastes primarios de mejora práctica rechazó tras Holm; todos sus p ajustados son 1,0000.** El bootstrap principal conserva bloques de 14 días y 4.999 réplicas. La mejora puntual SAE frente a persistencia es de **6,91 %**, pero su IC de ganancia incluye cero. Frente a GRU32, la diferencia puntual es menor que δ. Por tanto, la conclusión confirmatoria es limitada: **la menor RMSE observada no demuestra una mejora práctica general, y la ausencia de rechazo tampoco demuestra equivalencia**. Las sensibilidades, diagnósticos entre semillas y limitaciones se publican junto a la tabla.

## Qué se entrega ahora y qué sigue abierto

- **Ejecutado:** selección de trece recetas, 78/78 ajustes finales, 10/10 pilotos, tres comparadores, congelación y evaluación común de dieciséis sistemas; [métricas A2 originales](../_static/e2_benchmark_final_20261113/benchmark_A2.csv).
- **Auditable:** [análisis y contrastes](../_static/e2_benchmark_final_20261113/benchmark_analysis.json), [admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json), [congelación](../_static/e2_benchmark_final_20261113/final_freeze.json), [piloto y MDE](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json), [procedencia](../_static/e2_benchmark_final_20261113/source_evidence.json) y [manifiesto](../_static/e2_benchmark_final_20261113/manifest.json).
- **Limitaciones conservadas:** evaluación retrospectiva en el calendario ya conocido de E1; ninguna recuperación de cegamiento; R=6 por recursos y alcance del piloto declarado; dos fallos HPO y un desenlace desconocido; fallo ARIMA(0,0,0) de enero sin repetir la oportunidad, dos no convergencias y 15/18 órdenes completamente elegibles; ARIMA(2,0,1) y periodo estacional 7 seleccionados por validación. Los intervalos y los resultados negativos también se reportan.
- **Suplementos ejecutados:** [§50](50_comparaciones_f2a.md) incorpora comparaciones DM/MCS; [§51](51_eficiencia_costes.md), costes y Pareto; [§52](52_sintesis_literatura.md), literatura delimitada; [§54](54_robustez_interpretabilidad.md), robustez, explicaciones del Top 3 y perfiles A100. **[§55](55_reproduccion_ablaciones.md) cierra las 81 reproducciones del plan y las 18 ablaciones**, después de conservar y resolver los fallos anteriores de preparación y persistencia. Completar el benchmark no equivale a completar toda la guía; la [matriz de cumplimiento (§38)](38_cumplimiento_guia.md) conserva evidencias y carencias.

No se modificaron pesos, escaladores, calibradores ni hiperparámetros después de observar la evaluación para favorecer estas conclusiones. No se selecciona otra receta por el orden final de A2.

## Archivo histórico: cierre documental del 12/11/2026

**Todo el texto y las descargas que siguen conservan el corte del 12 de noviembre.** Sus estados «benchmark pendiente», sus contadores y su saldo presupuestario corresponden a ese antecedente. La actualización de arriba y §6 presentan el benchmark ejecutado del 13 de noviembre; el paquete ZIP histórico se mantiene sin sobrescribir sus bytes.

**Cierre solicitado: utilizar los experimentos y archivos existentes, sin iniciar nuevas corridas.** Este capítulo reúne resultados, discusión, conclusiones y archivos. Los diseños anteriores permanecen identificados en sus capítulos originales.

## Resumen

Se estudia el pronóstico diario de volatilidad de Bitcoin mediante siete días de OHLCV y horizonte de un día. El objetivo es la desviación estándar poblacional de siete retornos logarítmicos, sin anualización. Se conserva el contrato de E1 y su margen de relevancia práctica. El trabajo aporta un EDA orientado a decisiones, revisión de literatura, implementaciones y adaptaciones D7, una réplica parcial de TimeMixer en ETTh1 y evidencia trazable de selección experimental. El corte disponible reúne 1.705 resultados de la campaña optimizada y 1.802 registros anteriores: 3.507 registros entre ambos historiales. No representan 3.507 modelos finales ni una comparación terminada de los dieciséis sistemas obligatorios. La réplica externa, con cinco semillas, cumple su criterio previo de compatibilidad numérica; no demuestra precisión sobre Bitcoin. El antecedente E1 muestra que la persistencia es un comparador exigente. Los resultados disponibles sostienen conclusiones sobre datos, mecanismos y ejecución, pero no permiten proclamar un ganador del catálogo vigente. Se entrega la evidencia existente con configuraciones, costes, métricas y fuentes, sin completar tablas con resultados sintéticos o de recetas diferentes.

## Contenido de la entrega

| Componente | Evidencia disponible |
|---|---|
| Problema y contrato | D7×5 → σ₇, RMSE, δ=0,00025772842664981106, unidad temporal y preguntas: [§1](01_punto_partida.md), [§17](17_contrato_sincronizado.md) |
| Datos y EDA | Calidad, reconstrucción de etiquetas, transformaciones, dependencia, desplazamiento y decisiones: [§2](02_eda.md), [§10](10_auditoria_datos.md) |
| Literatura y modelos | Revisión, diez candidatos de cuatro familias, tres referencias E1 y tres comparadores: [§3](03_literatura.md), [§43](43_catalogo_obligatorio.md) |
| Arquitecturas y protocolo | Fuentes verificables, adaptación D7, nueve cuerpos definidos, NAS TiDE y calendario registrado: [§44](44_arquitecturas_publicadas.md), [§45](45_enmienda_protocolo.md) |
| Experimentos ejecutados | Réplica/piloto ETTh1, antecedentes E1 y selección histórica con identidades separadas: [§5](05_paper_guia.md), [§41](41_avance_optimizacion_11_origenes.md) y tablas siguientes |
| Reproducción y archivos | Configuraciones, métricas, curvas disponibles, recibos y dependencias: [§8](08_reproducibilidad.md) y descargas al final |

## Resultados existentes de selección

El corte de este archivo histórico corresponde al commit [`795844308c07`](https://github.com/jassan-bit/DeepLearning_202630/commit/795844308c07cbc160d3dd1feb6284b5f443bb20), cerrado el **12/11/2026 a las 12:53:37 UTC**. Su diario quedó `paused_campaign_budget`, con cierre de sesión registrado. Son archivos versionados, no una lectura en vivo de Colab.

| Indicador | Resultado verificado e interpretación |
|---|---|
| Campaña optimizada | **1.705 resultados completados; cero fallos** entre esos resultados |
| Historial anterior | **1.802 registros**, conservando sus recetas y contabilidad |
| Total entre historiales | **3.507 registros**; no equivale a sistemas o réplicas finales |
| Pasos de optimizador conocidos de la campaña optimizada | **2.245.556**; el intento físico original perdido conserva su incertidumbre |
| Selecciones modelo/origen cerradas | **50**, incluidas las que utilizan resultados compatibles del historial |
| Enero–junio de 2024 | HPO cerrado para FCN, ResNet, Inception, TSMixer, iTransformer, MLP16, LSTM32 y GRU32 |
| Julio de 2024 | FCN y ResNet cerrados; Inception, **13/36** intentos |
| Pilotos nuevos durante este cierre | **0**; R y k vigentes no quedan determinados |

Cada dominio HPO modelo/origen comprende 18 configuraciones y dos semillas. Esa pareja permite seleccionar configuraciones; **no satisface las repeticiones finales ni el piloto de potencia de E2**. Los 108 ajustes FCN/ResNet/Inception de enero son reutilizados y no se suman otra vez a los 1.705 nuevos.

Las recetas restauradas de SAE, TimesNet, TimeMixer y PatchTST tienen identidades nuevas; los resultados de sus variantes anteriores no se trasladan a ellas. Las métricas internas de selección tampoco se presentan como errores de prueba final común. [Instantánea](../_static/delivery_20261112/selection_snapshot.json) · [CSV de ensayos](../_static/delivery_20261112/trials.csv) · [Procedencia](../_static/delivery_20261112/source_index.json).

## Antecedente E1 en las 608 fechas conocidas

Esta tabla corresponde **exclusivamente a E1**, con sus recetas y semillas. Contextualiza la dificultad del problema; no rellena resultados de los diez candidatos E2.

| Sistema E1 | Semillas | RMSE media | DE entre semillas | IC95 % de la media entre semillas | MAE media |
|---|---:|---:|---:|---|---:|
| GRU32 | 309 | 0,005212686 | 0,001121207 | [0,005087180; 0,005338192] | 0,003732332 |
| LSTM32 | 309 | 0,006251147 | 0,000795628 | [0,006162086; 0,006340209] | 0,004417266 |
| MLP16 | 309 | 0,005228132 | 0,000541108 | [0,005167561; 0,005288702] | 0,003865300 |
| Persistencia | Determinista | **0,004085596** | No aplica | Sin réplicas artificiales | **0,002353164** |

Los IC de las redes describen incertidumbre de entrenamiento condicionada al calendario fijo. No incluyen nuevos mercados ni hacen independientes las 608 fechas conocidas. La persistencia tiene una trayectoria determinista; la ausencia de DE entre semillas no elimina su incertidumbre temporal. [Tabla fuente y alcance](../_static/delivery_20261112/e1_reference_metrics.csv).

**Interpretación:** en este antecedente, el RMSE de persistencia es menor que la media de cada referencia neuronal. La complejidad de una arquitectura no basta para justificar su uso. Esta diferencia descriptiva no resuelve los contrastes del catálogo vigente ni acredita superioridad futura.

## Réplica ejecutada del paper guía

TimeMixer se replicó en **ETTh1**, historia y horizonte 96, siete variables y cinco semillas: otra tarea y otra escala de métricas.

| Métrica ETTh1 | Referencia publicada | Réplica, media ± DE | IC95 % de la media |
|---|---:|---:|---|
| MSE | 0,375 | 0,377838 ± 0,002870 | [0,374274; 0,381401] |
| MAE | 0,400 | 0,401082 ± 0,003473 | [0,396769; 0,405394] |

La discrepancia MSE es +0,002838, aproximadamente +0,757 %. Su intervalo queda dentro de la tolerancia registrada: **réplica parcial numéricamente compatible**, con diferencias de entorno y alcance declaradas. No es una mejora del artículo ni un entrenamiento de la receta Bitcoin restaurada. El piloto asociado localiza una dependencia de los marcadores respecto al contexto del batch en esa implementación; no se generaliza a toda la familia multiescala. [Método, curvas, análisis y descarga](05_paper_guia.md).

## Discusión y conclusiones

**Ganancia predictiva:** los resultados disponibles no permiten elegir un ganador entre los dieciséis sistemas vigentes. El antecedente E1 muestra un naive competitivo. Los resultados HPO acreditan selección real, cuyo estimando difiere del error final fuera de selección. Por ello, la RQ de mejora predictiva del catálogo E2 permanece sin una respuesta confirmatoria.

**Dependencia y regímenes:** el EDA documenta un día OHLCV incompleto en 2023 y etiquetas solapadas. Las decisiones defendibles son conservar calendario y máscaras, ajustar transformadores únicamente en entrenamiento y utilizar evaluación temporal con embargo. La asimetría y el desplazamiento justifican errores por periodos; no autorizan a tratar fechas o semillas como mercados independientes. La ventaja del catálogo en alta volatilidad aún no se ha demostrado.

**Arquitectura y coste:** la réplica ETTh1 respalda reproducibilidad parcial de un mecanismo y permite localizar una limitación de implementación. Las adaptaciones D7 conservan identidad explícita. Los perfiles y la optimización sintética aportan evidencia técnica, pero no demuestran una mejora global de coste/precisión del benchmark Bitcoin. La RQ conjunta de error y coste requiere una comparación común todavía ausente.

Se entrega una cadena trazable de **datos → decisiones → modelos y protocolo → experimentos disponibles → interpretación**. La conclusión sostenible es mantener evaluación temporal y persistencia como referencias exigentes. Con lo existente no está demostrada la superioridad del Top 10 ni del híbrido histórico.

## Límites del cierre

Se cierra la versión documental con lo existente. **No se certifica cumplimiento integral de la guía.** El benchmark común de dieciséis sistemas, NAS TiDE, R/k compatibles, inferencia del catálogo completo, robustez, incertidumbre/explicabilidad comparables y reproducción independiente de la tabla principal carecen todavía de evidencia suficiente para marcarlos como completados. La [matriz de requisitos](38_cumplimiento_guia.md) conserva el detalle para una revisión posterior; esta lista no ordena otra campaña durante el cierre.

Los pesos privados no están incluidos en las descargas públicas verificadas: inventarios y hashes no sustituyen sus bytes. Los periodos históricos de 608 y 60 fechas ya fueron examinados y no se presentan como prueba virgen.

La contabilidad conciliada conserva **222.927,99 segundos cargados**, aproximadamente **61 h 55 min**, dentro de 64 horas. El saldo observado es **7.472,01 segundos**. La [evidencia presupuestaria](../_static/delivery_20261112/budget_reconciliation.json) no concede tiempo ni demuestra actividad actual. Este cierre no inicia ajustes ni amplía el límite.

## Archivos de entrega

- [Paquete: informe, tablas, configuraciones y procedencia](../_static/delivery_20261112/Entregable2_evidencia_disponible.zip).
- [Manifiesto de archivos y alcance](../_static/delivery_20261112/delivery_manifest.json).
- [Ensayos](../_static/delivery_20261112/trials.csv) · [Resumen de selección](../_static/delivery_20261112/selection_snapshot.json) · [Índice de fuentes](../_static/delivery_20261112/source_index.json).
- [Antecedente E1](../_static/delivery_20261112/e1_reference_metrics.csv) · [Réplica y curvas ETTh1](05_paper_guia.md) · [Reproducción](08_reproducibilidad.md).

La [preparación técnica de pilotos (§46)](46_artefactos_y_pilotos.md) queda como apéndice opcional para una continuación posterior.

````
