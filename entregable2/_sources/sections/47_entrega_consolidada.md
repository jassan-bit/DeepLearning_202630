# 47. Entrega consolidada con la evidencia disponible

**Consolidación vigente del 14/11:** [§60](60_calibracion_temporal_condicional.md) añade políticas temporales y cobertura condicional; [§61](61_reproduccion_consolidada.md), auditoría de entrega y Apéndice B; [§62](62_cierre_literatura_focal.md), decisiones bibliográficas focales; [§63](63_ensamble_calibracion.md), referencia empírica de ensemble. Los resultados adversos y límites prospectivos se conservan.

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
