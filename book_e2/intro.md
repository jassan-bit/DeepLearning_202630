# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

**Evaluación final completada. Corte verificado: 13/11/2026, 11:58:36 de Bogotá.** Se evaluaron los **16 sistemas en las mismas 608 fechas**. Están guardados los **78 ajustes finales neuronales**, los **10 pilotos**, los tres comparadores y la congelación de **81 predictores**, con pesos, escaladores y calibradores. La ejecución terminó con código de salida **0** y respaldo verificado en **GitHub**.

**[Resultados finales y discusión (§6)](sections/06_benchmark_final.md)** · **[Tabla A2 medida, CSV](_static/e2_benchmark_final_20261113/benchmark_A2.csv)** · **[20 contrastes primarios](_static/e2_benchmark_final_20261113/primary_contrasts.csv)** · **[Fuentes y hashes](_static/e2_benchmark_final_20261113/source_evidence.json)**

**Diagnósticos F2a añadidos el 13/11/2026:** [§49](sections/49_diagnosticos_f2a.md) documenta ajuste, brechas observadas, residuos, calibración y estabilidad temporal sobre los pronósticos guardados. La familia exploratoria tiene **826 contrastes y 707 rechazos tras Holm**. Se detecta dependencia residual en las 81 trayectorias; los 31 rechazos de cobertura son por sobrecobertura. No se acredita ausencia de sobreajuste ni estabilidad por falta de rechazo. Los resultados primarios se conservan.

**Suplementos de cierre:** [Comparaciones DM/MCS y rangos (§50)](sections/50_comparaciones_f2a.md) · [Costes medidos y Pareto (§51)](sections/51_eficiencia_costes.md) · [Síntesis bibliográfica delimitada (§52)](sections/52_sintesis_literatura.md) · [Ejecuciones complementarias y límites (§53)](sections/53_cierre_requisitos.md). Las tres primeras aportan resultados. La robustez está en ejecución con recibos individuales de GitHub; su agregado, la reproducción y las ablaciones todavía no se declaran completados.

## Resumen

Se estudia el pronóstico diario de volatilidad de Bitcoin a partir de siete días de OHLCV, con horizonte de un día y objetivo σ₇ poblacional de retornos de Close, sin anualización. El benchmark reúne diez candidatos de cuatro familias, tres referencias neuronales de E1 y tres comparadores estadísticos. La selección reducida usa seis configuraciones y tres orígenes internos por red; cada receta neuronal final se repite con seis semillas. Los 16 sistemas se evalúan sobre las mismas 608 fechas conocidas de E1, manteniendo congelados los pesos, escaladores, calibradores y el análisis.

El LSTM-autoencoder con predictor obtiene el menor RMSE medio observado, **0,00380343**, seguido de ARIMA, **0,00392162**, y GRU32, **0,00400116**. Persistencia obtiene el menor MAE y sMAPE. Ninguno de los 20 contrastes primarios demuestra la mejora práctica registrada con corrección de Holm; esto tampoco establece equivalencia. Los resultados describen una evaluación retrospectiva del calendario heredado y no una prueba prospectiva nueva. El piloto no alcanza la potencia objetivo para el efecto planificado. El informe conserva el presupuesto acumulado de 80 horas, los fallos y sus cargos. Quedan requisitos de robustez, explicabilidad, ablaciones, eficiencia, diagnóstico estadístico y reproducción por cerrar o verificar.

## Qué quedó completado

| Etapa | Resultado verificado |
|---|---|
| Selección de trece redes | **13 recetas cerradas**; 231 éxitos HPO de 234 oportunidades, dos fallos y un desenlace desconocido conservados |
| Ajustes finales de diez candidatos y tres referencias | **78/78**, seis semillas por red |
| Pilotos de variabilidad y potencia | **10/10**, cinco semillas de iTransformer y cinco de GRU32; [informe medido](_static/e2_benchmark_final_20261113/pilot_power_MDE.json) |
| Comparadores y congelación | **3/3 comparadores**, 81 predictores congelados antes de admitir datos finales |
| Evaluación y análisis registrados | **16/16 sistemas, 608/608 fechas**, tabla A2 y 20 contrastes primarios |
| Auditoría F2a posterior | **81 trayectorias**, 826 contrastes exploratorios, curvas de los 78 ajustes finales y 29 de 39 ajustes HPO seleccionados; [resultados y límites (§49)](sections/49_diagnosticos_f2a.md) |
| Comparaciones F2a posteriores | DM–HLN/HAC14, MCS y rangos condicionados: [§50](sections/50_comparaciones_f2a.md) |
| Eficiencia observada | 78 ajustes finales: 5.765,09 segundos de fases instrumentadas, parámetros y Pareto: [§51](sections/51_eficiencia_costes.md) |
| Persistencia | Resultados y comprobaciones de integridad en **GitHub** |

El piloto estima una potencia condicional de **12,30 %** para una ganancia de **2δ**, por debajo del objetivo de 80 %. Con R=6, la ganancia mínima estimada para esa potencia es **0,00087867**. Son cálculos sobre cinco semillas emparejadas de dos modelos y 53 fechas de desarrollo; no garantizan potencia temporal ni para todo el Top 10. R=6 se conserva por recursos, sin ampliar repeticiones después de observar la prueba.

Completar el benchmark no acredita cumplimiento integral de la guía. La [matriz de exigencias (§38)](sections/38_cumplimiento_guia.md) distingue los resultados existentes de las condiciones pendientes. Esta actualización documental utiliza los artefactos ya guardados.

## Cómo interpretar los resultados

La métrica primaria de cada red es la **media de sus seis RMSE individuales**, no la RMSE de un ensamble. La desviación y el IC entre semillas describen variabilidad de optimización condicionada a esta cronología. Los tres comparadores deterministas aportan una trayectoria cada uno; no se presentan como seis entrenamientos independientes.

La mejora puntual del autoencoder frente a persistencia es **0,00028217 de RMSE (6,91 %)**, con IC bootstrap temporal **[−0,00016895; 0,00072419]**. Aunque la estimación supera el margen heredado δ=0,00025772842664981106, su incertidumbre no demuestra superioridad práctica. Frente a GRU32 la mejora puntual es **0,00019773**, menor que δ. Los **20 p ajustados por Holm son 1,0** en la familia primaria registrada.

El análisis usa bootstrap pareado en bloques de 14 días, 4.999 remuestreos y sensibilidad diagnóstica a 7 y 28 días. Los diagnósticos por semillas no convierten esas repeticiones en mercados independientes. Los límites de esta batería frente a lo solicitado por la guía se explican en §38. No se cambian recetas ni calibradores después de observar la prueba.

## Guía de lectura

| Contenido | Dónde revisar |
|---|---|
| Tabla A2, incertidumbre, intervalos y discusión final | **[Benchmark final (§6)](sections/06_benchmark_final.md)** |
| Ajuste, sobreajuste, residuos, calibración y estabilidad temporal | **[Diagnósticos F2a posteriores (§49)](sections/49_diagnosticos_f2a.md)** |
| Informe consolidado y antecedentes del cierre documental | [§47](sections/47_entrega_consolidada.md) |
| Problema, hipótesis, datos y contrato heredado | [§1](sections/01_punto_partida.md) · [§17](sections/17_contrato_sincronizado.md) |
| EDA y auditoría temporal | [§2](sections/02_eda.md) · [§10](sections/10_auditoria_datos.md) |
| Literatura, catálogo y cuerpos publicados | [§3](sections/03_literatura.md) · [§43](sections/43_catalogo_obligatorio.md) · [§44](sections/44_arquitecturas_publicadas.md) |
| Protocolo reducido, ejecución y desviaciones | [§48](sections/48_protocolo_reducido.md) |
| Réplica TimeMixer en ETTh1 y límites de transferencia | [§5](sections/05_paper_guia.md) |
| Requisitos aún pendientes y reproducción | **[Matriz de la guía (§38)](sections/38_cumplimiento_guia.md)** · [§8](sections/08_reproducibilidad.md) |

## Catálogo y alcance temporal

Los diez candidatos son **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**. Las referencias son **MLP16, LSTM32 y GRU32**; los comparadores son **Persistence, SeasonalNaive y ARIMA**. El híbrido adicional y Ridge conservan alcance opcional e histórico.

La selección usa enero, julio y octubre de 2024; el ajuste final común termina el **31/10/2024**, seguido de embargo y calibración separada. La evaluación comprende **01/01/2025–31/08/2026**, el calendario heredado ya conocido en E1. Su interpretación es retrospectiva. La [admisión de datos](_static/e2_benchmark_final_20261113/data_admission.json) documenta las veinte fuentes originales verificadas, el puente causal de diciembre y el soporte común sin comprimir fechas.

La campaña mantiene sus **80 horas acumuladas autorizadas**, sin reiniciar cargos. Las incidencias HPO y el fallo de implementación de ARIMA quedan contabilizados; el ARIMA ganador, de orden (2,0,1), se elige entre **15 de los 18 órdenes** con tres folds válidos. Se conservan dos incidencias de no convergencia y el fallo de implementación, sin repetirlos. Las correcciones de adquisición y respaldo no alteran los predictores congelados. La [congelación](_static/e2_benchmark_final_20261113/final_freeze.json) y el [análisis completo](_static/e2_benchmark_final_20261113/benchmark_analysis.json) permiten revisar ese alcance.

## Qué falta para cerrar la guía

1. Conservar el alcance exploratorio de las comparaciones F2a ya ejecutadas en §50 y de los diagnósticos S7 de §49. La ausencia de un margen δg anterior al experimento no puede corregirse retrospectivamente; los elementos no estimables deben permanecer declarados.
2. Acreditar robustez con severidades, explicabilidad con fidelidad y estabilidad, y ablaciones de componentes bajo las condiciones de la guía.
3. Completar los perfiles de inferencia que no están en los recibos de costes de §51, el cierre global de literatura que excede el corpus delimitado de §52 y la reproducción independiente registrada en §53.
4. Revisar discusión, conclusiones, apéndices y enlaces de entrega. Los requisitos sin evidencia deben permanecer declarados; las métricas de prueba no se usan para volver a seleccionar modelos.

La [matriz de §38](sections/38_cumplimiento_guia.md) detalla las prioridades y qué puede aprovechar los artefactos actuales. El resultado nulo de los contrastes se conserva en las conclusiones.

## Cortes históricos y descargas conservadas

Los cortes anteriores mantienen sus fechas y archivos. Sus conteos parciales y paquetes no sustituyen a la tabla final del 13/11. Las campañas históricas conservaron 1.705 resultados optimizados y 1.802 registros previos: son antecedentes de selección, no nuevas evaluaciones finales. Los protocolos anteriores se consultan en [§40](sections/40_calendario_original.md), [§41](sections/41_avance_optimizacion_11_origenes.md), [§45](sections/45_enmienda_protocolo.md) y [§46](sections/46_artefactos_y_pilotos.md).

- [Instantánea y procedencia](_static/e2_progress_final_20261113_v2/progress_snapshot.json)
- [Avance de las trece redes, CSV](_static/e2_progress_final_20261113_v2/neural_progress.csv)
- [Descargar el paquete de entrega](_static/delivery_20261112/Entregable2_evidencia_disponible.zip)
- [Fuentes y hashes](_static/e2_progress_final_20261113_v2/source_evidence.json)
- [Manifiesto](_static/e2_progress_final_20261113_v2/manifest.json)
- [corte anterior del 13/11, 07:24](_static/e2_progress_final_20261113/progress_snapshot.json)
- [tabla de trece redes](_static/e2_progress_final_20261113/neural_progress.csv)
- [fuentes](_static/e2_progress_final_20261113/source_evidence.json)
- [manifiesto](_static/e2_progress_final_20261113/manifest.json)
- [corte anterior del 12/11, 22:22](_static/e2_progress_20261113/progress_snapshot.json)
- [Top 10, CSV](_static/e2_progress_20261113/top10_training_validation.csv)
- [trece redes, CSV](_static/e2_progress_20261113/all13_hpo_progress.csv)
- [fuentes](_static/e2_progress_20261113/source_evidence.json)
- [manifiesto](_static/e2_progress_20261113/manifest.json)
- [instantánea](_static/delivery_20261112/selection_snapshot.json)
- [CSV de ensayos](_static/delivery_20261112/trials.csv)
