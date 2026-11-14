# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

**Evaluación final y experimentos complementarios completados. Cierre A100 verificado: 13/11/2026, 22:42:55 de Bogotá; suplementos estadísticos y documentales del 13/11, posteriores a las 23:17.** La tabla A2 reúne **16 sistemas sobre las mismas 608 fechas**. El plan complementario cerró **99/99 ajustes: 81 reproducciones y 18 ablaciones**, con todas las comprobaciones de reproducción aprobadas y sin reservas abiertas. Se conservan los resultados originales y los respaldos en **GitHub**.

**[Resultados finales (§6)](sections/06_benchmark_final.md)** · **[Tabla A2 medida, CSV](_static/e2_benchmark_final_20261113/benchmark_A2.csv)** · **[Reproducción y ablaciones (§55)](sections/55_reproduccion_ablaciones.md)** · **[Estado y presupuesto final](_static/e2_reproduction_ablations_20261113/status.json)**

**Resultado de reproducción:** los recibos de los 81 predictores informan **diferencia máxima absoluta cero** frente a los originales en 53 pronósticos de calibración y 608 de prueba. Los 78 estados neuronales son iguales tensor a tensor y los tres clásicos conservan sus parámetros. Es una reproducción numérica con checkout y entorno Python separados sobre la misma cronología conocida.

**Resultado de ablaciones:** las tres variantes del autoencoder se entrenaron con seis semillas cada una. Retirar el dropout supervisado aumentó la RMSE un **13,90 %**, con **p Holm=0,00059994**. Las otras dos diferencias no se rechazan tras Holm; eso no prueba equivalencia. Los análisis son posteriores y mantienen las recetas y la A2 originales.

**Otros suplementos completados:** [ajuste y calibración (§49)](sections/49_diagnosticos_f2a.md) · [DM/MCS y rangos (§50)](sections/50_comparaciones_f2a.md) · [costes y Pareto (§51)](sections/51_eficiencia_costes.md) · [síntesis bibliográfica (§52)](sections/52_sintesis_literatura.md) · [estado de requisitos (§53)](sections/53_cierre_requisitos.md) · [robustez y explicaciones del Top 3 (§54)](sections/54_robustez_interpretabilidad.md). Los diagnósticos incluyen **79 predictores, 468 filas de robustez y 78 perfiles de inferencia A100**.

**Suplementos finales:** [20.000 remuestreos y Giacomini–Rossi (§56)](sections/56_inferencia_estabilidad_final.md) · [6.080 etiquetas de errores y 100 fechas por modelo (§57)](sections/57_errores_techo.md) · [FLOPs core de las trece redes (§58)](sections/58_flops_predictores.md) · [conciliación de 9.559 grupos bibliográficos (§59)](sections/59_auditoria_literatura_final.md). Son análisis posteriores con fecha y alcance propios, sin nuevas corridas ni sustitución de A2.

**Cierre documental:** [calibración temporal y cobertura condicional (§60)](sections/60_calibracion_temporal_condicional.md) · [entrega reproducible y Apéndice B actual (§61)](sections/61_reproduccion_consolidada.md) · [elegibilidad focal y P contextual (§62)](sections/62_cierre_literatura_focal.md) · [referencia empírica de ensemble (§63)](sections/63_ensamble_calibracion.md). La cobertura global se acerca al 90 % con ACI, aunque persiste infracobertura en rangos altos. El ensemble proporciona una referencia puntual más baja, con diferencias no concluyentes tras Holm.

**Estado bibliográfico vigente:** [§62](sections/62_cierre_literatura_focal.md) incorpora los cuatro originales antes no recuperados, **17 inclusiones aplicadas, once exclusiones y dos contextos adjudicados**, con cero accesos/revisiones focales pendientes. Publica calidad, comparación por estratos y límites de prioridad global. Las cadenas originales se recuperaron de GitHub; el flujo PRISMA global sigue incompleto por exportaciones privadas y cribado. Datos/código tienen publicación pública identificada para 9/9 fuentes del catálogo.

## Resumen

Se estudia el pronóstico diario de volatilidad de Bitcoin usando siete días de OHLCV, horizonte de un día y objetivo σ₇ poblacional de retornos de Close, sin anualización. Se comparan diez candidatos de cuatro familias, tres referencias neuronales de E1 y tres comparadores estadísticos. La selección reducida utiliza seis configuraciones y tres orígenes internos por red; cada receta neuronal final tiene seis semillas. Los 16 sistemas comparten 608 fechas de evaluación conocidas desde E1.

El LSTM-autoencoder obtiene la menor RMSE media observada, **0,00380343**, seguido de ARIMA, **0,00392162**, y GRU32, **0,00400116**. Persistencia presenta el menor MAE y sMAPE. Ninguno de los veinte contrastes primarios demuestra la mejora práctica registrada tras Holm; tampoco se establece equivalencia. La repetición de los 81 predictores reproduce sus resultados con discrepancia máxima cero. Las 18 ablaciones muestran degradación significativa al retirar el dropout supervisado; las otras dos comparaciones no son concluyentes.

Se publican residuos, calibración, comparaciones temporales, robustez, explicaciones del Top 3, perfiles A100, costes y Pareto. La evaluación es retrospectiva y el piloto no alcanza la potencia objetivo. El cierre conserva **84 horas totales autorizadas**, todos los cargos y los resultados negativos. Permanecen abiertos el cierre bibliográfico global y δg previo; el nuevo proxy empírico de ensemble tiene incertidumbre amplia y no identifica riesgo Bayes. El conteo FLOPs cubre matrices y convoluciones, con exclusiones declaradas.

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
| Reproducción numérica de los predictores | **81/81**, todas las comprobaciones aprobadas; máxima diferencia de pronósticos **0**: [§55](sections/55_reproduccion_ablaciones.md) |
| Ablaciones entrenadas del autoencoder | **18/18**, tres componentes × seis semillas; bootstrap de 10.000 réplicas y Holm: [comparaciones](_static/e2_reproduction_ablations_20261113/ablation_comparisons.csv) |
| Refinamiento numérico y estabilidad | **20.000 remuestreos** de los mismos veinte contrastes; Giacomini–Rossi y constancia separados: [§56](sections/56_inferencia_estabilidad_final.md) |
| Errores y dificultad | **6.080 etiquetas**, 100 fechas por cada modelo principal, IC temporales y entropía ordinal; γ ideal con límites: [§57](sections/57_errores_techo.md) |
| Operaciones de inferencia | FLOPs core D7×5 de **13 recetas** y Pareto, con exclusiones: [§58](sections/58_flops_predictores.md) |
| Literatura recibida | **9.559 grupos conciliados**, roles de 20 fuentes y puntajes con incógnitas explícitas: [§59](sections/59_auditoria_literatura_final.md) |
| Cierre contable | **84 h autorizadas**, **80,2561 h contabilizadas**, saldo **3 h 44 min 38 s**, sin reservas abiertas |
| Persistencia | Resultados y comprobaciones de integridad en **GitHub** |

El piloto estima una potencia condicional de **12,30 %** para una ganancia de **2δ**, por debajo del objetivo de 80 %. Con R=6, la ganancia mínima estimada para esa potencia es **0,00087867**. Son cálculos sobre cinco semillas emparejadas de dos modelos y 53 fechas de desarrollo; no garantizan potencia temporal ni para todo el Top 10. R=6 se conserva por recursos, sin ampliar repeticiones después de observar la prueba.

Completar el benchmark no acredita cumplimiento integral de la guía. La [matriz de exigencias (§38)](sections/38_cumplimiento_guia.md) distingue los resultados existentes de las condiciones pendientes. Esta actualización documental utiliza los artefactos ya guardados.

## Cómo interpretar los resultados

La métrica primaria de cada red es la **media de sus seis RMSE individuales**, no la RMSE de un ensamble. La desviación y el IC entre semillas describen variabilidad de optimización condicionada a esta cronología. Los tres comparadores deterministas aportan una trayectoria cada uno; no se presentan como seis entrenamientos independientes.

La mejora puntual del autoencoder frente a persistencia es **0,00028217 de RMSE (6,91 %)**, con IC bootstrap temporal **[−0,00016895; 0,00072419]**. Aunque la estimación supera el margen heredado δ=0,00025772842664981106, su incertidumbre no demuestra superioridad práctica. Frente a GRU32 la mejora puntual es **0,00019773**, menor que δ. Los **20 p ajustados por Holm son 1,0** en la familia primaria registrada.

El análisis original conserva bootstrap pareado en bloques de 14 días y 4.999 remuestreos. [§56](sections/56_inferencia_estabilidad_final.md) reproduce exactamente sus veinte contrastes y añade **20.000 réplicas con las mismas hipótesis y Holm**, sin cambiar la conclusión: todos los p ajustados son 1,0. Se mantienen la sensibilidad a 7 y 28 días y la fecha posterior del refinamiento. Los diagnósticos por semillas no convierten esas repeticiones en mercados independientes. Los límites de esta batería frente a lo solicitado por la guía se explican en §38. No se cambian recetas ni calibradores después de observar la prueba.

## Guía de lectura

| Contenido | Dónde revisar |
|---|---|
| Tabla A2, incertidumbre, intervalos y discusión final | **[Benchmark final (§6)](sections/06_benchmark_final.md)** |
| Ajuste, sobreajuste, residuos, calibración y estabilidad temporal | **[Diagnósticos F2a posteriores (§49)](sections/49_diagnosticos_f2a.md)** |
| Reproducción en entorno separado y tres ablaciones entrenadas | **[Cierre y diferencias observadas (§55)](sections/55_reproduccion_ablaciones.md)** |
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

El cierre complementario mantiene un límite de **84 horas acumuladas autorizadas**, con el ancla anterior de 80 horas y todos sus cargos conservados. El balance final y su alcance se documentan en [§55](sections/55_reproduccion_ablaciones.md). Las incidencias HPO y el fallo de implementación de ARIMA quedan contabilizados; el ARIMA ganador, de orden (2,0,1), se elige entre **15 de los 18 órdenes** con tres folds válidos. Se conservan dos incidencias de no convergencia y el fallo de implementación, sin repetirlos. Las correcciones de adquisición y respaldo no alteran los predictores congelados. La [congelación](_static/e2_benchmark_final_20261113/final_freeze.json) y el [análisis completo](_static/e2_benchmark_final_20261113/benchmark_analysis.json) permiten revisar ese alcance.

## Qué falta para cerrar la guía

1. Completar la elegibilidad global de la literatura, recuperar los 374 registros Scopus ausentes y resolver los componentes bibliométricos desconocidos. [§59](sections/59_auditoria_literatura_final.md) cierra la contabilidad de lo recibido y distingue el puntaje anterior de su sensibilidad contextual.
2. Conservar las desviaciones que no pueden corregirse después: δg no se definió previamente, la potencia fue insuficiente y la prueba ya se conocía desde E1. Giacomini–Rossi y el refinamiento de 20.000 réplicas están calculados en [§56](sections/56_inferencia_estabilidad_final.md); no crean un experimento prospectivo.
3. [§63](sections/63_ensamble_calibracion.md) añade una referencia empírica ajustada sólo en calibración y γ=0,28757; su incertidumbre no confirma una superioridad ni un techo óptimo. [§57](sections/57_errores_techo.md) conserva entropía ordinal y las referencias anteriores, sin convertirlas en ruido Bayes. La taxonomía de residuos es automática y posterior, con 100 fechas auditables por modelo.
4. Los FLOPs de [§58](sections/58_flops_predictores.md) cubren matrices y convoluciones, con FFT, activaciones y otras operaciones excluidas. Energía, dinero y coste total del servicio no se midieron. El [Apéndice B actual](sections/61_reproduccion_consolidada.md) mantiene estados verificables y un comando de auditoría; la defensa oral conserva su evaluación propia.

La [matriz de §38](sections/38_cumplimiento_guia.md) distingue cada evidencia de sus condiciones pendientes. El cierre de 99 ajustes corresponde al plan complementario; no es un porcentaje de cumplimiento integral de E2.

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
