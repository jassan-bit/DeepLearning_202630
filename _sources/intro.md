# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · GRU–LSTM–TCN, CV de Filipe y modelo final D7 · recuperación de literatura, curvas reproducidas y margen académico del 6 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para pronosticar volatilidad de **Bitcoin**, con datos oficiales de **Binance Spot BTCUSDT**. El problema es **tipo 4: regresión temporal**: cinco covariables OHLCV diarias, desviación estándar poblacional de siete retornos logarítmicos y horizonte de un día. Se conserva la validación **`split_train_val_forwardChaining` de `timeseries-cv`**, proyecto de DidierRLopes con coautoría de **Filipe R. Ramos**, con embargo y transformaciones ajustadas únicamente al pasado permitido.

El estudio tiene **dos fases identificadas por separado**. La fase I evalúa las cuatro entradas D=7/14/21/28 con diez semillas y un test histórico ya observado. La fase II mantiene **D7, n=7 y h=1**, registra una búsqueda equitativa, ejecuta un piloto antes de su ajuste definitivo y evalúa una reserva histórica distinta con métodos fijados previamente.

## Qué contiene el libro

- [Problema y dataset](sections/01_problema_dataset.md): formulación, fuente, reloj de información y definición de σ₇.
- [Análisis exploratorio](sections/02_eda.md): calidad, estructura temporal y decisiones derivadas del desarrollo 2023–2024.
- [Estado del arte](sections/03_estado_arte.md): extracción de GRU–LSTM–TCN y límites de recurrencia bibliográfica.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): arquitecturas, cuatro entradas iniciales, purgas y alcance de las dos fases.
- [Benchmark inicial](sections/05_benchmark.md): tablas de CV y del test histórico, baseline e inferencia exploratoria.
- [Procedimiento temporal](sections/06_procedimiento.md): ruta 6.4 de la guía y disponibilidad causal.
- [Limitaciones](sections/07_limitaciones_entregables.md): potencia entre semillas, dependencia y fuerza inferencial.
- [Entregables](sections/08_entregables.md): siete productos y ocho descargas directas.
- [Modelo final y evaluación D7](sections/09_estudio_planificado.md): HPO comparable, piloto, dimensionamiento y evaluación preespecificada.
- [Decisiones y pendientes](sections/protocolo_pendiente.md): evidencia de cierre y límites aún abiertos.

## Modelo final y evaluación D7

El [diseño y código](https://github.com/jassan-bit/DeepLearning_202630/commit/ac7dd7f67c0dec2397b519897cf6da8056b238f5) se publicaron **antes del primer entrenamiento** de esta fase. La búsqueda utiliza **ocho configuraciones y dos semillas por familia**, los mismos diez bloques mensuales de enero–octubre de 2024 y **320 ajustes por familia, 960 en total**. La puntuación es la media entre semillas del RMSE sobre las 305 fechas de validación concatenadas.

| Modelo | Receta elegida con desarrollo | Parámetros | RMSE enero–octubre, media ± DE |
|---|---|---:|---:|
| **GRU/D7** | Una capa de 16 unidades, LR 0,003, dropout 0,1 | **1.121** | **0,00513620 ± 0,00003068** |
| LSTM/D7 | Una capa de 16 unidades, LR 0,003, dropout 0 | 1.489 | 0,00626389 ± 0,00021402 |
| TCN/D7 | Tres bloques causales, ancho 8, LR 0,003, dropout 0 | 1.185 | 0,00543698 ± 0,00056998 |

El baseline final es **GRU/D7 de 16 unidades**, seleccionado y congelado por desarrollo antes del piloto y de la reserva. Es una receta distinta de la GRU de 32 unidades de fase I.

El piloto usa **diez semillas nuevas por familia**, treinta trayectorias y sesenta ajustes, sobre el **01/11–30/12/2024** con pesos fijos. Las diferencias pareadas de RMSE y su incertidumbre dimensionan **465 ejecuciones por familia** mediante t no central, α*=0,05/3 y potencia marginal objetivo 0,80. La planificación quedó [publicada antes de los ajustes definitivos](https://github.com/jassan-bit/DeepLearning_202630/commit/5a0acfbc2ff582e252f0e6fb18aaf652963221f8). Sus supuestos y MDE se desarrollan en el [capítulo 9](sections/09_estudio_planificado.md).

La nueva reserva comprende **01/09–30/10/2026, 60 días**. Los archivos ya habían sido auditados por integridad, timestamps y esquema; según los registros conservados, no se habían utilizado para construir objetivos o evaluar resultados. Esta fase es una evaluación preespecificada de una reserva histórica, no predicciones emitidas en vivo ni evidencia de otro mercado. La potencia planificada entre semillas **no acredita potencia temporal del 80 %** sobre esos 60 días.

### Resultados de la nueva reserva

| Modelo final | RMSE, media ± DE | R² medio | MAE medio |
|---|---:|---:|---:|
| GRU | 0,00511696 ± 0,00077519 | 0,348891 | 0,00391028 |
| LSTM | 0,00565123 ± 0,00056569 | 0,215843 | 0,00429363 |
| TCN | 0,00433967 ± 0,00099896 | 0,517956 | 0,00338399 |
| Persistencia | 0,00384478 | 0,640632 | 0,00212123 |

Las redes resumen 465 semillas sobre las mismas sesenta fechas; persistencia es determinista. Los [contrastes e intervalos de fase II](sections/09_estudio_planificado.md) distinguen aleatoriedad de entrenamiento e incertidumbre temporal.

En esta reserva, **persistencia tiene menor RMSE media que las tres redes y TCN el menor error entre las redes**. El baseline operativo permanece GRU/D7-c03, fijado por desarrollo. Las diferencias entre semillas no equivalen a superioridad temporal: ninguna comparación DM rechaza tras Holm-6 y el bootstrap temporal familiar no acredita una mejora superior al 5 %. Los diagnósticos rechazan normalidad de las diferencias; las t de 465 pares se interpretan como aproximaciones, con sus límites en el capítulo 9.

## Fase I: benchmark exploratorio conservado

La comparación inicial utiliza **D=7/14/21/28**, n=7, h=1, doce bloques mensuales de 2024 y diez semillas por red e historia. Completó **120 trayectorias, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones**. Las cuarenta trayectorias LSTM reentrenadas reprodujeron exactamente **14.640 predicciones de control**.

Tras cerrar configuraciones y épocas se realizaron **80 ajustes finales nuevos de GRU/TCN** y se reutilizaron **40 LSTM idénticos**, verificando pesos, escaladores y recetas. El test **01/01/2025–31/08/2026** contiene **608 fechas y 72.960 predicciones**: ochenta inferencias nuevas y cuarenta trayectorias LSTM conservadas. Sus veinte meses son intervalos de reporte, sin reajuste durante test.

| Modelo de fase I seleccionado con CV | RMSE CV 2024, media | RMSE test histórico, media ± DE | R² test, media |
|---|---:|---:|---:|
| **GRU/D7, 32 unidades** | **0,005155** | **0,003775 ± 0,000194** | **0,8542** |
| LSTM/D7, 32 unidades | 0,006335 | 0,005030 ± 0,000264 | 0,7411 |
| TCN/D7, ancho 16 | 0,005820 | 0,004175 ± 0,000373 | 0,8208 |

Las [tablas de CV de fase I](_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y [su test](_static/gru_lstm_tcn_20261105/metric_summary_test.csv) conservan las seis métricas y las cuatro entradas. Persistencia obtiene menor RMSE que GRU en esa CV; en su test GRU tiene menor RMSE, mientras persistencia conserva mejores MAE y MAPE.

**El test de fase I ya había sido observado.** Sus contrastes y margen son posteriores y exploratorios; el nuevo estudio no cambia esa cronología. El piloto y HPO D7 de fase II no se atribuyen a D14/21/28. Tampoco se comparan directamente métricas de 608 y 60 días para afirmar una ganancia causada por el HPO: cambian el periodo y el régimen.

## Literatura y límites científicos

La bibliografía canónica reúne **60 referencias**. La {ref}`síntesis homogénea actual <homogeneous-selection-e1>` concilia 47 identidades documentales y distingue **26 originales completos leídos y una transcripción completa**, fuentes parciales y textos no recuperados. Dentro del núcleo de volatilidad realizada cripto hay **12 estudios**, diez con comparadores básicos extraíbles: **LSTM 10/10, GRU 4/10, MLP 3/10 y TCN 1/10**. El estrato elegible por suficiencia de reporte contiene **ocho estudios**: LSTM 8/8, GRU 2/8, MLP 2/8 y TCN 1/8. N10 es la extracción descriptiva; los denominadores y exclusiones de reporte se publican; no se mezclan precios, dirección, volatilidad implícita o componentes híbridos para aumentar frecuencias.

La terna **GRU–LSTM–TCN** conserva el experimento registrado: dos reglas de actualización recurrente y una representación convolucional causal sobre la misma entrada D7. La tabla conjunta documenta recetas, parámetros y sesgos inductivos. **MLP es una alternativa más frecuente y más pequeña que TCN en la evidencia disponible**; existe también una receta histórica propia de 593 parámetros. La elección actual prioriza el contraste entre celdas y filtros y no se presenta como las tres primeras del ranking de frecuencia ni como una solución de complejidad mínima global. La guía exige justificar sesgos diferentes, sin imponer tres familias amplias distintas.

El flujo histórico conserva **244 ocurrencias y 182 grupos provisionales** de cuatro exportaciones. Los 22 casos que requirieron recuperación quedan al corte como **ocho decisiones de exclusión/contexto y 14 informes no obtenidos**. Es una disposición documentada del flujo, sin obligación de comprar esos textos ni afirmación de haberlos leído. Las {download}`alternativas de acceso gratuito <_static/literature_e1/open_access_replacement_20261107/README.md>` aportan respaldo por tema y función. El corte anterior de siete estudios y sus frecuencias permanecen como evidencia histórica, separado del corpus conciliado actual.

La nueva {ref}`bitácora nativa <current-native-execution-e1>` conserva las consultas comunicadas y las exportaciones reales: **271 resultados de Web of Science, 538 de Scopus y 32 de IEEE Xplore**. Se distinguen total de resultados, registros exportados y estudios incluidos. Los filtros difieren: Scopus incluye 2019–2026; el exportado WoS contiene años 2017–2026; IEEE muestra acceso abierto e intervalo 2018–2026. Esta ronda acredita la consulta efectiva de las tres bases con su procedencia y alcance, sin atribuirle las cadenas desconocidas de las sesiones históricas ni convertir los resultados en artículos leídos.

El margen académico δ≈0,00025772843 corresponde al **5 % del RMSE CV del GRU/D7 inicial**. Fue posterior al test de fase I y quedó fijado antes de la evaluación definitiva. Su [justificación](sections/01_problema_dataset.md) lo relaciona con el uso académico confirmado y las escalas de desarrollo: 1,60 DE del RMSE CV entre semillas y 2,64 % de la DE observada de σ₇ en 2024, con sensibilidad al 2,5 % y 10 %. Es una elección normativa documentada; no demuestra utilidad financiera, equivalencia entre modelos o generalización a otros mercados.

## Reproducibilidad

La actualización del 06/11 añade fichas de originales, comparación de familias, contexto académico del margen y curvas públicas de las recetas finales. La {ref}`reproducción de curvas <curvas-finales-reproducidas>` se registró y publicó antes de sus nuevos ajustes: **465 semillas por familia, 1.395 trayectorias y 2.790 etapas**, sólo con desarrollo 2023–2024. Es una ejecución posterior; no recupera las historias privadas originales ni puntúa de nuevo la reserva. Se publican también las diferencias respecto de los hashes históricos. El baseline, las métricas de evaluación y el margen registrado se conservan. La {ref}`auditoría numérica posterior <auditoria-numerica-e1>` añade dos ejecuciones independientes de seis celdas fijadas previamente: **18/18 coincidencias canónicas de pérdidas y pesos, con diferencia máxima cero**. Ese control acredita repetibilidad actual del subpanel; la comparación numérica con los archivos originales conserva su limitación de acceso.

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) distingue los comandos de ambas fases y la restauración desde los ZIP. Las descargas del [capítulo 8](sections/08_entregables.md) reúnen informe, código y evidencia pública. Construir HTML consume resultados guardados y no entrena redes ni abre archivos de mercado. Las observaciones y predicciones individuales, los escaladores y los checkpoints completos se distinguen de los agregados públicos conservados para revisión.

La reserva de fase II pasa a ser una evaluación conocida una vez publicados sus resultados. Reproducirla no recupera independencia para un futuro modelo diseñado con esa información.
