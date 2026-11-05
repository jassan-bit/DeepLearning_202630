# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · GRU–LSTM–TCN, CV de Filipe y evaluación histórica · 5 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para pronosticar volatilidad de **Bitcoin**, con datos oficiales de **Binance Spot BTCUSDT**. El problema corresponde al **tipo 4: regresión temporal**: cinco covariables OHLCV diarias, desviación estándar poblacional de siete retornos logarítmicos y horizonte de un día. Las entradas **D=7,14,21,28** cambian la historia observada y mantienen **n=7 y h=1**.

La terna vigente es **GRU–LSTM–TCN**. Las tres redes reciben la misma información y usan el mismo preprocesamiento causal, las mismas diez semillas y las mismas fechas. GRU y LSTM son recurrentes unidireccionales de una capa; TCN utiliza tres bloques residuales de convoluciones causales dilatadas. Sus configuraciones son adaptaciones compactas documentadas, con 3.777, 5.025 y 4.289 parámetros, respectivamente.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación, fuente, reloj de información y definición de σ₇.
- [Análisis exploratorio de datos](sections/02_eda.md): calidad, estructura temporal, diez modalidades con sus incisos y aplicación de la función de Filipe.
- [Estado del arte](sections/03_estado_arte.md): revisión trazable, extracción de GRU–LSTM–TCN y límites de la recurrencia bibliográfica.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): arquitecturas, cuatro entradas, semillas, purgas, cierre de configuración y trazabilidad.
- [Benchmark y baseline](sections/05_benchmark.md): tablas de CV y test histórico, comparación entre las tres redes e interpretación estadística.
- [Procedimiento temporal](sections/06_procedimiento.md): ruta 6.4 de la guía, validación de Filipe y evaluación final sin reajustes durante test.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): potencia, dependencia, fuerza inferencial y requisitos pendientes.
- [Entregables del proyecto](sections/08_entregables.md): los siete productos y sus descargas directas.
- [Decisiones y pendientes](sections/protocolo_pendiente.md): matriz de cierre y trabajo aún abierto.

## Alcance del proyecto

La validación utiliza **`split_train_val_forwardChaining` de `timeseries-cv`**, proyecto de DidierRLopes con coautoría de **Filipe R. Ramos**. Sus orígenes diarios se agrupan explícitamente en doce bloques mensuales de 2024. Cada ajuste y escalador usa únicamente el pasado permitido, con embargo de D días y validación interna cronológica. Se conserva **2023 como historia inicial** y la representación relativa al último cierre observable.

La nueva CV completó **120 trayectorias, 1.440 segmentos mensuales, 2.880 ajustes y 43.920 predicciones**. Las cuarenta trayectorias LSTM se reentrenaron como control y reprodujeron exactamente **14.640 predicciones históricas**. Las cuatro entradas se evaluaron para las tres redes; la media de RMSE de CV seleccionó **D=7 en cada familia y GRU/D7 como baseline neuronal**, antes de acceder al contenido del test en este ciclo.

Tras congelar configuraciones y épocas se realizaron **80 ajustes finales nuevos de GRU/TCN** y se reutilizaron **40 LSTM históricos idénticos**, verificando pesos, escaladores y recetas: **120 checkpoints finales**. La evaluación cubre **01/01/2025–31/08/2026**, con **608 fechas y 72.960 predicciones**. Se efectuaron 80 inferencias nuevas y se conservaron 40 trayectorias LSTM firmadas. Sus veinte meses son intervalos de reporte de pesos finales fijos. Septiembre–octubre de 2026 permanece sin asignar.

| Modelo seleccionado con CV | RMSE CV 2024, media | RMSE test histórico, media ± DE | R² test, media |
|---|---:|---:|---:|
| **GRU/D7** | **0,005155** | **0,003775 ± 0,000194** | **0,8542** |
| LSTM/D7 | 0,006335 | 0,005030 ± 0,000264 | 0,7411 |
| TCN/D7 | 0,005820 | 0,004175 ± 0,000373 | 0,8208 |

Las [tablas completas de CV](_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y [test](_static/gru_lstm_tcn_20261105/metric_summary_test.csv) conservan las seis métricas, las cuatro entradas, dispersiones, cobertura e intervalos. **Persistencia obtiene menor RMSE que GRU en CV; en test GRU tiene menor RMSE, mientras persistencia conserva mejores MAE y MAPE.** La selección se conserva por CV y no cambia con el ranking del test.

## Evidencia y límites

GRU, LSTM y TCN aparecen como comparadores independientes de volatilidad de Bitcoin en Akbiyik et al. (2023); Miura et al. (2019) también evalúa GRU y LSTM {cite:p}`akbiyik2023,miura2019`. Las frecuencias estrictas del corpus son **GRU 3/7, LSTM 5/7 y TCN 1/7**; incluyen publicaciones y manuscritos con niveles de verificación distintos. La limitada recurrencia de TCN y la justificación de no redundancia entre las dos recurrentes permanecen explícitas en el capítulo 3.

**El test histórico ya había sido observado en el proyecto.** Cerrar el nuevo ciclo antes de puntuar sus modelos conserva su trazabilidad, pero no crea un holdout desconocido ni confirmación temporal independiente. Los nuevos contrastes entre GRU–LSTM–TCN son **post-hoc y exploratorios**. Las semillas describen variación del entrenamiento condicionada a las fechas; no son otros mercados.

Las diez semillas constituyen un presupuesto fijo: **no se ejecutó un piloto prospectivo y dimensionamiento específico de la nueva terna**, ni se acredita potencia 0,80. El [estado de cierre](sections/protocolo_pendiente.md) conserva ese pendiente. Para un modelo avanzado futuro se propone un margen operativo académico del **5 % frente a GRU/D7 y persistencia**, definido después del test actual y sujeto a su propio protocolo, piloto y evaluación independiente. No se afirma utilidad financiera ni equivalencia.

## Reproducibilidad y privacidad

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) describe la construcción del libro y la reproducción. Las [evidencias vigentes](_static/gru_lstm_tcn_20261105/summary.json) y las descargas del [capítulo 8](sections/08_entregables.md) reúnen métricas, código y procedencia. El libro utiliza resultados guardados: construir HTML no entrena redes ni abre brutos. Los datos individuales, pesos y predicciones completos se distinguen de los resúmenes públicos conforme a la política de entrega.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
