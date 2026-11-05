# 8. Entregables: productos y evidencias

```{raw} html
<div class="admonition note" id="descargar-entregable-1">
  <p class="admonition-title">Descargas del Entregable 1 · GRU–LSTM–TCN</p>
  <p><a class="btn btn-primary btn-lg" href="../_static/e1_downloads/entregable1_completo.zip" download="entregable1_completo.zip">Descargar entrega completa · ZIP</a></p>
  <p>Los siete productos corresponden a la terna vigente GRU–LSTM–TCN. También pueden descargarse individualmente.</p>
</div>
```

Esta unidad reúne el informe, el código, las instrucciones de datos y las evidencias de **GRU, LSTM y TCN**. El baseline neuronal vigente es **GRU con siete días de entrada**, seleccionado mediante la validación de 2024 antes de puntuar los nuevos modelos en el test histórico. La actualización no cambia los resultados originales ni convierte el test ya observado en una evaluación independiente.

Los botones descargan directamente desde el JBook. El [catálogo](../_static/e1_downloads/catalogo.json) y las [sumas SHA-256](../_static/e1_downloads/SHA256SUMS.txt) identifican los archivos de esta versión.

| Ítem | Producto | Estado y alcance | Descarga directa |
|---|---|---|---|
| {ref}`8.1 <entrega-8-1>` | Informe en Jupyter Book | Capítulos y figuras de GRU–LSTM–TCN | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_1_informe_html.zip" download="8_1_informe_html.zip">Descargar ZIP</a> |
| {ref}`8.2 <entrega-8-2>` | Código EDA, modelos y estadística | Código versionado; reproducción en una salida nueva y verificación de evidencias guardadas | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_2_codigo.zip" download="8_2_codigo.zip">Descargar ZIP</a> |
| {ref}`8.3 <entrega-8-3>` | Dataset: instrucciones y scripts | Fuente oficial, calendario, descargador y manifiestos | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_3_dataset_instrucciones.zip" download="8_3_dataset_instrucciones.zip">Descargar ZIP</a> |
| {ref}`8.4 <entrega-8-4>` | Síntesis de literatura y flujo | Siete estudios cripto; selección GRU–LSTM–TCN documentada con límites de recurrencia | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_4_literatura.zip" download="8_4_literatura.zip">Descargar ZIP</a> |
| {ref}`8.5 <entrega-8-5>` | Benchmark, baseline e inferencia | Cuatro entradas, diez semillas, CV y test histórico; inferencia exploratoria | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_5_benchmark.zip" download="8_5_benchmark.zip">Descargar ZIP</a> |
| {ref}`8.6 <entrega-8-6>` | Dependencias, semillas y configuraciones | Locks, recetas y catálogos de 1.440 segmentos CV y 120 modelos finales | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_6_configuraciones.zip" download="8_6_configuraciones.zip">Descargar ZIP</a> |
| {ref}`8.7 <entrega-8-7>` | Métrica, margen, independencia y partición | Cronología explícita; margen futuro posterior al test; piloto del trío pendiente | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_7_declaracion.zip" download="8_7_declaracion.zip">Descargar ZIP</a> |

Descomprime los archivos antes de abrirlos. El informe HTML se inicia en `informe/intro.html`; si el navegador restringe recursos locales, ejecuta `python -m http.server 8000 --directory informe` y abre `http://localhost:8000`. MathJax y los vínculos a fuentes externas pueden requerir internet. Los ZIP públicos contienen código, metadatos, resultados agregados y figuras. Una ilustración temporal muestra observado/predicho para la semilla fija4101; no se distribuyen sus vectores tabulares. Las series diarias descargables, los escaladores y los checkpoints completos permanecen en la evidencia conservada.

La [declaración de esta entrega](../_static/gru_lstm_tcn_delivery_20261105/declaration_and_chronology.json) y el [manifiesto de productos](../_static/gru_lstm_tcn_delivery_20261105/delivery_manifest.json) distinguen disponibilidad de archivos de cierre metodológico. Los pendientes se conservan en la [matriz de cierre](protocolo_pendiente.md).

(entrega-8-1)=
## 8.1. Informe en Jupyter Book

El informe desarrolla problema y datos, EDA, literatura, implementación, benchmark, procedimiento, limitaciones y entregables de la terna vigente.

| Componente | Ubicación |
|---|---|
| Problema y definición de σ₇ | [Capítulo 1](01_problema_dataset.md) |
| EDA y estructura temporal | [Capítulo 2](02_eda.md) |
| Selección bibliográfica de GRU, LSTM y TCN | [Capítulo 3](03_estado_arte.md) |
| Implementación y evaluación | [Capítulo 4](04_implementacion_evaluacion.md) |
| Baseline, métricas e inferencia | [Capítulo 5](05_benchmark.md) |
| Procedimiento y limitaciones | [Capítulo 6](06_procedimiento.md) y [capítulo 7](07_limitaciones_entregables.md) |

La construcción HTML consume evidencia guardada y no ejecuta entrenamientos ni inferencias. Las tablas principales muestran únicamente GRU, LSTM, TCN y persistencia.

(entrega-8-2)=
## 8.2. Código completo, ejecutable y versionado

El [repositorio](https://github.com/jassan-bit/DeepLearning_202630) contiene el EDA y el flujo de reproducción de la terna actual. El adaptador público se incorpora después de la ejecución local: su hash no se atribuye retroactivamente al código que entrenó los resultados. La [procedencia](../_static/gru_lstm_tcn_20261105/summary.json) conserva ese alcance y los hashes de las fuentes de los agregados.

| Función | Código vigente |
|---|---|
| Adquisición oficial | [download_binance_spot.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/download_binance_spot.py) |
| GRU, LSTM y TCN | [dl_models_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models_gru_lstm_tcn.py) |
| Registro, CV y fases finales en una salida nueva | [run_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn.py) |
| Agregación y estadística de la terna | [build_gru_lstm_tcn_evidence.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/build_gru_lstm_tcn_evidence.py) |
| Estadística de una reproducción nueva | [analyze_gru_lstm_tcn_run.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_gru_lstm_tcn_run.py) |
| Diagnósticos temporales | [gru_lstm_tcn_diagnostics.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/gru_lstm_tcn_diagnostics.py) |

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) explica los entornos, las entradas y los comandos. Una nueva reproducción debe escribir en otro directorio; no se sobreescriben protocolos, curvas ni resultados cerrados.

La CV registrada contiene **120 trayectorias nuevas, 1.440 segmentos mensuales, 2.880 ajustes y 43.920 predicciones**. Las cuarenta trayectorias LSTM reprodujeron exactamente sus 14.640 predicciones de control. La fase final comprende **80 ajustes nuevos de GRU/TCN y 40 checkpoints LSTM idénticos reutilizados**: 120 modelos finales, con sus pesos cerrados antes de acceder al contenido del test en este procedimiento. En test se guardaron 72.960 predicciones, con 80 inferencias nuevas y cuarenta trayectorias LSTM reutilizadas.

(entrega-8-3)=
## 8.3. Dataset, instrucciones y scripts de obtención

La fuente es Binance Vision, BTCUSDT spot, OHLCV de un minuto. Las instrucciones exactas están en [data/README.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md); el descargador verifica los archivos y sus checksums. No se adquieren otros activos para este experimento por ampliar la revisión bibliográfica a otras criptomonedas.

| Periodo UTC | Función | Regla |
|---|---|---|
| 01/01/2023–31/12/2023 | Historia inicial | Orden cronológico |
| 01/01/2024–31/12/2024 | Validación forward chaining | Doce bloques mensuales de orígenes diarios, embargo D y validación interna con pasado permitido |
| 01/01/2025–31/08/2026 | Test histórico ya observado | 608 fechas, pesos finales fijos, sin reajuste durante test |
| Septiembre–octubre de 2026 | Sin asignación automática | No se abrió ni se incorporó al experimento vigente |

Se agregan días UTC completos y se conserva el calendario, incluyendo sus faltantes. La etiqueta es la desviación estándar poblacional de siete retornos logarítmicos diarios, **σ₇ con ddof=0**, sin anualización. Las entradas D=7/14/21/28 terminan antes del día objetivo, con horizonte de un día. La imputación y el acceso a información futura no forman parte del pipeline.

Los datos brutos no se incluyen en los ZIP públicos. El paquete 8.3 contiene las instrucciones, el script y los manifiestos necesarios para identificar su obtención.

(entrega-8-4)=
## 8.4. Síntesis de literatura y flujo de búsqueda

La revisión mantiene **40 referencias y siete estudios cripto con texto completo**. Todos incluyen Bitcoin y uno también otras diecinueve criptomonedas. Los subconjuntos se solapan; no se suman como grupos independientes. Artículos publicados, manuscritos y evidencia de código conservan su nivel de verificación.

| Modelo actual | Comparadores independientes en el corpus de siete | Respaldo publicado directo |
|---|---:|---|
| GRU | 3/7 | Akbiyik et al. (2023), sección 5 y Tabla 2; Miura et al. (2019), Tabla 1 |
| LSTM | 5/7 | Akbiyik et al. (2023); Miura et al. (2019) |
| TCN | 1/7 | Akbiyik et al. (2023), TCN independiente; D-TCN es su propuesta avanzada |

Los tres aparecen como comparadores independientes de volatilidad de Bitcoin en WSDM/ACM {cite:p}`akbiyik2023`; GRU y LSTM también se evalúan en ISNN/Springer LNCS {cite:p}`miura2019`. Se distingue publicación en actas de congreso de revista científica, y no se infiere indexación individual o cuartil sólo a partir de la editorial.

La síntesis, la búsqueda y el flujo están en el [capítulo 3](03_estado_arte.md) y en el ZIP 8.4. Las [frecuencias del corpus](../_static/literature_e1/crypto_extension_20261105/family_frequencies.csv) conservan todos los modelos realmente extraídos de los originales; esas menciones bibliográficas no los convierten en modelos propios de este benchmark.

**Pendiente bibliográfico:** ampliar el respaldo independiente de TCN y justificar la inclusión de dos variantes recurrentes. Las recetas propias son adaptaciones compactas, no réplicas exactas de los artículos. La selección actual no se presenta como anterior al benchmark histórico.

(entrega-8-5)=
## 8.5. Tabla comparativa, baseline e inferencia estadística

El baseline neuronal es **GRU/D7**, elegido por su menor RMSE medio en CV 2024: **0,00515457 ± 0,00016156**. Esa elección quedó cerrada antes de las nuevas inferencias de test. Se reportan las cuatro entradas y todas las semillas, aunque tengan peor resultado.

La siguiente tabla resume el **test histórico, 01/01/2025–31/08/2026**, con la entrada de siete días seleccionada por CV:

| Modelo | RMSE, media ± DE | R², media ± DE | MAE medio | MAPE medio (%) |
|---|---|---|---:|---:|
| **GRU** | **0,00377511 ± 0,00019410** | **0,854173 ± 0,015331** | **0,00279851** | **18,0667** |
| LSTM | 0,00502979 ± 0,00026367 | 0,741108 ± 0,026619 | 0,00391888 | 25,3406 |
| TCN | 0,00417465 ± 0,00037283 | 0,820819 ± 0,032191 | 0,00314430 | 19,8847 |
| Persistencia | 0,00408560 | 0,829605 | 0,00235316 | 13,7553 |

Las redes resumen diez semillas; persistencia es determinista y no tiene desviación de inicialización. La tabla completa con precisión guardada está en [métricas de test](../_static/gru_lstm_tcn_20261105/metric_summary_test.csv), junto con [métricas CV](../_static/gru_lstm_tcn_20261105/metric_summary_cv.csv), [selección por CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) y [resumen de ejecución](../_static/gru_lstm_tcn_20261105/summary.json).

La comparación entre los tres modelos actuales se desarrolla en el [capítulo 5](05_benchmark.md). Los contrastes dentro de la nueva terna se elaboran después de puntuar el test y se identifican como **post-hoc y exploratorios**. Las [comparaciones de CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv) y [de test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv) incluyen diferencias pareadas, IC y corrección de Holm. No equivalen a un contraste confirmatorio en otro mercado.

Persistencia tiene menor RMSE en CV que las redes actuales. En test, GRU presenta menor RMSE medio, mientras persistencia conserva mejores MAE y MAPE. Los veinte meses del test son intervalos descriptivos de los mismos pesos, no veinte folds de entrenamiento.

(entrega-8-6)=
## 8.6. Dependencias, semillas y configuración de cada corrida

| Modelo | Receta fija | Parámetros entrenables |
|---|---|---:|
| GRU | Una capa unidireccional de 32 unidades; estado final; LR 0,003 | 3.777 |
| LSTM | Una capa unidireccional de 32 unidades; estado final; LR 0,003 | 5.025 |
| TCN | Tres bloques residuales, dos convoluciones causales por bloque; ancho 16; kernel 3; dilataciones 1/2/4; LR 0,001 | 4.289 |

La TCN tiene un campo receptivo de 29 días, cubre las entradas de hasta 28 días y lee la última posición observada. Las tres redes tienen salida Softplus, AdamW, dropout 0,1, batch 32, decaimiento 0,0001, clipping de norma 1, máximo de cien épocas y patience 12 sobre validación interna. No se realizó HPO adicional de la nueva terna.

Las semillas son **4101–4110**. Cada mes usa la semilla de trayectoria más 100.000 por ordinal mensual; el refit fresco suma 10.000.000. Los índices temporales son deterministas y no tienen semilla de partición. Las diez repeticiones constituyen el presupuesto de esta evaluación; no son un tamaño derivado de potencia 0,80.

Los [1.440 registros de configuración CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv) y los [120 registros finales](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv) identifican modelo, D, semilla y receta; el catálogo final distingue ajustes nuevos y reutilización. Las [especificaciones](../_static/gru_lstm_tcn_20261105/model_specs.json) y el ZIP 8.6 completan la evidencia pública sin distribuir valores aprendidos de escaladores ni pesos.

(entrega-8-7)=
## 8.7. Declaración de métrica, margen, independencia y partición

| Elemento | Declaración y alcance |
|---|---|
| Métrica primaria | RMSE sobre todas las fechas de cada trayectoria (366 en CV y 608 en test), en unidades originales de σ₇; media y DE entre semillas |
| Métricas secundarias | MAE, MSE, R², MAPE y MASE, con sus coberturas y denominadores |
| Partición | Historia inicial 2023, forward chaining 2024, ajuste final con desarrollo permitido, test histórico enero de 2025–agosto de 2026 |
| Registro de la nueva ejecución | Protocolo y hashes congelados localmente antes de sus ajustes; publicación en el repositorio posterior a la ejecución |
| Unidad de repetición | Trayectoria de entrenamiento por semilla; fechas compartidas, sin interpretar meses o semillas como mercados independientes |
| Selección del baseline | GRU/D7 por CV, cerrada antes de las nuevas inferencias de test; no se selecciona por ranking del test |
| Comparaciones dentro del trío | Análisis posterior al test, con incertidumbre y multiplicidad; alcance exploratorio |
| Margen para modelos futuros | Operativo académico: 5 % del RMSE CV de GRU/D7, aproximadamente **0,00025772843**, con sensibilidad 2,5/10 %; definido después del test actual |
| Piloto y potencia | Pendientes para el procedimiento GRU–LSTM–TCN; no se reutilizan resultados de otros modelos como si fueran su piloto |
| Equivalencia | No afirmada; falta de significancia no demuestra igualdad |
| Fuera de alcance | Utilidad financiera, generalización temporal independiente y datos de septiembre–octubre de 2026 |

El margen futuro debe registrarse antes de una evaluación nueva y acompañarse de su propio dimensionamiento y de la comparación con persistencia. Su definición actual no prueba que un modelo avanzado supere el baseline, ni altera los protocolos originales del ensayo local.

La [cronología de la entrega](../_static/gru_lstm_tcn_delivery_20261105/declaration_and_chronology.json), los [resultados](../_static/gru_lstm_tcn_20261105/summary.json) y la [matriz de pendientes](protocolo_pendiente.md) conservan estas distinciones. La presencia de los siete archivos de entrega no significa que el piloto prospectivo o la corroboración bibliográfica adicional estén cerrados.
