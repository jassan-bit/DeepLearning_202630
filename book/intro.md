# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Entregable 1 · Modelos base LSTM–MLP–GRU · actualización del 7 de noviembre de 2026, Colombia**

E1 compara modelos básicos de Deep Learning para pronosticar la volatilidad de **Bitcoin BTCUSDT**, con datos oficiales de Binance Spot. La tarea es **regresión temporal, tipo 4**: cinco covariables OHLCV diarias, volatilidad poblacional de siete retornos logarítmicos y horizonte de un día. El procedimiento conserva la validación forward chaining de **timeseries-cv**, con coautoría de Filipe R. Ramos, purga temporal y transformaciones aprendidas únicamente del pasado permitido.

**El núcleo vigente es LSTM–MLP–GRU. TCN se conserva como comparación complementaria.** La selección utiliza las tres familias más recurrentes del estrato cripto con reporte suficiente y concreta su simplicidad y diversidad mediante arquitecturas de una capa oculta o recurrente. El cambio está respaldado por una matriz verificable y por una ejecución propia; se preservan nombres y puntuaciones de los experimentos anteriores.

## Selección de los modelos base

| Familia | Frecuencia en diez experimentos elegibles | Arquitectura básica y mecanismo |
|---|---:|---|
| **LSTM** | **10/10** | Una capa recurrente con estado de memoria separado y compuertas de entrada, olvido y salida |
| **MLP** | **4/10** | Una capa oculta densa sobre la ventana ordenada; aprende interacciones entre posiciones sin estado recurrente |
| **GRU** | **2/10** | Una capa recurrente con actualización y reinicio del estado |
| TCN, complementaria | 1/10 | Convoluciones causales dilatadas; evidencia y experimentos históricos conservados |

La {ref}`decisión de §3.14 <core-selection-e1>` deriva estos recuentos del corpus congelado, identifica autores, DOI, recetas y adaptaciones, y comprueba parámetros. Las frecuencias describen **el corpus revisado**, con sus búsquedas dirigidas y límites de cobertura. Todas las familias reciben la misma ventana; los estados recurrentes se reinician en cada muestra.

## Ruta de lectura

- [Problema y dataset](sections/01_problema_dataset.md): formulación, procedencia, reloj de información y objetivo σ₇.
- [Análisis exploratorio](sections/02_eda.md): calidad, dependencia temporal y decisiones del desarrollo 2023–2024.
- [Estado del arte](sections/03_estado_arte.md): búsquedas nativas, lectura crítica y selección vigente.
- [Implementación](sections/04_implementacion_evaluacion.md): modelos, cuatro entradas, entrenamiento y reproducibilidad.
- [Benchmark](sections/05_benchmark.md): comparación histórica compatible del núcleo y resultados complementarios.
- [Procedimiento](sections/06_procedimiento.md): validación forward chaining y disponibilidad causal.
- [Limitaciones](sections/07_limitaciones_entregables.md): alcance temporal, potencia y generalización.
- [Entregables y descargas](sections/08_entregables.md): informe, código, configuración y evidencia pública.
- [Estudio anterior con TCN](sections/09_estudio_planificado.md): resultados y cronología conservados.
- [Núcleo LSTM–MLP–GRU](sections/10_nucleo_mlp.md): nueva comparación D7 con HPO, piloto y planificación propios.
- [Matriz de cumplimiento](sections/protocolo_pendiente.md): requisitos, evidencia y alcance.

## Nueva comparación D7

El [protocolo y código](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) quedaron publicados y verificados antes de los nuevos ajustes. El {ref}`capítulo 10 <final-core-mlp-e1>` identifica esta ejecución y sus resultados propios.

El HPO completó **ocho candidatos y dos semillas por familia**, diez bloques enero–octubre de 2024 y **320 etapas por familia, 960 en total**. Comparte D7, n=7, h=1, preprocesamiento, optimizador y reglas de parada. La selección quedó congelada antes del piloto y de la evaluación:

| Modelo | Receta seleccionada con desarrollo | Parámetros | RMSE CV, media ± DE |
|---|---|---:|---:|
| **GRU** | 16 unidades, LR 0,003, dropout 0,1 | **1.121** | **0,00513620 ± 0,00003068** |
| MLP | Una capa oculta de 32 unidades, LR 0,003, dropout 0,1 | 1.185 | 0,00527943 ± 0,00003534 |
| LSTM | 16 unidades, LR 0,003, dropout 0 | 1.489 | 0,00626389 ± 0,00021402 |

El **baseline actual es GRU/D7-c03 de 16 unidades**, elegido por CV. La MLP nueva tiene 1.185 parámetros y conserva su propia procedencia; la MLP histórica de 593 parámetros permanece en su tabla anterior.

El piloto completó **diez semillas nuevas por familia, treinta trayectorias y sesenta etapas**, sobre noviembre–diciembre de 2024. Su dispersión pareada requiere **R=331** mediante t no central y multiplicidad; el piso prefijado produce **465 ejecuciones por modelo**, conservando δ=0,00025772842664981106. El [plan exacto](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) quedó publicado y verificado antes de los ajustes definitivos. Se cierran todos los pesos antes de puntuar.

La evaluación comprende **01/09–30/10/2026**, sesenta fechas con contexto de agosto. **Este calendario ya fue evaluado en el proyecto**: la extensión es histórica y exploratoria, con procedimiento registrado antes de sus nuevos ajustes. No constituye un holdout independiente nuevo ni predicciones emitidas en vivo.

### Evaluación final del núcleo

Se cerraron **1.395 modelos y 2.790 etapas definitivas** antes de puntuar; la ejecución completa, incluido HPO y piloto, comprende **3.810 etapas**. Los 32 archivos recibidos coinciden con los hashes originales. La evaluación conserva **83.700 predicciones** sobre las sesenta fechas comunes.

| Modelo final | RMSE, media ± DE | MAE medio | R² medio |
|---|---:|---:|---:|
| **GRU** | **0,00511696 ± 0,00077519** | 0,00391028 | 0,348891 |
| **MLP** | **0,00516664 ± 0,00048727** | 0,00397196 | 0,345286 |
| **LSTM** | **0,00565123 ± 0,00056569** | 0,00429363 | 0,215843 |
| Persistencia | 0,00384478 | 0,00212123 | 0,640632 |

Las redes resumen 465 semillas cada una; persistencia es determinista. **GRU–MLP no rechaza igualdad de medias tras Holm, p=0,243**, sin demostrar equivalencia. Ambas tienen menor RMSE que LSTM en la comparación entre inicializaciones. Ningún DM rechaza tras Holm-6; el bootstrap temporal familiar no acredita una mejora superior al 5 %. Persistencia conserva el menor error puntual.

El {ref}`capítulo 10 <final-core-mlp-e1>` publica las seis métricas, IC, contrastes y **44.201 registros de pérdidas de entrenamiento/validación**. Una revisión independiente recalculó las seis métricas desde las predicciones guardadas con diferencia máxima cero y verificó la cronología y los archivos públicos.

## Comparación histórica del núcleo

El [benchmark](sections/05_benchmark.md) reúne métricas reales de **GRU, MLP y LSTM** para **D=7/14/21/28**, diez semillas comunes 4101–4110, doce bloques de CV de 2024 y test histórico **01/01/2025–31/08/2026**. Los modelos comparten contrato de datos y calendario. Las recetas y el esfuerzo de HPO original difieren; este antecedente se distingue de la nueva búsqueda comparable.

| Modelo histórico D7 | Parámetros | RMSE CV, media | RMSE test de 608 días, media ± DE |
|---|---:|---:|---:|
| **GRU, 32 unidades** | **3.777** | **0,00515457** | **0,00377511 ± 0,00019410** |
| MLP, 16 unidades | 593 | 0,00604907 | 0,00479318 ± 0,00039878 |
| LSTM, 32 unidades | 5.025 | 0,00633541 | 0,00502979 ± 0,00026367 |

Las cuatro entradas y las seis métricas se conservan en el capítulo 5. D7 minimiza la RMSE CV de cada familia en ese antecedente; GRU es su baseline histórico. El test ya había sido observado y sus contrastes son exploratorios. La nueva selección documental conserva esta cronología.

## Literatura y búsqueda reproducible

La bibliografía canónica contiene **70 referencias**. La {ref}`síntesis acumulada <homogeneous-expansion-e1>` concilia 57 obras, **35 textos primarios completos consultados y una transcripción completa**. El constructo cripto contiene quince experimentos: trece con familias autónomas extraíbles y diez con reporte suficiente para el ranking principal. La {ref}`ampliación gratuita <free-model-expansion-e1>` documenta trece obras seleccionadas, de las que diez son incorporaciones; fuentes de otros mercados e híbridos conservan su papel de evidencia relacionada.

La {ref}`bitácora nativa <current-native-execution-e1>` acredita consultas del **6 de noviembre de 2026 en Colombia**: **WoS 271, Scopus 538 e IEEE Xplore 32 resultados**. Se conservan cadenas, campos, filtros y exportaciones reales, distinguiendo resultados de registros exportados: **271/538/25**. Los filtros difieren y se declaran: Scopus 2019–2026; WoS con años exportados 2017–2026; IEEE con acceso abierto e intervalo visible 2018–2026.

La recuperación histórica conserva ocho decisiones verificadas y catorce informes no obtenidos al corte, con identidad, motivo y alternativas gratuitas por función. Es una limitación de cobertura documentada. Las sesiones antiguas y los cortes bibliográficos fechados se conservan separados de la ronda nativa actual.

## Interpretación y reproducibilidad

El margen del 5 % es una convención para la **comparación académica** confirmada por el autor. Su [justificación](sections/01_problema_dataset.md) lo sitúa frente a la variación de desarrollo y presenta sensibilidad al 2,5/10 %. Las semillas caracterizan aleatoriedad del entrenamiento sobre las mismas fechas; no representan cientos de mercados independientes. Persistencia permanece visible y los resultados adversos se conservan.

El estudio anterior con TCN conserva curvas públicas y la {ref}`auditoría de repetibilidad <auditoria-numerica-e1>` de seis celdas: **18/18 coincidencias canónicas actuales**, separadas de la comparación limitada con archivos privados originales. La nueva ejecución mantiene sus propias curvas y procedencia.

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) y las [descargas](sections/08_entregables.md) documentan la restauración. Construir HTML consume resultados guardados: no entrena redes ni abre datos de mercado. Los precios, predicciones diarias, escaladores y checkpoints se mantienen separados de los agregados públicos.
