# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Entregable 1 · LSTM–MLP–GRU · benchmark principal de 608 días · actualización del 7 de noviembre de 2026, Colombia**

E1 compara modelos básicos de Deep Learning para pronosticar la volatilidad de **Bitcoin BTCUSDT**, con datos oficiales de Binance Spot. La tarea es **regresión temporal, tipo 4**: cinco covariables OHLCV diarias, volatilidad poblacional de siete retornos logarítmicos y horizonte de un día. Conserva la validación forward chaining de **timeseries-cv**, con coautoría de Filipe R. Ramos, purga temporal y transformaciones aprendidas únicamente del pasado permitido.

**La comparación principal conserva GRU y LSTM de 32 unidades e incorpora la MLP histórica de 16 unidades bajo las mismas condiciones de datos y evaluación.** Las tres tienen diez semillas y el mismo test de **608 días, del 01/01/2025 al 31/08/2026**. TCN permanece complementaria. El estudio posterior de sesenta días se conserva como comprobación adicional con sus propias recetas y resultados.

## Benchmark principal: mismas condiciones, 608 fechas

Las tres redes ya habían sido ejecutadas bajo este protocolo. La MLP no se infiere de otra arquitectura: sus diez puntuaciones por semilla están conservadas, junto al registro original de evaluación. La conciliación verifica contrato de datos, entrenamiento, calendarios, semillas y **480 coincidencias exactas de LSTM** entre las fuentes históricas. Esta actualización recupera esas ejecuciones existentes y no afirma un entrenamiento nuevo.

| Modelo principal D7 | Parámetros | RMSE test, media ± DE | R² test medio |
|---|---:|---:|---:|
| **GRU, 32 unidades** | **3.777** | **0,00377511 ± 0,00019410** | **0,854173** |
| **MLP, 16 unidades** | **593** | **0,00479318 ± 0,00039878** | **0,764012** |
| **LSTM, 32 unidades** | **5.025** | **0,00502979 ± 0,00026367** | **0,741108** |
| Persistencia | 0 | 0,00408560 | 0,829605 |

Las redes resumen **diez semillas 4101–4110**; persistencia es determinista. El [benchmark principal](sections/05_benchmark.md) publica las seis métricas, DE, IC y su procedencia. **GRU tiene el menor RMSE y mayor R² medios de este periodo; persistencia conserva mejores MAE y MAPE.** Las métricas no se suman para declarar un ganador universal.

### Protocolo común y recetas conservadas

- **Desarrollo:** 2023–2024; CV de doce bloques mensuales sobre las 366 fechas de 2024.
- **Entrada y objetivo:** D7×5 OHLCV, σ₇ poblacional sin anualizar y horizonte de un día.
- **Ajuste final:** las mismas 709 etiquetas elegibles, del 08/01/2023 al 24/12/2024, con embargo de siete días y escalas aprendidas de entrenamiento.
- **Evaluación:** 608 fechas comunes, pesos y escaladores fijos; veinte meses de reporte sin reajustar durante test.
- **Recetas:** GRU/LSTM H32, LR 0,003; MLP H16, LR 0,001; dropout 0,1 y reglas de optimización/parada comunes.

Las cuatro entradas **D=7/14/21/28** conservan sus resultados. D7 minimiza la RMSE CV de cada familia y **GRU/D7 de 32 unidades es el baseline neuronal principal**. Los esfuerzos históricos de HPO difieren: se comparan estas recetas originales y no un óptimo global de familias con igual búsqueda. Las diez repeticiones fueron un presupuesto histórico fijo; no se les atribuye el dimensionamiento posterior de R=465.

**Mayor cobertura temporal:** 608 días abarcan veinte meses, frente a los dos meses del estudio complementario. Priorizar ahora este benchmark es una decisión de presentación posterior a conocer ambos resultados. El test ya era conocido y la comparación es histórica y exploratoria; su mayor duración no acredita por sí sola representatividad de otros mercados o periodos.

## Selección bibliográfica de los modelos base

| Familia | Frecuencia en diez experimentos cripto con reporte suficiente | Mecanismo básico |
|---|---:|---|
| **LSTM** | **10/10** | Una capa recurrente con estado de memoria separado y compuertas |
| **MLP** | **4/10** | Una capa oculta densa sobre la ventana ordenada, sin estado recurrente |
| **GRU** | **2/10** | Una capa recurrente con actualización y reinicio del estado |
| TCN, complementaria | 1/10 | Convoluciones causales dilatadas |

La {ref}`selección de §3.14 <core-selection-e1>` deriva esos recuentos del corpus congelado, identifica autores, DOI y adaptaciones, y comprueba simplicidad y capacidad. La MLP principal tiene **una capa oculta y 593 parámetros**. Las frecuencias describen el corpus revisado, con búsquedas dirigidas y límites de cobertura; todas las familias reciben la misma información observable.

## Comprobación complementaria de sesenta días

El {ref}`estudio complementario D7 <final-core-mlp-e1>` conserva intactas sus recetas **GRU16, LSTM16 y MLP32**, HPO equitativo, piloto y evaluación del **01/09–30/10/2026**. La fuente se publicó en [91acdeae](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) antes de sus ajustes; el [dimensionamiento](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) también precedió al ajuste definitivo.

Ese estudio ejecutó 960 etapas de HPO, 60 de piloto y 2.790 definitivas: **465 semillas por familia, 1.395 modelos, 83.700 predicciones y 44.201 registros de pérdida**. Su baseline propio es GRU16, distinto del GRU32 principal. Sus tamaños, curvas y contrastes pertenecen a esas recetas y fechas.

**En esos sesenta días, persistencia obtiene el menor error.** Los R² medios son 0,640632 para persistencia, 0,348891 para GRU16, 0,345286 para MLP32 y 0,215843 para LSTM16. GRU–MLP no rechaza tras Holm (p=0,243); ningún DM rechaza tras Holm-6 ni se acredita mejora temporal relativa superior al 5 %. El libro conserva esta evidencia adversa y la dependencia del resultado respecto al periodo y la receta. Los dos estudios no se mezclan para atribuir diferencias únicamente a una arquitectura.

## Ruta de lectura

- [Problema y dataset](sections/01_problema_dataset.md): formulación, procedencia y reloj de información.
- [Análisis exploratorio](sections/02_eda.md): calidad y dependencia del desarrollo 2023–2024.
- [Estado del arte](sections/03_estado_arte.md): búsquedas, lectura crítica y selección.
- [Implementación](sections/04_implementacion_evaluacion.md): recetas principales y ejecuciones complementarias.
- [Benchmark principal](sections/05_benchmark.md): GRU32–LSTM32–MLP16, cuatro entradas y test de 608 días.
- [Procedimiento](sections/06_procedimiento.md): validación forward chaining y disponibilidad causal.
- [Limitaciones](sections/07_limitaciones_entregables.md): alcance, planificación y generalización.
- [Entregables y descargas](sections/08_entregables.md): informe, código, configuración y evidencia.
- [Estudio complementario con TCN](sections/09_estudio_planificado.md): archivo y cronología.
- [Estudio complementario del núcleo en 60 días](sections/10_nucleo_mlp.md): HPO, piloto, curvas y análisis propios.
- [Matriz de cumplimiento](sections/protocolo_pendiente.md): requisitos, evidencia y alcance.

## Literatura, margen y reproducibilidad

La bibliografía canónica conserva **70 referencias**. La {ref}`síntesis acumulada <homogeneous-expansion-e1>` concilia 57 obras, 35 textos primarios completos y una transcripción completa. El constructo cripto comprende quince experimentos, trece descriptivos y diez con reporte suficiente para el ranking principal. La {ref}`ampliación gratuita <free-model-expansion-e1>` mantiene los papeles de originales, fuentes relacionadas e híbridos.

La {ref}`bitácora nativa <current-native-execution-e1>` acredita las consultas del **6 de noviembre de 2026 en Colombia**: **WoS 271, Scopus 538 e IEEE 32 resultados**, con **271/538/25 registros exportados**. Cadenas, filtros y fechas conservan su procedencia; las sesiones antiguas no se reconstruyen con aproximaciones. Catorce informes no recuperados al corte conservan identidad, motivo y alternativas verificadas.

El margen académico δ=0,00025772842664981106 conserva el ancla del 5 % del RMSE CV del GRU32 inicial. Su [justificación](sections/01_problema_dataset.md) y el piloto del estudio complementario no retroceden en el tiempo para dimensionar las diez semillas históricas. Las semillas describen inicialización condicionada al mismo calendario y no son mercados independientes.

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) y las [descargas](sections/08_entregables.md) distinguen la reconstrucción de métricas existentes de un entrenamiento nuevo. Construir HTML no entrena redes ni abre observaciones de mercado. Las curvas, predicciones y configuraciones de cada ejecución conservan su versión; E2 mantiene sus propios registros inmutables.
