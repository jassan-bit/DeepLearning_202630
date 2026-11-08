# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Entregable 1 · LSTM–MLP–GRU · benchmark principal de 608 días · cierre de evidencias del 8 de noviembre de 2026 UTC (7 de noviembre en Colombia)**

E1 compara modelos básicos de Deep Learning para pronosticar la volatilidad de **Bitcoin BTCUSDT**, con datos oficiales de Binance Spot. La tarea es **regresión temporal, tipo 4**: cinco covariables OHLCV diarias, volatilidad poblacional de siete retornos logarítmicos y horizonte de un día. Conserva la validación forward chaining de **timeseries-cv**, con coautoría de Filipe R. Ramos, purga temporal y transformaciones aprendidas únicamente del pasado permitido.

**La comparación principal vuelve a entrenar GRU32, LSTM32 y MLP16 sobre las mismas 608 fechas**, con HPO equitativa, un piloto propio y **309 inicializaciones por red** dimensionadas antes de sus ajustes definitivos. TCN permanece complementaria. El registro original de diez semillas y la comprobación posterior de sesenta días conservan sus resultados y su cronología.

## Benchmark principal: HPO común, piloto y 608 fechas

La {ref}`ejecución registrada de 608 fechas <remediation-608-e1>` mantiene las arquitecturas básicas y el periodo **01/01/2025–31/08/2026**. Las tres familias reciben cuatro candidatos LR–dropout, dos semillas comunes y diez folds de enero–octubre de 2024: **480 etapas de HPO**, 160 por familia. **GRU32 es el baseline neuronal elegido con CV**, antes del piloto y de puntuar esta ejecución.

El piloto separado ejecuta diez semillas por red sobre noviembre–diciembre de 2024. Con **δ=0,00025772842664981106**, α/3 y potencia marginal objetivo 0,80, el cálculo con cota simultánea de la DE fija **R=309**, semillas **18101–18409**. El [plan íntegro](https://github.com/jassan-bit/DeepLearning_202630/blob/0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3/experiments/e1_608_remediation_20261107/planning/plan.json) se publicó y verificó antes de los ajustes definitivos. El presupuesto nuevo comprende **927 modelos finales y 2.394 etapas de ajuste** contando HPO, piloto y ajuste definitivo.

Las recetas elegidas son **GRU32, LR 0,003 y dropout 0,1; LSTM32, LR 0,003 y dropout 0; MLP16, LR 0,003 y dropout 0,1**. Las tres se reajustan sobre 709 etiquetas elegibles hasta el 24/12/2024, con embargo de siete días, transformaciones del entrenamiento y pesos fijos durante las 608 fechas. Se conservan las predicciones diarias de todas las semillas para los análisis temporales conjuntos.

| Modelo D7 | Parámetros | RMSE, media ± DE | R² medio |
|---|---:|---:|---:|
| GRU | 3.777 | 0,00521269 ± 0,00112121 | 0,709833 |
| LSTM | 5.025 | 0,00625115 ± 0,00079563 | 0,594658 |
| MLP | 593 | 0,00522813 ± 0,00054111 | 0,717998 |
| Persistencia | 0 | 0,00408560 | 0,829605 |

Las redes resumen **309 inicializaciones**, cada una sobre las mismas 608 fechas; persistencia es determinista. **Persistencia tiene menor RMSE y mayor R² que las tres redes en la ejecución nueva.** GRU–MLP no rechaza diferencia cero tras Holm-3 (**p=0,823**); no se concluye equivalencia. GRU conserva el papel de baseline neuronal por su selección de desarrollo, independiente del resultado del test.

La puntuación cerró **563.616 predicciones** y los registros contienen **33.141 pérdidas de entrenamiento/validación**. El [benchmark](sections/05_benchmark.md) publica seis métricas, IC, pruebas entre semillas, DM temporal, bootstrap por bloques y diagnósticos residuales de esta misma ejecución. La revisión numérica independiente verificó esas cifras desde sus predicciones cerradas. Las [descargas](sections/08_entregables.md) incluyen los agregados, curvas y configuraciones.

El {ref}`análisis posterior de errores <frozen-errors-e1>` añade observado frente a pronosticado, veinte meses, rangos comunes de volatilidad, sesgo y casos difíciles usando esas mismas predicciones congeladas. **Las 5.568 métricas individuales verificadas coinciden exactamente y los archivos originales conservan sus hashes.** No se reentrena ni se modifica el resultado final.

La potencia planificada describe diferencias entre inicializaciones bajo sus supuestos; no es potencia temporal ni potencia alcanzada. **El calendario ya era conocido: esta es una comparación histórica exploratoria.** La publicación previa del nuevo código y del plan acredita el orden de esta ejecución sin convertir sus fechas en una reserva independiente nueva.

### Resultados originales conservados

El {ref}`registro original de diez semillas <benchmark-core-historico-e1>` conserva GRU32 RMSE **0,00377511** y R² **0,854173**, MLP16 RMSE **0,00479318** y R² **0,764012**, LSTM32 RMSE **0,00502979** y R² **0,741108**. Persistencia presenta RMSE **0,00408560** y R² **0,829605**. GRU tenía el menor RMSE de ese registro; persistencia mantenía mejores MAE y MAPE.

Estos valores pertenecen a las ejecuciones originales y no se atribuyen a los nuevos pesos. Las entradas **D=7/14/21/28** y sus métricas originales siguen disponibles. D7 fue el mínimo CV de cada familia y se fija en la nueva comparación. El esfuerzo de HPO desigual y el presupuesto fijo de diez semillas se conservan como límites de ese antecedente; el piloto nuevo no se retrofecha.

## Selección bibliográfica de los modelos base

| Familia | Corte de selección, N=10 | Síntesis ampliada, N=18 | Mecanismo básico |
|---|---:|---:|---|
| **LSTM** | **10/10** | **15/18** | Una capa recurrente con estado de memoria separado y compuertas |
| **MLP** | **4/10** | **6/18** | Una capa oculta densa sobre la ventana ordenada, sin estado recurrente |
| **GRU** | **2/10** | **4/18** | Una capa recurrente con actualización y reinicio del estado |
| TCN, complementaria | 1/10 | 1/18 | Convoluciones causales dilatadas |

La {ref}`selección de §3.14 <core-selection-e1>` conserva su corte original. El {ref}`cierre de §3.15 <native-620-closure-e1>` integra los 620 grupos de la ronda nativa: añade ocho experimentos estrictos a los diez previos y confirma el mismo orden de las tres primeras familias. Cada experimento aporta como máximo un voto por familia autónoma; propuestas híbridas, variantes y versiones de autor se distinguen. La MLP principal tiene **una capa oculta y 593 parámetros**; las tres redes reciben la misma información observable. Estas frecuencias describen el corpus evaluado, con sus límites de búsqueda y calidad.

## Comprobación complementaria de sesenta días

El {ref}`estudio complementario D7 <final-core-mlp-e1>` conserva intactas sus recetas **GRU16, LSTM16 y MLP32**, HPO equitativo, piloto y evaluación del **01/09–30/10/2026**. La fuente se publicó en [91acdeae](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) antes de sus ajustes; el [dimensionamiento](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) también precedió al ajuste definitivo.

Ese estudio ejecutó 960 etapas de HPO, 60 de piloto y 2.790 definitivas: **465 semillas por familia, 1.395 modelos, 83.700 predicciones y 44.201 registros de pérdida**. Su baseline propio es GRU16, distinto del GRU32 principal. Sus tamaños, curvas y contrastes pertenecen a esas recetas y fechas.

**En esos sesenta días, persistencia obtiene el menor error.** Los R² medios son 0,640632 para persistencia, 0,348891 para GRU16, 0,345286 para MLP32 y 0,215843 para LSTM16. GRU–MLP no rechaza tras Holm (p=0,243); ningún DM rechaza tras Holm-6 ni se acredita mejora temporal relativa superior al 5 %. El libro conserva esta evidencia adversa y la dependencia del resultado respecto al periodo y la receta. Los dos estudios no se mezclan para atribuir diferencias únicamente a una arquitectura.

## Ruta de lectura

- [Problema y dataset](sections/01_problema_dataset.md): formulación, procedencia y reloj de información.
- [Análisis exploratorio](sections/02_eda.md): calidad y dependencia del desarrollo 2023–2024.
- [Estado del arte](sections/03_estado_arte.md): búsquedas, lectura crítica y selección.
- [Implementación](sections/04_implementacion_evaluacion.md): recetas principales y ejecuciones complementarias.
- [Benchmark principal](sections/05_benchmark.md): nueva comparación D7 y archivo original de cuatro entradas.
- [Procedimiento](sections/06_procedimiento.md): validación forward chaining y disponibilidad causal.
- [Limitaciones](sections/07_limitaciones_entregables.md): alcance, planificación y generalización.
- [Entregables y descargas](sections/08_entregables.md): informe, código, configuración y evidencia.
- [Estudio complementario con TCN](sections/09_estudio_planificado.md): archivo y cronología.
- [Estudio complementario del núcleo en 60 días](sections/10_nucleo_mlp.md): HPO, piloto, curvas y análisis propios.
- [Ejecución principal registrada](sections/11_reejecucion_608.md): HPO común, piloto, R=309, curvas y análisis de las 608 fechas.
- [Errores con predicciones congeladas](sections/12_errores_congelados.md): comparación visual, rangos de volatilidad y casos difíciles, conservando las métricas finales.
- [Matriz de cumplimiento](sections/protocolo_pendiente.md): requisitos, evidencia y alcance.

## Literatura, margen y reproducibilidad

La bibliografía base conserva **70 referencias**, acompañadas ahora de **32 adiciones** de documentos primarios evaluados. Citar una fuente excluida permite documentar el juicio y no la convierte en experimento elegible. La {ref}`síntesis acumulada <homogeneous-expansion-e1>` concilia 57 obras, 35 textos primarios completos y una transcripción completa. El constructo cripto comprende quince experimentos, trece descriptivos y diez con reporte suficiente para el ranking principal. La {ref}`ampliación gratuita <free-model-expansion-e1>` mantiene los papeles de originales, fuentes relacionadas e híbridos.

La {ref}`bitácora nativa <current-native-execution-e1>` acredita las consultas del **6 de noviembre de 2026 en Colombia**: **WoS 271, Scopus 538 e IEEE 32 resultados**, con **271/538/25 registros exportados**. Cadenas, filtros y fechas conservan su procedencia; las sesiones antiguas no se reconstruyen con aproximaciones. Los catorce informes no recuperados del corte anterior conservan su historia. El cierre posterior revisa **620 grupos**, con **509 exclusiones iniciales y 111 avances**: nueve exclusiones tras ampliar metadatos, **44 evaluaciones primarias** y **58 informes no recuperados al corte**, todos con disposición explícita. De los 44 evaluados, trece satisfacen el estrato estricto; cinco ya estaban incluidos y ocho son nuevos. Los siete resultados IEEE no exportados quedan fuera del universo recibido. El flujo y las fichas permiten distinguir revisión terminada de disponibilidad de texto completo.

El margen académico δ=0,00025772842664981106 conserva el ancla del 5 % del RMSE CV del GRU32 inicial. Su [justificación](sections/01_problema_dataset.md) y el piloto propio dimensionan los ajustes nuevos; ningún piloto posterior dimensiona retrospectivamente las diez semillas originales. Las semillas describen inicialización condicionada al mismo calendario y no son mercados independientes.

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) y las [descargas](sections/08_entregables.md) distinguen las métricas históricas del entrenamiento nuevo registrado. Construir HTML no entrena redes ni abre observaciones de mercado. Las curvas, predicciones y configuraciones de cada ejecución conservan su versión; E2 mantiene sus propios registros inmutables.

## Continuidad propuesta hacia E2 y E3

El candidato activo compartido de E2/E3 es **CNN-1D + LSTM + atención temporal**, propuesto y todavía no implementado ni entrenado. Mantiene como referencia el contrato de este E1 principal: D7×5 relativo, σ₇ poblacional y horizonte de un día. Los diez modelos de E2 y ese candidato no sustituyen las referencias MLP16, LSTM32 y GRU32 ni cambian los resultados anteriores.

El [contrato sincronizado](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) separa cohortes, preprocesamiento y métricas de la ejecución E1 de las decisiones pendientes del híbrido. Las 608 fechas ya conocidas sólo permiten una comparación histórica exploratoria; R=309, HPO y conclusiones de E1 no se transfieren al nuevo modelo. [E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html) · [E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html).
