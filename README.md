# DeepLearning_202630

Este repositorio reúne **tres Jupyter Books**, cada uno con índice, configuración y dirección propios:

| Entregable | Fuentes | Dirección del libro | Estado |
|---|---|---|---|
| E1 | [`book/`](book/) | [Abrir E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) | Informe y estudios realizados |
| E2 | [`book_e2/`](book_e2/) | [Abrir E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html) | Ruta F2a, piloto Bitcoin ejecutado (440 ajustes) y diez adaptadores finales revisados; HPO/resultados finales pendientes |
| E3 | [`book_e3/`](book_e3/) | [Abrir E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html) | Libro iniciado; guía específica pendiente |

[Selector de los tres entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html). La infraestructura de compilación es común; cada JBook se construye desde su propia carpeta. Los apartados experimentales que siguen corresponden a **E1**.

La [sección 5 de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/05_paper_guia.html) contiene la réplica externa de TimeMixer y el piloto sobre tres modelos en ETTh1. Su [paquete descargable](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip) reúne código y evidencias; estas ejecuciones no modifican los resultados de E1 ni completan el benchmark Bitcoin D7.

La [sección 6 de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/06_benchmark_final.html) prepara la comparación tabular, la eficiencia, la inferencia y la discusión del benchmark. Su [paquete de análisis y plantillas](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/benchmark_analysis_download/Benchmark_E2_analisis_plantillas.zip) contiene código verificable y pruebas con fixtures sintéticos. Las métricas y el ranking Bitcoin continúan pendientes de ejecuciones válidas.


El [protocolo P0 de E2](book_e2/sections/09_protocolo_registrado.md) se registró públicamente en `599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0` antes de entrenar. Integra veinte configuraciones, quince recetas ajustables T12, NAS condicional, familias e inferencia temporal, potencia objetivo y calibración disjunta. El [paquete actual](book_e2/_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) contiene código y evidencia agregada de la etapa P0/OOF de desarrollo; no acredita HPO, calibración de los mejores finales ni el benchmark definitivo. La [nueva implementación final](experiments/e2_completion_20261106/analysis_additions.json) incorpora revisión individual de diez adaptadores, HPO temporal T12 con puntos fuente, P1 y calibración A53 (08/11–30/12/2024), estadística, robustez y ablaciones. Su registro y ejecuciones son evidencia distinta del P0 ya cerrado; no se atribuyen resultados finales inexistentes. Faltan los 374 Scopus y la adjudicación bibliográfica completa. Los capítulos 7–14 desarrollan la guía ya recibida.


Proyecto de Deep Learning, 2026-30: **pronóstico de volatilidad de Bitcoin con GRU, LSTM y TCN**.

[Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [Estudio D7 planificado](https://jassan-bit.github.io/DeepLearning_202630/sections/09_estudio_planificado.html) · [Entregables descargables](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1) · [Reproducción](BUILD.md).

El objetivo es σ₇, desviación estándar poblacional de siete retornos logarítmicos diarios de Close, sin anualización, con horizonte de un día. Las entradas son cinco canales OHLCV diarios y la validación emplea los índices forward chaining de `timeseries-cv`, proyecto de DidierRLopes y Filipe R. Ramos, con historia inicial 2023 y embargo temporal.

## Dos fases identificadas

| Elemento | Fase I exploratoria | Fase II D7 planificada |
|---|---|---|
| Historias | D=7/14/21/28 | D=7, n=7, h=1 |
| Ajuste | Recetas fijas y diez semillas | Ocho candidatos, dos semillas y diez bloques comunes por familia; 960 ajustes HPO |
| Piloto | No precedió al benchmark inicial | Diez semillas nuevas por familia, 60 días de desarrollo y 60 ajustes |
| Repeticiones definitivas | Diez por configuración | 465 por familia, calculadas antes del ajuste definitivo |
| Evaluación | 01/01/2025–31/08/2026, 608 fechas ya observadas | 01/09–30/10/2026, 60 fechas de una reserva histórica no usada anteriormente para resultados |
| Inferencia | Posterior y exploratoria | Métodos, margen y comparaciones registrados antes de la nueva evaluación |

El diseño y código de fase II se [publicaron antes del HPO](https://github.com/jassan-bit/DeepLearning_202630/commit/ac7dd7f67c0dec2397b519897cf6da8056b238f5). Tras seleccionar las recetas y ejecutar el piloto, el [dimensionamiento se cerró públicamente antes del ajuste definitivo](https://github.com/jassan-bit/DeepLearning_202630/commit/5a0acfbc2ff582e252f0e6fb18aaf652963221f8). El R=465 se calcula mediante t no central, α*=0,05/3, potencia marginal objetivo 0,80 y límite superior simultáneo del 80 % para las DE pareadas del piloto. Supone normalidad y transferencia de dispersión; no acredita potencia temporal o conjunta del 80 %.

El baseline de fase II es **GRU/D7 de 16 unidades**, elegido por RMSE media de HPO enero–octubre de 2024 (**0,00513620 ± 0,00003068**). LSTM selecciona 16 unidades y TCN ancho 8; las tres usan LR 0,003, con dropout 0,1 para GRU y 0 para LSTM/TCN. El piloto y HPO D7 no se atribuyen retrospectivamente a las otras entradas.

El ejecutor nuevo es [run_gru_lstm_tcn_planned.py](scripts/run_gru_lstm_tcn_planned.py), con [datos reservados](scripts/planned_study_data.py), [estadística preespecificada](scripts/planned_study_statistics.py) y [diseño versionado](experiments/gru_lstm_tcn_planned_20261105/design.json). Sus fases explícitas separan registro, HPO, piloto, planificación, ajuste, puntuación y análisis. Todos los pesos y escaladores se cierran antes de construir objetivos de la nueva reserva. La adquisición previa ya había recorrido esos CSV para auditar integridad, timestamps y esquema: no se afirma que los archivos nunca se abrieron, ni que son pronósticos emitidos en vivo.

## Resultados de la nueva reserva de 60 días

| Modelo de fase II | RMSE, media ± DE | R² medio |
|---|---:|---:|
| GRU | 0,00511696 ± 0,00077519 | 0,34889070 |
| LSTM | 0,00565123 ± 0,00056569 | 0,21584267 |
| TCN | 0,00433967 ± 0,00099896 | 0,51795587 |
| Persistencia | 0,00384478 | 0,64063235 |

Persistencia tiene menor RMSE puntual que las tres redes; TCN ocupa el primer puesto descriptivo entre redes. El baseline operativo sigue siendo GRU/D7-c03, elegido antes del piloto y de la reserva. Ninguna de las seis comparaciones temporales DM rechaza tras Holm; el bootstrap temporal familiar no acredita una reducción superior al 5 %. Shapiro–Wilk rechaza normalidad de los tres pares de diferencias, por lo que las t con 465 pares se interpretan como aproximaciones, no como inferencia exacta bajo normalidad demostrada.

Estos resultados corresponden al 01/09–30/10/2026. No se comparan directamente con el test de 608 fechas para atribuir una mejora o deterioro al HPO.

## Benchmark histórico conservado

La fase I produjo 120 trayectorias CV, 2.880 ajustes y 43.920 predicciones. Su ajuste final comprende 80 redes GRU/TCN nuevas y 40 LSTM idénticas reutilizadas; el test de 608 fechas reúne 72.960 predicciones. El baseline inicial **GRU/D7 de 32 unidades** obtiene RMSE 0,00377511 ± 0,00019410 y R² 0,854173 en ese test. Es una receta y una cohorte diferentes de la fase II; comparar métricas entre los dos periodos no identifica una mejora causada por HPO.

Los resultados y contrastes de fase I permanecen exploratorios. Su [ejecutor portable](scripts/run_gru_lstm_tcn.py) y los [originales firmados](experiments/gru_lstm_tcn_20261105/original_registration/) se preservan. Una reproducción fresca de esa fase ajusta 120 modelos finales; no se confunde con los 80 nuevos y 40 reutilizados del experimento observado.

## Literatura y distribución

La revisión identifica 40 referencias y siete estudios cripto completos: GRU aparece como comparador independiente en 3/7, LSTM en 5/7 y TCN en 1/7. Las verificaciones editoriales, tipos de publicación y límites de no redundancia se mantienen explícitos. Un nuevo experimento no aumenta esas frecuencias.

El repositorio incluye los derivados diarios de desarrollo 2023–2024. Los ZIP de entrega los omiten y documentan su generación con los archivos oficiales. Se publican código, metadatos, agregados y figuras; las ilustraciones temporales fijan la semilla antes de comparar errores. Los derivados completos de evaluación, predicciones individuales, pesos y escaladores permanecen fuera de la exportación pública. Construir el libro no entrena redes ni abre datos de mercado.

[BUILD.md](BUILD.md) distingue clon y restauración conjunta de productos 8.2, 8.3 y 8.5. Sus manifiestos se verifican desde `book/_static/` en las descargas y también desde `results/` en el repositorio. Un estudio BTCUSDT y dos meses de reserva no representan otros mercados; la significación entre semillas y la incertidumbre temporal se interpretan por separado.

La reserva de fase II pasa a ser una evaluación conocida una vez publicados sus resultados. Reproducirla no recupera independencia para un futuro modelo diseñado con esa información.

La DE definitiva de GRU–TCN supera aproximadamente 1,64 veces su cota de planificación obtenida con el piloto. Junto con el rechazo de normalidad, esto limita la transferencia de los supuestos: el R se calculó antes de la ejecución, pero no se afirma potencia alcanzada o demostrada.
