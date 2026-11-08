# DeepLearning_202630

Este repositorio reúne **tres Jupyter Books**, cada uno con índice, configuración y dirección propios:

| Entregable | Fuentes | Dirección del libro | Estado |
|---|---|---|---|
| E1 | [`book/`](book/) | [Abrir E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) | Informe y estudios realizados |
| E2 | [`book_e2/`](book_e2/) | [Abrir E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/intro.html) | Diez modelos vigentes y candidato CNN-LSTM con atención; benchmark vigente pendiente |
| E3 | [`book_e3/`](book_e3/) | [Abrir E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html) | Continuidad CNN-LSTM con atención propuesta; guía específica pendiente |

[Selector de los tres entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html). La infraestructura de compilación es común; cada JBook se construye desde su propia carpeta. Los apartados experimentales que siguen corresponden a **E1**.

La [sección 5 de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/05_paper_guia.html) contiene la réplica externa de TimeMixer y el piloto sobre tres modelos en ETTh1. Su [paquete descargable](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip) reúne código y evidencias; estas ejecuciones no modifican los resultados de E1 ni completan el benchmark Bitcoin D7.

La [sección 6 de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/06_benchmark_final.html) prepara la comparación tabular, la eficiencia, la inferencia y la discusión del benchmark. Su [paquete de análisis y plantillas](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/benchmark_analysis_download/Benchmark_E2_analisis_plantillas.zip) contiene código verificable y pruebas con fixtures sintéticos. Las métricas y el ranking Bitcoin continúan pendientes de ejecuciones válidas.


Los [protocolos históricos de E2](book_e2/sections/09_protocolo_registrado.md) conservan sus registros y recibos. No definen el catálogo actual ni acreditan el nuevo híbrido. La nueva especificación, implementación, presupuesto y preregistro siguen pendientes; no se alteran los datos o resultados previos.


Proyecto de Deep Learning, 2026-30: **E1 principal con GRU32, LSTM32 y MLP16; E2/E3 proponen CNN-1D + LSTM + atención temporal**.

[Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [Estudio D7 planificado](https://jassan-bit.github.io/DeepLearning_202630/sections/09_estudio_planificado.html) · [Entregables descargables](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1) · [Reproducción](BUILD.md).

El objetivo es σ₇, desviación estándar poblacional de siete retornos logarítmicos diarios de Close, sin anualización, con horizonte de un día. Las entradas son cinco canales OHLCV diarios y la validación emplea los índices forward chaining de `timeseries-cv`, proyecto de DidierRLopes y Filipe R. Ramos, con historia inicial 2023 y embargo temporal.

## Archivo de fases anteriores

Los apartados siguientes documentan ejecuciones previas y no sustituyen el E1 principal de 608 fechas/R309 de `book/sections/11_reejecucion_608.md`. Sus recetas y resultados conservan su identidad. El [contrato sincronizado](book_e2/sections/17_contrato_sincronizado.md) define la continuidad editorial vigente; no registra ni ejecuta el nuevo híbrido.

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

| Modelo final | RMSE, media ± DE | R² medio |
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

El corte bibliográfico del **05/11/2026** conserva **40 referencias de antecedentes y siete estudios cripto con texto completo verificado**: GRU aparece como comparador independiente en **3/7**, LSTM en **5/7** y TCN en **1/7**. Las verificaciones editoriales, tipos de publicación y límites de no redundancia se mantienen explícitos.

La [ampliación nativa del 06/11/2026](references/literature_e1/native_search_20261106/README.md) incorpora cuatro exportaciones WoS/Scopus: **244 ocurrencias y 182 grupos provisionales** por DOI/título, con [recuentos](references/literature_e1/native_search_20261106/summary.json), [registros](references/literature_e1/native_search_20261106/records.csv) y [cribado](references/literature_e1/native_search_20261106/screening.csv) separados del corte histórico. El recuento de grupos no equivale al de estudios elegibles o nuevos textos completos verificados. La lectura de los restantes originales, la comprobación de fechas efectivas y la trazabilidad IEEE permanecen abiertas. Un resumen o un nuevo experimento no aumenta las frecuencias históricas. El [capítulo 3](book/sections/03_estado_arte.md) y las [descargas de literatura](book/sections/08_entregables.md) conservan ambas procedencias.

El primer complemento del 06/11 llevó la bibliografía a **42 referencias**, tras revisar [dos originales completos adicionales](references/literature_e1/primary_reviews_20261106/README.md): TCN/LSTM para un índice CVI de volatilidad implícita y redes como componentes híbridos GARCH. Las fichas distinguen sus objetivos y roles; no se cambia el denominador histórico a nueve ni se atribuyen a estos textos comparadores autónomos adicionales de σ₇.

La actualización añade el [cribado editorial por título/resumen](references/literature_e1/editorial_screening_20261106/README.md), una [matriz de selección de familias](references/literature_e1/selection_synthesis_20261106/README.md) y la [comparabilidad de datos y objetivos](references/e1_dataset_comparison_20261106/README.md). Los Q–Q, Friedman y la sensibilidad Wilcoxon se incorporan como [suplemento estadístico posterior](results/e1_statistics_supplement_20261106/README.md), calculado desde métricas guardadas. No modifican el protocolo, los resultados primarios ni la elección del baseline, y no añaden entrenamientos.

El repositorio incluye los derivados diarios de desarrollo 2023–2024. Los ZIP de entrega los omiten y documentan su generación con los archivos oficiales. Se publican código, metadatos, agregados y figuras; las ilustraciones temporales fijan la semilla antes de comparar errores. Los derivados completos de evaluación, predicciones individuales, pesos y escaladores permanecen fuera de la exportación pública. Construir el libro no entrena redes ni abre datos de mercado.

[BUILD.md](BUILD.md) distingue clon y restauración conjunta de productos 8.2, 8.3 y 8.5. Sus manifiestos se verifican desde `book/_static/` en las descargas y también desde `results/` en el repositorio. Un estudio BTCUSDT y dos meses de reserva no representan otros mercados; la significación entre semillas y la incertidumbre temporal se interpretan por separado.

La reserva de fase II pasa a ser una evaluación conocida una vez publicados sus resultados. Reproducirla no recupera independencia para un futuro modelo diseñado con esa información.

La DE definitiva de GRU–TCN supera aproximadamente 1,64 veces su cota de planificación obtenida con el piloto. Junto con el rechazo de normalidad, esto limita la transferencia de los supuestos: el R se calculó antes de la ejecución, pero no se afirma potencia alcanzada o demostrada.

El corte bibliográfico de §3.7 reunió **50 entradas**, con niveles de lectura separados. La [recuperación posterior por internet](references/literature_e1/internet_closure_20261106/README.md) revisa los 22 casos: cuatro originales y una transcripción completa, cinco decisiones de elegibilidad cerradas por exclusión/contexto y 17 casos pendientes de original o suplemento en ese corte. Los cuatro candidatos complementarios están separados. El [registro de intentos nativos](references/literature_e1/search_recovery_20261106/README.md) documenta bloqueos, cuatro exportaciones verificadas y doce cadenas propuestas no ejecutadas; no acredita IEEE ni recupera cadenas, fechas o conteos históricos faltantes.

El [modelo final](book/sections/09_estudio_planificado.md) incorpora una reproducción posterior de curvas con las 465 semillas por familia y sólo desarrollo 2023–2024, sin puntuar de nuevo la reserva ni cambiar el benchmark. El [contexto académico de δ](results/e1_academic_margin_20261106/README.md) añade escalas de variabilidad y mantiene el 5% fijo como elección normativa.

La bibliografía canónica actual contiene **60 referencias**. La [actualización de acceso gratuito](references/literature_e1/open_access_replacement_20261107/README.md) revisa tres PDF suministrados y aporta alternativas abiertas por tema y función. La cola nativa conserva 22 identidades: ocho decisiones de exclusión/contexto y 14 informes no recuperados al corte. Acceso, lectura, licencia y elegibilidad quedan separados; la búsqueda web no acredita el historial nativo de WoS/Scopus/IEEE. Esta ampliación literaria no reentrena modelos ni modifica las curvas o resultados científicos históricos.

La [bitácora provisional](references/literature_e1/provisional_native_log_20261107/README.md) registra cotas inferiores de resultados compatibles con los identificadores únicos exportados: WoS ≥8/≥59 y Scopus ≥19/≥158. El total IEEE y los campos nativos no recuperados permanecen desconocidos; las fechas de referencia y cadenas propuestas no se presentan como consultas ejecutadas. El registro documental no modifica el benchmark ni los totales bibliográficos verificados.
