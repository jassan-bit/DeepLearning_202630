# 8. Entregables: productos y evidencias

```{raw} html
<div class="admonition note" id="descargar-entregable-1">
  <p class="admonition-title">Descargas del Entregable 1 · LSTM–MLP–GRU</p>
  <p><a class="btn btn-primary btn-lg" href="../_static/e1_downloads/entregable1_completo.zip" download="entregable1_completo.zip">Descargar entrega completa · ZIP</a></p>
  <p>Los siete productos reúnen la selección bibliográfica LSTM–MLP–GRU, sus resultados históricos compatibles y el registro de una nueva ejecución común. El estudio anterior con TCN conserva sus resultados como comparación complementaria. También pueden descargarse individualmente.</p>
</div>
```

Esta unidad reúne el informe, código, instrucciones de datos y evidencias del **núcleo LSTM–MLP–GRU**. El [capítulo 10](10_nucleo_mlp.md) documenta su nueva ejecución común; el [capítulo 5](05_benchmark.md) conserva sus resultados históricos compatibles. **TCN pertenece al estudio complementario anterior** del [capítulo 9](09_estudio_planificado.md), cuyos resultados permanecen separados. La evaluación de septiembre–octubre de 2026 ya es conocida: una ejecución nueva no recupera independencia confirmatoria.

Los botones descargan directamente desde el JBook. El [catálogo](../_static/e1_downloads/catalogo.json) y las [sumas SHA-256](../_static/e1_downloads/SHA256SUMS.txt) identifican los archivos de esta versión.

| Ítem | Producto | Estado y alcance | Descarga directa |
|---|---|---|---|
| {ref}`8.1 <entrega-8-1>` | Informe en Jupyter Book | Núcleo LSTM–MLP–GRU, ejecución común e historia complementaria de TCN | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_1_informe_html.zip" download="8_1_informe_html.zip">Descargar ZIP</a> |
| {ref}`8.2 <entrega-8-2>` | Código EDA, modelos y estadística | Ejecutores separados por estudio, pruebas, locks y guía de restauración | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_2_codigo.zip" download="8_2_codigo.zip">Descargar ZIP</a> |
| {ref}`8.3 <entrega-8-3>` | Dataset: instrucciones y scripts | Fuente oficial, descargador, manifiestos y calendario | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_3_dataset_instrucciones.zip" download="8_3_dataset_instrucciones.zip">Descargar ZIP</a> |
| {ref}`8.4 <entrega-8-4>` | Síntesis de literatura y flujo | Bibliografía de 70 referencias, bitácora nativa acreditada y ampliación gratuita TCN/MLP con conciliación acumulada | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_4_literatura.zip" download="8_4_literatura.zip">Descargar ZIP</a> |
| {ref}`8.5 <entrega-8-5>` | Benchmark, baseline e inferencia | Agregados por cohorte del núcleo actual; resultados complementarios y controles de cierre | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_5_benchmark.zip" download="8_5_benchmark.zip">Descargar ZIP</a> |
| {ref}`8.6 <entrega-8-6>` | Dependencias, semillas y configuraciones | Recetas, búsqueda, piloto, dimensionamiento y configuraciones por fase | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_6_configuraciones.zip" download="8_6_configuraciones.zip">Descargar ZIP</a> |
| {ref}`8.7 <entrega-8-7>` | Métrica, margen, independencia y partición | Cierre y procedencia de la nueva ejecución; registros históricos separados | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_7_declaracion.zip" download="8_7_declaracion.zip">Descargar ZIP</a> |

Descomprime los archivos antes de abrirlos. El informe HTML se inicia en `informe/intro.html`; si el navegador restringe recursos locales, ejecuta `python -m http.server 8000 --directory informe` y abre `http://localhost:8000`. MathJax y los vínculos a fuentes externas pueden requerir internet. Los ZIP contienen código, metadatos, agregados y figuras. Sus ilustraciones temporales usan semillas fijas declaradas, sin escoger la de menor error. No incluyen brutos de un minuto, vectores individuales de predicción, escaladores o checkpoints. Los derivados diarios de desarrollo 2023–2024 están versionados en el repositorio, pero se omiten de los ZIP y se documenta cómo generarlos.

La [matriz de cierre](protocolo_pendiente.md) distingue disponibilidad de archivos, requisitos metodológicos y límites científicos.

(entrega-8-1)=
## 8.1. Informe en Jupyter Book

El informe desarrolla problema, EDA, literatura, implementación, benchmark histórico compatible, procedimiento, limitaciones y la nueva ejecución común del núcleo.

| Componente | Ubicación |
|---|---|
| Problema y definición de σ₇ | [Capítulo 1](01_problema_dataset.md) |
| EDA y estructura temporal | [Capítulo 2](02_eda.md) |
| Selección bibliográfica de LSTM, MLP y GRU; TCN complementaria | [Capítulo 3](03_estado_arte.md) |
| Implementación y requisitos de la guía | [Capítulo 4](04_implementacion_evaluacion.md) |
| Benchmark histórico compatible LSTM–MLP–GRU | [Capítulo 5](05_benchmark.md) |
| Procedimiento y limitaciones | [Capítulo 6](06_procedimiento.md) y [capítulo 7](07_limitaciones_entregables.md) |
| Estudio complementario registrado con TCN | [Capítulo 9](09_estudio_planificado.md) |
| Nueva ejecución común de LSTM–MLP–GRU | [Capítulo 10](10_nucleo_mlp.md) |

Construir HTML consume evidencia guardada y no ejecuta entrenamientos ni inferencias.

(entrega-8-2)=
## 8.2. Código completo, ejecutable y versionado

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) distingue clon del repositorio, restauración de ZIP y fases de cada estudio. Una reproducción escribe en otra salida y conserva los protocolos cerrados.

| Función | Código |
|---|---|
| Adquisición oficial | [download_binance_spot.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/download_binance_spot.py) |
| Arquitecturas del núcleo LSTM–MLP–GRU | [e1_core_models_mlp.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e1_core_models_mlp.py) |
| Registro, HPO, piloto, planificación y evaluación común del núcleo | [run_e1_core_mlp_extension.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_e1_core_mlp_extension.py) |
| Estadística de la nueva ejecución | [e1_core_mlp_statistics.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e1_core_mlp_statistics.py) |
| Exportación pública con controles de cierre | [export_e1_core_mlp_extension.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/export_e1_core_mlp_extension.py) |
| Arquitecturas complementarias anteriores con TCN | [dl_models_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models_gru_lstm_tcn.py) |
| Ejecutor del estudio complementario registrado | [run_gru_lstm_tcn_planned.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn_planned.py) |
| Carga verificada de la nueva reserva | [planned_study_data.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/planned_study_data.py) |
| Estadística y dimensionamiento preespecificados | [planned_study_statistics.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/planned_study_statistics.py) |
| Reproducción portable de fase I | [run_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn.py) |
| Estadística de una reproducción de fase I | [analyze_gru_lstm_tcn_run.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_gru_lstm_tcn_run.py) |

El diseño del núcleo exige publicar fuentes antes de sus nuevos ajustes, completar una búsqueda equitativa y un piloto nuevo, publicar el dimensionamiento antes del ajuste definitivo y congelar los pesos antes de puntuar. La [ejecución del capítulo 10](10_nucleo_mlp.md) conserva esos recibos y su estado real; ninguna métrica antigua de GRU/LSTM se mezcla con una nueva MLP para presentar una comparación común.

En el estudio complementario con TCN, el diseño y código se [publicaron antes de su HPO](https://github.com/jassan-bit/DeepLearning_202630/commit/ac7dd7f67c0dec2397b519897cf6da8056b238f5). Su [cierre de planificación](https://github.com/jassan-bit/DeepLearning_202630/commit/5a0acfbc2ff582e252f0e6fb18aaf652963221f8) fija recetas y R antes de sus ajustes definitivos. Estos registros conservan su contexto y no se atribuyen a la nueva ejecución del núcleo ni retrospectivamente a fase I.

El estudio anterior con TCN conserva 120 trayectorias CV, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones. Su cierre final contiene 80 ajustes nuevos y 40 LSTM idénticos reutilizados. Su evaluación D7 registrada conserva 960 ajustes HPO, 60 de piloto y **2.790 etapas definitivas para 1.395 modelos**, con su propia cohorte y configuración. Los conteos nuevos de LSTM–MLP–GRU proceden de su ejecución separada y del exportador cerrado.

(entrega-8-3)=
## 8.3. Dataset, instrucciones y scripts de obtención

La fuente es Binance Vision, BTCUSDT spot, OHLCV de un minuto. [data/README.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md) y el descargador identifican archivos y checksums oficiales.

| Periodo UTC | Función | Regla |
|---|---|---|
| 01/01/2023–31/12/2023 | Historia inicial de ambas fases | Orden cronológico |
| 01/01/2024–31/12/2024 | Desarrollo y CV de fase I | Doce bloques mensuales y entrenamiento creciente |
| 01/01–31/10/2024 | HPO de fase II D7 | Diez bloques comunes, ocho candidatos y dos semillas por familia |
| 01/11–30/12/2024 | Piloto de fase II | 60 días, recetas congeladas, diez semillas nuevas y pesos fijos |
| 01/01/2025–31/08/2026 | Test histórico de fase I | 608 fechas; periodo ya observado, alcance exploratorio |
| 01/09–30/10/2026 | Nueva reserva de fase II | Asignación explícita previa; 60 fechas y una puntuación por modelo cerrado |

La adquisición anterior recorrió los CSV de septiembre–octubre para auditar integridad, timestamps y esquema. Según los registros conservados, no se habían utilizado para construir objetivos o evaluar modelos. No se afirma que los archivos nunca se abrieron ni que las predicciones fueron emitidas en vivo. Octubre de 2026 termina en el **día 30**, porque el 31 no estaba adquirido en esta instantánea.

Se agregan días UTC completos y se conserva el calendario, sin imputar faltantes. La etiqueta es σ₇ poblacional de retornos logarítmicos diarios, `ddof=0`, sin anualización. Las entradas terminan antes del día objetivo. Agosto de 2026 aporta contexto observable para las primeras ventanas de fase II; no amplía sus 60 resultados de evaluación.

(entrega-8-4)=
## 8.4. Síntesis de literatura y flujo de búsqueda

**Corte y procedencia de las descargas.** El catálogo identifica el commit fuente de la actualización y los SHA-256 de cada ZIP. El producto 8.4 reúne la bibliografía canónica ampliada de **70 entradas**, la ingesta nativa, las fichas primarias, las decisiones editoriales, la recuperación posterior de originales y las nuevas fuentes gratuitas TCN/MLP. La ampliación de trece obras contiene tres referencias ya citadas y diez nuevas; la conciliación acumulada mantiene las identidades, el papel de cada arquitectura y el nivel de acceso. Estos documentos amplían la evidencia sin convertir un título, una cita o un fragmento indexado en lectura completa. Los manifiestos históricos se conservan con sus fechas y denominadores.

El **corte histórico del 05/11/2026** conserva **40 referencias de antecedentes y siete estudios cripto con texto completo verificado**. Los siete incluyen Bitcoin y uno otras diecinueve criptomonedas; los subconjuntos se solapan. Las frecuencias siguientes describen ese corte y su criterio de comparador independiente.

| Modelo | Comparadores independientes en siete estudios al 05/11 | Respaldo publicado directo |
|---|---:|---|
| GRU | 3/7 | Akbiyik et al. (2023), §5 y Tabla 2; Miura et al. (2019), Tabla 1 |
| LSTM | 5/7 | Akbiyik et al. (2023); Miura et al. (2019) |
| TCN | 1/7 | Akbiyik et al. (2023), TCN independiente; D-TCN es su propuesta avanzada |

Las tres familias aparecen como comparadores de volatilidad de Bitcoin en WSDM/ACM {cite:p}`akbiyik2023`; GRU y LSTM también en ISNN/Springer LNCS {cite:p}`miura2019`. Se distingue congreso, revista y manuscrito, sin inferir indexación individual o Q1 sólo por la editorial.

La **ampliación del 06/11/2026** incorpora cuatro archivos nativos suministrados de Scopus y Web of Science:

| Base | Exportaciones recibidas | Ocurrencias |
|---|---:|---:|
| Scopus | 2 | 19 + 158 = 177 |
| Web of Science | 2 | 8 + 59 = 67 |
| **Total** | **4** | **244** |

La agrupación por DOI/título produce **182 grupos provisionales**. Este recuento organiza el cribado y conserva las apariciones por fuente; no representa 182 artículos incluidos, textos completos revisados o estudios comparables con σ₇ de BTCUSDT. El año nominal de publicación tampoco basta para admitir registros de 2027: se debe verificar Early Access o fecha efectiva disponible antes del corte.

El {download}`registro de importación <../_static/literature_e1/native_search_20261106/README.md>`, el [manifiesto de fuentes](../_static/literature_e1/native_search_20261106/source_manifest.json), los [recuentos](../_static/literature_e1/native_search_20261106/summary.json), los [registros normalizados](../_static/literature_e1/native_search_20261106/records.csv) y el [cribado](../_static/literature_e1/native_search_20261106/screening.csv) permiten revisar el paso de exportación a candidatos. El refresco preparado de 8.4 reúne estos archivos, las ocurrencias, la deduplicación, la cola de lectura y el flujo de esta ampliación, junto con la evidencia histórica. El autor confirmó la correspondencia TCN (ocho/19 registros), GRU/LSTM (59/158) y filtro 2019–2026; las cadenas literales y cobertura total siguen sin acreditarse por los archivos.

El primer complemento, de **dos originales completos adicionales**, llevó la bibliografía a **42 referencias**: Nguyen et al. (2023) compara TCN/LSTM para CVI, un índice de volatilidad implícita; Zahid et al. (2022) emplea redes como componentes de modelos híbridos GARCH {cite:p}`nguyen2023cvi,zahid2022hybrid`. El {download}`registro de lectura <../_static/literature_e1/primary_reviews_20261106/README.md>`, su [matriz](../_static/literature_e1/primary_reviews_20261106/evidence_matrix.csv) y las [métricas extraídas](../_static/literature_e1/primary_reviews_20261106/metrics.csv) forman parte del refresco preparado de 8.4. Los objetivos, papeles de modelo y límites metodológicos se mantienen separados; no se modifica el corte N=7 ni se presentan nueve estudios como comparadores autónomos homogéneos de σ₇.

La {download}`recuperación posterior por internet <../_static/literature_e1/internet_closure_20261106/README.md>` revisó los 22 casos pendientes en aquel corte: **cuatro originales completos y una transcripción completa**, cinco decisiones estrictas cerradas por exclusión/contexto y **17 casos todavía pendientes entonces**. La matriz separa acceso, objetivo, papel autónomo o híbrido, receta y decisión. Los cuatro candidatos complementarios tienen su propia procedencia y no se suman a las 22 decisiones ni a las frecuencias históricas.

El producto 8.4 incorpora ahora la {download}`etapa de sustitución por acceso gratuito <../_static/literature_e1/open_access_replacement_20261107/README.md>`, su [catálogo de fuentes](../_static/literature_e1/open_access_replacement_20261107/sources.csv), la [matriz por tema y función](../_static/literature_e1/open_access_replacement_20261107/replacement_map.csv) y la [disposición actual de los 22 casos](../_static/literature_e1/open_access_replacement_20261107/native_dispositions.csv). Los tres PDF recibidos se revisaron sin redistribuirlos. Se cierran ocho decisiones de exclusión/contexto y se registran 14 informes no obtenidos al corte, conservando el historial y distinguiendo las nuevas fuentes abiertas de los documentos originales no recuperados. El código de compilación se distribuye en 8.2 y los resúmenes/manifiestos en 8.7.

El {download}`registro histórico de recuperación de búsquedas <../_static/literature_e1/search_recovery_20261106/README.md>` conserva cuatro intentos de acceso nativo bloqueados, sin inventar totales de resultados. Sus doce cadenas propuestas no se ejecutaron en esa etapa. Las cuatro exportaciones recibidas siguen verificadas; aquel registro no recuperó sus cadenas y fechas originales ni acreditó IEEE. La ronda nueva se documenta por separado a continuación. También se resuelven dos fechas efectivas de capítulos con año nominal 2027, conservando ambos campos.

El producto 8.4 conserva todos los metadatos de la {ref}`bitácora provisional de §3.9 <provisional-native-log-e1>`: el {download}`procedimiento <../_static/literature_e1/provisional_native_log_20261107/README.md>`, el {download}`CSV de preparación <../_static/literature_e1/provisional_native_log_20261107/provisional_log.csv>`, el [resumen](../_static/literature_e1/provisional_native_log_20261107/summary.json) y el [manifiesto](../_static/literature_e1/provisional_native_log_20261107/manifest.json). Las cotas ≥8/≥59/≥19/≥158 se derivan de exportaciones recibidas; son límites inferiores provisionales del universo compatible, no totales observados ni estimaciones puntuales. En aquella preparación IEEE no tenía consulta acreditada. La referencia documental del 06/11/2026 y las propuestas para repetir se distinguen de fechas y cadenas nativas no verificadas. Esa fotografía histórica no modifica PRISMA, 244 ocurrencias/182 grupos ni el N=7 histórico.

La {ref}`síntesis homogénea del corte anterior de §3.10 <homogeneous-selection-e1>` y la {ref}`ronda nativa documentada de §3.11 <current-native-execution-e1>` tienen directorios propios dentro de 8.4. Aquella síntesis audita 47 identidades y distingue **26 originales primarios leídos y una transcripción completa** de los informes parciales o no recuperados. Su estrato de volatilidad realizada histórica contiene 12 estudios; diez permiten extraer comparadores base, con **LSTM 10, GRU 4, MLP 3 y TCN 1**. Los estratos, la deduplicación y la sensibilidad de calidad de descripción permanecen en su {download}`procedimiento y resultados <../_static/literature_e1/homogeneous_synthesis_20261107/README.md>`. La terna registrada GRU–LSTM–TCN se conserva como contraste entre celdas recurrentes y filtros causales: no se presenta como las tres arquitecturas más frecuentes ni como el mínimo único de complejidad.

La nueva ronda acredita consultas en las tres bases y conserva sus diferentes alcances:

| Base | Total nativo | Registros exportados | Alcance acreditado |
|---|---:|---:|---|
| Web of Science | 271 | 271 UT distintos | La exportación incluye cuatro publicaciones de 2017–2018; no se atribuye un filtro 2019–2026 |
| Scopus | 538 | 538 EID distintos | 2019–2026 en la cadena literal |
| IEEE Xplore | 32 | 25 | Consulta con filtro de acceso abierto y años 2018–2026; exportación parcial, siete resultados no exportados |

El {download}`expediente de ejecución <../_static/literature_e1/native_execution_20261107/README.md>` distingue cadena, fecha, total, tamaño exportado y respaldo de cada campo. El autor confirma el **06/11/2026 en Colombia para las tres consultas**; los nombres de exportación son evidencia adicional, sin acreditar la hora exacta de búsqueda. Los 834 registros exportados se agrupan provisionalmente en 620 grupos por identificadores y título; no son 620 estudios leídos. Esta ronda acredita ejecuciones nuevas y no reconstruye las sesiones de las cuatro exportaciones históricas ni declara concluida una revisión completa de todos los nuevos candidatos.

La **búsqueda web complementaria del 07/11/2026 en Colombia** añade la {ref}`selección de siete trabajos TCN y seis MLP <free-model-expansion-e1>`. El {download}`catálogo de autores, DOI, rutas gratuitas y papeles de modelo <../_static/literature_e1/free_model_expansion_20261107/references.csv>` y sus {download}`referencias BibTeX <../_static/literature_e1/free_model_expansion_20261107/references.bib>` acompañan las fichas de lectura y el registro de consultas web. Akbiyik, Nguyen y Lei ya estaban citados: las trece obras aportan **diez referencias nuevas**. Ge et al. se conserva como preprint; Lensky y Hao tiene publicación en IEEE CAI 2024 y versión de autor gratuita. La lectura indexada de Vo se distingue de una descarga íntegra, y la propuesta híbrida de Agal no se convierte en un comparador TCN autónomo verificado.

La {ref}`conciliación acumulada <homogeneous-expansion-e1>` amplía el registro de 47 a **57 identidades**: **35 originales primarios leídos, una transcripción completa y 21 fuentes con otros niveles de acceso**. El constructo cripto contiene 15 experimentos y 13 con referencias extraíbles. El **estrato actual elegible por reporte tiene diez estudios**, con **LSTM 10, MLP 4, GRU 2 y TCN 1**; este N10 elegible es distinto del N10 descriptivo del corte anterior. Su {download}`matriz de adjudicación <../_static/literature_e1/homogeneous_expansion_20261107/adjudications.csv>` y {download}`frecuencias por estrato <../_static/literature_e1/homogeneous_expansion_20261107/frequencies.csv>` separan evidencia cripto, transferencias financieras, objetivo, lectura y suficiencia de reporte. Siete fuentes seleccionadas para TCN y seis para MLP no representan siete y seis usos elegibles del problema específico ni una estimación de prevalencia: el recuento se deriva de la matriz común. Esta ampliación conserva las consultas nativas, los 244/182 históricos y los resultados del benchmark.

La {download}`decisión de núcleo <../_static/literature_e1/core_selection_20261107/README.md>` selecciona **LSTM–MLP–GRU**, las tres mayores frecuencias observadas del estrato elegible, y conserva TCN como complemento. El {download}`ranking reproducible <../_static/literature_e1/core_selection_20261107/ranking.csv>`, las {download}`plantillas publicadas y adaptaciones propias <../_static/literature_e1/core_selection_20261107/source_anchors.csv>` y los {download}`candidatos de capacidad <../_static/literature_e1/core_selection_20261107/parameter_candidates.csv>` documentan simplicidad y diferencias de sesgo bajo D7 común. Las anchuras MLP 32/128 son candidatas anteriores a su nuevo HPO; la receta histórica de 593 parámetros no se presenta como un modelo final de esta ejecución.

El [capítulo 3](03_estado_arte.md) explica el respaldo del núcleo LSTM/MLP/GRU y el papel complementario de TCN. Encontrar una arquitectura en un resumen o en un componente híbrido no actualiza frecuencias de comparadores independientes. El HPO y el piloto tampoco aumentan esa recurrencia ni convierten las recetas propias en réplicas exactas de artículos.

(entrega-8-5)=
## 8.5. Tabla comparativa, baseline e inferencia estadística

### Núcleo actual LSTM–MLP–GRU

El {download}`paquete histórico compatible <../_static/e1_historical_core_20261107/README.md>` reúne **240 corridas agregadas** de CV2024 y del test histórico 01/01/2025–31/08/2026, cuatro historias D7/14/21/28, tres familias y diez semillas. Sus {download}`144 filas de resumen de métricas <../_static/e1_historical_core_20261107/metric_summary.csv>` y {download}`controles de cohorte <../_static/e1_historical_core_20261107/cohort_checks.json>` permiten revisar esa comparación sin ajustar modelos. Los seis contrastes pareados publicados corresponden sólo a D7 y conservan carácter posterior y exploratorio. No se reconstruyen pronósticos diarios de MLP ni se atribuye a esos agregados un piloto anterior que no tuvieron.

La {ref}`nueva ejecución común cerrada <final-core-mlp-e1>` completa **48 trayectorias HPO y 960 etapas**, **30 trayectorias piloto y 60 etapas**, y **1.395 modelos definitivos y 2.790 etapas**. El piloto exige R=331 y el mínimo registrado eleva el tamaño final a **465 semillas por familia**. Se verificaron las publicaciones de fuente y plan antes de sus ajustes correspondientes y se puntuaron los modelos después de congelar todos los pesos: **83.700 predicciones privadas en 60 fechas**, del 01/09 al 30/10/2026.

| Modelo del núcleo actual | RMSE, media ± DE entre semillas |
|---|---:|
| GRU | 0,00511696 ± 0,00077519 |
| MLP | 0,00516664 ± 0,00048727 |
| LSTM | 0,00565123 ± 0,00056569 |
| Persistencia determinista | 0,00384478 |

**Persistencia presenta menor RMSE que las tres redes.** GRU es el baseline elegido por CV y tiene la menor RMSE media entre redes; su diferencia con MLP no rechaza la hipótesis de igualdad tras Holm-3. La evaluación mantiene carácter histórico y exploratorio porque el calendario ya se conocía. Las diferencias entre semillas no acreditan superioridad sobre otros periodos ni utilidad financiera.

Se distribuyen el {download}`resumen y alcance <../_static/e1_core_mlp_extension_20261107/summary.json>`, las {download}`métricas agregadas <../_static/e1_core_mlp_extension_20261107/metric_summary.csv>`, las {download}`métricas por semilla <../_static/e1_core_mlp_extension_20261107/run_metrics.csv>`, los {download}`contrastes pareados <../_static/e1_core_mlp_extension_20261107/pairwise_rmse.csv>` y la {download}`cronología comprobada <../_static/e1_core_mlp_extension_20261107/chronology.json>`. Las {download}`curvas de entrenamiento y validación <../_static/e1_core_mlp_extension_20261107/learning_curves.png>` proceden de **44.201 filas reales de pérdidas**; están disponibles también las {download}`curvas por corrida <../_static/e1_core_mlp_extension_20261107/learning_curves_by_run.csv>` y su {download}`resumen <../_static/e1_core_mlp_extension_20261107/learning_curve_summary.csv>`. El reajuste no usa validación de reserva. El {download}`manifiesto de 25 archivos públicos <../_static/e1_core_mlp_extension_20261107/manifest.json>` acredita hashes y tamaños; los objetivos y pronósticos diarios, pesos y escaladores permanecen privados.

### Estudio complementario anterior con TCN

El [capítulo 9](09_estudio_planificado.md) presenta la evaluación preespecificada del estudio complementario GRU–LSTM–TCN, con recetas seleccionadas por HPO, **465 semillas por familia**, contrastes primarios Holm-3 y análisis temporal secundario. Su baseline seleccionado antes del piloto y de aquella reserva fue **GRU/D7 de 16 unidades**, con RMSE de HPO enero–octubre **0,00513620 ± 0,00003068**. Estos resultados no se atribuyen a las nuevas recetas LSTM–MLP–GRU.

El ZIP 8.5 incluye los [agregados de fase II](../_static/gru_lstm_tcn_planned_20261105/summary.json), junto con los registros de fase I y el suplemento estadístico posterior del [capítulo 9](09_estudio_planificado.md). Cada directorio conserva su manifiesto de integridad y su cohorte. Se añaden las {download}`curvas públicas reproducidas <../_static/e1_curve_reproduction_20261106/README.md>` de las tres recetas finales: 1.395 trayectorias y 2.790 etapas nuevas, sólo con desarrollo 2023–2024. Su comparación de hashes distingue identidad de configuración, curvas y pesos; la ejecución no sustituye los resultados originales ni puntúa la reserva. Los Q–Q, Friedman y la sensibilidad Wilcoxon del suplemento se calculan desde métricas ya guardadas: no formaron parte del análisis registrado ni modifican sus contrastes primarios.

La {download}`auditoría forense <../_static/e1_replay_forensics_20261107/README.md>` añade la búsqueda de originales, la comparación de archivos y los 29 cambios de época de TCN, con causa todavía no identificada. La {download}`repetibilidad actual controlada <../_static/e1_replay_repeatability_20261107/README.md>` ejecuta dos procesos independientes sobre seis celdas diagnósticas: **12 trayectorias, 24 etapas y 18/18 comparaciones canónicas exactas** entre réplicas; las seis celdas satisfacen también la tolerancia fijada antes del ajuste. Coinciden además con esas seis celdas de la reproducción completa anterior. No se leen precios de reserva ni se añaden observaciones independientes al benchmark; el resultado no acredita identidad con los archivos originales ni repetibilidad de las 1.395 trayectorias completas.

La reserva del estudio complementario anterior tiene los siguientes resultados:

| Modelo del estudio complementario | RMSE, media ± DE | R² medio | MAE medio |
|---|---:|---:|---:|
| GRU | 0,00511696 ± 0,00077519 | 0,348891 | 0,00391028 |
| LSTM | 0,00565123 ± 0,00056569 | 0,215843 | 0,00429363 |
| TCN | 0,00433967 ± 0,00099896 | 0,517956 | 0,00338399 |
| Persistencia | 0,00384478 | 0,640632 | 0,00212123 |

Las redes resumen 465 semillas sobre las mismas sesenta fechas; persistencia es determinista. Los [contrastes e intervalos de fase II](09_estudio_planificado.md) distinguen aleatoriedad de entrenamiento e incertidumbre temporal.

En aquella reserva, **persistencia tuvo menor RMSE media que las tres redes y TCN el menor error entre las redes**. El baseline de ese estudio permanece GRU/D7-c03, fijado por desarrollo. Las diferencias entre semillas no equivalen a superioridad temporal: ninguna comparación DM rechaza tras Holm-6 y el bootstrap temporal familiar no acredita una mejora superior al 5 %. Los diagnósticos rechazan normalidad de las diferencias; las t de 465 pares se interpretan como aproximaciones, con sus límites en el capítulo 9.

La tabla siguiente conserva el **test de fase I, 01/01/2025–31/08/2026**, para las recetas iniciales D7:

| Modelo de fase I | RMSE, media ± DE | R², media ± DE | MAE medio | MAPE medio (%) |
|---|---|---|---:|---:|
| **GRU, 32 unidades** | **0,00377511 ± 0,00019410** | **0,854173 ± 0,015331** | **0,00279851** | **18,0667** |
| LSTM, 32 unidades | 0,00502979 ± 0,00026367 | 0,741108 ± 0,026619 | 0,00391888 | 25,3406 |
| TCN, ancho 16 | 0,00417465 ± 0,00037283 | 0,820819 ± 0,032191 | 0,00314430 | 19,8847 |
| Persistencia | 0,00408560 | 0,829605 | 0,00235316 | 13,7553 |

Las redes de esa tabla resumen diez semillas. Persistencia es determinista y no tiene DE de inicialización. Las [métricas originales de test](../_static/gru_lstm_tcn_20261105/metric_summary_test.csv), las [de CV](../_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y la [selección de fase I](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) no cambian. Su inferencia sigue siendo posterior y exploratoria. En esa CV persistencia tiene menor RMSE; en ese test GRU tiene menor RMSE, mientras persistencia conserva mejores MAE y MAPE.

No se comparan directamente RMSE de 608 y 60 fechas para atribuir una ganancia al HPO. La diferencia de cohorte, régimen y longitud impide esa conclusión causal.

(entrega-8-6)=
## 8.6. Dependencias, semillas y configuración de cada corrida

El núcleo actual distribuye su diseño en `experiments/e1_core_mlp_extension_20261107/`, con dependencias fijadas, semillas HPO/piloto/definitivas separadas y una salida privada nueva. La {download}`selección real por desarrollo <../_static/e1_core_mlp_extension_20261107/selection.json>` congela las siguientes recetas D7, con salida Softplus:

| Modelo del núcleo actual | Receta final seleccionada | Parámetros entrenables |
|---|---|---:|
| GRU, c03 | Una capa de 16 unidades; LR 0,003; dropout 0,1 | 1.121 |
| LSTM, c02 | Una capa de 16 unidades; LR 0,003; dropout 0 | 1.489 |
| MLP, c03 | Una capa ReLU de 32 unidades, entrada D7×5 aplanada; LR 0,003; dropout 0,1 | 1.185 |

El {download}`plan del piloto <../_static/e1_core_mlp_extension_20261107/plan.json>` fija R requerido=331 y R final=465; su {download}`recibo de publicación <../_static/e1_core_mlp_extension_20261107/plan_publication.json>` precede al ajuste definitivo. Se conservan el {download}`registro de ejecución <../_static/e1_core_mlp_extension_20261107/registration.json>`, el {download}`runtime observado <../_static/e1_core_mlp_extension_20261107/runtime.json>` y la {download}`procedencia de fuentes <../_static/e1_core_mlp_extension_20261107/source_manifest.json>`. Las configuraciones históricas permanecen separadas. Los pesos y valores de escaladores permanecen privados.

La tabla siguiente conserva **las recetas del estudio complementario anterior con TCN**:

| Modelo complementario anterior | Receta seleccionada | Parámetros entrenables |
|---|---|---:|
| GRU | Una capa unidireccional de 16 unidades; LR 0,003; dropout 0,1 | 1.121 |
| LSTM | Una capa unidireccional de 16 unidades; LR 0,003; dropout 0 | 1.489 |
| TCN | Tres bloques residuales, dos convoluciones causales por bloque; ancho 8; kernel 3; dilataciones 1/2/4; LR 0,003; dropout 0 | 1.185 |

Se comparten AdamW, batch 32, decaimiento 0,0001, clipping 1, máximo de cien épocas y paciencia 12. La búsqueda conserva sus **24 candidatos**, 48 trayectorias de validación y 480 segmentos mensuales. Sus semillas son **6101–6102**; el piloto usa **7101–7110** y la comparación definitiva **8101–8565**, sin solapamiento entre etapas. R=465 resulta del dimensionamiento registrado, sin parada por significación.

La fase I conserva sus [1.440 configuraciones CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv) y [120 registros finales](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv), con anchos 32/32/16 y semillas 4101–4110. La fase II tiene su [catálogo definitivo](../_static/gru_lstm_tcn_planned_20261105/final_configurations.csv), [recetas](../_static/gru_lstm_tcn_planned_20261105/selected_recipes.json), [runtime](../_static/gru_lstm_tcn_planned_20261105/runtime.json) y [dimensionamiento](../_static/gru_lstm_tcn_planned_20261105/dimensioning.json). El ZIP 8.6 reúne dependencias, semillas y metadatos, sin pesos o valores de escaladores. Incluye por separado el protocolo publicado, el lock y los recibos de la reproducción posterior de curvas; conservar las semillas originales no convierte esas nuevas curvas en archivos originales recuperados.

También incorpora `experiments/e1_replay_audit_20261107/` y los registros de la comprobación controlada: publicación previa, registro, entorno y ejecución. Su fuente [79f8dcc](https://github.com/jassan-bit/DeepLearning_202630/commit/79f8dcc1a4c3bfacfd1fc56c0ecb43d8e0185501) fue verificada remotamente antes del registro y de las réplicas. La guía §4.3 pide semillas, dependencias, configuración, un procedimiento reproducible y declarar el alcance del determinismo; no exige recuperar archivos históricos idénticos bit a bit. Se acredita el procedimiento actual y se conserva como límite independiente la comparación numérica imposible con los archivos privados originales ausentes.

(entrega-8-7)=
## 8.7. Declaración de métrica, margen, independencia y partición

La nueva ejecución **LSTM–MLP–GRU** mantiene una métrica común RMSE, una selección de baseline por desarrollo, el margen académico fijo y la regla publicada de dimensionamiento. Evalúa **sesenta fechas históricas ya observadas**, por lo que sus comparaciones se interpretan como exploratorias. Las repeticiones independientes de entrenamiento no añaden calendarios ni mercados. El exportador conserva esa declaración y rechaza resultados mezclados, paneles incompletos, curvas de validación inventadas en el reajuste o metadatos sin sus controles de publicación y cierre.

La declaración siguiente pertenece al **estudio complementario anterior** y conserva su cronología original:

| Elemento | Declaración y alcance de fase II D7 |
|---|---|
| Métrica primaria | RMSE por trayectoria sobre las mismas 60 fechas; media y DE entre 465 semillas |
| Métricas secundarias | MAE, MSE, R², MAPE y MASE, con cobertura y denominadores |
| Partición | HPO enero–octubre de 2024; piloto noviembre–30 de diciembre; ajuste final con desarrollo permitido 2023–2024; nueva reserva 01/09–30/10/2026 |
| Registro inicial | Diseño y código públicos antes del primer ajuste HPO |
| Piloto y dimensionamiento | Diez semillas nuevas por modelo; R=465 calculado y publicado antes de los ajustes definitivos |
| Potencia | Objetivo marginal 0,80 por comparación, bajo supuestos gaussianos y transferencia de DE; no potencia conjunta ni temporal |
| Unidad de repetición | Semilla independiente de entrenamiento, emparejada entre modelos y condicionada al mismo calendario |
| Selección del baseline | GRU/D7 de 16 unidades, por HPO; congelado antes del piloto y de la reserva |
| Inferencia primaria | Tres comparaciones bilaterales de RMSE, Holm-3, IC marginales/familiares y efecto; métodos fijados antes de la reserva |
| Margen de planificación | δ=0,00025772842664981106, ancla académica del 5 % del RMSE CV del GRU inicial; posterior a fase I y previo a fase II |
| Incertidumbre temporal | DM/HAC6/HLN, familia secundaria de seis pares y bootstrap circular de bloques 7, sensibilidad 3/14; sin prometer potencia temporal |
| Equivalencia | No afirmada; ausencia de significación no demuestra igualdad |
| Fuera de alcance | Utilidad financiera, otro mercado independiente o pronósticos emitidos en vivo |

La [cronología de fase II](../_static/gru_lstm_tcn_planned_20261105/chronology.json), su [registro](../_static/gru_lstm_tcn_planned_20261105/registration.json) y el [cierre de pesos](../_static/gru_lstm_tcn_planned_20261105/final_freeze_provenance.json) permiten seguir el orden. La [cronología de fase I](../_static/gru_lstm_tcn_delivery_20261105/declaration_and_chronology.json) conserva su carácter anterior y exploratorio. Los límites bibliográficos y de generalización permanecen en la [matriz de pendientes](protocolo_pendiente.md).

El producto 8.7 añade el expediente de comparabilidad de datasets, el contexto académico del margen fijo y los registros de las reproducciones de curvas, junto con resúmenes y manifiestos de las ampliaciones editoriales, búsquedas nativas, síntesis homogénea, selección del núcleo y auditoría forense. Incluye también los resúmenes/manifiestos del histórico compatible LSTM–MLP–GRU y los registros de cierre de su nueva ejecución; las fichas/tablas literarias completas están en 8.4, los agregados experimentales en 8.5 y los constructores en 8.2. La {download}`justificación del margen <../_static/e1_academic_margin_20261106/README.md>` usa sólo desarrollo y no modifica el δ registrado. El producto 8.3 conserva íntegro el archivo original de adquisición; 8.6 conserva sus miembros históricos y añade la nueva configuración por separado. Los resultados científicos históricos incluidos en los demás productos mantienen sus bytes. La edición `20261107_core_mlp` conserva también los metadatos anteriores `20261107_free_models` y los registros previos del propio 07/11 y del 06/11.

La edición de descargas del núcleo requiere el cierre comprobado de la nueva ejecución. Su catálogo identifica LSTM–MLP–GRU, la receta y el baseline elegidos en esa validación y el R realmente completado. Los modelos, entrypoints y resultados de selección anteriores quedan separados en `historical_study_metadata`; TCN conserva su papel complementario.

También incorpora el [resumen de la bitácora provisional](../_static/literature_e1/provisional_native_log_20261107/summary.json) y su [manifiesto de procedencia](../_static/literature_e1/provisional_native_log_20261107/manifest.json); el CSV y el procedimiento completos se distribuyen en 8.4. Este registro conserva como desconocidos la cadena literal, el total y la fecha nativa, y dejaba abierta la consulta acreditada de IEEE en aquel corte. La ronda nativa posterior conserva esta versión previa y registra su propia fecha de ejecución confirmada.
