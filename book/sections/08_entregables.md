# 8. Entregables: productos y evidencias

```{raw} html
<div class="admonition note" id="descargar-entregable-1">
  <p class="admonition-title">Descargas del Entregable 1 · GRU–LSTM–TCN</p>
  <p><a class="btn btn-primary btn-lg" href="../_static/e1_downloads/entregable1_completo.zip" download="entregable1_completo.zip">Descargar entrega completa · ZIP</a></p>
  <p>Los siete productos reúnen el benchmark inicial, el estudio D7 con búsqueda equitativa, piloto y evaluación preespecificada, y la ampliación bibliográfica nativa del 6 de noviembre. También pueden descargarse individualmente.</p>
</div>
```

Esta unidad reúne el informe, código, instrucciones de datos y evidencias de **GRU, LSTM y TCN**, distinguiendo **fase I exploratoria** y **fase II D7 planificada**. Sus resultados tienen recetas, periodos y registros propios; el nuevo estudio no vuelve prospectivo el test anterior.

Los botones descargan directamente desde el JBook. El [catálogo](../_static/e1_downloads/catalogo.json) y las [sumas SHA-256](../_static/e1_downloads/SHA256SUMS.txt) identifican los archivos de esta versión.

| Ítem | Producto | Estado y alcance | Descarga directa |
|---|---|---|---|
| {ref}`8.1 <entrega-8-1>` | Informe en Jupyter Book | Capítulos, figuras y resultados de ambas fases | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_1_informe_html.zip" download="8_1_informe_html.zip">Descargar ZIP</a> |
| {ref}`8.2 <entrega-8-2>` | Código EDA, modelos y estadística | Ejecutores separados para fase I y fase II; pruebas, locks y guía de restauración | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_2_codigo.zip" download="8_2_codigo.zip">Descargar ZIP</a> |
| {ref}`8.3 <entrega-8-3>` | Dataset: instrucciones y scripts | Fuente oficial, descargador, manifiestos y calendario | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_3_dataset_instrucciones.zip" download="8_3_dataset_instrucciones.zip">Descargar ZIP</a> |
| {ref}`8.4 <entrega-8-4>` | Síntesis de literatura y flujo | 42 referencias actuales; corte histórico de siete estudios y ampliación nativa WoS/Scopus con 244 ocurrencias | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_4_literatura.zip" download="8_4_literatura.zip">Descargar ZIP</a> |
| {ref}`8.5 <entrega-8-5>` | Benchmark, baseline e inferencia | Agregados del benchmark inicial y del estudio D7 preespecificado | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_5_benchmark.zip" download="8_5_benchmark.zip">Descargar ZIP</a> |
| {ref}`8.6 <entrega-8-6>` | Dependencias, semillas y configuraciones | Recetas, búsqueda, piloto, dimensionamiento y configuraciones por fase | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_6_configuraciones.zip" download="8_6_configuraciones.zip">Descargar ZIP</a> |
| {ref}`8.7 <entrega-8-7>` | Métrica, margen, independencia y partición | Registros previos de fase II y cronología original de fase I | <a class="btn btn-primary btn-sm" href="../_static/e1_downloads/8_7_declaracion.zip" download="8_7_declaracion.zip">Descargar ZIP</a> |

Descomprime los archivos antes de abrirlos. El informe HTML se inicia en `informe/intro.html`; si el navegador restringe recursos locales, ejecuta `python -m http.server 8000 --directory informe` y abre `http://localhost:8000`. MathJax y los vínculos a fuentes externas pueden requerir internet. Los ZIP contienen código, metadatos, agregados y figuras. Sus ilustraciones temporales usan semillas fijas declaradas, sin escoger la de menor error. No incluyen brutos de un minuto, vectores individuales de predicción, escaladores o checkpoints. Los derivados diarios de desarrollo 2023–2024 están versionados en el repositorio, pero se omiten de los ZIP y se documenta cómo generarlos.

La [matriz de cierre](protocolo_pendiente.md) distingue disponibilidad de archivos, requisitos metodológicos y límites científicos.

(entrega-8-1)=
## 8.1. Informe en Jupyter Book

El informe desarrolla problema, EDA, literatura, implementación, benchmark inicial, procedimiento, limitaciones y el nuevo estudio D7.

| Componente | Ubicación |
|---|---|
| Problema y definición de σ₇ | [Capítulo 1](01_problema_dataset.md) |
| EDA y estructura temporal | [Capítulo 2](02_eda.md) |
| Selección bibliográfica de GRU, LSTM y TCN | [Capítulo 3](03_estado_arte.md) |
| Implementación y requisitos de la guía | [Capítulo 4](04_implementacion_evaluacion.md) |
| Benchmark exploratorio de fase I | [Capítulo 5](05_benchmark.md) |
| Procedimiento y limitaciones | [Capítulo 6](06_procedimiento.md) y [capítulo 7](07_limitaciones_entregables.md) |
| HPO, piloto, dimensionamiento y nueva evaluación D7 | [Capítulo 9](09_estudio_planificado.md) |

Construir HTML consume evidencia guardada y no ejecuta entrenamientos ni inferencias.

(entrega-8-2)=
## 8.2. Código completo, ejecutable y versionado

[BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) distingue clon del repositorio, restauración de ZIP y fases de cada estudio. Una reproducción escribe en otra salida y conserva los protocolos cerrados.

| Función | Código |
|---|---|
| Adquisición oficial | [download_binance_spot.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/download_binance_spot.py) |
| Arquitecturas GRU, LSTM y TCN | [dl_models_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models_gru_lstm_tcn.py) |
| Registro, HPO, piloto, planificación y evaluación de fase II | [run_gru_lstm_tcn_planned.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn_planned.py) |
| Carga verificada de la nueva reserva | [planned_study_data.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/planned_study_data.py) |
| Estadística y dimensionamiento preespecificados | [planned_study_statistics.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/planned_study_statistics.py) |
| Reproducción portable de fase I | [run_gru_lstm_tcn.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/run_gru_lstm_tcn.py) |
| Estadística de una reproducción de fase I | [analyze_gru_lstm_tcn_run.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_gru_lstm_tcn_run.py) |

El diseño y código de fase II se [publicaron antes del HPO](https://github.com/jassan-bit/DeepLearning_202630/commit/ac7dd7f67c0dec2397b519897cf6da8056b238f5). El [cierre de planificación](https://github.com/jassan-bit/DeepLearning_202630/commit/5a0acfbc2ff582e252f0e6fb18aaf652963221f8) fija las recetas y R antes de los ajustes definitivos. Estos registros no se atribuyen retrospectivamente a fase I, cuyo adaptador público se incorporó después de sus ejecuciones originales.

La fase I conserva 120 trayectorias CV, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones. Su cierre final contiene 80 ajustes nuevos y 40 LSTM idénticos reutilizados. La fase II D7 usa una salida nueva y ajustes frescos: 960 de HPO, 60 de piloto y **2.790 etapas de ajuste definitivo para 1.395 modelos finales**. La puntuación nueva se realiza después del cierre de todos sus pesos y escaladores.

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

**Descargas de la ampliación del 06/11/2026.** Los metadatos nuevos están disponibles directamente en los enlaces de este apartado y en los ZIP del catálogo actualizado. Sus SHA-256 identifican la versión distribuida. El refresco conserva byte a byte los productos 8.3, 8.5 y 8.6, y actualiza informe, código, literatura, declaración y paquete conjunto.

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

El {download}`registro de importación <../_static/literature_e1/native_search_20261106/README.md>`, el [manifiesto de fuentes](../_static/literature_e1/native_search_20261106/source_manifest.json), los [recuentos](../_static/literature_e1/native_search_20261106/summary.json), los [registros normalizados](../_static/literature_e1/native_search_20261106/records.csv) y el [cribado](../_static/literature_e1/native_search_20261106/screening.csv) permiten revisar el paso de exportación a candidatos. El ZIP 8.4 reúne estos archivos, las ocurrencias, la deduplicación, la cola de lectura y el flujo de esta ampliación, junto con la evidencia histórica. El autor confirmó la correspondencia TCN (ocho/19 registros), GRU/LSTM (59/158) y filtro 2019–2026; las cadenas literales y cobertura total siguen sin acreditarse por los archivos.

La lectura de **dos originales completos adicionales** lleva la bibliografía actual a **42 referencias**: Nguyen et al. (2023) compara TCN/LSTM para CVI, un índice de volatilidad implícita; Zahid et al. (2022) emplea redes como componentes de modelos híbridos GARCH {cite:p}`nguyen2023cvi,zahid2022hybrid`. El {download}`registro de lectura <../_static/literature_e1/primary_reviews_20261106/README.md>`, su [matriz](../_static/literature_e1/primary_reviews_20261106/evidence_matrix.csv) y las [métricas extraídas](../_static/literature_e1/primary_reviews_20261106/metrics.csv) forman parte del ZIP 8.4. Los objetivos, papeles de modelo y límites metodológicos se mantienen separados; no se modifica el corte N=7 ni se presentan nueve estudios como comparadores autónomos homogéneos de σ₇.

El [capítulo 3](03_estado_arte.md) distingue antecedentes, corte histórico y ampliación. Permanecen abiertas la lectura de los restantes originales y clasificación de sus objetivos, el respaldo independiente de TCN, la justificación de GRU/LSTM y la trazabilidad IEEE. Encontrar una arquitectura en un resumen no actualiza las frecuencias de texto completo. El HPO y el piloto tampoco las aumentan ni convierten las recetas propias en réplicas exactas de artículos. La revisión no se declara cerrada y los nuevos resultados no se seleccionan por favorecer la terna o el ranking experimental.

(entrega-8-5)=
## 8.5. Tabla comparativa, baseline e inferencia estadística

El [capítulo 9](09_estudio_planificado.md) presenta la evaluación preespecificada de fase II D7, con recetas seleccionadas por HPO, **465 semillas por familia**, contrastes primarios Holm-3 y análisis temporal secundario. El baseline seleccionado antes del piloto y la reserva es **GRU/D7 de 16 unidades**, con RMSE de HPO enero–octubre **0,00513620 ± 0,00003068**. La elección no se reabre según el ranking de la reserva.

El ZIP 8.5 incluye los [agregados de fase II](../_static/gru_lstm_tcn_planned_20261105/summary.json), junto con los registros de fase I. Cada directorio conserva su manifiesto de integridad y su cohorte.

La reserva de fase II tiene los siguientes resultados:

| Modelo de fase II | RMSE, media ± DE | R² medio | MAE medio |
|---|---:|---:|---:|
| GRU | 0,00511696 ± 0,00077519 | 0,348891 | 0,00391028 |
| LSTM | 0,00565123 ± 0,00056569 | 0,215843 | 0,00429363 |
| TCN | 0,00433967 ± 0,00099896 | 0,517956 | 0,00338399 |
| Persistencia | 0,00384478 | 0,640632 | 0,00212123 |

Las redes resumen 465 semillas sobre las mismas sesenta fechas; persistencia es determinista. Los [contrastes e intervalos de fase II](09_estudio_planificado.md) distinguen aleatoriedad de entrenamiento e incertidumbre temporal.

En esta reserva, **persistencia tiene menor RMSE media que las tres redes y TCN el menor error entre las redes**. El baseline operativo permanece GRU/D7-c03, fijado por desarrollo. Las diferencias entre semillas no equivalen a superioridad temporal: ninguna comparación DM rechaza tras Holm-6 y el bootstrap temporal familiar no acredita una mejora superior al 5 %. Los diagnósticos rechazan normalidad de las diferencias; las t de 465 pares se interpretan como aproximaciones, con sus límites en el capítulo 9.

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

| Modelo de fase II | Receta seleccionada | Parámetros entrenables |
|---|---|---:|
| GRU | Una capa unidireccional de 16 unidades; LR 0,003; dropout 0,1 | 1.121 |
| LSTM | Una capa unidireccional de 16 unidades; LR 0,003; dropout 0 | 1.489 |
| TCN | Tres bloques residuales, dos convoluciones causales por bloque; ancho 8; kernel 3; dilataciones 1/2/4; LR 0,003; dropout 0 | 1.185 |

Se comparten AdamW, batch 32, decaimiento 0,0001, clipping 1, máximo de cien épocas y paciencia 12. La búsqueda conserva sus **24 candidatos**, 48 trayectorias de validación y 480 segmentos mensuales. Sus semillas son **6101–6102**; el piloto usa **7101–7110** y la comparación definitiva **8101–8565**, sin solapamiento entre etapas. R=465 resulta del dimensionamiento registrado, sin parada por significación.

La fase I conserva sus [1.440 configuraciones CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv) y [120 registros finales](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv), con anchos 32/32/16 y semillas 4101–4110. La fase II tiene su [catálogo definitivo](../_static/gru_lstm_tcn_planned_20261105/final_configurations.csv), [recetas](../_static/gru_lstm_tcn_planned_20261105/selected_recipes.json), [runtime](../_static/gru_lstm_tcn_planned_20261105/runtime.json) y [dimensionamiento](../_static/gru_lstm_tcn_planned_20261105/dimensioning.json). El ZIP 8.6 reúne dependencias, semillas y metadatos, sin pesos o valores de escaladores.

(entrega-8-7)=
## 8.7. Declaración de métrica, margen, independencia y partición

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
