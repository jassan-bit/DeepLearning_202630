# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

(e2-entregables-portada)=
## Entregables

**Descargas independientes para ejecutar en el computador del profesor.** Los doce ZIP v3 incluyen su propio `run.py`, instrucciones `LEEME.md`, entradas necesarias y manifiesto SHA-256. Los suplementos posteriores indican abajo sus ejecutores e instrucciones propios. Se puede usar cada paquete por separado, sin clonar el repositorio ni copiar archivos de la página web.

**Edición v3 · 16/11/2026:** incorpora las entradas reales y la evidencia de §65 en el corte identificado por cada manifiesto. El seguimiento posterior de §66 se distribuye en los archivos adicionales indicados abajo; no cambia los bytes de v3. La [edición v2 del 15/11](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v2_20261115/index.json) y la [primera edición](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v1/index.json) conservan sus descargas. El libro offline identifica su fecha de corte.

1. Descargar el ZIP elegido y **extraer todo su contenido** en una carpeta propia.
2. Abrir la carpeta `D01`, `D02`, etc. y leer `LEEME.md`.
3. Ejecutar `python run.py` —o `py -3.12 run.py` en Windows— y abrir el informe o los archivos generados en `output/`.

Se recomienda **Python 3.12**. D02 requiere instalar sus dependencias de CPU antes del primer forward; los otros once comandos predeterminados usan la biblioteca estándar. Cada ZIP incluye también `ejecutar.cmd` y `ejecutar.sh`.

| Archivo independiente | Qué genera al ejecutarlo | Requisitos |
| --- | --- | --- |
| [D01 · Libro completo y fuentes](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/01_informe_jupyter_book.zip) | Extrae el libro HTML offline, fórmulas locales y fuentes para reconstruirlo. | Python estándar; rebuild opcional con Jupyter Book. |
| [D02 · Código completo de modelos](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/02_codigo_modelos.zip) | Ejecuta las 13 arquitecturas en CPU; incluye fuentes de autores y entrenamiento CUDA local opcional. | Instalar requirements_cpu.txt; luego python run.py. |
| [D03 · Dataset, Data Card y contrato](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/03_datos_contrato.zip) | Reconstruye σ₇ y verifica las 1.316 ventanas train655/cal53/test608; incluye OHLCV diario y matrices reales. La descarga de minutos es opcional. | Python estándar. |
| [D04 · Preregistro y auditoría de fuga](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/04_preregistro_auditoria.zip) | Genera el informe con cronología, entradas reales, controles ejecutados y su aceptación; incluye los dos paquetes portables de diarios originales. | Python estándar. |
| [D05 · Ficha de ruta F2a](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/05_ficha_ruta_F2a.zip) | Genera la ficha y tablas de pruebas, unidades, p-valores, efectos e IC disponibles. | Python estándar. |
| [D06 · Benchmark y comparaciones](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/06_benchmark_comparaciones.zip) | Recalcula A2 original desde 81 pronósticos; incluye resultados CPU separados y reanálisis estadístico opcional. | Python estándar; NumPy/SciPy sólo para repetir pruebas. |
| [D07 · Registro experimental y fichas](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/07_registro_experimentos.zip) | Genera tablas y fichas desde configuraciones, semillas, resultados y costes incluidos. | Python estándar. |
| [D08 · Techo, incertidumbre y errores](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/08_techo_incertidumbre.zip) | Reconstruye cobertura original, 6.080 categorías y el proxy de techo con pesos guardados. | Python estándar; reanálisis completo opcional. |
| [D09 · Arquitecturas: NAS no aplica](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/09_arquitecturas_no_aplica.zip) | Genera y valida la justificación del catálogo actual con arquitecturas definidas. | Python estándar; no ejecuta una búsqueda NAS. |
| [D10 · Paper guía y réplica parcial](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/10_paper_guia_replica.zip) | Reproduce los agregados de cinco semillas; incluye ETTh1, autores y entrenamiento CPU opcional. | Python estándar; PyTorch sólo para entrenar. |
| [D11 · Reproducibilidad: Apéndice B](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/11_reproducibilidad.zip) | Genera la lista de 21 ítems, sus estados, evidencia y recibos. | Python estándar. |
| [D12 · Presentación y notas de defensa](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/12_presentacion_defensa.zip) | Genera 16 diapositivas HTML, notas del orador y tablas de apoyo; se pueden imprimir a PDF. | Python estándar; no requiere PowerPoint. |

**Alcance de la ejecución:** los paquetes de análisis reproducen cálculos desde los pronósticos y datos incluidos. D02 verifica los modelos con entradas artificiales y pesos iniciales; las opciones de entrenamiento están separadas y documentadas. Los entrenamientos opcionales requieren las dependencias y el hardware indicados en cada `LEEME.md`; no se lanzan al ejecutar el comando predeterminado.

D01 contiene el HTML, sus fuentes y MathJax locales. Dos paquetes grandes de procedencia se ofrecen como descargas opcionales con commit y SHA fijados; **D04 y D11 los incluyen completos**, con sus diarios originales. La lectura y reconstrucción del libro no requieren esos dos ZIP.

**NAS no aplica al catálogo actual**, incluida TiDE R0-v2; D09 entrega su justificación ejecutable. D12 contiene las diapositivas y notas preparadas, mientras que la realización y evaluación de la defensa oral conservan su carácter pendiente.

[Instrucciones de la publicación](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/LEEME.md) · [Inventario y SHA-256 publicados](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/index.json) · [Comprobación publicada de los doce ZIP](https://jassan-bit.github.io/DeepLearning_202630/entregable2/_static/entregables_independientes_v3_20261116/execution_report.json) · [Data Card original](_static/e2_point1_review_v1/dataset_card_current.json).

La [entrega consolidada](sections/47_entrega_consolidada.md), el [Apéndice B](sections/61_reproduccion_consolidada.md) y la [matriz de requisitos](sections/38_cumplimiento_guia.md) conservan el estado científico de cada componente. Que un archivo se ejecute correctamente no convierte los controles experimentales pendientes en resultados aprobados.

**Evidencia nueva del 16/11/2026:** [entradas reales, controles y comparaciones completadas (§65)](sections/65_cierre_cientifico_real.md). Se recuperaron los veinte ZIP originales, se auditaron las **608 entradas de prueba**, se entrenaron **13 controles con etiquetas permutadas** y se ejecutó la **RQ3 completa en CPU**. El clasificador adversarial detecta cambio de distribución y se conserva su alerta. La nueva repetición CPU de las recetas ganadoras también está completa: 78 ajustes neuronales, tres comparadores y evaluación/análisis propios, sin repetir HPO.

El [suplemento independiente anterior del 16/11](_static/e2_pending_closure_20261116/suplemento_cierre_20261116.zip) conserva sus 18 etapas y su estado histórico. Los doce ZIP v2 mantienen su snapshot del 15/11 y sus bytes originales; no heredan automáticamente los resultados nuevos.

### Archivos independientes del seguimiento de §66

Estos cinco ZIP incluyen sus propias entradas e instrucciones; no requieren descargar otro entregable ni clonar el repositorio. Los reanálisis parten de pronósticos incluidos; el paquete CPU incluye pesos y fuentes para generar pronósticos reales. Cada modo conserva su alcance y recibo.

| Descarga | Ejecución y alcance |
| --- | --- |
| [Controles negativos: reanálisis](_static/e2_closure_followup_20261116/audit/reanalisis_controles_negativos_v1.zip) | `python run.py`: seis controles, RMSE, correlaciones, IC, TOST y Holm. Requiere NumPy/SciPy; no entrena. |
| [Validación HPO y variabilidad](_static/e2_closure_followup_20261116/inference/reanalisis_inferencial_v1.zip) | `python run.py`: comparación OOF249–prueba608 y descomposición finita con IC. Requiere NumPy; no entrena. |
| [Dependencia de volumen y OHLC](_static/e2_closure_followup_20261116/shortcut/diagnostico_entradas_reanalisis_v1.zip) | Repite 26 comparaciones desde los pronósticos de las tres condiciones. Requiere NumPy; las instrucciones identifican el ejecutor. |
| [HPO descriptivo y recuperación bibliográfica](_static/e2_closure_followup_20261116/hpo/reanalisis_hpo_descriptivo_v1.zip) | Reconstruye asociaciones y costes registrados con Python estándar; incluye formulario del historial Scopus/WoS. No ejecuta ajustes HPO. |
| [Flujo CPU con fuentes, datos y pesos](_static/e2_closure_followup_20261116/reproduction/reproduccion_cpu_portable_v1.zip) | Incluye verificación, reanálisis, inferencia de 81 predictores y entrenamiento de recetas fijas como modos separados. Requiere instalar `requirements_cpu.txt`; leer `README.md` antes del modo elegido. |

[Resultados y límites del seguimiento (§66)](sections/66_seguimiento_pendientes.md). La revisión de controles negativos conserva un resultado adverso: **no está establecida la cercanía al azar**. Los nuevos análisis no completan una campaña HPO ampliada ni restituyen el historial bibliográfico original faltante.

## Estado del benchmark y suplementos

**Ampliaciones del 15 y 16/11/2026:** [§64](sections/64_correcciones_metodologicas.md) conserva veinte tamaños de efecto con IC, RQ2 ejecutada, trece contrastes de brecha y 12.000 mediciones de latencia. [§65](sections/65_cierre_cientifico_real.md) añade datos reales train655/cal53/test608, controles de los puertos, causalidad clásica, permutación de etiquetas, diagnóstico adversarial y comparación formal de calibración–prueba. La nueva IUT CPU genera **47.424 pronósticos** y no demuestra ninguna mejora conjunta tras Holm10. La tolerancia CUDA del suplemento anterior conserva su resultado fallido y no se cambia.

**Evaluación final y experimentos complementarios completados. Cierre A100 verificado: 13/11/2026, 22:42:55 de Bogotá; suplementos estadísticos y documentales del 13/11, posteriores a las 23:17.** La tabla A2 reúne **16 sistemas sobre las mismas 608 fechas**. El plan complementario cerró **99/99 ajustes: 81 reproducciones y 18 ablaciones**, con todas las comprobaciones de reproducción aprobadas y sin reservas abiertas. Se conservan los resultados originales y los respaldos en **GitHub**.

**[Resultados finales (§6)](sections/06_benchmark_final.md)** · **[Tabla A2 medida, CSV](_static/e2_benchmark_final_20261113/benchmark_A2.csv)** · **[Reproducción y ablaciones (§55)](sections/55_reproduccion_ablaciones.md)** · **[Estado y presupuesto final](_static/e2_reproduction_ablations_20261113/status.json)**

**Resultado de reproducción:** los recibos de los 81 predictores informan **diferencia máxima absoluta cero** frente a los originales en 53 pronósticos de calibración y 608 de prueba. Los 78 estados neuronales son iguales tensor a tensor y los tres clásicos conservan sus parámetros. Es una reproducción numérica con checkout y entorno Python separados sobre la misma cronología conocida.

**Resultado de ablaciones:** las tres variantes del autoencoder se entrenaron con seis semillas cada una. Retirar el dropout supervisado aumentó la RMSE un **13,90 %**, con **p Holm=0,00059994**. Las otras dos diferencias no se rechazan tras Holm; eso no prueba equivalencia. Los análisis son posteriores y mantienen las recetas y la A2 originales.

**Otros suplementos completados:** [ajuste y calibración (§49)](sections/49_diagnosticos_f2a.md) · [DM/MCS y rangos (§50)](sections/50_comparaciones_f2a.md) · [costes y Pareto (§51)](sections/51_eficiencia_costes.md) · [síntesis bibliográfica (§52)](sections/52_sintesis_literatura.md) · [estado de requisitos (§53)](sections/53_cierre_requisitos.md) · [robustez y explicaciones del Top 3 (§54)](sections/54_robustez_interpretabilidad.md). Los diagnósticos incluyen **79 predictores, 468 filas de robustez y 78 perfiles de inferencia A100**.

**Suplementos finales:** [20.000 remuestreos y Giacomini–Rossi (§56)](sections/56_inferencia_estabilidad_final.md) · [6.080 etiquetas de errores y 100 fechas por modelo (§57)](sections/57_errores_techo.md) · [FLOPs core de las trece redes (§58)](sections/58_flops_predictores.md) · [conciliación de 9.559 grupos bibliográficos (§59)](sections/59_auditoria_literatura_final.md). Son análisis posteriores con fecha y alcance propios, sin nuevas corridas ni sustitución de A2.

**Cierre documental:** [calibración temporal y cobertura condicional (§60)](sections/60_calibracion_temporal_condicional.md) · [entrega reproducible y Apéndice B actual (§61)](sections/61_reproduccion_consolidada.md) · [elegibilidad focal y P contextual (§62)](sections/62_cierre_literatura_focal.md) · [referencia empírica de ensemble (§63)](sections/63_ensamble_calibracion.md). La cobertura global se acerca al 90 % con ACI, aunque persiste infracobertura en rangos altos. El ensemble proporciona una referencia puntual más baja, con diferencias no concluyentes tras Holm.

**Estado bibliográfico vigente: cierre Scopus con 7.500 registros.** Por decisión expresa del autor se conserva la cohorte Scopus original ya analizada y se cierra su recuperación adicional. [§62](sections/62_cierre_literatura_focal.md) mantiene el **flujo reconciliado de 9.559 grupos multibase**, **42 estudios aplicados únicos**, calidad en doce dimensiones y cero pendientes operativos. Los 174 informes no recuperados, el desfase histórico Scopus de 374 y los rangos/totales WoS no confirmados se documentan como límites; no se solicita otra exportación Scopus. [Decisión fechada](_static/e2_literature_scope_7500_20261116/scope_decision.json) · [ZIP independiente del cierre](_static/e2_literature_scope_7500_20261116/cierre_scopus_7500.zip).

## Organización del informe

El índice principal sigue los **catorce capítulos de la guía de E2, en su mismo orden**. Los documentos complementarios aparecen bajo el tema correspondiente; los avances y diseños anteriores se reúnen en el [archivo de antecedentes](sections/archivo_antecedentes.md).

Las menciones **§15–§66** identifican documentos complementarios y de trabajo, no capítulos adicionales de la guía. La [tabla de concordancia](sections/concordancia_guia.md) permite localizar cada uno en la organización actual. Los enlaces, resultados y descargas anteriores se conservan.

## Resumen

Se estudia el pronóstico diario de volatilidad de Bitcoin usando siete días de OHLCV, horizonte de un día y objetivo σ₇ poblacional de retornos de Close, sin anualización. Se comparan diez candidatos de cuatro familias, tres referencias neuronales de E1 y tres comparadores estadísticos. La selección reducida utiliza seis configuraciones y tres orígenes internos por red; cada receta neuronal final tiene seis semillas. Los 16 sistemas comparten 608 fechas de evaluación conocidas desde E1.

El LSTM-autoencoder obtiene la menor RMSE media observada, **0,00380343**, seguido de ARIMA, **0,00392162**, y GRU32, **0,00400116**. Persistencia presenta el menor MAE y sMAPE. Ninguno de los veinte contrastes primarios demuestra la mejora práctica registrada tras Holm; tampoco se establece equivalencia. La repetición de los 81 predictores reproduce sus resultados con discrepancia máxima cero. Las 18 ablaciones muestran degradación significativa al retirar el dropout supervisado; las otras dos comparaciones no son concluyentes.

Se publican residuos, calibración, comparaciones temporales, robustez, explicaciones del Top 3, perfiles A100, costes y Pareto. La evaluación es retrospectiva y el piloto no alcanza la potencia objetivo. El cierre conserva **84 horas totales autorizadas**, todos los cargos y los resultados negativos. El flujo bibliográfico recibido está reconciliado con límites de cobertura y acceso. δg tiene registro previo y [trece contrastes posteriores en CPU](sections/64_correcciones_metodologicas.md), con dos tolerancias condicionales; no certifican paridad CUDA ni ausencia general de sobreajuste. El proxy empírico de ensemble tiene incertidumbre amplia y no identifica riesgo Bayes. El conteo FLOPs cubre matrices y convoluciones, con exclusiones declaradas.

## Qué quedó completado

| Etapa | Resultado verificado |
|---|---|
| Selección de trece redes | **13 recetas cerradas**; 231 éxitos HPO de 234 oportunidades, dos fallos y un desenlace desconocido conservados |
| Ajustes finales de diez candidatos y tres referencias | **78/78**, seis semillas por red |
| Pilotos de variabilidad y potencia | **10/10**, cinco semillas de iTransformer y cinco de GRU32; [informe medido](_static/e2_benchmark_final_20261113/pilot_power_MDE.json) |
| Comparadores y congelación | **3/3 comparadores**, 81 predictores congelados antes de admitir datos finales |
| Evaluación y análisis registrados | **16/16 sistemas, 608/608 fechas**, tabla A2 y 20 contrastes primarios |
| Auditoría F2a posterior | **81 trayectorias**, 826 contrastes exploratorios, curvas de los 78 ajustes finales y 29 de 39 ajustes HPO seleccionados; [resultados y límites (§49)](sections/49_diagnosticos_f2a.md) |
| Comparaciones F2a posteriores | DM–HLN/HAC14, MCS y rangos condicionados: [§50](sections/50_comparaciones_f2a.md) |
| Eficiencia observada | 78 ajustes finales: 5.765,09 segundos de fases instrumentadas, parámetros y Pareto: [§51](sections/51_eficiencia_costes.md) |
| Reproducción numérica de los predictores | **81/81**, todas las comprobaciones aprobadas; máxima diferencia de pronósticos **0**: [§55](sections/55_reproduccion_ablaciones.md) |
| Ablaciones entrenadas del autoencoder | **18/18**, tres componentes × seis semillas; bootstrap de 10.000 réplicas y Holm: [comparaciones](_static/e2_reproduction_ablations_20261113/ablation_comparisons.csv) |
| Refinamiento numérico y estabilidad | **20.000 remuestreos** de los mismos veinte contrastes; Giacomini–Rossi y constancia separados: [§56](sections/56_inferencia_estabilidad_final.md) |
| Errores y dificultad | **6.080 etiquetas**, 100 fechas por cada modelo principal, IC temporales y entropía ordinal; γ ideal con límites: [§57](sections/57_errores_techo.md) |
| Operaciones de inferencia | FLOPs core D7×5 de **13 recetas** y Pareto, con exclusiones: [§58](sections/58_flops_predictores.md) |
| Literatura recibida | **9.559 grupos cribados, 42 estudios aplicados**, doce dimensiones y disposiciones de acceso: [§62](sections/62_cierre_literatura_focal.md) |
| Entradas y controles reales | Train655/cal53/test608; **78 puertos, 36 comprobaciones clásicas y 13 controles con etiquetas permutadas**: [§65](sections/65_cierre_cientifico_real.md) |
| Comparación conjunta RQ3 en CPU | **47.424 pronósticos nuevos**, diez IUT; ninguna mejora conjunta establecida tras Holm10: [§65](sections/65_cierre_cientifico_real.md) |
| Registro y contabilidad recuperados | Cronología de **170 HPO nuevos y 78 ajustes finales**; **233 cargos observados** del catálogo reducido, con hardware y relojes diferentes: [§65](sections/65_cierre_cientifico_real.md) |
| Presupuesto | **84 h acumuladas autorizadas**. El cierre A100 del 13/11 dejó 80,2561 h contabilizadas; los controles y la nueva repetición CPU conservan sus cargos posteriores, sin reiniciar el límite. |
| Repetición CPU con recetas fijas | **78 ajustes neuronales + tres clásicos**, cal53/freeze81/test608 y primaria20.000 completos; [evidencia y costes](sections/65_cierre_cientifico_real.md) |
| Persistencia | Resultados y comprobaciones de integridad en **GitHub** |

El piloto estima una potencia condicional de **12,30 %** para una ganancia de **2δ**, por debajo del objetivo de 80 %. Con R=6, la ganancia mínima estimada para esa potencia es **0,00087867**. Son cálculos sobre cinco semillas emparejadas de dos modelos y 53 fechas de desarrollo; no garantizan potencia temporal ni para todo el Top 10. R=6 se conserva por recursos, sin ampliar repeticiones después de observar la prueba.

Completar el benchmark no acredita cumplimiento integral de la guía. La [matriz de exigencias (§38)](sections/38_cumplimiento_guia.md) distingue los resultados existentes de las condiciones pendientes; [§65](sections/65_cierre_cientifico_real.md) identifica los controles y cálculos nuevos con sus propios recibos.

## Cómo interpretar los resultados

La métrica primaria de cada red es la **media de sus seis RMSE individuales**, no la RMSE de un ensamble. La desviación y el IC entre semillas describen variabilidad de optimización condicionada a esta cronología. Los tres comparadores deterministas aportan una trayectoria cada uno; no se presentan como seis entrenamientos independientes.

La mejora puntual del autoencoder frente a persistencia es **0,00028217 de RMSE (6,91 %)**, con IC bootstrap temporal **[−0,00016895; 0,00072419]**. Aunque la estimación supera el margen heredado δ=0,00025772842664981106, su incertidumbre no demuestra superioridad práctica. Frente a GRU32 la mejora puntual es **0,00019773**, menor que δ. Los **20 p ajustados por Holm son 1,0** en la familia primaria registrada.

El análisis original conserva bootstrap pareado en bloques de 14 días y 4.999 remuestreos. [§56](sections/56_inferencia_estabilidad_final.md) reproduce exactamente sus veinte contrastes y añade **20.000 réplicas con las mismas hipótesis y Holm**, sin cambiar la conclusión: todos los p ajustados son 1,0. Se mantienen la sensibilidad a 7 y 28 días y la fecha posterior del refinamiento. Los diagnósticos por semillas no convierten esas repeticiones en mercados independientes. Los límites de esta batería frente a lo solicitado por la guía se explican en §38. No se cambian recetas ni calibradores después de observar la prueba.

## Guía de lectura

| Contenido | Dónde revisar |
|---|---|
| Tabla A2, incertidumbre, intervalos y discusión final | **[Benchmark final (§6)](sections/06_benchmark_final.md)** |
| Ajuste, sobreajuste, residuos, calibración y estabilidad temporal | **[Diagnósticos F2a posteriores (§49)](sections/49_diagnosticos_f2a.md)** |
| Reproducción en entorno separado y tres ablaciones entrenadas | **[Cierre y diferencias observadas (§55)](sections/55_reproduccion_ablaciones.md)** |
| Informe consolidado y antecedentes del cierre documental | [§47](sections/47_entrega_consolidada.md) |
| Problema, hipótesis, datos y contrato heredado | [§1](sections/01_punto_partida.md) · [§17](sections/17_contrato_sincronizado.md) |
| EDA y auditoría temporal | [§2](sections/02_eda.md) · [§10](sections/10_auditoria_datos.md) |
| Literatura, catálogo y cuerpos publicados | [§3](sections/03_literatura.md) · [§43](sections/43_catalogo_obligatorio.md) · [§44](sections/44_arquitecturas_publicadas.md) |
| Protocolo reducido, ejecución y desviaciones | [§48](sections/48_protocolo_reducido.md) |
| Réplica TimeMixer en ETTh1 y límites de transferencia | [§5](sections/05_paper_guia.md) |
| Requisitos aún pendientes y reproducción | **[Matriz de la guía (§38)](sections/38_cumplimiento_guia.md)** · [§8](sections/08_reproducibilidad.md) |

## Catálogo y alcance temporal

Los diez candidatos son **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**. Las referencias son **MLP16, LSTM32 y GRU32**; los comparadores son **Persistence, SeasonalNaive y ARIMA**. El híbrido adicional y Ridge conservan alcance opcional e histórico.

La selección usa enero, julio y octubre de 2024; el ajuste final común termina el **31/10/2024**, seguido de embargo y calibración separada. La evaluación comprende **01/01/2025–31/08/2026**, el calendario heredado ya conocido en E1. Su interpretación es retrospectiva. La [admisión de datos](_static/e2_benchmark_final_20261113/data_admission.json) documenta las veinte fuentes originales verificadas, el puente causal de diciembre y el soporte común sin comprimir fechas.

El cierre complementario mantiene un límite de **84 horas acumuladas autorizadas**, con el ancla anterior de 80 horas y todos sus cargos conservados. El balance final y su alcance se documentan en [§55](sections/55_reproduccion_ablaciones.md). Las incidencias HPO y el fallo de implementación de ARIMA quedan contabilizados; el ARIMA ganador, de orden (2,0,1), se elige entre **15 de los 18 órdenes** con tres folds válidos. Se conservan dos incidencias de no convergencia y el fallo de implementación, sin repetirlos. Las correcciones de adquisición y respaldo no alteran los predictores congelados. La [congelación](_static/e2_benchmark_final_20261113/final_freeze.json) y el [análisis completo](_static/e2_benchmark_final_20261113/benchmark_analysis.json) permiten revisar ese alcance.

## Pendientes y límites que se conservan

1. **Literatura cerrada en el alcance recibido:** el componente Scopus se congela en 7.500 registros originales y no requiere nuevas exportaciones. El [cierre de §62](sections/62_cierre_literatura_focal.md) conserva los 9.559 grupos multibase y sus disposiciones; las diferencias históricas de cobertura, los informes inaccesibles y los límites de comparabilidad permanecen declarados.
2. Mantener el alcance de las comparaciones: [§64](sections/64_correcciones_metodologicas.md) contrasta δg en train/cal de los 78 predictores CPU; dos redes establecen tolerancia condicional. [§65](sections/65_cierre_cientifico_real.md) ejecuta la comparación formal de calibración–prueba original y la IUT CPU completa. Calibración no es validación HPO independiente; esos cálculos no establecen adecuación conjunta ni identifican la causa de sobreajuste. El calendario conocido conserva alcance retrospectivo y el piloto su limitación de potencia/MDE declarada. La nueva repetición CPU de las recetas está completa; no repite la búsqueda ni toda la campaña histórica.
3. [§63](sections/63_ensamble_calibracion.md) añade una referencia empírica ajustada sólo en calibración y γ=0,28757; su incertidumbre no confirma una superioridad ni un techo óptimo. [§57](sections/57_errores_techo.md) conserva entropía ordinal y las referencias anteriores, sin convertirlas en ruido Bayes. La taxonomía de residuos es automática y posterior, con 100 fechas auditables por modelo.
4. Conservar el cambio de distribución detectado y su tratamiento explícito en [§65](sections/65_cierre_cientifico_real.md). Los costes recuperados describen cargos observados del catálogo reducido, con hardware y relojes diferentes; faltan una duración desconocida y 51 desgloses de fase. Los FLOPs de [§58](sections/58_flops_predictores.md) tienen exclusiones declaradas; energía, dinero y servicio completo no se midieron. El [Apéndice B actual](sections/61_reproduccion_consolidada.md) mantiene estados verificables; la defensa oral conserva su evaluación propia.

La [matriz de §38](sections/38_cumplimiento_guia.md) distingue cada evidencia de sus condiciones pendientes. El cierre de 99 ajustes corresponde al plan complementario; no es un porcentaje de cumplimiento integral de E2.

## Cortes históricos y descargas conservadas

Los cortes anteriores mantienen sus fechas y archivos. Sus conteos parciales y paquetes no sustituyen a la tabla final del 13/11. Las campañas históricas conservaron 1.705 resultados optimizados y 1.802 registros previos: son antecedentes de selección, no nuevas evaluaciones finales. Los protocolos anteriores se consultan en [§40](sections/40_calendario_original.md), [§41](sections/41_avance_optimizacion_11_origenes.md), [§45](sections/45_enmienda_protocolo.md) y [§46](sections/46_artefactos_y_pilotos.md).

- [Instantánea y procedencia](_static/e2_progress_final_20261113_v2/progress_snapshot.json)
- [Avance de las trece redes, CSV](_static/e2_progress_final_20261113_v2/neural_progress.csv)
- [Paquete histórico del 12/11](_static/delivery_20261112/Entregable2_evidencia_disponible.zip)
- [Fuentes y hashes](_static/e2_progress_final_20261113_v2/source_evidence.json)
- [Manifiesto](_static/e2_progress_final_20261113_v2/manifest.json)
- [corte anterior del 13/11, 07:24](_static/e2_progress_final_20261113/progress_snapshot.json)
- [tabla de trece redes](_static/e2_progress_final_20261113/neural_progress.csv)
- [fuentes](_static/e2_progress_final_20261113/source_evidence.json)
- [manifiesto](_static/e2_progress_final_20261113/manifest.json)
- [corte anterior del 12/11, 22:22](_static/e2_progress_20261113/progress_snapshot.json)
- [Top 10, CSV](_static/e2_progress_20261113/top10_training_validation.csv)
- [trece redes, CSV](_static/e2_progress_20261113/all13_hpo_progress.csv)
- [fuentes](_static/e2_progress_20261113/source_evidence.json)
- [manifiesto](_static/e2_progress_20261113/manifest.json)
- [instantánea](_static/delivery_20261112/selection_snapshot.json)
- [CSV de ensayos](_static/delivery_20261112/trials.csv)
