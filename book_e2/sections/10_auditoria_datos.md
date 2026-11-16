(contrato-de-datos-y-auditoria-formal-de-fuga)=
(e2-auditoria-datos)=
# 10. Contrato de Datos y Auditoría Formal de Fuga de Información

**Lectura vigente:** la cohorte real de **655 objetivos train, 53 de calibración y 608 de evaluación** ya está reconstruida desde las fuentes verificadas. Los controles actuales incluyen datos, 78 predictores neuronales, tres clásicos, 35 regresiones univariadas, trece ajustes con etiquetas permutadas y un clasificador adversarial. Sus resultados y límites se reúnen en el [cierre científico actual](65_cierre_cientifico_real.md). **Todos esos procedimientos están ejecutados; la aceptación de rendimiento cercano al azar con etiquetas permutadas no está establecida.** La nueva réplica CPU con objetivos verdaderos está completa en su alcance de recetas fijas.

Los **veinte ZIP oficiales locales** coinciden con el manifiesto, con **875.520 filas de minuto y 608 días completos**. El [recibo de agregación](../_static/e2_scientific_closure_20261116/data_audit/evaluation_aggregation_receipt.json) y el [resultado de datos reales](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json) verifican entradas, objetivos y escalado; el vector de objetivos608 coincide exactamente con el guardado. Este resultado supera la limitación de contenido de test del corte del 15/11, cuyo [recibo anterior](../_static/e2_corrections_20261115/data_audit/audit_result.json) permanece intacto. El auditor de datos informa **12 PASS, 3 PARTIAL y 1 NA**: los tres PARTIAL delimitan ports, etiquetas permutadas y adversarial fuera de ese programa; sus nuevos recibos separados se presentan abajo, sin reescribir aquel resultado.

La [admisión ejecutada](../_static/e2_benchmark_final_20261113/data_admission.json), la [congelación](../_static/e2_benchmark_final_20261113/final_freeze.json), la [Data Card vigente](../_static/e2_point1_review_v1/dataset_card_current.json) y la [auditoría de partición](../_static/e2_eda_current_v1/partition_audit.json) son las fuentes actuales. La [lista verificable](../_static/e2_data_training_uncertainty_current_v1/guide_checklist.csv), la [trazabilidad](../_static/e2_data_training_uncertainty_current_v1/decision_traceability.csv), el [índice de evidencias](../_static/e2_data_training_uncertainty_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_data_training_uncertainty_current_v1/manifest.json) distinguen los alcances de los puntos 10–12. El {download}`índice de esta consolidación <../_static/e2_data_training_uncertainty_current_v1/README.md>` y la {download}`versión exacta anterior de este capítulo <../_static/e2_data_training_uncertainty_current_v1/10_auditoria_datos_before_update.md>` conservan la procedencia.

```{important}
**Regla de oro de la guía, p.46:** si una verificación de la tabla 10.1 falla, **el resultado del benchmark no es válido hasta corregir la causa y repetir todo el experimento afectado**. Una nota al pie no subsana ese fallo. Un control no ejecutado es una comprobación pendiente, no evidencia de aprobación ni prueba automática de fuga.
```

La herencia del test y del margen de E1 se acepta conforme a la continuidad E1–E2–E3. Su conocimiento previo se declara y limita la confirmación prospectiva; **no se considera fuga por la mera herencia de la partición**. A la vez, ni los hashes ni la reproducción numérica acreditan por sí solos todas las verificaciones exigidas.

(e2-auditoria-verificaciones-vigentes)=
## 10.1 Verificaciones obligatorias: evidencia y acción concreta

| Verificación de la guía | Evidencia actual y estado | Qué falta o cómo debe interpretarse |
|---|---|---|
| **Intersección de identificadores** | **Verificada sobre las cohortes reales 655/53/608** y sus ventanas; los tres folds conservan IDs train–validación disjuntos. [Resultado real](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json) · [soportes](../_static/e2_scientific_closure_20261116/data_audit/support_intersections.json). | El ID es la fecha UTC del objetivo. Una serie BTCUSDT puede aparecer en todas las fases; IDs disjuntos no hacen independientes las fechas ni separan toda la historia causal compartida. |
| **Duplicados exactos** | **Cotejo real ejecutado:** seis comparaciones de contenido de ventanas D7, con igualdad de arrays para confirmar coincidencias de hash y cero alertas; se excluye el mismo ID de las consultas. [Resultado D10](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json). | Cubre las ventanas y representación auditadas. Ausencia de duplicados exactos no demuestra independencia ni ausencia de decisiones posteriores al test. |
| **Casi duplicados** | **Diagnóstico real ejecutado:** seis ámbitos, 2.585 consultas, distancia RMS con umbral 0,01 fijado y escalador de train655; cero alertas. [Resultado D11](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json). | El umbral es descriptivo y no una prueba universal. No se eliminan ventanas legítimas próximas por continuidad temporal; sin alertas no se inventa una ablación con/sin casos. |
| **Fuga temporal** | **Verificada en los soportes auditados:** embargo 7, 1.316 ventanas causales reales y cero atributos de la barra objetivo o del futuro; firmas de los 78 ajustes y escaladores físicos cotejados. Los tres clásicos añaden 36 ramas de historia futura perturbada/suprimida, sin cambios en los prefijos ni coeficientes. [Datos](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json) · [clásicos](../_static/e2_scientific_closure_20261116/classical_causality/readout.json). | La historia pasada compartida es legítima. Los probes finitos no certifican todos los historiales ni cada lectura histórica anterior al registro; el test E1 ya era conocido. |
| **Fuga espacial** | **No aplica:** una serie de mercado sin coordenadas o unidades espaciales. | No se sustituye por un test espacial ficticio ni se elimina por ello la auditoría temporal. |
| **Fuga por transformaciones** | **Preparación común real verificada:** cambiar datos posteriores al 31/10 no altera train ni su escalador. Además, la [auditoría de ports](../_static/e2_scientific_closure_20261116/port_audit/readout.json) ejecutó **78 predictores de 13 modelos, seis ventanas train/cal y 390 condiciones**, sin alterar pesos/buffers/entradas consultadas/escaladores. | Pasan los probes y tolerancias fijadas, no una prueba exhaustiva sobre cualquier entrada. La auditoría de ports no incluyó intervenciones sobre todas las ventanas test608 ni acredita la paridad CPU/GPU original. Las cinco guardas con fixtures siguen como evidencia complementaria. |
| **Atributos posteriores al evento** | **Disponibilidad causal comprobada** para las 1.316 ventanas y **35 regresiones OLS univariadas** ajustadas sólo en train655, con 105 resultados train/cal/test; ninguna activa el umbral previo R²≥0,95. [Datos D06](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json) · [pantalla univariada](../_static/e2_scientific_closure_20261116/univariate_attributes/result_v1/summary.json). | Las 35 coordenadas requieren barras anteriores a t; los precios relativos también usan Close(t−1). Los clásicos usan objetivos pasados declarados. La pantalla conserva R² negativos, no selecciona atributos y no excluye fugas multivariadas o no lineales. |
| **Validación adversarial** | **Ejecutada:** clasificador de dominio train655/test608, 35 coordenadas D7×5, validación OOF temporal purgada y 999 permutaciones con reajuste. **AUC=0,952539, p=0,001**: los periodos son discriminables. [Resultado](../_static/e2_scientific_closure_20261116/controls/adversarial_result.json). | Es diagnóstico de distribución; sus folds pueden usar filas futuras para ese fin y no son pronósticos causales. La diferencia significativa exige explicar límites de transporte y revisar regímenes/cobertura; por sí sola no demuestra fuga ni justifica retuning con test. |
| **Fuga por preentrenamiento** | **Sin pesos fundacionales externos:** las redes se inicializan de nuevo; SAE usa X de train en AE1/AE2 antes de supervisión. Los registros y controles del port conservan estados y escalas. | No aplica una búsqueda de contaminación de un corpus externo no usado. El uso de código público o de caché HPO no sustituye la comprobación train-only de las fases auxiliares. |
| **Sanidad con etiquetas aleatorias** | **Ejecutada para las trece redes:** una semilla fija por modelo, recetas/épocas congeladas, etiquetas supervisadas permutadas por bloques de 7 y evaluación en 608 fechas. La IUT de ganancia sobre la media de train y asociación positiva usa bootstrap de bloques14 y Holm13; **cero flags ajustados**. [Especificación](../_static/e2_scientific_closure_20261116/controls/specification.json) · [resumen](../_static/e2_scientific_closure_20261116/controls/negative_summary.json). | **Rendimiento cercano al azar no establecido:** iTransformer y TSMixer conservan ganancias positivas y ρ=0,522/0,493; sus p conjuntos nominales 0,00740/0,04460 pasan a 0,09619/0,53515 con Holm13. No rechazar la regla familiar no demuestra equivalencia al azar ni fuga. Se conservan ambos resultados sin repetir permutaciones para obtener una decisión deseada. Los clásicos tienen controles causales de prefijos. |

**Ejecutar estos controles no equivale a certificar validez universal.** La AUC adversarial alta documenta desplazamiento entre periodos. Se trata reportando límites temporales, resultados por regímenes y cobertura condicional, sin cambiar la receta o escoger otra arquitectura mediante el test. No identifica por sí sola la causa económica del cambio ni prueba entrada de información futura. Ljung–Box/BDS desfavorables, por su parte, cuestionan adecuación o dependencia residual y no son diagnósticos automáticos de fuga.

Las [36 ramas clásicas](../_static/e2_scientific_closure_20261116/classical_causality/readout.json) cubren tres modelos y seis prefijos fijados: los pronósticos de prefijo permanecen idénticos al perturbar o suprimir información futura y no se reestiman coeficientes. ARIMA utiliza el checkpoint recuperado de la reproducción, cuyo hash de pickle difiere del original; su [linaje y bindings](../_static/e2_scientific_closure_20261116/classical_causality/predictor_bindings.json) se conservan explícitos. Igualdad de parámetros/pronósticos no se transforma en igualdad de bytes de ambos pickles.

### 10.1.1 Qué separaciones están comprobadas

| Ámbito actual | Entrenamiento admisible | Validación/calibración o evaluación | Evidencia |
|---|---|---|---|
| HPO, origen 01/01/2024 | **253** ejemplos, hasta 25/09/2023 | **83**, 03/10–24/12/2023 | IDs train–validación disjuntos y embargo 7 |
| HPO, origen 01/07/2024 | **435**, hasta 25/03/2024 | **83**, 02/04–23/06/2024 | IDs train–validación disjuntos y embargo 7 |
| HPO, origen 01/10/2024 | **527**, hasta 25/06/2024 | **83**, 03/07–23/09/2024 | IDs train–validación disjuntos y embargo 7 |
| Ajuste final | **655**, 08/01/2023–31/10/2024 | **53 calibración**, 08/11–30/12/2024 | Embargo 01–07/11; soportes de cierres de etiquetas separados |
| Prueba heredada común | Pesos/escaladores/calibradores ya fijados | **608**, 01/01/2025–31/08/2026 | Mismo calendario guardado para 16 sistemas/81 trayectorias |

Las cuentas y fechas se cotejan con [registros y máscaras actuales](../_static/e2_eda_current_v1/partition_audit.json). Las 83 fechas de cada fold no son las 305 fechas del EDA forward de diez folds ni las 60 del antiguo P0. Los 78 recibos finales neuronales reportan 655 ejemplos; en HPO parte de las reutilizaciones carece del contador train directo y se coteja mediante firmas y metadatos de disponibilidad. No se presenta esa reconstrucción como una reapertura de todos los tensores originales.

La máscara de entrenamiento neuronal exige entrada D7 y objetivo válidos. La evaluación usa además la disponibilidad de los rezagos causales necesarios para los comparadores registrados; las 608 fechas resultan elegibles para todos. La disponibilidad adicional de una referencia endógena se declara, sin atribuir a todas las redes historia que no recibieron.

### 10.1.2 IDs, ventanas y disponibilidad no son la misma comprobación

Para emitir al inicio de t, $X_t$ usa OHLCV de **[t−7,t−1]**; $Y_t$ usa **ocho cierres [t−7,t]** para siete retornos. El embargo separa los soportes de etiquetas en cada train–validación HPO y en train final–calibración. En cambio, calibración y test tienen IDs distintos pero sus soportes comparten **seis cierres y cinco retornos**, del cierre de calibración 30/12/2024 al primer objetivo 01/01/2025. El 31/12 es historia causal, no fecha de ajuste o recalibración.

Esa reutilización de historia ya observable es legítima al avanzar el origen con parámetros congelados. También los folds expansivos pueden incorporar observaciones que eran validación de un origen anterior y ya son pasadas en el siguiente. **No equivale a reutilizar etiquetas futuras para seleccionar recetas**, ni convierte las fechas o semillas en observaciones iid. La [auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json) distingue ambas situaciones; no afirma intersección cero de todo el dato bruto entre todas las etapas.

### 10.1.3 Guardas comunes verificadas con fixtures

Se reejecutaron **cinco tests unitarios existentes del código científico común: 5/5 aprobados, cero fallos y cero errores**. El [informe detallado](../_static/e2_data_training_uncertainty_current_v1/targeted_data_guard_checks.json) y el {download}`registro de ejecución <../_static/e2_data_training_uncertainty_current_v1/targeted_data_guard_checks.log>` identifican los seis archivos de fuente/pruebas cotejados, idénticos al commit científico `1e8745497b484518f766b6da5c19be8362d9add7`.

| Guarda comprobada | Resultado en los fixtures |
|---|---|
| Transformación train-only | Alterar observaciones sintéticas posteriores al 31/10 no cambia la firma de entrenamiento final ni su escalador. |
| Ajuste final con épocas fijas | El worker común, con ajuste/inferencia simulados, no recibe calibración como validación para entrenar y conserva el vínculo predictor/calibrador. |
| Puerta de acceso | El descargador de mercado no se invoca si falla la comprobación de congelación física completa. |
| Puente causal y objetivo | La primera ventana de prueba sintética utiliza 31/12 como historia observable y σ₇ se construye como desviación poblacional de siete retornos logarítmicos. |
| Calendario y geometría | Los días ausentes y las geometrías OHLCV inválidas se rechazan; no se comprime el calendario para ocultarlos. |

La ejecución de esas cinco guardas usó **CPU, Python 3.12.14 y Torch 2.8.0+cpu**, con fixtures y mocks; no fue entrenamiento ni inferencia de mercado. Fue posterior al benchmark y conserva ese alcance. Los controles reales de datos, ports, etiquetas permutadas y adversarial ya tienen los recibos separados de §10.1: no se atribuyen retroactivamente a estas pruebas unitarias ni se suman como evidencia independiente de una misma hipótesis.

### 10.1.4 Límites de los controles históricos y acciones de cierre

El [preflight P0/OOF](../_static/remediation_preflight/data_support_audit.json) conserva su cohorte histórica de desarrollo. Los nuevos controles actuales tienen planes y fuentes propios: [datos655/53/608](../_static/e2_scientific_closure_20261116/data_audit/audit_result.json), [78 predictores](../_static/e2_scientific_closure_20261116/port_audit/readout.json), [trece negativos](../_static/e2_scientific_closure_20261116/controls/negative_summary.json), [dominios](../_static/e2_scientific_closure_20261116/controls/adversarial_result.json) y [causalidad clásica](../_static/e2_scientific_closure_20261116/classical_causality/readout.json). La nueva [IUT CPU completa](../_static/e2_scientific_closure_20261116/cpu_test_iut/statistics/summary.json) usa proyecciones CPU y costes CPU, con cero mejoras conjuntas; evalúa los pesos congelados recuperados y conserva un alcance distinto de los nuevos ajustes CPU ya completados. Es un estudio de rendimiento/coste, no una certificación adicional de ausencia de fuga.

Permanecen límites concretos: cobertura finita de los probes por port, supuestos de bloques, una única serie y calendario conocido, y explicación/transportabilidad del cambio de distribución. La auditoría de ports delimita train/cal, aunque la cohorte test608 ya está disponible y otro experimento ejecutó sus proyecciones CPU. La **nueva réplica CPU de recetas fijas con objetivos verdaderos está completa**, con su evaluación y análisis; no se declara equivalente numéricamente a A100 ni repetición de todo el HPO. [Estado y alcance](65_cierre_cientifico_real.md).

Si una comprobación aplicable detecta un fallo, se identifica su causa y todo el alcance afectado, se corrige el pipeline y **se repite el experimento afectado antes de sostener su validez**. No basta anotar la incidencia, borrar fechas desfavorables o elegir otro diagnóstico que no rechace. Un control no estimable o aún no ejecutado permanece identificado como tal; no se confunde con un fallo ya demostrado.

(e2-contrato-datos-vigente)=
## 10.2 Contrato de datos: esquema, disponibilidad, linaje y filtros

### 10.2.1 Esquema versionado y diccionario de atributos

La [Data Card actual, schema_version 2](../_static/e2_point1_review_v1/dataset_card_current.json), el [manifiesto de adquisición heredado](../_static/dataset_manifest_e1.json) y el [contrato vigente](17_contrato_sincronizado.md) fijan fuente, tipos, unidades y reglas. La ficha histórica `dataset_card.json` conserva sus bytes; sus antiguos estados no describen el cierre actual. La ruta es **F2a**, sin crear una ruta tabular por representar entradas como una matriz ni añadir Y rezagada como sexto canal.

| Cantidad | Tipo, unidad y validación | Fuente y disponibilidad |
|---|---|---|
| `open_time`, `close_time` | Marcas enteras; milisegundos antes de 2025 y microsegundos desde 2025 según el proveedor; normalización a UTC y controles de secuencia | CSV oficial de velas de 1 minuto; no confundir cambio de unidad con cambio de frecuencia |
| OHLC | Valores numéricos positivos, finitos y geometría coherente; **USDT por BTC** antes de transformar | Binance Spot; la barra completa del día t−k es observable al inicio de t−k+1 |
| `volume` | Numérico finito y no negativo; **BTC** | Volumen del activo base, agregado por día completo |
| Otros campos crudos | CSV de **12 columnas**: volumen cotizado, número de operaciones, volúmenes taker y campo ignorado, además de los anteriores | Documentados en la ficha; no se convierten en canales neuronales adicionales |
| Fecha diaria / ID de ejemplo | Medianoche UTC, única y con posición conservada en el calendario | Fecha del objetivo t; h=1, sin comprimir huecos |
| Entrada D7 | Tensor **7×5**, barras completas t−7…t−1 y máscara de calidad | Cinco canales OHLCV disponibles antes de emitir; excluye la barra completa de t |
| σ₇(t) | Desviación poblacional, `ddof=0`, siete retornos logarítmicos; **adimensional y sin anualizar**; finita y no negativa cuando el soporte es válido | Requiere cierres t−7…t; sólo se conoce al terminar t. No es varianza realizada intradiaria ni una etiqueta de clase |

El [cargador de desarrollo](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_development_data.py) y la [admisión final](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_reduced_final_data.py) contienen comprobaciones de hashes, calendario UTC, valores finitos, geometría OHLCV y coherencia de σ₇. La nueva agregación y auditoría ejecutaron estas verificaciones sobre los veinte ZIP locales y las cohortes reales; §10.1 distingue ese resultado de las guardas sintéticas complementarias. El [diccionario de disponibilidad](../_static/eda_decisions/heredado/eda_temporal_specific/availability_contract.csv) diferencia OHLCV futuro, σ₇(t) y el rezago observable σ₇(t−1) usado en referencias o regímenes.

### 10.2.2 Transformaciones y trazabilidad desde el dato original

El linaje conserva este orden:

1. **Fuente:** ZIP oficiales y checksums, identificados por ruta, tamaño y SHA-256 en el manifiesto. Se conservan los brutos y sus fechas.
2. **Agregación diaria UTC:** primera apertura, máximo de máximos, mínimo de mínimos, último cierre y suma del volumen. Una barra incompleta no se rellena usando el futuro.
3. **Objetivo y máscaras:** retornos logarítmicos diarios y σ₇ de siete retornos con ocho cierres consecutivos; la invalidez se propaga a etiquetas y ventanas dependientes.
4. **Representación causal:** precios como `log(P)−log(C[t−1])`, anclados al último cierre observable de la ventana, y `log1p(volume)`.
5. **Escalado aprendido:** media/DE poblacional de cada canal sobre las posiciones N×7 de train; DE cero se sustituye por 1. Y se divide por su media positiva de train, sin centrar. Cada fold ajusta sus parámetros sólo con train.
6. **Congelación y scoring:** los escaladores finales, pesos y calibradores se fijan antes de puntuar; las métricas invierten la escala de Y y conservan las mismas fechas. Las actualizaciones causales de estado ARIMA no reestiman sus coeficientes.

La [admisión](../_static/e2_benchmark_final_20261113/data_admission.json) identifica los derivados de evaluación y el puente 31/12; [§48](48_protocolo_reducido.md) registra la política de cada etapa. Un SHA identifica bytes concretos, no permisos de acceso causal o independencia. El hash de un CSV serializado de nuevo tampoco se equipara silenciosamente al de otro formato o dominio tensorial.

### 10.2.3 Poblaciones y exclusiones con denominadores explícitos

| Ámbito y fuente del recuento | Población, filtros y resultado |
|---|---|
| **Calidad de desarrollo 2023–2024** | De **731 días de calendario**, el 24/03/2023 está incompleto; quedan **730 barras completas**. La fuente identifica **80 minutos ausentes y una fila presente con cierre temporal irregular**, no 81 filas ausentes. Se conservan 350/366 etiquetas σ₇ válidas de 2023/2024: **716** en total. [Calidad por año](../_static/eda_decisions/heredado/eda_general_quality/quality_by_year.csv). |
| **Entrenamiento final hasta 31/10/2024** | **670 posiciones de calendario y 655 ejemplos admisibles**. Los 15 IDs excluidos son 01–07/01/2023 por historia inicial insuficiente y 24–31/03/2023 por el día incompleto y sus soportes dependientes. No se presentan como 15 días OHLCV incompletos. [Máscaras y fechas actuales](../_static/e2_eda_current_v1/partition_audit.json). |
| **Embargo, calibración y puente** | 01–07/11 se excluye del ajuste/calibración por diseño; **53 fechas de calibración, 08/11–30/12**, y 31/12 sólo como puente. El embargo es una regla temporal, no un filtro de calidad ni un error eliminado por su pérdida. |
| **Admisión de prueba actual** | **20 ZIP, 875.520 minutos registrados/esperados y 608 días completos**, sin minutos ausentes/irregulares ni barras diarias excluidas en ese recibo; **608/608 fechas comunes elegibles**. [Data Card](../_static/e2_point1_review_v1/dataset_card_current.json), [nuevo recibo por archivo](../_static/e2_scientific_closure_20261116/data_audit/evaluation_aggregation_receipt.json) y [cohortes reales](../_static/e2_scientific_closure_20261116/data_audit/cohort_array_manifest.json). |

Estos denominadores no se suman como una cascada única: calendario, minutos, barras, etiquetas, ventanas y cortes de protocolo son objetos diferentes. No se borran eventos extremos legítimos ni se seleccionan fechas por la magnitud del error. La admisión actual no abrió septiembre–octubre de 2026 para su scoring; el inventario heredado completo puede contener archivos de ese periodo con otro alcance.

Los recuentos finales ya proceden de una agregación ejecutada de las 875.520 filas y de la reconstrucción real de X/Y para 655/53/608, con hashes y diferencias numéricas publicados. El CSV diario serializado de nuevo conserva su SHA propio; no se lo confunde con el de otro derivado E1. **La aprobación integral no se deduce de disponer de los archivos ni de reconstruir A2**: se leen por separado los controles ejecutados, sus resultados adversos o no concluyentes y los límites de §10.1 y del [Apéndice B](61_reproduccion_consolidada.md).

````{admonition} Archivo documental: contenido anterior del capítulo 10
:class: dropdown

Este bloque conserva íntegramente el texto anterior, sus encabezados y enlaces. Sus recibos, fechas y estados corresponden a sus cortes originales; el protocolo reducido y las verificaciones actuales se presentan arriba.

**Orientación de entrega del 14/11/2026:** el protocolo ejecutado es [§48](48_protocolo_reducido.md), los resultados comunes están en [§6](06_benchmark_final.md), la reproducción registrada en [§55](55_reproduccion_ablaciones.md) y el Apéndice B actual en [§61](61_reproduccion_consolidada.md). Las cifras científicas y los recibos históricos conservan su fecha y su alcance.
**Alcance vigente:** esta página separa el preflight histórico P0 de los controles de la ejecución final. La [admisión actual](../_static/e2_benchmark_final_20261113/data_admission.json), la [congelación](../_static/e2_benchmark_final_20261113/final_freeze.json) y el [protocolo de §48](48_protocolo_reducido.md) acreditan soporte, identidades y secuencia de la campaña de dieciséis sistemas. No equivalen a la aprobación de todos los criterios de la auditoría integral de fuga.


**Estado:** la auditoría ejecutada de soporte P0/OOF de desarrollo está aprobada. Esto no certifica todas las verificaciones del pipeline final de configuraciones del registro histórico exigidas en la tabla10.1 de la guía (PDF pp44–45). La regla de p46 exige corregir cualquier fallo y repetir el experimento afectado; no permite publicar un benchmark inválido con una nota al pie. La ausencia de un control todavía no ejecutado se informa como pendiente, sin declararlo aprobado.

## 10.1 Evidencia ejecutada, fecha y frontera de acceso

El [recibo `data_support_audit.json`](../_static/remediation_preflight/data_support_audit.json) se generó el **06/11/2026 a las 14:52:31 UTC** con `scripts/e2_remediation_data_audit.py` de la fuente registrada [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0). Lee exclusivamente los dos derivados diarios de desarrollo 2023–2024, comprueba su integridad y produce agregados e índices, sin precios individuales en el recibo.

| Comprobación | Resultado y alcance |
|---|---|
| Calendario | 731 fechas diarias UTC de 2023–2024, únicas y sin compresión |
| Día OHLCV incompleto | 24/03/2023; la máscara conserva su posición |
| Reconstrucción σ₇ | Diferencia absoluta máxima0,0 en la reconstrucción comparada por este auditor; consistencia aritmética, no ausencia de ruido de proxy |
| Orígenes de desarrollo | Once: diez mensuales para OOF y uno para noviembre–diciembre |
| OOF enero–octubre de 2024 | 305 fechas de resultado únicas |
| P0 noviembre–30 de diciembre de 2024 | Sesenta fechas de resultado únicas |
| Identidades de resultados | Cero intersecciones train/evaluación e inner-train/inner-validación dentro de cada separación auditada |
| Duplicados exactos D7 cruzados | Cero colisiones de contenido entre train/evaluación en los once orígenes |
| Corte temporal | Última etiqueta y ajuste de escalado no posteriores a origen−8; embargo de siete días |
| Preflight de datos | Cero ajustes y cero predicciones realizados por **este auditor**, sin acceso a precios de test de 2025/2026 |

El último contador describe el proceso de auditoría, no el estado global posterior del proyecto: **P0 ya se ejecutó y analizó**, con 220 trayectorias/440 ajustes de desarrollo cerrados y sus resultados en §9. No son resultados del benchmark final, ni nuevos mercados independientes.

Los hashes de fuentes son:

```text
daily_market_development.csv
bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171
daily_volatility_development.csv
20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a
```

Un hash identifica bytes y detecta cambios respecto al registro; **no prueba por sí solo ausencia de fuga**. El calendario, la disponibilidad, los soportes y el código de transformación requieren controles adicionales. Tampoco la ausencia de duplicados exactos excluye dependencia, casi duplicados o decisiones hechas con test.

### 10.1.1 Enmienda de soportes para el procedimiento final

La [enmienda preparada](../_static/completion_protocol/analysis_additions.json) reconoce selección de hiperparámetros hasta **31/10/2024**. P1 y A usan origen nominal 08/11, corte de ajuste/escalado/selección 31/10, embargo 01–07/11 y **53 resultados 08/11–30/12**. El antiguo P0 conserva sus sesenta fechas y soportes ya ejecutados; no se reescriben los recibos de `599a29f`.

Para OOF final, las tres recetas históricas E1 declaran su selección hasta el 31/10 incluso en folds anteriores; esa exposición impide calificarlas como selección anidada insesgada. Los cortes mensuales de pesos, escaladores y épocas siguen vigentes para todos; las otras diecisiete configuraciones deben respetar también selección de configuración anterior al origen. La excepción histórica no permite nuevos ajustes con información futura.

El código final añade auditoría de estos soportes y guardas globales: **todos los pesos primarios, A y de controles negativos cerrados antes de cualquier score de calibración**, calibradores sellados antes del test y misma identidad del predictor A calibrado/evaluado. La lectura posterior de historia 2025–agosto de 2026 para referencias endógenas conserva calendario y NaN, con hash/auditoría de archivos; no autoriza refit ni elimina la condición de test conocido. La ejecución integrada de esos controles todavía debe producir sus recibos.

## 10.2 Contrato operativo de las observaciones

Se hereda la [Data Card](../_static/dataset_card.json) y el [contrato científico E1](../_static/e1_reference.json), sin cambiar D7×5, objetivo, h=1, δ, unidad ni partición. La [ruta F2a](07_ruta_temporal.md) añade sus referencias explícitas; no añade Y rezagada a las redes.

| Campo | Regla al cargar y construir ejemplos |
|---|---|
| Identificador temporal | Fecha diaria UTC; su posición no se sustituye por número de fila retenida |
| Covariables | Cinco magnitudes OHLCV; valores finitos, OHLC positivos y coherentes, volumen no negativo en barras completas |
| Disponibilidad | Para resultado t, solo barras completas t−7…t−1; no OHLCV del propio día t |
| Etiqueta | Desviación poblacional de siete retornos logarítmicos consecutivos, ocho cierres, sin anualización |
| Ausencia | No imputar ni unir días separados por un hueco; excluir ejemplos dependientes de información inválida |
| Transformación | OHLC relativos al último cierre observable de la ventana y log1p de volumen; escalas y media de Y aprendidas en entrenamiento admisible |
| Selección y parada | Validación interna anterior al origen, con su embargo; datos de evaluación no eligen época ni parámetros |
| Referencias endógenas | Persistencia alcanza t−8; naive semanal alcanza t−14; ARIMA usa historia declarada, con parámetros y actualizaciones separados |

El 24/03/2023 reúne 80 minutos ausentes y una vela presente con cierre temporal irregular según E1. No se presentan como 81 filas ausentes. Se conserva el día incompleto y se propaga su máscara a retornos, σ₇ y ventanas. Los recuentos detallados de calidad y elegibilidad, incluidas las 350/366 etiquetas válidas de desarrollo, permanecen en §2; 731 días no equivalen a 731 muestras supervisadas ni a 731 unidades independientes.

El driver P0 descodifica para cada ajuste únicamente el prefijo admisible hasta su corte; el scoring de desarrollo es una fase posterior a los cierres programados. La auditoría previa sí inspecciona desarrollo completo para comprobar su esquema y calendarios: no se confunde ese control con una evaluación desconocida ni con permiso para ajustar escalas con validación.

## 10.3 Once separaciones auditadas

| Origen | Último resultado train/escalado | Train elegible | Inner-train | Inner-validación | Evaluación |
|---|---|---:|---:|---:|---:|
| 01/01/2024 | 24/12/2023 | 343 | 253 | 83 | 31 |
| 01/02/2024 | 24/01/2024 | 374 | 284 | 83 | 29 |
| 01/03/2024 | 22/02/2024 | 403 | 313 | 83 | 31 |
| 01/04/2024 | 24/03/2024 | 434 | 344 | 83 | 30 |
| 01/05/2024 | 23/04/2024 | 464 | 374 | 83 | 31 |
| 01/06/2024 | 24/05/2024 | 495 | 405 | 83 | 30 |
| 01/07/2024 | 23/06/2024 | 525 | 435 | 83 | 31 |
| 01/08/2024 | 24/07/2024 | 556 | 466 | 83 | 31 |
| 01/09/2024 | 24/08/2024 | 587 | 497 | 83 | 30 |
| 01/10/2024 | 23/09/2024 | 617 | 527 | 83 | 31 |
| 01/11/2024 | 24/10/2024 | 648 | 558 | 83 | 60 |

Los diez primeros orígenes reúnen 305 fechas OOF; la última fila representa todo el bloque P0 hasta el 30/12, no un undécimo mes OOF. La validación interna nominal de noventa días deja 83 resultados elegibles por su separación temporal; inner-train más inner-validación no suman todo train porque también existe embargo interno.

El activo BTCUSDT aparece legítimamente en todas las fases: se evalúa una sola serie, no generalización a activos excluidos de entrenamiento. Para esta ruta se comprueban fechas de resultados y soportes cronológicos; no se exige inventar identificadores de mercados disjuntos. Una fecha validada puede incorporarse a un entrenamiento posterior cuando ya es pasada y respeta embargo. Ese uso expansivo no vuelve independientes los folds, las semillas o las etiquetas solapadas.

## 10.4 Correspondencia con las verificaciones obligatorias

| Tabla10.1, PDF pp44–45 | Evidencia disponible | Estado y límite |
|---|---|---|
| Intersección de identificadores | Índices de resultados train/evaluación e inner-train/validación en once orígenes | Aprobado para P0/OOF; una serie común no se presenta como particiones iid por mercado |
| Duplicados exactos | Hash de los tensores D7 numéricos entre train/evaluación | Cero colisiones en desarrollo auditado; controles finales/test por completar |
| Casi duplicados | Vecinos en representación relativa, escalado de entrenamiento y umbral0,01 documentados en §2 | Diagnóstico de desarrollo ejecutado; sensibilidad de desempeño con/sin casos no acreditada por este recibo |
| Fuga temporal | Train y escalado hasta origen−8; controles internos y calendario no comprimido | Aprobado en once separaciones D7; cada referencia con otro soporte requiere su comprobación específica |
| Fuga espacial | Un mercado sin coordenadas/unidades espaciales | No aplica a la tarea F2a; no sustituirlo por un test ficticio |
| Transformaciones | Escalado de train comprobado y pruebas sintéticas de adaptadores; EDA con parámetros train-only | Evidencia P0 parcial; auditoría completa de cada pipeline final pendiente |
| Atributos posteriores al evento | Diccionario causal y prefijos de lectura anteriores al origen; análisis de rezagos en §2 | Disponibilidad documentada; revisión final y controles univariados aplicables pendientes de cierre |
| Validación adversarial | Implementación de clasificador train/evaluación con las 35 entradas, cinco folds bloqueados/purgados y gap de siete días; escalado dentro de fold | Experimento final pendiente; no se sustituye por KS/Wasserstein ni se seleccionan modelos con su resultado |
| Modelos preentrenados | P0 desde inicialización; adaptadores históricos revisados con fuentes/presets por hash | El catálogo vigente debe verificar procedencia de pesos, archivos importados y licencias antes de entrenar |
| Sanidad con etiquetas permutadas | Implementación para dieciséis redes y Ridge; permutación separada por bloques en train/selección de épocas; nuevas semillas y pesos cerrados antes de test | Ejecución y análisis pendientes; las referencias endógenas requieren auditoría causal y perturbaciones de historia, no una regresión supervisada ficticia |

Los controles de adaptadores verifican forma, gradientes finitos y que una predicción no cambie con otros orígenes en el batch. La revisión independiente y la prueba de memorización sintética del catálogo histórico se conservan para auditoría en [§14](14_arquitecturas.md). Son controles relevantes para disponibilidad causal, pero **no equivalen** a entrenar con etiquetas permutadas, demostrar ausencia de atajos o aprobar automáticamente los pipelines del catálogo vigente. Si una comprobación futura falla, se conservará el fallo y se repetirá el experimento afectado tras corregir su causa; no se seleccionarán fechas favorables ni un test que produzca el resultado deseado.

### 10.4.1 Controles negativos del diseño histórico

El módulo `scripts/e2_leakage_controls.py` separa diecisiete controles supervisados —dieciséis redes y Ridge— de las referencias endógenas. Permuta etiquetas por bloques de siete observaciones admitidas, declara huecos y separa permutaciones de entrenamiento, selección temprana y validación interna. Usa semillas de permutación 690101/690118/690132, reajuste desde cero y la misma receta primaria congelada. Los **diecisiete predictores de control deben cerrar junto a todos los primarios y A antes de calibración o test**; no se ajusta un control después de observar resultados. Los errores se evalúan sobre la misma reserva conocida de sesenta fechas, dentro de su fase única.

Se compara con el pronóstico constante de media de entrenamiento. La señal sospechosa predefinida requiere conjuntamente ganancia RMSE>δ y correlación Spearman positiva, mediante IUT y **Holm sobre diecisiete hipótesis, α=0,01**, con 2.000 remuestreos. No rechazar no demuestra ausencia de fuga; los controles son diagnósticos con potencia y supuestos limitados, no una certificación automática.

El clasificador adversarial usa las 35 entradas, cinco particiones bloqueadas/purgadas, embargo de siete días, escalado aprendido dentro de cada train y regresión logística L2 balanceada con parámetros fijos. La distribución nula usa **2.000 permutaciones de etiquetas de dominio por bloques con reajuste de todos los folds**. Se ejecuta tras los cierres globales; explica desplazamiento entre periodos sin retroalimentar HPO o adaptar escaladores a test. ARIMA y los dos ingenuos reciben auditoría de estado causal y perturbaciones reales de la historia observable, manteniendo coeficientes fijos. Estos controles pertenecen al diseño histórico y están implementados; no se atribuyen al preflight P0 ni se declaran ejecutados para los dieciséis sistemas de la entrega actual.

## 10.5 Reconstrucción del preflight histórico y límites actuales

Desde el checkout registrado, en el entorno de §8 y con una ruta de salida separada, este comando regenera el recibo agregado sin ajustar modelos:

```bash
python scripts/e2_remediation_data_audit.py --output ../e2-data-audit-reproduction.json
```

El [registro de entrada](../_static/remediation_preflight/registration.json) preserva sus hashes y el [recibo publicado](../_static/remediation_preflight/data_support_audit.json) preserva los resultados observados. La [descarga unificada](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) incluye el procedimiento y las evidencias de esta etapa. Una reproducción puede cambiar la fecha de generación del recibo; comparar esa fecha o el hash del JSON completo no sustituye contrastar los mismos campos científicos e identidades de entrada.

El scoring actual está cerrado y sus predictores, cohortes y checkpoints tienen recibos propios en §§6/48/55. Permanece parcial la auditoría integral de todos los criterios y controles negativos sobre ese pipeline; no se declara aprobada por el preflight de soporte. El periodo 2025–2026 conocido de E1 conserva alcance retrospectivo. Un hash nuevo no crea un test nuevo y una auditoría de soporte no demuestra generalización independiente.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/10_auditoria_datos.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.


````
