(ruta-metodologica-series-de-tiempo-y-controles-causales)=
(e2-ruta-temporal)=
# 7. Ruta Metodológica según el Tipo de Problema

**Lectura vigente:** corresponde la familia **F2 temporal y la ruta F2a de series de tiempo**, con controles de disponibilidad de covariables desarrollados también en F2b. Aunque las redes reciben D7×5 OHLCV sin un canal explícito de Y rezagada, el diagnóstico de referencia mostró autocorrelación residual fuerte: la regla **RT1 de F2b exige reclasificar a series de tiempo**. Esta decisión conserva el contrato E1–E2–E3; no añade variables, cambia la partición ni ordena entrenamientos nuevos.

El benchmark de [§6](06_benchmark_final.md) está ejecutado: **16 sistemas, 81 predictores congelados y 608 fechas comunes**. La selección, el ajuste y sus incidencias se conservan en [§48](48_protocolo_reducido.md); [§49](49_diagnosticos_f2a.md), [§50](50_comparaciones_f2a.md), [§56](56_inferencia_estabilidad_final.md) y [§60](60_calibracion_temporal_condicional.md) aportan diagnósticos posteriores con hipótesis y alcances propios. **Ejecución del benchmark y adecuación estadística de un modelo son conclusiones distintas:** los resultados actuales no satisfacen conjuntamente los cuatro criterios de adecuación de la guía. La [ampliación del 16/11 (§65)](65_cierre_cientifico_real.md) incorpora los controles sobre entradas reales, la comparación formal de los periodos originales y una IUT CPU completa; sus resultados conservan esa distinción.

Este capítulo sigue el orden **7.1–7.5 de la guía, pp. 20–40**. La [lista verificable](../_static/e2_route_current_v1/guide_checklist.csv), la [trazabilidad de decisiones](../_static/e2_route_current_v1/decision_traceability.csv), el [mapa de batería por cada uno de los veinte pares primarios](../_static/e2_route_current_v1/contrast_battery_map.csv), el [índice de evidencias](../_static/e2_route_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_route_current_v1/manifest.json) distinguen pruebas ejecutadas, requisitos pendientes y alternativas no aplicables. El {download}`índice de esta consolidación <../_static/e2_route_current_v1/README.md>` y la {download}`versión exacta anterior <../_static/e2_route_current_v1/07_ruta_temporal_before_update.md>` conservan sus fuentes; no se recalculan resultados. Esas listas conservan su corte documental; la evidencia adicional del 16/11 se consulta en [§65](65_cierre_cientifico_real.md).

(e2-ruta-mapa-vigente)=
## 7.1 Mapa de rutas y justificación de F2a

| Familia de la guía | Tipos que agrupa | Aplicación al proyecto |
|---|---|---|
| F1 · Tabular | Clasificación y regresión tabular | No es la ruta principal: presentar covariables como una matriz no elimina el orden, la dependencia ni el objetivo temporal. |
| **F2 · Temporal** | **Series de tiempo y regresión temporal** | **Aplica F2a** por el diagnóstico RT1. Los controles causales de covariables de F2b complementan la justificación. |
| F3 · Espacial | Regresión espacial y espacio-temporal | No aplica: no hay unidades geográficas o una dimensión espacial que deban bloquearse. |
| F4 · Imágenes | Clasificación y segmentación de imágenes | No aplica: las arquitecturas transferidas no convierten las entradas OHLCV en imágenes. |
| F5 · Lenguaje y secuencias | Clasificación de texto; etiquetado secuencial o secuencia a secuencia | No aplica como ruta de lenguaje: no hay documentos, etiquetas por token ni una salida textual/secuencial de esa naturaleza. |

La {ref}`referencia Ridge forward de desarrollo <e2-diagnostico-forward>` utilizó las cinco covariables D7, α=1 fijado y diez folds históricos de 2024, con **305 fechas**. Sus residuos mostraron **ACF(1)=0,632228** y Ljung–Box lag7 **Q=266,4062, p≈8,88183×10⁻⁵⁴**, p Holm≈5,77319×10⁻⁵² dentro de 66 diagnósticos de EDA. La regla RT1 y el EDA temporal de la guía conducen a F2a, decisión documentada del 06/11/2026. La [trazabilidad histórica de la reclasificación](../_static/eda_traceability/decisiones_vigentes.csv) distingue esa decisión de su incorporación editorial posterior. La [resolución de ruta y alcance vigentes](../_static/e2_route_current_v1/route_decision.json) conserva ese linaje sin importar los catálogos o calendarios anteriores como protocolo actual.

Ridge fue una referencia diagnóstica de desarrollo, no un cuarto modelo base E1 ni un sistema obligatorio adicional. Ese resultado no demuestra no linealidad, que añadir Y rezagada mejore la predicción ni que todos los residuos futuros se comporten igual. El objetivo σ₇ comparte seis retornos entre fechas consecutivas, lo que también contribuye a la dependencia.

Se conserva **D7×5 OHLCV → σ₇ poblacional de siete retornos, h=1**, sin anualización, y la **media de las RMSE individuales por semilla**. El margen práctico es **δ=0,00025772842664981106**. La [herencia de métrica, margen, unidad y partición](01_punto_partida.md) se acepta conforme a la guía: el conocimiento previo del test de E1 limita la confirmación prospectiva, sin exigir otro holdout por la mera continuidad del trabajo.

(e2-ruta-protocolo-comun-vigente)=
## 7.2 Protocolo común: once pasos y criterios de adecuación

### 7.2.1 Cómo se ejecutó cada paso

| Paso de la guía | Evidencia y resultado actuales | Alcance o límite |
|---|---|---|
| **1. Preregistrar** | [§1](01_punto_partida.md), [§9](09_protocolo_registrado.md) y [§48](48_protocolo_reducido.md) conservan contrato, δ, semillas, comparaciones y enmiendas previas a sus ajustes. RQ1 se concreta en veinte pares, Holm20, α familiar 0,05. | No se atribuye carácter prospectivo al test conocido ni a los suplementos posteriores. RQ2/RQ3 conservan su alcance posterior; RQ2 ya fue ejecutada en [§64](64_correcciones_metodologicas.md), sin recuperar confirmación prospectiva. |
| **2. Auditar datos** | [EDA vigente](02_eda.md) y [auditoría de partición](../_static/e2_eda_current_v1/partition_audit.json): máscaras, disponibilidad, IDs, soporte y hashes comprobados mediante registros guardados. | El cotejo documental no reabre todos los archivos privados ni certifica por sí solo ausencia integral de fuga. |
| **3. Particionar por la unidad correcta** | Tres folds HPO cronológicos: **253/435/527** ejemplos de entrenamiento y **83** de validación cada uno. Final: **655 entrenamiento, 53 calibración y 608 prueba**. Embargo heredado de siete días en train–validación/calibración. | La unidad inferencial es el calendario de una serie; semillas y ventanas solapadas no son mercados independientes. Calibración y prueba comparten historia causal, sin compartir IDs de objetivos ajustados. |
| **4. Pipeline causal** | [Implementación](04_implementacion.md): transformaciones ajustadas sólo con train de cada fold; ventanas hasta t−1, máscaras sin comprimir huecos y recetas finales congeladas. | La normalización relativa de la ventana y el escalador aprendido son operaciones diferentes; no se usa OHLCV completo de t al emitir. |
| **5. Referencias** | MLP16, LSTM32 y GRU32 de E1, más Persistence, SeasonalNaive y ARIMA, evaluados junto al Top 10. | Son 16 sistemas; los clásicos aportan R=1. No superar significativamente al ingenuo impide declarar valor añadido demostrado, pero sus resultados se conservan. |
| **6. Igualar ajuste** | [§11](11_hpo_equidad.md): seis configuraciones por red, mismos tres folds y semilla HPO 271201; **231/234 oportunidades válidas**, incidencias conservadas. | Igual número de plazas no demuestra igual coste total: SAE tiene tres fases; existen políticas históricas y hardware distintos en selección. Las seis plazas no son la NAS condicional de 1.000 arquitecturas. |
| **7. Repetir y guardar cada ejecución** | **Seis semillas 271801–271806 por red**, 78 ajustes finales, más tres clásicos; pesos y trayectorias identificados. | R=6 supera el mínimo numérico; el piloto estimó potencia condicional **12,3 %**, inferior a 80 %. Una trayectoria determinista no se convierte en seis entrenamientos al emparejarla. |
| **8. Diagnosticar ajuste/sobreajuste** | [§49](49_diagnosticos_f2a.md) recupera curvas y ejecuta residuos/calibración; [§56](56_inferencia_estabilidad_final.md) añade estabilidad y brechas descriptivas. | Hay evidencia residual adversa. [§64](64_correcciones_metodologicas.md) añade trece contrastes de brecha en CPU con dos tolerancias condicionales; no certifica adecuación del benchmark CUDA. Los diagnósticos posteriores no se presentan como filtro anterior al entrenamiento. |
| **9. Evaluar en prueba con IC** | A2 común en **608 fechas**, 81 predictores previamente congelados, sin retuning. Original B=4.999; [refinamiento B=20.000](56_inferencia_estabilidad_final.md) conserva exactamente hipótesis y conclusiones. | El calendario se hereda de E1 y sigue siendo conocido. Los remuestreos reutilizan pronósticos fijos; no son nuevas evaluaciones de recetas seleccionadas con test. |
| **10. Comparar y controlar multiplicidad** | [§6](06_benchmark_final.md) y [§50](50_comparaciones_f2a.md): efectos, IC y p brutos/ajustados; Holm20 primario, DM–HLN120, MCS y rangos condicionados con familias separadas. | Se ejecutaron tres procedimientos por cada par primario, con sus ajustes Holm20. §7.3 identifica los supuestos y dominios de incertidumbre que deben justificar su validez, sin certificarla por un contador. |
| **11. Analizar y discutir** | Robustez/oclusión en [§54](54_robustez_interpretabilidad.md), tres ablaciones en [§55](55_reproduccion_ablaciones.md), errores/techo en [§57](57_errores_techo.md)/[§63](63_ensamble_calibracion.md), calibración temporal en [§60](60_calibracion_temporal_condicional.md). | Son evidencias diferenciadas y mayoritariamente posteriores; no prueban causalidad, adecuación universal o techo Bayes. [§64](64_correcciones_metodologicas.md) ejecuta RQ2 con su alcance exploratorio propio. |

El ajuste final usa objetivos hasta **31/10/2024**; el embargo es **01–07/11**, la calibración **08/11–30/12** y el **31/12** sólo aporta historia causal antes del test **01/01/2025–31/08/2026**. La [auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json) verifica separación train–calibración y train–prueba. Entre calibración y prueba hay **seis cierres y cinco retornos históricos compartidos**, con IDs de objetivo distintos. Ese uso causal no equivale a reajustar parámetros, pero impide afirmar independencia o soportes globalmente disjuntos. El avance de estado ARIMA con coeficientes congelados se distingue de un nuevo ajuste.

### 7.2.2 Las cuatro condiciones deben cumplirse simultáneamente

| Criterio de adecuación de la guía | Evidencia actual y decisión |
|---|---|
| **i. Superar al trivial con mejora >δ y p ajustado <0,05** | Ninguno de los diez candidatos rechaza la nula de ganancia ≤δ frente a Persistence en Holm20. SAE tiene la mejor ganancia puntual, pero p Holm=1 e IC que incluye cero. **No acreditado.** |
| **ii. Brecha train–validación menor que δg con evidencia estadística** | **δg sí fue registrado previamente**: 0,00025772842664981106. [§64](64_correcciones_metodologicas.md) ejecuta una prueba de media de brecha en CPU, con seis semillas y Holm13: dos tolerancias condicionales y once no establecidas. La ausencia general de sobreajuste y la adecuación conjunta siguen **no establecidas**. |
| **iii. Calibración y ausencia de estructura residual** | Mincer–Zarnowitz rechaza 37/81 restricciones de calibración puntual; Ljung–Box rechaza en **81/81 trayectorias** a lag7 y lag14 tras Holm826. La cobertura también depende de modelo/grupo. **No se acredita ausencia de estructura ni calibración universal.** |
| **iv. Prueba consistente con validación** | [§65](65_cierre_cientifico_real.md) ejecuta la comparación formal de **calibración53 frente a prueba608 de los mismos predictores A100 originales**, con t sobre seis diferencias y Holm13: seis rechazos. La sensibilidad por signos da cero rechazos. Calibración no es validación HPO independiente; la diferencia de periodos no identifica sobreajuste, y un no rechazo no prueba consistencia o equivalencia. **La adecuación de este criterio no queda establecida.** |

No se declara adecuado a un candidato por su menor RMSE o por un no rechazo aislado. El benchmark aporta resultados ejecutados, incluidos los desfavorables; la falta de adecuación conjunta no se confunde con falta de ejecución del trabajo.

(e2-ruta-bateria-vigente)=
## 7.3 Batería de pruebas: aplicación, ejecución y límites

### 7.3.1 Tablas A, B y C aplicadas a este problema

| Tabla de la guía | Evidencia aplicable ejecutada | Pendientes y alternativas |
|---|---|---|
| **A · Ajuste y adecuación** | Comparación con ingenuo; Mincer–Zarnowitz HAC14; Ljung–Box de residuos y cuadrados lag7/14; ARCH-LM7; BDS2/3; Jarque–Bera; cobertura con IC temporales. La familia residual tiene **707/826 rechazos**. [Resultados completos](../_static/e2_f2a_audit_20261113/tests.csv). | [§65](65_cierre_cientifico_real.md) ejecuta además trece controles reales con etiquetas de train permutadas por bloques: no hay alertas tras Holm13, sin certificar ausencia universal de fuga. No se ejecutaron Shapiro/Breusch–Pagan/White; JB y diagnósticos temporales sí tienen resultados, sin equivalencia entre sus nulas. PIT no es estimable con un solo intervalo90 %; Kupiec/Christoffersen no están implementados. La cobertura HAC/ACI no los sustituye. Pruebas de clasificación y espaciales no aplican. |
| **B · Ausencia de sobreajuste** | Curvas de época: **78 finales y 29 HPO**, con diez HPO sin pérdidas originales; comparación calibración–prueba y brechas descriptivas. GR y constancia: veinte diagnósticos de cada tipo, **cero rechazos Holm40**, B=20.000. [Brechas guardadas](../_static/e2_temporal_final_audit_20261114/saved_training_calibration_gaps.csv). | [§64](64_correcciones_metodologicas.md) publica trece pruebas de δg en CPU, con riesgos train/cal comparables y alcance condicional. [§65](65_cierre_cientifico_real.md) añade la comparación formal calibración–prueba original, sin establecer equivalencia ni validación independiente. La paridad CUDA del suplemento anterior no se acredita. Las curvas por época **no son curvas por tamaño de entrenamiento**. Bootstrap de optimismo con reentrenamiento, curvas 10–100 % y otras alternativas de Tabla B no se atribuyen a estos artefactos ni se ordenan como una lista íntegramente obligatoria. |
| **C · Diferencias entre modelos** | Bootstrap primario pareado; t y cambios de signo condicionales; DM–HLN **35/120 rechazos Holm120**; MCS90 % de cinco sistemas; Friedman/Iman–Davenport condicionado a seis semillas y trece redes; **78 Wilcoxon exactos, cero rechazos Holm78** y [diagrama de rangos](../_static/e2_benchmark_current_v1/conditional_rank_groups.svg). | Los tres métodos por par están ejecutados; su contador no acredita automáticamente los supuestos ni el cumplimiento integral. [§64](64_correcciones_metodologicas.md) añade d de Cohen pareado e IC entre semillas para los veinte pares; permanecen los efectos e IC temporales RMSE/MSE en sus unidades originales. TOST/ROPE no están ejecutados: no se afirma equivalencia. Nemenyi es alternativa a Wilcoxon–Holm, no otra prueba obligatoria adicional. Clark–West requiere anidamiento pertinente; no se inventa. |

Para una pérdida, la orientación del margen de Tabla B es **g = RMSE_validación − RMSE_train**, con nula de sobreajuste inaceptable **mediana(g)≥δg** y alternativa **mediana(g)<δg** al usar Wilcoxon. Una t sobre ejecuciones evaluaría la **media** de las brechas, con sus propios supuestos y unidad justificados. El [linaje de δg](../_static/e2_point1_review_v1/margin_lineage.json) fija su existencia previa en `599a29f` y la dirección en `ef2be289`; campos históricos `delta_g=null` no prueban ausencia del registro. Los riesgos de entrenamiento guardados se evaluaron con el lote configurado; los pronósticos de calibración, con lote1 y otro periodo. El suplemento posterior [§64](64_correcciones_metodologicas.md) sí genera predicciones por fecha train/cal comparables en CPU y contrasta la media entre semillas, sin atribuirles incertidumbre temporal o paridad CUDA. Los resultados del P0 histórico tampoco sustituyen el catálogo vigente.

### 7.3.2 Mínimo de tres pruebas por comparación confirmatoria

La guía, p.23, exige **al menos tres pruebas distintas de Tabla C por cada contraste confirmatorio**, con: **i)** una paramétrica o de remuestreo pareado; **ii)** una no paramétrica o bayesiana; **iii)** un procedimiento global o con control de comparaciones múltiples, justificando sus supuestos. No basta aplicar Holm a una sola prueba. El [protocolo registrado](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_reduced_protocol_20261112/protocol.json) lo explicita: **«Holm alone is not a third test»**.

El [motor original](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_reduced_final_analysis.py) publica **tres procedimientos realmente ejecutados en los veinte pares**: bootstrap temporal/semillas, t pareada y cambios de signo. Comparten la ganancia puntual y δ; cada lista recibe su **Holm20 separado**. El [recibo de congelación](../_static/e2_benchmark_final_20261113/final_freeze.json) registra `analysis_frozen=true` y el hash del análisis antes del scoring de esta campaña. Se reconoce esa evidencia, sin convertir el test conocido desde E1 en confirmación prospectiva nueva.

- El **bootstrap pareado** representa la ganancia de RMSE>δ, variando semillas y bloques del calendario. El refinamiento B=20.000 conserva las veinte decisiones; cambiar B o la longitud de bloque no crea pruebas adicionales.
- La **t pareada** y la **enumeración de signos** también están ejecutadas con ese margen, condicionadas al calendario fijo. La primera requiere independencia entre trayectorias y una aproximación normal adecuada de seis ganancias; la segunda, simetría respecto a la nula. Enumerar64 signos no demuestra ese supuesto ni transforma la variación entre semillas en incertidumbre entre mercados.
- **Holm aporta el control de comparaciones múltiples previsto por la guía**, junto a los tres métodos existentes. No se cuenta como una cuarta prueba ni se exige Friedman/MCS adicional para sustituir la alternativa explícita «global o con control de comparaciones múltiples». Tampoco se exige independencia entre los tres métodos como un requisito añadido.

El [mapa de los veinte pares](../_static/e2_route_current_v1/contrast_battery_map.csv) conserva la presencia de cada método, margen, familia y alcance. Hay tres procedimientos, pero el contador **no demuestra sus supuestos ni tres confirmaciones temporales homogéneas**: bootstrap y diagnósticos de semillas representan fuentes de incertidumbre distintas. La normalidad residual no verifica la normalidad de seis diferencias, y la validez de la aproximación temporal requiere dependencia suficientemente corta y bloques adecuados. Estas limitaciones impiden certificar el cumplimiento integral y la adecuación por la mera cantidad de salidas; no se informa que los tres métodos falten.

DM y MCS posteriores usan **MSE diaria media de trayectorias fijas e igualdad contra cero**; Friedman/Wilcoxon describen rangos o diferencias entre seis semillas, excluyendo clásicos de esos bloques. Son comparaciones exploratorias adicionales con hipótesis y familias propias, no tres confirmaciones retrospectivas del margen δ ni votos para cambiar la decisión primaria. Correcciones como Nadeau–Bengio pertenecen a comparaciones pertinentes con particiones solapadas; no se trasladan automáticamente a esta batería de calendario y semillas.

(e2-ruta-detalle-temporal-vigente)=
## 7.4 Ruta F2a y controles complementarios de F2b

### 7.4.1 F2a: aplicación de S1–S8

| Paso F2a | Aplicación y evidencia | Límite actual |
|---|---|---|
| **S1. Horizonte, frecuencia y exógenas** | Frecuencia diaria UTC, **h=1**, una salida σ₇; ventana de siete días completos hasta t−1. [Contrato](17_contrato_sincronizado.md). | No hay varios horizontes ni covariables futuras conocidas que se hayan incorporado. |
| **S2. EDA temporal y transformaciones** | [§2](02_eda.md) amplía E1: STL, ADF/KPSS, ACF/PACF, heterocedasticidad, continuidad y cambios, con fechas y decisiones. | Evidencia de tramos distintos no prueba estacionariedad global ni ruido irreducible; no se diferencia Y por automatismo. |
| **S3. Origen móvil y embargo** | Tres orígenes internos de selección con embargo7; final train655/cal53/test608. [Auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json). | Son tres folds HPO, no diez folds actuales. El test diario usa política de pesos fijos, sin reajuste en cada fecha. |
| **S4. Ajustes sólo con el pasado** | Transformaciones por train, ventanas y rezagos causales; sin imputación con futuro ni compresión de huecos. | El historial compartido entre ventanas es dependencia; no se presenta como intersección de información cero. |
| **S5. Baselines y presupuesto** | Persistence, SeasonalNaive7 y ARIMA(2,0,1), Top10 y tres redes E1. Misma selección interna de las redes; oportunidades y fallos conservados. | ARIMA satisface la alternativa ARIMA/ETS; ETS no es un requisito adicional automático. Igualdad de oportunidades no acredita igualdad total de cómputo. |
| **S6. Evaluación por horizonte/origen** | Un horizonte y **608 fechas de emisión**; A2, métricas por trimestre y ventanas temporales guardadas. | Trimestres y ventanas solapadas no son series independientes. No se mezclan medias mensuales con el estimando primario por trayectoria. |
| **S7. Residuos, calibración y estabilidad** | Ljung–Box, ARCH, BDS, JB, Mincer–Zarnowitz y cobertura; [GR](56_inferencia_estabilidad_final.md) y [calibración temporal](60_calibracion_temporal_condicional.md). | Residuos con estructura; δg evaluado posteriormente en CPU con alcance propio; no PIT ni garantía condicional universal. |
| **S8. Comparación y robustez** | [DM–HLN/MCS/rangos](50_comparaciones_f2a.md); perturbaciones de ruido, extremos, faltantes y nivel, tres severidades. [§54](54_robustez_interpretabilidad.md). | Friedman es condicionado a semillas, no poblacional sobre series/orígenes independientes. Estrés sintético no demuestra resistencia a otro mercado. |

La guía permite **MASE o una razón frente al ingenuo**. La [tabla de error relativo de los dieciséis sistemas](../_static/e2_route_current_v1/relative_error_vs_naive.csv) añade una presentación aritmética de las RMSE ya publicadas:

| Sistema | RMSE media / RMSE Persistence | RMSE media / RMSE SeasonalNaive |
|---|---:|---:|
| SAE | 0,930936 | 0,313076 |
| ARIMA | 0,959866 | 0,322806 |
| GRU32 | 0,979334 | 0,329353 |

Un cociente menor que 1 indica menor RMSE puntual que esa referencia. Son **cocientes de la media de las RMSE individuales respecto a la RMSE del comparador R=1**, sobre las mismas 608 fechas; no son MASE, nuevas pruebas ni una demostración de mejora mayor que δ. La [procedencia del cálculo descriptivo](../_static/e2_route_current_v1/ratio_provenance.json) identifica los datos A2. No se cambia la primaria ni se añade SeasonalNaive a Holm20.

La primaria **RMSE heredada** se mantiene; las métricas recomendadas de F2a no autorizan cambiarla tras ver la prueba. Se reportan MAE, R², sMAPE, cobertura, anchura y ganancias frente a naive. Pinball y score de intervalo están en §60. QLIKE, CRPS o acierto direccional no se presentan como calculados; su uso depende de la cantidad pronosticada y de una distribución o decisión pertinente. El proyecto no emite VaR por disponer de intervalos de σ₇. Las comparaciones locales/globales entre múltiples series y el control por múltiples horizontes no aplican a una sola serie y h=1.

### 7.4.2 F2b: qué conserva la comprobación de covariables

| Paso F2b de la captura de la guía | Resolución para el contrato actual |
|---|---|
| **RT1. Confirmar regresión temporal o reclasificar** | Las entradas excluyen un canal explícito Y rezagada, pero la referencia Ridge deja autocorrelación fuerte: **se sigue F2a**, con la evidencia de §7.1. |
| **RT2. Disponibilidad y desfases** | OHLCV completo de t−k, k≥1, disponible al emitir; OHLCV completo de t y σ₇(t), aún no. σ₇(t−1) puede usarse en referencias/regímenes causales, sin añadirlo a las redes. [Contrato de disponibilidad](../_static/eda_decisions/heredado/eda_temporal_specific/availability_contract.csv). |
| **RT3. Ventana y asociaciones rezagadas** | Se conservan D7 heredado y las 140 asociaciones de rezagos 1–28 sobre cohorte común de 308 resultados de 2023. Los máximos marginales no se presentan como selección optimizada ni causal de una nueva ventana. |
| **RT4. Partición y embargo** | Cronología, embargo7 entre train y validación/calibración y transformaciones sólo con pasado; soportes y puente 31/12 explícitos. No se traslada a F1 por el formato de matriz. |
| **RT5. Referencias y presupuesto** | La referencia Ridge de desarrollo documenta RT1. **No se añade Ridge ni una media móvil al catálogo obligatorio vigente**: los baselines finales corresponden a F2a. |
| **RT6. Error, residuos y deriva** | RMSE/MAE/R² finales; Ljung–Box y ARCH de 81 trayectorias, 112 filas trimestrales y 8.304 ventanas de 90 días. Descripciones dependientes, no miles de nuevas réplicas. |
| **RT7. Comparar** | DM–HLN, Wilcoxon y Friedman–Holm disponibles con los alcances de §7.3; bootstrap temporal emparejado y familias declaradas. No se considera una segunda batería confirmatoria independiente. |

Estos controles conservan la disponibilidad de las covariables y la interpretación de la dependencia. La reclasificación no requiere cambiar silenciosamente X, Y o los comparadores ya congelados. Una sensibilidad con rezagos de Y necesitaría su diseño y evaluación de desarrollo propios; no es un resultado nuevo de esta actualización documental.

(e2-ruta-decision-vigente)=
## 7.5 Guía de decisión aplicada al estado actual

| Situación observada | Decisión que permite la evidencia |
|---|---|
| No hay superioridad práctica ajustada frente a naive | Conservar A2 y resultados negativos; describir SAE como menor RMSE observada, sin declarar valor añadido confirmado ni seleccionar otra receta con test. |
| Brechas y δg previo | Conservar el margen y la orientación correcta. [§64](64_correcciones_metodologicas.md) ejecuta trece pruebas con riesgos train/cal comparables en CPU: dos tolerancias condicionales y once no establecidas. [§65](65_cierre_cientifico_real.md) añade la comparación formal de los riesgos originales calibración–prueba. No afirmar ausencia general de sobreajuste, equivalencia entre periodos o adecuación conjunta. |
| Deriva entre train y prueba reales | El adversarial obtiene AUC 0,95254, p=0,001. La separación se localiza descriptivamente en volumen (AUC 0,90265 frente a 0,49734 con OHLC). Conservar la alerta, las métricas por trimestre/régimen y los límites de transporte; no retirar atributos o reajustar con test. [§65](65_cierre_cientifico_real.md). |
| Mejora conjunta de error y coste | La nueva IUT CPU está ejecutada, con ambos componentes del mismo backend y Holm10: cero mejoras conjuntas. El resultado se limita a esa política de forward residente y no resuelve por sí solo el coste A100 o del servicio. [§65](65_cierre_cientifico_real.md). |
| Residuos autocorrelacionados y heterocedásticos | Mantener F2a, incertidumbre temporal robusta y limitaciones. La recomendación de revisar rezagos/capacidad es trabajo de desarrollo con protocolo propio; no modifica los predictores publicados. |
| Cobertura global próxima a 90 %, pero desigual por grupos | Reportar longitud y score junto a cobertura; no confundir adaptación ACI con garantía condicional, PIT completo ni cumplimiento universal. |
| Heterogeneidad Friedman, sin pares Wilcoxon ajustados | Mostrar rangos y sus conexiones de no rechazo. Con seis pares, la resolución es limitada; ni significancia global ni no rechazo justifican una pareja superior o equivalente. |
| Potencia y variación entre semillas limitadas | Reportar R=6, DE, IC y potencia piloto 12,3 %. No aumentar R automáticamente después de observar prueba ni crear repeticiones clásicas ficticias. |
| Proxy de techo y costes parciales | Mantener la incertidumbre de γ y distinguir recursos medidos de desconocidos; no justificar más complejidad con un techo óptimo no identificado. |

La siguiente decisión sobre un modelo requiere evidencia de desarrollo y un protocolo compatible con la continuidad del proyecto. **Esta consolidación mantiene congelados modelos, recetas, pronósticos y calibradores originales**; los suplementos de calibración/ensemble conservan sus reglas y familias separadas. El estado es **ruta F2a justificada y benchmark ejecutado, con adecuación conjunta no demostrada y requisitos inferenciales concretos pendientes**, no ausencia de trabajo ni aprobación automática de todos los requisitos. Los capítulos y recibos históricos se conservan íntegros a continuación.

````{admonition} Archivo documental: contenido anterior del capítulo 7
:class: dropdown

Este bloque conserva el texto anterior y sus encabezados para mantener sus enlaces. Sus fechas y estados describen sus cortes originales; la lectura vigente sigue los apartados 7.1–7.5 presentados arriba.

**Lectura vigente:** el protocolo ejecutado es [§48](48_protocolo_reducido.md), los resultados comunes están en [§6](06_benchmark_final.md), la reproducción registrada en [§55](55_reproduccion_ablaciones.md) y el Apéndice B actual en [§61](61_reproduccion_consolidada.md). Las cifras científicas y los recibos históricos conservan su fecha y su alcance.

**Estado:** el benchmark reducido está completado en [§6](06_benchmark_final.md), con el calendario efectivamente ejecutado de [§48](48_protocolo_reducido.md). La auditoría posterior de ajuste, residuos, calibración y estabilidad temporal está en [§49](49_diagnosticos_f2a.md); las comparaciones S8 se documentan en [§50](50_comparaciones_f2a.md), mientras robustez y otros requisitos conservan su estado en [§53](53_cierre_requisitos.md). Los estados y calendarios anteriores son antecedentes, no trabajos finales pendientes.

**Antecedente de planificación del 12/11/2026:** ese corte fue sustituido por la ejecución documentada en [§48](48_protocolo_reducido.md). Su texto exacto se conserva en el archivo enlazado al final; no describe trabajos finales pendientes.

**Calendario ejecutado:** tres folds de selección, con **253/435/527 ejemplos de entrenamiento y 83 de validación en cada fold**; ajuste final de **655 ejemplos**, calibración de **53** y evaluación heredada de **608**, con política de pesos fijos. La [auditoría del EDA vigente](../_static/e2_eda_current_v1/partition_audit.json) distingue IDs y soportes. Los diez folds mensuales del EDA inicial y los diseños de §§40/45 conservan su identidad histórica.



**Guía y aplicación:** los controles F2a actuales están en §§49–57. La NAS es condicional a cuerpos sin arquitectura definida; las seis plazas HPO de §48 no constituyen NAS. Los planes anteriores se conservan como antecedentes y no ordenan nuevas corridas.


La ruta vigente es **F2a, series de tiempo**, decidida el 06/11/2026 tras recibir la guía completa de 65 páginas. El [catálogo obligatorio de 16 sistemas](43_catalogo_obligatorio.md) conserva esa clasificación, el objetivo y las precauciones causales. El protocolo efectivamente ejecutado y sus enmiendas están en §48, y los recibos finales en §§6/55. El híbrido adicional y Ridge son opcionales/históricos, sin superioridad demostrada.

El [registro de decisiones del EDA](../_static/eda_traceability/decisiones_vigentes.csv) conserva la procedencia de la reclasificación y la fila histórica `RESID`. Su incorporación documental el 08/11/2026 no es una decisión experimental retroactiva.

## 7.1 Motivo de la reclasificación y contrato que se conserva

La {ref}`referencia Ridge forward <e2-diagnostico-forward>` dejó ACF(1)=0,632228 en 305 fechas de validación de 2024. Ljung–Box de lag7 dio Q=266,4062 y p≈8,88×10⁻⁵⁴, con sus limitaciones asintóticas y de dependencia. Es evidencia histórica del EDA, no un modelo adicional del catálogo vigente. La regla RT1 (PDF p32) y el diagnóstico del EDA C (p6) determinan seguir F2a; excluir Y rezagada de las entradas no convierte por sí solo la tarea en regresión iid.

Se conservan **D7×5 OHLCV → σ₇, horizonte h=1**, escala, primaria RMSE media por trayectoria y **δ=0,00025772842664981106**. Los periodos ya evaluados mantienen su condición de conocidos. Las referencias obligatorias de E1 son MLP16, LSTM32 y GRU32, adicionales a los diez modelos; Persistence, SeasonalNaive y ARIMA completan los 16 sistemas.

La dependencia entre etiquetas σ₇ consecutivas, con seis retornos compartidos, contribuye a la autocorrelación. Ese diagnóstico no demuestra que un canal nuevo mejore fuera de muestra ni prueba no linealidad. Reclasificar la ruta tampoco añade automáticamente Y rezagada como sexto canal de las diez redes.

## 7.2 Información endógena y alcance de los controles

Los controles temporales actuales se vinculan al protocolo ejecutado de [§48](48_protocolo_reducido.md), a la [admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json) y a los diagnósticos posteriores de [§49](49_diagnosticos_f2a.md). Persistence, SeasonalNaive y ARIMA ya aportan resultados de Bitcoin en la cohorte común. Los controles del diseño sustituido quedan en el [archivo del protocolo](../_static/remediation_protocol/preregistration.json), sin trasladar su catálogo o presupuesto al actual.

El objetivo σ₇ necesita ocho cierres consecutivos para siete retornos. Usar una etiqueta histórica exige auditar todos esos cierres, no solo la fecha nominal de la etiqueta. Una referencia con mayor historia puede ser causal y, aun así, disponer de información distinta de las redes D7×5. Las comparaciones deben informar esa diferencia y usar una cohorte elegible común sin comprimir huecos.

El último objetivo de calibración, 30/12/2024, y el primero de evaluación, 01/01/2025, comparten seis cierres y cinco retornos. Sus IDs son distintos; la historia compartida y el puente del 31/12 son admisibles al avanzar el origen con parámetros congelados. Esa dependencia debe conservarse en la incertidumbre temporal. La [auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json) no declara intersección cero entre todas las etapas.

**Antecedente opcional:** CNN–LSTM–atención conservó el mismo soporte D7×5 y controles causales propuestos. El híbrido y Ridge no amplían los 16 sistemas obligatorios ni aportan superioridad acreditada. Sus ablaciones propuestas no se confunden con las tres ablaciones entrenadas del autoencoder que sí publica [§55](55_reproduccion_ablaciones.md). [Contrato vigente](17_contrato_sincronizado.md).

## 7.3 Política causal de ajuste y actualización

El protocolo ejecutado distingue selección de hiperparámetros, ajuste de pesos, escalado, actualización de estado y scoring. Cada operación identifica las fechas que puede usar. El horizonte h=1 no elimina el embargo heredado ni permite ajustar escaladores con validación o test. Los tres folds HPO preceden a la evaluación externa; su receta global no se atribuye a pronósticos externos de enero–octubre de 2024.

El ajuste final usa etiquetas hasta el **31/10/2024**; el embargo es **01–07/11**, la calibración **08/11–30/12**, y el **31/12** aporta sólo historia causal. Los pesos, escaladores y calibradores permanecen fijos en las **608 fechas de 01/01/2025–31/08/2026**. La incorporación de una observación ya disponible al estado de ARIMA con coeficientes congelados no equivale a reestimar sus parámetros. Las fechas faltantes permanecen en el calendario y la elegibilidad se decide por soporte, no por el error obtenido.

Las restricciones de salida y la gestión de fallos conservan las reglas registradas. Una predicción no finita se registra como fallo; no se sustituye por cero ni se elimina una fecha favorablemente. La auditoría actual consulta registros guardados y no vuelve a certificar mediante ejecución los kernels de los modelos.

## 7.4 EDA temporal heredado: evidencia reutilizada y cobertura

El EDA de E1 se amplía, como exige la guía; no se desecha ni se repite automáticamente. Los siguientes resultados ya existen en el [capítulo EDA de E1](https://jassan-bit.github.io/DeepLearning_202630/sections/02_eda.html). El {ref}`bloque B del EDA actual <e2-eda-series-heredado>` explicita para cada uno la decisión de preprocesamiento, arquitectura, validación y riesgos. Se distinguen de los diagnósticos finales ya ejecutados en §49, posteriores al benchmark.

| Requisito F2a / EDA B | Evidencia ejecutada de E1 | Uso y límite para E2 |
|---|---|---|
| Tendencia y estacionalidad | STL robusto de log-cierres diarios, 25/03–31/12/2023, periodo7; E1 §2.5.1 | Apoya representación causal; periodo impuesto no prueba estacionalidad ni habilita STL centrado como predictor |
| Estacionariedad | ADF con constante/7 rezagos y KPSS constante/rezagos automáticos; σ₇ de 01/04–31/12/2023, n275, pADF≈0,002423 y pKPSS>0,10; E1 §2.5.2 | Compatibilidad de ese tramo con estacionariedad, sin garantía para 2024/2026; no diferencia automáticamente Y |
| Dependencia temporal | ACF/PACF y periodograma de retornos diarios, 26/03–31/12/2023, n281; E1 §2.5.3 | Distingue unidades y retornos frente a σ₇; no selecciona D o temporada por máximo visual |
| Heterocedasticidad | ARCH-LM sobre retornos diarios centrados, siete rezagos, p≈0,341; E1 §2.5.3 | No rechazo exploratorio no prueba homocedasticidad; ARCH-LM de las trayectorias actuales se reporta por separado en §49 |
| IID residual | BDS y Ljung–Box del estudio histórico de E1 por fold y semilla; E1 §2.5.6 | BDS contrasta iid; no atribuir todo rechazo a no linealidad ni trasladar el resultado al catálogo vigente |
| Anomalías y cambios | Máscaras de continuidad, extremos sin borrado; candidato CUSUM 15/10/2023 sin confirmación; E1 §2.5.1/2.5.5 | Preservar eventos legítimos y revisar sensibilidad; el candidato no prueba una ruptura causal |
| Dificultad de pronóstico | Persistencia forward de desarrollo: RMSE0,00474310, MAE0,00292428; E2 §2.2.3 | Referencia informal de predictibilidad; no es ruido irreducible ni techo validado |

Los CSV heredados de [estacionariedad diaria](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/stationarity_daily.csv), [estacionariedad de σ₇ y sensibilidades](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/stationarity_volatility.csv) y el [resumen de E1](https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/main/results/eda_development/summary.json) contienen especificaciones, rangos y advertencias; el capítulo fuente conserva sus figuras. No se exige recalcularlos por cambiar el nombre de la ruta. El {ref}`diagnóstico forward de E2 <e2-diagnostico-forward>` amplía esta evidencia con Ridge/persistencia sobre 305 fechas de 2024.

El **bloque C** permanece desarrollado en §2.2.1–2.2.3: estructura diaria y huecos, disponibilidad de cada covariable, asociaciones rezagadas sobre cohorte común, relaciones móviles y diagnóstico residual de referencia. La [trazabilidad actual](../_static/e2_eda_current_v1/decision_traceability.csv) y la [lista de cotejo](../_static/e2_eda_current_v1/guide_checklist.csv) cubren conjuntamente 2.1, B y C. El [perfil actual del objetivo](../_static/e2_eda_current_v1/target_profiles.csv) y su [ECDF](../_static/e2_eda_current_v1/target_ecdf.svg) describen la partición ejecutada sin inferencia iid. La [copia anual de covariables E1](../_static/e2_eda_current_v1/inherited_annual_profiles.csv) conserva variables del mismo día y referencia hasta el 03/12/2024; no representa entradas causales D7×5 ni se mezcla con las distancias actuales.

**Decisión de preprocesamiento:** mantener escalado causal, máscaras y calendario, sin diferenciación automática ni imputación de observaciones. **Impacto en arquitectura y validación:** mantener F2a y las operaciones causales de §48; discutir por separado los residuos finales de §49 y la predictibilidad informal del EDA. **Riesgos:** heredar conclusiones de otra frecuencia/tramo, confundir un contraste iid con prueba exclusiva de no linealidad o tratar el error naive como techo de Bayes. La entropía ordinal posterior de [§57](57_errores_techo.md) y el proxy empírico de [§63](63_ensamble_calibracion.md) conservan sus supuestos: ni identifican ruido externo de σ₇ ni cierran un techo de desempeño.

## 7.5 Verificación y estado

El módulo histórico `scripts/e2_temporal_baselines.py` y sus pruebas sintéticas se conservan sin cambios. Los resultados actuales de Persistence, SeasonalNaive y ARIMA están en §6, con su selección y congelación en §48; la evidencia sintética anterior mantiene un papel distinto de esas ejecuciones de Bitcoin.

Estado: **ruta F2a definida, benchmark de 16 sistemas completado y partición ejecutada comprobada mediante registros guardados**. La [auditoría actual](../_static/e2_eda_current_v1/partition_audit.json) distingue el soporte compartido legítimo de las etiquetas de ajuste. [§49](49_diagnosticos_f2a.md) publica diagnósticos de las 81 trayectorias finales; no se presentan como análisis anteriores al entrenamiento. El [P0](09_protocolo_registrado.md) conserva su identidad histórica. La herencia del test de E1 se acepta con exposición previa declarada; las limitaciones de ruido externo, cegamiento y otros pendientes de [§61](61_reproduccion_consolidada.md) impiden atribuir cierre integral a esta actualización.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/07_ruta_temporal.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.


````
