# Seguimiento de auditoría, inferencia, HPO y portabilidad

**Suplemento posterior del 16/11/2026.** Se ejecutan los análisis adicionales autorizados sobre los datos disponibles, con planes y recibos propios. La [inscripción del alcance](../_static/e2_closure_followup_20261116/scope_registration.json) continúa el presupuesto acumulado de 84 horas. Los resultados originales de A2/RQ1 y las versiones anteriores de los archivos se conservan. La tarea sigue siendo **F2a, serie de tiempo; NAS del capítulo 14 no aplica** al catálogo actual.

## Controles negativos: investigación y resultado adverso

La revisión de trazas de TSMixer e iTransformer verifica que el optimizador supervisado recibe únicamente las 655 ventanas de entrenamiento y sus etiquetas permutadas, sin selección con calibración o prueba. La escala del objetivo procede de entrenamiento y no cambia al permutar. Los `forward` revisados reciben X, sin etiquetas ni fechas futuras. [Revisión estática y límites](../_static/e2_closure_followup_20261116/audit/trace_review.json).

El [plan adicional](../_static/e2_closure_followup_20261116/audit/plan.json) fija tres nuevas permutaciones por bloques de siete días para cada una de esas dos redes, con la misma inicialización, receta y épocas anteriores. Se completaron **seis ajustes**. Las arquitecturas fueron elegidas para investigar los resultados anteriores; este suplemento es diagnóstico y no una familia confirmatoria nueva.

| Red | Ganancia media de RMSE frente a la constante de entrenamiento | IC90 % temporal de esa media finita | p TOST ajustado Holm2 |
| --- | ---: | --- | ---: |
| TSMixer | 0,000606372 | [0,000410671; 0,000845935] | 1 |
| iTransformer | 0,000627953 | [0,000506498; 0,000752111] | 1 |

La TOST utiliza el margen heredado **±δ=±0,00025772842664981106**, 10.000 remuestreos por bloques de catorce días y el promedio de las tres funciones registradas por red. **No establece equivalencia con la constante y no aprueba el criterio de rendimiento cercano al azar.** Cinco de los seis IC95 % marginales de Spearman están por encima de cero; la tercera permutación de TSMixer conserva asociación inconclusa. Se publican [los seis resultados y el resumen](../_static/e2_closure_followup_20261116/audit/summary.json), sin sustituir permutaciones por sus puntuaciones.

También se evalúan las dos funciones antes de entrenar, verificando su estado inicial. La asociación inicial y las diferencias de escala **no explican íntegramente la señal posterior**, especialmente en iTransformer. Los seis retornos ya conocidos que intervienen en σ₇ y los momentos de la ventana son información causal legítima; constituyen una hipótesis de investigación, no una explicación demostrada de estos resultados. {download}`Expediente y alcance de la auditoría <../_static/e2_closure_followup_20261116/audit/README.md>`.

La revisión no encontró una vía explícita de etiquetas futuras, pero eso no certifica ausencia universal de fuga. Las recetas y los cuerpos ya habían sido seleccionados con etiquetas reales: los controles no repiten toda la selección HPO bajo permutación. **B03 permanece parcial.** No se redefine retrospectivamente el azar para obtener aprobación.

## Validación HPO frente a prueba

Se recuperan **39 trayectorias de los ganadores HPO**, tres folds de 83 fechas por cada una de las trece redes. El contraste compara su RMSE concatenada de **249 fechas** con la media de las seis RMSE individuales de prueba sobre **608 fechas**. No utiliza calibración53 como validación HPO.

El [plan corregido antes del cálculo numérico](../_static/e2_closure_followup_20261116/inference/recovery_v3/analysis_plan.json) conserva dos admisiones fallidas por formatos históricos de fecha. La recuperación posterior del manifiesto sólo sella archivos existentes; no recalcula los resultados. El análisis usa **5.000 remuestreos**, bloques de catorce días y sensibilidades de siete y veintiocho, con remuestreo separado de los calendarios y sin emparejar artificialmente semillas entre periodos.

Sólo SAE rechaza igualdad con Holm13: **prueba−OOF=−0,001810259**, IC95 % marginal **[−0,003041891; −0,000958075]**, p ajustado **0,0103979**. El error observado es menor en prueba. Los otros doce no rechazan; eso no establece equivalencia. [Las 39 filas con sensibilidades](../_static/e2_closure_followup_20261116/inference/recovery_v3/result_v1/conditional_OOF_test_comparisons_39.csv).

**Son entrenamientos, tamaños de muestra y periodos distintos.** Las mismas fechas OOF participaron en la selección del ganador y sus épocas; el remuestreo no repite esa selección. Este contraste está ejecutado, pero no restaura validación externa independiente del procedimiento ni identifica sobreajuste, deriva o su ausencia como causa de las diferencias.

## Descomposición actual de variabilidad

Se descompone la variación de las **pérdidas cuadradas de 13 redes × 6 inicializaciones × 608 fechas** mediante cinco proyecciones ortogonales con pesos empíricos uniformes. Se conserva la dependencia temporal al remuestrear fechas comunes por bloques e inicializaciones anidadas en cada red. Se publican magnitudes, proporciones e IC95 % marginales.

| Componente finito | Proporción observada | IC95 % temporal |
| --- | ---: | --- |
| Diferencias entre modelos del catálogo | 5,51 % | [3,92 %; 14,59 %] |
| Inicialización anidada en modelo | 0,25 % | [0,12 %; 0,76 %] |
| Calendario común | 35,99 % | [24,63 %; 43,12 %] |
| Modelo × calendario | 49,51 % | [38,98 %; 56,58 %] |
| Resto inicialización × calendario | 8,75 % | [5,34 %; 15,23 %] |

Las cinco magnitudes suman la varianza empírica total con error relativo máximo **1,10×10⁻¹⁵**. Las pérdidas tienen unidades σ₇² y su varianza σ₇⁴; las proporciones no tienen unidades. [Componentes e intervalos completos](../_static/e2_closure_followup_20261116/inference/recovery_v3/result_v1/finite_loss_variance_components_15.csv).

**Es una descomposición finita del catálogo vigente**, no un ajuste poblacional de efectos aleatorios ni variabilidad entre mercados, particiones nuevas o selecciones HPO. El componente restante incluye interacción con el calendario; no se denomina ruido independiente. Los IC son condicionales y aproximados bajo los supuestos del remuestreo por bloques.

## Diagnóstico específico de dependencia de entradas

El plan fija tres condiciones: entrada original, sustitución del volumen de siete días y sustitución conjunta de los cuatro canales OHLC. Los donantes se extraen exclusivamente de entrenamiento, con asignación prefijada, conservando cada camino de siete días dentro de su grupo. Se completaron **142.272 forwards: 78 predictores × 608 fechas × tres condiciones**, sin ajustar pesos o escaladores. Los originales CPU tienen discrepancia máxima **0** con sus referencias CPU admitidas.

Se publican **26 comparaciones** con 3.000 remuestreos por bloques de catorce días. Por ejemplo, sustituir volumen aumenta la RMSE de iTransformer en **0,0001773213**, IC95 % marginal **[0,0000247184; 0,0003260195]**. [Tabla completa](../_static/e2_closure_followup_20261116/shortcut/study_v1/statistics/input_dependency_by_model.csv) y {download}`resultados y límites <../_static/e2_closure_followup_20261116/shortcut/result_notes.md>`.

Este diagnóstico estudia un candidato señalado por la deriva del EDA. **Dependencia del volumen no demuestra señal espuria, fuga o causalidad.** Las sustituciones pueden romper el soporte conjunto entre grupos y la continuidad entre ventanas; los IC son condicionales a un único sorteo de donantes. No se ejecuta una ablación entrenada sin volumen ni se selecciona una receta nueva con prueba.

## Cobertura HPO y costes

El inventario conserva las **234 oportunidades, 231 resultados completos y trece ganadores**. Las seis plazas por red acreditan la alternativa de igualdad de número de configuraciones de §11.1; la igualdad de segundos no se impone como requisito adicional a esa alternativa. La contabilidad mantiene los **233 cargos observados, 25.833,898918 s**, separados de la asignación desconocida de 1.860 s y de los tiempos de fase contenidos en esos cargos.

La nueva [figura de puntuaciones individuales](../_static/e2_closure_followup_20261116/hpo/recorded_configuration_scores.svg), las [237 asociaciones descriptivas](../_static/e2_closure_followup_20261116/hpo/marginal_associations_descriptive.csv) y el [inventario de identificabilidad](../_static/e2_closure_followup_20261116/hpo/configuration_pair_identifiability_195.csv) muestran las limitaciones del diseño: sólo **uno de 195 pares** de configuraciones difiere en un único parámetro. **No se presentan como importancia independiente ni como variabilidad de selección estimada.**

El [dominio posterior propuesto](../_static/e2_closure_followup_20261116/hpo/followup_registration_not_executed.json) contiene dieciséis tuplas por red y un diseño provisional de repeticiones. Conserva explícitamente las dimensiones familiares sin adjudicar o explorar. Ese diseño no está ejecutado, no completa por sí solo §11.2 y no activa NAS.

El piloto temporal separado usa **C03 del dominio original**, elegida por posición antes de los ajustes, primer fold train253/validación83, semilla 271211, máximo 200 épocas y paciencia 20. MLP16 completó 200 épocas e iTransformer 24; **TimesNet quedó censurado por su límite temporal después de 39 épocas y 624 pasos**, sin una puntuación HPO completa. Se conservan la primera admisión bloqueada por el hash de un ejecutor ajeno y la recuperación que restringe las dependencias verificadas, sin repetir los dos ajustes exitosos. El [resumen del piloto](../_static/e2_closure_followup_20261116/hpo/pilot_v3/pilot_summary.json) cobra **291,208123 s** por todos los intentos y admisiones, dentro del límite de 450 s. Es una medición de tres representantes, no una estimación válida del coste completo de las trece familias ni evidencia de cobertura o importancia ampliadas.

## Bibliografía y reproducción

La comprobación local distingue los archivos ya recibidos de las consultas posteriores. La búsqueda acotada en Drive tampoco localizó las exportaciones originales faltantes. **Continúan sin recuperarse los 374 registros de la consulta original Scopus de 7.874 ni la correspondencia completa entre consultas, totales y lotes WoS.** Una búsqueda nueva no reconstruye ese historial. El [formulario de recuperación](../_static/e2_closure_followup_20261116/literature/native_export_recovery_form.json) precisa los datos necesarios; los desconocidos permanecen vacíos. La cola de 174 informes sigue identificada, sin inventar su lectura o elegibilidad.

El enlace de §8.4 se actualiza al Apéndice B vigente y los cortes anteriores quedan identificados como históricos. Los nuevos reanálisis se prueban fuera del repositorio, con entradas y código incluidos. La ejecución desde pronósticos guardados se distingue de inferencia sobre pesos y de entrenamiento desde cero.

El nuevo paquete CPU incluye **79 checkpoints binarios y 81 predictores**, fuentes fijadas, dependencias y entradas diarias/matrices. Se ejecutaron fuera del repositorio la reconstrucción de A2 y la inferencia real de las 81 trayectorias: **discrepancia máxima de pronósticos y A2 igual a cero** con su propia referencia CPU. Se hicieron además tres ajustes reales de comprobación del software: MLP16, las tres fases de SAE y ARIMA, con train655/cal53, sin evaluar prueba o producir una nueva A2 completa. Los fallos previos de redondeo aritmético y conversión de un resultado clásico se conservan con sus cargos. [Resumen y alcances verificados](../_static/e2_closure_followup_20261116/reproduction/summary.json).

La entrada de ese paquete es el derivado diario y las matrices auditadas. Su ejecución no vuelve a descargar ni decodificar los veinte ZIP de minutos. La prueba usa el intérprete CPU identificado en una carpeta aislada, sin afirmar que se instaló otro entorno desde cero o paridad A100. El modo `train-fixed` está separado y las pruebas específicas no equivalen a otra campaña completa de 78 ajustes. La reproducción integral de HPO, robustez y toda la campaña histórica conserva sus condiciones pendientes; **B21 no se completa por agregar un ejecutor**.

## Descargas independientes y estado verificable

Extraer todo el ZIP, instalar las dependencias indicadas y seguir su archivo de instrucciones. Cada archivo incluye sus propias entradas; los cinco alcances se pueden ejecutar sin clonar el repositorio.

| Archivo | Alcance comprobado fuera del repositorio |
| --- | --- |
| [Reanálisis de controles negativos](../_static/e2_closure_followup_20261116/audit/reanalisis_controles_negativos_v1.zip) | NumPy/SciPy; 80 valores de las seis funciones, sus IC y TOST/Holm, discrepancia cero. No ajusta modelos. |
| [OOF–prueba y variabilidad finita](../_static/e2_closure_followup_20261116/inference/reanalisis_inferencial_v1.zip) | NumPy; dos tablas completas con B=5.000 y tres longitudes de bloque, reproducidas exactamente. |
| [Dependencia de entradas](../_static/e2_closure_followup_20261116/shortcut/diagnostico_entradas_reanalisis_v1.zip) | NumPy; 26 comparaciones y 208 cifras reconstruidas, discrepancia cero. No repite los forwards. |
| [HPO descriptivo y bibliografía](../_static/e2_closure_followup_20261116/hpo/reanalisis_hpo_descriptivo_v1.zip) | Python estándar; 702 cifras de asociaciones y la contabilidad observada, discrepancia cero. Incluye formulario bibliográfico; no lanza HPO. |
| [Flujo CPU con pesos y fuentes](../_static/e2_closure_followup_20261116/reproduction/reproduccion_cpu_portable_v1.zip) | PyTorch y dependencias fijadas; verificación, reanálisis, inferencia real y entrenamiento de recetas fijas como modos separados. Leer `README.md`. |

El [estado consolidado](../_static/e2_closure_followup_20261116/delivery/closure_state.json), el [Apéndice B CSV](../_static/e2_closure_followup_20261116/delivery/appendix_B_current.csv), su [JSON](../_static/e2_closure_followup_20261116/delivery/appendix_B_current.json) y el [cierre presupuestario](../_static/e2_closure_followup_20261116/delivery/budget_final.json) conservan los alcances y condiciones parciales. La cuenta continúa la cadena previa; se cobran los nuevos cálculos y fallos sin reiniciar las 84 horas. Los archivos anteriores y A2/RQ1 permanecen intactos.
