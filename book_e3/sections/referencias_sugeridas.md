# Referencias metodológicas sugeridas por la guía

Las 37 referencias siguientes proceden de las páginas 54–57 de *Entregable3DeepLearningProject.pdf*, aportado por el usuario. Se conserva su numeración original y se estructuran sus metadatos bibliográficos. Esta incorporación documental no registra nuevas búsquedas, lecturas de originales ni verificación externa de esos metadatos.

La aplicabilidad se refiere a la ruta temporal F2a del proyecto: pronóstico de σ₇, con RMSE primaria, dependencia temporal y variabilidad entre semillas. Cada procedimiento deberá corresponder al estimando, las unidades y los supuestos del protocolo que se ejecute; una referencia sugerida no acredita que su prueba ya se haya aplicado.

## Relación con la ruta temporal

| N.º de la guía | Tema y referencia | Aplicabilidad a F2a |
|---|---|---|
| 1 | Comparación de clasificadores entre datasets. {cite:p}`guia_demsar2006` | Orienta la comparación entre conjuntos. Las fechas correlacionadas de una serie y las semillas sobre una misma cronología no se convierten en datasets independientes para Friedman. |
| 2 | Comparaciones por pares posteriores al análisis global. {cite:p}`guia_garcia2008` | Ayuda a organizar familias y post-hoc entre conjuntos; requiere unidades y diseño compatibles, además del control temporal dentro de cada conjunto. |
| 3 | Aproximación de Iman–Davenport para Friedman. {cite:p}`guia_iman1980` | Pertinente para el análisis entre conjuntos u orígenes justificables como unidades del diseño. No resuelve por sí sola la dependencia de los errores diarios. |
| 4 | Comparación bayesiana y regiones de equivalencia práctica. {cite:p}`guia_benavoli2017` | Una ROPE requiere margen, escala y modelo probabilístico prefijados. Debe explicarse la adaptación de una referencia sobre clasificadores al pronóstico temporal. |
| 5 | Áreas ROC correlacionadas y prueba de DeLong. {cite:p}`guia_delong1988` | Referencia de clasificación. El objetivo continuo σ₇ y la RMSE no son un área ROC; su aplicación exigiría una tarea de clasificación definida por separado. |
| 6 | Inferencia del error de generalización y corrección de varianza. {cite:p}`guia_nadeau2003` | La corrección de Nadeau–Bengio corresponde a determinados diseños de particiones con entrenamientos solapados. No se aplica automáticamente a una única trayectoria de prueba ni reemplaza una estimación temporal de varianza. |
| 7 | Contrastes para comparar algoritmos de clasificación. {cite:p}`guia_dietterich1998` | Los diseños de particiones de clasificación requieren adaptación justificada. Aleatorizar fechas para imitar una validación 5×2 destruiría la disponibilidad causal del pronóstico. |
| 8 | Selección de pruebas de significancia en lenguaje. {cite:p}`guia_dror2018` | Aporta criterios de correspondencia entre métrica, prueba y unidad. En F2a, el pareado y el remuestreo deben respetar la dependencia temporal, en lugar de asumir ejemplos intercambiables. |
| 9 | Fuentes de varianza en benchmarks. {cite:p}`guia_bouthillier2021` | Separar variación por semilla, partición, datos y ajuste de hiperparámetros. Seis semillas sobre las mismas fechas no representan seis mercados independientes. |
| 10 | Reporte del proceso experimental. {cite:p}`guia_dodge2019` | Registrar intentos, presupuesto, paradas y fallos, además de la receta seleccionada en validación. Ayuda a documentar el esfuerzo de diseño y ajuste. |
| 11 | Claridad y límites de las afirmaciones científicas. {cite:p}`guia_lipton2019` | Distinguir el mecanismo, la evidencia que lo sustenta y el alcance de cada conclusión. Un nombre nuevo o una explicación intuitiva requiere respaldo adicional. |
| 12 | Fuga de información y reproducibilidad. {cite:p}`guia_kapoor2023` | Auditar disponibilidad temporal, transformaciones, selección, particiones y pipeline efectivo. Los hashes de archivos y las fechas guardadas sólo cubren parte de esa auditoría. |
| 13 | Calibración de redes de clasificación. {cite:p}`guia_guo2017` | ECE y escalado de temperatura se refieren a probabilidades de clase en su aplicación habitual. Para σ₇ continuo se necesitan diagnósticos e intervalos de regresión o pronóstico adecuados. |
| 14 | Comparación de árboles y redes en datos tabulares. {cite:p}`guia_grinsztajn2022` | Motiva comparadores fuertes y control del protocolo. Sus resultados tabulares no determinan el desenlace de una ventana temporal OHLCV ni una superioridad universal de una familia. |
| 15 | Transformers y baselines de series de tiempo. {cite:p}`guia_zeng2023` | Pertinente para justificar baselines y contrastar complejidad con beneficio observado. El resultado en otra tarea o longitud de contexto no se atribuye directamente a D7. |
| 16 | Robustez ante corrupción y perturbaciones. {cite:p}`guia_hendrycks2019` | El diseño por tipos y severidades puede orientar el estrés temporal. Las corrupciones de imágenes requieren adaptación; el estrés OHLCV debe explicitar su coherencia y su alcance sintético. |
| 17 | Diferencias de precisión predictiva: Diebold–Mariano. {cite:p}`guia_diebold1995` | Contrasta diferencias de pérdidas con supuestos sobre su dependencia. Un contraste de pérdida cuadrática debe identificarse como tal; no acredita automáticamente una mejora de RMSE mayor que δ. |
| 18 | Corrección de Harvey–Leybourne–Newbold. {cite:p}`guia_harvey1997` | Ajusta el contraste bajo su diseño y horizonte. La corrección de muestra finita no elimina supuestos de dependencia o estacionariedad ni garantiza potencia suficiente. |
| 19 | Capacidad predictiva condicional: Giacomini–White. {cite:p}`guia_giacomini2006` | Deben especificarse información condicionante, esquema de estimación y condiciones de validez. Su uso no se decide después de observar qué prueba produce el p más favorable. |
| 20 | Contraste de Clark–West para modelos anidados. {cite:p}`guia_clark2007` | Requiere anidamiento y un esquema compatible con el contraste. Comparar dos redes, o hacer una red más grande, no demuestra por sí solo el anidamiento necesario. |
| 21 | Model Confidence Set. {cite:p}`guia_hansen2011` | Ofrece un conjunto de modelos bajo una pérdida, familia y nivel declarados. Pertenecer al conjunto no demuestra equivalencia práctica ni identifica obligatoriamente un ganador único. |
| 22 | Bootstrap estacionario. {cite:p}`guia_politis1994` | Pertinente para remuestrear dependencia temporal bajo sus supuestos. Debe fijarse la política de longitud de bloques y conservar el pareado entre modelos. |
| 23 | Tamaños de efecto para datos ordinales. {cite:p}`guia_romano2006` | Ayuda a contextualizar efectos por rangos y los umbrales citados por la guía. Su origen en encuestas ordinales no convierte esos umbrales en relevancia económica validada para volatilidad. |
| 24 | Reglas de puntuación estrictamente propias. {cite:p}`guia_gneiting2007` | Orienta la evaluación coherente de pronósticos probabilísticos. Debe distinguirse el objetivo puntual, los intervalos y la distribución completa que realmente se estiman. |
| 25 | Predicción conforme. {cite:p}`guia_angelopoulos2023` | Sus garantías dependen del método y de supuestos como intercambiabilidad. Para series de tiempo se debe justificar la adaptación y medir cobertura temporal y condicional. |
| 26 | Explicaciones visuales con Grad-CAM. {cite:p}`guia_selvaraju2017` | Su aplicación original es visual. Un uso en convoluciones temporales necesita definir la adaptación, la agregación y las pruebas de fidelidad; no se aplica directamente por tener una CNN. |
| 27 | Integrated Gradients y atribuciones axiomáticas. {cite:p}`guia_sundararajan2017` | Puede aplicarse a una red diferenciable con una referencia de entrada justificada. Deben respetarse el preprocesamiento y las dependencias OHLCV, y comprobarse fidelidad y estabilidad. |
| 28 | SHAP y explicación de predicciones. {cite:p}`guia_lundberg2017` | Requiere documentar el fondo de referencia y el tratamiento de variables correlacionadas. Las atribuciones describen el modelo y no establecen causas económicas. |
| 29 | Energía y políticas de entrenamiento en lenguaje. {cite:p}`guia_strubell2019` | Motiva registrar costes reales y recursos. Sus cifras de NLP no se trasladan como mediciones de energía del proyecto ni se deducen sólo de latencia. |
| 30 | Reporte de energía y huella de carbono. {cite:p}`guia_henderson2020` | Registrar método, hardware, alcance y factores necesarios cuando se mida energía. CPU, GPU, memoria y consumo energético son magnitudes distintas. |
| 31 | Fichas de modelos. {cite:p}`guia_mitchell2019` | Aplicación directa a `MODEL_CARD.md`: propósito, dominio, evaluación, limitaciones y usos. Una ficha debe describir el artefacto y la evidencia disponibles. |
| 32 | Reproducibilidad de investigación. {cite:p}`guia_pineau2021` | Sustenta documentación del entorno, semillas, datos, código, artefactos y repetición. Reproducir una cronología conocida no crea una nueva evaluación independiente. |
| 33 | Optuna y optimización de hiperparámetros. {cite:p}`guia_akiba2019` | Puede instrumentar el ajuste en validación con presupuesto comparable, espacios registrados y fallos conservados. La herramienta no garantiza por sí sola una comparación justa. |
| 34 | Procedimiento de Holm y control FWER. {cite:p}`guia_holm1979` | Corresponde al control familiar exigido para los contrastes declarados. Se conserva la familia completa y se reportan p crudos y ajustados, sin elegir sólo la prueba favorable. |
| 35 | Benjamini–Hochberg y control FDR. {cite:p}`guia_benjamini1995` | Controla otra noción de error múltiple. Usar FDR en un análisis secundario no sustituye el control FWER con Holm requerido para la familia principal. |
| 36 | Equivalencia y dos pruebas unilaterales: TOST. {cite:p}`guia_schuirmann1987` | Una afirmación de equivalencia necesita margen prefijado y estimación adecuada a la dependencia del diseño. No rechazar una diferencia o una superioridad no demuestra equivalencia. |
| 37 | Fundamentos del bootstrap. {cite:p}`guia_efron1993` | Orienta estimación de incertidumbre y remuestreo. Para observaciones temporales se requiere un procedimiento que conserve su estructura, como una política de bloques justificada. |

## Estimando temporal y margen práctico

La primaria del proyecto es RMSE en unidades originales. Los contrastes de diferencias de pérdida cuadrática deben conservar esa identificación: un p-valor de igualdad en MSE no prueba una ganancia de RMSE superior al margen δ. La comparación frente a δ requiere definir la diferencia que se estima y su incertidumbre en la escala pertinente. Los contrastes condicionales, la corrección por horizonte y los métodos para modelos anidados tienen condiciones específicas que deben verificarse en el protocolo.

Las repeticiones por semillas describen variabilidad de optimización sobre una cronología; los métodos temporales evalúan otras fuentes de incertidumbre. La corrección de Nadeau–Bengio se justifica a partir del diseño de particiones usado. La unidad efectiva, el pareado y la dependencia también deben corresponder al bootstrap y al análisis entre conjuntos. El control FDR y el control FWER responden a errores diferentes y se identifican por separado.

## Calibración e interpretación del pronóstico

En σ₇ continuo, los diagnósticos de regresión, cobertura de intervalos y calidad probabilística deben corresponder a la salida producida. AUC, ROC y la calibración de probabilidades de clasificación sólo serían pertinentes para otra tarea definida expresamente. Las explicaciones visuales y de lenguaje requieren adaptación al modelo temporal, referencias de entrada coherentes y comprobaciones de fidelidad; sus nombres no acreditan una explicación causal.

## Bibliografía estructurada

{download}`Descargar las 37 referencias metodológicas sugeridas en BibTeX <../referencias_metodologicas_v1.bib>`.

Este archivo versionado conserva los campos aportados por la guía y usa claves con prefijo `guia_`. Los nombres abreviados y las listas con *et al.* se mantienen sin completar autores no suministrados; tampoco se añadieron DOI o URL no aportados. Las siete referencias del mecanismo candidato se conservan en su fichero independiente.
