# Decisiones, revisión y pendientes de E1

El **04/11/2026**, el usuario autorizó correr los tres modelos e incorporar sus resultados al libro del Entregable 1. Esa instrucción amplía el alcance anterior de documentación y EDA: ya se ejecutaron **MLP, LSTM y CNN1D poco profunda**. Se distinguen **V1**, con búsqueda/piloto/comparación estática, y **V2**, añadido para corregir el escalado y ejecutar entrenamiento real con validación forward chaining de Filipe. El test previsto de 2025–2026 continúa reservado; el ciclo DL registró **cero archivos de test abiertos**, aparte de los controles previos de timestamps/esquema/conteos durante la adquisición. No se declara cerrado el cumplimiento global de la guía.

## Decisiones incorporadas

| Elemento | Criterio o evidencia vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT, archivos oficiales de un minuto |
| Modelado | Diario UTC, cinco covariables OHLCV |
| Categoría literal de la guía | Tipo 4: regresión temporal, ruta 6.4; sin rezagos explícitos de volatilidad como entrada |
| Historia de entrada | Se compararon D=7/14/21/28; la regla global registrada seleccionó **D=7 común** para el primer ciclo |
| Objetivo principal | Desviación poblacional de n=7 retornos logarítmicos diarios; sin anualizar |
| Horizonte | Un día; próximo valor de volatilidad histórica móvil, no siete días enteramente futuros |
| Sensibilidad del objetivo | n=14/21/28 conservados como sensibilidad; no entrenados en este ciclo |
| Pérdida y métrica | MSE y RMSE principal; MAE/MSE/MAPE/R²/MASE complementarias |
| Relevancia operativa | Reducción relativa mínima del 5 % de RMSE frente a persistencia de las mismas fechas; V1 no la alcanzó. V2 se reporta por periodo sin convertir una mejora puntual en confirmación |
| Política de MAPE | Solo objetivos positivos; cobertura y ceros excluidos; sin epsilon arbitrario |
| Política de MASE | Rezago de un día de calendario; denominador del entrenamiento permitido, sin comprimir huecos; no definido si no hay escala positiva |
| Independencia | Una serie BTCUSDT; semillas condicionadas al calendario, fechas dependientes y ventanas solapadas; no son nuevos mercados |
| Longitud de bloques | V1: bootstrap circular L=14, sensibilidad 7/28, 2.000 réplicas y semilla 20261104. V2: 12 bloques mensuales de validación con ajuste real; no son bloques bootstrap ni repeticiones independientes |
| Faltantes | Exclusión estricta del 24/03/2023 y ventanas afectadas, sin interpolar ni comprimir calendario |
| Embargo | D días omitidos antes del origen t; candidatos de ajuste u≤t−D−1, además de calidad y disponibilidad |
| Calendario ejecutado | Entrenamiento inicial 2023; forward chaining 2024. V1 conserva su ajuste estático y su selección enero–junio/piloto julio–diciembre; V2 reajusta por mes con el pasado admisible, sin usar el mes evaluado para entrenarlo |
| Ajuste final y calendario reservado | Ajuste final previsto con desarrollo 2023–2024 una vez cerradas las configuraciones; todavía no ejecutado. Test enero de 2025–agosto de 2026; septiembre–octubre de 2026 permanece sin asignación automática |
| Reserva bruta de test | 20 ZIP mensuales previstos del 01/01/2025 al 31/08/2026, identificados por metadatos y hashes; muestras supervisadas y protocolo final pendientes |
| Referencia diagnóstica del bloque C | Cuatro Ridge con intercepto libre y α=1 fijo; 366 predicciones de 2024 por D, sin tuning; evidencia histórica separada del ciclo DL |
| Familias implementadas | MLP de ancho 16 y 593 parámetros; LSTM unidireccional de ancho 32 y 5.025 parámetros; CNN1D causal de dos capas, ancho 16 y 1.057 parámetros |
| Selección operacional | La búsqueda eligió **MLP**; su identidad no se reemplazó tras observar el piloto o la comparación posterior |
| Resultado descriptivo de V1 | LSTM tiene menor RMSE entre las redes en julio–diciembre; persistencia supera a las tres en ese ciclo. Las cifras V2 se presentan separadas por periodo en §5 |
| CV de Filipe ejecutable | `timeseries-cv` 0.1.5: `split_train_val_forwardChaining(np.arange(731),7,1,1)`; orígenes diarios agrupados en 12 bloques mensuales, con los índices efectivos de train/inner/validación suministrados a las redes |
| Comparación V2 | Escalado absoluto frente a relativo al último cierre de cada ventana; mismos D7/H/LR/modelos y diez semillas 4101–4110 emparejadas entre variantes, sin añadir rezagos de volatilidad |
| Perfiles de fuentes conservados | LSTM, TCN y encoder Transformer son perfiles documentales; no sustituyen las implementaciones propias ni el respaldo aplicado pendiente de MLP/CNN1D simple |

El margen del 5 % es un criterio operativo frente a persistencia, no utilidad financiera calibrada ni margen entre redes, de equivalencia o para modelos avanzados. El piloto **V1** lo convirtió a **0,000241124 unidades de RMSE** para planificar la detección de una diferencia frente a cero. Ese cálculo no demuestra una mejora superior al 5 %. El [registro de decisiones](../_static/e1/decisiones.json) conserva la cronología y separa los análisis históricos del ciclo actual.

## Revisión integral contra la guía

La revisión del **03/11/2026** contrastó 18 requisitos de las secciones 1–7 con la evidencia entonces disponible. Su {download}`informe <../_static/e1/revision_integral_e1_20261103.md>` y [registro JSON](../_static/e1/revision_integral_e1_20261103.json) se conservan como antecedentes fechados; las afirmaciones de cero ajustes DL de esos cortes no describen la ejecución posterior.

| Punto revisado | Evidencia actual o límite |
|---|---|
| Definición y categoría | Tipo 4, entradas OHLCV, salida continua, MSE y RMSE explícitos |
| Ventanas y temporalidad | D de entrada, n del objetivo y h de pronóstico diferenciados; primer ciclo D7/n7/h1, evaluación histórica |
| Independencia y margen | Una serie; inferencia por semillas y por fechas diferenciada; 5 % limitado a persistencia |
| Particiones | Manifiesto de preparación versión 2 conservado; V1 estático. V2 entrena en 12 bloques mensuales reales de Filipe, con embargo de 7 días y validación interna causal; no son 366 reajustes |
| Datos y trazabilidad | CSV de desarrollo y hashes conservados; reserva bruta de 20 ZIP de test intacta |
| EDA | Perfiles, calidad, etiquetas, desplazamiento marginal y bloque C con 1.464 residuos Ridge; mecanismo de faltantes y ruido externo no identificados |
| Literatura | 35 referencias, cinco textos aplicados exploratorios y 81 cifras de sus autores; respaldo de MLP/CNN1D simple todavía 0/5 en el corpus |
| Implementación | V1: **201 corridas y 402 ajustes**. V2 completó **60 trayectorias, 720 modelos mensuales y 1.440 ajustes efectivos**, con 21.960 predicciones; ejecución y conteos en su [resumen](../_static/dl_r2_cv_20261104/execution_summary.json) |
| Comparación e inferencia | V1: 30 semillas y nueve contrastes/Holm. V2: diez semillas y tres contrastes pareados de RMSE absoluto frente a relativo/Holm; resumen anual separado de la comparación descriptiva común julio–diciembre |
| Potencia | V1: piloto requiere máximo 1.212, ejecutado límite 30. V2: diez semillas exploratorias sin nuevo cálculo de potencia; la dispersión V1 no se transfiere automáticamente |

El cumplimiento global del Entregable 1 **no se declara cerrado**. Ejecutar las redes no completa el respaldo bibliográfico, la potencia requerida, la evaluación reservada ni el margen para modelos avanzados.

Los módulos previos conservan sus evidencias: [EDA general](../_static/e1/revision_eda_general_20261103.json), [EDA específico](../_static/eda_temporal_specific/summary.json), [referencia Ridge](../_static/eda_temporal_reference/summary.json) y [partición temporal](../_static/cv_forward_chaining/summary.json). Los 12 diagnósticos Ljung–Box/Holm del módulo Ridge son exploratorios y no se mezclan con la familia diagnóstica de las redes. La dependencia residual no añade automáticamente rezagos de volatilidad ni cambia el tipo 4.

El ciclo **V1** está documentado mediante [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json), [selección](../_static/dl_benchmark_20261104/selected_configuration.json), [piloto](../_static/dl_benchmark_20261104/pilot_summary.json), [plan de comparación posterior al piloto](../_static/dl_benchmark_20261104/protocol_final.json) y [resumen de ejecución](../_static/dl_benchmark_20261104/execution_summary.json). El registro precedió al primer ajuste DL, pero fue posterior al EDA y al periodo histórico; no se presenta como registro previo a observar todos los datos.

La CV **V2** tiene su [protocolo propio](../_static/dl_r2_cv_20261104/protocol.json), [índices efectivos](../_static/dl_r2_cv_20261104/folds.json), [cortes mensuales](../_static/dl_r2_cv_20261104/folds.csv), [corridas](../_static/dl_r2_cv_20261104/runs.csv) y [resumen de ejecución](../_static/dl_r2_cv_20261104/execution_summary.json). El registro de código previo a sus ajustes corresponde a `1f0ed23`. La [fe de erratas](../_static/dl_r2_cv_20261104/protocol_erratum.json) corrige una frase descriptiva heredada sobre “full2023”; los índices efectivos ya incluían correctamente el pasado mensual de 2024. No cambia pesos, datos ni optimización.

La representación relativa utiliza `log(OHLC)−log(último cierre observado de la propia ventana)`, conserva volumen con `log1p`, escalas aprendidas solo con train y salida en unidades originales. Así se estudia el fallo de extrapolación del nivel de precios detectado con los R² negativos. El contraste controlado compara ambas representaciones bajo la misma CV. La comparación V1 frente a V2 cambia también actualización, corte interno y semillas, por lo que no identifica por sí sola el efecto exclusivo del escalado.

Los borradores [de diseño](../_static/experiments/protocolo_e2_documental_20261103.json) y [de benchmark](../_static/experiments/benchmark_documental_20261103.json), y los informes del 03/11 de [implementación](../_static/e1/revision_implementacion_evaluacion_20261103.json), [benchmark](../_static/e1/revision_benchmark_baseline_20261103.json), [procedimiento](../_static/e1/revision_procedimiento_20261103.json) y [limitaciones](../_static/e1/revision_limitaciones_entregables_20261103.json), permanecen como antecedentes. Sus plantillas vacías no sustituyen las métricas [V1](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv) o [V2](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv) ni se rellenan retrospectivamente como si hubieran sido mediciones del 03/11.

## Estado de los siete cambios solicitados

| Prioridad | Producto | Estado verificable |
|---|---|---|
| 1 | Fuentes y búsqueda trazable | Tabla de las 35 referencias, enlaces y nivel de evidencia; consultas públicas e historial de acceso conservados |
| 2 | Cribado y flujo | 118 ocurrencias, 12 duplicadas, 106 repositorios, 20 inspeccionados, siete textos evaluados y cinco incluidos |
| 3 | Matriz de literatura | Cinco textos, 81 cifras publicadas y evaluación de calidad con ubicaciones; resultados de autores, no propios |
| 4 | Tres modelos básicos | MLP–LSTM–CNN1D implementados con configuraciones de desarrollo; la extracción aplicada MLP/CNN1D simple sigue incompleta |
| 5 | Fichas técnicas | Perfiles fuente LSTM/TCN/encoder conservados; implementaciones propias en §4; faltan apoyos aplicados independientes para MLP/CNN1D simple |
| 6 | Comparación crítica | Diferencias de objetivo/mercado/escala en literatura y resultados propios de desarrollo con límites separados |
| 7 | Definición y coherencia | n7/h1, D7 seleccionada bajo regla común, RMSE, 5 % frente a persistencia, MAPE/MASE, ruta 6.4 y procedencia sincronizados |

Las frecuencias sobre cinco textos completos se conservan: familia LSTM 4/5 incluyendo BiLSTM, unidireccional 3/5, GRU 2/5, TCN 1/5 y encoder Transformer 2/5. MLP independiente y CNN1D simple no tienen apoyo acreditado en ese corpus. La elección curricular y su implementación no se presentan como los tres modelos más frecuentes de toda la literatura ni como complejidad mínima absoluta.

## Límites que permanecen abiertos

- **Cobertura académica:** las doce consultas públicas ejecutadas y la verificación posterior no acreditan búsquedas internas en WoS/Scopus, indexación individual completa ni cuartil Q1. La recuperación es acotada y falta respaldar MLP/CNN1D simple.
- **Perfiles suplementarios de atención:** dos apoyos son manuscritos sin revisión por pares acreditada y uno carece de fecha editorial corroborada. Son límites de esos antecedentes, separados del trío implementado.
- **Extracción de los tres básicos:** la guía pide recurrencia, sencillez y diversidad; el corpus actual no cierra esos criterios conjuntamente para el trío curricular.
- **EDA y referencia contextual:** el diagnóstico Ridge está ejecutado; no se ha incorporado como otro competidor bajo el protocolo DL. El mecanismo de faltantes y el ruido externo de etiquetas siguen sin identificarse.
- **Configuraciones de fuentes:** permanecen límites documentales de `keras-tcn` y S6, sin atribuirlos a las implementaciones PyTorch propias. Los ganadores de S1 sí están recuperados.
- **Potencia:** en V1, el máximo de 30 semillas es menor que los 1.212/172/724 del piloto. V2 usa diez semillas por variante/modelo sin nuevo dimensionamiento; sus folds no sustituyen repeticiones independientes. Ningún ciclo acredita potencia 0,80 para una mejora pequeña del 5 %.
- **Generalización:** 2024 fue expuesto al EDA y a la selección de V1; enero–junio intervino en H/LR/D y julio–diciembre en piloto/comparación. La CV V2 respeta el tiempo dentro de cada ajuste, pero no es validación anidada de esa selección previa ni confirmación independiente. Se comparan los mismos 184 días con V1 solo descriptivamente: V1 tiene 30 semillas y V2 diez distintas. El ajuste final con desarrollo y el test supervisado 2025–agosto de 2026 permanecen pendientes.
- **Baseline y margen avanzado:** MLP conserva la identidad operacional de búsqueda de V1; el ranking posterior de ese ciclo y los resultados de V2 quedan identificados por fase y periodo. No se reemplaza retrospectivamente la referencia, ni se ha fijado un margen de equivalencia o de mejora para modelos avanzados.

## Proyección a E2

Las futuras etapas heredan código, configuraciones, resultados y limitaciones mediante versiones. Cualquier cambio de arquitectura, entradas, esfuerzo de búsqueda o cadencia requerirá una enmienda identificable; el resultado desfavorable frente a persistencia no justifica ocultar este ciclo o elegir retrospectivamente otra referencia.

Antes de consultar la reserva se deben cerrar sus muestras supervisadas, reglas de actualización, análisis y comparaciones. Las guías completas de E2/E3 no se han aportado; no se inventan requisitos adicionales. El diseño de noviembre de 2026 es posterior a los periodos de mercado: una evaluación de 2025–2026 será un backtest con reserva, no un pronóstico emitido prospectivamente. Si se usa ese test para diseñar un modelo posterior, hará falta otra evaluación independiente para confirmarlo.

## Evidencia y reproducción

Los scripts `scripts/dl_data.py`, `scripts/dl_models.py`, `scripts/train_dl_benchmark.py` y `scripts/analyze_dl_benchmark.py` reproducen **V1** bajo el entorno fijado en `requirements-training.lock.txt`. Los resultados originales están en `results/dl_benchmark_20261104/`: protocolos fechados, runtime, hashes, registros por corrida, configuraciones, escaladores, curvas y predicciones; se conservaron checkpoints de piloto y comparación. Los [registros agregados](../_static/dl_benchmark_20261104/runs.csv) y el [runtime observado](../_static/dl_benchmark_20261104/runtime.json) permiten identificar las ejecuciones, sin contar como réplicas independientes los dos ajustes internos de cada corrida.

El análisis **V1** separa IC t por semillas sobre calendario fijo, nueve contrastes pareados/Holm, seis DM principales/Holm y 15 diagnósticos/Holm de la semilla 1001. Su bootstrap remuestrea fechas, sin reentrenar. Esos contrastes, residuos y curvas no se reutilizan como evidencia calculada para V2.

Para **V2**, `scripts/prepare_dl_r2_cv.py` se ejecuta en el entorno separado de `timeseries-cv` 0.1.5/NumPy 1.26; `scripts/train_dl_r2_cv.py` consume los índices para ajustar redes en PyTorch. `scripts/dl_data_v2.py` define la representación relativa y `scripts/analyze_dl_r2_v2.py` recalcula las métricas desde predicciones. Se publican [métricas anuales](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), [cohorte común julio–diciembre](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2.csv) y [tres comparaciones controladas](../_static/dl_r2_cv_20261104/analysis/comparisons.csv). RMSE y R² anuales se calculan agrupando las 366 predicciones de una trayectoria, sin promediar R² de folds unitarios ni confundir media de RMSE entre semillas con RMSE de un ensemble. MASE utiliza el denominador permitido de cada entrenamiento mensual.

Las instrucciones completas están en `BUILD.md`; construir HTML valida presentación y enlaces, sin volver a ajustar las redes ni certificar superioridad o cumplimiento académico total.
