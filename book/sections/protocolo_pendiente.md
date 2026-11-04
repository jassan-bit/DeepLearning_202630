# Decisiones, revisión y pendientes de E1

El **04/11/2026**, el usuario autorizó correr los tres modelos e incorporar sus resultados al libro del Entregable 1. Esa instrucción amplía el alcance anterior de documentación y EDA: ya se ejecutaron **MLP, LSTM y CNN1D poco profunda**, con búsqueda, piloto y comparación de desarrollo. El test previsto de 2025–2026 continúa reservado; el ciclo DL registró **cero archivos de test abiertos**, aparte de los controles previos de timestamps/esquema/conteos durante la adquisición. No se declara cerrado el cumplimiento global de la guía.

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
| Relevancia operativa | Reducción relativa mínima del 5 % de RMSE frente a persistencia de las mismas fechas; no se alcanzó en desarrollo |
| Política de MAPE | Solo objetivos positivos; cobertura y ceros excluidos; sin epsilon arbitrario |
| Política de MASE | Rezago de un día de calendario; denominador del entrenamiento permitido, sin comprimir huecos; no definido si no hay escala positiva |
| Independencia | Una serie BTCUSDT; semillas condicionadas al calendario, fechas dependientes y ventanas solapadas; no son nuevos mercados |
| Longitud de bloques | Bootstrap circular pareado L=14, sensibilidad 7/28; 2.000 réplicas, semilla 20261104; predicciones guardadas sin reentrenar ni remuestrear semillas |
| Faltantes | Exclusión estricta del 24/03/2023 y ventanas afectadas, sin interpolar ni comprimir calendario |
| Embargo | D días omitidos antes del origen t; candidatos de ajuste u≤t−D−1, además de calidad y disponibilidad |
| Calendario ejecutado | Ajuste estático con 2023 permitido; selección enero–junio de 2024; piloto y comparación con semillas nuevas en julio–diciembre de 2024 |
| Reserva bruta de test | 20 ZIP mensuales previstos del 01/01/2025 al 31/08/2026, identificados por metadatos y hashes; muestras supervisadas y protocolo final pendientes |
| Referencia diagnóstica del bloque C | Cuatro Ridge con intercepto libre y α=1 fijo; 366 predicciones de 2024 por D, sin tuning; evidencia histórica separada del ciclo DL |
| Familias implementadas | MLP de ancho 16 y 593 parámetros; LSTM unidireccional de ancho 32 y 5.025 parámetros; CNN1D causal de dos capas, ancho 16 y 1.057 parámetros |
| Selección operacional | La búsqueda eligió **MLP**; su identidad no se reemplazó tras observar el piloto o la comparación posterior |
| Resultado descriptivo posterior | LSTM tiene menor RMSE entre las redes en julio–diciembre; persistencia supera a las tres |
| Perfiles de fuentes conservados | LSTM, TCN y encoder Transformer son perfiles documentales; no sustituyen las implementaciones propias ni el respaldo aplicado pendiente de MLP/CNN1D simple |

El margen del 5 % es un criterio operativo frente a persistencia, no utilidad financiera calibrada ni margen entre redes, de equivalencia o para modelos avanzados. El piloto lo convirtió a **0,000241124 unidades de RMSE** para planificar la detección de una diferencia frente a cero. Ese cálculo no demuestra una mejora superior al 5 %. El [registro de decisiones](../_static/e1/decisiones.json) conserva la cronología y separa los análisis históricos del ciclo actual.

## Revisión integral contra la guía

La revisión del **03/11/2026** contrastó 18 requisitos de las secciones 1–7 con la evidencia entonces disponible. Su {download}`informe <../_static/e1/revision_integral_e1_20261103.md>` y [registro JSON](../_static/e1/revision_integral_e1_20261103.json) se conservan como antecedentes fechados; las afirmaciones de cero ajustes DL de esos cortes no describen la ejecución posterior.

| Punto revisado | Evidencia actual o límite |
|---|---|
| Definición y categoría | Tipo 4, entradas OHLCV, salida continua, MSE y RMSE explícitos |
| Ventanas y temporalidad | D de entrada, n del objetivo y h de pronóstico diferenciados; primer ciclo D7/n7/h1, evaluación histórica |
| Independencia y margen | Una serie; inferencia por semillas y por fechas diferenciada; 5 % limitado a persistencia |
| Particiones | Manifiesto v2 con embargo D; el ciclo DL usa cortes de 2023 y ajuste estático, sin 366 reentrenamientos |
| Datos y trazabilidad | CSV de desarrollo y hashes conservados; reserva bruta de 20 ZIP de test intacta |
| EDA | Perfiles, calidad, etiquetas, desplazamiento marginal y bloque C con 1.464 residuos Ridge; mecanismo de faltantes y ruido externo no identificados |
| Literatura | 35 referencias, cinco textos aplicados exploratorios y 81 cifras de sus autores; respaldo de MLP/CNN1D simple todavía 0/5 en el corpus |
| Implementación | Tres redes ejecutadas; 96 corridas de búsqueda, 15 de piloto y 90 de comparación: **201 corridas y 402 ajustes** |
| Comparación e inferencia | 30 semillas nuevas por red; métricas, costos, IC, contrastes y figuras de julio–diciembre de 2024; no es test independiente |
| Potencia | Piloto calculó R requerido máximo 1.212; se ejecutó el límite registrado de 30. La insuficiencia para el efecto pequeño se declara, no se presenta como potencia 0,80 |

El cumplimiento global del Entregable 1 **no se declara cerrado**. Ejecutar las redes no completa el respaldo bibliográfico, la potencia requerida, la evaluación reservada ni el margen para modelos avanzados.

Los módulos previos conservan sus evidencias: [EDA general](../_static/e1/revision_eda_general_20261103.json), [EDA específico](../_static/eda_temporal_specific/summary.json), [referencia Ridge](../_static/eda_temporal_reference/summary.json) y [partición temporal](../_static/cv_forward_chaining/summary.json). Los 12 diagnósticos Ljung–Box/Holm del módulo Ridge son exploratorios y no se mezclan con la familia diagnóstica de las redes. La dependencia residual no añade automáticamente rezagos de volatilidad ni cambia el tipo 4.

El ciclo actual está documentado mediante [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json), [selección](../_static/dl_benchmark_20261104/selected_configuration.json), [piloto](../_static/dl_benchmark_20261104/pilot_summary.json), [plan de comparación posterior al piloto](../_static/dl_benchmark_20261104/protocol_final.json) y [resumen de ejecución](../_static/dl_benchmark_20261104/execution_summary.json). El registro precedió al primer ajuste DL, pero fue posterior al EDA y al periodo histórico; no se presenta como registro previo a observar todos los datos.

Los borradores [de diseño](../_static/experiments/protocolo_e2_documental_20261103.json) y [de benchmark](../_static/experiments/benchmark_documental_20261103.json), y los informes del 03/11 de [implementación](../_static/e1/revision_implementacion_evaluacion_20261103.json), [benchmark](../_static/e1/revision_benchmark_baseline_20261103.json), [procedimiento](../_static/e1/revision_procedimiento_20261103.json) y [limitaciones](../_static/e1/revision_limitaciones_entregables_20261103.json), permanecen como antecedentes. Sus plantillas vacías no sustituyen las [métricas actuales](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv) ni se rellenan retrospectivamente como si hubieran sido mediciones del 03/11.

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
- **Potencia:** el máximo de 30 semillas por red es menor que los 1.212/172/724 estimados por el piloto. La MDE y el supuesto condicional se publican; no se afirma suficiencia para una mejora pequeña del 5 %.
- **Generalización:** julio–diciembre de 2024 fue expuesto al EDA y a la calibración del piloto. Semillas nuevas no lo convierten en test. El test supervisado 2025–2026 y su protocolo final permanecen pendientes.
- **Baseline y margen avanzado:** MLP conserva la identidad operacional de búsqueda; LSTM lidera descriptivamente entre redes después, y persistencia tiene menor error que todas. No se ha fijado un margen de equivalencia ni de mejora para modelos avanzados.

## Proyección a E2

Las futuras etapas heredan código, configuraciones, resultados y limitaciones mediante versiones. Cualquier cambio de arquitectura, entradas, esfuerzo de búsqueda o cadencia requerirá una enmienda identificable; el resultado desfavorable frente a persistencia no justifica ocultar este ciclo o elegir retrospectivamente otra referencia.

Antes de consultar la reserva se deben cerrar sus muestras supervisadas, reglas de actualización, análisis y comparaciones. Las guías completas de E2/E3 no se han aportado; no se inventan requisitos adicionales. El diseño de noviembre de 2026 es posterior a los periodos de mercado: una evaluación de 2025–2026 será un backtest con reserva, no un pronóstico emitido prospectivamente. Si se usa ese test para diseñar un modelo posterior, hará falta otra evaluación independiente para confirmarlo.

## Evidencia y reproducción

Los scripts `scripts/dl_data.py`, `scripts/dl_models.py`, `scripts/train_dl_benchmark.py` y `scripts/analyze_dl_benchmark.py` reproducen el ciclo bajo el entorno fijado en `requirements-training.lock.txt`. Los resultados originales están en `results/dl_benchmark_20261104/`: protocolos fechados, runtime, hashes, registros por corrida, configuraciones, escaladores, curvas y predicciones; se conservaron checkpoints de piloto y comparación. Los [registros agregados](../_static/dl_benchmark_20261104/runs.csv) y el [runtime observado](../_static/dl_benchmark_20261104/runtime.json) permiten identificar las ejecuciones, sin contar como réplicas independientes los dos ajustes internos de cada corrida.

El análisis separa IC t por semillas sobre calendario fijo, nueve contrastes pareados/Holm, seis DM principales/Holm y 15 diagnósticos/Holm de la semilla 1001. El bootstrap remuestrea fechas, no reentrena. Las curvas graficadas son registros reales de parada interna y no curvas de test. Las instrucciones completas están en `BUILD.md`; construir HTML valida presentación y enlaces, sin volver a ajustar las redes ni certificar superioridad o cumplimiento académico total.
