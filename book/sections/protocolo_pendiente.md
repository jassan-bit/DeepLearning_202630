# Decisiones, revisión y pendientes de E1

Esta página registra el alcance autorizado: **definición del problema, EDA y revisión documental**, incluido el desarrollo del bloque C con una referencia lineal regularizada autorizado por el usuario. Se ejecutaron cuatro referencias Ridge para diagnosticar residuos de desarrollo. El entrenamiento de las tres redes, tuning, piloto, benchmark y evaluación de test se reservan para E2. El PDF adjunto sí ubica ejecución en E1; esta organización de trabajo no modifica la guía ni acredita una excepción formal del docente.

## Decisiones incorporadas

| Elemento | Criterio o evidencia vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT, archivos oficiales de un minuto |
| Modelado | Diario UTC, cinco covariables OHLCV |
| Categoría literal de la guía | Tipo 4: regresión temporal, ruta 6.4; no hay rezagos explícitos de volatilidad como entrada |
| Historia de entrada | D=7/14/21/28 días; no se eligió una D principal |
| Objetivo principal | Desviación poblacional de n=7 retornos logarítmicos diarios; sin anualizar |
| Horizonte | Un día; próximo valor de volatilidad histórica móvil, no siete días enteramente futuros |
| Sensibilidad del objetivo | n=14/21/28, manteniendo separadas las definiciones y sus escalas |
| Pérdida prevista y métrica | MSE y RMSE principal; MAE/MSE/MAPE/R²/MASE complementarias |
| Relevancia operativa | Reducción relativa mínima del 5 % de RMSE frente a persistencia de las mismas fechas |
| Política de MAPE | Solo objetivos positivos; informar cobertura, ceros excluidos y casos no definidos; sin epsilon arbitrario |
| Política de MASE | Rezago de un día de calendario; escala calculada solo con entrenamiento; no definida si no hay pares válidos o el denominador es cero |
| Independencia | Una serie BTCUSDT, Nseries=1; dependencia temporal y ventanas solapadas; filas y semillas no son mercados independientes |
| Longitud de bloques | Debe justificarse antes de la inferencia de E2; no se inventa con resultados ajenos al E1 |
| Faltantes | Exclusión estricta del día incompleto en los derivados actuales, sin interpolar ni comprimir el calendario |
| Embargo aplicado en los índices | D días omitidos antes del origen t; candidatos de ajuste u≤t−D−1, además de calidad y disponibilidad |
| Calendario | Historia de 2023, validación de 2024 y test previsto del 01/01/2025 al 31/08/2026; septiembre–octubre no se asignan automáticamente |
| Reserva bruta de test | 20 ZIP mensuales identificados mediante metadatos y hashes; el test supervisado y su protocolo aún no están congelados |
| Referencia diagnóstica del bloque C | Cuatro Ridge con intercepto libre y $\alpha=1$ fijo, una por D; ajuste permitido de 2023, 366 predicciones de 2024 por referencia; sin tuning ni selección de D |
| Familias candidatas documentales | LSTM unidireccional, TCN y encoder Transformer; selección final por recurrencia y mínima complejidad todavía no acreditada |

El margen del 5 % es un criterio operativo aceptado frente a persistencia, no una utilidad financiera calibrada ni un margen ya fijado entre redes, para equivalencia o para modelos avanzados. Alcanzarlo puntualmente no equivale a demostrarlo con un intervalo de incertidumbre. La conversión a escala absoluta para dimensionamiento se hará en la etapa experimental mediante una regla registrada, sin cambiar retrospectivamente el porcentaje. El [registro de decisiones](../_static/e1/decisiones.json) separa lo fijado de lo que requiere E2.

## Revisión integral contra la guía

La revisión del **03/11/2026** contrasta 18 requisitos de las secciones 1–7 de la guía con evidencia del libro y sus archivos. La matriz completa está disponible como {download}`informe de revisión <../_static/e1/revision_integral_e1_20261103.md>` y [registro JSON](../_static/e1/revision_integral_e1_20261103.json).

| Punto revisado | Corrección o estado vigente |
|---|---|
| Definición y categoría | Tipo 4, entradas OHLCV, salida continua, pérdida MSE e hipótesis operativa explícitos |
| Ventanas y temporalidad | D de entrada, n del objetivo y h de pronóstico diferenciados; evaluación como backtest histórico |
| Independencia y margen | Una serie; dependencia entre fechas y semillas diferenciada; 5 % limitado a persistencia |
| Particiones | Folds corregidos con embargo explícito D; 5.856 filas, 366 por configuración; una muestra de ajuste adicional retirada en cada fila |
| Datos y trazabilidad | Fuente y hashes conservados; copia de términos y reserva bruta de 20 ZIP de test documentadas sin abrir esos ZIP |
| EDA | Ocho perfiles objetivo, colas, máscaras, reconstrucción numérica y seis marginales por bloques/Holm; bloque C con disponibilidad causal, correlaciones en cohorte común, relaciones móviles y 1.464 residuos de cuatro Ridge. Mecanismo de faltantes y ruido externo no identificados |
| Literatura | 35 referencias separadas de cinco textos aplicados exploratorios; candidatos de trabajo distinguidos de los tres básicos definitivos exigidos |
| Implementación de las tres redes y benchmark | No ejecutados por el alcance vigente; los cuatro ajustes lineales son diagnósticos de EDA y no cubren estos requisitos de la guía original |

El cumplimiento global del Entregable 1 **no se declara cerrado**. Corregir la redacción y los índices no sustituye las evidencias bibliográficas o experimentales que faltan.

La ampliación específica de la sección 2.1 está en el {download}`informe de EDA general <../_static/e1/revision_eda_general_20261103.md>` y su [matriz de 18 puntos](../_static/e1/revision_eda_general_20261103.json). Se ejecutaron diagnósticos sobre desarrollo, sin modelos predictivos ni archivos de test. La consistencia del cálculo de etiquetas se separa del ruido externo, y la referencia de permutación por bloques se mantiene exploratoria y condicional.

El bloque C aplicable se desarrolla en §2.2 con dos módulos adicionales: el [EDA específico](../_static/eda_temporal_specific/summary.json), que usa las mismas 308 fechas elegibles de 2023 para los 140 pares covariable–rezago, y la [referencia lineal diagnóstica](../_static/eda_temporal_reference/summary.json). Esta última conserva cuatro ajustes estáticos de 2023 y 366 residuos de 2024 por D. Los 12 diagnósticos Ljung–Box en rezagos 7/14/28 rechazan con ajuste Holm bajo su referencia asintótica; la dependencia del objetivo móvil y la limitación de la referencia impiden atribuir ese rechazo exclusivamente a una falta de rezagos del objetivo. Se mantiene el tipo 4 y no se usaron archivos de test. El ajuste estático y $\alpha=1$ describen este diagnóstico, sin congelar el experimento posterior.

La revisión del capítulo 4 se conserva en el {download}`informe de implementación y evaluación <../_static/e1/revision_implementacion_evaluacion_20261103.md>` y su [matriz](../_static/e1/revision_implementacion_evaluacion_20261103.json). La [especificación documental actual](../_static/experiments/protocolo_e2_documental_20261103.json) sincroniza decisiones aceptadas y campos abiertos sin sustituir el borrador histórico. Se distinguen detección frente a cero y superación del 5 %, corridas y ajustes, selección de 2024 y validación interna, escalado del subentrenamiento y máscara de MASE. No se calcularon R/MDE ni se ejecutaron redes, piloto, benchmark o test.

La revisión del capítulo 5 está en el {download}`informe de benchmark y baseline <../_static/e1/revision_benchmark_baseline_20261103.md>` y su [matriz](../_static/e1/revision_benchmark_baseline_20261103.json). El [contrato documental del benchmark](../_static/experiments/benchmark_documental_20261103.json) fija agregación descriptiva y roles de referencia, con inferencia final y resultados pendientes. Se distingue el baseline seleccionado en desarrollo del ranking descriptivo de test, y la media de RMSE entre semillas de la pérdida temporal examinada por DM.

## Estado de los siete cambios solicitados

| Prioridad | Producto | Estado verificable |
|---|---|---|
| 1 | Fuentes y búsqueda trazable | Tabla de las 35 referencias citadas, con enlaces y nivel de evidencia; consultas complementarias e historial de acceso conservados |
| 2 | Cribado y flujo | 118 ocurrencias, 12 duplicadas, 106 repositorios, 14 inspeccionados, 5 textos completos y 3 incluidos; unidades diferenciadas |
| 3 | Matriz de literatura | Tres textos incluidos, 42 valores publicados y evaluación de calidad con ubicaciones |
| 4 | Candidatos para tres modelos básicos | LSTM–TCN–encoder Transformer documentados; la selección definitiva por frecuencia y mínima complejidad permanece parcial |
| 5 | Fichas técnicas | Ecuaciones, diagramas, parámetros analíticos, configuraciones fuente, resultados publicados, fortalezas y límites |
| 6 | Comparación crítica | Objetivos/mercados/escalas comparados, riesgos y brecha limitada al corpus |
| 7 | Definición y coherencia | n=7, h=1, cuatro longitudes de entrada, RMSE, 5 %, MAPE/MASE, ruta 6.4 y procedencia sincronizados |

Las frecuencias se calculan sobre cinco textos completos: familia LSTM 4/5 incluyendo BiLSTM, variante unidireccional 3/5, GRU 2/5, TCN 1/5 y encoder Transformer 2/5. Se conserva el corte original y la sensibilidad sin el estudio sin fecha. La selección no se presenta como los tres modelos más frecuentes de toda la literatura ni como mínima complejidad absoluta. MLP independiente y CNN1D simple no tienen respaldo comparativo suficiente en este corpus.

## Límites que permanecen abiertos

- **Cobertura académica:** no se ejecutaron consultas bibliográficas en WoS/Scopus/IEEE/arXiv cuando su interfaz quedó bloqueada. La revisión complementaria no es exhaustiva.
- **Fuente de atención:** los dos apoyos aplicados recuperados son manuscritos sin revisión por pares acreditada; uno no acredita fecha editorial. Si el criterio académico exige solo artículos revisados por pares, esa parte de la selección debe ampliarse.
- **Extracción de los tres básicos:** la guía pide recurrencia, sencillez y diversidad; priorizar únicamente diversidad en un corpus pequeño no cierra conjuntamente esos criterios.
- **EDA y referencia de contexto:** el diagnóstico de cuatro Ridge y sus residuos está ejecutado; queda justificar la especificación y el tratamiento de la dependencia para el experimento de E2. La autocorrelación residual no reclasifica automáticamente el problema. No se han identificado el mecanismo de faltantes ni el ruido externo de etiquetas con evidencia suficiente.
- **Configuraciones originales:** los ganadores de S1 sí están publicados y se corrigió su omisión; quedan sin fijar la versión de `keras-tcn`, algunos detalles de S6 y las configuraciones OHLCV propias. Las fichas separan fuente, cálculo analítico y adaptación futura.
- **Datos y ejecución:** quedan para E2 las muestras definitivas de entrenamiento/test, configuración implementada, recursos, número de semillas, piloto y benchmark. No se presentan como resultados inexistentes de E1.

## Proyección a E2

E2 deberá fijar el procedimiento experimental y el esfuerzo comparable entre modelos, ajustar transformaciones con pasado, registrar semillas/fallos y proteger test. Las longitudes de bloque, potencia y tamaños definitivos requerirán justificación propia. El criterio y las decisiones de E1 se heredan mediante versiones; cambios posteriores se declaran como enmiendas.

El plan documentado en noviembre de 2026 es posterior a los periodos de mercado considerados: su evaluación futura será un backtest con reserva, no un pronóstico emitido prospectivamente en 2025. Reutilizar el mismo test para diseñar nuevos modelos en entregables posteriores no constituye confirmación independiente.

## Evidencia y reproducción

La evidencia documental está en `references/literature_e1/`, el registro bibliográfico inicial en `references/bibliography/`, el EDA en `results/eda_development/` y el complemento descriptivo en `results/eda_covariates_documental/`. Los módulos del bloque C se reproducen con `scripts/eda_temporal_specific.py` y `scripts/eda_temporal_reference.py`, con resultados en sus directorios homónimos bajo `results/`; conservan los CSV fuente, la exclusión del 24/03/2023 y el calendario. La [tabla consolidada de referencias](../_static/literature_e1/references_access_35.csv) distingue las 35 referencias de los cinco textos del corpus exploratorio. La revisión del 03/11/2026 inspeccionó seis repositorios adicionales, añadió dos textos grises y actualizó el flujo, las frecuencias y sus sensibilidades. El [resumen de partición vigente](../_static/cv_forward_chaining/summary.json) identifica el generador, la regla de embargo y los hashes. La [reserva bruta del test](../_static/data/test_raw_reserve_20250101_20260831.json) se reproduce con `scripts/reserve_test_raw_manifest.py`. El preregistro piloto anterior se conserva como borrador histórico, sin sustituir las decisiones documentales actuales ni un futuro protocolo congelado. Las predicciones lineales incorporadas corresponden exclusivamente al diagnóstico autorizado de EDA; no se incorporan resultados de redes del trabajo separado fuera de E1. Construir HTML comprueba presentación y enlaces, no superioridad estadística ni cumplimiento académico total.
