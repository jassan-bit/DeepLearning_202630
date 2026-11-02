# Decisiones, evidencia y cierre documental de E1

Esta página registra el alcance autorizado: **definición del problema, EDA y revisión documental**. El entrenamiento, tuning, piloto, benchmark y pruebas de modelos propios se reservan para E2 por instrucción del usuario. El PDF adjunto sí ubica ejecución en E1; esta organización de trabajo no modifica la guía ni acredita una excepción formal del docente.

## Decisiones incorporadas

| Elemento | Criterio documental vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT, archivos oficiales de un minuto |
| Modelado | Diario UTC, cinco covariables OHLCV |
| Categoría literal de la guía | Regresión temporal, ruta 6.4; no hay rezagos explícitos de volatilidad como entrada |
| Historia de entrada | D=7/14/21/28 días; no se eligió una D principal |
| Objetivo principal | Desviación poblacional de n=7 retornos logarítmicos diarios; sin anualizar |
| Horizonte | Un día; próximo valor de volatilidad histórica móvil, no siete días enteramente futuros |
| Sensibilidad del objetivo | n=14/21/28, manteniendo separadas las definiciones y sus escalas |
| Pérdida prevista y métrica | MSE y RMSE principal; MAE/MSE/MAPE/R²/MASE complementarias |
| Relevancia operativa | Reducción relativa mínima del 5 % de RMSE frente a persistencia de las mismas fechas |
| Política de MAPE | Solo objetivos positivos; informar cobertura, ceros excluidos y casos no definidos; sin epsilon arbitrario |
| Política de MASE | Rezago de un día de calendario; escala calculada solo con entrenamiento; no definida si no hay pares válidos o el denominador es cero |
| Independencia | Dependencia temporal y ventanas solapadas; no tratar filas o semillas como nuevos mercados independientes |
| Longitud de bloques | Debe justificarse antes de la inferencia de E2; no se inventa con resultados ajenos al E1 |
| Faltantes | Exclusión estricta del día incompleto en los derivados actuales, sin interpolar ni comprimir el calendario |
| Calendario | Historia de 2023, validación de 2024 y test previsto del 01/01/2025 al 31/08/2026; septiembre–octubre no se asignan automáticamente |
| Familias documentales | LSTM unidireccional, TCN y encoder Transformer, condicionadas al corpus y sus límites |

El margen del 5 % es un criterio operativo aceptado, no una utilidad financiera calibrada. La conversión a escala absoluta para dimensionamiento se hará en la etapa experimental mediante una regla registrada, sin cambiar retrospectivamente el porcentaje. El [registro de decisiones](../_static/e1/decisiones.json) separa lo fijado de lo que requiere E2.

## Estado de los siete cambios solicitados

| Prioridad | Producto | Estado verificable |
|---|---|---|
| 1 | Búsqueda trazable | Consultas complementarias registradas; fuentes académicas bloqueadas diferenciadas de cero resultados |
| 2 | Cribado y flujo | 118 ocurrencias, 12 duplicadas, 106 repositorios, 14 inspeccionados, 5 textos completos y 3 incluidos; unidades diferenciadas |
| 3 | Matriz de literatura | Tres textos incluidos, 42 valores publicados y evaluación de calidad con ubicaciones |
| 4 | Selección de tres familias | LSTM–TCN–encoder Transformer; diversidad antes de frecuencia, con compromiso y fuente gris de atención explícitos |
| 5 | Fichas técnicas | Ecuaciones, diagramas, parámetros analíticos, configuraciones fuente, resultados publicados, fortalezas y límites |
| 6 | Comparación crítica | Objetivos/mercados/escalas comparados, riesgos y brecha limitada al corpus |
| 7 | Definición y coherencia | n=7, h=1, cuatro longitudes de entrada, RMSE, 5 %, MAPE/MASE, ruta 6.4 y procedencia sincronizados |

Las frecuencias se calculan sobre tres textos completos: familia LSTM 3/3 incluyendo BiLSTM, variante unidireccional 2/3, GRU 2/3, TCN 1/3 y encoder Transformer 1/3. La selección no se presenta como los tres modelos más frecuentes de toda la literatura ni como mínima complejidad absoluta. MLP independiente y CNN1D simple no tienen respaldo comparativo suficiente en este corpus.

## Límites que permanecen abiertos

- **Cobertura académica:** no se ejecutaron consultas bibliográficas en WoS/Scopus/IEEE/arXiv cuando su interfaz quedó bloqueada. La revisión complementaria no es exhaustiva.
- **Fuente de atención:** el único apoyo aplicado recuperado es un manuscrito sin revisión por pares acreditada. Si el criterio académico exige solo artículos revisados por pares, esa parte de la selección debe ampliarse.
- **Configuraciones originales:** no se recuperaron todos los hiperparámetros ganadores; las fichas separan parámetros derivados de configuraciones publicadas y de adaptaciones futuras.
- **Datos y ejecución:** quedan para E2 las muestras definitivas de entrenamiento/test, configuración implementada, recursos, número de semillas, piloto y benchmark. No se presentan como resultados inexistentes de E1.

## Proyección a E2

E2 deberá fijar el procedimiento experimental y el esfuerzo comparable entre modelos, ajustar transformaciones con pasado, registrar semillas/fallos y proteger test. Las longitudes de bloque, potencia y tamaños definitivos requerirán justificación propia. El criterio y las decisiones de E1 se heredan mediante versiones; cambios posteriores se declaran como enmiendas.

El plan documentado en noviembre de 2026 es posterior a los periodos de mercado considerados: su evaluación futura será un backtest con reserva, no un pronóstico emitido prospectivamente en 2025. Reutilizar el mismo test para diseñar nuevos modelos en entregables posteriores no constituye confirmación independiente.

## Evidencia y reproducción

La evidencia documental está en `references/literature_e1/`, el registro bibliográfico inicial en `references/bibliography/`, el EDA en `results/eda_development/` y el complemento descriptivo en `results/eda_covariates_documental/`. No se incorporan entrenamientos, predicciones ni métricas propias del trabajo separado fuera de E1. Construir HTML comprueba presentación y enlaces, no superioridad estadística ni cumplimiento académico total.
