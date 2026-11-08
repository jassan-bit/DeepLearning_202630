# 6. Benchmark vigente: diez modelos y un candidato adicional

**Estado al 08/11/2026: sin resultados de mercado del nuevo catálogo.** La presentación de E2 considera únicamente los diez modelos indicados abajo y HAR + red residual como candidato 11. La expresión «modelo superior» representa la mejora que se desea investigar; no es una conclusión respaldada por evidencia.

Hay cuatro prototipos compactos y un corrector con HAR externo, con diez pruebas sintéticas documentadas. Los otros seis modelos tienen código histórico pendiente de integración y validación nueva. No se han ejecutado el HPO, el dimensionamiento ni la evaluación de este conjunto. Las métricas de E1, P0 o ETTh1 y las de artículos externos no rellenan sus resultados.

## 6.1 Tabla de los once candidatos

| N.º | Modelo | Estado de implementación para este diseño | RMSE y coste de mercado |
|---|---|---|---|
| 1 | FCN-1D | Prototipo compacto aislado; receta pendiente | No medidos |
| 2 | ResNet-1D | Prototipo compacto aislado; receta pendiente | No medidos |
| 3 | Inception-1D | Prototipo compacto de una red; receta pendiente | No medidos |
| 4 | LSTM-autoencoder con predictor | Reconstrucción y predictor disponibles; entrenamiento por definir | No medidos |
| 5 | TimesNet | Código histórico; integración y validación pendientes | No medidos |
| 6 | TSMixer | Código histórico; integración y validación pendientes | No medidos |
| 7 | TiDE | Código histórico; integración y validación pendientes | No medidos |
| 8 | TimeMixer | Código histórico; integración y validación pendientes | No medidos |
| 9 | PatchTST | Código histórico; integración y validación pendientes | No medidos |
| 10 | iTransformer | Código histórico; integración y validación pendientes | No medidos |
| 11 | HAR + red residual | Corrector con HAR externo; falta pipeline HAR/OOF y resolver pérdida/enlace | No medidos; superioridad no demostrada |

**No medido no significa cero ni un empate.** Esta tabla es el catálogo y su estado, no un ranking. Los controles HAR solo, red sola y soporte comparable se reportarán dentro del estudio del candidato 11 cuando se definan; no reintroducen el catálogo descartado ni acreditan ensayos ya realizados.

## 6.2 Qué debe medir una comparación válida

La tarea neural conserva D7×5 OHLCV y predicción a un día de σ₇ en unidades originales. El componente HAR requiere más historia del objetivo, que debe identificarse explícitamente. Las fechas comunes, el calentamiento y las exclusiones se fijan antes de comparar; no se eliminan fechas a conveniencia según el modelo o el error observado.

La métrica primaria es la **media de las RMSE de cada trayectoria/semilla**, calculadas sobre el mismo calendario elegible:

$$
q_{m,s}=\sqrt{|\mathcal T|^{-1}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2},
\qquad \widehat\mu_m=R^{-1}\sum_s q_{m,s}.
$$

La expresión define la agregación, **no fija R**. No se reemplaza por la RMSE de predicciones promediadas, una raíz de errores mezclados entre semillas ni una media no ponderada de RMSE mensuales. Si una receta usa un ensemble, deben definirse de antemano sus miembros y qué constituye una réplica de esa receta.

Las métricas secundarias y diagnósticos se fijarán en el nuevo preregistro. Al informar MAE, MSE, R², MAPE o MASE, se declararán denominadores, cobertura y valores no estimables. MAPE no oculta objetivos cero; MASE usa únicamente pares temporales elegibles del entrenamiento, sin comprimir huecos. Si se evalúan varianza o QLIKE, debe registrarse la transformación de σ₇ y cualquier suelo numérico antes de puntuar.

Cada resultado debe acompañarse de receta, origen, semilla, número de fechas, escala, fallos y procedencia de los pesos. Repetir una predicción determinista para una resta pareada no crea entrenamientos independientes.

## 6.3 Cómo evaluar al candidato 11

HAR + red residual sólo podrá presentarse como mejor bajo una comparación definida y respaldada por datos. Antes de entrenar se necesitan:

1. **Objetivo y enlace:** especificar qué pronostica HAR, qué residuo aprende la red, cómo se invierte cualquier transformación y cómo se tratan valores negativos. El truncamiento del prototipo puede anular gradientes.
2. **Cross-fitting causal:** obtener residuos de predicciones realmente fuera de muestra dentro del entrenamiento autorizado. Ningún error de validación o test se reutiliza como objetivo del corrector en ese fold.
3. **Controles:** HAR solo con el mismo calendario y tratamiento de salida; red sola; comparadores con soporte de información equiparable, o una limitación explícita de la atribución arquitectónica.
4. **Equidad:** cargar estimación HAR, generación OOF, entrenamiento del corrector, selección y reajustes al presupuesto del híbrido. No conceder búsquedas extra hasta lograr ventaja.
5. **Decisión previa:** comparador, mejora mínima relevante, familia de contrastes, multiplicidad, incertidumbre y criterio de coste definidos antes de resultados.

La literatura aporta tanto motivación como resultados desfavorables de híbridos. Una menor varianza residual no garantiza mayor capacidad predictiva. Se aceptarán resultados nulos o peores; no se seleccionará retrospectivamente la métrica o cohorte que favorezca al candidato.

## 6.4 Incertidumbre, eficiencia y límites

**Calendario y semillas representan incertidumbres distintas.** Los remuestreos temporales deben usar bloques comunes de fechas y recalcular cada RMSE antes de agregar. La variación entre semillas describe el entrenamiento condicionado a un calendario; no equivale a observar más mercados. Longitudes de bloque, sensibilidades, número de remuestreos y familias requieren especificación propia, no copia automática del diseño retirado.

Los intervalos del score no son intervalos predictivos de σ₇. Una rama de calibración exige separación del ajuste y de la selección, un predictor congelado y reglas anteriores al ranking. Su diseño y ejecución para estos once candidatos están pendientes.

Los costes se medirán con hardware, precisión y protocolo comunes: búsqueda, preentrenamiento, ajuste final, preprocesamiento e inferencia separados. Parámetros, memoria, latencia y FLOPs/MACs deben indicar método y operaciones omitidas. Una magnitud no medida es ausente, no cero. Las fronteras de Pareto y la dominancia puntual tampoco demuestran superioridad con incertidumbre.

La comparación temporal puede requerir DM y Model Confidence Set conforme a la guía, con pérdidas y unidades apropiadas. DM sobre pérdidas cuadráticas no prueba por sí mismo una ganancia práctica de RMSE. El MCS no establece equivalencia. No se fabricarán datasets independientes usando semillas o meses solapados para pruebas globales entre datasets.

La reserva ya consultada en etapas anteriores conserva carácter **exploratorio**. Registrar nuevos ajustes no elimina esa exposición. No hay evidencia actual para afirmar generalización a otros activos o regímenes.

## 6.5 Condiciones de cierre

| Etapa del catálogo vigente | Evidencia necesaria | Estado |
|---|---|---|
| Recetas y fuentes | Arquitecturas, adaptaciones, objetivos y controles del híbrido identificados | Pendiente de congelación |
| Pipeline | Integración de los once candidatos y controles, causalidad, escalado y fallos comprobados | Parcial; sólo prototipos aislados |
| Preregistro | Fuente ejecutable, calendario, presupuesto, hipótesis y análisis publicados antes de ajustes | Pendiente |
| Selección y dimensionamiento | HPO propio, recetas congeladas y planificación de repeticiones que contemple tiempo y entrenamiento | No ejecutados |
| Evaluación | Pesos y calibradores cerrados, predicciones comparables, costes y auditorías | No ejecutada |
| Conclusiones | Efectos, incertidumbre, multiplicidad, resultados negativos y límites | Sin ganador demostrado |

La [implementación](04_implementacion.md), el [preregistro pendiente](09_protocolo_registrado.md), la [equidad de HPO](11_hpo_equidad.md) y las [arquitecturas](14_arquitecturas.md) desarrollan estas condiciones. El [plan editorial de fases](../_static/literature_recovery_second_20261108/next_phases.json) es una ayuda de trabajo, no un preregistro ni un recibo de ejecución.

## 6.6 Evidencia histórica separada

El [recibo P0/OOF](../_static/remediation_preflight/execution_summary.json) conserva 220 trayectorias y 440 ajustes de desarrollo anteriores; sus [métricas](../_static/remediation_pilot_results/aggregate_metrics.csv) y [análisis](../_static/remediation_pilot_results/summary.json) no son la tabla de este benchmark. El [recibo HPO del 06/11/2026](../_static/completion_preflight/execution_progress.json) acredita un inicio parcial histórico, no una búsqueda actual del catálogo vigente. Se preservan protocolos y resultados originales sin reproducir aquí el catálogo retirado ni comprometer su continuación.
