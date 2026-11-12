# 6. Benchmark vigente: diez modelos y un candidato adicional

**Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

**Vigencia al 11/11/2026:** Hay selección CUDA parcial documentada en [§36](36_avance_campana_cuda.md) y un corte vigente con 881 ajustes HPO nuevos completados en [§41](41_avance_optimizacion_11_origenes.md); el benchmark final, R y el ranking siguen pendientes. Los estados «sin HPO» de abajo pertenecen al corte del 08/11/2026, no al estado actual. [Matriz de exigencias y evidencia](38_cumplimiento_guia.md) · [Calendario vigente de once orígenes](40_calendario_original.md).


**Actualización de planificación:** [§19: auditoría real de desarrollo y plan propuesto](19_plan_auditoria.md). Sin entrenamiento de mercado, HPO ni preregistro nuevo.

**Avance de software:** fábrica de los once candidatos, híbrido y flujo sintético implementados y verificados; recetas provisionales, sin datos de mercado. Véase [§18: integración y pruebas sintéticas](18_integracion_sintetica.md). No hay HPO ni benchmark vigente ejecutado.

**Estado al 08/11/2026: sin resultados de mercado del nuevo catálogo.** La presentación de E2 considera únicamente los diez modelos indicados abajo y CNN-1D + LSTM + atención temporal como candidato 11. La expresión «modelo superior» representa la mejora que se desea investigar; no es una conclusión respaldada por evidencia.

Se conservan cuatro prototipos compactos y un corrector con HAR externo, ahora histórico inactivo, con diez pruebas sintéticas documentadas. El nuevo híbrido tiene receta provisional: está implementado provisionalmente y probado con datos sintéticos, sin entrenamiento de mercado. Las diez pruebas anteriores no lo validan; sus nuevos controles están en §18. Los seis adaptadores existentes se integraron sin reducir sus mecanismos y pasaron el smoke sintético; la auditoría del flujo de mercado sigue pendiente. No se han ejecutado el HPO, el dimensionamiento ni la evaluación de este conjunto. Las métricas de E1, P0 o ETTh1 y las de artículos externos no rellenan sus resultados.

## 6.1 Tabla de los once candidatos

| N.º | Modelo | Estado de implementación para este diseño | RMSE y coste de mercado |
|---|---|---|---|
| 1 | FCN-1D | Prototipo integrado y comprobado sintéticamente; receta definitiva pendiente | No medidos |
| 2 | ResNet-1D | Prototipo integrado y comprobado sintéticamente; receta definitiva pendiente | No medidos |
| 3 | Inception-1D | Prototipo integrado de una red; receta definitiva pendiente | No medidos |
| 4 | LSTM-autoencoder con predictor | Pretrain y ajuste supervisado sintéticos; entrenamiento de mercado por definir | No medidos |
| 5 | TimesNet | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 6 | TSMixer | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 7 | TiDE | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 8 | TimeMixer | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 9 | PatchTST | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 10 | iTransformer | Adaptador integrado y smoke sintético aprobado; receta definitiva pendiente | No medidos |
| 11 | CNN-1D + LSTM + atención temporal | Prototipo integrado y pruebas sintéticas; receta definitiva pendiente | No medidos; superioridad no demostrada |

**No medido no significa cero ni un empate.** Esta tabla es el catálogo y su estado, no un ranking. Los controles CNN sola, LSTM sola y CNN–LSTM sin atención se reportarán dentro del estudio del candidato 11 cuando se definan; no reintroducen el catálogo descartado ni acreditan ensayos ya realizados.

## 6.2 Qué debe medir una comparación válida

La tarea común conserva D7×5 derivado de OHLCV mediante `relative_window` y predicción a un día de σ₇ poblacional, sin anualización, en unidades originales, según el [contrato sincronizado E1–E2–E3](17_contrato_sincronizado.md). El híbrido no recibe historia adicional ni rezagos del objetivo. Las fechas comunes y las exclusiones se fijan antes de comparar; no se eliminan fechas a conveniencia según el modelo o el error observado.

La métrica primaria es la **media de las RMSE de cada trayectoria/semilla**, calculadas sobre el mismo calendario elegible:

$$
q_{m,s}=\sqrt{|\mathcal T|^{-1}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2},
\qquad \widehat\mu_m=R^{-1}\sum_s q_{m,s}.
$$

La expresión define la agregación, **no fija R**. No se reemplaza por la RMSE de predicciones promediadas, una raíz de errores mezclados entre semillas ni una media no ponderada de RMSE mensuales. Si una receta usa un ensemble, deben definirse de antemano sus miembros y qué constituye una réplica de esa receta.

Las métricas secundarias y diagnósticos se fijarán en el nuevo preregistro. Al informar MAE, MSE, R², MAPE o MASE, se declararán denominadores, cobertura y valores no estimables. MAPE no oculta objetivos cero; MASE usa únicamente pares temporales elegibles del entrenamiento, sin comprimir huecos. Si se evalúan varianza o QLIKE, debe registrarse la transformación de σ₇ y cualquier suelo numérico antes de puntuar.

Cada resultado debe acompañarse de receta, origen, semilla, número de fechas, escala, fallos y procedencia de los pesos. Repetir una predicción determinista para una resta pareada no crea entrenamientos independientes.

## 6.3 Cómo evaluar al candidato 11

CNN-1D + LSTM + atención temporal sólo podrá presentarse como mejor bajo una comparación definida y respaldada por datos. El diseño enlaza convoluciones 1D causales que conservan siete posiciones, LSTM unidireccional con `return_sequences=True`, atención temporal sobre esos siete estados observados, vector de contexto y cabeza escalar de σ₇. Antes de entrenar se necesitan:

1. **Receta y salida:** definir dimensiones, kernels, profundidad, función de atención, enlace y pérdida sin presentarlos como hiperparámetros ya elegidos. Es una adaptación propuesta, no una reproducción exacta verificada del curso o de una publicación.
2. **Validación causal:** comprobar la ventana y sus transformaciones, padding exclusivamente izquierdo, estado recurrente reiniciado y ausencia de información futura. La atención no añade observaciones fuera de D7.
3. **Controles:** CNN sola, LSTM sola y CNN–LSTM sin atención, con entradas, calendario y evaluación comunes. Son ablaciones, no ampliaciones del catálogo ni resultados ya obtenidos.
4. **Equidad:** contabilizar la búsqueda, entrenamiento de todos los bloques, reajustes y ablaciones. No conceder búsquedas extra hasta lograr ventaja.
5. **Decisión previa:** comparador, mejora mínima relevante, familia de contrastes, multiplicidad, incertidumbre y criterio de coste definidos antes de resultados.

Combinar mecanismos no acredita novedad ni mejora garantizada. Se aceptarán resultados nulos o peores; no se seleccionará retrospectivamente la métrica o cohorte que favorezca al candidato. HAR + red residual y sus exigencias de soporte adicional/cross-fitting quedan como antecedente inactivo, no como obligaciones del nuevo diseño.

## 6.4 Incertidumbre, eficiencia y límites

**Calendario y semillas representan incertidumbres distintas.** Los remuestreos temporales deben usar bloques comunes de fechas y recalcular cada RMSE antes de agregar. La variación entre semillas describe el entrenamiento condicionado a un calendario; no equivale a observar más mercados. Longitudes de bloque, sensibilidades, número de remuestreos y familias requieren especificación propia, no copia automática del diseño retirado.

Los intervalos del score no son intervalos predictivos de σ₇. Una rama de calibración exige separación del ajuste y de la selección, un predictor congelado y reglas anteriores al ranking. Su diseño y ejecución para estos once candidatos están pendientes.

Los costes se medirán con hardware, precisión y protocolo comunes: búsqueda, preentrenamiento, ajuste final, preprocesamiento e inferencia separados. Parámetros, memoria, latencia y FLOPs/MACs deben indicar método y operaciones omitidas. Una magnitud no medida es ausente, no cero. Las fronteras de Pareto y la dominancia puntual tampoco demuestran superioridad con incertidumbre.

La comparación temporal puede requerir DM y Model Confidence Set conforme a la guía, con pérdidas y unidades apropiadas. DM sobre pérdidas cuadráticas no prueba por sí mismo una ganancia práctica de RMSE. El MCS no establece equivalencia. No se fabricarán datasets independientes usando semillas o meses solapados para pruebas globales entre datasets.

La reserva ya consultada en etapas anteriores conserva carácter **exploratorio**. Registrar nuevos ajustes no elimina esa exposición. No hay evidencia actual para afirmar generalización a otros activos o regímenes.

## 6.5 Condiciones de cierre: tabla histórica del 08/11

Los estados de implementación/HPO de esta tabla son antecedentes. Para selección ejecutada consultar [§41](41_avance_optimizacion_11_origenes.md); el benchmark y análisis finales siguen pendientes.

| Etapa del catálogo vigente | Evidencia necesaria | Estado |
|---|---|---|
| Recetas y fuentes | Arquitecturas, adaptaciones, objetivos y controles del híbrido identificados | Pendiente de congelación |
| Pipeline | Integración de los once candidatos y controles, causalidad, escalado y fallos comprobados | Once candidatos implementados provisionalmente y smoke sintético verificado (§18); desarrollo real auditado (§19), ejecutor de mercado aún sin integrar |
| Preregistro | Fuente ejecutable, calendario, presupuesto, hipótesis y análisis publicados antes de ajustes | Pendiente |
| Selección y dimensionamiento | HPO propio, recetas congeladas y planificación de repeticiones que contemple tiempo y entrenamiento | No ejecutados |
| Evaluación | Pesos y calibradores cerrados, predicciones comparables, costes y auditorías | No ejecutada |
| Conclusiones | Efectos, incertidumbre, multiplicidad, resultados negativos y límites | Sin ganador demostrado |

La [implementación](04_implementacion.md), el [preregistro pendiente](09_protocolo_registrado.md), la [equidad de HPO](11_hpo_equidad.md) y las [arquitecturas](14_arquitecturas.md) desarrollan estas condiciones. El [plan editorial de fases](../_static/literature_recovery_second_20261108/next_phases.json) es una ayuda de trabajo, no un preregistro ni un recibo de ejecución.

## 6.6 Evidencia histórica separada

El [recibo P0/OOF](../_static/remediation_preflight/execution_summary.json) conserva 220 trayectorias y 440 ajustes de desarrollo anteriores; sus [métricas](../_static/remediation_pilot_results/aggregate_metrics.csv) y [análisis](../_static/remediation_pilot_results/summary.json) no son la tabla de este benchmark. El [recibo HPO del 06/11/2026](../_static/completion_preflight/execution_progress.json) acredita un inicio parcial histórico, no una búsqueda actual del catálogo vigente. Se preservan protocolos y resultados originales sin reproducir aquí el catálogo retirado ni comprometer su continuación.
