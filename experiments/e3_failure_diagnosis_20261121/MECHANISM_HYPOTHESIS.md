# Hipótesis de trabajo: interacción del objetivo auxiliar con la media

**Estado: hipótesis priorizada para medición, no causa demostrada ni candidata nueva.** La ruta se genera después de observar desarrollo y así se registra. Asistencia de IA declarada.

## Evidencia que la motiva

1. V3 empeora RMSE en3/3 semillas pero mejora MAE; V4 empeora RMSE y MAE, con5/5 semillas de peor RMSE.
2. Las recetas atómicas de cada cohorte comparten estructura/inicialización y cambian la receta de objetivo. Sus diferencias posteriores también incluyen trayectoria de optimización y checkpoint elegido.
3. En CPU ambos auxiliares pierden stop-MSE frente a MSE-only en5/5 semillas, y una semilla presenta parada temprana muy distinta.
4. No existe registro de MSEtrain/CRPS separados, gradientes por término, clipping o átomos/masas por caso. La pérdida total no permite atribuir conflicto de gradientes.

Estos hechos motivan medir la interacción. No prueban que CRPS sea impropio, que domine el gradiente, que falten épocas, que una mayor GPU ayude o que el auxiliar sea la única causa. V4 cambió coordenada y escala simultáneamente; tampoco aísla ambas causas.

## Pregunta falsable de diagnóstico

En estados y lotes de desarrollo fijados antes de medir, ¿el auxiliar cambia de forma relevante la dirección y tamaño del desplazamiento del optimizador respecto del objetivo MSE, y qué parte se explica por estado de Adam, clipping o escala?

Con g_M=gradiente de MSE normalizada y g_A=gradiente del auxiliar normalizado **sin lambda**, se define g_total=g_M+lambda·g_A, usando el peso real de cada receta una sola vez. Interesa el coseno cuando ambas normas son no nulas. En MSE-only el peso real es0: su gradiente auxiliar efectivo es cero; cualquier geometría auxiliar evaluada allí es contrafactual y debe etiquetarse, sin atribuirle influencia histórica.

Un coseno negativo es conflicto geométrico local, no por sí solo degradación de validación o generalización. Debe registrarse también el producto con un desplazamiento hipotético calculado con la regla y el estado coherente del optimizador; no se observa un paso histórico. Una proyección euclídea no garantiza descenso tras Adam, su momento y pasos finitos.

No se elige una frecuencia mínima de conflicto a partir de estas predicciones puntuales. Se necesita primero un protocolo de medición con soporte, escala y precisión definidos. La ausencia de datos para medirlo se informa como falta de evidencia, no como presencia o ausencia del mecanismo.

## Antecedentes que impiden llamarla novedosa por defecto

Se hizo una consulta focal nueva a las siguientes fuentes primarias; se leyeron sus páginas de resumen/metadatos, no se atribuye aquí lectura integral de los teoremas:

| Fuente | Aporte ya conocido relevante | Consecuencia para la propuesta |
|---|---|---|
| [Du et al., Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224v2), 2018/2020 | Usa similitud entre gradientes para modular ayuda del auxiliar al objetivo principal. | Un gating por coseno no es una contribución nueva por sí mismo. |
| [Yu et al., Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782v4),NeurIPS 2020 | PCGrad proyecta componentes de gradientes en conflicto entre tareas. | Aplicar PCGrad a MSE/CRPS requiere atribución; no acredita adaptación sustantiva automáticamente. |
| [Stirn et al., Faithful Heteroscedastic Regression](https://proceedings.mlr.press/v206/stirn23a.html),AISTATS 2023 | Modificaciones de optimización para preservar la precisión de la media frente a un modelo equivalente de sólo media en su ámbito. | Separar auxiliares o proteger una media tiene antecedentes. No se traslada sin prueba el resultado gaussiano a una ley atómica transformada. |

No se afirma una búsqueda exhaustiva o ausencia de trabajos posteriores. Una propuesta concreta necesitaría revisar sus formulaciones completas y antecedentes específicos de esta tarea antes de defender R1.

## Qué adaptación podría estudiarse si la medición la fundamenta

El objeto de investigación sería una regla que gestione la señal auxiliar de la ley atómica transformada preservando un objetivo puntual declarado, con efecto definido sobre la **actualización real** del optimizador. Falta especificar su diferencia sustantiva respecto de los antecedentes y demostrarla bajo hipótesis explícitas. No se implementa ni se selecciona esa regla aquí.

Una propiedad local de descenso bajo SGD no garantiza descenso con Adam ni RMSE fuera del lote. Una variante cuya media reproduzca exactamente MSE-only no puede afirmar estricta superioridad puntual frente a ese mismo control por construcción. La mejora distribucional podría ser un hallazgo distinto, pero no sustituye la métrica y familia actuales de R3.

## Regla de descarte y continuidad

- Si no se identifica un mecanismo medible que la regla controle, o ésta resulta equivalente a un método conocido sin adaptación sustantiva para el problema, no se propone como candidata R1.
- Si controlar la interacción elimina todo aporte auxiliar o produce exactamente la trayectoria MSE-only, se reconoce ese alcance y no se promete superioridad RMSE.
- Si posteriormente se justifica un piloto nuevo, deberá tener identidad, controles, datos de desarrollo, reglas y presupuesto anteriores a sus resultados. Perder su primaria bajo la regla fijada obliga a conservar el fracaso y detener esa receta; no a escoger MAE, fechas o semillas favorables.
- Cualquier confirmación exigiría después batería válida, datos apropiados y potencia, con independencia de que el diagnóstico geométrico resulte favorable.

**Próxima acción concreta preparada:** tras la auditoría de disponibilidad y hashes ya realizada, inspeccionar el esquema seguro de los estados, verificar el acceso limitado a las entradas de ajuste/selección y congelar la medición descrita en MECHANISM_MEASUREMENT_PLAN.json. Se vuelven a verificar los hashes antes de cargar. No autoriza un entrenamiento nuevo ni reabre V3/V4.
