# 3. Desarrollo y evidencias

**Estado actual: diseño conceptual propuesto; no hay implementación, entrenamiento ni ejecuciones de E3 registrados.** El contenido de este capítulo se completará con el trabajo que corresponda a la guía específica.

## Procedencia del trabajo

Se distinguirán las aportaciones propias de E3, los componentes reutilizados y las referencias externas. Cuando se incorpore un componente previo, se citará una versión verificable y se explicará su función en el nuevo desarrollo.

Las evidencias se enlazarán cuando existan: código, configuraciones, documentos o artefactos que se hayan generado para E3. No se publican tablas de ejecuciones vacías con valores cero ni métricas de otros entregables como sustituto de evidencia nueva.

## Estado de las evidencias

| Producto | Estado |
|---|---|
| Desarrollo propio de E3 | Candidato CNN-1D + LSTM + atención temporal; implementación pendiente |
| Componentes reutilizados | Contrato de datos de E1 seleccionado como referencia; código concreto y versiones pendientes |
| Configuraciones o registros de ejecución | No generados |
| Artefactos de respaldo | No generados |

Estas categorías organizan el trabajo futuro; no acreditan cumplimiento de requisitos todavía no recibidos.

El diagrama conceptual es: D7×5 relativo → CNN-1D causal → LSTM unidireccional → atención temporal → salida σ₇. No usa la etiqueta del día objetivo ni rezagos mensuales del objetivo. Los pesos de atención no se interpretan automáticamente como explicaciones causales.
