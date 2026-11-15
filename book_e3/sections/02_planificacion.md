# 2 Plan de trabajo

El arranque utiliza el benchmark existente de E2 y la guía oficial recibida. El presupuesto anterior de E2 no se traslada como autorización o saldo de cómputo para E3. Fecha de entrega, hardware y presupuesto nuevo siguen abiertos.

## Secuencia y condiciones de cierre

| Paso | Trabajo | Evidencia necesaria para cerrar |
|---|---|---|
| 1 | Inventariar E2 y revisar las filas P0 de la trazabilidad. | Rutas, versiones, hashes y acceso a resultados, pesos, escaladores, calibradores y particiones. Identificar también artefactos privados. |
| 2 | Completar controles de datos y análisis formal pendiente de brecha con δg. | Recibos por control; marcar no estimable cuando falte evidencia. Si aparece un error real, corregir y recalcular sus productos dependientes. |
| 3 | Cerrar Bloque A y congelar el conjunto de comparación. | Tabla de impacto y ranking antes/después si corresponde; manifiesto con hashes. Conservar recetas y resultados históricos. |
| 4 | Revisar novedad y desarrollar el mecanismo sólo en desarrollo. | Búsqueda focal, trabajos cercanos, hipótesis, propiedad, bitácora y tres ablaciones sustantivas. Descartar una propuesta que no cumpla R1/R2. |
| 5 | Seleccionar fuentes adicionales y diseñar la confirmación. | Tres datasets de la misma tarea, distintos en dos ejes, fuente/licencia/versión/hash; calendario y separación documentados antes de observar resultados. |
| 6 | Ejecutar piloto y dimensionar. | Al menos cinco semillas por rival para variabilidad en desarrollo; análisis de potencia por contraste y conjunto. Determinar R≥10 y costes antes del cierre final. |
| 7 | Congelar diseño y registrar protocolo. | `experiments/estandar.md`, `protocolo.md`, `design_log.md` y `procedencia.md` completos, con fecha y tag Git reales. |
| 8 | Ejecutar confirmación y evaluación ampliada. | Misma política para todos, tres pruebas por contraste y resultados negativos conservados. |
| 9 | Exportar e integrar dashboard e informe. | Round-trip, perfiles CPU, tabla maestra única, Docker, pruebas de consistencia, Render y prueba de humo. |

El inventario de exportación puede avanzar durante la investigación. El dashboard puede diseñarse con esquema de datos y pruebas sintéticas claramente identificadas, pero no publicarse como producto final sin sus artefactos y resultados.

## Primera tarea operativa

Crear un registro por cada predictor congelado con: modelo, receta elegida en validación, semilla cuando aplique, checkpoint, escalador, calibrador, métricas, predicciones, ubicación, hash y estado de acceso. Las 81 trayectorias documentadas son el universo inicial a conciliar; una mención en el libro no verifica por sí sola que los pesos estén disponibles aquí.

Después se vinculan los controles de fuga al pipeline real y se identifica qué insumos permiten verificar la brecha con δg. Esta revisión no requiere reabrir HPO ni seleccionar nuevas semillas de E2.

## Confirmación y exposición previa

Las 608 fechas de enero de 2025 a agosto de 2026 ya se evaluaron en E1/E2. Septiembre–octubre de 2026 también tiene resultados publicados en un antecedente E1. Repetir esos periodos puede servir como comparación histórica exploratoria; no recupera cegamiento. La herencia E1→E2 está permitida y no es por sí misma una infracción.

Para E3 se deberá fijar una separación real entre desarrollo y confirmación. Una fecha posterior no garantiza independencia si ya se usó para decidir el modelo. Los datasets adicionales se seleccionan antes de conocer su desempeño y no se usan para cambiar el mecanismo; cualquier ajuste de hiperparámetros permitido debe estar preespecificado y respetar sus particiones internas.

## Decisiones abiertas antes de entrenar

- Origen verificable del mecanismo y revisión de novedad.
- Acceso a artefactos completos de E2 y resultado de los controles pendientes.
- Conjuntos adicionales y calendario de confirmación, sin selección por resultados.
- Tres pruebas temporales compatibles con sus estimandos y supuestos; alcance de Holm sobre rivales y pruebas; familia de ablaciones separada.
- Presupuesto común, capacidad, piloto, R definitivo y tratamiento de comparadores deterministas.
- Componentes suficientes para tres ablaciones interpretables; no añadir piezas sólo para completar el número.

La bitácora deberá distinguir hipótesis concebidas de variantes efectivamente evaluadas. Este plan no constituye un protocolo preregistrado ni autoriza una campaña GPU o un servicio de pago.
