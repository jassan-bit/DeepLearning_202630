# 13 Sustentación

La guía exige **20 minutos de exposición sobre el dashboard desplegado**, seguidos de preguntas. Este capítulo es un guion de preparación; no acredita una defensa realizada, una demostración funcional o comprensión individual del código. La aplicación y su URL Render siguen pendientes.

## 13.1 Distribución del tiempo

Los cinco bloques que enumera la guía suman 18 minutos. La agenda siguiente conserva esos tiempos y propone los dos minutos restantes para transiciones y margen operativo. Las preguntas van después de la exposición.

| Ventana propuesta | Duración | Contenido obligatorio | Evidencia que deberá acompañarlo |
|---|---:|---|---|
| 00:00–02:00 | 2 min | Problema, datos y hallazgos principales del EDA. | Objetivo σ7 de siete retornos, h=1 sin anualización, fuente/periodo, figura temporal interpretada y límites de datos. |
| 02:00–05:00 | 3 min | Benchmark, mejores configuraciones y predicción en vivo con explicación local. | Tabla de 17 sistemas con estados honestos; configuración elegida en validación; entrada admitida, pronóstico, intervalo y explicación del mismo caso/artefacto. |
| 05:00–10:00 | 5 min | Modelo original, limitación, relación con paper guía, fuente, mecanismo, demostración y novedad. | Ecuaciones, pseudocódigo, código, supuestos, verificación numérica y revisión de antecedentes con su nivel de evidencia. |
| 10:00–16:00 | 6 min | Protocolo y evaluación estadística/ampliada. | Familia completa, pruebas/supuestos, Holm, efectos/IC, potencia por contraste/conjunta, calibración, robustez, coste y veredicto respaldado. |
| 16:00–18:00 | 2 min | Conclusiones, limitaciones y trabajo futuro. | Qué está demostrado, qué no, escenarios adversos y tareas restantes, sin declarar superioridad no acreditada. |
| Margen hasta 20:00 | 2 min | Transiciones, cambio de vista y contingencia operativa. | Ensayo cronometrado; no sustituye contenido ni extiende la exposición a la sesión de preguntas. |

La demo deberá usar la URL pública y una entrada precargada verificada por cada capacidad que se exhiba. Se registrarán versión, estado del servicio y procedencia del ejemplo. Las capturas o el vídeo de contingencia se identificarán como material grabado y no como una predicción ejecutada en vivo.

## 13.2 Preparación de la demostración

Antes de ensayar deben estar construidas y probadas las páginas Contexto, EDA y Modelos y la inferencia habilitada para los artefactos que realmente cumplan el contrato de producción. Los 17 sistemas se representan; un sistema precalculado o el candidato sin entrenamiento no se convierte en inferencia habilitada para completar la demo.

La [prueba de humo](10_despliegue_render.md) recorrerá todas las páginas y cada modelo con inferencia en vivo, con sus estados, errores, intervalo/explicación cuando estén disponibles y tiempos medidos. Los criterios de aceptación se fijan antes de medir, distinguiendo arranque frío y respuesta caliente. Se comprobará la URL con antelación y se preparará contingencia sólo a partir de una aplicación realmente funcionando. Hoy esos registros están pendientes.

La explicación debe distinguir el líder por validación disponible de E2 del líder descriptivo de prueba, una métrica de cohorte de una predicción de semilla representante y un nominal de cobertura de su verificación empírica. Se mantendrán visibles la exposición histórica del test y los límites de la evaluación del candidato.

## 13.3 Preguntas y comprobaciones técnicas

| Pregunta prevista por la guía | Preparación y estado actual |
|---|---|
| ¿Qué limitación concreta del paper guía o métodos existentes resuelve el mecanismo y qué evidencia la sustenta? | Estudiar [6](06_modelo_original.md) y [7](07_diseno_riguroso.md); el estrés retrospectivo motiva una hipótesis, pero su corrección por el candidato no se ha demostrado en mercado. |
| Si proviene de otra aplicación, ¿qué adaptación algorítmica hizo y por qué era necesaria? | Explicar fuente, diferencias y equivalencias con trabajos cercanos. La revisión de novedad y la verificación externa siguen abiertas. |
| ¿Qué propiedad demuestra y qué supuestos no cumplen exactamente los datos reales? | Derivar la cota latente de la sección 7, sus normas/restricciones y límites; no extenderla al codificador o OHLCV crudo sin una prueba adicional. |
| ¿Cuántas variantes Nd probó y cómo controló selección sobre validación? | Declarar cuatro variantes funcionales artificiales y Nd de mercado igual a cero en el corte existente; conciliar esfuerzo total y registrar futuros intentos/fallos. |
| Al retirar un componente, ¿qué predijo la hipótesis y qué observó? | Separar intervenciones de software de la ablación estadística, todavía no ejecutada. El factorial y el control de capacidad no tienen métricas BTC. |
| ¿Quién integra la familia y cómo corrigió comparaciones múltiples? | Enumerar los 16 rivales más el candidato; explicar Holm y el máximo entre pruebas como plan, sin presentar p del candidato inexistentes. |
| ¿Qué MDE y potencia conjunta alcanzó? | El piloto y dimensionamiento del candidato están pendientes. No trasladar la potencia histórica de E2 ni multiplicar marginales bajo independencia no justificada. |
| ¿Dónde no supera a los benchmarks y por qué? | Mostrar resultados adversos cuando existan; hoy no hay comparación de mercado del candidato que permita atribuirle aciertos o fallos de otros modelos. |
| ¿Qué cambia al retirar el conjunto adicional más desfavorable? | Los tres conjuntos siguen sin elegir/evaluar; la sensibilidad se prefija y no permite retirar un dataset para mejorar la afirmación principal. |

Durante las preguntas el docente podrá pedir derivar un gradiente, modificar una pérdida o parte del mecanismo y ejecutar una prueba unitaria o un experimento reducido. La persona autora debe poder explicar qué cambia en la formulación, los supuestos, la cota y los tests. Que el código pase CI no demuestra ese dominio técnico. La imposibilidad de explicarlo es evidencia en contra de las condiciones R1 y R2, según la guía.

## 13.4 Ensayo y trazabilidad

El ensayo deberá registrar fecha, participantes, versión, duración por bloque, ejemplo usado y fallos/acciones pendientes. Se prepararán comandos concretos en el entorno fijado y con insumos verificados; la prueba de software no se describirá como una campaña científica completa.

No se afirma que el ensayo o la sustentación hayan ocurrido. El [anexo final](cierre_reproducibilidad.md) mantiene separados el uso de IA, la revisión personal pendiente y las firmas; la [sección 14](14_evaluacion_calificacion.md) conserva los criterios de evaluación sin asignar calificación actual.
