# A.1 Plantilla de declaración del estándar

**PLANTILLA SIN COMPLETAR. No es `experiments/estandar.md`, no está suscrita y no registra un compromiso fechado o un tag.** Los campos pendientes deben resolverse con evidencia y revisión antes de emitir la declaración real y antes de congelar el protocolo. No se retrofechan registros.

| Campo de A.1 | Contenido exigido o campo pendiente |
|---|---|
| Estándar | Profundidad **teórica**: al menos una propiedad demostrada formalmente, con supuestos explícitos y verificación numérica; familia completa; al menos tres conjuntos adicionales; al menos tres pruebas válidas por contraste; potencia por contraste ≥0,90 y potencia conjunta reportada. Al menos diez semillas independientes cuando exista aleatoriedad, o más según potencia. |
| Justificación de viabilidad | **PENDIENTE:** cómputo disponible, perfil del equipo, costes previstos y por qué permite el estándar completo. No usar un presupuesto supuesto como si estuviera asignado. |
| Justificación de semillas | **PENDIENTE:** piloto, desviación de diferencias pareadas, alternativa frente a la frontera δ, ajuste familiar, unidades independientes, MDE y potencia por contraste/conjunta. R≥10 no certifica potencia. |
| Rivales comprometidos | **PENDIENTE DE RATIFICAR Y VINCULAR A HASHES:** inventario real de 16 rivales en `experiments/e3_bloques_ab_20261115/master_configurations.csv`; no seleccionar un subconjunto según resultados. |
| Tres conjuntos adicionales | **PENDIENTE:** nombre, fuente/licencia/versión/hash, diferencias de naturaleza y selección anterior a conocer el desempeño; no usarlos para diseñar. |
| Propiedad y supuestos | **PENDIENTE:** enunciado a comprometer, demostración/verificación previstas, alcance y pertinencia. La cota ya disponible de la cabeza latente no certifica el pipeline desde OHLCV crudo. |
| Presupuesto justo | **PENDIENTE:** intentos, espacios relativos, épocas/parada, horas/hardware, fallos y capacidad comparables; distinguir esfuerzo de diseño y HPO. |
| Familia, pruebas y decisión | **PENDIENTE:** tres procedimientos admisibles F2a y coherentes por rival, supuestos, unidades, Holm y máximo de p ajustados; decisión global sólo si todos los rivales cumplen. |
| Desarrollo y confirmación | **PENDIENTE:** cortes, exposición previa, separación efectiva, Nd y criterios de éxito/fracaso. |
| Fecha de la declaración | **PENDIENTE.** Registrar la fecha real cuando se cierre y revise el contenido. |
| Tag Git y commit | **PENDIENTE.** Identificar el registro real anterior al congelamiento; no inventar un tag. |
| Responsables y revisión | **PENDIENTE DE VALIDACIÓN HUMANA.** Esta plantilla no expresa una firma o compromiso personal confirmado. |

No se reduce el estándar después de observar resultados. Si la potencia requiere más semillas, se dimensionan antes del cierre; si hay limitaciones, se declaran sin presentar ausencia de rechazo como equivalencia. La hipótesis y el hallazgo alternativo se fijarán antes de confirmar.

Fuente normativa: Apéndice A.1 y secciones 2, 7, 8 y 14 de la guía. Fuente de trabajo: [sección 8 del informe](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/08_verificacion_estadistica.html).
