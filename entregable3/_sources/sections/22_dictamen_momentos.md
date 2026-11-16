# 22 Dictamen de admisibilidad: qué sostiene la cabeza de momentos

**16 de noviembre de 2026. Decisión interna de desarrollo:** conservar la [formulación por momentos](21_propuesta_momentos.md) como adaptación experimental documentada; **no seleccionarla todavía como aporte original de E3 ni iniciar otro piloto en este corte**. La sustantividad no está acreditada. Tampoco se demostró que la receta completa ya esté publicada o carezca de novedad. R1 permanece pendiente, con R2 y R3 sin acreditar.

La base sigue siendo CNN causal → LSTM unidireccional → atención temporal. La cabeza aditiva v2 continúa como referencia predictiva de desarrollo; no se anuncia un ganador frente al Top 10. El [dictamen completo](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/README.md) distingue la decisión técnica de una calificación o aprobación del docente.

## 22.1 Qué aporta la revisión adicional

La cota inferior y los scores ya tienen antecedentes atribuidos. La reapertura focal de [ConstraintNet](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf) confirma que una guarda puede usar regiones suministradas por otra red. Por tanto, aprender los límites no basta como argumento de originalidad. La inferencia sobre nuestra propuesta es que su diferencia debe justificarse por el vínculo conjunto específico entre momentos y media, no por usar una guarda o límites aprendidos.

En [Faithful](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf), el comparador de media sola conserva todos los ancestros de la media. En nuestra cabeza `m,q` son ancestros de `mu=L+rho(H-L)`: sus scores pueden cambiar la predicción aunque se detenga su gradiente al codificador. No se hereda la garantía de ese artículo. La [comparación primaria y los localizadores corregidos](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/primary_comparison.md) separan estos hechos de nuestras inferencias; Faithful corresponde a **§2 Methods**, páginas físicas 2–4, no a §3 como indicaba el primer registro.

El control que faltaba es la **misma cabeza de momentos con MSE sola**, retirando únicamente las pérdidas auxiliares. Una cabeza directa con igual número de parámetros responde otra pregunta y no sustituye ese control.

## 22.2 Dos límites que pueden demostrarse

Con lecturas funcionales arbitrarias, para cualquier `mu>sqrt(A)` se puede elegir `m=(mu²-A)/c`, `q=m²`, obteniendo `L=H=mu`. La región propuesta no amplía el rango de medias puntuales; restringe su compatibilidad con los momentos reportados. Eso no demuestra igualdad de clases de funciones de dos redes finitas ni ausencia de novedad. Su diferencia puede actuar como sesgo inductivo, con efectos que requieren evaluación específica.

**Momentos incorrectos.** Sea `A=0,V=1` casi seguramente y `c=6/49`; la media verdadera es `sqrt(c)`. Con reportes auxiliares válidos pero erróneos `m=4,q=32`, la envolvente obliga a

$$
\widehat\mu\ge L=\sqrt{2c}>\sqrt c,
\qquad\operatorname{MSE}_{\min}=c(\sqrt2-1)^2>0.
$$

La media libre puede acertar conservando los mismos dos reportes y sus scores. Se fijan esos valores para aislar la restricción; no se afirma que la red vaya a aprenderlos en BTC. La envolvente estimada no garantiza una mejora del error.

**Actualización auxiliar.** En esa misma ley, con la inicialización propuesta `m=1,b=√8,d=1`, se calcularon pasos escalares desde las mismas tres coordenadas de lectura. Para pasos suficientemente pequeños, los auxiliares desplazan la media más lejos del valor verdadero que MSE sola. Con paso `1e-5`, la MSE normalizada es aproximadamente `0,1111105461` frente a `0,1111104078`. Se comprueban además los pasos prefijados `1e-6` y `1e-4`.

Son **seis actualizaciones escalares independientes de SGD**, con contexto fijo, no un entrenamiento neuronal ni una simulación del Adam futuro. Los [resultados descargables](../_static/e3_dictamen_momentos_20261116/counterexamples.json) y la [derivación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/math_review.md) documentan las hipótesis, derivadas y tolerancias. Ningún ejemplo invalida el teorema de consistencia condicional existente ni explica causalmente el resultado del piloto de Bitcoin.

## 22.3 Cómo tendría que aislarse una contribución

| Comparación futura | Qué resolvería |
|---|---|
| Misma cabeza por momentos, con y sin auxiliares; ambas 1236 parámetros | Efecto de las pérdidas auxiliares sobre el mismo mecanismo. |
| Cabeza por momentos frente a media directa + mismos auxiliares; ambas 1236 | Efecto de la compatibilidad entre media y momentos. |
| Media directa con auxiliares separados y gradiente detenido al contexto, frente a su propia media sola | Control inspirado en Faithful, sujeto a comprobar sus condiciones. |
| Referencia aditiva v2 congelada y ablaciones Jensen / `q=9m²` | Utilidad frente a una referencia fuerte y necesidad de cada componente. |

Igualar parámetros no prueba igual expresividad, coste o trayectoria. No se implementaron estos controles ni se reajustan los rivales congelados de E2. La evaluación confirmatoria, todavía pendiente, conserva la familia completa, potencia, conjuntos adicionales y el margen heredado.

## 22.4 Revisión externa preparada, sin aval simulado

La extensión realizó ocho consultas web —seis EN y dos ES— y añadió lecturas focales de dos antecedentes, con [registro exacto](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/search_extension.json). Una ruta por citas terminó en lectura primaria y otra quedó incompleta por falta de acceso. No identificar la composición exacta no prueba su ausencia. Las consultas web no completan las búsquedas nativas exigidas por §6.3.

Se prepararon un {download}`paquete para la persona revisora <../_static/e3_dictamen_momentos_20261116/human_review.md>` y [cinco familias bilingües planificadas](../_static/e3_dictamen_momentos_20261116/native_search_plan.csv). Las plantillas no están diligenciadas, no se ejecutaron esas búsquedas nativas y no existe firma o concepto humano. La persona externa debe repetir al menos dos cadenas y emitir su propia valoración; la revisión entre agentes de IA no la sustituye. No se contactó a terceros.

La [decisión estructurada](../_static/e3_dictamen_momentos_20261116/decision.json), la [procedencia y uso de IA](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/provenance.md) y el [audit reproducible](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e3_audit_moment_admissibility.py) hacen revisable este corte. No hubo datos o inferencias nuevas de mercado, carga de NPZ/checkpoints, selección de semillas o nuevos conjuntos de confirmación. Se conservan las cinco recetas neuronales entrenadas y el presupuesto anterior; el `N_d` global sigue pendiente de conciliación.
