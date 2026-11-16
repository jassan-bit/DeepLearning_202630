# Dictamen interno de admisibilidad de la cabeza de momentos

**16 de noviembre de 2026. Decisión de desarrollo: conservar la formulación como adaptación experimental documentada; no seleccionarla todavía como aporte original de E3 ni pasarla a una campaña confirmatoria.** No se inicia otro piloto en este corte. Es una decisión técnica del proyecto con la evidencia disponible, no una calificación, una sanción o un aval del docente.

La base sigue siendo **CNN causal → LSTM unidireccional → atención temporal**. La cabeza aditiva v2 conserva su papel de referencia predictiva de desarrollo. No se acredita un ganador frente al Top 10. Este dictamen profundiza y delimita la [propuesta anterior](../e3_mecanismo_momentos_20261116/README.md), sin sustituir sus resultados o su demostración condicional válida.

## 1. Qué cambió al revisar con mayor profundidad

| Pregunta | Hallazgo verificable | Consecuencia para la decisión |
|---|---|---|
| ¿Es nueva la cota inferior? | Se especializa de Jensen reponderado de Merhav (2023), ec. (8). | Se atribuye; no se presenta como desigualdad inventada aquí. |
| ¿Son nuevos los scores de momentos? | Son scores conocidos de media, relacionados con QLIKE/Bregman; Gneiting (2011), §3.1. | Elicitar `E[V]` y `E[V²]` no constituye por sí solo la contribución. |
| ¿Distinguen la propuesta los límites aprendidos? | ConstraintNet contempla restricciones por muestra, incluso suministradas por otra red, y guardas de intervalo. | La procedencia neuronal de los límites, por sí sola, no es diferencia sustantiva suficiente. |
| ¿Protege la media como Faithful? | `m,q` son ancestros de la media; sus scores pueden cambiarla aunque se detenga su gradiente al codificador. | No se hereda fidelidad de trayectoria. Falta el control de **la misma cabeza con MSE sola**, no sólo una media directa de igual tamaño. |
| ¿Amplía las medias puntuales representables? | Con lecturas funcionales arbitrarias, toda `mu>sqrt(A)` ya es representable por una cabeza positiva directa y por la de momentos. | El argumento de mayor expresividad puntual no sostiene el aporte. Esto no prueba equivalencia de dos redes finitas. |
| ¿La envolvente siempre mejora el error? | Un ejemplo con momentos válidos pero incorrectos fuerza una media peor; otro produce pasos SGD distintos y peor MSE que MSE sola. | La pertenencia a la región calculada no certifica precisión ni beneficio de los auxiliares. |

La [comparación primaria](primary_comparison.md), el [análisis matemático](math_review.md) y los [contraejemplos reproducibles](counterexamples.json) permiten revisar cada conclusión. Las inferencias sobre nuestro mecanismo se distinguen de lo que afirman los artículos. La corrección de localizadores del primer corte está en [errata.md](errata.md).

## 2. Diferencia que queda en pie y qué no demuestra

La diferencia específica es **restringir conjuntamente** la media de `sqrt(A+cV)` y dos momentos de `V`, mediante una guarda derivada de su identidad algebraica. Esa restricción puede actuar como sesgo inductivo y modificar la estimación y sus gradientes. No es exactamente la misma red que una cabeza libre, y no se encontró una fuente que permita afirmar que la receta completa ya está publicada.

Tampoco se acreditó que esa diferencia sea una adaptación sustantiva para la tarea. Compartir la clase puntual o el óptimo Bayes con otros métodos no prueba ausencia de novedad; muchos mecanismos distintos comparten esos objetos. Lo que se descarta es justificar la originalidad por una desigualdad nueva, un funcional de media nuevo, mayor expresividad puntual o una garantía general de proteger la MSE. No se formula otro candidato para eludir este resultado.

**Estado académico:** R1 sigue pendiente, sin declaración de novedad; esta cabeza no implementada no acredita R2; R3 no se evaluó. No se transforma una búsqueda incompleta en una declaración de «modelo no novedoso» ni una propiedad matemática correcta en una certificación de modelo original.

## 3. Controles necesarios antes de una eventual ejecución

Los controles se especifican conceptualmente; no se entrenaron ni se modificó la familia congelada de E2.

| Control | Comparación que permite | Capacidad analítica |
|---|---|---:|
| **Misma cabeza de momentos, `lambda_m=lambda_q=0`** | Aísla el efecto de las dos pérdidas auxiliares; conserva todos los ancestros de la media. | 1236 |
| Media directa + mismas dos cabezas/scores auxiliares | Aísla la restricción conjunta de salida con tres lecturas activas. | 1236 |
| Media directa con auxiliares cuyo gradiente se detiene en el contexto compartido | Control inspirado en Faithful con ramas auxiliares separadas de los ancestros de su media; necesita su propio comparador directo MSE solo y condiciones de actualización idénticas. | 1236, con 1210 parámetros ancestros de la media |
| Cabeza aditiva v2 conservada | Referencia más fuerte observada en desarrollo; no aísla una sola intervención. | 1210 |
| Plug-in Jensen y versión con `q=9m²` | Ablaciones del uso de `q` libre y de la posición de media. | 1210 / 1223 |

Detener sólo el gradiente al codificador en **nuestra cabeza acoplada** no basta para heredar Faithful: actualizar las lecturas `m,q` todavía cambia la media. El control directo inspirado en ese trabajo es otra arquitectura y su garantía, si se demuestra bajo las mismas condiciones, corresponde a su propia media directa. Igualar 1236 parámetros no demuestra igual expresividad, trayectoria o coste: se deben medir y declarar esas diferencias.

La igualdad de actualizaciones requeriría además las mismas condiciones de inicialización, aleatoriedad y optimizador. Un clipping global de todos los parámetros o un scheduler guiado por la pérdida conjunta puede volver a acoplar las ramas; su tratamiento debe fijarse explícitamente y la selección de época debe compararse bajo el mismo criterio.

Para reabrir esta línea se necesita una justificación sustantiva frente a los antecedentes, una revisión reproducible completa y externa y un diseño que aísle la contribución con los controles anteriores. Sólo entonces tendría sentido decidir una eventual prueba de desarrollo con presupuesto y criterios prefijados. Mejorar un piloto no resolvería por sí solo R1 ni probaría R3. La evaluación confirmatoria conserva Top 10, baselines fuertes y el margen heredado `delta=0,00025772842664981106`.

## 4. Alcance bibliográfico real y paquete para revisión externa

Se añadieron **ocho consultas web reales**, seis en inglés y dos en español, registradas con sus textos exactos en [search_extension.json](search_extension.json). Se examinaron focalmente Hines y Hines (2025), sobre aprendizaje con restricción de momento global, y Wunder et al. (2021), seguido desde Merhav. Difieren de nuestra región condicional y no certifican la ausencia de la composición completa. La ruta desde GenUT hacia Julier/Uhlmann quedó incompleta por falta de acceso al cuerpo y se declara como tal. Las [referencias nuevas](references_extension.bib) distinguen lectura de metadatos.

Se reabrieron también Merhav, Gneiting, Faithful y ConstraintNet, con localizadores y alcances focales; no se declara una lectura íntegra nueva de todos los artículos. Las ocho consultas se suman como extensión a las doce web del primer registro, **no como consultas nativas a bases académicas**. Los tres intentos API previos no recibieron respuesta bibliográfica; no se repitieron ni se transformaron en búsquedas exitosas. Los conteos de resultados siguen siendo desconocidos para esos intentos.

Quedan pendientes la revisión nativa con la cobertura de §6.3, el seguimiento por citas exigido, su repetición antes de entrega y una persona externa al diseño que repita al menos dos cadenas y emita su concepto. El [paquete de verificación humana](human_review.md) y las [cinco familias bilingües planificadas](native_search_plan.csv) están listos para diligenciar. **Son plantillas sin ejecución, firma o aval**, no evidencia de cumplimiento. El revisor puede disentir del dictamen y debe aportar sus propios localizadores. No se enviaron mensajes a terceros.

## 5. Operaciones y reproducción

El [audit matemático](../../scripts/e3_audit_moment_admissibility.py) calcula dos ejemplos artificiales y seis actualizaciones escalares de tres coordenadas, con derivadas analíticas y comprobación por diferencias centrales. No instancia una red, no ejecuta una campaña Adam y no busca hiperparámetros. El ejemplo SGD no predice la trayectoria del optimizador futuro ni el desempeño de BTC.

```bash
python scripts/e3_audit_moment_admissibility.py --check
```

El comando comprueba el JSON guardado con tolerancias explícitas. El libro copia exactamente ese archivo y la plantilla humana; CI comprueba las copias y la reproducción. La bitácora y la asistencia de IA se declaran en [provenance.md](provenance.md).

Los **cinco entrenamientos de recetas neuronales de mercado** entre v1/v2 y el saldo de desarrollo se conservan: consumo `556,212659362 / 1200 s`, saldo aritmético `643,787340638 s`. El `N_d` global sigue pendiente de conciliación de propuestas, variantes y ensayos; no se reduce a cinco por contar sólo recetas entrenadas. No se cargaron datos NPZ ni checkpoints, no hubo inferencia de mercado y no se abrieron nuevos conjuntos de confirmación.
