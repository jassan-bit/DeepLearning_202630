# Paquete de verificación externa de novedad — SIN DILIGENCIAR

**Estado: plantilla preparada el 16 de noviembre de 2026; ninguna revisión humana, firma o aprobación registrada.** La revisión asistida por IA no cumple esta casilla. Este documento no solicita contactar a nadie ni atribuye un concepto al docente.

## Qué revisar

Objeto exacto: misma CNN causal → LSTM → atención, con cabeza `m>0`, `q=m²(1+b²)`, `rho=d²/(1+d²)` y media `L(A,m,q)+rho(H(A,m)-L(A,m,q))` de `Y=sqrt(A+cV)`. La pérdida suma MSE y scores conocidos de media para `V,V²`. El [dictamen interno](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/README.md), la [formulación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_mecanismo_momentos_20261116/theory.md), el [análisis crítico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/math_review.md) y la [comparación primaria](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/primary_comparison.md) deben revisarse juntos.

No se reclama invención de CNN/LSTM/atención, Jensen, QLIKE o guarda de intervalo. La pregunta es si la diferencia conjunta constituye una adaptación algorítmica sustantiva para esta tarea, debidamente atribuida. No se pide verificar que supera al Top 10: no hay evidencia confirmatoria de ello.

## Búsqueda y repetición independiente pendientes

La guía original, §6.3, páginas físicas 11–12, pide al menos tres bases, al menos cinco cadenas con sinónimos EN/ES, matriz de cercanos, seguimiento de citas y búsqueda repetida antes de entregar. Otra persona ajena al diseño debe repetir al menos dos cadenas y anexar su concepto. Los nombres de bases son ejemplos según el problema; no se da por aprobada la equivalencia de servicios públicos de metadatos.

El [plan CSV](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/native_search_plan.csv) contiene cinco familias bilingües **planificadas, no ejecutadas**. Las consultas web ya realizadas están en el [registro de extensión](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_dictamen_momentos_20261116/search_extension.json) y el [registro inicial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_mecanismo_momentos_20261116/literature/search_log.json). Las solicitudes API bloqueadas no tienen conteos de resultados conocidos. Primero deben completarse y guardarse búsquedas nativas reproducibles con cobertura apropiada. El revisor externo repetirá al menos dos cadenas exactas, dejando constancia de diferencias de fecha, base y filtros; no se prellenan resultados.

Para cada ejecución guardar fecha/hora, base, consulta exacta adaptada a su sintaxis, campos y filtros, total mostrado, exportación o evidencia de resultados, artículos incluidos/excluidos y motivo. «Sin acceso» se registra como tal, no como cero artículos. Scopus, Web of Science e IEEE Xplore son opciones para evaluar según disponibilidad y cobertura; arXiv y otros repositorios pueden complementar la revisión. La lista no registra acceso ni impone pagar una suscripción.

## Campos que debe completar la persona revisora

| Campo | Respuesta humana pendiente |
|---|---|
| Nombre, fecha y relación con el proyecto | |
| Declaración de independencia respecto del diseño | |
| Cadena 1 exacta, base, filtros y resultados | |
| Cadena 2 exacta, base, filtros y resultados | |
| Evidencias o exportaciones enlazadas | |
| Comparación con Merhav, Gneiting, ConstraintNet y Faithful | |
| Otras aplicaciones y antecedentes cercanos encontrados | |
| ¿Qué cambia exactamente en ecuaciones, pérdida o entrenamiento? | |
| ¿Qué antecedente contiene o hace insuficiente esa diferencia? | |
| Concepto razonado: diferencia sustantiva / insuficiente / no concluyente | |
| Límites de la revisión y trabajo faltante | |
| Confirmación de autoría del concepto, si corresponde | |

La persona puede rechazar la interpretación interna. Si encuentra un antecedente idéntico o una diferencia sólo de aplicación debe aportar el localizador; si no lo encuentra, eso no demuestra ausencia absoluta. No se generará una firma, fecha retroactiva o aval en su nombre. La clasificación académica final corresponde al procedimiento del curso.
