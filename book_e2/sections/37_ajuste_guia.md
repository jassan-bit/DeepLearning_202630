# 37. Ajuste de cómputo conforme a la guía

**Enmienda del 11/11/2026: tres orígenes de selección en lugar de once.** La guía pide origen móvil y el mismo esfuerzo entre modelos (p. 30), pero no fija once selecciones mensuales. Se conservan `2024-01-01`, `2024-10-01` y `2025-01-01`: dos orígenes de desarrollo y la selección anterior al test histórico. Octubre conserva la base de recetas del piloto de potencia existente. La elección usa calendario y dependencias del diseño, no rankings ni métricas de prueba.

**Este cambio está implementado en un nuevo programador; no cambia una sesión de Colab que ya está ejecutándose.** El notebook de transición se enlaza abajo cuando queda fijado al código publicado. No se ha entrenado un modelo nuevo durante esta modificación.

## Qué exige la guía y qué conservamos

- NAS de 1.000 arquitecturas cuando el artículo no define la arquitectura utilizada (pp. 13, 52–54).
- Exención de NAS para el cuerpo publicado con configuración definida (p. 53). No se atribuye esa exención por nombre a nuestras variantes propias.
- Una semilla en el cribado reducido, k ≥ 10 finalistas y al menos cinco semillas diferentes para confirmación; k debe justificarse por ruido (pp. 54–55).
- Selección y evaluación temporal sin fuga y política común para todos los sistemas (pp. 30–31).

Se mantienen las seis variantes NAS y los cinco cuerpos publicados exentos de §27, además de los tres controles E1. No se cambia retrospectivamente su identidad. Se conserva k=40 y las configuraciones, semillas, máximos de 50/200 épocas y parada temprana ya registrados. Reducir k por haber observado un ganador o llamar publicado a un cuerpo propio sería otro cambio y no se hace aquí.

La guía no fija 18 configuraciones, 50/200 épocas, k=40 ni once orígenes: son elecciones metodológicas. Esta enmienda reduce los orígenes y documenta que se obtiene menor cobertura temporal; no certifica aceptación institucional ni cumplimiento integral de todos los entregables.

## Presupuesto comparado

| Búsqueda neuronal | Diseño anterior: 11 orígenes | Enmienda: 3 orígenes |
|---|---:|---:|
| HPO: 14 redes × 18 × 2 | 5.544 | 1.512 |
| NAS-1: 6 familias × 1.000 | 66.000 | 18.000 |
| NAS-2: hasta 41 finalistas × 5 × 6 | 13.530 | 3.690 |
| **Máximo de intentos lógicos de selección** | **85.074** | **23.202** |

La reducción del máximo es **72,7 %**. No es una promesa de reducir el tiempo en esa misma proporción: cambian las duraciones por receta y origen. Persisten entrenamientos costosos y los límites de sesión/presupuesto pueden pausar antes de completar la selección. Estos totales no incluyen reajustes finales, comparadores clásicos, robustez, ablaciones, calibración ni inferencia. R definitivo continúa pendiente. El presupuesto monetario y la duración total deben medirse; no se deducen de este conteo.

## Conservar resultados y cambiar de programador

El programador conserva el plan científico original, sus hashes de trabajo y todos los recibos. Recorre sólo los tres orígenes autorizados; el resumen distingue `completed_amended_three_origin_selection` de completar el dominio original de once. La cobertura indica tres orígenes y `partial_origins: true`. Los resultados compatibles del primer origen se leen de la caché/registro validado, sin entrenarlos otra vez. Resultados de otros orígenes, si existiesen, se conservan pero no se convierten en evidencia del agregado de tres orígenes.

La enmienda reduce el calendario de ejecución admitido, no altera las configuraciones ni mueve sus resultados a un plan con otra identidad. El código científico de ajustes permanece fijado en A3; el nuevo programador está versionado por separado. La procedencia de cada resultado sigue siendo el diario original. No se cambia la rama de campaña desde esta actualización del libro.

1. Cerrar el controlador anterior mediante la interrupción normal de su celda y esperar su informe de salida. No desconectar ni matar la VM para cambiar el diseño.
2. No ejecutar ambos controladores a la vez. El nuevo programador comprueba procesos y conserva el bloqueo del diario.
3. Usar el notebook nuevo, preparar el mismo entorno y activar la ejecución sólo después del cierre. No habilitar una nueva repetición R3.
4. Si aparece `reconciliation_required`, detenerse: hay un intento sin resultado confirmado. El programador no lo repite automáticamente. Interrumpir un ajuste puede requerir revisar su recibo antes de continuar.

El notebook no amplía el presupuesto: conserva las extensiones ya registradas y los límites originales para reconstruir el mismo plan. No cambia hardware, precisión ni determinismo.

## Alcance temporal y entregables pendientes

Los tres orígenes usan el mismo calendario y soporte para todos los modelos, con preprocesamiento ajustado sólo en entrenamiento. Se retiene octubre para la selección previa al piloto de potencia de noviembre–diciembre de 2024. Los bloques de evaluación no ejecutados del diseño mensual anterior no se rellenan ni se presentan como completos. La futura evaluación debe declarar estos tres orígenes, reportar por origen y usar sólo sus bloques comunes.

No se hereda automáticamente una conclusión de once orígenes a tres ni se trata cada semilla como una serie independiente. El test histórico de 608 fechas sigue siendo conocido y exploratorio. El benchmark, los reajustes y el informe estadístico deben aplicar esta enmienda antes de publicarse; no están completados por el programador de selección.

[Enmienda estructurada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e2_guide_schedule_20261111/amendment.json) · [Programador](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_guide_schedule.py).
