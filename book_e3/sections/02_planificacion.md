# 2. Planificación

El plan de trabajo se cerrará después de conocer el objetivo y los productos del Entregable 3. Se eligió la continuidad CNN-1D + LSTM + atención temporal, pero no se han congelado configuración experimental ni presupuesto de ejecución para E3.

## Registro de decisiones

Cada decisión que resulte aplicable deberá conservar su fecha, motivo, versión y evidencia. Las decisiones abiertas se identificarán como pendientes; una propuesta todavía no ejecutada no figurará como resultado.

| Decisión | Estado |
|---|---|
| Alcance del desarrollo | Pendiente de guía |
| Recursos y materiales necesarios | Por determinar |
| Uso de antecedentes de E1 o E2 | Contrato E1 actual como referencia y candidato compartido con E2; versiones y reutilización de código pendientes |
| Procedimiento y criterios de evaluación | Por definir según la tarea |

Si el entregable requiere una evaluación experimental, las decisiones de selección y comparación deberán fijarse antes de consultar sus resultados. Un periodo ya analizado en otro entregable se identificará como información conocida; su repetición no creará una evaluación independiente.

Este apartado se actualizará con el plan efectivamente adoptado y conservará las revisiones que cambien su alcance.

## Secuencia propuesta, no ejecutada

1. Cotejar la guía de E3 y formalizar su relación con el candidato compartido.
2. Especificar CNN causal sobre D7×5 → LSTM unidireccional con siete estados → atención temporal sobre esos estados → cabeza escalar de σ₇. Anchos, kernels, capas, regularización y enlace no están congelados.
3. Implementar pruebas de forma, gradientes, causalidad, independencia entre muestras y preprocesamiento dentro de cada fold. Las pruebas del prototipo HAR no validan este híbrido.
4. Registrar selección temporal, HPO equitativo, piloto y dimensionamiento propios; no copiar R=309 de E1.
5. Evaluar con condiciones comparables, controles de componentes e inferencia prefijada. Sobre las 608 fechas conocidas, declarar alcance exploratorio; una evaluación independiente necesitaría otro periodo definido y no usado para selección.

Ningún paso de entrenamiento o puntuación se ejecuta con esta actualización editorial.
