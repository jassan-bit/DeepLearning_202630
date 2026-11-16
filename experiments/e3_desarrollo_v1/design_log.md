# Continuación de la bitácora de diseño: primer piloto real

Corte: 16 de noviembre de 2026. La
[bitácora del 15 de noviembre](../e3_bloque_c_20261115/design_log.md) conserva
el estado histórico de software y teoría. Esta continuación registra la nueva
evidencia sin reescribir aquel corte.

| Variante | Datos y selección | Evidencia | Decisión |
|---|---|---|---|
| `full` | 532 ajuste, purga 7 días, validación interna 116; semilla 20261116; mejor época por RMSE. | RMSE 0.0060453143; MAE 0.0044823116; época 35, de 40 ejecutadas. | No hay mejora observada sobre M00; conservar resultado adverso. |
| `M00` | Mismas fechas, semilla e inicialización compartida; sin penalización ni saturación. | RMSE 0.0053767665; MAE 0.0039659159; época 39, de 40 ejecutadas. | Control con menor error descriptivo; no acredita novedad ni superioridad confirmatoria. |

Fuente inmutable: commit
[`f7826d1`](https://github.com/jassan-bit/DeepLearning_202630/tree/f7826d120f3a6eb8e9a0952dd9842cd77f152f34/runs/desarrollo_v1),
corrida `9e40ce379a7b4ee1a3e2fb306071cdcb`. Tiempo conjunto registrado:
369.477959639 segundos. El máximo de 40 épocas terminó ambas recetas; no se
agotó el presupuesto ni se activó la paciencia de ocho épocas.

## Recuento acumulado y alcance

- **Cinco estructuras implementadas y examinadas como software:** `full`, sin
  penalización, sin saturación, sin contenido y el control conjunto `M00`.
- **Dos variantes evaluadas con métricas de mercado** (`N_d` de selección por
  desempeño en este corte): `full` y `M00`, una semilla cada una. Las tres
  intervenciones individuales siguen sin resultados de entrenamiento de mercado.
- **Tres diagnósticos analíticos posteriores, sin ajuste:** proyección de full,
  proyección de M00 y piso solo. Se registran como análisis explorados; no se
  suman como redes entrenadas ni se omiten de la trazabilidad.

La validación usada para elegir época y examinar estos diagnósticos continúa
siendo desarrollo. No existe protocolo confirmatorio congelado o preregistro
anterior a este piloto. La revisión posterior no altera esa cronología.

La [revisión posterior](../e3_revision_post_piloto_20261116/README.md) conserva
insumos, diagnóstico reproducible, teoría, fuentes y la decisión de continuación.

## Ficha propuesta después de la revisión

La [ficha CNN-LSTM-atención v2](../e3_ficha_cnn_lstm_atencion_v2_20261116/README.md)
mantiene la misma arquitectura base y propone tres recetas neuronales nuevas:
cabeza de innovación con pérdida mixta, su ablación sin NLL auxiliar y un control
de cabeza positiva aditiva. Permanecen en estado de diseño, sin implementación
de las redes ni resultados de mercado. El recuento de variantes evaluadas con
datos de mercado sigue siendo dos.

Se implementó únicamente el operador matemático escalar de la cabeza y se
verificó con entradas sintéticas. Las órdenes 64/128/256 comparadas pertenecen
al análisis de precisión numérica, no a una búsqueda de arquitectura o de
desempeño de mercado. El plan propone 600 segundos adicionales dentro del límite
registrado existente; todavía no se ha ejecutado ni reservado esa nueva corrida.

## Implementación v2 preparada, sin nueva evaluación de mercado

El [piloto v2 implementado](../e3_innovacion_v2_20261116/README.md) añade las tres
recetas neuronales propuestas, manteniendo la CNN-LSTM-atención. Las pruebas de
software emplean datos sintéticos. El recuento de variantes evaluadas con datos
de mercado permanece en dos; no hay resultados v2 al publicar esta actualización.
El nuevo notebook conserva el presupuesto y los artefactos de v1 y recupera una
única corrida v2 desde GitHub.
