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
