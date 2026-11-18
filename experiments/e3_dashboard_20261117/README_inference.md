# Inferencia del dashboard: procedencia y reproducción

La aplicación usa ONNX Runtime **1.23.2** en CPU, sin Torch. Los cuatro modelos
de `artifacts/e3_dashboard_v1` conservan los pesos ya entrenados de la semilla
271801. Las tres referencias proceden de los TorchScript guardados en
`artifacts/e3_handoff`; E3 usa `best_model_state` del checkpoint completo del
commit `510c3bddfdc9926322e3210ac61ee1f5eeffad2e`, mejor época 38.

El SHA256 de ese checkpoint es
`4d440b71db042b916bc4daa95031bd222fd1c3c7a94974cff9dca7b43d9df5c9`.
El archivo está en `runs/estabilidad_aditiva_v1/seeds/271801/checkpoint.pt`
de ese commit; el exportador lo verifica antes de `torch.load(weights_only=True)`.
Las fuentes matemáticas coinciden por hash con la campaña congelada.

## Comprobación sin framework de entrenamiento

Desde la raíz y el entorno de inferencia instalado:

```bash
python -m unittest discover -s tests -p test_e3_dashboard_inference.py -v
python experiments/e3_dashboard_20261117/verify_serving.py \
  --npz work/e3_dashboard/current_cohort_arrays.npz --download \
  --output work/e3_dashboard/serving_verification.json
```

El segundo comando descarga, sólo si falta, la cohorte inmutable del commit
`af3f1223ad5899ed63016d2af2d09bae8470ab00`, comprueba su SHA256 y reproduce
las cuatro trayectorias de 608 fechas frente al paquete de predicciones
verificado. No importa Torch, entrena o recalibra. Los once casos unitarios
incluyen admisión CSV, cálculo independiente del piso, intervalos heredados,
integridad, caché y contrastes de oclusión. Sus alcances no certifican una
explicación causal o una campaña científica reproducida por una persona ajena.

## Reexportación científica separada

El exportador necesita un entorno separado: Torch **2.8.0+cpu**, ONNX
**1.19.1**, ONNX Runtime **1.23.2** y NumPy **2.3.5**. Esas dependencias no
forman parte del arranque del dashboard. Sus argumentos requieren:

- La cohorte auditada `current_cohort_arrays.npz`, SHA256
  `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
- La admisión histórica `runs/innovacion_v2/data_admission.json` del commit
  `1ca029e869a517cb8de704e573c78ca1e0cbad3c`, SHA256
  `7de4ccec3e415484253c3867b4451040a1e9a2fd49e4b999599cbe785f9a9d32`.
- El checkpoint E3 anterior y los dos archivos de predicciones conservados:
  `combined_predictors.json` y `e3_evaluation_predictions.csv`, admitidos por
  el ejecutor del análisis posterior de `experiments/e3_estadistica_posterior_20261117`.

```bash
python experiments/e3_dashboard_20261117/export_models.py \
  --npz work/inputs/current_cohort_arrays.npz \
  --admission work/inputs/data_admission.json \
  --e3-checkpoint work/inputs/checkpoint_271801.pt \
  --prediction-data work/e3_statistical_inputs \
  --output work/e3_dashboard_reexport \
  --report work/e3_dashboard_reexport/export_verification.json
```

La salida de reproducción queda en `work/`: las mediciones de latencia y la
fecha pueden variar entre máquinas. La tolerancia sigue siendo 10⁻⁶ y los
pesos no se seleccionan con el resultado de esa comparación. La exportación
original registró memoria PyTorch, ORT y comparación con pronósticos congelados;
su proceso incluye Torch y **no mide la RAM del servicio final**.

La oclusión reemplaza una barra diaria completa por una referencia de ajuste
con precios alineados al último cierre de la entrada. Mantiene coherencia
OHLC por barra, pero puede alterar la dinámica entre días. Los siete efectos
son contrastes independientes, no necesariamente aditivos; quedan pendientes
la fidelidad explicativa, estabilidad entre semillas y comparación con el
mejor benchmark bajo el diseño científico.
