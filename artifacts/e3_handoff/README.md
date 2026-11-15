# Exportaciones iniciales de CPU para E3

MLP16, LSTM32 y GRU32 se exportaron a TorchScript desde sus pesos reales de E2,
verificados contra la congelación y el respaldo de GitHub Releases. La regla de
selección de semilla es el menor ID final registrado (271801), sin consultar su
desempeño de prueba. La receta de hiperparámetros se coteja con su recibo de
selección en validación.

Cada carpeta contiene `model.torchscript.pt`, `metadata.json` y
`roundtrip_report.json`. La ejecución usó Python 3.12 y Torch 2.8.0+cpu; las
versiones concretas y los hashes de las fuentes heredadas están en los reportes.

La verificación utiliza las 53 fechas históricas de calibración de E2,
08/11–30/12/2024, reconstruidas de los CSV públicos de desarrollo. Usa las
medias, desviaciones y escala objetivo guardadas; no ajusta pesos, escaladores ni
calibradores. El error máximo de predicción y la diferencia de RMSE entre
memoria y exportación fueron cero en los tres modelos, con tolerancia absoluta
declarada de 10⁻⁶ en unidades de σ₇. Esta es una comprobación numérica sobre
calibración conocida; no verifica aún la métrica de la prueba A2 de 608 fechas ni
aporta confirmación independiente de E3.

## Inferencia

Los artefactos reciben `(batch, 7, 5)` de `float32` ya transformados y
estandarizados según el metadata, ordenados de antiguo a reciente. Devuelven
σ₇ dividido por `target_scale`; la aplicación invierte esa escala. Las
validaciones Python del modelo original no se conservan en el trazado. Debe
usarse la guarda externa del exportador antes de inferir:

```python
from scripts.e3_export_readiness import predict_exported_reference

# windows es una lista de ventanas D7×5 ya transformadas y estandarizadas.
predictions_sigma7 = predict_exported_reference(
    "artifacts/e3_handoff/MLP16", windows
)
```

La función verifica hash del artefacto, forma, valores finitos y escala. No
transforma OHLCV crudo, no detecta por sí sola datos fuera de dominio y no
entrena. Los rangos para aviso fuera de dominio siguen pendientes.

## Perfil de CPU

Cada modelo se midió en un proceso separado con un hilo Torch, lote uno, diez
llamadas de calentamiento y cien medidas. Las latencias son del forward con
entrada ya preprocesada; excluyen lectura de archivos y transformación OHLCV.
La memoria reportada es el pico RSS del proceso completo, con intérprete,
runtime, pesos y cohorte; no es RAM incremental del modelo ni memoria de
Render. Los FLOPs no se estiman en este ejecutor.

Los `selected_model.pt` de origen contienen pesos, escalador y arquitectura, pero
no estado del optimizador en ese archivo. La disponibilidad del checkpoint
completo de entrenamiento debe verificarse por separado en los archivos de
fase; no se deduce de esta exportación.
