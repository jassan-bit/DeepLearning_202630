# Referencia lineal regularizada: diagnóstico del bloque C

Se ajustan cuatro referencias Ridge con α=1, una por D=7/14/21/28. Cada una usa
solo muestras permitidas de 2023 y omite los D días de resultado anteriores al
primer origen de validación (01/01/2024). Los índices coinciden con el manifiesto
CV vigente. Estandarización train-only (media y desviación poblacional), intercepto
libre y objetivo sin transformar. La solución SciPy Cholesky se verifica con un
sistema aumentado de mínimos cuadrados independiente; no se instala sklearn.

Los coeficientes permanecen fijos todo 2024. Se generan 366 residuos fuera del
ajuste por D, sin recortar predicciones negativas ni seleccionar D o α. Esto no
define la política de reajuste de E2. No se calcula un ranking de desempeño, no se
entrenan redes y no se abren ZIP ni datos de test.

- `reference_configurations.csv`: cortes, muestras, hashes de índices, interceptos
  y descripciones de residuos/predicciones.
- `training_outcomes.csv`: exactamente los índices y fechas usados para ajustar.
- `model_parameters.csv`: orden de 5D covariables, media/escala de 2023 y pesos.
- `validation_predictions_residuals.csv`: etiquetas, predicciones sin recorte y
  residuos de 2024, con instante de disponibilidad y soporte de entrada.
- `residual_acf.csv`: rezagos 1–28, con referencia puntual aproximada de ruido
  blanco ±1,96/√366; no son bandas simultáneas o robustas a cualquier dependencia.
- `ljung_box_holm.csv`: Q a 7/14/28 días, `model_df=0`, referencia χ² asintótica y
  Holm para los doce contrastes. Las colas se guardan también en logaritmos;
  un underflow numérico no se presenta como probabilidad cero exacta.

El objetivo móvil comparte seis retornos entre días consecutivos. La dependencia
residual puede reflejar ese solapamiento, límites de una relación lineal o cambios
de régimen; no decide por sí sola cambiar a series de tiempo ni incorporar rezagos
del objetivo. La clasificación vigente sigue el contrato OHLCV sin esos rezagos.
Los contrastes son diagnósticos exploratorios, no inferencia confirmatoria del
benchmark. Holm no corrige la calibración de la referencia asintótica.

Reproducir desde el repositorio:

```bash
/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_temporal_reference.py
```

Fuentes y hashes, software, controles, cuatro ajustes de referencia, cero redes,
cero piloto y cero archivos de test se conservan en `summary.json`. Fuente de datos:
Binance Vision/BTCUSDT; procesamiento Jassan Arteta; derivados y figura sujetos a
CC BY-NC-SA 4.0 y condiciones aplicables del proveedor.
