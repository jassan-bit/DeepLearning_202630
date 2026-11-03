# Complemento del EDA general: objetivo, calidad y máscaras

Se reutilizan únicamente cuatro archivos de `results/eda_development/`, con sus hashes:
OHLCV diario, volatilidades, resumen del EDA previo y cobertura mensual. No se abren ZIP,
no se accede a 2025–2026, no se ajusta ningún modelo y no se imputa el 24/03/2023.

Reproducción: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_general_quality.py`.

- `target_profiles.csv`: ocho perfiles (dos años por cuatro definiciones de volatilidad),
  percentiles, momentos, ceros y banderas IQR. Los umbrales se calculan solo con 2023 y
  se aplican sin reajustar a 2024; las banderas no eliminan observaciones.
- `quality_by_year.csv`: completitud de días, clave temporal, valores OHLCV repetidos,
  coherencia OHLC y disponibilidad de retornos, sobre derivados existentes.
- `label_reconstruction.csv` y `label_reconstruction_summary.csv`: reconstrucción directa
  por fechas con `math.log`, `math.fsum` y divisor poblacional, sin reutilizar rolling/std
  de pandas. Se comprueban todas las posiciones, incluidos calentamiento y huecos.
  La diferencia numérica no estima ruido de precios ni del proxy económico.
- `calendar_masks.csv`: posiciones diarias preservadas para OHLCV, retornos, etiquetas
  y ventanas OHLCV de las cuatro historias frente al objetivo principal.
- `window_eligibility.csv`: combinaciones año/D/n antes de reglas específicas de folds,
  embargo o entrenamiento; no son tamaños de muestra independientes.
- `missingness_evidence.json`: conserva el episodio de 80 minutos ausentes y un cierre
  no elegible del resumen previo; el mecanismo MCAR/MAR/MNAR permanece no identificado.
- `summary.json`: métodos, límites, hashes y controles; generación determinista.

Las figuras muestran cuantiles/cola empírica de sigma7 y máscaras de calendario 2023/24.
Fuente: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta.
Derivados y figuras: CC BY-NC-SA 4.0 y condiciones aplicables del proveedor.
