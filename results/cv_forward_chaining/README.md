# Forward chaining con timeseries-cv 0.1.5

Generado por `scripts/prepare_forward_chaining.py`. Ejemplo oficial: `np.arange(27)`, input4/forecast3/jump2, ocho folds. Firma real0.1.5: `(sequence, numInputs, numOutputs, numJumps)`.

Para BTCUSDT se particionan índices del calendario diario2023–2024 y después se mapean OHLCV y volatilidad diaria por separado. Entradas7/14/21/28, horizonte1día, salto1día, etiquetas de validación2024. Se rechazan entradas/etiquetas inválidas y se verifica disponibilidad temporal de cada etiqueta de entrenamiento. El resultado tiene16configuraciones y366folds por configuración; no son observaciones independientes ni entrenamientos ejecutados. Los ZIP de test no se abren.

El resumen conserva la procedencia y SHA-256 de la figura oficial del notebook, bajo MIT, copyright2021 DidierRLopes y crédito a la coautoría de Filipe R.Ramos. El aviso completo se conserva en `book/_static/licenses/timeseries-cv-LICENSE.txt`. Fuente de datos y derivados de los manifiestos: Binance Vision/Binance Spot, procesamiento Jassan Arteta, CC BY-NC-SA4.0 y condiciones aplicables; ver capítulo2 del libro.
