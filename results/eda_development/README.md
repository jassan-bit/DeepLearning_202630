# EDA de desarrollo BTCUSDT

Generado por `scripts/eda_binance_development.py` desde 24 ZIP mensuales oficiales 2023–2024 con verificación SHA-256. No se abren ZIP2025–2026. `summary.json` registra métodos, periodos y resultados; CSV incluyen OHLCV diario, volatilidades candidatas, pruebas y correlaciones. Las figuras PNG están en `book/_static/eda/`.

Días incompletos no se interpolan. Los valores diarios OHLCV de un día son observables solo al finalizarlo. Cada sigma está etiquetada por el día de resultado; sus predictores deben terminar el día anterior. Las ventanas7/14/21/28 son definiciones distintas, no modelos ganadores. PCA se ajusta solo en2023 y se proyecta2024. Diagnósticos exploratorios no implican capacidad predictiva ni equivalen a pruebas sobre residuos de redes.

Fuente: Binance Vision, Binance Spot BTCUSDT, https://data.binance.vision/ ; procesamiento: Jassan Arteta. Derivados de datos y figuras bajo CC BY-NC-SA4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/ y condiciones aplicables Binance Vision Dataset Terms. Se seleccionan días completos, agreganOHLCV, calculan retornos y estadísticas; los ZIP originales no cambian.
