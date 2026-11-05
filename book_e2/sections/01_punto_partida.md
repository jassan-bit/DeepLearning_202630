# 1. Punto de partida desde E1

El antecedente es el [Entregable 1](https://jassan-bit.github.io/DeepLearning_202630/intro.html), cuya versión de referencia queda fijada al [commit `94a4b03`](https://github.com/jassan-bit/DeepLearning_202630/tree/94a4b0324e93361e8928d708fdfeb50110e3686f). Los [metadatos de referencia](../_static/e1_reference.json) conservan procedencia y hashes; no contienen observaciones individuales ni pesos.

E1 estudió regresión temporal de volatilidad de Bitcoin con OHLCV diario de Binance Spot BTCUSDT, objetivo σ₇ poblacional de retornos logarítmicos y horizonte de un día.

| Antecedente | Evidencia de E1 |
|---|---|
| Referencia seleccionada con desarrollo | GRU/D7-c03, 16 unidades, 1.121 parámetros, LR 0,003 y dropout 0,1 |
| Menor RMSE observada en su reserva | Persistencia: 0.00384478 |
| Mejor red observada en esa reserva | TCN: 0.00433967 |
| GRU registrado | RMSE 0.00511696; su selección no se cambió por el ranking del test |
| Alcance | Una reserva histórica de 60 días; las semillas no representan mercados independientes |

Las fechas del 01/09 al 30/10/2026 **ya fueron evaluadas y publicadas**. No se presentarán como un test desconocido de E2. La GRU de 32 unidades de la fase inicial y la GRU-c03 de 16 unidades son recetas distintas. La versión de referencia que adopte E2 deberá registrarse explícitamente.
