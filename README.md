# DeepLearning_202630

Proyecto de Deep Learning, 2026-30: **pronóstico de volatilidad de Bitcoin con GRU, LSTM y TCN**.

[Jupyter Book](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [Benchmark](https://jassan-bit.github.io/DeepLearning_202630/sections/05_benchmark.html) · [Siete entregables descargables](https://jassan-bit.github.io/DeepLearning_202630/sections/08_entregables.html#descargar-entregable-1) · [Reproducción](BUILD.md).

El objetivo es σ₇, desviación estándar poblacional de siete retornos logarítmicos diarios de Close, sin anualización, con horizonte de un día. Los modelos reciben cinco canales OHLCV diarios y entradas de 7/14/21/28 días. La validación usa los índices forward chaining de `timeseries-cv` de DidierRLopes y Filipe R. Ramos: historia inicial 2023, doce bloques mensuales de 2024 y diez semillas por modelo/entrada.

El baseline neuronal vigente es **GRU/D7**, elegido por CV antes de las nuevas inferencias en el test histórico:

| Fase | RMSE media ± DE de GRU/D7 | R² medio |
|---|---|---:|
| CV 2024 | 0,00515457 ± 0,00016156 | 0,721214 |
| Test histórico 2025–agosto2026 | 0,00377511 ± 0,00019410 | 0,854173 |

La ejecución local produjo120 trayectorias CV,2.880 ajustes y43.920 predicciones. El ajuste final comprende80 redes GRU/TCN nuevas y40 LSTM idénticas reutilizadas; sus120 checkpoints preceden al acceso al derivado del test en este procedimiento. Test reúne608 fechas y72.960 predicciones:80 inferencias nuevas y40 trayectorias LSTM reutilizadas. Sus veinte meses son intervalos de reporte, sin refits. Septiembre–octubre de 2026 permanece sin asignar ni abrir.

**Alcance:** el test ya había sido observado en el proyecto. La comparación actual dentro de GRU–LSTM–TCN se elaboró después del test y es exploratoria; no constituye una confirmación temporal independiente. Persistencia conserva mejores MAE/MAPE y menor RMSE en CV. El nuevo trío no tiene aún piloto prospectivo ni un número de semillas derivado de potencia. El margen académico para modelos futuros se declara después del test y no afirma utilidad financiera.

La revisión identifica40 referencias y siete estudios cripto completos. Las frecuencias de comparadores independientes son GRU 3/7, LSTM 5/7 y TCN 1/7, con tipos de publicación y niveles de verificación diferenciados. La recurrencia TCN y la justificación de dos variantes recurrentes conservan sus límites.

El flujo público es [run_gru_lstm_tcn.py](scripts/run_gru_lstm_tcn.py), con [modelos](scripts/dl_models_gru_lstm_tcn.py), [agregación de evidencia](scripts/build_gru_lstm_tcn_evidence.py) y [diagnósticos](scripts/gru_lstm_tcn_diagnostics.py). El adaptador portable se publica después de la ejecución; no se le atribuye retroactivamente el hash del código entrenado. Los originales se preservan en `experiments/gru_lstm_tcn_20261105/original_registration/` y en el historial. Las tablas del libro corresponden únicamente a la terna vigente y persistencia.

Los archivos públicos contienen código, metadatos, agregados y figuras; una ilustración observado/predicho usa la semilla fija4101. No se redistribuyen los archivos tabulares de observaciones/predicciones individuales, los pesos o los escaladores conservados para auditoría. Construir o descargar el libro no entrena redes ni abre datos de mercado.
