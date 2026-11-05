# Ampliación EDA E2 orientada a decisiones

Ejecución desde la raíz: `python scripts/eda_e2_decisions.py`.
Requiere Python y NumPy, pandas, SciPy y Matplotlib; las versiones observadas se
registran en provenance.json. Los dos CSV de desarrollo 2023–2024 y los índices
archivados de E1 se conservan íntegros. No descarga datos, abre ZIPs ni lee
observaciones del test. Los archivos heredados se copian byte por byte desde E1.
Para restaurar las dependencias mínimas observadas en un entorno aislado:
`python -m pip install -r results/e2_eda_decisions_20261105/requirements.txt`.

- Transformaciones: lambda ajustada solo con los 343 outcomes iniciales permitidos
  hasta el 24/12/2023. Perfiles de todos los datos válidos de cada año son
  descriptivos; no se adopta una transformación del objetivo o pérdida nueva.
- Ventanas: D7, log OHLC relativo al último cierre observable y log1p volumen;
  escalas por cinco canales ajustadas en los 343 train iniciales (N×D posiciones).
  Distancia RMS euclídea en 35 valores, umbral fijo 0,01. Similitud no prueba
  copia ni autoriza borrar patrones; controles por fecha/composición son aparte.
  El cero numérico de distancia relativa usa tolerancia 1e-14; las coincidencias
  de ventanas OHLCV brutas se comprueban por igualdad de todos sus valores.
  También se publica sensibilidad a umbrales 0,01/0,05/0,10/0,25/0,50 sin
  seleccionar un corte para eliminar ejemplos. Se separan train/train,
  validación/train y validación/validación con el mismo escalado de train.
- Folds: los diez bloques mensuales Jan–Oct2024 proceden de índices diarios de
  timeseries-cv agrupados en E1. El embargo es u<=origen−8. Fechas ya validadas
  pueden entrar al train de meses posteriores, pero los folds no son réplicas
  independientes. El CSV histórico llama test_indices a validación de desarrollo.
- Ridge alpha1: una referencia diagnóstica ajustada por mes sin HPO, con escalas
  solo del train admitido e intercepto no penalizado. Se contrasta numéricamente
  con otra solución por mínimos cuadrados aumentados. No es uno de los diez
  modelos E2 seleccionados y no altera las recetas GRU/LSTM/TCN heredadas.
- ACF: centrada y normalizada por suma de cuadrados, con calendario íntegro.
  Ljung–Box: referencia chi-cuadrado df=rezago (sin parámetros ARMA estimados),
  con Holm sobre los 66 diagnósticos publicados. P-valores aproximados y
  exploratorios: muestras mensuales cortas, solapamiento y cambios de ajuste
  limitan calibración. Se conservan log-p para evitar ceros por underflow.

Los CSV de residuos de desarrollo son derivados, no cotizaciones brutas.
Datos: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta.
Datos y derivados sujetos a CC BY-NC-SA 4.0 y términos de la fuente conservados
en E1; no se atribuye esta licencia automáticamente al código del repositorio.
