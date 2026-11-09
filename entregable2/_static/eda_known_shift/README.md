# Distribuciones de periodos conocidos de E1 · EDA E2

Análisis descriptivo posterior a los resultados de E1. No entrena ni puntúa
modelos, no cambia fechas, inputs, objetivo, métricas o margen. Se publican solo
agregados y figuras. No incluye vectores de precios ni predicciones individuales.

Desde la raíz del repositorio, Python 3.12 con numpy, pandas, scipy y matplotlib:

```bash
python scripts/eda_e2_known_shift.py --known-market-csv /ruta/daily_market_reserved.csv
```

La opción anterior exige el SHA-256 del derivado publicado en el audit de E1.
Para reconstruir desde la adquisición original sin entrenar ningún modelo:

```bash
python scripts/eda_e2_known_shift.py --raw-root /ruta/adquisicion/oficial --rebuild-cache /tmp/e2_known_market_nuevo
```

El directorio privado de caché debe ser nuevo y estar fuera del repositorio.
El script verifica los 32 ZIP ya conocidos (agosto como contexto, septiembre
mensual y treinta diarios de octubre) contra el manifiesto fijado de E1 y usa
la rutina de agregación existente. No llama al ejecutor del benchmark ni a su
control de primer acceso. No modifica los originales. Diferencias de versión
en la serialización del CSV pueden cambiar el hash derivado reconstruido; se
registran el hash real y los hashes originales de cada ZIP.

La referencia de distancias es el entrenamiento inicial D7 admisible hasta
24/12/2023 (343 ejemplos). Las tablas de periodos presentan su calendario entero,
incluido 2023 (350 ejemplos); esto NO ajusta escaladores de modelos en validación.
KS es distancia, sin p-valor iid; Wasserstein se divide por DE poblacional de
la referencia. El componente de último cierre relativo es constante cero y no
se divide por su desviación cero. No hay afirmación de nuevos datos sellados,
significación estadística, utilidad financiera o mejor desempeño de un modelo.

Fuente: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta. CC BY-NC-SA 4.0
y condiciones adicionales de la fuente documentadas en la Data Card de E2.
