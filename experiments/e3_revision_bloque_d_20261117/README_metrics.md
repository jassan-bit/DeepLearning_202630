# Suplemento descriptivo de métricas: 17 de noviembre de 2026

Este suplemento posterior usa únicamente las predicciones congeladas de los
17 sistemas sobre las mismas 608 fechas conocidas. No carga pesos, entrena,
repite inferencia, recalibra ni calcula nuevos contrastes de hipótesis.
El código reutiliza la admisión de once archivos inmutables del análisis
estadístico anterior, incluyendo fechas, objetivos, 133 predictores E2 y diez
predictores E3. La semilla histórica 20261116 permanece fuera de la media E3.

## Qué aporta

- `metrics_extended.csv`: tabla numérica completa con RMSE media individual,
  DE muestral entre semillas, MAE, sMAPE, R², cobertura y anchura disponibles.
- `metrics_extended_table.csv`: presentación de ocho columnas con redondeo
  explícito para el libro; se ordena por la RMSE ya obtenida.
- `metrics_uncertainty.csv`: IC95 % bootstrap de la media de RMSE entre
  inicializaciones. Usa 10000 remuestras y semilla 20261117, con los mismos
  índices para los catorce sistemas neuronales. Reproduce exactamente el IC
  heredado de E3. Los clásicos con una trayectoria no reciben un IC de semillas.
- `metrics_per_predictor.csv`: las métricas de las 143 trayectorias que sustentan
  los agregados, sin promediar primero las predicciones para formar un ensamble.
- `metrics_supplement.json`: definiciones, SHA256, ambiente, alcance y controles.

sMAPE se expresa en porcentaje como
`200*mean(abs(y-p)/(abs(y)+abs(p)))`, con cero para el cociente 0/0. R² se calcula
por predictor como `1-SSE/SST`, usando la media del mismo objetivo común, y luego
se promedia entre predictores. Con objetivo constante R² sería indefinido: el
suplemento no aplica la convención de reemplazarlo por uno o cero. La cohorte
real tiene variación y se comprueba que su R² está definido.

## Intervalos heredados y campos ausentes

Cada predictor de E2 conserva un `calibrator_q` y el soporte de las 53 fechas
de calibración. Se reconstruye su intervalo ya definido como
`[max(0, prediction-q), prediction+q]`, sin estimar otra vez q. Se promedian por
predictor la cobertura empírica y la anchura en las 608 fechas; después se
promedian esas métricas por sistema. El truncamiento inferior importa para la
anchura. Como control, las mismas fórmulas reproducen la tabla histórica de seis
semillas, pero esa tabla no se usa como sustituto de los agregados actuales de
diez semillas.

E3 no utilizó el bloque de calibración y no conserva un q ni intervalos
predictivos congelados. Sus columnas de cobertura y anchura permanecen vacías
en los datos numéricos y aparecen como `Pendiente` en la tabla de lectura.
El IC de la media entre semillas no se convierte en un intervalo predictivo.

Los IC son marginales y condicionados al calendario conocido, sin cobertura
simultánea, temporal o predictiva. Las coberturas de los intervalos E2 son
descriptivas: el nominal 90 % no es garantía bajo dependencia temporal o cambio
de distribución. E2 conserva 655 ejemplos de ajuste y 53 de calibración; E3,
532 de ajuste y 116 de validación, sin calibración. El suplemento no corrige esa
diferencia ni acredita potencia, adecuación o R3.

## Reproducción

Desde la raíz del repositorio, con NumPy 2.3.5 y Python 3.12:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python experiments/e3_revision_bloque_d_20261117/metrics_supplement.py \
  --data-dir work/e3_statistical_inputs --download --check \
  --mirror book_e3/_static/e3_revision_bloque_d_20261117

python -m unittest discover -s experiments/e3_revision_bloque_d_20261117 \
  -p test_metrics_supplement.py -v
```

`--check` recalcula los agregados y compara sus bytes con los CSV guardados, sus
metadatos estables y sus copias publicadas. También comprueba un control negativo
de alineación: una copia en memoria con una fecha E3 alterada debe rechazarse.
Las nueve pruebas unitarias usan ejemplos calculados a mano, límites de
cobertura/anchura, la diferencia entre media de RMSE y RMSE de un ensamble,
cuantiles bootstrap, entradas incompatibles y rechazo de alteración de un hash.
Las fuentes originales no se modifican.
