# Pipeline científico separado del dashboard

El servidor importa `e3_dashboard/`, las páginas y los artefactos de inferencia.
No importa los ejecutores científicos ni utiliza este directorio al arrancar.
Los scripts históricos conservan sus rutas y hashes para mantener la
reproducción de los experimentos ya ejecutados:

- `src/e3_development.py`, `src/e3_innovation.py` y sus ejecutores en `scripts/`:
  pilotos y entrenamiento históricos, con sus planes y contadores.
- `experiments/e3_estadistica_posterior_20261117/`: análisis de predicciones
  congeladas, sin nuevos ajustes del modelo.
- `experiments/e3_dashboard_20261117/`: construcción del bundle y exportación
  de pesos ya entrenados para el servicio, con verificación de paridad.

Los entornos científicos usan sus locks CPU/CUDA específicos; no se instalan
en la imagen del dashboard. `requirements.txt` corresponde exclusivamente al
servidor CPU con ONNX Runtime. Preparar o ejecutar el dashboard no inicia otra
campaña de entrenamiento ni reinicia los presupuestos históricos.
