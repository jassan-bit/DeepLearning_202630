# Curvas reproducidas de las recetas finales D7

Se ejecutaron de nuevo las tres recetas congeladas, con 465 semillas por familia, sólo desarrollo 2023–2024 y código histórico intacto. Esta es una reproducción posterior; no se recuperaron los logs privados originales. Se publicaron protocolo, código y semillas antes de esta nueva ejecución.

Fuente inmutable publicada: https://github.com/jassan-bit/DeepLearning_202630/commit/fe41b8ad1eaa1425aad900f6c3cd966843d600c7.

Instalación del entorno con el paquete CPU fijado en el lock:

```bash
python -m pip install -r requirements-e1-curves.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

La comparación con el manifiesto histórico encuentra 0 de 2790 CSV de curvas idénticos byte a byte. La tabla de hashes conserva todas las diferencias y las coincidencias; no se sustituyen semillas ni recetas según ese resultado.

Las figuras agregadas muestran media y rango intercuartil. Cada época incluye sólo trayectorias que la alcanzaron; consultar n_trajectories. Las bandas describen variación por inicialización sobre el mismo calendario y no son IC de generalización temporal. La semilla 8101 se fijó para la figura individual antes del ajuste.

La selección interna tiene pérdidas train/validación; el reajuste conserva sólo train durante el número cerrado de épocas. La ausencia de validación en el reajuste es intencional. La escala del objetivo se ajusta por separado con los datos permitidos de cada etapa; no comparar directamente los valores entre inner y refit.

Este producto no consulta la reserva de 2026, no genera pronósticos nuevos y no cambia el baseline, δ o los resultados originales. Pesos, observaciones, predicciones y valores de escaladores quedan fuera de la publicación.
