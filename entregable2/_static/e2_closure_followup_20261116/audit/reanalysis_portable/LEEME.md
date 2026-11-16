# Reanálisis independiente de controles negativos

Este paquete recalcula el seguimiento de seis controles negativos con sus
pronósticos reales guardados. Incluye todos los insumos necesarios y no necesita
el repositorio original, datos de Binance, pesos, PyTorch, GPU ni acceso a red
durante el reanálisis. La instalación inicial de dependencias sí puede requerir
Internet.

Use Python 3.11 o posterior. En una carpeta de trabajo:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell, en lugar de la línea anterior:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --output resultados
```

La ejecución predeterminada verifica el manifiesto SHA256 y recalcula ganancias
de RMSE, correlaciones de Spearman, intervalos por bloques de calendario,
TOST aproximado y ajuste de Holm. Usa exactamente 10.000 remuestreos, bloques de
14 días y la semilla del plan incluido. Compara todos los campos del resumen
con la referencia y escribe `summary_recomputed.json` y `verification.json`.
La carpeta de salida debe ser nueva. Una diferencia numérica mayor que 1e-12
produce un error; se informa también la máxima diferencia realmente observada.

Para verificar solamente los archivos, sin importar NumPy o SciPy:

```sh
python run.py --verify-only
```

`analysis_plan.json` es la copia exacta del plan congelado antes de los seis
ajustes. Sus rutas históricas sirven como procedencia y no se resuelven para
ejecutar este paquete. `training_reference.json` conserva la constante exacta de
la media del objetivo de entrenamiento, su fuente y hash. Los seis NPZ incluyen
fechas, objetivo, predicción y permutación original de las etiquetas.

El paquete no vuelve a entrenar ni ejecuta modelos. Reproduce análisis de
pronósticos guardados y no reconstruye toda la campaña histórica. Los intervalos
están condicionados a las tres funciones fijas nuevas por modelo y al calendario
observado; no identifican toda la variabilidad de permutaciones o de selección
de hiperparámetros.

La referencia conserva el resultado adverso: ninguna de las dos medias fijas
estableció equivalencia de RMSE con la constante dentro del delta registrado.
El reanálisis no certifica ausencia de fuga ni cambia la aceptación pendiente
de la auditoría integral.
