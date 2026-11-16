# Orquestación trazable de la reproducción analítica E2

`python scripts/reproduce_e2_campaign.py --check` funciona con Python estándar,
sin Git, red o ZIP presentes. Comprueba los bytes del catálogo y de las pruebas
incluidas, sondea los intérpretes indicados y enumera los doce ZIP esperados con
sus hashes. La salida predeterminada es una carpeta nueva bajo `work/`; nunca
sobrescribe recibos publicados. El código de salida **2** significa parcial y es
el resultado esperado mientras falten entrenamiento o puertas obligatorias.
El código **1** indica fallo o evidencia alterada; **0** sólo puede describir un
alcance analítico completo y nunca certifica toda la campaña original.

## Repetición local

Descargar los doce ZIP v2 completos en una carpeta y ejecutar:

```sh
python scripts/reproduce_e2_campaign.py --replay --archives-dir RUTA_ZIP --output NUEVA_SALIDA --python PYTHON_CORRECCIONES --analysis-python PYTHON_PRIMARIA --model-python PYTHON_CPU --book-python PYTHON_JUPYTERBOOK --cap-seconds 600
```

El script reserva y cobra su reloj de pared en el diario aditivo existente de la
campaña ya autorizada de 84 horas. No necesita otra envoltura del presupuesto,
no lo reinicia y no inicia ajustes. `--replay` requiere el diario existente
`book_e2/_static/e2_pending_closure_20261116/compute_ledger.jsonl` y su ancla histórica;
`--check` no necesita ese diario. No instala dependencias automáticamente.

Correcciones: NumPy2.3.5/SciPy1.17.0. Primaria e incertidumbre originales:
NumPy2.3.5/pandas2.2.3/SciPy1.16.3/statsmodels0.14.5/Matplotlib3.10.8.
HPO: NumPy2.3.5/Matplotlib3.10.8. Forward fixture: Torch2.8.0+cpu,
NumPy2.3.5/pandas2.2.3/einops0.8.1/reformer-pytorch1.4.4.
Libro: JupyterBook1.0.4.post1 y su lock incluido en D01.

Las 18 etapas distinguen extracción/reconstrucción del libro, forwards con
fixtures, validación de informes guardados, reconstrucción aritmética de
métricas/datos/techo/paper/presentación y replays numéricos de primaria,
calibración/errores, efectos/RQ2, diagnósticoCPU y metadatos HPO. Cada comando,
retorno, tiempo y hash del log queda registrado. Las etapas bloqueadas no se
marcan ejecutadas. Los análisis que vuelven a calcular números no se confunden
con los informes que sólo validan evidencia guardada.

No ofrece un `--full-training`: faltan un ejecutor integral validado y puertas
necesarias. No repite los 78 ajustes originales, las tres reconstrucciones
clásicas ni las 18 ablaciones. Las pruebas reales de fuga pendientes y la puerta
CUDA original siguen separadas. El argumento algebraico RQ3v2 se conserva como
prueba lógica de un componente CPU; no estima una nueva p ni ejecuta la IUT.

Los costes HPO conservan tres ámbitos solapados: fases medidas, cargos de la
campaña y ajustes históricos reutilizados. No se suman entre sí. La asignación
conservadora de1860s desconocida no es tiempo de entrenamiento medido; energía,
dinero y coste integral histórico desconocidos permanecen vacíos.

## Archivos necesarios para un `--check` autónomo

Mantener el layout del suplemento: `scripts/reproduce_e2_campaign.py`,
`book_e2/_static/e2_pending_closure_20261116/reproduction_inputs.json` y estos
archivos, enumerados también con tamaño/SHA256 en el catálogo:

- `scripts/e2_corrections_budget.py`
- `book_e2/_static/e2_reproduction_ablations_20261113/budget_final.json`
- `book_e2/_static/e2_corrections_20261115/hpo/summary.json`
- `book_e2/_static/e2_corrections_20261115/data_audit/audit_result.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/summary.json`
- `book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component/v2/necessary_component_bounds_10.csv`
- `book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component/v2/README.md`
- `book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component/v2/necessary_component_proof.json`
- `book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component/v2/execution_receipt.json`
- `book_e2/_static/e2_pending_closure_20261116/rq3_necessary_component/v2/manifest.json`
- `scripts/e2_rq3_necessary_component.py`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/CPU_cost_only_Holm10.csv`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/manifest.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/results/execution_receipt.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/statistics/statistical_plan_v3.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/plan_v3.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/latency/latency_report.json`
- `book_e2/_static/e2_corrections_20261115/cpu_diagnostic/latency/raw_cpu_latency.csv`

No hacen falta los doce ZIP para ese modo: su ausencia se informa como bloqueo
de replay. El suplemento conserva la fuente v1 realmente ejecutada bajo
`replay_receipts/v1/`; las correcciones posteriores de CLI se validan con tests
y comprobación de dependencias, sin atribuirles otra ejecución estadística.

El diario activo conserva íntegramente la cadena previa de10591bytes; la copia histórica de8763bytes de `e2_corrections_20261115/compute_ledger.jsonl` conserva los bytes publicados en532a2bb y no vuelve a recibir cargos. La migración mantiene el mismo ancla y todos los cargos/reservas, sin reiniciar presupuesto. El suplemento C3 conserva su snapshot coherente anterior y no se modifica.
