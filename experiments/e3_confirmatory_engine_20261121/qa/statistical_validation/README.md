# Evidencia estadística sintética

La especificación vigente es **V4**, en `v4/plan.json`, `v4/freeze.json` y
`v4/report.json`. Conserva las referencias fixed-b escalar/vectorial y el
bootstrap-t/HAC de V3. Endurece el alpha operacional a 0,025 con nuevas 512×5
réplicas independientes y 2000 CPU s, sin relajar los criterios CP. Los CI99
operacionales y CI95 diagnósticos conservan sus nombres y cobertura reales.
Las 21 pruebas deterministas pasan; no equivalen a calibración estocástica.
El código/plan quedaron públicos antes de ejecutar V4, en commit
`70f0838cdf8809a469ea1ca837e5011faff551e2`.

Reproducción vigente (requiere directorio nuevo):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python qa/statistical_validation/v4/run_validation.py --output-dir /tmp/temporal-v4-reproduction
```

V3 tiene un inventario completo de 960 réplicas y dictamen inconcluso. Tras el
cambio de entorno se reconstruyó únicamente su informe, sin repetir réplicas;
`v3/RECOVERY_RECEIPT.json` conserva hashes y el consumo original desconocido.
Su fuente y pruebas originales están en `v3/statistics_frozen.py` y
`v3/tests_frozen.py`. Restaurarlas en una copia temporal permite reproducir
el runner V3 sin alterar la versión actual. Los archivos de réplicas se
publican como gzip determinista con SHA del comprimido y del JSONL original.

Los archivos y cifras descritos a continuación son **registro histórico
v1/v2**, no la calibración operativa V4. `statistics_v2.py` preserva además la
fuente exacta de v2 con su hash original. El runner v2 usa esa fuente en una
copia temporal para reproducción; no debe ejecutarse encima de v3.

**Dictamen histórico v2: FAILED_FINITE_SAMPLE_VALIDATION. No admitida.**

Los resultados v1 y v2 se conservaron con planes, hashes de fuente, salidas por
réplica e intervalos Clopper–Pearson exactos. No se entrenaron modelos de
mercado ni se abrió la prueba final. El presupuesto combinado consumió
168.46 segundos de CPU; la segunda ejecución respetó su límite de 240 s.

- `plan_v1.json`, `freeze_v1.json`, `statistics_v1.py`, `runner_v1.py`:
  protocolo y fuente originales, fijados antes de la primera simulación.
- `report_v1.json`, `replicates_v1.jsonl`: fallo de FWER/cobertura a T192.
- `plan_v2.json`, `freeze_v2.json`: nuevo plan T1536, HAC48/bloque64, mismas
  tolerancias, corrección de invariancia numérica de unidades.
- `run_validation.py`: runner de la v2; verifica sus tres hashes y se niega a
  sobrescribir resultados existentes.
- `report_v2.json`, `replicates_v2.jsonl`, `verdict_v2.json`: persiste un fallo
  de FWER en la frontera práctica y fallos de cobertura. Otros límites son
  inconclusos; la alternativa grande tiene potencia alta pero no acredita la
  calibración nula.
- `deterministic_results_v2.json`: 14 pruebas de corrección pasan.
- `environment_versions.json`: versiones realmente utilizadas.

El informe conserva los p-valores agregados mediante sus eventos familiares;
la API `analyze` conserva todos los p crudos y ajustados, estadísticos,
direcciones e intervalos para una evaluación real. Los JSONL sintéticos
conservan los eventos y CIs de cada réplica especificados antes de ejecutar,
no simulan un fichero de predicciones de mercado.

Reproducción v2 desde la raíz del paquete:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_statistics.py -v
OPENBLAS_NUM_THREADS=1 python qa/statistical_validation/run_validation.py --output-dir /tmp/temporal-v2-reproduction
```

Para v1, reconstruir una copia temporal con `statistics_v1.py` como
`engine/statistics.py` y `runner_v1.py` como
`qa/statistical_validation/run_validation.py`; copiar plan/freeze v1 al mismo
directorio relativo. El runner original comprobará los hashes originales.
No se restaura v1 sobre el código vigente ni se borran los resultados adversos.

La explicación de hipótesis, estimando, independencia, solapamiento, semillas,
multiplicidad y límites se encuentra en `docs/TEMPORAL_BATTERY.md`.
