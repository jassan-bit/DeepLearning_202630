# Evidencias del EDA vigente

Este expediente acompaña el capítulo 2 de E2. Separa el EDA de desarrollo heredado, la comprobación del calendario ejecutado y los perfiles descriptivos posteriores. No modifica las recetas, los pesos, la calibración, los resultados del benchmark ni los archivos históricos.

## Contenido y alcance

| Archivo | Qué permite comprobar |
|---|---|
| `partition_audit.json` | Tres folds HPO con 253/435/527 ejemplos de entrenamiento y 83 de validación por fold; entrenamiento final de 655, calibración de 53 y evaluación de 608 fechas. Identidades, máscaras, firmas y soportes temporales, con límites explícitos de la comprobación sobre registros guardados. |
| `target_profiles.csv` | Nueve perfiles del objetivo σ₇: entrenamiento/validación de cada fold y las tres particiones finales. Media, dispersión, cuantiles, asimetría, curtosis y distancias descriptivas con la referencia de entrenamiento correspondiente. |
| `target_ecdf.svg` | Distribuciones acumuladas empíricas de las 655/53/608 etiquetas finales en unidades originales. |
| `summary.json` | Métodos, versiones de dependencias, hashes de fuentes y salidas, accesos y límites del cálculo descriptivo. |
| `inherited_annual_profiles.csv` | Copia exacta del contexto anual de E1. OHLCV del mismo día y otras variables de 2023/2024/2025/enero–agosto de 2026; referencia de escala hasta el 03/12/2024. No representa el tensor causal de 35 componentes ni usa el corte final de octubre como referencia. |
| `decision_traceability.csv` | Hallazgos, evidencias, decisiones de preprocesamiento, impacto en arquitectura y validación, riesgos y alcance por requisito. |
| `guide_checklist.json` y `.csv` | Cotejo de EDA general y bloques B/C de la guía; distingue evidencia disponible, no aplicabilidad y limitaciones. |
| `manifest.json` | Tamaños y SHA-256 de este expediente y los dos scripts de reproducción. |

La partición es cronológica, sin semilla de barajado. Los IDs finales de entrenamiento, calibración y evaluación son disjuntos. Calibración y evaluación comparten seis cierres pasados y cinco retornos por el avance de las ventanas: no se afirma independencia ni intersección nula de todos los soportes. Los diez folds mensuales y la reserva histórica de 60 días se conservan como antecedentes distintos del calendario actual.

Los perfiles actuales leen las etiquetas ya guardadas de las 608 fechas conocidas desde E1. Son posteriores a la evaluación y no cambian la selección de modelos. KS se calcula únicamente como distancia entre distribuciones empíricas, sin p-valores. Las ventanas y los entrenamientos expansivos no son muestras independientes.

La reconstrucción de σ₇ verifica consistencia numérica. No identifica el ruido económico externo del proxy ni un techo de desempeño. Los perfiles anuales heredados tampoco sustituyen una auditoría nueva de las covariables causales del test.

## Reproducción

Desde la raíz de un checkout completo del repositorio, utilizar Python 3.12 con NumPy 2.3.5, Pandas 2.2.3, SciPy 1.16.3 y Matplotlib 3.10.8; `summary.json` conserva las versiones utilizadas. Los scripts son:

```bash
python scripts/audit_e2_eda_partition.py --output /tmp/e2-eda-reproduccion/partition_audit.json
python scripts/e2_eda_current_profiles.py --output /tmp/e2-eda-perfiles-reproduccion
```

Elegir rutas de salida nuevas. La auditoría rechaza sobrescribir un archivo existente. El generador descriptivo acepta salidas existentes idénticas y rechaza cambios. Los resultados no contienen rutas privadas del entorno; el SVG omite fecha y fija la semilla de sus identificadores gráficos. `inherited_annual_profiles.csv` debe coincidir byte por byte con `book/_static/eda_dataset_context_20261105/distribution_by_period.csv`.

Las fuentes están en el checkout: `results/eda_development/`, el protocolo reducido y los artefactos finales ya exportados bajo `book_e2/_static/`. Sus hashes figuran en los JSON. La auditoría usa el guardián de vectores guardados de `scripts/e2_temporal_final_audit.py`, sin ejecutar modelos. Los dos comandos no abren ZIP brutos, no ajustan escaladores o modelos, no ejecutan inferencia ni recalculan puntuaciones predictivas. No requieren Colab ni GPU.
