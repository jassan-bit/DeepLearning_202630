# E2: revisión de diez originales completos

Se leyeron diez PDFs completos, incluidas referencias y apéndices: 215 páginas físicas. Los recibos fijan versión, SHA256 y tamaño. Las fuentes completas y el texto extraído no se redistribuyen.

La disponibilidad de diez implementaciones de autor se documentó antes en un registro de código. Su estado histórico de cero PDFs pertenece a esa etapa. El estado actual es diez originales completos verificados. En esta revisión hubo cero ajustes de modelos, cero reproducciones numéricas, cero resultados BTC y cero descargas de benchmarks validadas. Describir un dataset en un artículo no valida una descarga.

Cada par siguiente pertenece a una misma fila y variante. Hay diferencias de horizonte, contexto, split y escalado: estas cifras externas no ordenan modelos para BTC σ7 ni para D7×5,h1.

| Modelo / variante | Dataset | H | L | MSE | MAE | Tabla; página PDF / impresa | Versión |
|---|---|---:|---:|---:|---:|---|---|
| TSMixer: TSMixer básico multivariante | ETTh1 | 96 | 512 | 0.361 | 0.392 | 3; 10 / 10 | 2303.06053v5 |
| TiDE: TiDE, Table2; ETTh1 settings Table8 including RevIN=True, all residuals present; NOT TiDE(no res.) | ETTh1 | 96 | 720 | 0.375 | 0.398 | 2; 8 / 8 | 2304.08424v5 |
| TimeMixer: TimeMixer oficial, configuración unificada | ETTh1 | 96 | 96 | 0.375 | 0.400 | 13; 17 / NR | 2405.14616v1 |
| PatchTST: PatchTST/64 supervisado | ETTh1 | 96 | 512 | 0.370 | 0.400 | 3; 7 / NR | 2211.14730v2 |
| Informer: Informer con ProbSparse y distilling; columna Informer, no Informer†. | ETTh1 | 24 | NR | 0.577 | 0.549 | 2; 6 / NR | 2012.07436v3 |
| Autoformer: Autoformer speedup (todos los resultados del paper), 2 encoder + 1 decoder, d_model512, 8 heads, moving average25; c exacto por fila NReportado. | ETTh1 | 24 | 96 | 0.384 | 0.425 | 5; 13 / 13 | 2106.13008v5 |
| TimesNet: TimesNet (Ours), backbone Inception compartido final | ETTh1 | 96 | 96 | 0.384 | 0.402 | 13; 21 / 21 | 2210.02186v3 |
| S-Mamba: Simple-Mamba (S-Mamba): bi-Mamba VC + FFN TD; columna S-Mamba principal | ETTh1 | 96 | 96 | 0.386 | 0.405 | 3; 7 / 7 | 2403.11144v3 |
| iTransformer: Original complete iTransformer backbone, iTransformer (Ours) column | ETTh1 | 96 | 96 | 0.386 | 0.405 | 10; 23 / NR | 2310.06625v4 |
| Crossformer: Original full Crossformer (DSW+TSA+HED) without covariates | ETTh1 | 24 | NR | 0.305 | 0.367 | 1; 7 / NR | ICLR 2023 conference PDF |

## Calidad de reporte para la fila seleccionada

NR significa no reportado tras leer la versión completa; no equivale a cero. NV significa no verificado. Una DE de otros horizontes, datasets o variantes no se traslada a la fila seleccionada. Los campos completos y sus evidencias están en models.json y quality_matrix.csv.

| Modelo | Split | Escalado | Scaler train-only | Repeticiones | Semillas | Incertidumbre de esta fila | Test formal | Presupuesto HPO igual |
|---|---|---|---|---|---|---|---|---|
| TSMixer | reported | reported | NR | NR | NR | NR | NR | NR |
| TiDE | reported | reported | reported | reported: 5 | NR | reported | NR | NR |
| TimeMixer | reported | NR | NR | reported: 3 | NR | NR | partial | NR |
| PatchTST | NR | partial | NR | reported: 1 | reported: [2021] | NR | NR | NR |
| Informer | reported | reported | NR | reported: 5 | NR | NR | NR | NR |
| Autoformer | reported | NR | NR | reported: 3 | NR | NR | NR | NR |
| TimesNet | reported | partial | NR | reported: 3 | NR | NR | NR | NR |
| S-Mamba | NR | partial | NR | NR | NR | NR | NR | NR |
| iTransformer | reported | partial | NR | NV | NR | NR | NR | NR |
| Crossformer | reported | reported | reported | reported: 5 | NR | NR | NR | NR |

## Controles críticos de interpretación

- Informer y Crossformer: el original ofrece ETTh1 H24; no atribuirle ETTh1 H96. El contexto ganador de esa fila es NR.
- Autoformer: ETTh1 H24 aparece en Tabla 5, PDF p13; ETT* H96 del cuerpo es ETTm2. Su operador Roll es circular.
- TimeMixer: el par canónico es la configuración unificada L96, M=3, cuatro escalas; la búsqueda optimizada es otra condición. La Tabla 6 imprime 15min para ETTh1/h2: conservar la inconsistencia para verificarla, sin corregirla por costumbre.
- PatchTST: /64 supervisado L512/P16/S8 se distingue de /42, pretraining, fine-tuning y linear probing.
- TiDE: el artículo declara 7:1:2 para todos los datasets long-term, incluido ETT; no sustituirlo por otro split conocido. Sus errores estándar no son desviaciones estándar.
- TSMixer: Table 2 imprime 7:2:1 para Weather/Electricity/Traffic; Table 8 contiene rangos/columnas incongruentes. Son anomalías de reporte observadas; no se infiere una conducta impropia.
- TimesNet: la fórmula del paper usa ceil(T/f), mientras el código inspeccionado usa floor(T/f); mantener distintas las fuentes. El código selecciona periodos compartidos por batch: para combinar distintos orígenes, la adaptación propuesta exige inferencia B1 o selección independiente por origen, con top-k sólo sobre frecuencias positivas y k≤floor(L_eff/2). Es un riesgo del adaptador, no una acusación sobre experimentos publicados. Costes de imputation no son costes del forecast seleccionado.
- S-Mamba: versión arXiv v3 de 2024, 14 páginas físicas; no es el cuerpo editorial de 2025. Optimizador, splits y semillas son NR. Costes de Fig5 llevan su propio batch/hardware/tarea.
- iTransformer: cinco runs de robustez en Table 5 cubren otros seis datasets; no documentan repeticiones ni dispersión de la fila ETTh1 elegida.

## Eficiencia y reproducción

efficiency_matrix.csv y models.json conservan unidades, hardware, batch, longitudes y tareas. No se convierten MB en cantidades de pesos, barras sin etiquetas en tiempos exactos, ni costes de M5/imputation en costes BTC. FLOPs no reportados permanecen NR. Las fórmulas del paper, las derivaciones del analista y el código se identifican como fuentes distintas.

Los datos públicos congelados permiten regenerar este paquete sin acceso a adjuntos privados: `python scripts/compile_e2_fulltext_review.py --check`. provenance.json y SHA256SUMS registran los hashes de las entradas y salidas públicas.
