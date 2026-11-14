# Taxonomía de errores y referencias de predictibilidad

Este complemento se calculó después de conocer el test y sus resultados originales. Las reglas se conservaron antes de generar sus etiquetas automáticas. No es un prerregistro del experimento original ni una revisión humana. Sólo utiliza pronósticos guardados y la volatilidad de desarrollo 2023–2024: no hubo entrenamiento, inferencia de modelos ni ajuste de pesos.

Reglas registradas en 2026-11-13T23:23:42.193648-05:00 (America/Bogota). Exportación congelada SHA-256: 65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187.

## Categorías y muestra auditable

La unidad principal es una fecha por modelo; se usa la media aritmética de sus seis pronósticos congelados. Es un predictor agregado diagnóstico y no sustituye la tabla A2, cuya estadística sigue siendo la media de los RMSE individuales. Se publican además las etiquetas por semilla y su concordancia para hacer visible la cancelación de errores en la media.

Con e = pronóstico − observado, los cortes son a = 0.00130577631562 y b = 0.00728725980319: Q50 y Q90 de los 714 cambios absolutos diarios consecutivos del objetivo de desarrollo. No se atraviesan huecos. Son escalas descriptivas, no márgenes confirmatorios de relevancia ni ruido irreducible.

| Categoría exclusiva | Regla |
|---|---|
| Banda pequeña | abs(e) ≤ a |
| Subestimación moderada | a < abs(e) ≤ b y e < 0 |
| Sobreestimación moderada | a < abs(e) ≤ b y e ≥ 0 |
| Subestimación grande | abs(e) > b y e < 0 |
| Sobreestimación grande | abs(e) > b y e ≥ 0 |

Se etiquetaron las 608 fechas de cada uno de los diez modelos: **6.080 casos fecha-modelo**. La muestra conserva **100 fechas distintas por modelo**, con las seis semillas por fecha (**600 errores individuales por modelo**). Son casos con residuos continuos; no se afirma que cada uno sea un fallo grave. La selección común se estratificó por año y régimen observado del objetivo, con Q25/Q75 calculados sólo en desarrollo y semilla 272101. No depende de los errores. Estos regímenes son descriptivos, no entradas de predicción ni causas económicas.

| Modelo | Fechas etiquetadas | Fechas auditables | Casos por semilla | Concordancia de etiquetas |
|---|---:|---:|---:|---:|
| FCN-1D | 608 | 100 | 600 | 75.2% |
| ResNet-1D | 608 | 100 | 600 | 67.2% |
| Inception-1D | 608 | 100 | 600 | 70.0% |
| LSTM-autoencoder con predictor | 608 | 100 | 600 | 81.6% |
| TimesNet | 608 | 100 | 600 | 70.8% |
| TSMixer | 608 | 100 | 600 | 76.5% |
| TiDE | 608 | 100 | 600 | 56.6% |
| TimeMixer | 608 | 100 | 600 | 69.5% |
| PatchTST | 608 | 100 | 600 | 76.8% |
| iTransformer | 608 | 100 | 600 | 91.3% |

Las frecuencias del calendario completo son recuentos exactos de este conjunto finito. Los IC del 95% son intervalos marginales de sensibilidad temporal: 10.000 remuestreos de bloques circulares de 14 días, con sensibilidad a 7 y 28 días. Suponen dependencia corta y estacionariedad aproximada; no son incertidumbre de contar estos 608 días ni garantizan cobertura simultánea de todas las categorías. Los cambios de régimen limitan la extrapolación.

| Modelo | Banda pequeña | Subest. moderada | Sobreest. moderada | Subest. grande | Sobreest. grande |
|---|---:|---:|---:|---:|---:|
| FCN-1D | 35.0% [30.8%, 39.3%] | 24.8% [19.7%, 30.1%] | 35.0% [29.3%, 41.3%] | 5.1% [2.1%, 8.9%] | 0.0% [0.0%, 0.0%] |
| ResNet-1D | 32.4% [28.1%, 36.8%] | 16.4% [12.7%, 20.4%] | 46.4% [40.3%, 52.8%] | 4.6% [2.0%, 8.1%] | 0.2% [0.0%, 0.5%] |
| Inception-1D | 33.2% [29.1%, 37.2%] | 19.7% [15.3%, 24.5%] | 42.3% [36.0%, 48.8%] | 4.3% [2.0%, 7.4%] | 0.5% [0.0%, 1.2%] |
| LSTM-autoencoder con predictor | 31.1% [26.0%, 36.2%] | 17.9% [13.7%, 22.4%] | 46.9% [40.1%, 53.8%] | 3.9% [1.8%, 6.7%] | 0.2% [0.0%, 0.5%] |
| TimesNet | 25.3% [21.2%, 29.4%] | 19.4% [14.3%, 24.8%] | 49.8% [42.8%, 57.1%] | 3.1% [1.6%, 4.9%] | 2.3% [0.7%, 4.4%] |
| TSMixer | 30.3% [24.8%, 35.9%] | 37.3% [30.9%, 43.4%] | 21.7% [16.6%, 27.5%] | 10.4% [6.2%, 15.0%] | 0.3% [0.0%, 0.8%] |
| TiDE | 28.0% [23.4%, 32.7%] | 36.3% [29.9%, 42.9%] | 26.3% [21.2%, 31.9%] | 7.6% [4.3%, 11.3%] | 1.8% [0.7%, 3.3%] |
| TimeMixer | 23.4% [19.2%, 27.6%] | 16.8% [12.3%, 21.4%] | 56.2% [48.8%, 63.8%] | 3.0% [1.2%, 5.3%] | 0.7% [0.2%, 1.3%] |
| PatchTST | 10.7% [7.6%, 14.0%] | 12.2% [7.9%, 16.8%] | 33.2% [27.8%, 38.7%] | 11.0% [5.3%, 17.4%] | 32.9% [25.3%, 41.1%] |
| iTransformer | 22.9% [18.4%, 27.5%] | 21.5% [16.3%, 27.1%] | 41.0% [34.0%, 48.4%] | 10.5% [5.9%, 15.8%] | 4.1% [2.3%, 6.2%] |

Los archivos detallados conservan los pesos de inclusión N_h/n_h, frecuencias muestrales ponderadas, la concordancia de semillas y correlación/intersección/Jaccard de errores grandes para los 45 pares de modelos. Las categorías describen residuos; no sustituyen una taxonomía semántica anotada por expertos.

## Referencias de predictibilidad y distancia γ

Se calculó entropía de permutación de Bandt y Pompe (2002; [DOI 10.1103/PhysRevLett.88.174102](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.174102)), con órdenes 3, 4 y 5 y retardo 1, sin atravesar faltantes. Es complejidad ordinal, no una conversión a error Bayes, un porcentaje de predictibilidad o un RMSE mínimo alcanzable.

| Periodo | Orden | Ventanas | Entropía normalizada |
|---|---:|---:|---:|
| development_2023_2024 | 3 | 712 | 0.971268 |
| development_2023_2024 | 4 | 710 | 0.934492 |
| development_2023_2024 | 5 | 708 | 0.897266 |
| known_test_2025_2026 | 3 | 606 | 0.954079 |
| known_test_2025_2026 | 4 | 605 | 0.918404 |
| known_test_2025_2026 | 5 | 604 | 0.875382 |

El ensemble uniforme de los diez modelos, emparejado por semilla y sin ajustar pesos, alcanza RMSE medio **0.0041200562**. SAE conserva **0.0038034296** y Persistence **0.0040855959**. El ensemble resulta peor: **no constituye un techo empírico válido**. Se conserva el resultado negativo, sin buscar una combinación más favorable.

Con s = −RMSE, la fórmula de la guía se orienta así:

$$\gamma = \frac{RMSE_{mejor}-RMSE_{referencia}}{RMSE_{base}-RMSE_{referencia}}.$$

Con el ensemble, el cociente algebraico es 9.188146; el denominador es negativo y **no se interpreta como fracción pendiente**. Frente a la cota matemática ideal RMSE = 0, **γ_ideal = 0.930936**, IC temporal de 14 días [0.844480, 1.029743]. Es la fracción del error de Persistence que permanece respecto a una predicción perfecta; **no estima la brecha frente a un techo real alcanzable**. El riesgo Bayes y el ruido irreducible permanecen sin identificar.

El complemento documenta las referencias ensayadas, la distancia condicionada y sus límites; no acredita un techo de desempeño identificado. No modifica la ausencia de superioridad práctica confirmada ni convierte el test conocido desde E1 en prueba independiente.

## Archivos auditables

- taxonomy_protocol.json: reglas, fecha, umbrales, semillas y alcance post hoc.
- all_6080_date_model_labels.csv: etiquetas del calendario completo.
- audit_100_dates_per_main_model.csv y audit_600_seed_cases_per_main_model.csv: muestra y semillas.
- error_category_frequencies.csv: frecuencias e IC de 7, 14 y 28 días.
- shared_error_diversity.csv: solapamiento y correlación de los 45 pares.
- permutation_entropy.csv y ceiling_gamma_references.json: referencias y límites de γ.
- manifest.json: hashes y declaración de ausencia de ajuste/inferencia.
