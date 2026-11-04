# Auditoría numérica del análisis de los tres modelos base

Fecha UTC: 2026-11-04T14:53:23.082102+00:00

**Resultado: PASS.** Se auditaron predicciones almacenadas de desarrollo; cero archivos de mercado o test abiertos y cero ajustes de modelos. No se modificó código.

## Comprobaciones

| Comprobación | Cantidad | Resultado |
|---|---:|---|
| Filas de predicción / corridas de desarrollo | 16.560 / 90 | Coinciden |
| Semillas por modelo / fechas comunes | 30 / 184 | 1001–1030; 01/07–31/12/2024 |
| Agregados de métricas | 24 | 6 métricas × 4 métodos |
| Contrastes pareados y corrección Holm | 9 | Coinciden |
| DM: HAC14 primario y HAC7/28 de sensibilidad | 18 | Holm en los 6 primarios |
| Diagnósticos residuales LB/BDS | 15 | Semilla 1001; Holm conjunto |
| Comparaciones con intervalos bootstrap | 18 | 2.000 réplicas por longitud 14/7/28 |
| Filas de ranking bootstrap | 21 | Coinciden |
| Pruebas unitarias del analizador | 10 | Aprobadas |

Las definiciones de métricas verificadas son RMSE, MAE, MSE, R², MAPE y MASE. Los intervalos t usan desviación estándar muestral y 29 grados de libertad entre semillas. Persistencia cuenta como un método determinista y no recibe un intervalo entre semillas.

## Errores numéricos máximos

| Cantidad contrastada | Máxima diferencia absoluta |
|---|---:|
| `metric_mean_absolute_error` | 8.8817841970012523e-16 |
| `metric_sample_sd_absolute_error` | 4.4408920985006262e-16 |
| `metric_ci_endpoint_absolute_error` | 8.8817841970012523e-16 |
| `per_run_metric_absolute_error` | 1.7763568394002505e-15 |
| `paired_t_statistic_absolute_error` | 7.1054273576010019e-15 |
| `paired_p_value_absolute_error` | 9.8227697740672036e-25 |
| `paired_effect_dz_absolute_error` | 0 |
| `paired_shapiro_p_absolute_error` | 0 |
| `paired_ci_endpoint_absolute_error` | 0 |
| `paired_holm_p_absolute_error` | 5.9970594410094506e-24 |
| `dm_loss_difference_absolute_error` | 0 |
| `dm_long_run_variance_absolute_error` | 6.3527471044072525e-22 |
| `dm_hln_statistic_absolute_error` | 8.8817841970012523e-16 |
| `dm_p_value_absolute_error` | 4.5102810375396984e-17 |
| `dm_primary_holm_p_absolute_error` | 1.7347234759768071e-18 |
| `diagnostic_statistic_absolute_error` | 0 |
| `diagnostic_p_value_absolute_error` | 1.275966915836473e-65 |
| `diagnostic_holm_p_absolute_error` | 1.275966915836473e-65 |
| `bootstrap_difference_ci_endpoint_absolute_error` | 0 |
| `bootstrap_relative_gain_ci_endpoint_absolute_error` | 0 |
| `bootstrap_ranking_frequency_absolute_error` | 0 |

Son diferencias de coma flotante y serialización; todas cumplen las tolerancias explícitas del JSON. Las cantidades de distintas unidades no se combinan en un único error.

## Resultados principales

| Método | RMSE medio | DE entre semillas | IC 95 % entre semillas |
|---|---:|---:|---|
| MLP | 0.0159903092278324 | 0.00285446425962948 | [0.0149244347561301; 0.0170561836995347] |
| LSTM | 0.00890346290242051 | 0.000485441210822708 | [0.00872219617526403; 0.00908472962957699] |
| CNN1D | 0.0175866890763792 | 0.00519211888965756 | [0.0156479200202034; 0.019525458132555] |
| Persistencia | 0.00482247875241335 | No aplica | No aplica |

Los tres modelos neuronales tienen mayor RMSE que persistencia. LSTM es la mejor red observada en desarrollo; MLP fue la familia de referencia operativa fijada previamente por HPO.

| Comparación pareada | Diferencia media | IC 95 % entre semillas | p sin ajustar | p Holm (familia 9) |
|---|---:|---|---:|---:|
| MLP - LSTM | 0.00708684632541192 | [0.00598691208221072; 0.00818678056861312] | 9.02387966031e-14 | 5.41432779618e-13 |
| MLP - CNN1D | -0.0015963798485468 | [-0.00373266602498707; 0.000539906327893479] | 0.1372642435 | 0.549056974 |
| LSTM - CNN1D | -0.00868322617395872 | [-0.0105831181372806; -0.00678333421063679] | 2.9818440515e-10 | 1.49092202575e-09 |
| MLP - Persistencia | 0.0111678304754191 | [0.0101019560037168; 0.0122337049471214] | 2.5084884824e-19 | 2.00679078592e-18 |
| LSTM - Persistencia | 0.00408098415000715 | [0.00389971742285067; 0.00426225087716364] | 1.13893696396e-28 | 1.02504326757e-27 |
| CNN1D - Persistencia | 0.0127642103239659 | [0.0108254412677901; 0.0147029793801417] | 5.25203752598e-14 | 3.67642626819e-13 |
| MLP relative gain > 5% | -2.31578635153763 | [-2.53680847033264; -2.09476423274262] | 1 | 1 |
| LSTM relative gain > 5% | -0.846242017751737 | [-0.883829892465704; -0.80865414303777] | 1 | 1 |
| CNN1D relative gain > 5% | -2.64681525399737 | [-3.04884274975472; -2.24478775824002] | 1 | 1 |

Las diferencias de RMSE se expresan como A−B; valores negativos favorecen A. Las tres filas de ganancia relativa usan fracciones y contrastan ganancia > 0,05. Sus p-valores son 1.

Para LSTM−persistencia, el intervalo bootstrap temporal primario L14 de la diferencia de RMSE es [0,002234065604; 0,005794735698]. Este intervalo condiciona en las redes almacenadas y no incluye reentrenamiento.

## Alcance y observaciones científicas

1. La evidencia es exploratoria de desarrollo y calibración sobre el calendario fijo del 01/07 al 31/12 de 2024, utilizado previamente en el piloto y el EDA; no demuestra generalización en el test reservado.
2. El HPO fijó MLP como familia de referencia operativa antes del benchmark. LSTM es la mejor arquitectura neuronal observada posteriormente; cambiar la referencia al ver el benchmark constituye una nueva selección de desarrollo.
3. Todas las medias de RMSE neuronales superan el error de persistencia determinista. Las tres pruebas unilaterales de ganancia superior al 5 % tienen p ajustado igual a 1; ningún modelo cumple ese objetivo en estas fechas.
4. Los intervalos t entre semillas condicionan en las fechas fijas y describen aleatoriedad del entrenamiento. Los intervalos temporales por bloques condicionan en los modelos almacenados, sin reentrenamiento ni remuestreo de semillas; son intervalos marginales y no simultáneos.
5. MLP−CNN1D tiene p de Holm entre semillas igual a 0,549056974; la evidencia de pérdidas temporales y bootstrap de fechas favorece MLP condicionada en los modelos almacenados. Describen fuentes de incertidumbre distintas y no deben sustituirse selectivamente.
6. Shapiro señala las distribuciones de diferencias MLP−LSTM y MLP−persistencia al nivel nominal 0,05. Se conservó el procedimiento t preregistrado; no rechazar en los demás contrastes no verifica normalidad.
7. DM usa pérdidas cuadráticas promediadas entre semillas, no la pérdida de una predicción promedio ni el RMSE medio entre semillas. HAC14 es primario con una familia Holm de seis comparaciones; HAC7/28 son sensibilidades sin pretensión de confirmación adicional.
8. Los 15 diagnósticos residuales de la semilla designada rechazan tras Holm. BDS rechaza una hipótesis i.i.d., no exclusivamente una estructura no lineal; el objetivo de volatilidad con ventanas superpuestas de siete días también induce dependencia serial. Los diagnósticos no establecen una mejora de desempeño alcanzable.
9. La planificación piloto exigía hasta 1212 semillas y el límite preregistrado de recursos permitió 30. No se alcanza la potencia planificada 0,80 para el pequeño efecto objetivo; la significancia de diferencias mucho mayores no elimina esta limitación de planificación.
10. Persistencia usa ocho cierres históricos para calcular la volatilidad previa de siete retornos; las entradas neuronales seleccionadas contienen siete días OHLCV. Esta diferencia de información registrada limita la interpretación de la ventaja del comparador.
11. Las frecuencias de ranking bootstrap iguales a 1 o 0 son frecuencias finitas de remuestreo bajo este diseño, no probabilidades posteriores ni demostración del ranking poblacional.
12. La auditoría recalculó el análisis exclusivamente desde resultados de desarrollo almacenados: cero archivos de mercado o test abiertos, cero ajustes de modelos y ningún cambio al código de entrenamiento o análisis.

## Trazabilidad

`revision_dl_analysis_20261104.json` conserva cifras completas, tolerancias, hashes de las entradas y salidas auditadas, las 9 pruebas, los 18 DM, los 15 diagnósticos, los 18 intervalos y las 21 filas de ranking.

Script independiente de recálculo: `/tmp/audit_real_dl_analysis.py` (SHA-256 `064cc07a67a9c8269b11ba4ead4e38975f88a6efac377685805e58541599c6e3`). Es un archivo temporal del workspace y no un script versionado; no importa el módulo de análisis del proyecto.

Comando de pruebas: `/workspace/.venvs/deeplearning-training/bin/python -m unittest discover -s tests -p test_dl_analysis.py -v`.

No se ejecutó entrenamiento, no se evaluó test 2025+ y no se editaron documentos del libro para esta auditoría.
