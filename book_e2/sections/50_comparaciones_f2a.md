# 50. Comparaciones F2a posteriores: DM, MCS y rangos por semilla

**Análisis ejecutado el 13/11/2026 sobre pronósticos congelados.** La nueva familia DM–HLN rechaza **35 de 120 igualdades de pérdida media**, con Holm a α=0,05. El Model Confidence Set al **90 % conserva cinco sistemas: FCN-1D, LSTM-autoencoder con predictor, GRU32, Persistence y ARIMA**. El diagnóstico de rangos entre las trece redes detecta heterogeneidad global entre ejecuciones, pero sus 78 comparaciones Wilcoxon no rechazan después de Holm.

Estos resultados son **exploratorios y posteriores a conocer la prueba**. Las [veinte comparaciones primarias y la A2 original](06_benchmark_final.md) conservan sus resultados, su métrica RMSE, su margen práctico y su familia Holm. Los nuevos procedimientos responden a hipótesis distintas; no constituyen tres pruebas confirmatorias de cada contraste original. Tampoco prueban equivalencia, generalización a otros mercados ni ausencia de sobreajuste. La [auditoría residual de §49](49_diagnosticos_f2a.md) sigue siendo pertinente para interpretar las aproximaciones temporales.

## Entrada, plan y unidad de análisis

Se leen las mismas **81 trayectorias, 16 sistemas y 608 fechas diarias contiguas**, del 01/01/2025 al 31/08/2026. Las trece redes aportan seis semillas cada una; los tres clásicos aportan una trayectoria determinista. Se verifica el SHA256 completo de la entrada antes de calcular:

```text
65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187
```

El [plan de análisis](../_static/e2_f2a_s8_20261113/analysis_plan.json) se guardó antes de ejecutar estos cálculos, cuando los resultados originales ya eran conocidos: **no es un preregistro prospectivo**. Fija una familia DM de todas las parejas, MCS con α=0,10 y la evaluación condicionada de rangos por semilla. No se entrenan modelos, no se ejecuta inferencia con pesos, no se reabren HPO o calibración y no se vuelve a ejecutar el benchmark.

Para las comparaciones temporales se define, por sistema y fecha:

\[
L_{m,t}=\frac{1}{R_m}\sum_{s=1}^{R_m}(y_t-\hat y_{m,s,t})^2,
\qquad R_m=6\text{ para redes},\quad R_m=1\text{ para clásicos}.
\]

Se conserva cada semilla como trayectoria **fija**: el remuestreo temporal mueve fechas conjuntamente entre sistemas y no genera nuevas semillas. Esta pérdida estima el error cuadrático medio de las ejecuciones guardadas; **no es el error del pronóstico promedio de un ensamble**. Su promedio temporal tampoco coincide con la métrica primaria original:

\[
\bar L_m=\frac1{R_m}\sum_s\mathrm{RMSE}_{m,s}^{2},
\qquad
\mathrm{RMSE}_{\mathrm{original},m}=\frac1{R_m}\sum_s\mathrm{RMSE}_{m,s}.
\]

Por ello, una comparación sobre MSE no se interpreta como otra prueba del margen δ expresado en RMSE. El [resumen de pérdidas](../_static/e2_f2a_s8_20261113/system_loss_summary.csv) conserva MSE, su raíz, la media original de RMSE y el MSE de la predicción promedio para hacer visible la diferencia de estimandos.

Se verificaron aritméticamente las 81 RMSE individuales frente a las originales, con diferencia máxima **5,20417×10⁻¹⁸** y tolerancia `1e-10`. La representación canónica de `original_analysis` conserva el mismo SHA256 antes y después: `5276613f21eccd23aab062acdee4be3b121e127ab704a35aa71c016d4baf5b45`. Los veinte contrastes se leen y se identifican por hash; sus hipótesis no se recalculan.

## Diebold–Mariano con HLN y HAC14

Se forman las **120 parejas de 16 sistemas**. Para cada pareja A–B, el diferencial es \(d_t=L_{A,t}-L_{B,t}\): un valor positivo favorece a B. La hipótesis bilateral es \(H_0:E[d_t]=0\), con horizonte real **h=1**. La superposición de los objetivos σ₇ motiva considerar dependencia serial adicional; no convierte el horizonte en siete o catorce pasos.

La varianza de la media usa Newey–West con pesos Bartlett y **14 rezagos**:

\[
\widehat V(\bar d)=\frac1T\left[\hat\gamma_0+
2\sum_{\ell=1}^{14}\left(1-\frac{\ell}{15}\right)\hat\gamma_\ell\right],
\quad
\hat\gamma_\ell=\frac1T\sum_{t=\ell+1}^T(d_t-\bar d)(d_{t-\ell}-\bar d).
\]

Se aplica el factor Harvey–Leybourne–Newbold \(c=\sqrt{(T-1)/T}\) para h=1 y se contrasta \(c\bar d/\sqrt{\widehat V(\bar d)}\) frente a una t de **607 grados de libertad**. Los IC nominales del 95 % invierten esta misma prueba. No se añade otra corrección de grados de libertad a la covarianza HAC. Combinar HLN con un ancho de banda superior a h−1 produce una aproximación exploratoria; no se afirma exactitud finita ante el solapamiento y la heterocedasticidad observados.

Holm se aplica **una sola vez a las 120 hipótesis nuevas**. No se mezcla esta familia con las veinte originales, la auditoría residual o los rangos por semilla. Todos los contrastes fueron estimables y **35/120 rechazan** a α=0,05. Los [resultados completos](../_static/e2_f2a_s8_20261113/dm_hln_hac14_all120.csv) incluyen diferencia media, IC, estadístico, p bruto, p ajustado y decisión.

La tabla siguiente muestra comparaciones relevantes. Las diferencias e IC están en **10⁻⁶ unidades de σ₇²**. Los IC son marginales, no simultáneos; un IC que excluye cero no implica rechazo con Holm120.

| A − B | Diferencia MSE | IC nominal 95 % | p bruto | p Holm120 | Rechaza |
|---|---:|---:|---:|---:|:---:|
| LSTM-autoencoder − GRU32 | −1,519 | [−2,977; −0,062] | 0,04109 | 1,00000 | No |
| LSTM-autoencoder − Persistence | −2,176 | [−5,542; 1,191] | 0,20482 | 1,00000 | No |
| FCN-1D − GRU32 | 4,873 | [−3,747; 13,493] | 0,26740 | 1,00000 | No |
| FCN-1D − Persistence | 4,216 | [−2,607; 11,039] | 0,22538 | 1,00000 | No |
| MLP16 − GRU32 | 4,421 | [2,270; 6,572] | 0,0000613 | 0,00656 | Sí |
| PatchTST − GRU32 | 86,818 | [42,310; 131,326] | 0,0001410 | 0,01382 | Sí |
| PatchTST − Persistence | 86,161 | [43,708; 128,615] | 0,0000754 | 0,00784 | Sí |

La menor MSE del autoencoder frente a GRU32 no supera esta corrección múltiple. PatchTST presenta pérdida mayor que GRU32 y Persistence con rechazo ajustado. No rechazar una pareja no establece igualdad ni equivalencia práctica. La familia completa permite comprobar las otras 113 parejas sin seleccionar únicamente las favorables.

## Model Confidence Set de Hansen

Se implementa el procedimiento de rango **T_R** sobre las mismas 16 columnas de pérdida diaria. Usa **10.000 remuestreos circulares por bloques de 14 días**, semilla RNG `202611132`. Los mismos índices se aplican a todos los sistemas y etapas para conservar el emparejamiento y la dependencia conjunta. Los bloques circulares unen el final y el inicio de esta cronología como aproximación de remuestreo; esa unión no representa una continuidad económica observada.

Para cada pareja, el error estándar se estima mediante la media de \((\bar d^*_{ij}-\bar d_{ij})^2\). El estadístico observado es \(T_R=\max_{i,j}|\bar d_{ij}/\widehat{se}_{ij}|\). La distribución nula usa **diferenciales bootstrap centrados**, \((\bar d^*_{ij}-\bar d_{ij})/\widehat{se}_{ij}\), y no el máximo de pérdidas bootstrap sin centrar. Los p-valores Monte Carlo usan \((1+\text{excedencias})/(10000+1)\).

Si se rechaza igualdad de precisión en el conjunto activo, se elimina el sistema con mayor desventaja estandarizada \(e_R=\arg\max_i\max_j\bar d_{ij}/\widehat{se}_{ij}\). Este criterio no equivale a quitar siempre la mayor MSE. El máximo acumulado de los p-valores secuenciales produce el p MCS por sistema. Se conserva la secuencia completa para calcularlos; las etapas posteriores al primer no rechazo **no excluyen sistemas del conjunto al 90 %**.

| Sistema | MSE media ×10⁶ | p MCS | Integra MCS 90 % |
|---|---:|---:|:---:|
| FCN-1D | 20,908 | 0,20458 | Sí |
| ResNet-1D | 20,882 | 0,08169 | No |
| Inception-1D | 19,108 | 0,01580 | No |
| LSTM-autoencoder con predictor | 14,516 | 1,00000 | Sí |
| TimesNet | 23,384 | 0,00540 | No |
| TSMixer | 24,930 | 0,00840 | No |
| TiDE | 34,358 | 0,00010 | No |
| TimeMixer | 20,515 | 0,01580 | No |
| PatchTST | 102,853 | 0,00320 | No |
| iTransformer | 27,610 | 0,00400 | No |
| MLP16 | 20,457 | 0,00440 | No |
| LSTM32 | 21,764 | 0,00010 | No |
| GRU32 | 16,036 | 0,14559 | Sí |
| Persistence | 16,692 | 0,20458 | Sí |
| SeasonalNaive | 147,588 | 0,01580 | No |
| ARIMA | 15,379 | 0,44286 | Sí |

La primera etapa arroja \(T_R=7,11561\) y p=0,00009999. Tras once exclusiones, el conjunto de cinco sistemas arroja \(T_R=2,04852\) y p=0,14559: se detiene el procedimiento al nivel α=0,10. Permanecer en el MCS no significa que los cinco sistemas sean equivalentes o todos mejores que cualquier excluido. La inclusión de FCN-1D junto con la exclusión de Inception-1D, pese a la menor MSE puntual de este último, refleja la estandarización y la incertidumbre conjunta.

El nivel **90 %** corresponde al procedimiento exploratorio bajo sus supuestos; no certifica una probabilidad de éxito en otro periodo. No se añade Holm a los p de eliminación: MCS ya es un procedimiento conjunto secuencial, diferente de las pruebas DM por pareja. [Pertenencia y p MCS](../_static/e2_f2a_s8_20261113/mcs_membership.csv) · [Secuencia completa](../_static/e2_f2a_s8_20261113/mcs_steps.csv).

## Friedman y post-hoc: diagnóstico condicionado de optimización

La guía admite ejecuciones como bloques en el catálogo general y series u orígenes en la ruta F2a. Aquí solo existe **una cronología de mercado**. Por tanto, no se dispone de un Friedman poblacional sobre series u orígenes independientes. Los trimestres consecutivos tampoco se convierten en réplicas iid mediante agregación.

Sí se presenta un **diagnóstico condicionado al mercado y recetas congelados**: cada uno de los seis identificadores de semilla aporta un vector de las trece RMSE neurales. Los identificadores son comunes a priori, pero compartir un número no acredita por sí solo efectos aleatorios comunes o emparejamiento sustantivo entre arquitecturas. La interpretación requiere independencia entre vectores de semillas, condicionada a los datos, y una hipótesis de intercambiabilidad de rangos entre modelos dentro de cada bloque. Estas condiciones son supuestos del diagnóstico, no resultados demostrados por observar seis ejecuciones. Los clásicos quedan excluidos: **no se duplica una trayectoria determinista seis veces para simular ejecuciones**.

El estadístico Friedman con corrección de empates es **Q=46,98901**, gl=12, p χ²=0,000004682. La transformación Iman–Davenport da **F(12,60)=9,39367**, p=7,02787×10⁻¹⁰. Dado el tamaño de solo seis bloques, se añade una comprobación de **10.000 permutaciones independientes de etiquetas de modelo dentro de cada fila de rangos**, semilla `202611133`: p Monte Carlo=**0,00009999**. Este valor refleja la resolución mínima observada del remuestreo, no una probabilidad exacta nula. Las tres cifras globales evalúan el mismo diagnóstico de rangos, no tres hipótesis independientes.

Los rangos medios más bajos corresponden al autoencoder (**1,500**) y GRU32 (**2,167**); PatchTST obtiene rango **13 en las seis semillas**. Los [rangos de las trece redes](../_static/e2_f2a_s8_20261113/conditional_seed_ranks.csv) conservan además mínimo, máximo y media de RMSE.

Al superar la puerta global Iman–Davenport a α=0,05 se ejecutan las **78 parejas Wilcoxon bilaterales exactas**, sobre diferencias de RMSE entre los seis identificadores, con Holm78. Las diferencias no tienen ceros ni empates absolutos en esta ejecución. La interpretación adicional exige simetría de las diferencias y validez del emparejamiento; los seis resultados por pareja no son seis mercados independientes.

**Ninguna de las 78 parejas rechaza y todos sus p Holm son 1,0.** Con seis pares sin empates, el menor p bilateral exacto posible es **2/2⁶=0,03125**; incluso ese mínimo, multiplicado por 78 en el primer paso Holm, supera uno. Por tanto, este post-hoc carece de resolución para declarar parejas significativas en esta familia. El rechazo global no autoriza atribuir significancia a una pareja escogida por su rango. [Post-hoc completo](../_static/e2_f2a_s8_20261113/conditional_wilcoxon_holm78.csv).

## Alcance y reproducción

DM y MCS consideran dependencia serial mediante HAC14 o bloques14, respectivamente. Requieren que la dependencia débil o estacionariedad local y la longitud elegida sean adecuadas. Los diagnósticos residuales previos muestran que no es defendible tratar los días como iid. Estos ajustes tampoco recuperan la independencia entre mercados, el cegamiento del periodo conocido o la variabilidad de nuevas ejecuciones de entrenamiento. No se utilizaron las conclusiones para seleccionar otra receta o volver a entrenar. Las comprobaciones de robustez ante perturbaciones constituyen evidencia adicional distinta de estas comparaciones.

El programa `scripts/e2_f2a_s8_comparisons.py` y sus diez pruebas cubren la fórmula HAC/HLN e inversión del IC, la distinción de pérdidas individuales y ensamble, el centrado de la nula MCS, la eliminación estandarizada, varianzas degeneradas, repetibilidad, bloques circulares y la resolución del post-hoc. Incluyen una comprobación simulada de tamaño bajo igualdad iid; es una prueba de funcionamiento, no una demostración de calibración en esta serie. Una revisión estadística independiente comprobó las fórmulas; el HAC coincidió con OLS-HAC de statsmodels sin corrección adicional, con discrepancia máxima de 6,51×10⁻¹⁹ en un vector de referencia.

Entorno de ejecución: Python 3.12.14, NumPy 2.3.5 y SciPy 1.16.3. Se reproducen únicamente los cálculos posteriores a partir de la entrada ya verificada, dejando el plan en el directorio de salida:

```bash
python -m unittest discover -s tests -p 'test_e2_f2a_s8_comparisons.py' -v
python scripts/e2_f2a_s8_comparisons.py --input /ruta/e2_f2a_input.json --output /ruta/statistics
```

El [resumen JSON](../_static/e2_f2a_s8_20261113/comparisons_summary.json) registra alcance, parámetros, hashes originales y resultados. El [manifiesto SHA256](../_static/e2_f2a_s8_20261113/manifest.json) identifica código, pruebas, plan y productos compactos; no se guardan las matrices bootstrap completas. Este suplemento aporta comparaciones S8 con sus restricciones, sin convertir el protocolo retrospectivo en un estudio prospectivo ni modificar los veinte resultados primarios.
