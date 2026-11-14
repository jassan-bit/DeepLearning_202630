# Inception individual: evidencia numérica recuperada

Sí existen resultados exactos de redes individuales en el repositorio autor de *InceptionTime: Finding AlexNet for Time Series Classification* (DOI 10.1007/s10618-020-00710-y). Permiten un **nuevo escenario contextual de artefactos UCR2018/128 con dos variantes**, conservando el escenario previo.

| Variante | Error de clasificación medio |
|---|---:|
| Una red Inception: media de las cinco ejecuciones individuales en los 128 datasets | 0.1639239119738714 |
| InceptionTime: ensemble de cinco redes, mismos 128 nombres de datasets | 0.1564056859794546 |

La operación es `L_single = mean_dataset(mean_run(1 - accuracy))`. Promediar los errores de cinco redes entrenadas estima el rendimiento repetido de **una red**; no combina sus probabilidades ni crea un ensemble. Se usan todas las ejecuciones y todos los datasets, sin elegir el mejor run. La fórmula `(maxloss - L_single)/(maxloss - minloss)` da **P=0** dentro de este pool de dos variantes. Significa mayor pérdida entre estas dos variantes; no acredita baja eficacia absoluta.

El [README autor congelado](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/README.md#L45) enlaza el resultado ensemble128 en la línea50 y define el archivo individual en la línea55. Los [datos individuales](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/results-inception-128.csv) contienen 640 filas: 128 datasets por cinco iteraciones, sin celdas faltantes/duplicadas; el [ensemble128](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/results-InceptionTime-128.csv) contiene128 filas con exactamente el mismo conjunto de nombres y etiqueta `UCRArchive_2018`. Ambos CSV numéricos se preservan en `author_raw/`; hashes y objetos Git están en `candidate.json`.

La [versión v3 del original](https://arxiv.org/html/1909.04939v3) distingue single/ensemble en §3.2, ecuación1, y §5, Figura11. No publica en texto una cifra exacta de accuracy/MPCE individual para reconstruir el pool de seis variantes de esa figura. No se digitalizó el gráfico. El CSV85 sólo incluye doce variantes ensemble; no se mezcla con UCR2018/128. Esta métrica es error de clasificación medio, **no MPCE**.

La coincidencia del archivo de datos, nombres y marco autor respalda una comparación contextual reportada. El original §6.6 describe los splits train/test del archivo UCR, pero los CSV128 no incluyen identificadores/checksums de instancias que permitan auditar los splits exactos, ni su procesamiento completo. No se afirma control independiente de HPO o presupuesto. Tampoco se equipara esta arquitectura clasificadora con la adaptación E2/D7 ni con pronóstico de volatilidad cripto.

Se hizo una búsqueda adicional de pools SOTA compatibles. Apareció [hfawaz/dl-4-tsc, results/results-ucr-128.csv](https://github.com/hfawaz/dl-4-tsc/blob/master/results/results-ucr-128.csv), con ocho clasificadores y 5.120 filas. Su `utils/utils.py` actual, líneas74–99 y130–159, normaliza cada serie de UCR2018; el código expuesto de InceptionTime no conserva esa misma rama de lectura2018. No quedó probada la identidad de preprocessing y splits de las ejecuciones históricas, por lo que no se importa ese pool ni se mezclan resultados2015/2018.

Este candidato requiere declarar una extensión retrospectiva explícita del escenario P9, cuyo pool original no incluía este artefacto. `P_original` y P9 permanecen intactos. No se presenta una tabla del manuscrito ni un ganador global. `python recompute.py` reproduce sólo aritmética sobre los CSV congelados; no entrena modelos ni usa datos de mercado.
