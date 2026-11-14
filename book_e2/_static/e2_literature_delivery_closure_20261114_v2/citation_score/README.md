# Auditoría bibliométrica C/P v2

Fecha de auditoría: 14 de noviembre de 2026 UTC. Este directorio contiene artefactos nuevos. No se modificaron los capítulos, los resultados publicados, la pertenencia al Top10 ni los componentes originales.

## C: resultado y cobertura

Se conservan las seis C verificadas del corte recibido de Scopus. Siguen pendientes cuatro sistemas correspondientes a tres estudios: FCN y ResNet comparten Wang2016/2017; también faltan Inception y LSTM-SAE. Ningún dato ausente se convierte en cero.

`local_export_identity_search.csv` documenta 27 búsquedas: nueve estudios por tres corpus de metadatos derivados. Se revisaron 7.499 registros Scopus, 1.298 WoS y 7.833 combinados, con coincidencia exacta de título normalizado y DOI. Para los tres estudios pendientes se buscaron tanto el identificador arXiv como el editorial cuando existe. No hubo coincidencias. Este resultado se limita a las exportaciones recibidas y parciales; no demuestra ausencia del estudio en Scopus/WoS. Los nativos crudos no están en el repositorio actual; se inspeccionaron los derivados bibliográficos y sus manifiestos.

Los DOI editoriales se documentan como alias de búsqueda, conservando la versión y el año original:

| Estudio | Identificador utilizado | Alias editorial verificado |
|---|---|---|
| Wang, FCN/ResNet | arXiv:1611.06455, 2016 | [10.1109/IJCNN.2017.7966039](https://ieeexplore.ieee.org/abstract/document/7966039/footnotes), IJCNN2017 |
| InceptionTime | arXiv:1909.04939, 2019 | [10.1007/s10618-020-00710-y](https://link.springer.com/article/10.1007/s10618-020-00710-y), 2020; el vínculo aparece también en [arXiv](https://arxiv.org/abs/1909.04939) |
| LSTM-SAE | DOI editorial, 2019 | [10.1038/s41598-019-55320-6](https://www.nature.com/articles/s41598-019-55320-6) |

No se reintentaron las nueve consultas OpenAlex bloqueadas previamente. Se probó un proveedor independiente, la [API oficial de métricas de Dimensions](https://platform.digital-science.com/api-reference/dimensions-metrics). El sondeo también devolvió proxy403 y se detuvo; `dimensions_probe_receipt.json` conserva la respuesta. No se obtuvo un nuevo corte uniforme de nueve estudios.

`external_observations.json` separa observaciones de páginas oficiales de los conteos admisibles para C. La página principal de Springer mostró 1.914 citas para Inception, mientras su página de métricas mostró 1.807 con actualización explícita del 19 de octubre. El resultado de búsqueda de Nature mostró 326 para SAE, con rastreo del mes anterior y sin acceso directo exitoso. Esos números no completan C: difieren el proveedor, la fecha de corte o la vía de observación. El [detalle de métricas de Springer](https://link.springer.com/article/10.1007/s10618-020-00710-y/metrics) identifica Dimensions como proveedor.

`current10_C_status_v2.csv` conserva las C originales. Su normalizador continúa siendo 487 citas por año en la cohorte recibida anterior, con edad desde primera publicación. No es un máximo bibliométrico mundial; tampoco se recalcula mezclando conteos nuevos de editorial con los antiguos de Scopus. La etiqueta histórica de carpeta20261105 se conserva, pero los manifiestos documentan un nombre de exportación Scopus y DA WoS del20261106 sin corte horario homogéneo verificable. No se presenta como una consulta en vivo de noviembre14.

## P: comparación contextual

Los artefactos de `contextual_p/` evalúan por separado clasificación UCR y pronóstico, conservan el P original y usan resultados de tablas publicadas. No se exige que todos estudien sigma7 de Bitcoin para presentar contexto bibliográfico, pero cada P contextual identifica su tarea, conjunto de datos, métrica, protocolo y conjunto de comparadores. No se mezclan las escalas ni se afirma una clasificación global controlada del Top10.

La lectura adicional permite utilizar el agregado MPCE de UCR44 para FCN/ResNet, en lugar de seleccionar una sola fila favorable. Para SAE se conserva la sensibilidad a RMSE/MAE/SMAPE de la tabla completa. Las tablas de iTransformer y TimeMixer proporcionan comparadores bajo el mismo experimento declarado en cada fuente. Inception continúa pendiente para la red individual, porque el resultado verificado corresponde al ensemble de cinco redes.

Un P contextual dentro de una tabla responde a una comparación distinta del P originalmente definido. No vuelve prospectivo el análisis posterior ni valida eficacia sobre Bitcoin, y no altera los intervalos originales. El protocolo y el resumen de `contextual_p/` precisan qué sistemas reciben una comparación contextual calculable.

## Escenario P9 y sensibilidad del score

`contextual_p/preferred_intrapaper_scenario_rule.json` fija una regla retrospectiva antes de generar la selección de P9; los valores de las tablas ya se conocían, por lo que no es un protocolo ciego ni una preinscripción. Se eligen MPCE UCR44, RMSE Beijing y MSE ETTh1/H96/L96. Entre tablas ETTh1 elegibles se prefiere el conjunto con más comparadores publicados y se desempata por DOI de fuente ascendente, sin usar el rendimiento. Cada denominador incluye todos los comparadores de esa tabla. Table23 usa exclusivamente las columnas de parámetros unificados; L96 deriva del enlace declarado con ese protocolo y no de una reproducción independiente.

`conditional_score_by_scenario.csv` expande las nueve P a los diez sistemas y mantiene Inception vacío. Aplica sólo como sensibilidad la fórmula y los pesos ya existentes. Una suma de métricas minmax de distintos conjuntos supone una equivalencia de utilidad entre esas escalas que no está demostrada: los números no son un ranking global controlado. No se sustituye el P original. Los límites son algebraicos al variar los campos desconocidos en[0,1], no intervalos de confianza.

| Escenario | C conocidas | P contextuales | Componentes conocidos | Scores condicionales completos |
|---|---:|---:|---:|---:|
| Auditoría original conservada | 6 | 0 | 36/50 | 0/10 |
| Anterior ETTh1 nominal, conservado | 6 | 6 | 42/50 | 6/10 |
| Anterior ETTh1 separado por lookback | 6 | 5 | 41/50 | 5/10 |
| Nuevo P9 dentro de tablas | 6 | 9 | 45/50 | 6/10 |

El escenario P9 se prefiere como descripción bibliográfica por conservar comparadores dentro de una misma tabla y sus ajustes declarados. La falta de calibración entre estudios impide interpretarlo como puntaje definitivo. MAE y SMAPE de SAE generan sensibilidades separadas; ambas producen P=1 frente a P=0 con RMSE, sin escoger ni promediar la métrica más favorable. `prior_contextual_sensitivity_retained.csv` es una copia byte a byte del artefacto anterior.

## Reproducción

Los cuatro scripts buscan primero la variable `E2_REPOSITORY_ROOT` y, si no está definida, recorren los ancestros de su propio archivo hasta encontrar el repositorio que contiene `book_e2`. La raíz configurada explícitamente debe contener ese directorio; una ruta inválida produce un error claro. Cada script escribe en su propio directorio, también los dos ubicados en `contextual_p`. No dependen del directorio de trabajo actual ni de una ruta absoluta de esta sesión.

Una vez publicada la carpeta dentro del repositorio, ejecutar desde ella, en este orden:

```bash
python build_citation_audit.py
python contextual_p/build_contextual_p.py
python contextual_p/build_preferred_intrapaper_scenario.py
python build_conditional_score_scenarios.py
```

Si los scripts están en una carpeta de caché o salida externa al repositorio, definir antes `E2_REPOSITORY_ROOT` con la ruta del checkout que contiene los insumos. Los manifiestos conservan SHA256 de los insumos; el protocolo del score usa referencias relativas con prefijos `repository:` y `artifact:`. Antes de puntuar, el integrador comprueba los hashes del CSV P9 y de la regla contra su recibo de generación, y el hash de los conjuntos de comparación contra la regla fija. Los scripts no utilizan la red. Los recibos de generación incluyen tiempos nuevos al reproducir; la validación de portabilidad compara byte a byte los CSV y las fuentes extraídas, manteniendo intacta la regla retrospectiva fija.
