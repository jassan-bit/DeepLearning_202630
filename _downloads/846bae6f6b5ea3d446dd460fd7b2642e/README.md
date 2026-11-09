# Recuperación documental focalizada: estudio S8

Esta carpeta usa la fecha UTC 05/11/2026; la lectura corresponde al 04/11/2026 en Bogotá (UTC−5). Los JSON registran ambas zonas. Es una revisión **posterior al benchmark**, sin cambios de modelos, pesos, predicciones o decisiones históricas.

Se leyó el capítulo completo de **Ryotaro Miura, Lukáš Pichl y Taisei Kaizoji (2019), Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series**, ISNN2019, LNCS11554, pp.165–172. El DOI **10.1007/978-3-030-22796-8_18** figura en portada y metadatos PDF. No se acredita una consulta nueva al resolver, un registro WoS/Scopus ni un cuartil.

Fuente completa consultada, fijada al commit de una biblioteca académica:
https://raw.githubusercontent.com/Anton21a/Thesis/8bc3aa3c96fd214b34b914d2057f4339f2c8ed1e/ML%20articles/Artificial%20Neural%20Networks.pdf

SHA-256: `2364a462ed212803e2c4b751d27d66091af5773159c601f950df17578853dfeb`; 1.678.552 bytes. El PDF y su texto completo permanecen fuera de esta carpeta pública. Se entregan hechos bibliográficos, extracción, valores numéricos y enlaces; se conserva el copyright Springer2019.

## Archivos

- `primary_source.json`: identidad, fuente y comprobante de lectura.
- `extraction.json`: datos, objetivo, arquitecturas, localizaciones, protocolo y campos no reportados.
- `quality_assessment.json`: evaluación por dimensiones; no certifica ausencia de fuga.
- `miura_table1.csv`: ocho filas de Tabla1, p.171: **32 valores MSE/RMSE y diez varianzas**. Las varianzas no son desviaciones estándar.
- `synthesis_six_columns.csv`: las cinco filas históricas más S8.
- `family_frequencies.csv`: seis estudios; un conteo por familia y estudio, sin duplicar variantes.
- `references_access_36.json` y `.csv`: los 35 registros anteriores más `miura2019`.
- `references_36.bib`: las 36 referencias de la bibliografía vigente del libro.
- `selection_update.json`: apoyo documental y límites de la adaptación del proyecto.
- `search_retrieval_trace.json` y `retrieval_flow.svg`: recorrido focalizado y candidatos contextuales; búsquedas y repositorios no son artículos.
- `manifest.json`: inventario SHA-256 de los metadatos públicos.

## Resultado y límites

S8 proporciona una ocurrencia aplicada de **MLP independiente** y otra de **CNN independiente de dos capas más Dense** para RV de Bitcoin. El corte pasa de cinco a seis textos. No prueba uso ampliamente recurrente, selección previa al entrenamiento ni réplica de nuestras redes.

**El capítulo no especifica la dimensión de la CNN original**, sus filtros, kernels, padding, causalidad o anchuras. Por eso se separan CNN independiente 1/6 y CNN1D explícitamente verificada 0/6. La Conv1D causal del proyecto es una adaptación declarada.

El artículo suma retornos minuto cuadrados en intervalos de tres horas y usa RV pasada. El objetivo sigma7 diario y las entradas OHLCV del proyecto son distintos. Las cifras publicadas no se ordenan contra nuestras métricas como si fueran una misma evaluación.

El flujo anterior 118−12=106 repositorios y 7=5+2 textos se conserva como corte histórico. El diagrama nuevo documenta una incorporación, sin inventar un PRISMA combinado. El manuscrito de Djanga2025 recuperado durante la revisión cita Aaltio2022; **no se le atribuye una cita a Miura que no aparece allí**.
