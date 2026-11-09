# Repetibilidad actual del subpanel diagnóstico E1 D7

Las dos réplicas independientes completaron las seis celdas prepublicadas:
**12 trayectorias, 24 etapas de ajuste**, sin reintentos ni sustituciones.
Coinciden los 18/18 hashes canónicos de curvas inner/refit y tensores A–B, así
como los 18/18 frente a la reproducción completa anterior en cada réplica.
Las diferencias máximas absolutas y relativas A–B son cero. También pasa el
criterio secundario fijado (`rtol=1e-6`, `atol=1e-8`) en las seis celdas.

| Familia | Semilla | Época original | Reproducción completa | Nueva A | Nueva B |
|---|---|---|---|---|---|
| GRU | 8101 | 65 | 65 | 65 | 65 |
| LSTM | 8101 | 12 | 12 | 12 | 12 |
| TCN | 8101 | 75 | 75 | 75 | 75 |
| TCN | 8110 | 74 | 19 | 19 | 19 |
| TCN | 8416 | 55 | 6 | 6 | 6 |
| TCN | 8552 | 59 | 97 | 97 | 97 |

El protocolo y ejecutor se publicaron en el commit
`79f8dcc1a4c3bfacfd1fc56c0ecb43d8e0185501`; se observó su existencia remota a las
04:15:54 UTC. El registro privado ocurrió a las 04:16:30 UTC, antes del primer
ajuste a las 04:16:42 UTC. A terminó a las 04:17:20 UTC y B empezó en otra
invocación de Python a las 04:17:27 UTC, terminando a las 04:18:00 UTC. Cada CLI
creó tres procesos `spawn` nuevos y registró su entorno efectivo. Se mantuvieron
intactos el protocolo, código científico y checkout publicado. El recibo conserva
la observación remota; la fecha del commit sola no demuestra cuándo se publicó.

La reproducción previa utilizó cuatro trabajadores; estas dos réplicas usaron
tres y reprodujeron los mismos resultados en el subpanel. Ese cambio no explicó
la discrepancia histórica en estas seis celdas bajo el entorno actual. La causa
exacta de los cambios frente al estudio original continúa sin identificar.
Los 18 hashes de archivo de curvas/checkpoints de cada réplica siguen distintos
de los históricos. No se dispone de los payloads originales para comparar sus
valores o tensores; no se afirma igualdad canónica con el original.

El subpanel es un diagnóstico dirigido: tres familias con la primera semilla y
las tres mayores diferencias de época ya observadas en TCN. Sus dos réplicas no
agregan muestras independientes al benchmark ni acreditan repetibilidad del panel
completo R=465, exactitud de la reserva o recuperación de curvas originales.
Los resultados históricos, baseline y margen académico permanecen intactos.
Se leyeron sólo derivados de desarrollo 2023–2024: cero lecturas de precios de la
reserva, cero puntuaciones de reserva y cero pronósticos emitidos. Las pasadas
forward para pérdidas de entrenamiento/validación son parte intrínseca del ajuste.

`comparison.csv` conserva las épocas y comparaciones exactas y numéricas.
`artifact_hashes.csv` publica 72 hashes verificados contra los completados
privados; los hashes de preprocesamiento no revelan valores de escaladores.
`registration.json`, `execution.json`, `runtime.json` y
`publication_verification.json` documentan registro, orden y entorno. El manifiesto
público admite exclusivamente estos metadatos, resúmenes y hashes; no exporta
pesos, escaladores, observaciones ni predicciones diarias.
