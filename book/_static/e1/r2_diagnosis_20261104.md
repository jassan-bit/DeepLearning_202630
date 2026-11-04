# Diagnóstico independiente de R² — primer ciclo DL

Lectura de fuentes, protocolos, curvas y checkpoints existentes; cero entrenamientos nuevos y cero lecturas de raw/test2025+. No se modificó ningún archivo firmado del experimento. Los cálculos se hicieron con los CSV de desarrollo registrados y replay de checkpoints, sin ajuste de parámetros.

## El cálculo de R² es correcto

El runner calcula `1 - SSE / sum((y - mean(y))**2)` en unidades originales, después de invertir la escala de Y. Para las mismas 184 fechas y cada semilla, esto equivale a `1 - MSE / var(y, ddof=0)`. El R² medio publicado es la media de los R² por semilla. La desviación poblacional del objetivo julio–diciembre es 0,0083947979291: una RMSE superior a ese valor produce R² negativo. LSTM RMSE media0,00890346 queda ligeramente por encima; MLP/CNN mucho más. La media del periodo es el denominador descriptivo de R², no una referencia desplegable que pudiera conocerse al origen.

Persistencia tiene RMSE0,00482248, R²0,67000, correlación0,83475 con el objetivo y sesgo+0,00005116; su ventaja es real dentro del desarrollo. Usa ocho cierres para σ7 previa, mientras las redes D7 observan siete días OHLCV. No se debe corregir la métrica, recortar valores negativos ni cambiar el objetivo para hacerla favorable.

## Hallazgo principal: extrapolación del nivel de precio

El preprocesamiento aplica log a OHLC y estandariza con los días únicos de2023 admitidos. Esto mantiene el nivel de precio absoluto, aun si se reduce su escala.

| Canal | Máximo z en ventanas de entrenamiento | Media z julio–diciembre2024 | Mínimo z julio–diciembre2024 |
|---|---:|---:|---:|
| Open | 2,38569 | 4,77471 | 3,42910 |
| High | 2,35897 | 4,80111 | 3,42751 |
| Low | 2,39966 | 4,75999 | 3,01117 |
| Close | 2,37859 | 4,78735 | 3,42510 |

En los cuatro precios, incluso el mínimo de la evaluación es superior al máximo del entrenamiento. La evaluación exige extrapolar un nivel de BTC no visto. El volumen estandarizado presenta media−0,70455 en evaluación y rangos que se solapan con train; el problema más extremo está en OHLC. Esto es evidencia de desplazamiento y una causa plausible de extrapolación espuria, no una demostración causal de que una única modificación resolverá el desempeño.

## Sesgo y falta de seguimiento temporal

Promedios de los estadísticos de cada una de las30semillas; `bias²/MSE` se calcula por semilla y luego se promedia.

| Modelo | Media predicha | Media real | Sesgo pred−real | SD predicha | SD real | Correlación media | Fracción MSE por sesgo² |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0,03518810 | 0,02316302 | +0,01202508 | 0,00764929 | 0,00839480 | 0,16094 | 55,28% |
| LSTM | 0,02578369 | 0,02316302 | +0,00262067 | 0,00079573 | 0,00839480 | 0,14765 | 12,53% |
| CNN1D | 0,03628650 | 0,02316302 | +0,01312348 | 0,00761669 | 0,00839480 | 0,10275 | 56,82% |

MLP y CNN1D sobreestiman el nivel de volatilidad y una parte grande de su MSE es sesgo. LSTM es casi constante: la SD de sus predicciones es aproximadamente9,5% de la SD real. No sigue las subidas y bajadas del objetivo. La correlación baja de las tres sugiere que corregir solo la media de salida no bastaría. No calibrar interceptos con julio–diciembre y reportar luego sobre esas mismas fechas como si fuera confirmación.

## Ajuste y parada: límites del primer presupuesto

| Familia | Mejores épocas: mediana | Media | Mínimo–máximo |
|---|---:|---:|---:|
| MLP | 22 | 21,33 | 3–43 |
| LSTM | 3 | 3,73 | 1–14 |
| CNN1D | 11,5 | 17,50 | 1–57 |

Para la semilla designada1001, las verdaderas curvas internas de2023 en su época elegida registran:

| Familia | Época elegida | RMSE train interno | RMSE validación interna | RMSE train final del refit |
|---|---:|---:|---:|---:|
| MLP | 22 | 0,00725094 | 0,00784684 | 0,00736031 |
| LSTM | 3 | 0,00765287 | 0,00845441 | 0,00756980 |
| CNN1D | 8 | 0,00830863 | 0,00792544 | 0,00801279 |

El reajuste final usa otra inicialización, más muestras343vs251 y nuevos escaladores. Transferir únicamente el número de épocas elegido es una decisión válida y registrada, pero no garantiza el mismo grado de convergencia, especialmente con solo3épocas. No es un bug del cálculo de R². Tampoco basta aumentar épocas indiscriminadamente: la curva interna de LSTM empeora después de su mínimo y la deriva de entradas no desaparece con más entrenamiento.

El RMSE del propio entrenamiento final es peor que la persistencia calculada en esas mismas fechas (0,00435268); esto señala que el primer modelo no capturó bien una referencia fuerte incluso dentro del ajuste. La persistencia es un comparador contextual y no sustituye una comparación justa de información.

## Propuesta acotada para V2, sin tocar V1 ni el test

1. **Prioridad: representación invariante al nivel de precio.** Para cada ventana ya observada, transformar cada OHLC como `log(P_i / C_last)` con C_last el último cierre disponible de esa ventana. Mantener D×5 canales, el volumen, la misma transformación para las tres redes y escaladores ajustados solo con2023. Esta normalización elimina el nivel absoluto y preserva relaciones entre los precios dentro de la ventana. El denominador es observable antes del resultado; no añade rezagos explícitos del objetivo ni cambia la ruta tipo4. Comparar la representación original frente a esta variante en una ablación de desarrollo, conservando configuraciones y semillas comunes.
2. **Separar la evidencia de optimización.** Revisar train loss y error interno, inicialización de la cabeza, tamaño de learning rate y regla de transferencia de duración. Probar un número mínimo de épocas o una regla de selección robusta a semillas solo mediante particiones cronológicas previas; no forzar mínimos después de ver resultados para declararlos confirmados. Una segunda etapa puede seleccionar duraciones con medianas entre semillas y registrar presupuesto idéntico por familia.
3. **Actualizar con pasado disponible como experimento distinto.** Un esquema de reajuste creciente durante2024, con embargo D y parada interna cronológica, podría incorporar regímenes más recientes. Debe compararse como otro protocolo y no atribuir sus mejoras exclusivamente a arquitectura. No mezclarlo con el efecto de la normalización si se desea saber qué corrigió el problema.
4. **Mantener objetivos y evaluación.** σ7/ddof0/h1, mismas fechas dentro de cada comparación, RMSE principal y R² complementario, sin seleccionar por el R² más bonito ni quitar días difíciles. El periodo ya expuesto continúa siendo desarrollo; reservar confirmación posterior según protocolo cerrado.
5. **No proponer como arreglo invisible** añadir σ7(d−1), porque sería una nueva entrada objetivo y exigiría revisar el contrato tipo4; ni añadir un cierre anterior extra a una sola red. Si se desea esa hipótesis, se registra aparte para todas las familias.

Una mejora de R² es posible, pero no se garantiza R²>0 ni superar persistencia hasta ejecutar la comparación. La prioridad empírica es quitar la dependencia espuria del nivel nominal de BTC y comprobar que la red aprende variación temporal útil.
