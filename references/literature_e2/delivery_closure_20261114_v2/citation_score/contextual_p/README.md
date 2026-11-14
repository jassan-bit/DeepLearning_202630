# Evaluación contextual de P, propuesta v2

Hay resultados externos comparables dentro de las tablas de los artículos para **9 de los 10 sistemas**, sin exigir que el objetivo sea BTC σ7. Esto permite describir rendimiento en el contexto de sus arquitecturas. No completa un P universal comparable entre los diez sistemas: las tareas, métricas, variantes y conjuntos de comparación siguen siendo distintos. Se conservaron vacíos todos los P originales y no se calculó un nuevo score total.

Los 27 candidatos numéricos están en `current_model_contextual_P_candidates.csv`; los 89 valores que fijan sus denominadores, en `source_local_performance_pools.csv`. Cada `pool_id` identifica una normalización propia, con `P_contextual_source_local=(máximo-error)/(máximo-mínimo)`. El CSV incluye también todos los comparadores de cada conjunto. Un 0 observado indica el peor resultado de ese conjunto; una métrica ausente sigue vacía.

## Clasificación

[Wang et al., versión 1611.06455v4](https://arxiv.org/html/1611.06455v4), Table I, publica MPCE agregado sobre los 44 datasets UCR. Se usan los **11 candidatos** DTW, COTE, MCNN, BOSSVS, PROP, BOSS, SE1, TSBF, MLP, FCN y ResNet. FCN registra 0.0219 y ResNet 0.0231; extremos 0.0219/0.0407. Sus P contextuales son 1.0000 y 0.9362. Se evita escoger una única fila favorable, como Adiac. P es una transformación nueva del MPCE publicado; el artículo no publica este P. Ambos sistemas proceden de un solo estudio. La prueba por datasets del artículo no establece una diferencia significativa entre FCN y ResNet ni incertidumbre por semillas.

[InceptionTime, versión 1909.04939v3](https://arxiv.org/html/1909.04939v3), Figure 10, compara conjuntos de cinco redes: 54 victorias, 8 empates, 23 derrotas sobre 85 datasets frente a ResNet(5). El crédito descriptivo `(54+8/2)/85=0.68235` se conserva en JSON con su variante. No se atribuye a la red individual Inception-1D actual. Figure 11 incluye InceptionTime(1), pero no se extrajo una tabla numérica suficiente para asignarle P. Por ello este sistema sigue sin candidato verificable de P individual.

## Forecasting

[LSTM-SAE, Scientific Reports 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911101/), Table 2, permite comparar cuatro configuraciones sobre Beijing PM2.5. Se mantienen RMSE, MAE y SMAPE separados. La variante SAE de dos capas obtiene 24.041/12.060/9.864: es la peor en RMSE y la mejor en MAE y SMAPE, de modo que sus P locales son 0/1/1. No se promedian ni se escoge la métrica retrospectivamente. Los lags y tamaños de red seleccionados varían entre filas; la cronología del split y la igualdad de presupuesto HPO no quedan verificados.

La extracción local verificada de [iTransformer v4](https://arxiv.org/abs/2310.06625v4), Table 10, p23, contiene los 11 modelos del mismo experimento ETTh1/H96/L96. Incluye iTransformer, TimesNet, PatchTST y TiDE. Para MSE: 0.386, 0.384, 0.414 y 0.479, respectivamente. No se sustituyen por las cifras de sus artículos originales. El caso TiDE muestra por qué importa esta separación: su original con L720 registra 0.375 y esta reproducción con L96 registra 0.479.

[TimeMixer v1](https://arxiv.org/html/2405.14616v1), Table 13, ofrece ETTh1/H96 con input96 y parámetros unificados declarados para 11 modelos; incluye TimeMixer, TimesNet y PatchTST. Table 23 agrega cuatro modelos, incluido TSMixer reproducido por esos autores en PyTorch. Se mantienen conjuntos separados, con MSE y MAE separados; no se mezclan columnas de búsqueda ni el TSMixer original. La escala global y el costo total HPO siguen sin resolverse.

[TSMixer v5](https://arxiv.org/html/2303.06053v5), Table 3, también compara TSMixer y PatchTST sobre ETTh1/H96, ambos con L512. Se usan siete candidatos y se omite TMix-Only conforme al pie de tabla. Algunos comparadores se importan de PatchTST y conservan otros contextos. Es una descripción del conjunto publicado; no se declara que todos fueron reentrenados en condiciones iguales. El resultado cambia si cambia el conjunto de comparadores.

## Interpretación y estado

El protocolo es nuevo y posthoc. No causó la selección del catálogo, no reemplazó P, no leyó resultados A2 y no demuestra eficacia de las adaptaciones BTC. Se revisaron las tablas HTML identificadas y sus pies; se reutilizó la extracción PDF local de iTransformer. No se afirma una nueva lectura íntegra de cada paper ni revisión humana.

La mejora frente al artefacto previo es distinguir «tarea distinta de BTC» de «ausencia de una comparación documental válida». Clasificación, contaminación y ETTh1 pueden tener su propio contexto de rendimiento. Una puntuación conjunta requeriría fijar antes la métrica y el conjunto comparable por estrato, justificar su calibración y mantener explícita la cobertura incompleta. Con la evidencia disponible, `P_contextual_global` permanece vacío para los diez modelos.

Ejecutar `python build_contextual_p.py` regenera los CSV/JSON y comprueba rangos, cobertura 9/10, conservación de P y ausencia de uso de A2. `protocol_and_provenance.json` registra los SHA256 de las entradas locales. Sólo se escribieron artefactos en este directorio.
