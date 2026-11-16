# Revisión focal de novedad y recomendación después de V2

Fecha real de revisión: **16 de noviembre de 2026**. Se leyeron los registros públicos de main y fuentes originales con el alcance indicado abajo. No se modificaron modelos ni se ejecutaron nuevos entrenamientos. La base permanece **CNN causal → LSTM unidireccional → atención temporal por contenido**, como M00.

Esta revisión complementa la [ficha V2](../../e3_ficha_cnn_lstm_atencion_v2_20261116/README.md) y la [revisión posterior a V1](../../e3_revision_post_piloto_20261116/README.md). No convierte las lecturas heredadas en lecturas nuevas ni una búsqueda web general en una revisión sistemática de tres bases. No se realizó verificación humana externa.

## Evidencia consultada y alcance

Se consultaron en GitHub los registros sources.json de la ficha y review_matrix.json de la revisión anterior. [sources.json](sources.json) registra sus SHA de blob observados, nueve antecedentes, metadatos, localizaciones y límites de lectura. [search_log.json](search_log.json) conserva las nueve consultas realmente realizadas: cuatro del revisor principal y cinco del revisor focal delegado. Todas fueron búsquedas web generales; no se informan consultas nativas a Scopus, Web of Science o IEEE ni un número ficticio de resultados.

Los originales de ConstraintNet, DeepAR y GARCH-NN se reabrieron y revisaron focalmente. El cuerpo escaneado de folded-t, antes documentado sólo mediante metadatos y cubierta, recibió ahora una lectura visual focal delegada a otro agente de IA de las páginas 2718–2720. GARCHNet se reabrió en esa revisión delegada. Chan, SurVAE y Spulbar–Ene conservan sus lecturas focales previas, identificadas como heredadas. No se atribuye una lectura íntegra de estos artículos ni una revisión humana independiente.

La búsqueda encontró además el original de **Acevedo-Prins y Velásquez**, publicado el **4 de septiembre de 2026**, en *Journal of Risk and Financial Management*, **19(9), artículo 688**, [DOI 10.3390/jrfm19090688](https://doi.org/10.3390/jrfm19090688). Se verificaron esos metadatos en la página editorial y se leyó focalmente la sección 2.3, ecuaciones 6–8, junto a la definición del objetivo de la sección 3.1. Se usa como antecedente de diseño; no se adoptan sin revisión sus pruebas de equivalencia ni sus conclusiones empíricas.

## Matriz: mecanismo conocido, adaptación y evidencia pendiente

| Componente | Antecedente comprobado | Qué pertenece a esta adaptación y qué sigue pendiente |
|---|---|---|
| Separar seis retornos conocidos y uno desconocido | [Chan, Golub y LeVeque (1983)](https://doi.org/10.1080/00031305.1983.10483115), p.242, ecuaciones 1.3a–b: actualización de suma de cuadrados. | El coeficiente 6/49 y el piso de sigma7 especializan esa identidad. No son por sí mismos una propiedad matemática nueva ni un predictor de la innovación. |
| Piso dependiente de la muestra más salida positiva | [ConstraintNet (2021)](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), pp.6814–6816: dominio por muestra y guarda escalar exp(z)+b. | El piso A procede de nuestro objetivo. Sustituir exponencial por Softplus y conservar una red convencional no acredita por sí solo una modificación algorítmica sustantiva. |
| Escala positiva producida por una red | [DeepAR, versión 2019](https://arxiv.org/html/1704.04110), sección 3.1; [GARCHNet](https://link.springer.com/article/10.1007/s10614-023-10390-7), sección 2.3. | La lectura de M00 se usa como escala de magnitud. Red→parámetro positivo y Softplus ya tienen antecedentes; no se reclama novedad por ellos. |
| Magnitud Student plegada | [Psarakis y Panaretos (1990)](https://mpra.ub.uni-muenchen.de/6257/1/MPRA_paper_6257.pdf), sección 2, ecuación 2.1. | Modelar U como una half-t de escala es una hipótesis de esta ficha sobre una familia conocida. No identifica signos ni acredita una distribución correcta para BTC. |
| Transporte de densidad y pérdida de signo | [SurVAE (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/9578a63fbe545bd82cc5bbe749636af1-Paper.pdf), sección 2 y [apéndice D original](https://arxiv.org/pdf/2007.02731). | El enlace Y=sqrt(A+cU²) se deriva para el objetivo heredado. No se adopta una arquitectura flow ni se atribuye al artículo el predictor exacto de esta ficha. |
| Student-t para volatilidad neuronal | [GARCH-NN (2024)](https://arxiv.org/html/2402.06642), ecuaciones 4–7; GARCHNet, ecuaciones 19–23. | La varianza condicional y la escala Student de esas fuentes deben distinguirse de sigma7 y de s. La elección nu=5 no constituye una innovación. |
| Mezclar error puntual y Student-NLL | [Acevedo-Prins y Velásquez (2026)](https://www.mdpi.com/1911-8074/19/9/688), sección 2.3, ecuaciones 6–8. | Allí se combinan error de varianza frente a retorno cuadrado y likelihood del retorno; aquí, MSE sobre sigma7 y NLL de U normalizada. La diferencia existe, pero no acredita automáticamente novedad de la combinación. |
| Parte conocida de un objetivo móvil | [Spulbar y Ene (2026)](https://www.mdpi.com/2227-7390/14/20/3661), secciones 3.1 y 4.3. | Ese antecedente usa cinco retornos cuadrados sin centrar. La fórmula centrada de sigma7 debe distinguirse, sin presentar como inédita la idea general de incorporar historia ya observada. |
| Media de sigma mediante cuadratura positiva | Regla numérica y propiedades verificadas en la ficha V2; integración y mezclas positivas son herramientas conocidas. | La normalización y las cotas del operador documentan una implementación verificable. No demuestran novedad, una cota de toda la red o superioridad estadística. |

## Por qué el mejor RMSE no resuelve R1

El control additive_positive_head impone un piso observable y añade una salida positiva del mismo backbone. Eso puede mejorar coherencia y predicción, pero ambas propiedades son distintas de demostrar una contribución original. ConstraintNet ya formula una guarda escalar desplazada; la identidad que produce el piso es conocida. El cambio de activo, definición de objetivo, activación o escala requiere una justificación adicional para superar el criterio de adaptación algorítmica de la guía.

Por tanto, **no se promueve ese control a “modelo original” por haber obtenido la menor RMSE**. Tampoco se concluye que toda combinación posible de esta ficha carezca de novedad: esta revisión limitada no prueba ausencia o existencia de un antecedente idéntico. Para R1 faltan completar la revisión exigida, comparar mecanismos cercanos con precisión y obtener la verificación humana externa. La comprensión y defensa académica de R2 y las condiciones de R3 permanecen separadas.

El enlace específico de V2 combina etiqueta de magnitud recuperable, densidad condicionada por A, media aproximada de sigma y pérdida mixta en el backbone preservado. Esa combinación se declara **adaptación propia asistida por IA, bajo revisión**. Esta revisión no la atribuye retrospectivamente a un paper ni acredita que sea inédita.

## Recomendación de selección y lectura de regímenes

El [resumen público de V2](https://github.com/jassan-bit/DeepLearning_202630/blob/e3-desarrollo-resultados-v2/runs/innovacion_v2/summary.json), consultado en esta revisión, identifica additive_positive_head como el mejor control por RMSE global: 0.0031446147843954175. innovation_density_mean obtuvo 0.003198296978507994, aproximadamente 1.707 % más error; **no superó la regla exploratoria de continuar con al menos 2 % de mejora sobre el mejor control**. Se conserva ese resultado adverso para la hipótesis de la cabeza propuesta.

La recomendación es conservar **additive_positive_head como referencia predictiva de desarrollo**, dentro de la misma CNN–LSTM–atención, y no presentarlo como originalidad acreditada o ganador confirmatorio de E3. La comparación tiene una semilla, seis alternativas y validación ya usada para seleccionar época y orientar el diseño; no equivale a evaluar la familia completa de E2 bajo el protocolo final.

El resumen también muestra que frozen_M00_projected presenta menor RMSE en las 18 fechas de alta volatilidad realizada que el control aditivo. Es una limitación de la recomendación global y debe reportarse, sin ocultar que distintos errores favorecen alternativas diferentes. **Ese grupo se define usando el objetivo realizado del día**, comparado con un umbral fijado en ajuste: sirve para diagnóstico posterior, no está disponible para escoger modelo al emitir el pronóstico. No se propone un router que utilice esa etiqueta futura.

Las coberturas nominales cercanas al 90 % de las dos variantes probabilísticas no acreditan calibración con esta muestra dependiente. Tampoco una diferencia por régimen establece significancia. Se mantienen los resultados y las recetas ensayadas; esta revisión no selecciona más semillas, épocas, métricas o arquitecturas para buscar una victoria.

La acción siguiente autorizada es cerrar el informe con desempeño y límites por régimen, procedencia y estado de admisibilidad. La selección de un modelo original para E3 sigue pendiente de evidencia de mecanismo y revisión; no se sustituye esa evidencia por el color de una tabla o por una mejora puntual de RMSE.

## Reproducibilidad documental

[references.bib](references.bib) contiene los nueve registros utilizados. [sources.json](sources.json) identifica el alcance de cada lectura y [search_log.json](search_log.json) conserva consultas, accesos y límites. Se citan y parafrasean los originales, sin copiar implementaciones ni redistribuir sus PDF. No se entrenaron modelos, no se cambiaron resultados y no se publicó al repositorio desde esta revisión.
