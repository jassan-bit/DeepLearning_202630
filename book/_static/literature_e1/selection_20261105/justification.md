# Selección condicionada por diversidad para el siguiente estudio D7

Decisión tomada el **04/11/2026 22:13:11 de Bogotá (UTC−5)**, equivalente a 05/11/2026 03:13:11 UTC. Es posterior al benchmark histórico y a las subsanaciones ya ejecutadas. El protocolo siguiente debe fijar el hash de `decision.json` **antes de comenzar su nuevo piloto o sus corridas definitivas**. No se presenta esta fecha como antecedente de experimentos anteriores.

## Regla de selección

Se reserva exactamente un lugar para cada papel: **una referencia feedforward plana**, **una referencia recurrente con memoria** y **una referencia convolucional local**. Dentro de un mismo papel se consideran la evidencia aplicada, la recurrencia del corpus y la simplicidad del diseño. Esta es una **selección condicionada por diversidad**, no un ranking mundial ni la elección de las tres familias con mayor frecuencia global.

1. **MLP:** representa interacciones no lineales sobre la ventana aplanada, sin estado recurrente ni compartición convolucional. S8 aporta una ocurrencia aplicada independiente.
2. **LSTM:** ocupa el lugar recurrente. En el corpus, la familia LSTM aparece en cinco de seis estudios y GRU en tres. Se elige LSTM dentro de ese lugar; no se declara que GRU sea inválida, idéntica o empíricamente inferior.
3. **CNN1D del proyecto:** representa filtros temporales locales compartidos. S8 acredita una CNN independiente de dos capas más Dense, pero **no especifica su dimensión**. Conv1D se adopta porque las entradas tienen estructura tiempo×covariables, sin una rejilla espacial. La causalidad, canales y kernels son decisiones explícitas del proyecto, no datos atribuidos al artículo.

La recurrencia permanece **MLP1/6, CNN genérica1/6 y CNN1D simple explícita0/6**. No se transforma una mención genérica en prueba de Conv1D. Tampoco se fusiona la CNN genérica de S8 con la TCN causal de S1 para inventar una configuración publicada que reúna ambas propiedades.

## Simplicidad verificable de las recetas

| Modelo D7 | Receta fijada | Parámetros implementados |
|---|---|---:|
| MLP | Una capa oculta de16, ReLU y dropout; salida escalar positiva | 593 |
| LSTM | Una capa unidireccional de32; último estado y salida escalar | 5.025 |
| CNN1D | Dos Conv1D causales de16canales/k3, promedio temporal; sin dilatación, residuos o atención | 1.057 |

Los conteos corresponden a **nuestro código con entradas7×5**, no al artículo. Demuestran capacidad paramétrica de estas implementaciones, sin acreditar que sean las configuraciones óptimas o que todos los artículos puedan ordenarse por complejidad cuando faltan anchuras y kernels.

GRU sigue siendo una alternativa válida dentro del lugar recurrente. La TCN documentada en S1 tiene seis bloques dilatados/residuales y anchura287; el lugar convolucional de este estudio se limita a la receta ordinaria de dos capas. El encoder de S2/S6 aporta atención, un papel que no forma parte de los tres lugares declarados. Su frecuencia2/6 se reconoce; no se inventan parámetros o resultados para descartarlo.

## Evidencia aplicada y adaptación

La fuente primaria de S8 es **Miura, Pichl y Kaizoji (2019), Artificial Neural Networks for Realized Volatility Prediction in Cryptocurrency Time Series**, ISNN2019, LNCS11554, pp.165–172, DOI impreso **10.1007/978-3-030-22796-8_18**. La lista de comparadores está en pp.168–169 y la Tabla1 en p.171. El texto completo consultado está fijado a un commit y hash en `decision.json`; no se redistribuye el PDF.

S8 usa RV de tres horas construida con retornos minuto de Bitcoin y entradas de RV pasada. El proyecto pronostica sigma7 diaria con OHLCV pasada. La evidencia es pertinente a las familias aplicadas a volatilidad de Bitcoin, **sin igualdad de objetivo, entradas, escala o configuración**. La arquitectura fuente de MLP tiene cuatro capas y la CNN no publica dimensionalidad, filtros, kernels o padding: nuestras recetas no son réplicas exactas.

S1 —Akbiyik y colaboradores, DOI10.1145/3539597.3570387— aporta LSTM, GRU y TCN causal/dilatada, con sus configuraciones y límites. Un nuevo examen focalizado no recuperó otro texto primario que permita ampliar los conteos MLP/CNN1D; el corpus permanece en **seis textos y36referencias**.

## Ajuste histórico y configuración del estudio siguiente

El HPO histórico tuvo **96corridas:32porfamilia**, con igual rejilla de cuatro historias, dos anchuras, dos learning rates y dos semillas. Se conserva como evidencia de esfuerzo de ajuste igualado en aquel procedimiento. No equivale a una búsqueda nueva en cada representación posterior ni demuestra optimalidad.

Para el siguiente estudio D7 se fijan **una receta por familia y los mismos controles de entrenamiento/evaluación**; no habrá otro ajuste de arquitectura dentro de esa comparación. El objetivo es comparar estas recetas de referencia, no estimar el mejor desempeño alcanzable por cualquier miembro de cada familia.

La guía4.2 permite configuración reportada o mínima justificada; aquí se declara esta segunda vía. La guía3.2 remite al Entregable2, sección14, cuando falte arquitectura fuente y exige declarar la búsqueda antes de entrenar. Este documento **no inventa esos detalles ni afirma haber ejecutado ahora esa búsqueda**. El límite entre apoyo de familia, adaptación mínima y reproducción exacta se mantiene visible.

Las nuevas semillas no crean una serie de mercado independiente ni convierten un calendario histórico ya examinado en test desconocido. El protocolo siguiente deberá definir el estimando condicional, el piloto, el dimensionamiento y las inferencias que realmente puedan sostenerse.
