# Demostraciones de aprendizaje profundo

## Bonificación

**Jassan Alberto Arteta Chavarro**

[Volver al selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html)

Este libro desarrolla los ejercicios y resultados reunidos en *Demostraciones Bonificacion.docx* y el ejemplo numérico del PDF *Cinco iteraciones de gradiente descendente*. El recorrido conecta el descenso de gradiente y la derivada de la sigmoide con la retropropagación, los gradientes de una capa convolucional, el campo receptivo y la capacidad de representación de las redes profundas.

Cada capítulo presenta el enunciado, establece las condiciones necesarias y muestra los pasos de la demostración. Se conserva la notación de las capturas y su numeración original cuando existe. Las verificaciones numéricas complementan las demostraciones; no sustituyen sus argumentos.

## Contenido

| Capítulo | Pregunta que responde |
|---|---|
| [Entrenamiento de MLP](capitulos/01_entrenamiento_mlp.md) | ¿Cómo se calculan y registran cinco actualizaciones de los seis pesos? |
| [Derivada de la sigmoide](capitulos/02_derivada_sigmoide.md) | ¿Por qué $f'(z)=a f(z)(1-f(z))$? |
| [Tamaño de paso](capitulos/03_tamano_paso.md) | ¿Qué condición sobre $\mu$ garantiza la convergencia del descenso de gradiente cuadrático? |
| [Gradientes convolucionales](capitulos/04_gradientes_convolucionales.md) | ¿Cómo se deriva la pérdida respecto del filtro, la entrada y el sesgo? |
| [Retropropagación](capitulos/05_retropropagacion.md) | ¿Por qué el gradiente se obtiene mediante un recorrido inverso? |
| [Campo receptivo](capitulos/06_campo_receptivo.md) | ¿Cómo se acumula el alcance de los filtros y sus strides? |
| [Beneficio de la profundidad](capitulos/07_beneficio_profundidad.md) | ¿Cómo produce una red de ancho dos un número exponencial de tramos lineales? |

## Dos precisiones para la lectura

**Ejercicio del MLP.** El primer capítulo utiliza los datos y el grafo del PDF proporcionado. Incluye los diez nodos, los seis gradientes, la primera actualización paso a paso y la tabla completa de los estados $t=0,\ldots,5$. Para $L=(y-\hat y)^2$ se conserva el factor $2$ de la derivada: $\partial L/\partial\hat y=2(\hat y-y)$. La pérdida disminuye aproximadamente un $9.46\%$ en cinco actualizaciones; el primer cambio entre pérdidas consecutivas inferior a $10^{-6}$ se alcanza en $t=255$, es decir, tras $250$ actualizaciones adicionales.

**Convención del tamaño de paso.** El capítulo correspondiente distingue el gradiente de $J$ del gradiente de $J/2$. Esta precisión permite interpretar correctamente el factor dos de la cota del enunciado.

## Cómo usar el libro

Lee los capítulos en el orden del índice o accede directamente a un resultado. El [cuaderno de comprobaciones actual](notebooks/comprobaciones_v2.ipynb) contiene cálculos reproducibles. El {ref}`capítulo del MLP <reproducir-los-resultados>` ofrece además el programa específico del ejemplo, sus tablas CSV y la comprobación de la tolerancia. El [anexo de enunciados originales](anexos/enunciados_originales.md) permite contrastar las transcripciones con las capturas del documento.

El [cuaderno inicial](notebooks/comprobaciones.ipynb) se conserva en los anexos como archivo histórico. La versión actual mantiene sus cálculos y salidas y actualiza dos notas para enlazar al ejercicio MLP ya completo.

Los símbolos se interpretan dentro de cada capítulo: por ejemplo, $L$ representa la pérdida en el ejercicio del MLP y el número de capas en algunos teoremas. Esta diferencia proviene de los enunciados originales.

La [guía de edición y compilación](anexos/guia_de_uso.md) explica cómo modificar las páginas y reconstruir el sitio.
