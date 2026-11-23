# Fuentes y documentación

## Material de los enunciados

**Demostraciones Bonificacion.docx.** Documento proporcionado para elaborar este libro. Contiene ocho imágenes agrupadas en siete temas: entrenamiento de MLP, derivada de la sigmoide, tamaño de paso, gradientes convolucionales, retropropagación, campo receptivo y beneficio de la profundidad.

Las imágenes se reproducen en el [anexo de enunciados originales](enunciados_originales.md). Los desarrollos matemáticos de los capítulos son demostraciones elaboradas a partir de esos enunciados. Las comprobaciones auxiliares del cuaderno se identifican como ejemplos didácticos; los datos del ejercicio MLP proceden del PDF descrito a continuación.

La captura del tamaño de paso menciona *Theodoridis, 2020, sección 5.3*. Esa cita forma parte del material original; el recorte no contiene la referencia bibliográfica completa. La prueba del capítulo se presenta de forma autocontenida mediante el gradiente, la Hessiana y la descomposición espectral de la matriz de segundos momentos.

(fuente-mlp-pdf)=
## Ejercicio numérico del MLP

**Cinco iteraciones de gradiente descendente.** PDF de seis páginas proporcionado como material del ejemplo de clase, con nombre de archivo *Ejercicio_para_el_estudiante__iteraciones_de_gradiente_descendente (1).pdf*.

La página 1 especifica las entradas $y_1=-1$, $y_2=2$, el objetivo $y=1$, la tasa $\alpha=0.01$, los seis pesos iniciales y los diez nodos del grafo. Las páginas 1 y 2 establecen la derivada correcta de $L=(y-\hat y)^2$ y los seis gradientes. Las páginas 2 y 3 desarrollan la primera actualización; las páginas 3 y 4 contienen las tablas de nodos, gradientes y estados. Las páginas 5 y 6 documentan la reducción de la pérdida y el criterio de parada.

El [capítulo de entrenamiento de MLP](../capitulos/01_entrenamiento_mlp.md) reproduce estos datos y resultados. Se ha comprobado la coincidencia de la tabla a seis decimales y la primera satisfacción de $|L^{(t)}-L^{(t-1)}|<10^{-6}$ en $t=255$. La implementación y las tablas de mayor precisión se ofrecen como descargas en ese capítulo.

## Herramientas de edición

- **Jupyter Book 1.** [Construcción del libro](https://jupyterbook.org/v1/start/build.html).
- **Jupyter Book 1.** [Estructura de la tabla de contenidos](https://jupyterbook.org/v1/structure/toc.html).
- **Jupyter Book.** [Migración de la configuración de la versión 1 a la versión 2](https://jupyterbook.org/latest/resources/upgrade/).
- **MathJax 3.2.** [Uso de una copia local del motor matemático](https://docs.mathjax.org/en/v3.2/web/hosting.html).

Este proyecto fija Jupyter Book 1.0.4.post1 para conservar el formato de configuración `_config.yml` y `_toc.yml`. La versión web utiliza MathJax 3.2.2 desde el mismo CDN que los otros libros del repositorio. Estas versiones se eligen para la reproducción del proyecto; no se presentan como las versiones más recientes.
