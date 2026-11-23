# Fuentes y documentación

## Material de los enunciados

**Demostraciones Bonificacion.docx.** Documento proporcionado para elaborar este libro. Contiene ocho imágenes agrupadas en siete temas: entrenamiento de MLP, derivada de la sigmoide, tamaño de paso, gradientes convolucionales, retropropagación, campo receptivo y beneficio de la profundidad.

Las imágenes se reproducen en el [anexo de enunciados originales](enunciados_originales.md). Los desarrollos matemáticos de los capítulos son demostraciones elaboradas a partir de esos enunciados. Los datos de las comprobaciones computacionales se identifican como ejemplos didácticos.

La captura del tamaño de paso menciona *Theodoridis, 2020, sección 5.3*. Esa cita forma parte del material original; el recorte no contiene la referencia bibliográfica completa. La prueba del capítulo se presenta de forma autocontenida mediante el gradiente, la Hessiana y la descomposición espectral de la matriz de segundos momentos.

## Herramientas de edición

- **Jupyter Book 1.** [Construcción del libro](https://jupyterbook.org/v1/start/build.html).
- **Jupyter Book 1.** [Estructura de la tabla de contenidos](https://jupyterbook.org/v1/structure/toc.html).
- **Jupyter Book.** [Migración de la configuración de la versión 1 a la versión 2](https://jupyterbook.org/latest/resources/upgrade/).
- **MathJax 3.2.** [Uso de una copia local del motor matemático](https://docs.mathjax.org/en/v3.2/web/hosting.html).

Este proyecto fija Jupyter Book 1.0.4.post1 para conservar el formato de configuración `_config.yml` y `_toc.yml`. La versión web utiliza MathJax 3.2.2 desde el mismo CDN que los otros libros del repositorio. Estas versiones se eligen para la reproducción del proyecto; no se presentan como las versiones más recientes.
