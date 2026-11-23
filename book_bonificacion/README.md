# Demostraciones de bonificación

Libro de demostraciones de aprendizaje profundo de Jassan Alberto Arteta Chavarro.

Sitio publicado: https://jassan-bit.github.io/DeepLearning_202630/bonificacion/intro.html

## Contenido

Siete capítulos: entrenamiento de MLP, derivada de la sigmoide, tamaño de paso para el error cuadrático, gradientes convolucionales, retropropagación, campo receptivo y beneficio de la profundidad. Incluye los enunciados, demostraciones paso a paso, un cuaderno con salidas ejecutadas y las capturas originales del documento proporcionado.

El capítulo del MLP incluye el ejemplo numérico completo del PDF proporcionado *Cinco iteraciones de gradiente descendente*: entradas y pesos iniciales, diez nodos, seis gradientes, primera actualización desarrollada y tabla de los estados 0 a 5. Utiliza la derivada correcta de la pérdida, `2 * (yhat - y)`. La reducción de la pérdida en cinco actualizaciones es del 9.4557004%; el primer cambio consecutivo inferior a `1e-6` ocurre en el estado 255, con 250 actualizaciones adicionales después de las cinco solicitadas. Los resultados se acompañan de un programa específico, una figura y archivos CSV/JSON.

## Compilar desde el repositorio

Con Python 3.12, desde la raíz de `DeepLearning_202630`:

```bash
python -m pip install -r requirements-book.lock.txt
jupyter-book build book_bonificacion --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book_bonificacion
```

Abre `book_bonificacion/_build/html/intro.html` en un navegador. La configuración web carga MathJax 3.2.2 desde el mismo CDN que los otros libros del repositorio; la presentación de las ecuaciones requiere acceso a ese recurso.

Los capítulos se editan en `capitulos/`; el índice está en `_toc.yml` y los datos del libro en `_config.yml`. Las imágenes originales están en `anexos/capturas/`.

## Comprobaciones en Python

```bash
python -m pip install -r book_bonificacion/requirements-notebooks.txt
python book_bonificacion/codigo/comprobaciones.py
```

Para recalcular el ejemplo del MLP y su criterio de parada:

```bash
python book_bonificacion/codigo/mlp_ejemplo_clase_v1.py
```

Las tablas están en `resultados/mlp_cinco_iteraciones_v1.csv`, `resultados/mlp_nodos_v1.csv` y `resultados/mlp_gradientes_v1.csv`; el criterio de parada y sus valores se documentan en `resultados/mlp_tolerancia_v1.json`. `codigo/mlp_cinco_iteraciones.py` se conserva como plantilla general independiente.

El cuaderno actual es [notebooks/comprobaciones_v2.ipynb](notebooks/comprobaciones_v2.ipynb). Conserva todos los cálculos y salidas del inicial y actualiza dos notas para enlazar al ejercicio MLP completo. [notebooks/comprobaciones.ipynb](notebooks/comprobaciones.ipynb) se mantiene sin modificaciones como archivo histórico, disponible en Anexos. La compilación del sitio no ejecuta notebooks ni entrena modelos.

## Publicación

El workflow existente `.github/workflows/publish-book.yml` conserva los bytes vinculados al registro científico de E2. Su paso de construcción compila `book`, `book_e2` y `book_e3`; después, la entrada de línea de comandos de `scripts/assemble_books_site.py` compila este libro con advertencias tratadas como errores y comprueba sus enlaces antes de ensamblar los cuatro. Un fallo detiene la publicación.

El ensamblador conserva E1 en la raíz, E2 en `entregable2/`, E3 en `entregable3/` y coloca este libro en `bonificacion/`.

La publicación mantiene los controles descritos en `PUBLICACION_SEGURA.md`: sólo el primer intento de un nuevo push a `main` puede publicar, después de superar todos los jobs requeridos. La verificación HTTP comprueba también la introducción del libro de bonificación. Las ejecuciones manuales y los reintentos no publican.
