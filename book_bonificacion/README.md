# Demostraciones de bonificación

Libro de demostraciones de aprendizaje profundo de Jassan Alberto Arteta Chavarro.

Sitio publicado: https://jassan-bit.github.io/DeepLearning_202630/bonificacion/intro.html

## Contenido

Siete capítulos: entrenamiento de MLP, derivada de la sigmoide, tamaño de paso para el error cuadrático, gradientes convolucionales, retropropagación, campo receptivo y beneficio de la profundidad. Incluye los enunciados, demostraciones paso a paso, un cuaderno con salidas ejecutadas y las capturas originales del documento proporcionado.

Las seis demostraciones teóricas están completas. La tabla numérica del MLP requiere la página del ejemplo de clase con su arquitectura, entradas, objetivo y pesos iniciales; esos datos no aparecían en el documento original. El capítulo explica el procedimiento y el código está preparado sin inventar esos valores.

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

El cuaderno `notebooks/comprobaciones.ipynb` conserva sus salidas. La compilación del sitio no ejecuta notebooks ni entrena modelos.

## Publicación

El workflow existente `.github/workflows/publish-book.yml` conserva los bytes vinculados al registro científico de E2. Su paso de construcción compila `book`, `book_e2` y `book_e3`; después, la entrada de línea de comandos de `scripts/assemble_books_site.py` compila este libro con advertencias tratadas como errores y comprueba sus enlaces antes de ensamblar los cuatro. Un fallo detiene la publicación.

El ensamblador conserva E1 en la raíz, E2 en `entregable2/`, E3 en `entregable3/` y coloca este libro en `bonificacion/`.

La publicación mantiene los controles descritos en `PUBLICACION_SEGURA.md`: sólo el primer intento de un nuevo push a `main` puede publicar, después de superar todos los jobs requeridos. La verificación HTTP comprueba también la introducción del libro de bonificación. Las ejecuciones manuales y los reintentos no publican.
