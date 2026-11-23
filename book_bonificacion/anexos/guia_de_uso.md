# Cómo leer y editar el libro

## Leer la versión publicada

El libro está integrado en el sitio de [DeepLearning 202630](https://jassan-bit.github.io/DeepLearning_202630/intro.html). Su dirección es:

[Demostraciones de bonificación](https://jassan-bit.github.io/DeepLearning_202630/bonificacion/intro.html).

El índice lateral permite recorrer los siete capítulos, el cuaderno de comprobaciones y los anexos. El [selector de entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) permite acceder también a E1, E2 y E3.

Las ecuaciones utilizan MathJax 3.2.2 desde el mismo CDN que los otros libros del repositorio. La versión web necesita acceso a ese recurso para presentarlas.

## Editar el contenido

Los archivos fuente están en [book_bonificacion dentro del repositorio](https://github.com/jassan-bit/DeepLearning_202630/tree/main/book_bonificacion).

| Archivo o carpeta | Función |
|---|---|
| `intro.md` | Portada y orientación de lectura |
| `capitulos/` | Enunciados, demostraciones y ecuaciones LaTeX |
| `_config.yml` | Título, autor, idioma, tema y presentación matemática |
| `_toc.yml` | Orden de las páginas en el índice |
| `notebooks/comprobaciones.ipynb` | Comprobaciones ejecutadas y figuras |
| `codigo/` | Código reproducible y función para las cinco iteraciones del MLP |
| `anexos/capturas/` | Ocho imágenes originales del documento |
| `figuras/` | Gráficas generadas por las comprobaciones |
| `assets/custom.css` | Estilo del libro |

Edita las fuentes y vuelve a compilar. Los cambios directos en `_build/html` se sustituyen durante la siguiente compilación.

## Compilar en Windows

Con Python 3.12 instalado, abre PowerShell en la raíz del repositorio y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-book.lock.txt
.\.venv\Scripts\jupyter-book.exe build book_bonificacion --all --warningiserror --keep-going
.\.venv\Scripts\python.exe scripts/check_book_links.py --book-dir book_bonificacion
```

Después abre `book_bonificacion/_build/html/intro.html` en el navegador. La instalación inicial requiere internet; la presentación matemática de esta configuración web también carga MathJax desde el CDN.

## Compilar con un entorno ya activado

```bash
python -m pip install -r requirements-book.lock.txt
jupyter-book build book_bonificacion --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book_bonificacion
```

El proyecto utiliza **Jupyter Book 1.0.4.post1**, con `_config.yml` y `_toc.yml`. La versión 2 utiliza otra configuración; una actualización requiere migrar el proyecto.

## Ejecutar las comprobaciones

Desde la raíz del repositorio:

```bash
python -m pip install -r book_bonificacion/requirements-notebooks.txt
python book_bonificacion/codigo/comprobaciones.py
```

También puedes abrir el cuaderno en VS Code con la extensión Jupyter y seleccionar el entorno que contiene esas dependencias. Las salidas ya están incorporadas al cuaderno y se muestran en el sitio sin ejecutar código durante la compilación.

## Despliegue del sitio

El workflow de publicación compila los cuatro libros, comprueba enlaces, ensambla el sitio y verifica su inventario. Después de superar sus jobs requeridos, el primer intento de un nuevo push a `main` puede actualizar `gh-pages` mediante un commit descendiente, conservando las rutas y las descargas anteriores.

Una ejecución manual o un reintento comprueba el proyecto, pero no publica. Los controles y la verificación de la versión servida se describen en [PUBLICACION_SEGURA.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/PUBLICACION_SEGURA.md).

## Documentación oficial

- [Jupyter Book 1 y comando de construcción](https://jupyterbook.org/v1/start/build.html).
- [Estructura del índice](https://jupyterbook.org/v1/structure/toc.html).
- [Migración a Jupyter Book 2](https://jupyterbook.org/latest/resources/upgrade/).
