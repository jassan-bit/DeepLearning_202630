# Construir el Jupyter Book

El libro documenta el Entregable 1 y el estado real de sus evidencias. Construir el HTML no descarga datos ni entrena modelos. La adquisición directa de Binance se reproduce con `scripts/download_binance_spot.py`; véase `data/README.md`.

## Entorno probado

Python 3.12, Jupyter Book 1.0.4.post1. Esta configuración usa Jupyter Book **v1**, no v2.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-book.lock.txt
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py
python -m http.server 8000 --directory book/_build/html
```

Abrir http://localhost:8000. En Windows, activar con `.venv\Scripts\activate`.
El HTML también está disponible en `book/_build/html/index.html`.

`requirements-book.txt` fija la dependencia directa; `requirements-book.lock.txt` fija el entorno completo probado. Recrear el lock solo tras comprobar un build limpio. La descarga de dependencias necesita internet; MathJax usa CDN para renderizar fórmulas en el navegador.

## Verificación

El build trata advertencias como errores. El comprobador valida enlaces locales y anclas en el HTML generado, sin solicitar URLs externas. No reemplaza la revisión de licencia/datos ni las pruebas experimentales futuras.

Los archivos originales del curso están en `references/course/`; verificar con:

```bash
cd references/course
sha256sum -c SHA256SUMS.txt
```

Los hashes corresponden a **materiales del curso**, no a los datos de Bitcoin. Los SHA-256 de los ZIP de Binance se conservan en el manifiesto de `data/manifests/`.

## Privacidad

El workflow `.github/workflows/publish-book.yml` construye y comprueba el libro y actualiza la rama `gh-pages` al recibir cambios en `main`. En Settings → Pages debe seleccionarse Deploy from a branch, rama `gh-pages`, carpeta `/ (root)`. Esto requiere permisos de administración y un plan compatible si el repositorio es privado. Construir localmente no publica ni cambia la visibilidad del repositorio. El README y .gitignore originales se conservan sin modificación. Los generados se excluyen mediante `book/.gitignore`.
