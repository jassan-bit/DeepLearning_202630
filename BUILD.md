# Construir el Jupyter Book

Alcance: Sección 1 en español, borrador de protocolo. No descarga datos ni entrena modelos.

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

Los hashes corresponden a **materiales del curso**, no al dataset de Bitcoin, que aún no se ha adquirido.

## Privacidad

No hay workflow de despliegue ni configuración de GitHub Pages. Construir no publica el sitio ni cambia la visibilidad del repositorio. El README y .gitignore originales se conservan sin modificación. Los generados se excluyen mediante `book/.gitignore`.
