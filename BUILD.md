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

El build trata advertencias como errores. El comprobador valida enlaces locales y anclas en el HTML generado, sin solicitar URLs externas. La validación experimental se ejecuta por separado.

Los archivos originales del curso están en `references/course/`; verificar con:

```bash
cd references/course
sha256sum -c SHA256SUMS.txt
```

Los hashes corresponden a **materiales del curso**, no a los datos de Bitcoin. Los SHA-256 de los ZIP de Binance se conservan en el manifiesto de `data/manifests/`.

## Privacidad

El workflow `.github/workflows/publish-book.yml` construye y comprueba el libro y actualiza la rama `gh-pages` al recibir cambios en `main`. En Settings → Pages debe seleccionarse Deploy from a branch, rama `gh-pages`, carpeta `/ (root)`. Construir localmente no publica ni cambia la visibilidad del repositorio. Los generados se excluyen mediante `book/.gitignore`; los archivos individuales de corridas se conservan juntos en un comprimido versionado.

## Entrenamiento y análisis de desarrollo

El experimento real del 04/11/2026 está en [results/dl_benchmark_20261104/README.md](results/dl_benchmark_20261104/README.md). Entorno CPU probado: Python 3.12.14, PyTorch 2.15.0+cpu y versiones completas registradas en `runtime.json`. Usa un entorno separado del libro:

```bash
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.lock.txt
.venv-training/bin/python -m unittest discover -s tests -p 'test_dl_*.py'
tar -xzf results/dl_benchmark_20261104/run_artifacts.tar.gz -C results/dl_benchmark_20261104
.venv-training/bin/python scripts/verify_dl_artifacts.py
.venv-training/bin/python scripts/analyze_dl_benchmark.py --results-dir results/dl_benchmark_20261104
.venv-training/bin/python scripts/verify_dl_analysis.py
```

El análisis anterior no entrena. La restauración permite inspeccionar los 105 checkpoints y las curvas/configuraciones de las 201 corridas. El manifiesto incluye hashes de los 1.512 archivos. Las otras 96 corridas de búsqueda no guardan pesos.

Para replicar los ajustes, usar un **checkout separado** y conservar el resultado publicado antes de iniciar. Los dos CSV diarios de desarrollo ya están versionados; no hace falta abrir ZIP ni descargar test:

```bash
mv results/dl_benchmark_20261104 results/dl_benchmark_20261104_published_snapshot
.venv-training/bin/python scripts/train_dl_benchmark.py --phase validate
.venv-training/bin/python scripts/train_dl_benchmark.py --phase all
.venv-training/bin/python scripts/analyze_dl_benchmark.py --results-dir results/dl_benchmark_20261104
```

`--phase all` ejecuta búsqueda, piloto y comparación. Es una replicación con nuevos registros de ejecución, no un nuevo preregistro histórico. El entrenador comprueba las fuentes y la firma de sus scripts y entorno; ante una firma diferente no reutiliza artefactos previos. No deben sobrescribirse los resultados publicados ni editarse los scripts firmados para mezclar otro diseño con este experimento. Un cambio de arquitectura o protocolo requiere una versión explícita nueva.

Las 25 pruebas verifican modelos, particiones/preprocesamiento y cálculos estadísticos sobre casos controlados. Los resultados propios proceden de las corridas registradas, no de esas pruebas. La comparación usa calendario de desarrollo ya expuesto al piloto; 30 semillas no aportan 30 mercados independientes. El límite registrado de 30 no alcanzó el dimensionamiento del efecto pequeño planificado. El test de 2025–2026 permanece pendiente.
