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

## Entrenamiento y validación forward chaining

El procedimiento publicado utiliza `timeseries-cv` 0.1.5, proyecto de DidierRLopes con coautoría de Filipe R. Ramos. Sus índices diarios se agrupan en doce bloques mensuales de 2024. Cada red se ajusta de nuevo al comenzar el mes con el pasado permitido, embargo de siete días y parada interna cronológica. El horizonte sigue siendo diario.

El generador CV requiere NumPy menor que 2; utilizar un entorno separado del entrenamiento PyTorch:

```bash
python3.12 -m venv .venv-cv
.venv-cv/bin/python -m pip install -r requirements-cv.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.lock.txt
.venv-cv/bin/python -m unittest discover -s tests -p 'test_dl_r2_cv.py'
.venv-cv/bin/python scripts/prepare_dl_r2_cv.py --output-dir /tmp/dl-cv-index-check
.venv-training/bin/python -m unittest discover -s tests -p 'test_dl_*.py'
tar -xzf results/dl_r2_cv_20261104/run_artifacts.tar.gz -C results/dl_r2_cv_20261104
.venv-training/bin/python scripts/verify_dl_r2_cv.py
.venv-training/bin/python scripts/analyze_filipe_statistics.py
.venv-training/bin/python scripts/verify_filipe_statistics.py
.venv-training/bin/python scripts/export_filipe_book.py
```

El generador invoca `split_train_val_forwardChaining(np.arange(731),7,1,1)`. Su adaptación conserva 366 predicciones diarias y doce procedimientos mensuales por trayectoria. Los índices de entrenamiento, validación interna y evaluación alimentan directamente el ajuste de pesos y escaladores.

La exportación usa únicamente las predicciones y registros de CV, verifica sus cifras y genera los CSV y gráficos que muestra el libro. No entrena modelos. Los archivos publicados se limitan a esta metodología; las instantáneas de investigación se conservan en el repositorio como trazabilidad.

## Replicar los ajustes

Usar un checkout separado y conservar la instantánea publicada antes de iniciar. El manifiesto versionado es una entrada firmada: su timestamp forma parte del hash. Para comprobar la librería se generan los índices en otro directorio, como arriba, y se comparan las listas de índices, sin sobrescribir la entrada firmada.

```bash
mv results/dl_r2_cv_20261104 results/dl_r2_cv_20261104_published_snapshot
mkdir results/dl_r2_cv_20261104
cp results/dl_r2_cv_20261104_published_snapshot/folds.json results/dl_r2_cv_20261104/folds.json
cp results/dl_r2_cv_20261104_published_snapshot/folds.csv results/dl_r2_cv_20261104/folds.csv
.venv-training/bin/python scripts/train_dl_r2_cv.py --phase validate
.venv-training/bin/python scripts/train_dl_r2_cv.py --phase run
.venv-training/bin/python scripts/export_filipe_book.py
```

Se registraron diez semillas por modelo y representación, 60 trayectorias, 720 segmentos modelo–fold y 1.440 ajustes efectivos. Los 720 checkpoints, escaladores, máscaras y curvas están en el paquete de artefactos con hashes. Las semillas describen aleatoriedad de entrenamiento condicionada a 2024; no constituyen mercados independientes ni un dimensionamiento de potencia 0,80.

La evaluación sigue siendo exploratoria sobre desarrollo previamente examinado. El ajuste final con desarrollo 2023–2024 y el test previsto de enero de 2025–agosto de 2026 permanecen pendientes. Septiembre–octubre de 2026 queda sin asignar al test.

## Revisión de todos los ítems y estadística complementaria

`experiments/filipe_statistics_20261104/protocol.json` registra una enmienda exploratoria posterior a los descriptivos de 2024 y anterior al cálculo complementario. No es un preregistro anterior al entrenamiento ni una confirmación en datos desconocidos. El analizador utiliza las predicciones guardadas y las curvas internas del corte de diciembre con semilla 4101; restaurar primero los archivos por corrida mediante el tar indicado arriba.

```bash
.venv-training/bin/python -m unittest discover -s tests -p 'test_filipe_statistics.py'
.venv-training/bin/python scripts/analyze_filipe_statistics.py
.venv-training/bin/python scripts/verify_filipe_statistics.py
.venv-training/bin/python scripts/export_filipe_book.py
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py
```

El suplemento ejecuta nueve contrastes de redes/referencia/margen, incorpora los tres controles de representación en Holm global-12, seis DM principales con sensibilidades HAC, quince diagnósticos BDS/Ljung–Box, bootstrap pareado de fechas y sensibilidad retrospectiva de diseño. No ajusta pesos, remuestrea semillas ni evalúa el test. Los IC por semillas, pérdidas temporales y bootstrap responden preguntas distintas. El verificador independiente reconstruye los cálculos desde las predicciones sin importar el analizador numérico principal.

El inventario de la revisión identifica 390 subsecciones numeradas. Revisar y actualizar sus contenidos no equivale a declarar todos los requisitos científicamente completos: el piloto prospectivo, los márgenes pendientes, el ajuste final, el test y el respaldo aplicado faltante siguen identificados.
