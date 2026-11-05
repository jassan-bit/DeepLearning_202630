"""Package the current seven GRU/LSTM/TCN products and their combined ZIP.

This command only reads committed public code, the current rendered report,
allowlisted aggregate evidence and public literature/data-source/runtime
metadata. It never trains, scores,
downloads market data, updates a book, or publishes a branch.

Run only after the final book build and source commit. The output directory
must be empty and outside both the checkout and the immutable experiment.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from package_e1_downloads import archive, digest, encode, verify_report


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BOOK = 'https://jassan-bit.github.io/DeepLearning_202630/'
DOWNLOAD_RAW = 'https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/entregables/'
IMMUTABLE_EXPERIMENT = Path('/workspace/e1_lstm_gru_tcn_local_20261105')
BENCHMARK = ROOT / 'book/_static/gru_lstm_tcn_20261105'
DELIVERY = ROOT / 'book/_static/gru_lstm_tcn_delivery_20261105'
LITERATURE = ROOT / 'book/_static/literature_e1'
SELECTION = LITERATURE / 'gru_lstm_tcn_selection_20261105'
MODELS = ['GRU', 'LSTM', 'TCN']
ITEMS = [
    ('8.1', '8_1_informe_html.zip', 'Informe HTML vigente'),
    ('8.2', '8_2_codigo.zip', 'Código EDA y GRU–LSTM–TCN'),
    ('8.3', '8_3_dataset_instrucciones.zip', 'Dataset: instrucciones y adquisición'),
    ('8.4', '8_4_literatura.zip', 'Literatura y selección de GRU–LSTM–TCN'),
    ('8.5', '8_5_benchmark.zip', 'Benchmark vigente e inferencia exploratoria'),
    ('8.6', '8_6_configuraciones.zip', 'Dependencias, semillas y configuraciones vigentes'),
    ('8.7', '8_7_declaracion.zip', 'Declaración y cronología vigentes'),
]
METADATA_SUFFIXES = {'.md', '.json', '.csv', '.txt', '.bib', '.svg', '.png'}
FORBIDDEN_SUFFIXES = {'.zip', '.gz', '.tar', '.pt', '.pth', '.npz', '.npy', '.pkl', '.pickle', '.pdf', '.docx'}
FORBIDDEN_NAMES = {'predictions.csv', 'daily_market_development.csv', 'daily_volatility_development.csv',
    'daily_market_test.csv', 'daily_volatility_test.csv', 'causal_daily_profile.csv', 'label_reconstruction.csv',
    'validation_predictions_residuals.csv', 'model_parameters.csv', 'reference_configurations.csv',
    'preprocessing.json', 'splits.json', 'history_inner.csv', 'history_refit.csv'}
FORBIDDEN_CSV_FIELDS = {'y_true', 'y_pred', 'prediction', 'residual', 'x_mean', 'x_scale', 'y_scale', 'mase_train_scale', 'train_indices'}
FOLD_FILES = ['results/dl_r2_cv_20261104/folds.json',
              *[f'results/filipe_multihistory_20261104/D{d}/folds.json' for d in (14, 21, 28)]]


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe archive member: ' + name)
    return path


def add(files, path, name=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular source file: ' + str(path))
    member = name if name is not None else path.relative_to(ROOT).as_posix()
    safe_member(member)
    if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError('Refusing private data, manuscript, weights or nested archive: ' + str(path))
    data = path.read_bytes()
    if path.suffix.lower() == '.csv':
        columns = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))), [])
        if set(columns) & FORBIDDEN_CSV_FIELDS:
            raise ValueError('Individual observations/scalers in supposedly aggregate CSV: ' + str(path))
    if member in files and files[member] != data:
        raise ValueError('Conflicting duplicate archive member: ' + member)
    files[member] = data


def metadata_tree(files, directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Expected public metadata directory: ' + str(directory))
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in METADATA_SUFFIXES:
            add(files, path)


def allowlist_tree(files, directory, manifest_name, mapping_key, include=None):
    directory = Path(directory)
    manifest_path = directory / manifest_name
    record = json.loads(manifest_path.read_text())
    mapping = record[mapping_key]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Nonempty explicit aggregate allowlist required')
    for name, expected in sorted(mapping.items()):
        relative = safe_member(name)
        path = directory / str(relative)
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise ValueError('Allowlist escapes its directory: ' + name)
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError('Changed allowlisted public artifact: ' + str(path))
        if path.suffix.lower() not in METADATA_SUFFIXES:
            raise ValueError('Unexpected allowlisted type: ' + name)
        if include is None or include(name):
            add(files, path)
    add(files, manifest_path)
    return record


def rendered_report_files():
    """Current HTML, with excluded downloads rewritten to their online URLs."""
    build = ROOT / 'book/_build/html'
    if not (build / 'intro.html').is_file():
        raise ValueError('Build the current Jupyter Book before packaging')
    if 'GRU' not in (build / 'intro.html').read_text():
        raise ValueError('The rendered introduction does not identify the current GRU/LSTM/TCN report')
    sensitive_names = {'_static/eda/results/summary.json', '_static/eda_temporal_reference/summary.json'}
    sensitive_hashes = {digest((build / name).read_bytes()) for name in sensitive_names if (build / name).is_file()}
    excluded = set()
    for path in build.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(build).as_posix()
        if (relative.startswith('_static/e1_downloads/') or path.name in FORBIDDEN_NAMES
                or path.suffix.lower() in FORBIDDEN_SUFFIXES or relative in sensitive_names
                or (path.name == 'summary.json' and digest(path.read_bytes()) in sensitive_hashes)):
            excluded.add(relative)
    files, rewritten = {}, 0
    for path in sorted(build.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(build).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError('Unexpected rendered report symlink: ' + str(path))
        data = path.read_bytes()
        if path.suffix == '.html':
            def replace(match):
                nonlocal rewritten
                parts = urlsplit(match.group(2))
                if parts.scheme or parts.netloc:
                    return match.group(0)
                target = (path.parent / unquote(parts.path)).resolve()
                if target.is_relative_to(build.resolve()):
                    name = target.relative_to(build.resolve()).as_posix()
                    if name in excluded:
                        rewritten += 1
                        query = '?' + parts.query if parts.query else ''
                        fragment = '#' + parts.fragment if parts.fragment else ''
                        return match.group(1) + PUBLIC_BOOK + name + query + fragment + match.group(3)
                return match.group(0)
            data = re.sub(r'((?:href|src)=[\"\'])([^\"\']+)([\"\'])', replace, data.decode()).encode()
        files['informe/' + relative] = data
    files['ARTEFACTOS_EXTERNOS.json'] = encode({'scope': 'Current HTML, aggregate assets and an explicitly fixed-seed PNG illustration; no recursive downloads or machine-readable individual daily tables',
        'excluded_from_this_archive': sorted(excluded), 'rewritten_links': rewritten,
        'online_base': PUBLIC_BOOK, 'mathjax_may_require_network': True})
    return files


def source_revision(requested):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if requested is not None:
        resolved = subprocess.check_output(['git', 'rev-parse', '--verify', requested + '^{commit}'], cwd=ROOT, text=True).strip()
        if resolved != revision:
            raise ValueError('--source-commit must identify the current HEAD used for the report and packages')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError('Commit final source changes before recording the package source commit')
    return revision


def verify_archive(path):
    """Verify CRC, safe member paths and every member SHA against MANIFIESTO."""
    with ZipFile(path) as archive_file:
        if archive_file.testzip() is not None:
            raise ValueError('ZIP CRC failure: ' + str(path))
        names = archive_file.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member')
        for name in names:
            safe_member(name)
        manifest = json.loads(archive_file.read('MANIFIESTO.json'))
        if set(names) != set(manifest['files']) | {'MANIFIESTO.json'}:
            raise ValueError('ZIP members differ from the member manifest')
        for name, metadata in manifest['files'].items():
            data = archive_file.read(name)
            if len(data) != metadata['bytes'] or digest(data) != metadata['sha256']:
                raise ValueError('ZIP member hash mismatch: ' + name)
    return {'CRC': 'PASS', 'member_hashes': 'PASS', 'safe_paths': 'PASS', 'members': len(names)}


def common_readme(revision):
    return (f'\n\nTerna experimental vigente: **GRU–LSTM–TCN**. Baseline neural: **GRU/D7**, '
        f'seleccionado en CV antes de la nueva evaluación del test histórico ya conocido. '
        f'Fuente versionada: `{revision}`.\n\n'
        'Las comparaciones actuales entre los tres modelos y los diagnósticos adicionales se definieron '
        'después de observar el test: su alcance es exploratorio. Las diez semillas son un presupuesto fijo; '
        'no se atribuye un piloto prospectivo de los modelos nuevos ni potencia 0,80. El margen operativo '
        'futuro se declara después del test y no se aplica retroactivamente. Persistencia conserva su papel de referencia.\n\n'
        'Los paquetes públicos contienen código, metadatos, agregados y figuras. Se excluyen archivos crudos, '
        'pesos, escaladores y series diarias o vectores individuales legibles por máquina. '
        'Se incluye una ilustración PNG de observado frente a predicho con la semilla4101 fija, '
        'sin su CSV individual y sin seleccionar la semilla por su error. El adaptador público de reproducción '
        'es posterior a la ejecución original y no sustituye sus firmas históricas.\n\n'
        'La licencia MIT incluida corresponde a timeseries-cv; no asigna una licencia general al proyecto. '
        'Los términos de la fuente de datos se conservan en el producto 8.3.\n')


def package(output, requested_commit=None):
    output = Path(output).resolve()
    if (output.is_relative_to(ROOT) or output.is_relative_to(IMMUTABLE_EXPERIMENT)
            or (output.exists() and any(output.iterdir()))):
        raise ValueError('Use an empty external output directory, outside the checkout and immutable experiment')
    revision = source_revision(requested_commit)
    summary = json.loads((BENCHMARK / 'summary.json').read_text())
    if (summary['models'] != MODELS or summary['baseline_selected_by_cv']['model'] != 'GRU'
            or summary['baseline_selected_by_cv']['history_days'] != 7
            or not summary['trio_pairwise_analysis_posthoc_after_test']):
        raise ValueError('Only the current completed GRU/LSTM/TCN aggregate export is permitted')
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    packages = {item: {} for item, *_ in ITEMS}
    packages['8.1'] = rendered_report_files()
    report_check = verify_report(packages['8.1'])
    tracked = subprocess.check_output(['git', 'ls-files', 'scripts', 'tests'], cwd=ROOT, text=True).splitlines()
    code = [name for name in tracked if name.endswith('.py')]
    active = {'scripts/run_gru_lstm_tcn.py', 'scripts/dl_models_gru_lstm_tcn.py',
              'scripts/gru_lstm_tcn_final.py', 'scripts/build_gru_lstm_tcn_evidence.py',
              'scripts/gru_lstm_tcn_diagnostics.py', 'scripts/download_binance_spot.py'}
    if not active <= set(code):
        raise ValueError('Commit all current code before packaging: ' + str(active - set(code)))
    for name in code:
        add(packages['8.2'], ROOT / name)
    for path in sorted(ROOT.glob('requirements-*.txt')):
        add(packages['8.2'], path)
        add(packages['8.6'], path)
    for name in ('README.md', 'BUILD.md', 'DESCARGAS.md'):
        add(packages['8.2'], ROOT / name)
    for name in FOLD_FILES:
        add(packages['8.2'], ROOT / name)
        add(packages['8.6'], ROOT / name)
    add(packages['8.3'], ROOT / 'data/README.md')
    add(packages['8.3'], ROOT / 'scripts/download_binance_spot.py')
    metadata_tree(packages['8.3'], ROOT / 'data/manifests')
    metadata_tree(packages['8.3'], ROOT / 'book/_static/data_source')
    metadata_tree(packages['8.4'], LITERATURE)
    # The relocated recovery metadata is already inside LITERATURE.
    add(packages['8.4'], ROOT / 'book/references.bib')
    if not (SELECTION / 'selection.json').is_file():
        raise ValueError('The current three-model literature selection is required')
    allowlist_tree(packages['8.5'], BENCHMARK, 'public_allowlist_manifest.json', 'file_sha256')
    config_names = {'configuration_catalog_cv.csv', 'configuration_catalog_final.csv', 'model_specs.json', 'runtime.json',
                    'protocols_manifest.json', 'source_manifest.json', 'selected_by_cv.json', 'public_analysis_plan.json'}
    allowlist_tree(packages['8.6'], BENCHMARK, 'public_allowlist_manifest.json', 'file_sha256', lambda name: name in config_names)
    allowlist_tree(packages['8.7'], DELIVERY, 'public_manifest.json', 'files_sha256')
    declarations = {'selected_by_cv.json', 'future_margin.json', 'protocols_manifest.json', 'public_analysis_plan.json'}
    allowlist_tree(packages['8.7'], BENCHMARK, 'public_allowlist_manifest.json', 'file_sha256', lambda name: name in declarations)
    notes = {
        '8.1': 'Descomprimir y abrir `informe/intro.html`. Alternativa: `python -m http.server 8000 --directory informe`. '
               'Los ZIP y otros artefactos excluidos de este informe mantienen enlaces a su ruta pública online; '
               'se evita la recursión de descargas. MathJax puede necesitar conexión.',
        '8.2': 'Leer `BUILD.md`. Entrada activa: `scripts/run_gru_lstm_tcn.py`; sus fases explícitas registran, '
               'entrenan, verifican y cierran nuevas reproducciones en otro directorio. La fase predeterminada '
               'sólo valida entradas. Se incluyen EDA, análisis, tests y los módulos auxiliares necesarios. '
               'Los módulos heredados son dependencias o archivo técnico; no designan la terna experimental actual '
               'ni son lanzadores recomendados. Los cuatro manifiestos de folds contienen índices y metadatos, '
               'sin valores de mercado. Los derivados diarios se reconstruyen con la adquisición y preparación documentadas.',
        '8.3': 'Leer `data/README.md`. El script de descarga y los manifiestos identifican fuente oficial, '
               'fechas UTC y SHA256. Se entregan instrucciones, código de adquisición y términos; no archivos crudos.',
        '8.4': 'La selección vigente está en `book/_static/literature_e1/gru_lstm_tcn_selection_20261105/`. '
               'Se incluyen síntesis, flujo y registros bibliográficos. Las menciones a otros modelos en artículos '
               'o cortes históricos no los convierten en modelos experimentales actuales. No se redistribuyen manuscritos.',
        '8.5': 'Abrir `book/_static/gru_lstm_tcn_20261105/README.md` y `report.md`. Los CSV contienen métricas '
               'agregadas de CV y test, contrastes entre GRU/LSTM/TCN, diagnósticos y figuras. '
               '`public_allowlist_manifest.json` usa la clave `file_sha256` y verifica cada agregado.',
        '8.6': 'Los catálogos vigentes contienen 1.440 segmentos de CV y 120 configuraciones finales, con '
               'semillas, recetas y hashes; no valores de escaladores ni pesos. Se incluyen locks, especificaciones '
               'de modelos, folds y la metadata de runtime registrada cuyo hash figura en la procedencia pública.',
        '8.7': 'Leer la declaración y cronología de `gru_lstm_tcn_delivery_20261105/`, junto con '
               '`selected_by_cv.json`, `protocols_manifest.json`, `public_analysis_plan.json` y `future_margin.json`. '
               'Distinguen selección previa en CV, evaluación de un test ya observado, análisis posteriores y requisitos futuros.',
    }
    license_path = ROOT / 'book/_static/licenses/timeseries-cv-LICENSE.txt'
    records, verifications = [], {}
    common = common_readme(revision)
    for item, filename, title in ITEMS:
        add(packages[item], ROOT / 'BUILD.md', 'BUILD.md')
        add(packages[item], license_path, 'AVISOS/timeseries-cv-LICENSE.txt')
        for name in packages[item]:
            safe_member(name)
        readme = f'# {item}. {title}\n\n{notes[item]}{common}'
        record = archive(output / filename, packages[item], item, title, revision, readme)
        verifications[filename] = verify_archive(output / filename)
        records.append(record)
    combined = {row['file']: (output / row['file']).read_bytes() for row in records}
    combined['CATALOGO_PRODUCTOS.json'] = encode(records)
    complete = archive(output / 'entregable1_completo.zip', combined, '8.1–8.7',
        'Entregable1 GRU–LSTM–TCN completo', revision,
        '# Entregable 1: siete productos vigentes\n\nDescomprimir este ZIP y después cada producto. '
        'Los siete ZIP conservan LEEME.md, MANIFIESTO.json y hashes individuales.\n' + common)
    verifications[complete['file']] = verify_archive(output / complete['file'])
    catalog = {'source_commit': revision, 'models': MODELS, 'active_entrypoint': 'scripts/run_gru_lstm_tcn.py',
        'baseline_selected_by_cv': summary['baseline_selected_by_cv'], 'packages': records, 'complete_package': complete,
        'report_verification': report_check, 'archive_verification': verifications,
        'new_model_fits_by_packaging': 0, 'new_checkpoint_forecasts_by_packaging': 0,
        'raw_market_archive_reads_by_packaging': 0, 'publications_by_packaging': 0,
        'scope': 'Current GRU/LSTM/TCN public delivery; exploratory already-observed test; code adapter postdates recorded fits',
        'public_aggregate_manifest_sha256': digest((BENCHMARK / 'public_allowlist_manifest.json').read_bytes()),
        'declaration_manifest_sha256': digest((DELIVERY / 'public_manifest.json').read_bytes()),
        'scientific_limitations_preserved': True}
    (output / 'catalogo.json').write_bytes(encode(catalog))
    (output / 'README.md').write_text('# Descargas vigentes del Entregable 1\n\n'
        'Los ocho archivos ZIP conservan los nombres de los botones de la unidad8. '
        'Los siete productos y el paquete completo describen la terna actual GRU–LSTM–TCN.\n' + common +
        '\nVerificar integridad con `sha256sum -c SHA256SUMS.txt`. '
        'El manifiesto interno de cada ZIP verifica además bytes y SHA256 por miembro.\n')
    checksum_paths = sorted(output.glob('*.zip')) + [output / 'catalogo.json', output / 'README.md']
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{digest(path.read_bytes())}  {path.name}\n' for path in checksum_paths))
    # Ensure a changed checkout cannot be mislabeled by a previously captured revision.
    if source_revision(revision) != revision:
        raise ValueError('Source revision changed during packaging')
    print(json.dumps({'packages': 8, 'source_commit': revision, 'output_dir': str(output),
        'complete_bytes': complete['bytes'], 'report_verification': report_check,
        'archive_verification': 'CRC and all member hashes PASS', 'model_fits': 0, 'forecasts': 0},
        ensure_ascii=False, indent=2))
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-commit', help='Current source commit; defaults to git HEAD and must resolve to HEAD')
    args = parser.parse_args()
    package(args.output_dir, args.source_commit)
