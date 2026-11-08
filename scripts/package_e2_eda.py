"""Package E2 EDA code, aggregate evidence and figures for direct book download.

The archive supplements the repository checkout. It excludes raw market files,
the private known-test cache, model checkpoints and individual test predictions.
"""
from hashlib import sha256
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "book_e2/_static/eda_descargas"
FILES = (
    "scripts/eda_e2_decisions.py", "scripts/eda_e2_known_shift.py",
    "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/planned_study_data.py",
    "scripts/package_e2_eda.py", "requirements-eda.lock.txt",
    "book_e2/sections/02_eda.md", "book_e2/_static/e1_reference.json",
    "book_e2/_static/dataset_card.json", "book_e2/_static/dataset_manifest_e1.json",
    "book_e2/_static/guia_e2_recibida.json",
)
DIRECTORIES = (
    "book_e2/_static/eda_decisions", "book_e2/_static/eda_known_shift",
    "book_e2/_static/eda_traceability",
)
README = """# EDA del Entregable 2: código, tablas y figuras

Este paquete permite revisar los diagnósticos y sus evidencias. Se utiliza
junto con el checkout del repositorio que publica este archivo:
https://github.com/jassan-bit/DeepLearning_202630

Los SHA-256 de MANIFIESTO.json identifican cada archivo incluido. Para ejecutar,
extraer conservando directorios sobre una copia del repositorio y crear el
entorno Python 3.12 con requirements-eda.lock.txt. Desde la raíz:

    python scripts/eda_e2_decisions.py

La entrada de desarrollo está versionada en results/eda_development del repo.
Los índices forward y los materiales heredados se conservan allí también.
La reconstrucción del perfil de la reserva ya conocida se explica en
book_e2/_static/eda_known_shift/README.md; admite caché verificado o los ZIP
oficiales, sin entrenar ni volver a puntuar modelos.

El contenido corresponde a EDA: Ridge alpha=1 es una referencia diagnóstica
mensual, distinta de la configuración Ridge del benchmark registrado de veinte
configuraciones. El benchmark final sigue pendiente; el diagnóstico no sustituye
sus resultados. La ruta F2a quedó cerrada el 06/11/2026. El CSV histórico se
conserva y decisiones_vigentes.csv documenta ese cierre y su enlace a la sección 7.
El perfil del test es retrospectivo y no constituye confirmación nueva. Los
resultados adversos y las limitaciones de ruido y sellado se conservan.

Se incluyen agregados y derivados diagnósticos de desarrollo; los originales
de mercado y el caché de precios de test se obtienen/conservan por separado.
Fuente de datos: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta.
Los derivados mantienen atribución CC BY-NC-SA 4.0 y condiciones de la fuente,
documentadas en la Data Card. La licencia de datos se distingue de la de código.
"""


def main():
    paths = {ROOT / name for name in FILES}
    for folder in DIRECTORIES:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    contents = {"LEEME.md": README.encode()}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Publication source must remain inside the repository")
        contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    manifest = {
        "schema_version": 1, "scope": "E2 EDA review materials; repository supplement",
        "e1_source_commit": "94a4b0324e93361e8928d708fdfeb50110e3686f",
        "raw_market_files_included": False, "individual_test_predictions_included": False,
        "new_deep_learning_fits": 0,
        "files": {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
                  for name, data in sorted(contents.items())},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    contents["MANIFIESTO.json"] = manifest_bytes
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / "EDA_E2_codigo_tablas_figuras.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, payload)
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for name, record in manifest["files"].items():
            assert sha256(stream.read(name)).hexdigest() == record["sha256"]
    (OUTPUT / "MANIFIESTO.json").write_bytes(manifest_bytes)
    (OUTPUT / "LEEME.md").write_text(README)
    print(json.dumps({"archive": archive.relative_to(ROOT).as_posix(),
                      "entries": len(contents), "bytes": archive.stat().st_size,
                      "sha256": sha256(archive.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
