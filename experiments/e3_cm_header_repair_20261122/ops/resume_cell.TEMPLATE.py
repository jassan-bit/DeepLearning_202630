# Ejecuta en la MISMA sesión Colab. Conserva SOURCE, SETTINGS y los recibos.
# Reanuda sólo F02/F03; cero entrenamientos, cero apertura de prueba.
from pathlib import Path
import hashlib, importlib.util, json, re, urllib.request

REPAIR_COMMIT = "__REPAIR_COMMIT__"
REPAIR_MANIFEST_SHA256 = "__REPAIR_MANIFEST_SHA256__"
assert re.fullmatch(r"[0-9a-f]{40}", REPAIR_COMMIT), "Usar la celda publicada con pins completos"
assert re.fullmatch(r"[0-9a-f]{64}", REPAIR_MANIFEST_SHA256)
assert SOURCE_COMMIT == "db74afd0763596f71bcbafb4b9150a44a2e83339"
assert MANIFEST_SHA256 == "2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b"
assert isinstance(SETTINGS, dict) and SETTINGS["source_commit"] == SOURCE_COMMIT
repair_dir = Path(PRIVATE).parent / "operational-addons" / ("coinm-header-" + REPAIR_COMMIT)
repair_prefix = "experiments/e3_cm_header_repair_20261122/"

def repair_fetch(relative, expected):
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    target = repair_dir / relative
    assert not target.is_symlink()
    if target.exists():
        assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        return target
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + REPAIR_COMMIT + "/" + repair_prefix + relative
    with urllib.request.urlopen(url, timeout=90) as response:
        raw = response.read(262145)
    assert len(raw) <= 262144 and hashlib.sha256(raw).hexdigest() == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    assert target.resolve().is_relative_to(repair_dir.resolve())
    with target.open("xb") as stream:
        stream.write(raw)
    return target

repair_manifest = repair_fetch("ops/REPAIR_MANIFEST.json", REPAIR_MANIFEST_SHA256)
repair_inventory = json.loads(repair_manifest.read_text())
for relative, digest in repair_inventory["files"].items():
    repair_fetch(relative, digest)
repair_spec = importlib.util.spec_from_file_location("e3_separately_pinned_cm_repair", repair_dir / "ops/resume_development.py")
repair_helper = importlib.util.module_from_spec(repair_spec)
repair_spec.loader.exec_module(repair_helper)
repair_result = repair_helper.resume(globals(), repair_dir, REPAIR_MANIFEST_SHA256)
print("Cuatro paneles verificados. Continúa sólo con la etapa 3 original; planificación sigue desactivada.")
