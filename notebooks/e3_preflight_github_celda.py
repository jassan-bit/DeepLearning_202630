import hashlib
import json
import urllib.request
from pathlib import Path

E3_SOURCE_COMMIT = "1ff136f71c0ffcd1dbd918efbc078a8d9653ed21"
url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
       f"{E3_SOURCE_COMMIT}/notebooks/e3_remediacion_preflight_colab.ipynb")
contenido = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(contenido).hexdigest() == (
    "172b576256217ff488e264e77d438dd0bd2486c44e8715f5a1ba8d348c97de57"
)
celdas = json.loads(contenido)["cells"]

reintentar = (
    all(n in globals() for n in ("github_api", "_e3_github_token",
        "OUTPUT", "EVIDENCE_ZIP", "HEAD", "RUN_ID"))
    and HEAD == E3_SOURCE_COMMIT
    and (Path(OUTPUT) / "preflight_receipt.json").is_file()
    and Path(EVIDENCE_ZIP).is_file()
)
if reintentar:
    print("Reintentando sólo el guardado; los modelos no se repiten.")
    celdas = [celdas[11]]

for numero, celda in enumerate(celdas, 1):
    if celda["cell_type"] == "code":
        print(f"\nEjecutando etapa {numero}...")
        exec(compile("".join(celda["source"]), f"etapa_{numero}", "exec"), globals())

print("Resultados guardados:", GITHUB_PUBLICATION["run_url"])
