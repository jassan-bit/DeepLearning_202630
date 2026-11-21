# Celda de inicio: prepara el piloto; no entrena.
from google.colab import drive
drive.mount("/content/drive")

import hashlib
import urllib.request
from pathlib import Path

SOURCE_COMMIT = '8777477fcde6c34890e447cc801df2f9c60a43c1'
SOURCE_MANIFEST_SHA256 = '7f32590bb954fb110a8755525c3ba74ef50fb4ef15e116f0274bad7a957a3dc8'
BOOTSTRAP_SHA256 = 'd687e6c0075af80230972ed930b67ec60edd4e68428a45a8d3a7ff16ae7d95c4'
CASE_ROOT = "/content/drive/MyDrive/DeepLearning_E3/SE_OI_V5/pilot_01"

if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Ejecuta primero la celda que monta Drive.")
url = (
    "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    + SOURCE_COMMIT
    + "/experiments/e3_seoi_v5_pilot_20261121/bootstrap_colab.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    code = response.read(2 * 1024 * 1024)
if hashlib.sha256(code).hexdigest() != BOOTSTRAP_SHA256:
    raise RuntimeError("El código no coincide con la versión fijada; no se ejecuta.")
E3_MODULE = {"__name__": "e3_v5_pinned_setup"}
exec(compile(code, "e3_v5_pinned_setup", "exec"), E3_MODULE)
E3 = E3_MODULE["setup"](
    source_commit=SOURCE_COMMIT,
    manifest_sha256=SOURCE_MANIFEST_SHA256,
    case_root=CASE_ROOT,
)
