# Preparación fijada: auditoría y admisión; cero entrenamientos.
import hashlib
import urllib.request

SOURCE_COMMIT = "6e004b5ddbfc414fe3ae57901d09b81fab717662"
BOOTSTRAP_SHA256 = "c862b9932eac67edb7710736b58b2bf09b9521f1978d1cf4ccbd7f9c8365aec2"
MANIFEST_SHA256 = "d6b9cc8170fa8bd3d9d076e970dab75f2270f7f59afa785495f2cab80437f814"
url = (
    "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    + SOURCE_COMMIT
    + "/experiments/e3_input_only_candidate_20261121/prepare_colab.py"
)
with urllib.request.urlopen(url, timeout=90) as response:
    code = response.read(128 * 1024 + 1)
if len(code) > 128 * 1024 or hashlib.sha256(code).hexdigest() != BOOTSTRAP_SHA256:
    raise RuntimeError("La fuente descargada no coincide con la versión fijada.")
E3_INPUT_ONLY = {"__name__": "e3_input_only_pinned_preparation"}
exec(compile(code, "e3_input_only_preparation", "exec"), E3_INPUT_ONLY)
E3_INPUT_ONLY["main"]([
    "--source-commit", SOURCE_COMMIT,
    "--manifest-sha256", MANIFEST_SHA256,
    "--cache", "/content/e3-input-only-dossier-1",
])
