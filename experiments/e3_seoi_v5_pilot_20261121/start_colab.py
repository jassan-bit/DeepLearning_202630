# Inicio V5 con GitHub: prepara/restaura y guarda; no entrena.
import hashlib
import urllib.request

SOURCE_COMMIT = 'c4ca86f1431afeaa958ff0b544b453ab17a4d7b8'
SOURCE_MANIFEST_SHA256 = '4706f20a00508e492a54e25873b4885cead3fa7660d43bdb18c5aeda4fce2509'
BOOTSTRAP_SHA256 = '3a6bce99e149e3a1e6cfbb07b8e23fb6482d9bb876597d4f8d5df198cde07941'
REPOSITORY = "jassan-bit/DeepLearning_202630"
RUN_BRANCH = "e3-seoi-v5-run-01"
RESULT_PREFIX = "runs/e3_seoi_v5/run-01"
CASE_ROOT = "/content/e3-seoi-v5-cases/" + hashlib.sha256((REPOSITORY + RUN_BRANCH).encode()).hexdigest()[:16]
url = (
    "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    + SOURCE_COMMIT
    + "/experiments/e3_seoi_v5_pilot_20261121/bootstrap_colab.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    code = response.read(2 * 1024 * 1024)
if hashlib.sha256(code).hexdigest() != BOOTSTRAP_SHA256:
    raise RuntimeError("El código descargado no coincide con la versión fijada.")
E3_MODULE = {"__name__": "e3_v5_github_setup"}
exec(compile(code, "e3_v5_github_setup", "exec"), E3_MODULE)
E3 = E3_MODULE["setup"](
    source_commit=SOURCE_COMMIT,
    manifest_sha256=SOURCE_MANIFEST_SHA256,
    case_root=CASE_ROOT,
    prepare_case=False,
)
print("Entorno verificado. La celda siguiente guardará o restaurará el caso en GitHub.")

from getpass import getpass

def private_value(name, prompt):
    try:
        from google.colab import userdata
        value = userdata.get(name)
    except Exception:
        value = getpass(prompt)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Falta la credencial privada solicitada.")
    return value.strip()

E3_CREDENTIALS = {
    "token": private_value("GITHUB_TOKEN", "Token GitHub con Contents: Read and write: "),
    "backup_secret": private_value("E3_V5_BACKUP_KEY", "Clave privada de respaldo (mínimo32caracteres; consérvala): "),
}
if len(E3_CREDENTIALS["backup_secret"]) < 32:
    raise ValueError("La clave de respaldo debe tener al menos32caracteres y conservarse para reanudar.")

def E3_GITHUB(action):
    return E3_MODULE["github_command"](
        E3, action, source_commit=SOURCE_COMMIT,
        token=E3_CREDENTIALS["token"], backup_secret=E3_CREDENTIALS["backup_secret"],
        repository=REPOSITORY, branch=RUN_BRANCH, prefix=RESULT_PREFIX,
    )

E3_GITHUB("setup")
print("Caso verificado en GitHub. No se ha iniciado entrenamiento en estas dos celdas.")
