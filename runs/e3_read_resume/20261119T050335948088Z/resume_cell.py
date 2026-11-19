import time
import urllib.request

E3_EXTENDED_SOURCE_COMMIT = "5a07d6157a9796ca54416dc5dea1e1527e731597"
E3_EXTENDED_MASTER_RUN = "20261119T031104515520Z"
E3_EXTENDED_START_NEW = False

url = (
    "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    f"{E3_EXTENDED_SOURCE_COMMIT}/"
    "experiments/e3_evaluacion_ampliada_20261119/"
    "orchestration/launch_colab.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    launcher = response.read()

for intento in range(1, 5):
    try:
        exec(compile(launcher, "e3_reanudar", "exec"), globals())
        break
    except Exception as error:
        corte_lectura = (
            type(error).__name__ == "StorageError"
            and str(error).startswith("github_transport_network_read_failed ")
        )
        if not corte_lectura or intento == 4:
            raise
        print(f"Corte de lectura de GitHub. Reanudando en 30 s ({intento}/3)...")
        time.sleep(30)
