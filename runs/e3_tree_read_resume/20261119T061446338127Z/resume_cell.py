import time
import re
from pathlib import Path
import urllib.request

E3_EXTENDED_SOURCE_COMMIT = "5a07d6157a9796ca54416dc5dea1e1527e731597"
E3_EXTENDED_MASTER_RUN = "20261119T031104515520Z"
E3_EXTENDED_START_NEW = False

_transport_file = (
    Path("/content") / ("e3-extended-source-" + E3_EXTENDED_SOURCE_COMMIT)
    / "experiments/e3_evaluacion_ampliada_20261119/orchestration/github_transport.py"
).resolve()
_allowed_route = re.compile(
    r"/repos/jassan-bit/DeepLearning_202630/git/"
    r"(?:trees/[0-9a-f]{40}(?:\?recursive=1)?|"
    r"ref/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z)"
)

def _e3_read_retry(error):
    if type(error).__name__ != "StorageError":
        return None
    message = str(error)
    if message.startswith("github_transport_network_read_failed "):
        return "corte de lectura de red"
    if message != "github_transport_http_error operation=git_trees_read status=404 attempts=1":
        return None
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if (frame.f_code.co_name == "_perform"
                and Path(frame.f_code.co_filename).resolve() == _transport_file):
            values = frame.f_locals
            route = values.get("path")
            if (values.get("method") == "GET"
                    and type(values.get("status")) is int
                    and values["status"] == 404
                    and values.get("allow_missing") is False
                    and isinstance(route, str)
                    and _allowed_route.fullmatch(route)):
                return route
        trace = trace.tb_next
    return None

url = (
    "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
    f"{E3_EXTENDED_SOURCE_COMMIT}/"
    "experiments/e3_evaluacion_ampliada_20261119/orchestration/launch_colab.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    launcher = response.read()

for intento in range(1, 5):
    try:
        exec(compile(launcher, "e3_reanudar_misma_campana", "exec"), globals())
        break
    except Exception as error:
        route = _e3_read_retry(error)
        if route is None or intento == 4:
            raise
        print(f"GitHub no devolvió una lectura requerida: {route}")
        print(f"Reanudación acotada en 30 s ({intento}/3).")
        time.sleep(30)
