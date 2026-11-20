"""Paste/run in Colab CPU after Manual Deploy; preserve the exact receipt ZIP.

Declared AI assistance. Uses the unchanged reviewed V2 helper/client. The SHA
must be the effective deployment shown by Render, never an assumed main HEAD.
No model training, calibration, Render mutation or Zenodo write is performed.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import re
import time
import types
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPOSITORY = "jassan-bit/DeepLearning_202630"
HELPER_SOURCE = "3a637d1f93b81363df1070d0e8506556ac90794e"
HELPER_PATH = "experiments/e3_evidence_close_20261120/release/render_public_acceptance_colab.py"
HELPER_SHA256 = "75996051e8cbfddb2c4c19880b91e9ca149a4e55bf11834c8d2e4f41c6977eff"


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def preserve_zip_and_receipt(archive, deployment_commit):
    """Small sequential Git-data write to a new evidence branch; never main.

    Reuse durable local state on retry. API commit attribution is supplied by
    GitHub's authenticated account; this code invents no author/signature.
    """
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("Preserva el ZIP regular de aceptación antes de guardar.")
    output = archive.with_suffix("")
    report_file = output / "report.json"
    if report_file.is_symlink() or not report_file.is_file():
        raise ValueError("Falta el recibo real del cliente; no se inventa aceptación.")
    raw_zip, raw_report = archive.read_bytes(), report_file.read_bytes()
    if len(raw_zip) > 8_000_000:
        raise ValueError("ZIP supera el límite declarado de8MB; conserva el archivo y usa un paquete por partes.")
    stamp = output.name.removeprefix("e3-render-public-smoke-")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", stamp):
        raise ValueError("El identificador durable del recibo no tiene el formato esperado.")
    prefix = "runs/e3_public_acceptance/" + stamp
    branch = "e3-public-acceptance-" + stamp
    files = {prefix + "/" + archive.name: raw_zip, prefix + "/report.json": raw_report}
    binding = {"source_commit": deployment_commit, "branch": branch,
               "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                 "git_blob_sha": git_blob(raw)} for name, raw in files.items()}}
    state_file = output / "github_preservation_state.json"
    if state_file.is_symlink():
        raise ValueError("No se admite un estado de preservación simbólico.")
    state = json.loads(state_file.read_text()) if state_file.exists() else {"binding": binding, "blobs": {}}
    if state.get("binding") != binding:
        raise ValueError("El estado conservado pertenece a otro ZIP, recibo o despliegue.")
    def save_state():
        state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    save_state()
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        token = getpass.getpass("Token GitHub (secreto local; Contents:Read and write; nunca pegar en el chat): ").strip()
    if not token:
        raise ValueError("El ZIP queda conservado localmente; no se intentó escribir sin autenticación.")
    last_write = 0.0
    def api(method, path, payload=None, missing=False):
        nonlocal last_write
        for attempt in range(3):
            if method != "GET":
                time.sleep(max(0, 2.1 - (time.monotonic() - last_write)))
            data = None if payload is None else json.dumps(payload).encode()
            request = Request("https://api.github.com/repos/" + REPOSITORY + path,
                              method=method, data=data, headers={"Authorization": "Bearer " + token,
                              "Accept": "application/vnd.github+json", "Content-Type": "application/json",
                              "User-Agent": "E3-public-acceptance-preservation/1"})
            try:
                with urlopen(request, timeout=60) as response:
                    raw = response.read()
                if method != "GET":
                    last_write = time.monotonic()
                return json.loads(raw)
            except HTTPError as error:
                if missing and error.code == 404:
                    return None
                response = error.read(2000).decode("utf-8", "replace")
                retry = error.headers.get("Retry-After")
                if error.code in {403, 429} and attempt < 2 and (
                        retry or "secondary rate limit" in response.lower()):
                    seconds = min(60, max(2, int(retry) if retry and retry.isdigit() else 60))
                    print("GitHub limita temporalmente la escritura; se conserva el estado y se espera", seconds, "s.")
                    time.sleep(seconds)
                    continue
                raise RuntimeError("GitHub HTTP" + str(error.code) + ": " + response[:900] +
                                   ". El ZIP y el estado local se conservan; no se modifica main.") from None
        raise RuntimeError("Se conserva el estado para reintentar el mismo recibo.")
    source = api("GET", "/git/commits/" + deployment_commit)
    if source.get("sha") != deployment_commit:
        raise ValueError("El SHA efectivo no identifica un commit del repositorio.")
    for name, raw in files.items():
        expected = git_blob(raw)
        if state["blobs"].get(name) != expected:
            result = api("POST", "/git/blobs", {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
            if result.get("sha") != expected:
                raise ValueError("GitHub devolvió otra huella de blob; conserva el estado.")
            state["blobs"][name] = expected
            save_state()
    if "tree" not in state:
        tree = api("POST", "/git/trees", {"base_tree": source["tree"]["sha"], "tree": [
            {"path": name, "mode": "100644", "type": "blob", "sha": git_blob(raw)}
            for name, raw in files.items()]})
        state["tree"] = tree["sha"]
        save_state()
    if "commit" not in state:
        commit = api("POST", "/git/commits", {"message": "Preservar recibo público Render " + stamp,
                    "tree": state["tree"], "parents": [deployment_commit]})
        state["commit"] = commit["sha"]
        save_state()
    ref = api("GET", "/git/ref/heads/" + branch, missing=True)
    if ref is None:
        api("POST", "/git/refs", {"ref": "refs/heads/" + branch, "sha": state["commit"]})
        ref = api("GET", "/git/ref/heads/" + branch)
    if ref.get("object", {}).get("sha") != state["commit"]:
        raise ValueError("La rama de evidencia contiene otro commit; no se sobrescribe.")
    current_tree = state["tree"]
    for component in prefix.split("/"):
        tree = api("GET", "/git/trees/" + current_tree)
        if tree.get("truncated"):
            raise ValueError("GitHub truncó un árbol directo; se conserva la evidencia.")
        entries = {row["path"]: row for row in tree["tree"]}
        entry = entries.get(component, {})
        if entry.get("type") != "tree":
            raise ValueError("La carpeta final de evidencia no fue verificada.")
        current_tree = entry["sha"]
    tree = api("GET", "/git/trees/" + current_tree)
    entries = {row["path"]: row for row in tree["tree"]}
    for name, raw in files.items():
        if entries.get(Path(name).name, {}).get("sha") != git_blob(raw):
            raise ValueError("No se verificó el blob final en la rama; no declarar guardado completo.")
    report = json.loads(raw_report)
    result = {"saved_and_blob_references_verified": True, "commit": state["commit"],
        "url": "https://github.com/" + REPOSITORY + "/tree/" + branch + "/" + prefix,
        "archive_sha256": hashlib.sha256(raw_zip).hexdigest(), "main_updated": False,
        "render_public_smoke_passed": bool(report.get("render_public_smoke_passed")),
        "new_model_fits": 0, "new_calibrations": 0, "AI_assistance_declared": True}
    (output / "github_preservation_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    actual = input("SHA40 efectivo copiado del log del Manual Deploy terminado en Render: ").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", actual):
        raise ValueError("Pega el SHA40 efectivo del despliegue; no usar main como identificador.")
    url = "https://raw.githubusercontent.com/" + REPOSITORY + "/" + HELPER_SOURCE + "/" + HELPER_PATH
    with urlopen(url, timeout=60) as response:
        code = response.read()
    if hashlib.sha256(code).hexdigest() != HELPER_SHA256:
        raise ValueError("El helper descargado no coincide con la versión revisada.")
    helper = types.ModuleType("e3_existing_V2_public_helper")
    helper.__file__ = "/content/e3-existing-public-helper.py"
    exec(compile(code, helper.__file__, "exec"), helper.__dict__)
    archive = helper.main(actual, download=True)
    return preserve_zip_and_receipt(archive, actual)


if __name__ == "__main__":
    main()
