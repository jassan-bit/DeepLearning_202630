"""Pinned fresh-CPU entry: metadata durability precedes market acquisition.

No work on import. Reuses the existing permission, not the lost receipt bytes.
Data/model execution remains in the published isolated S2 CPU interpreter.
The metadata client reuses S2's authenticated encryption in the notebook kernel.
AI assistance: implementation and review; no human signature is supplied.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

S2 = "db74afd0763596f71bcbafb4b9150a44a2e83339"
CM = "6c247cf0914cf28532bcfa5ceb07267a35c28c48"
CM_MANIFEST = "21ce8729f7db6c9e683adf57dfcb94e97eeb27c081c973a0791af5a39e9ac0a2"
CPU_START = "476edfac97b67d3cb671eba98543c96e982abeed"
CPU_SHA = "24aacfb0955fe4d0f318ab1bcf717d745ed098694a00fe8a23e743ee1ad36b9d"
REPO = "jassan-bit/DeepLearning_202630"
PREFIX = "experiments/e3_cpu_fresh_rebuild_20261122/"
JOURNAL_BRANCH = "e3-cpu-development-metadata-rebuild-20261122-01"
JOURNAL_PREFIX = "runs/e3_development_metadata/rebuild-20261122-01"
JOURNAL_ID = "data-acquisition-rebuild-20261122-01"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch(commit, prefix, relative, expected, root):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Ruta publicada inválida.")
    target = Path(root) / relative
    require(not any(p.is_symlink() for p in (target, *target.parents)) and
            target.resolve().is_relative_to(Path(root).resolve()), "Conservar rutas sin symlinks.")
    if target.exists():
        require(target.is_file() and sha(target.read_bytes()) == expected, "Archivo local distinto; se conserva.")
        return target
    url = "https://raw.githubusercontent.com/" + REPO + "/" + commit + "/" + prefix + relative.as_posix()
    with urllib.request.urlopen(url, timeout=90) as response:
        raw = response.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024 and sha(raw) == expected, "Descarga o SHA256 del componente distinto.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    return target


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def immutable(path, raw):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ruta de metadatos distinta.")
    if path.exists():
        require(path.read_bytes() == raw, "Metadatos originales distintos; no se sobrescriben.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return path


def metadata_paths(ns, kit):
    """Do not restore admission.json into a panel whose NPZ files are absent.

    A recovery bundle contains its admission and raw-manifest original bytes.
    Archive the bundle at a separate export path, then let S2 reacquire install
    the actual panel atomically. Raw CSV/ZIP/NPZ and mutable attempt logs stay out.
    """
    work, private = Path(ns["WORK"]), Path(ns["PRIVATE"])
    for panel in ("F00", "F01", "F02", "F03"):
        source = private / "panels" / panel / "development-v1/recovery_metadata.json"
        if source.is_file():
            immutable(private / "recovery-exports" / (panel + ".json"), source.read_bytes())
    result = set()
    for root in (private / "inputs", private / "operational-repairs", private / "recovery-exports"):
        if root.exists():
            result.update(p for p in root.rglob("*.json") if p.is_file())
    for name in ("development-panels.json", "development-split-rule.json", "planning-context.json"):
        if (private / name).is_file():
            result.add(private / name)
    for root in (Path(kit), Path(ns["repair_dir"])):
        result.update(p for p in root.rglob("*") if p.is_file() and
                      "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md", ".txt"))
    require(result and all(p.resolve().is_relative_to(work.resolve()) for p in result),
            "El respaldo de metadatos debe permanecer dentro de WORK.")
    return sorted(result)


def run_fresh_cpu(ns):
    commit = ns.get("FRESH_REBUILD_COMMIT", "")
    manifest_sha = ns.get("FRESH_REBUILD_MANIFEST_SHA256", "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) and re.fullmatch(r"[0-9a-f]{64}", manifest_sha),
            "Usar la celda publicada con commit y manifest fijados.")
    require(Path("/content").is_dir(), "Esta entrada corresponde a Google Colab.")
    work = Path("/content/e3-r3-crypto-engine-v2")
    kit = work / "operational-addons" / ("fresh-cpu-" + commit)
    manifest_path = fetch(commit, PREFIX, "RUNTIME_MANIFEST.json", manifest_sha, kit)
    manifest = read(manifest_path)
    require(manifest.get("version") == "fresh-cpu-development-rebuild-1", "Versión del kit distinta.")
    for relative, expected in manifest["files"].items():
        fetch(commit, PREFIX, relative, expected, kit)
    cm_dir = work / "operational-addons" / ("coinm-header-" + CM)
    cm_manifest = fetch(CM, "experiments/e3_cm_header_repair_20261122/",
                        "ops/REPAIR_MANIFEST.json", CM_MANIFEST, cm_dir)
    for relative, expected in read(cm_manifest)["files"].items():
        fetch(CM, "experiments/e3_cm_header_repair_20261122/", relative, expected, cm_dir)
    ns.update(FRESH_KIT_DIR=kit, FRESH_WORKER_SHA256=manifest["files"]["ops/fresh_worker.py"],
              repair_dir=cm_dir, CM_REPAIR_COMMIT=CM)
    driver = load(kit / "ops/fresh_rebuild.py", "e3_pinned_fresh_cpu_driver")
    journal_module = load(kit / "storage/acquisition_backup.py", "e3_pinned_metadata_journal")
    credentials = None
    journal = None
    original_credentials = None
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode

    def before_metadata(current):
        nonlocal credentials, journal, original_credentials
        if credentials is None:
            credentials = current["github_credentials"]()
            original_credentials = current["github_credentials"]
            current["github_credentials"] = lambda: dict(credentials)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(current["SOURCE"]))
        from engine import storage
        require(Path(storage.__file__).resolve() == Path(current["SOURCE"]) / "engine/storage.py",
                "Cliente de respaldo de otra fuente; se detiene sin descargar mercado.")
        journal = journal_module.MetadataJournal(storage, token=credentials["token"],
            backup_secret=credentials["backup_secret"], repository=REPO,
            branch=JOURNAL_BRANCH, prefix=JOURNAL_PREFIX, work_root=current["WORK"],
            journal_dir=Path(current["WORK"]) / "metadata-journals/rebuild-20261122-01",
            case_id=JOURNAL_ID)
        remote = journal.store.status()
        if remote.get("exists") and not journal.case.exists():
            restored = journal.restore()
            require(restored and restored.get("remote_verified") is True,
                    "No continuar sin autenticar el respaldo de metadatos existente.")
            print(json.dumps({"status":"metadata_restored_before_new_forms",
                              "metadata_files_restored":restored.get("metadata_files_restored"),
                              "raw_inputs_restored":False,"new_market_fits":0}), flush=True)
        context = Path(current["PRIVATE"]) / "planning-context.json"
        if context.is_file():
            saved = read(context)
            require(saved.get("source_commit") == S2 and saved.get("repository") == REPO,
                    "Identidad operativa guardada distinta.")
            require(Path(saved["case_root"]).parent == Path(current["WORK"]) / "cases" and
                    saved["branch"].startswith("e3-r3-crypto-engine-v2-planning-") and
                    saved["prefix"].startswith("runs/e3_r3_crypto_engine_v2/planning-"),
                    "Rutas o rama guardadas distintas; no iniciar otro caso.")
            current.update(PLANNING_CASE=Path(saved["case_root"]), PLANNING_BRANCH=saved["branch"],
                           PLANNING_PREFIX=saved["prefix"])
        branch = urllib.parse.quote(current["PLANNING_BRANCH"], safe="")
        prior = journal.store._request("GET", "/repos/" + REPO + "/git/ref/heads/" + branch,
                                       allow_missing=True)
        if prior is not None and not (Path(current["PLANNING_CASE"]) / "plan.json").is_file():
            current["run_cli"](["restore", *current["remote_args"](current["PLANNING_CASE"]),
                                "--workspace-root", str(current["WORK"])], remote=True)
        case = Path(current["PLANNING_CASE"])
        if (case / "plan.json").is_file():
            plan = read(case / "plan.json")
            require(plan.get("stage") == "planning" and plan.get("source_commit") == S2,
                    "El caso recuperado no es planificación S2.")
            for panel in plan["data_panels"]:
                binding = panel.get("recovery_metadata")
                require(binding and panel["id"] in ("F00","F01","F02","F03"), "Falta metadata original del panel.")
                raw = (case / binding["path"]).read_bytes()
                require(sha(raw) == binding["sha256"], "Metadata del caso recuperado distinta.")
                immutable(Path(current["PRIVATE"]) / "recovery-exports" / (panel["id"] + ".json"), raw)

    def checkpoint(current, stage):
        require(journal is not None, "Inicializar respaldo antes de adquirir.")
        value = {"kind":"operational_planning_location_not_preregistration","source_commit":S2,
                 "repository":REPO,"case_root":str(current["PLANNING_CASE"]),
                 "branch":current["PLANNING_BRANCH"],"prefix":current["PLANNING_PREFIX"]}
        immutable(Path(current["PRIVATE"]) / "planning-context.json",
                  json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        result = journal.checkpoint(metadata_paths(current, kit), event=stage)
        print(json.dumps({"status":"development_metadata_backup_verified", "stage":stage,
                          "remote_commit":result.get("commit"),"new_market_fits":0}), flush=True)
        return result

    try:
        rebuilt = driver.fresh_rebuild(ns, (kit / "assets/pinned_T2_notebook.json").read_bytes(),
                                       checkpoint, before_metadata_callback=before_metadata)
        if ns.get("FRESH_START_PLANNING", False):
            url = "https://raw.githubusercontent.com/" + REPO + "/" + CPU_START + "/experiments/e3_cpu_planning_start_20261122/start_cpu_planning.py"
            with urllib.request.urlopen(url, timeout=90) as response:
                raw = response.read(262145)
            require(len(raw) <= 262144 and sha(raw) == CPU_SHA, "Inicio CPU distinto del publicado.")
            ns.update(CPU_PLANNING_START_COMMIT=CPU_START, CPU_PLANNING_START_SHA256=CPU_SHA,
                      CPU_PLANNING_START_BYTES=raw)
            exec(compile(raw, url, "exec"), ns)
            return ns["start_cpu_planning"](ns)
        return rebuilt
    finally:
        if original_credentials is not None:
            ns["github_credentials"] = original_credentials
        if journal is not None:
            journal.close()
        if credentials is not None:
            credentials.clear()
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
