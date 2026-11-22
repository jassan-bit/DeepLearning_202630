from pathlib import Path
import hashlib, importlib.util, json, re, time, urllib.request

STATISTICS_SOURCE_COMMIT = 'db5a75bb66a0b28b1d05dacc41b66daf161f9120'
STATISTICS_MANIFEST_SHA256 = '45ad0706f293e487128905d6529d203aea14f8c5fc5a8907020781bd44a1c12f'
STATISTICS_BOOTSTRAP_SHA256 = '6645af3889e356ec687ade7a20d63cd4cbfb8689cc4b91c63269aa367d619061'
assert re.fullmatch(r"[0-9a-f]{40}", STATISTICS_SOURCE_COMMIT), "Usar la edición publicada con pins completos"
assert re.fullmatch(r"[0-9a-f]{64}", STATISTICS_MANIFEST_SHA256)
assert re.fullmatch(r"[0-9a-f]{64}", STATISTICS_BOOTSTRAP_SHA256)
STATISTICS_WORK = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "e3-primary-joint-validation-cpu"
setup_start = time.perf_counter()
bootstrap_url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" +
    STATISTICS_SOURCE_COMMIT + "/experiments/e3_primary_joint_repair_20261122/colab/bootstrap_cpu.py")
with urllib.request.urlopen(bootstrap_url, timeout=90) as response:
    bootstrap_raw = response.read(262145)
assert len(bootstrap_raw) <= 262144
assert hashlib.sha256(bootstrap_raw).hexdigest() == STATISTICS_BOOTSTRAP_SHA256
bootstrap_file = STATISTICS_WORK / ("bootstrap-statistics-" + STATISTICS_SOURCE_COMMIT + ".py")
assert not any(p.is_symlink() for p in (bootstrap_file, *bootstrap_file.parents))
if bootstrap_file.exists():
    assert bootstrap_file.read_bytes() == bootstrap_raw
else:
    bootstrap_file.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap_file.open("xb") as stream:
        stream.write(bootstrap_raw)
bootstrap_spec = importlib.util.spec_from_file_location("e3_pinned_statistics_bootstrap", bootstrap_file)
statistics_bootstrap = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(statistics_bootstrap)
candidate_python = None
for previous in (globals().get("VALIDATION_SETTINGS"), globals().get("SETTINGS")):
    if isinstance(previous, dict) and previous.get("source_commit") in {
        STATISTICS_SOURCE_COMMIT, "db74afd0763596f71bcbafb4b9150a44a2e83339",
        "c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c"}:
        candidate = previous.get("python")
        if isinstance(candidate, (str, Path)) and Path(candidate).is_file():
            try:
                statistics_bootstrap.check_python(candidate)
                candidate_python = str(candidate)
                break
            except RuntimeError:
                pass
VALIDATION_SETTINGS = statistics_bootstrap.setup(STATISTICS_SOURCE_COMMIT,
    STATISTICS_MANIFEST_SHA256, STATISTICS_WORK / "cache", existing_python=candidate_python)
STATISTICS_SOURCE = Path(VALIDATION_SETTINGS["source_root"])
SOFTWARE_SETUP_WALL_SECONDS = time.perf_counter() - setup_start
print(json.dumps({"cost_scope":"software_setup", "wall_seconds":SOFTWARE_SETUP_WALL_SECONDS,
    "monte_carlo_executed":False, "model_training_executed":False}))

def statistics_command(args):
    return statistics_bootstrap.command(VALIDATION_SETTINGS, args)

def statistics_result(args):
    result = statistics_command(args)
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("Falta el resultado JSON del comando estadístico; conservar el caso")

