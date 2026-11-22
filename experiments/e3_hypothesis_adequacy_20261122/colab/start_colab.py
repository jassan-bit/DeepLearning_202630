from pathlib import Path
import hashlib, importlib.util, json, re, subprocess, urllib.request

SOURCE_COMMIT = 'db74afd0763596f71bcbafb4b9150a44a2e83339'
MANIFEST_SHA256 = '2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b'
BOOTSTRAP_SHA256 = '3c78071ea0a25cd771037ef53eea1edd8c63cbc0a586cad1da3cda96b9fe560b'
assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT), "Usar la edición publicada con pins resueltos"
assert re.fullmatch(r"[0-9a-f]{64}", MANIFEST_SHA256)
assert re.fullmatch(r"[0-9a-f]{64}", BOOTSTRAP_SHA256)
WORK = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "e3-r3-crypto-engine-v2"
WORK.mkdir(parents=True, exist_ok=True)
url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
       + SOURCE_COMMIT + "/experiments/e3_hypothesis_adequacy_20261122/runtime/bootstrap_colab.py")
with urllib.request.urlopen(url, timeout=90) as response:
    bootstrap_bytes = response.read(1024 * 1024 + 1)
assert len(bootstrap_bytes) <= 1024 * 1024
assert hashlib.sha256(bootstrap_bytes).hexdigest() == BOOTSTRAP_SHA256
bootstrap_path = WORK / ("bootstrap-" + SOURCE_COMMIT + ".py")
if bootstrap_path.exists():
    assert bootstrap_path.read_bytes() == bootstrap_bytes
else:
    bootstrap_path.write_bytes(bootstrap_bytes)
spec = importlib.util.spec_from_file_location("e3_pinned_bootstrap", bootstrap_path)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
previous_info = globals().get("SETTINGS")
previous_python = None
if isinstance(previous_info, dict) and previous_info.get("source_commit") in {"c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c", SOURCE_COMMIT}:
    candidate_python = previous_info.get("python")
    if isinstance(candidate_python, (str, Path)) and Path(candidate_python).is_file():
        previous_python = str(candidate_python)
SETTINGS = bootstrap.setup(SOURCE_COMMIT, MANIFEST_SHA256, WORK / "cache", existing_python=previous_python, run_tests=True)
SOURCE = Path(SETTINGS["source_root"])
PRIVATE = WORK / "private"
PRIVATE.mkdir(exist_ok=True)
print(json.dumps({"source_commit": SOURCE_COMMIT, "manifest_sha256": MANIFEST_SHA256,
                  "source_root": str(SOURCE), "new_market_fits": 0, "test_opened": False}))
