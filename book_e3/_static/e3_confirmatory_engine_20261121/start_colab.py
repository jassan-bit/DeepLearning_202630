from pathlib import Path
import hashlib, importlib.util, json, re, subprocess, urllib.request

SOURCE_COMMIT = 'c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c'
MANIFEST_SHA256 = 'cd96def4f0a792d0d26d82f94048b0dbc9bf10b35cc7d3a561cf0e0db6d5b12b'
BOOTSTRAP_SHA256 = '86b4e619fe204092dc02cde3e882e999e4f78837dc89c985e239be776bc2b3e1'
assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT), "Usar la edición publicada con pins resueltos"
assert re.fullmatch(r"[0-9a-f]{64}", MANIFEST_SHA256)
assert re.fullmatch(r"[0-9a-f]{64}", BOOTSTRAP_SHA256)
WORK = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "e3-r3-engine"
WORK.mkdir(parents=True, exist_ok=True)
url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
       + SOURCE_COMMIT + "/experiments/e3_confirmatory_engine_20261121/bootstrap_colab.py")
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
SETTINGS = bootstrap.setup(SOURCE_COMMIT, MANIFEST_SHA256, WORK / "cache", run_tests=True)
SOURCE = Path(SETTINGS["source_root"])
PRIVATE = WORK / "private"
PRIVATE.mkdir(exist_ok=True)
print(json.dumps({"source_commit": SOURCE_COMMIT, "manifest_sha256": MANIFEST_SHA256,
                  "source_root": str(SOURCE), "new_market_fits": 0, "test_opened": False}))
