"""Run this CPU-only cell from a verified immutable repository commit.

Set the two literals only to the integration commit and manifest bytes supplied
with the published bundle. They are intentionally unresolved in the DRAFT.
No subprocess, pip, torch, GitHub write, inference or training is performed.
"""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

INTEGRATION_COMMIT = "REPLACE_WITH_REVIEWED_IMMUTABLE_40_HEX_COMMIT"
MANIFEST_SHA256 = "REPLACE_WITH_PUBLISHED_MANIFEST_SHA256"
REPO = "jassan-bit/DeepLearning_202630"
RELATIVE_BUNDLE = "experiments/e3_completion_20261120/protocol_power"

if not re.fullmatch(r"[0-9a-f]{40}", INTEGRATION_COMMIT) or not re.fullmatch(r"[0-9a-f]{64}", MANIFEST_SHA256):
    raise RuntimeError("This DRAFT has no published source lock. Set the verified integration commit and manifest hash first; no model has run.")

root = Path("/content/e3-v3-preparation-" + INTEGRATION_COMMIT[:12])
root.mkdir(parents=True, exist_ok=True)

def fetch(relative, expected):
    # A commit, never main/latest, binds all source reads. Paths are fixed below.
    url = f"https://raw.githubusercontent.com/{REPO}/{INTEGRATION_COMMIT}/{RELATIVE_BUNDLE}/{relative}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError("Source hash mismatch: " + relative)
    destination = root / relative
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError("Existing preparation source differs; preserve it and use a new directory")
    destination.write_bytes(content)
    return content

manifest_bytes = fetch("delivery_manifest.json", MANIFEST_SHA256)
manifest = json.loads(manifest_bytes)
for relative in ("protocol.DRAFT.json", "preflight.py"):
    fetch(relative, manifest["files"][relative])

# exec only the just-verified standard-library preparation source. Its __main__
# block is disabled; the exposed callable has no model execution path.
namespace = {"__name__": "e3_verified_CPU_preparation", "__file__": str(root / "preflight.py")}
exec(compile((root / "preflight.py").read_bytes(), str(root / "preflight.py"), "exec"), namespace)
result = namespace["prepare"](root / "protocol.DRAFT.json", root / "CPU_preparation_receipt.json")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("Receipt preserved:", root / "CPU_preparation_receipt.json")
print("Preparation only. No training or prediction is implemented by this cell.")
