"""Verify the compact distribution and reconstruct its source-bound ZIP_STORED.

Only container compression differs. All archived member bytes are retained.
No network, model import, training, prediction or statistical analysis occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_documentary_archive import LIMIT, NAME, archive_bytes


def verify(archive, manifest, canonical_output=None):
    archive, manifest = Path(archive), Path(manifest)
    receipt = json.loads(manifest.read_text())
    raw = archive.read_bytes()
    if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("Distribution archive differs")
    prefix = NAME + "/"
    with zipfile.ZipFile(archive) as stream:
        if sum(info.file_size for info in stream.infolist()) > LIMIT:
            raise ValueError("Distribution exceeds the documentary size limit")
        if stream.testzip() is not None:
            raise ValueError("Distribution CRC differs")
        names = stream.namelist()
        if len(names) != len(set(names)) or any(not n.startswith(prefix) for n in names):
            raise ValueError("Distribution members differ")
        bodies = {n[len(prefix):]: stream.read(n) for n in names}
    canonical = archive_bytes(bodies)
    if (len(canonical) != receipt["canonical_stored_bytes"] or
            hashlib.sha256(canonical).hexdigest() != receipt["canonical_stored_sha256"]):
        raise ValueError("Canonical source-bound archive cannot be reconstructed")
    if canonical_output:
        target = Path(canonical_output)
        if target.is_symlink():
            raise ValueError("Canonical output cannot be a symlink")
        if target.exists():
            if target.read_bytes() != canonical:
                raise ValueError("Existing canonical output differs; preserve it")
        else:
            target.write_bytes(canonical)
    return {"status": "distribution_and_canonical_member_identity_verified",
            "distribution_bytes": len(raw), "members": len(bodies),
            "canonical_stored_sha256": receipt["canonical_stored_sha256"],
            "canonical_reconstruction_verified": True,
            "documentation_source_commit": receipt["documentation_source_commit"],
            "new_forecasting_model_fits": 0, "new_forecasting_model_predictions": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.manifest, args.canonical_output), sort_keys=True))
