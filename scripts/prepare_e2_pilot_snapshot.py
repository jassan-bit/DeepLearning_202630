"""Prepare the exact registered local protocol snapshot in a disposable checkout.

The pilot used a two-LF local copy; the observed GitHub checkout has one LF.
Both concrete byte hashes and semantic equality are required. This helper
never accepts an arbitrary equivalent JSON reformat, never changes the frozen
training/runner sources, and never uses a network or dataset loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

VERSION = "e2-pilot-snapshot-preparation-v1"
PROTOCOL_PATH = Path("book_e2/_static/search_protocol_20261109/protocol.json")
SNAPSHOT_PATH = Path("experiments/e2_development_pilot_20261109/source_protocol.json")
BACKUP_PATH = Path("work/old_protocol_bytes.json")
REGISTERED_SHA256 = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
GITHUB_CHECKOUT_SHA256 = "5195337af13f4193da16bf4499c8396586c621d4ab084a5b296a297f2141edeb"
CANONICAL_SHA256 = "1bf33fb22e3452e0fbde93f202628b666c06490fba1f0f4fd9377cad385c7e18"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular protocol/snapshot file is required: "+str(path))
    return path.read_bytes()


def _canonical(data):
    value=json.loads(data.decode("utf-8"))
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(root):
    """Normalize only the known one-LF checkout to the registered exact bytes.

    Validation finishes before any backup/write. Existing exact bytes are an
    idempotent no-write case. A normalization creates an exclusive durable
    backup, then atomically replaces the checkout JSON from the pinned snapshot.
    Existing backups are never replaced. Call on a checkout/CI copy rather than
    changing a running experiment's input files.
    """
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("An existing nonsymlink repository root is required")
    target=root/PROTOCOL_PATH
    snapshot=root/SNAPSHOT_PATH
    backup=root/BACKUP_PATH
    exact=_read_regular(snapshot)
    if _sha(exact)!=REGISTERED_SHA256:
        raise ValueError("Snapshot differs from the exact registered pilot bytes")
    current=_read_regular(target)
    before=_sha(current)
    if before not in (GITHUB_CHECKOUT_SHA256,REGISTERED_SHA256):
        raise ValueError("Checkout protocol is neither the known GitHub bytes nor the registered snapshot")
    canonical_exact=_canonical(exact)
    canonical_current=_canonical(current)
    if (canonical_current!=canonical_exact or _sha(canonical_exact)!=CANONICAL_SHA256):
        raise ValueError("Semantic protocol change is forbidden")
    receipt={
        "version":VERSION,"status":"already_registered_snapshot","modified":False,
        "protocol_path":str(PROTOCOL_PATH),"snapshot_path":str(SNAPSHOT_PATH),
        "input_sha256":before,"registered_sha256":REGISTERED_SHA256,
        "known_github_checkout_sha256":GITHUB_CHECKOUT_SHA256,
        "canonical_json_sha256":CANONICAL_SHA256,
        "normalized_sha256":REGISTERED_SHA256,"backup_created":False,"backup_path":None,
        "semantic_change":False,"network_used":False,"market_data_read":False,
        "training_executed":False,
    }
    if current==exact:
        return receipt
    # One concrete known checkout -> one concrete registered snapshot only.
    if current+b"\n"!=exact:
        raise ValueError("The accepted checkout must differ only by the known final LF")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError("Refusing to overwrite prior protocol-byte backup")
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.parent.is_symlink():
        raise ValueError("Protocol-byte backup directory must not be a symlink")
    with backup.open("xb") as handle:
        handle.write(current);handle.flush();os.fsync(handle.fileno())
    _sync_directory(backup.parent)
    descriptor,temporary=tempfile.mkstemp(prefix="protocol-exact-",dir=target.parent)
    temporary=Path(temporary)
    try:
        with os.fdopen(descriptor,"wb") as handle:
            handle.write(exact);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,stat.S_IMODE(target.stat().st_mode))
        if _read_regular(target)!=current:
            raise ValueError("Checkout protocol changed while preparing its exact snapshot")
        os.replace(temporary,target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if _read_regular(target)!=exact or _read_regular(backup)!=current:
        raise RuntimeError("Protocol snapshot/backup verification failed")
    receipt.update(status="normalized_to_registered_snapshot",modified=True,
                   backup_created=True,backup_path=str(BACKUP_PATH),
                   backup_sha256=before)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    args=parser.parse_args()
    print(json.dumps(prepare(args.root),sort_keys=True,indent=2))


if __name__=="__main__":
    main()

