"""Small file/identity primitives shared by the controller, never credentials."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe(path):
    path=Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path,*path.parents)), "Symlink path refused")
    return path


def sha(path):
    result=hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"Duplicate JSON key")
            result[key]=value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: "+value)
    return json.loads(safe(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)


def atomic(path, raw, *, immutable=False):
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes()==raw,"Immutable evidence changed: "+path.name)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=".atomic-",delete=False) as stream:
        tmp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:tmp.unlink(missing_ok=True)


def write_json(path,value,*,immutable=False):
    atomic(path,canonical(value)+b"\n",immutable=immutable)


def write_csv(path,rows):
    require(bool(rows),"Empty results cannot stand for successful evaluation")
    output=io.StringIO(newline="")
    writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(path,output.getvalue().encode())


@contextmanager
def case_lock(case):
    case=safe(case);case.mkdir(parents=True,exist_ok=True)
    with (case/"case.lock").open("a+") as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Another process owns this case") from None
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def verify_sources(root):
    root=safe(root);manifest=read_json(root/"SOURCE_MANIFEST.json")
    files=manifest.get("files")
    require(isinstance(files,dict) and files,"Pinned source manifest required")
    for relative,expected in files.items():
        path=Path(relative)
        require(not path.is_absolute() and ".." not in path.parts,"Unsafe manifest path")
        require(sha(root/path)==expected,"Source hash mismatch: "+relative)
    return {"manifest_sha256":sha(root/"SOURCE_MANIFEST.json"),"files_verified":len(files)}
