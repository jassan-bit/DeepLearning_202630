"""Download/verify the twenty original E1/E2 test archives, without fitting.

Uses the published historical manifest, the normal HTTPS/proxy configuration
and SHA256/size checks. Existing good files are reused; differing files are not
overwritten. Run on a computer whose network permits data.binance.vision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/"book/_static/data/test_raw_reserve_20250101_20260831.json"
MANIFEST_SHA="d2eb4e84fff85ee957e6b02890cead3cec37fd593ec2a8db487dbe137d2d3a10"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    if digest(path)!=MANIFEST_SHA:
        raise ValueError("Historical archive manifest changed")
    payload=json.loads(Path(path).read_text())
    rows=payload["archives"]
    months=[f"2025-{i:02d}" for i in range(1,13)]+[f"2026-{i:02d}" for i in range(1,9)]
    if [r["archive"] for r in rows]!=[f"BTCUSDT-1m-{m}.zip" for m in months]:
        raise ValueError("Only the twenty original BTCUSDT one-minute months are admitted")
    for row in rows:
        url=urlsplit(row["url"])
        if (url.scheme!="https" or url.hostname!="data.binance.vision"
            or not url.path.endswith('/'+row["archive"])
            or not isinstance(row["bytes"],int) or row["bytes"]<=0
            or len(row["sha256"])!=64):
            raise ValueError("Invalid original archive identity")
    return rows


def verify(path,row):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing regular archive: "+str(path))
    if path.stat().st_size!=row["bytes"] or digest(path)!=row["sha256"]:
        raise ValueError("Archive differs from the original version; preserve and investigate: "+str(path))


def obtain(row,folder,*,validate_only=False):
    path=folder/row["relative_path"]
    if path.exists():
        verify(path,row)
        return path,"reused_verified"
    if validate_only:
        raise ValueError("Required archive missing: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+".",suffix=".partial",dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            with urllib.request.urlopen(row["url"],timeout=60) as response:
                while chunk:=response.read(1024*1024):
                    stream.write(chunk)
            stream.flush()
        verify(temporary,row)
        if path.exists():
            raise ValueError("Archive appeared during download; will not overwrite")
        temporary.rename(path)
    except BaseException:
        # Retain failed/incomplete bytes for a diagnostic; never admit them.
        raise
    return path,"downloaded_verified"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    p.add_argument("--output",type=Path,default=Path("datos_binance_E2"))
    p.add_argument("--validate-only",action="store_true")
    args=p.parse_args()
    rows=records(args.manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in rows:
        path,status=obtain(row,args.output,validate_only=args.validate_only)
        item={"archive":row["archive"],"path":str(path),"bytes":row["bytes"],"sha256":row["sha256"],"status":status}
        results.append(item)
        print(json.dumps(item),flush=True)
    report={"manifest_sha256":digest(args.manifest),"archives_verified":len(results),
        "bytes":sum(r["bytes"] for r in results),"market_fits":0,"source":"historical original manifest; no changed dataset accepted", "files":results}
    (args.output/"download_receipt.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
