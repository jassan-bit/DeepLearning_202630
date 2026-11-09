"""Explicit retrieval of eight immutable public iTransformer source files.

No network action occurs at import. URL destinations, repository, commit and
paths are fixed constants, never supplied by a model or data file. TLS
verification and inherited urllib proxy/CA behavior remain enabled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

MANIFEST = {
    "repository": "thuml/iTransformer",
    "commit": "c2426e68ca13f74aaec08045c5c724d8ad328124",
    "files": {
        "model/iTransformer.py": "6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db",
        "layers/Embed.py": "2de7e5049a696d320a23a8bdb6e08e823cf7fa3f246a0a49555fcd729c5c8398",
        "layers/Transformer_EncDec.py": "985c31f4ae187b08afaa9663dc960a40544b2ec33ab55389ceab3930edc1ce02",
        "layers/SelfAttention_Family.py": "38ac67428d528b8e145e4944ca059b40629a8446c891142602d5aaadc38dfd9e",
        "utils/masking.py": "02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef",
        "scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh": "81f8409cff60b754b45ed2e76133c7e3b6e585a50c8361f04fa3daf3b6997cc0",
        "run.py": "650873cfd6fee1f0415d4e8fd69943ad736fd1bab6495b640aa27b1a769f59f9",
        "LICENSE": "29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849"
    }
}
MAX_FILE_BYTES = 1024*1024


def fetch_sources(output, *, downloader=None):
    """Verify all bytes in a temporary tree, then install into a new directory."""
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Author-source destination must be new")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix="itransformer-source-",dir=output.parent))
    result={"status":"complete","repository":MANIFEST["repository"],"commit":MANIFEST["commit"],
            "verified_files":{},"weights_downloaded":False,"market_data_read":False}
    try:
        for relative,expected in MANIFEST["files"].items():
            url="https://raw.githubusercontent.com/"+MANIFEST["repository"]+"/"+MANIFEST["commit"]+"/"+relative
            if downloader is None:
                with urllib.request.urlopen(url,timeout=30) as response:
                    payload=response.read(MAX_FILE_BYTES+1)
            else:
                payload=downloader(url)
            if not isinstance(payload,bytes) or len(payload)>MAX_FILE_BYTES:
                raise ValueError("Unexpected source response size/type: "+relative)
            actual=hashlib.sha256(payload).hexdigest()
            if actual!=expected:
                raise ValueError("Pinned source hash mismatch: "+relative)
            target=temporary/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
            result["verified_files"][relative]=actual
        (temporary/"source_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Refusing to replace author-source destination")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    receipt=fetch_sources(args.output)
    print(json.dumps({"status":receipt["status"],"verified_files":len(receipt["verified_files"]),
                      "output":str(args.output),"weights_downloaded":False}))


if __name__=="__main__":
    main()

