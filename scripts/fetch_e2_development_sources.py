"""Fetch only the three hash-pinned development inputs; never parse prices."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

SOURCE_REPOSITORY = "jassan-bit/DeepLearning_202630"
SOURCE_COMMIT = "0b079b50625a936456a70695532853fc13298f31"


@dataclass(frozen=True)
class Source:
    path: str
    size: int
    sha256: str


SOURCES = (
    Source("experiments/dl_r2_cv_20261104/protocol.json", 5718,
           "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec"),
    Source("results/eda_development/daily_market_development.csv", 60427,
           "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171"),
    Source("results/eda_development/daily_volatility_development.csv", 76492,
           "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a"),
)


def _target(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or str(part) != relative:
        raise ValueError("Development source path must be normalized and relative")
    target = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Symlinks are not admitted in the development cache")
    return target


def _verify(raw, source):
    if len(raw) != source.size or hashlib.sha256(raw).hexdigest() != source.sha256:
        raise ValueError("Development source bytes do not match the pinned input: "+source.path)


def fetch_sources(output, *, opener=None):
    """Verify every payload before publishing any new cache files.

    Existing valid files are reused. Existing mismatches are retained and cause
    failure; the caller must inspect them rather than silently replacing data.
    """
    root = Path(output).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Development cache root cannot resolve through a symlink")
    opener = urlopen if opener is None else opener
    staged, records = [], []
    for source in SOURCES:
        target = _target(root, source.path)
        reused = target.exists()
        if reused:
            if not target.is_file() or target.stat().st_size != source.size:
                raise ValueError("Existing development input has a different size")
            raw = target.read_bytes()
        else:
            url = ("https://raw.githubusercontent.com/"+SOURCE_REPOSITORY+"/"
                   +SOURCE_COMMIT+"/"+source.path)
            with opener(url, timeout=30) as response:
                final_url = response.geturl()
                parsed = urlsplit(final_url)
                if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
                    raise ValueError("Development download redirected outside its public source")
                raw = response.read(source.size+1)
        _verify(raw, source)
        staged.append((target, raw, reused))
        records.append({"path": source.path, "bytes": len(raw),
                        "sha256": source.sha256, "reused": reused})
    for target, raw, reused in staged:
        if reused:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _target(root, target.relative_to(root).as_posix())
        fd, temporary = tempfile.mkstemp(prefix=".development-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
            "files": records, "numerical_market_values_parsed": False,
            "final_market_files_requested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_sources(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
