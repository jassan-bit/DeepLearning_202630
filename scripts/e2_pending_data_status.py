"""Show the recorded E2 data-source probe; optionally make ONE fresh probe.

Default execution reads the existing receipt without network activity. Explicit
--probe requires a NEW --output path and makes one curl request per pinned ZIP
and CHECKSUM URL, each capped at20seconds. It never retries, changes proxies,
fits models, reconstructs data or modifies historical plans.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "book_e2/_static/e2_pending_closure_20261116/data_status.json"
MANIFEST = ROOT / "data/manifests/test_raw_reserve_20250101_20260831.json"
ARCHIVE = "BTCUSDT-1m-2025-01.zip"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def single_probe(kind, url, work, archive_name):
    target = work / (archive_name if kind == "zip" else archive_name + ".CHECKSUM")
    header = work / (kind + ".headers.txt")
    if target.exists() or header.exists():
        raise ValueError("Use a fresh work directory; previous probe files are preserved")
    write_out = ('{"http_code":"%{http_code}","http_connect":"%{http_connect}",'
                 '"size_download":%{size_download},"time_total":%{time_total},'
                 '"url_effective":"%{url_effective}"}')
    command = ["curl", "--silent", "--show-error", "--fail", "--max-time", "20",
               "--connect-timeout", "10", "--dump-header", str(header), "--output", str(target),
               "--write-out", write_out, url]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=False)
        try:
            transfer = json.loads(result.stdout)
        except json.JSONDecodeError:
            transfer = {"curl_stdout": result.stdout[:1000]}
        code, error = result.returncode, result.stderr.strip()[:2000]
    except subprocess.TimeoutExpired:
        code, error, transfer = 124, "The25-second process guard stopped curl; no retry", {}
    return {"kind": kind, "url": url, "returncode": code,
            "elapsed_wall_seconds": time.monotonic() - started, "transfer": transfer,
            "error": error.replace(str(ROOT), "[repository]"), "file_exists": target.exists(),
            "file_bytes": target.stat().st_size if target.exists() else 0,
            "response_sha256": sha(target.read_bytes()) if target.exists() else None,
            "header_sha256": sha(header.read_bytes()) if header.exists() else None}


def probe(output, work):
    if output.exists():
        raise ValueError("A probe receipt already exists; it must not be overwritten or repeated")
    manifest_raw = MANIFEST.read_bytes()
    row = next(r for r in json.loads(manifest_raw)["archives"] if r["archive"] == ARCHIVE)
    budget = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/e2_corrections_budget.py"), "status"], text=True))
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(single_probe, "zip", row["url"], work, ARCHIVE),
                pool.submit(single_probe, "checksum", row["checksum_url"], work, ARCHIVE)]
        probes = [job.result() for job in jobs]
    verified = all(r["returncode"] == 0 for r in probes)
    if verified:
        tokens = (work / (ARCHIVE + ".CHECKSUM")).read_text().strip().split()
        zipped = work / ARCHIVE
        verified = (zipped.stat().st_size == row["bytes"] and sha(zipped.read_bytes()) == row["sha256"]
                    and len(tokens) == 2 and tokens[0].lower() == row["sha256"]
                    and tokens[1].lstrip("*") == ARCHIVE)
    record = {"version": "e2-pending-data-status-v1", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "OFFICIAL_PROBE_VERIFIED" if verified else "BLOCKED_INPUTS",
              "probe_scope": "One curl attempt per original ZIP and CHECKSUM URL; max20seconds each; no retries or proxy changes",
              "probe_elapsed_wall_seconds": time.monotonic() - started, "original_archive": row,
              "manifest_sha256": sha(manifest_raw), "probe_records": probes,
              "proxy_tunnel_403_observed": any(r["transfer"].get("http_connect") == "403" for r in probes),
              "official_zip_and_checksum_verified": verified, "all20_original_archives_recovered": False,
              "current_evaluation_features_available": False, "current_evaluation_target_metadata_available": True,
              "scientific_recomputation_executed": False, "budget_before_probe": budget,
              "new_model_fits": 0, "model_inference_calls": 0, "proxy_configuration_changed": False,
              "alternative_proxy_or_data_source_used": False, "earlier_large_local_search_repeated": False,
              "scope_limit": "A successful one-month probe alone is not recovery or validation of all608 evaluation dates."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=ROOT / "work/e2-pending-data-20261116")
    parser.add_argument("--probe", action="store_true", help="One explicit fresh probe; default only reads the existing receipt")
    args = parser.parse_args(argv)
    record = probe(args.output, args.work) if args.probe else json.loads(args.output.read_text())
    print(json.dumps({"status": record["status"],
                      "official_zip_and_checksum_verified": record["official_zip_and_checksum_verified"],
                      "proxy_tunnel_403_observed": record["proxy_tunnel_403_observed"],
                      "network_probe_executed_by_this_invocation": args.probe,
                      "current_evaluation_features_available": record["current_evaluation_features_available"],
                      "new_model_fits": 0, "model_inference_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
