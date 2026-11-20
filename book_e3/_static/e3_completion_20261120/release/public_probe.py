"""Read-only public E3 service/DOI observation with unchanged frozen V2 client.

Declared AI assistance: prepared by an AI technical auditor. Nothing here trains,
calibrates, deploys, changes a repository, or writes to Zenodo. A service-reported
commit is an observation, not an operator statement or platform attestation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REFERENCE_COMMIT = "bbd2604a9b78bab7fc8289ad07aae0a6117e884e"
V2_PATH = "experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py"
V2_SHA256 = "54ba00ebf49308fe525f7f4b27b4befa5f98912f2abdc61fc2d818eb314583ad"
LEGACY_PATH = "experiments/e3_bloques_fg_20261117/smoke_render.py"
LEGACY_SHA256 = "4dbbc83603d28224cfef4d1f8159eab55fa8dbef7c23081b5feebccf5e039190"
RENDER = "https://deeplearning-e3-dashboard.onrender.com"
DOI = "10.5281/zenodo.24356217"
REPO = "jassan-bit/DeepLearning_202630"
SHA40 = re.compile(r"[0-9a-f]{40}")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, directory, name, timeout=10, redirects=True):
    """Keep public upstream bytes. Explicitly distinguish CONNECT/proxy failure."""
    result = {"requested_url": url, "started_at_utc": utc_now(),
              "origin_http_status": None, "upstream_response_received": False}
    opener = build_opener() if redirects else build_opener(NoRedirect())
    try:
        with opener.open(Request(url, headers={"User-Agent": "E3-read-only-public-probe/1",
                                             "Accept": "application/json,text/html;q=0.8"}),
                         timeout=timeout) as response:
            raw = response.read(1_000_001)
            result.update(origin_http_status=response.status, upstream_response_received=True,
                          effective_url=response.url, headers=dict(response.headers))
    except HTTPError as error:
        raw = error.read(1_000_001)
        headers = dict(error.headers)
        proxy_blocked = any(k.lower() == "x-mitmproxy-blocked-reason" for k in headers)
        result.update(origin_http_status=None if proxy_blocked else error.code,
                      upstream_response_received=not proxy_blocked,
                      effective_url=error.url, headers=headers,
                      status="proxy_block_before_origin" if proxy_blocked else "HTTP_error_response")
    except (URLError, OSError, TimeoutError) as error:
        result.update(status="transport_failed_before_origin", error=str(error)[:1000],
                      finished_at_utc=utc_now())
        return result
    if len(raw) > 1_000_000:
        result.update(status="body_exceeds_declared_one_megabyte_limit", truncated=True)
        raw = raw[:1_000_000]
    directory.mkdir(parents=True, exist_ok=True)
    body_path = directory / (name + ".body")
    body_path.write_bytes(raw)
    result.update(body_file=body_path.name, bytes=len(raw), sha256=digest(raw), finished_at_utc=utc_now())
    try:
        result["json"] = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        result["body_preview"] = raw[:700].decode("utf-8", "replace")
    return result


def warm_render(output, seconds):
    """One total readiness budget; no redirect is accepted as the service health."""
    started = time.monotonic()
    deadline = started + seconds
    rows = []
    while time.monotonic() < deadline:
        row = request(RENDER + "/healthz", output, "render_health_latest",
                      timeout=max(.1, min(10, deadline - time.monotonic())), redirects=False)
        rows.append(row)
        health = row.get("json", {})
        if row["origin_http_status"] == 200 and isinstance(health, dict) and health.get("status") == "ok":
            return {"status": "ready", "attempts": rows, "budget_seconds": seconds,
                    "elapsed_seconds": time.monotonic() - started, "health": health,
                    "proven_cold_start_measurement": False}
        # A local CONNECT proxy rejection cannot improve by waking the origin.
        if row.get("status") in {"proxy_block_before_origin", "transport_failed_before_origin"} and (
                "Tunnel connection failed" in row.get("error", "") or row.get("status") == "proxy_block_before_origin"):
            break
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return {"status": "not_ready_or_transport_unavailable", "attempts": rows,
            "budget_seconds": seconds, "elapsed_seconds": time.monotonic() - started,
            "proven_cold_start_measurement": False}


def verify_reference_checkout(root):
    observed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if observed != REFERENCE_COMMIT:
        raise ValueError("Frozen reference checkout does not match the declared source pin.")
    records = {}
    for relative, expected in ((V2_PATH, V2_SHA256), (LEGACY_PATH, LEGACY_SHA256)):
        path = root / relative
        if path.is_symlink():
            raise ValueError("A frozen client must not be a symbolic link.")
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError("Frozen client SHA-256 mismatch: " + relative)
        records[relative] = {"bytes": len(raw), "sha256": digest(raw)}
    spec = importlib.util.spec_from_file_location("e3_frozen_public_v2", root / V2_PATH)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    example, models = client.load_references(root)
    return client, {"commit": observed, "client_files": records,
                    "scientific_reference_files": example["_reference_files"],
                    "expected_live_models": [m["id"] for m in models],
                    "fixed_case_csv_sha256": example["csv_sha256"]}


def doi_verdict(rows):
    api = rows["zenodo_public_api"]
    record = api.get("json", {})
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    observed_doi = record.get("doi", metadata.get("doi")) if isinstance(record, dict) else None
    public_record = bool(api.get("origin_http_status") == 200 and isinstance(record, dict)
                         and str(record.get("id")) == "24356217" and observed_doi == DOI
                         and isinstance(metadata, dict) and metadata.get("title"))
    resolver = rows["doi_resolver"]
    destination = urlsplit(resolver.get("effective_url", ""))
    resolved = bool(resolver.get("origin_http_status") == 200 and destination.hostname == "zenodo.org"
                    and destination.path.rstrip("/") in {"/records/24356217", "/record/24356217"})
    return {"doi": DOI, "public_record_metadata_verified": public_record,
            "resolver_to_public_record_verified": resolved,
            "publication_verified": public_record and resolved,
            "state": "published_public_record_observed" if public_record and resolved else "publication_not_verified",
            "files_downloaded_or_hash_verified": False,
            "archive_completeness_verified": False,
            "note": "A reserved DOI or private /uploads draft alone is not evidence of publication. A missing or blocked response does not establish nonexistence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wake-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.wake_seconds <= 180:
        raise SystemExit("The readiness budget must be positive and at most 180 seconds.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "started_at_utc": utc_now(), "status": "running",
              "script_sha256": digest(Path(__file__).read_bytes()),
              "service": RENDER, "reference_commit": REFERENCE_COMMIT,
              "new_model_fits": 0, "new_calibrations": 0, "deployments_created": 0,
              "zenodo_writes": 0, "operator_confirmation": False,
              "independent_platform_attestation": False,
              "identity_basis": "Service-reported environment value observed at /healthz; repository existence is checked separately.",
              "public_five_model_browser_acceptance_passed": False, "errors": []}
    targets = {"zenodo_public_api": "https://zenodo.org/api/records/24356217",
               "zenodo_public_record": "https://zenodo.org/records/24356217",
               "doi_resolver": "https://doi.org/" + DOI}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(request, url, output / "upstream", name, 20)
                   for name, url in targets.items()}
        report["render_readiness"] = warm_render(output / "upstream", args.wake_seconds)
        report["DOI_observations"] = {name: future.result() for name, future in futures.items()}
    report["DOI"] = doi_verdict(report["DOI_observations"])
    try:
        client, reference = verify_reference_checkout(args.reference_root.resolve())
        report["frozen_reference_validation"] = reference
        health = report["render_readiness"].get("health", {})
        observed = health.get("source_commit")
        field = health.get("source_commit_source")
        if not isinstance(observed, str) or not SHA40.fullmatch(observed.lower()) or field not in {"RENDER_GIT_COMMIT", "APP_SOURCE_COMMIT"}:
            report["browser_acceptance_status"] = "not_started_identity_unavailable"
        else:
            observed = observed.lower()
            report["observed_service_commit"] = observed
            report["observed_service_commit_source"] = field
            existence = request("https://api.github.com/repos/" + REPO + "/git/commits/" + observed,
                                output / "upstream", "service_source_commit", 20)
            report["service_commit_repository_check"] = existence
            if existence.get("origin_http_status") != 200 or existence.get("json", {}).get("sha") != observed:
                report["browser_acceptance_status"] = "not_started_repository_commit_unverified"
            else:
                # Preserve the existing client. Its historical CLI name is retained;
                # this wrapper explicitly records the observation as its true source.
                options = client.parser().parse_args([
                    "--url", RENDER, "--deployment-commit", observed,
                    "--output", str(output / "public_smoke_v2.json"),
                    "--screenshots-dir", str(output / "captures"),
                    "--wake-seconds", "10", "--navigation-seconds", "30",
                    "--prediction-seconds", "30", "--tolerance", "0.000001"])
                report["client_argument_provenance"] = {
                    "deployment_commit": "observed /healthz source_commit; not operator declared",
                    "historical_client_identity_field_requires_this_wrapper_context": True,
                    "frozen_references_match_remote_source_files_claimed": False}
                receipt = client.run(options)
                report["browser_acceptance_status"] = receipt["status"]
                report["public_five_model_browser_acceptance_passed"] = bool(receipt.get("render_public_smoke_passed"))
                report["browser_client_report_sha256"] = digest((output / "public_smoke_v2.json").read_bytes())
                report["browser_summary"] = {key: receipt.get(key) for key in (
                    "status", "pages", "models", "expected_live_models", "errors", "case", "finished_at_utc")}
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)[:1800]})
        (output / "wrapper_error.txt").write_text(traceback.format_exc())
    report.update(status="observation_completed", finished_at_utc=utc_now(),
                  workflow_success_means="Probe ran and retained a receipt; inspect acceptance and DOI flags independently.")
    dump(output / "public_release_report.json", report)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "files_manifest.json":
            raw = path.read_bytes()
            files[str(path.relative_to(output))] = {"bytes": len(raw), "sha256": digest(raw)}
    dump(output / "files_manifest.json", {"files": files})
    summary = {"status": report["status"], "Render_five_model_acceptance": report["public_five_model_browser_acceptance_passed"],
               "observed_service_commit": report.get("observed_service_commit"),
               "browser_acceptance_status": report.get("browser_acceptance_status"),
               "DOI_publication_verified": report["DOI"]["publication_verified"],
               "report_sha256": files["public_release_report.json"]["sha256"],
               "errors": report["errors"], "new_fits": 0, "operator_confirmation": False}
    print("E3_PUBLIC_RELEASE_RESULT=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    # The receipt is public, modest in size, and contains no credentials. Retain
    # this JSON in job logs as well as the artifact to permit audit without a ZIP.
    print("E3_PUBLIC_RELEASE_REPORT=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
