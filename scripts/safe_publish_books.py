"""Fail-closed, fast-forward publication of the complete three-book site.

`inventory` and `validate` never publish. `publish` defaults to a dry run;
only --execute permits a normal push. No force, branch creation, retry, source
checkout substitution, or rollback override is implemented.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from check_book_links import check_html

INVENTORY = "publication_inventory.json"
BOOKS = {"entregable1": ("book", ""), "entregable2": ("book_e2", "entregable2"),
         "entregable3": ("book_e3", "entregable3")}
DOWNLOAD_SUFFIXES = {".zip", ".pdf", ".csv", ".tsv", ".xlsx", ".xls", ".ipynb",
                     ".tar", ".gz", ".7z", ".pkl", ".parquet"}


class PublicationError(ValueError):
    """A publication precondition failed. Do not retry automatically."""


def require_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise PublicationError("A full lowercase source commit SHA is required")
    return value


def git(repo, *args, env=None, binary=False):
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    # Scope the optional token to the official GitHub HTTPS host, not URLs or logs.
    token = process_env.get("GH_TOKEN")
    if token:
        index = int(process_env.get("GIT_CONFIG_COUNT", "0"))
        process_env.update({"GIT_CONFIG_COUNT": str(index + 1),
            f"GIT_CONFIG_KEY_{index}": "http.https://github.com/.extraheader",
            f"GIT_CONFIG_VALUE_{index}": "AUTHORIZATION: basic " +
                base64.b64encode(("x-access-token:" + token).encode()).decode()})
    process_env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", "-C", str(repo), *args], env=process_env,
                            capture_output=True, check=False)
    if result.returncode:
        # Do not relay credentials or remote stderr into publication reports.
        raise PublicationError(f"git {args[0]} failed (exit {result.returncode}); no automatic retry")
    return result.stdout if binary else result.stdout.decode().strip()


def file_inventory(site):
    site = Path(site)
    if site.is_symlink() or not site.is_dir():
        raise PublicationError("Site must be a regular directory")
    files = {}
    for path in sorted(site.rglob("*")):
        relative = path.relative_to(site).as_posix()
        parts = PurePosixPath(relative).parts
        mode = path.lstat().st_mode
        if any(part in {".git", ".github"} for part in parts):
            raise PublicationError("Repository metadata must never enter the site")
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise PublicationError(f"Non-regular site entry: {relative}")
        if stat.S_ISREG(mode):
            if path.stat().st_nlink != 1:
                raise PublicationError(f"Hard-linked site entry: {relative}")
            if relative != INVENTORY:
                data = path.read_bytes()
                files[relative] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
    if not files:
        raise PublicationError("Empty site")
    return files


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"Invalid JSON file: {Path(path).name}") from error


def validate_site(site, source_sha, require_inventory=True, return_inventory=False):
    source_sha = require_sha(source_sha)
    site = Path(site)
    files = file_inventory(site)
    manifest = load_json(site / "books_manifest.json")
    if not isinstance(manifest, dict):
        raise PublicationError("Manifest must be a JSON object")
    if manifest.get("source_commit") != source_sha:
        raise PublicationError("Manifest source differs from the checked-out source")
    records = manifest.get("books", [])
    if not isinstance(records, list) or len(records) != 3:
        raise PublicationError("Manifest must contain exactly three books")
    seen = set()
    for book in records:
        if not isinstance(book, dict):
            raise PublicationError("Each book must be a JSON object")
        identity = book.get("id")
        if not isinstance(identity, str) or identity not in BOOKS or identity in seen:
            raise PublicationError("Missing, duplicate or unexpected book identity")
        seen.add(identity)
        source, prefix = BOOKS[identity]
        entry = (prefix + "/" if prefix else "") + "intro.html"
        if (book.get("source"), book.get("prefix"), book.get("entrypoint")) != (source, prefix, entry):
            raise PublicationError("Book route or source changed")
        for name in ("intro.html", "index.html", "objects.inv", "searchindex.js"):
            route = (prefix + "/" if prefix else "") + name
            if route not in files or not files[route]["bytes"]:
                raise PublicationError(f"Incomplete book: {route}")
        if book.get("intro_sha256") != files[entry]["sha256"]:
            raise PublicationError(f"Introduction checksum mismatch: {entry}")
    if "entregables.html" not in files or ".nojekyll" not in files:
        raise PublicationError("Missing selector or .nojekyll")
    links = check_html(site)
    if require_inventory:
        try:
            inventory_bytes = (site / INVENTORY).read_bytes()
            inventory = json.loads(inventory_bytes)
        except (OSError, ValueError) as error:
            raise PublicationError("Missing or invalid publication inventory") from error
        if (not isinstance(inventory, dict) or inventory.get("schema_version") != 1 or inventory.get("source_commit") != source_sha
                or inventory.get("files") != files):
            raise PublicationError("Site changed after inventory creation")
    result = {"source_commit": source_sha, "files": len(files),
              "bytes": sum(item["bytes"] for item in files.values()), **links}
    if return_inventory:
        expected = {name: item["sha256"] for name, item in files.items()}
        expected[INVENTORY] = sha256(inventory_bytes).hexdigest()
        return result, expected
    return result


def write_inventory(site, source_sha):
    validate_site(site, source_sha, require_inventory=False)
    path = Path(site) / INVENTORY
    path.write_text(json.dumps({"schema_version": 1, "source_commit": source_sha,
        "files": file_inventory(site)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return validate_site(site, source_sha)


def remote_heads(repo, remote):
    heads = {}
    for line in git(repo, "ls-remote", "--heads", remote, "refs/heads/main", "refs/heads/gh-pages").splitlines():
        sha, ref = line.split()
        heads[ref] = require_sha(sha)
    if set(heads) != {"refs/heads/main", "refs/heads/gh-pages"}:
        raise PublicationError("Both existing main and gh-pages branches are required")
    return heads


def check_source(repo, source_sha, remote="origin"):
    require_sha(source_sha)
    if git(repo, "rev-parse", "HEAD") != source_sha:
        raise PublicationError("Checkout differs from the requested source")
    git(repo, "diff", "--quiet", "HEAD", "--")
    if git(repo, "ls-files", "--others", "--exclude-standard"):
        raise PublicationError("Untracked source files are present; use a clean checkout")
    heads = remote_heads(repo, remote)
    if heads["refs/heads/main"] != source_sha:
        raise PublicationError("Obsolete source: main advanced; keep the current site")
    return heads


def previous_tree(repo, base):
    result = {}
    for record in git(repo, "ls-tree", "-r", "-z", base, binary=True).split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, sha = metadata.decode().split()
        path = name.decode("utf-8")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise PublicationError(f"Unsupported previous entry: {path}")
        result[path] = sha
    return result


def is_download(path):
    return "_downloads" in PurePosixPath(path).parts or PurePosixPath(path).suffix.lower() in DOWNLOAD_SUFFIXES


def preserve_previous(repo, base, site, source_sha):
    previous = previous_tree(repo, base)
    current = file_inventory(site)
    missing = sorted(set(previous) - set(current) - {INVENTORY})
    if missing:
        raise PublicationError("Previously published paths disappeared: " + ", ".join(missing[:10]))
    count = 0
    for path, blob in previous.items():
        if is_download(path):
            old_hash = sha256(git(repo, "cat-file", "blob", blob, binary=True)).hexdigest()
            if old_hash != current[path]["sha256"]:
                raise PublicationError(f"Published download changed: {path}; use a new versioned URL")
            count += 1
    try:
        manifest = json.loads(git(repo, "show", f"{base}:books_manifest.json"))
        if not isinstance(manifest, dict):
            raise PublicationError("Previous manifest must be a JSON object")
        previous_source = require_sha(manifest["source_commit"])
    except (KeyError, json.JSONDecodeError) as error:
        raise PublicationError("Previous source provenance is missing or invalid") from error
    git(repo, "merge-base", "--is-ancestor", previous_source, source_sha)
    return {"previous_source_commit": previous_source, "preserved_paths": len(previous),
            "unchanged_downloads": count}


def candidate_commit(repo, site, source_sha, base, expected_files=None):
    with tempfile.TemporaryDirectory(prefix="book-publication-index-") as temporary:
        env = {"GIT_INDEX_FILE": str(Path(temporary) / "index"),
               "GIT_AUTHOR_NAME": "github-actions[bot]",
               "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
               "GIT_COMMITTER_NAME": "github-actions[bot]",
               "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com"}
        git(repo, "read-tree", "--empty", env=env)
        git(repo, "--work-tree=" + str(Path(site).resolve()), "add", "--all", "--force", "--", ".", env=env)
        tree = git(repo, "write-tree", env=env)
        if expected_files is None:
            raise PublicationError("An independently validated inventory is required")
        staged = previous_tree(repo, tree)
        if set(staged) != set(expected_files):
            raise PublicationError("Staged paths differ from the validated site")
        for path, blob in staged.items():
            if sha256(git(repo, "cat-file", "blob", blob, binary=True)).hexdigest() != expected_files[path]:
                raise PublicationError("Staged content changed after validation: " + path)
        return git(repo, "commit-tree", tree, "-p", base, "-m",
                   "Publish complete books from " + source_sha, env=env)


def publish(repo, site, source_sha, remote="origin", execute=False, report_path=None):
    if Path(site).is_symlink():
        raise PublicationError("Site must not be a symlink")
    repo, site = Path(repo).resolve(), Path(site).resolve()
    heads = check_source(repo, source_sha, remote)
    base = heads["refs/heads/gh-pages"]
    git(repo, "fetch", "--no-tags", remote, source_sha, base)
    result, expected_files = validate_site(site, source_sha, return_inventory=True)
    result.update(preserve_previous(repo, base, site, source_sha))
    result.update({"previous_pages_commit": base, "mode": "execute" if execute else "dry-run",
                   "push_attempted": False, "pushed": False,
                   "push_outcome": "not_attempted", "web_deployment_verified": False})
    if execute:
        candidate = candidate_commit(repo, site, source_sha, base, expected_files)
        # The immutable Git tree is made only after validation. Do not reuse it
        # if either remote ref moved; no automatic rebase or force-with-lease.
        if check_source(repo, source_sha, remote) != heads:
            raise PublicationError("Publication race: remote refs moved; keep the current site")
        result["candidate_pages_commit"] = candidate
        # Persist uncertainty BEFORE invoking Git. If the process or network
        # disappears, a surviving report must not incorrectly claim no write.
        result.update({"push_attempted": True, "pushed": None, "push_outcome": "unknown"})
        if report_path:
            Path(report_path).write_text(json.dumps(result, indent=2) + "\n")
        try:
            git(repo, "push", remote, candidate + ":refs/heads/gh-pages")
        except PublicationError as error:
            if remote_heads(repo, remote)["refs/heads/gh-pages"] != candidate:
                raise PublicationError("Push failed or outcome uncertain; inspect remote before any new action") from error
        after = remote_heads(repo, remote)
        if after["refs/heads/gh-pages"] != candidate:
            raise PublicationError("Remote changed after push; do not claim current deployment")
        result["pushed"] = True
        result["push_outcome"] = "confirmed"
        result["source_current_after_push"] = after["refs/heads/main"] == source_sha
        if report_path:
            Path(report_path).write_text(json.dumps(result, indent=2) + "\n")
        if not result["source_current_after_push"]:
            raise PublicationError("main advanced during publication; branch was updated, but source is no longer current. No rollback performed")
    if report_path:
        Path(report_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def pages_workflow_status(repository, pages_sha):
    """Read the exact branch deployment, without credentials or write requests.

    This optional diagnostic supports public repositories. API unavailability
    must not turn into a failed deployment or a reason to republish anything.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise PublicationError("A GitHub owner/repository name is required")
    require_sha(pages_sha)
    request = Request(
        f"https://api.github.com/repos/{repository}/actions/runs?head_sha={pages_sha}&per_page=100",
        headers={"Accept": "application/vnd.github+json", "Cache-Control": "no-cache",
                 "User-Agent": "book-publication-verifier"}, method="GET")
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read(2 * 1024 * 1024 + 1))
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise PublicationError("Invalid workflow status response")
    matching = [run for run in payload["workflow_runs"] if isinstance(run, dict)
        and run.get("head_sha") == pages_sha and run.get("head_branch") == "gh-pages"
        and run.get("event") == "dynamic"
        and run.get("path") == "dynamic/pages/pages-build-deployment"
        and isinstance(run.get("id"), int) and not isinstance(run["id"], bool)
        and isinstance(run.get("run_attempt"), int)]
    if not matching:
        return None
    run = max(matching, key=lambda item: (item["id"], item["run_attempt"]))
    return {key: run.get(key) for key in ("id", "head_sha", "status", "conclusion", "run_attempt")} | {
        "url": f"https://github.com/{repository}/actions/runs/{run['id']}"}


def verify_live(site, source_sha, base_url, attempts=1, interval=30, report_path=None,
                repository=None):
    """Read back the expected manifest and the four entry pages; never roll back."""
    report = load_json(report_path) if report_path else {}
    if not isinstance(report, dict):
        raise PublicationError("Publication report must be a JSON object")
    # Never leave a prior successful verification in a report for a new check.
    for key in ("verified_source_commit", "verified_entrypoints", "verification_error",
                "observed_source_commit", "pages_workflow", "pages_diagnostic_error"):
        report.pop(key, None)
    report.update({"web_deployment_verified": False, "verification_status": "checking",
                   "expected_source_commit": source_sha, "verification_attempts": 0})

    def save_report():
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2) + "\n")

    def fail(message):
        report.update({"web_deployment_verified": False, "verification_status": "failed",
                       "verification_error": message})
        save_report()
        raise PublicationError(message)

    save_report()
    try:
        validate_site(site, source_sha)
        if urlsplit(base_url).scheme not in {"https", "http"} or attempts < 1 or interval < 0:
            raise PublicationError("A HTTP(S) site URL and valid retry bounds are required")
        pages_sha = None
        if repository:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
                raise PublicationError("A GitHub owner/repository name is required")
            if report.get("pushed") is not True:
                raise PublicationError("Pages diagnostics require a confirmed push report")
            pages_sha = require_sha(report.get("candidate_pages_commit"))
            if report.get("source_commit") != source_sha:
                raise PublicationError("Publication report source differs from the requested source")
    except (OSError, ValueError) as error:
        fail(str(error))
    base_url = base_url.rstrip("/") + "/"
    expected = load_json(Path(site) / "books_manifest.json")
    paths = [book["entrypoint"] for book in expected["books"]] + ["entregables.html"]
    expected_hashes = {path: sha256((Path(site) / path).read_bytes()).hexdigest() for path in paths}
    last_error = "not checked"
    for attempt in range(attempts):
        report["verification_attempts"] = attempt + 1
        try:
            def fetch(path):
                request = Request(urljoin(base_url, path) + "?source_check=" + source_sha,
                                  headers={"Cache-Control": "no-cache", "User-Agent": "book-publication-verifier"})
                with urlopen(request, timeout=30) as response:
                    return response.read(10 * 1024 * 1024 + 1)
            manifest = json.loads(fetch("books_manifest.json"))
            if not isinstance(manifest, dict):
                raise PublicationError("Public manifest is not a JSON object")
            report["observed_source_commit"] = manifest.get("source_commit")
            if manifest != expected:
                raise PublicationError("Public manifest is not the expected artifact; served source: " +
                                       str(manifest.get("source_commit")))
            for path, digest in expected_hashes.items():
                if sha256(fetch(path)).hexdigest() != digest:
                    raise PublicationError("Public page differs from the validated artifact: " + path)
            result = {"web_deployment_verified": True, "verified_source_commit": source_sha,
                      "verified_entrypoints": paths, "site_url": base_url,
                      "verification_status": "verified", "verification_attempts": attempt + 1}
            report.update(result)
            report.pop("verification_error", None)
            save_report()
            return result
        except (OSError, ValueError, HTTPException) as error:
            last_error = str(error)
        report["verification_error"] = last_error
        if pages_sha:
            try:
                workflow = pages_workflow_status(repository, pages_sha)
                report["pages_workflow"] = workflow
                report.pop("pages_diagnostic_error", None)
            except (OSError, ValueError, HTTPException) as error:
                workflow = None
                report["pages_diagnostic_error"] = str(error)
            if (workflow and workflow["status"] == "completed"
                    and workflow["conclusion"] in {"failure", "cancelled", "timed_out"}):
                fail("Pages workflow for the published commit ended with " + workflow["conclusion"]
                     + ": " + workflow["url"] + "; website delivery remains unverified: " + last_error)
        save_report()
        if attempt + 1 < attempts:
            time.sleep(interval)
    fail("Branch publication does not confirm website delivery: " + last_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "validate", "check-source", "publish", "verify-live"))
    parser.add_argument("--site", type=Path, default=Path("_build/site"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--repository", help="Optional public GitHub owner/repo for read-only Pages diagnostics")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            result = write_inventory(args.site, args.source_commit)
        elif args.command == "validate":
            result = validate_site(args.site, args.source_commit)
        elif args.command == "check-source":
            result = check_source(args.repo, args.source_commit, args.remote)
        elif args.command == "verify-live":
            if not args.url:
                raise PublicationError("--url is required for verify-live")
            result = verify_live(args.site, args.source_commit, args.url, args.attempts, args.interval,
                                 args.report, args.repository)
        else:
            result = publish(args.repo, args.site, args.source_commit, args.remote, args.execute, args.report)
        print(json.dumps(result, indent=2))
    except (PublicationError, ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()

