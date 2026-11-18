"""Persist one bibliographic snapshot using the existing GitHub integrity helpers.

Only an immutable new results branch is created. Main and old evidence stay intact.
AI assistance declared, 2026-11-18.
"""
from pathlib import Path
import base64
import hashlib
import json
import re


def snapshot(output, token, source, run_id, api):
    if re.fullmatch(r"[0-9a-f]{40}", source) is None or re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", run_id) is None:
        raise api.StorageError("invalid_source_or_run_id")
    output = api._regular_path(Path(output), directory=True)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise api.StorageError("symlink_refused")
        if not path.is_file():
            continue
        name = path.relative_to(output).as_posix()
        if token in name or not api._safe_relative(name) or path.suffix not in {".json", ".xml", ".csv"}:
            raise api.StorageError("unexpected_evidence_file")
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024 or token.encode() in raw:
            raise api.StorageError("evidence_size_or_secret_refused")
        files[name] = raw
    if sum(map(len, files.values())) > 32 * 1024 * 1024:
        raise api.StorageError("snapshot_exceeds_32MiB")
    receipt = json.loads(files["search_receipt.json"])
    if (receipt.get("kind") != "e3_r1_native_bibliographic_collection" or receipt.get("source_commit") != source
            or receipt.get("R1_accredited") is not False or receipt.get("human_external_review_completed") is not False
            or any(type(receipt.get(k)) is not int or receipt[k] != 0
                   for k in ("new_real_data_fits", "new_market_predictions", "new_statistical_contrasts"))):
        raise api.StorageError("bibliographic_scope_contract_failed")
    expected_files = {"search_receipt.json", "screening.csv"}
    for row in receipt.get("attempts", []):
        if row.get("response_file"):
            name = row["response_file"]
            if (name not in files or not name.startswith("responses/")
                    or len(files[name]) != row.get("response_bytes")
                    or hashlib.sha256(files[name]).hexdigest() != row.get("response_sha256")):
                raise api.StorageError("native_response_integrity_mismatch")
            expected_files.add(name)
    if set(files) != expected_files:
        raise api.StorageError("unexpected_or_missing_evidence_files")
    manifest = {"kind": "e3_r1_search_storage_manifest", "source_commit": source, "run_id": run_id,
                "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                  "git_blob_sha": api._git_sha(raw)} for name, raw in files.items()}}
    files["storage_manifest.json"] = api._canonical(manifest)
    return files


def save(output, token, source, run_id, api):
    token = api._token(token)
    files = snapshot(output, token, source, run_id, api)
    api.validate_credentials(token)
    root = "/repos/" + api.REPOSITORY
    branch = "e3-r1-search-" + run_id
    prefix = "runs/r1_search/" + run_id
    source_data = api._request("GET", root + "/git/commits/" + source, token)
    if source_data.get("sha") != source:
        raise api.StorageError("source_commit_identity_mismatch")
    base_tree = source_data["tree"]["sha"]
    old = api._tree_map(api._request("GET", root + "/git/trees/" + base_tree + "?recursive=1", token))
    if any(p.startswith(prefix + "/") for p in old):
        raise api.StorageConflict("run_already_in_source")
    expected = {prefix + "/" + name: api._git_sha(raw) for name, raw in files.items()}
    ref_path = root + "/git/ref/heads/" + branch

    def verify(ref):
        if ref.get("ref") != "refs/heads/" + branch or ref.get("object", {}).get("type") != "commit":
            raise api.StorageConflict("unexpected_results_ref")
        sha = ref["object"]["sha"]
        commit = api._request("GET", root + "/git/commits/" + sha, token)
        if commit.get("sha") != sha or [p.get("sha") for p in commit.get("parents", [])] != [source]:
            raise api.StorageConflict("existing_run_has_different_source")
        api._verify_tree(api._request("GET", root + "/git/trees/" + commit["tree"]["sha"] + "?recursive=1", token), old, expected, prefix)
        return sha

    existing = api._request("GET", ref_path, token, allow_missing=True)
    if existing is None:
        entries = []
        for name, raw in sorted(files.items()):
            blob = api._request("POST", root + "/git/blobs", token,
                                {"encoding": "base64", "content": base64.b64encode(raw).decode()})
            if blob.get("sha") != api._git_sha(raw):
                raise api.StorageError("uploaded_blob_integrity_mismatch")
            entries.append({"path": prefix + "/" + name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        tree = api._request("POST", root + "/git/trees", token, {"base_tree": base_tree, "tree": entries})
        api._verify_tree(api._request("GET", root + "/git/trees/" + tree["sha"] + "?recursive=1", token), old, expected, prefix)
        commit = api._request("POST", root + "/git/commits", token,
                              {"message": "Guardar búsqueda bibliográfica R1 " + run_id,
                               "tree": tree["sha"], "parents": [source]})
        verify({"ref": "refs/heads/" + branch, "object": {"type": "commit", "sha": commit["sha"]}})
        try:
            api._request("POST", root + "/git/refs", token, {"ref": "refs/heads/" + branch, "sha": commit["sha"]})
        except api.StorageError:
            recovered = api._request("GET", ref_path, token, allow_missing=True)
            if recovered is None:
                raise
            verify(recovered)
    sha = verify(api._request("GET", ref_path, token))
    return {"saved": True, "commit_sha": sha, "main_updated": False,
            "run_url": "https://github.com/" + api.REPOSITORY + "/tree/" + branch + "/" + prefix,
            "persisted_files": len(files), "all_files_verified": True}
