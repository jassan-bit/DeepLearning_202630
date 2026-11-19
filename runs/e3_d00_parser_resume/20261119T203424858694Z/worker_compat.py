"""Disclosed post-freeze D00 parser compatibility amendment.

The immutable scientific checkout and input bytes are never edited.  Only the
in-memory auxiliary BTC-history parser is replaced.  The original CLI,
coordinator, model recipes, completed-job cache, cohorts and budget remain in
force.  A blank complete calendar bar stays absent; it is never imputed.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = "5a07d6157a9796ca54416dc5dea1e1527e731597"
MASTER = "20261119T031104515520Z"
PROTOCOL_SHA256 = "5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a"
TAG_OBJECT = "ba8b5c6fde85cef3deb57b072ab3570a53585b29"
SCOPE = "experiments/e3_evaluacion_ampliada_20261119"
FROZEN_FILE_SHA256 = {
    "evaluation/run.py": "55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3",
    "evaluation/data.py": "22615ba4c1b3ffbb6523036b14d98b7459da290955a8ea416fcf6249c0872bbf",
    "admission/d00_history.py": "a603bc32b71323b060f36985f4adeb248ca3fc191b222dfcecd02cf93fb6709e",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "Parser compatibility refuses symlinked source/input paths")
    require(path.is_file(), "Parser compatibility requires an existing regular file")
    return path


def load_frozen(path):
    """Verify the three parser/coordinator files before importing any of them."""
    run_path = regular(path)
    require(run_path.name == "run.py" and run_path.parent.name == "evaluation",
            "Expected immutable evaluation/run.py")
    scope = run_path.parent.parent
    checked = {}
    for relative, expected in FROZEN_FILE_SHA256.items():
        p = regular(scope / relative)
        require(sha(p) == expected, "Frozen source digest differs: " + relative)
        checked[relative] = expected
    spec = importlib.util.spec_from_file_location("e3_d00_compat_frozen_run", run_path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    history = original.load(scope / "admission/d00_history.py", "e3_d00_compat_pinned_history")
    require(sha(original.data_api.__file__) == FROZEN_FILE_SHA256["evaluation/data.py"],
            "Imported data module differs from pinned frozen file")
    original._d00_compat_history = history
    original._d00_compat_source_checks = checked
    return original


def install_parser(original):
    """Change one function in memory; use the original NumPy arithmetic."""
    history_api = original._d00_compat_history
    data_api = original.data_api
    np = data_api.np

    def btc_history(paths, expected_hashes):
        if len(paths) != 2 or sorted(data_api.sha(p) for p in paths) != sorted(expected_hashes):
            raise ValueError("Both pinned daily BTC source files are required for classical causal history")
        # This frozen validator checks hashes, complete ordered calendars,
        # headers, coherent finite bars, and rejects partially missing bars.
        # Its records retain the all-missing 2023-03-24 calendar entry.
        records = history_api.history(paths)
        closes = {r["date"]: r["values"][3] for r in records if r["values"] is not None}
        history = {}
        for day in sorted(closes):
            end = date.fromisoformat(day)
            support = [(end - timedelta(days=j)).isoformat() for j in range(7, -1, -1)]
            if all(d in closes for d in support):
                history[day] = float(np.std(np.diff(np.log([closes[d] for d in support])), ddof=0))
        return history

    original.data_api.btc_history = btc_history
    return btc_history


def parse_forwarded(arguments):
    """Validate required preflight inputs without consuming original CLI args."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("output", "source-commit", "admission", "protocol", "freeze-receipt", "npz"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--btc-history", nargs=2, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--package-path")
    return parser.parse_args(arguments)


def _job_inventory(output):
    result = {}
    if (output / "jobs").exists():
        for p in sorted((output / "jobs").rglob("result.json")):
            regular(p)
            result[p.relative_to(output).as_posix()] = sha(p)
    raw = canonical(result).encode()
    return {"completed_job_results": len(result), "result_inventory_sha256": hashlib.sha256(raw).hexdigest()}


def preflight(original, args):
    """Data-only checks. Never construct, fit or predict with a model."""
    require(args.source_commit == SOURCE, "Compatibility amendment is limited to its frozen source commit")
    protocol_path = regular(args.protocol)
    freeze_path = regular(args.freeze_receipt)
    npz_path = regular(args.npz)
    regular(args.admission)
    paths = [regular(p) for p in args.btc_history]
    require(sha(protocol_path) == PROTOCOL_SHA256, "Frozen protocol SHA256 differs")
    p, freeze = original.protocol(protocol_path, freeze_path, SOURCE)
    require(freeze["tag_object_sha"] == TAG_OBJECT and
            freeze["annotated_tag"] == "e3-extended-audit-protocol-" + MASTER,
            "Annotated protocol tag differs from this master campaign")
    output = Path(args.output).absolute()
    require(not any(x.is_symlink() for x in (output, *output.parents)),
            "Compatibility output cannot traverse symlinks")
    require(output.is_dir(), "Existing campaign output required; no new campaign is created")
    before = _job_inventory(output)
    input_hashes = {str(path): sha(path) for path in [npz_path, *paths]}
    history_api = original._d00_compat_history
    records = history_api.history(paths)
    parity = history_api.verify_npz(records, npz_path)
    data = original.data_api.btc(npz_path, p, evaluation=True, history_paths=paths)
    cohorts = {k: len(v) for k, v in data["positions"].items()}
    require(cohorts == {"fit": 532, "validation": 116, "calibration": 53, "test": 608},
            "D00 frozen cohort geometry changed")
    missing_bars = [r["date"] for r in records if r["values"] is None]
    require(missing_bars == ["2023-03-24"], "Pinned D00 missing-bar calendar differs")
    axis = original.classical._axis(data)
    missing_labels = [day for day in axis if day not in data["history"]]
    expected_missing = [(date(2023, 3, 24) + timedelta(days=k)).isoformat() for k in range(8)]
    require(missing_labels == expected_missing, "Missing daily sigma7 states were altered")
    lookup = {day: i for i, day in enumerate(axis)}
    required = {str(day) for split in ("val", "calibration", "test")
                for day in data[split + "_dates"]}
    unavailable = []
    for day in sorted(required):
        i = lookup[day]
        for lag in (1, 7):
            if i < lag or not original.data_api.np.isfinite(data["history"].get(axis[i-lag], float("nan"))):
                unavailable.append({"date": day, "lag": lag})
    require(not unavailable, "A required held-out temporal-control lag is unavailable")
    fit_diagnostics, excluded = original.classical.fit_diagnostic_mask(data)
    require(before == _job_inventory(output), "Completed-job evidence changed during data-only preflight")
    require(input_hashes == {str(path): sha(path) for path in [npz_path, *paths]},
            "Input bytes changed during data-only preflight")
    report = {
        "kind": "e3_disclosed_D00_parser_compatibility_preflight",
        "status": "passed_data_only", "source_commit": SOURCE, "master_run_id": MASTER,
        "protocol_sha256": PROTOCOL_SHA256, "protocol_tag_object_sha": TAG_OBJECT,
        "worker_sha256": sha(__file__), "frozen_source_sha256": original._d00_compat_source_checks,
        "post_freeze_execution_amendment": True, "AI_assistance_declared": True,
        "amendment_scope": "In-memory auxiliary BTC history parser accepts the already documented all-missing calendar bar; immutable source and input bytes stay unchanged.",
        "arithmetic": "Original numpy.std(numpy.diff(numpy.log(eight consecutive observed closes)), ddof=0)",
        "numpy_version": original.data_api.np.__version__,
        "calendar_rows": len(records), "missing_OHLCV_dates": missing_bars,
        "retained_missing_sigma7_calendar_states": missing_labels,
        "daily_axis_rows": len(axis), "observed_sigma7_dates": len(data["history"]),
        "cohorts": cohorts, "fit_diagnostic_rows": len(fit_diagnostics),
        "predeclared_fit_diagnostic_exclusions": len(excluded),
        "missing_required_control_lags": unavailable, "cohort_parity": parity,
        "input_files": [{"file": x.name, "sha256": sha(x), "bytes": x.stat().st_size}
                        for x in [npz_path, *paths]],
        "existing_jobs_before_and_after": before,
        "rows_filled": 0, "cohort_rows_removed": 0, "new_real_data_fits": 0,
        "new_market_predictions": 0, "models_constructed": 0,
        "new_campaign_started": False, "R1_accredited": False, "R3_accredited": False,
    }
    destination = output / "parser_compatibility" / "d00_preflight.json"
    require(not any(x.is_symlink() for x in (destination, *destination.parents)),
            "Compatibility report cannot traverse symlinks")
    destination.parent.mkdir(exist_ok=True)
    raw = (canonical(report) + "\n").encode()
    if destination.exists():
        existing = json.loads(regular(destination).read_bytes())
        # Keep the first preflight as immutable evidence. New successful jobs
        # may legitimately have accumulated before a later resume.
        old_binding = {k: v for k, v in existing.items() if k != "existing_jobs_before_and_after"}
        new_binding = {k: v for k, v in report.items() if k != "existing_jobs_before_and_after"}
        require(old_binding == new_binding,
                "Existing parser preflight bindings differ; preserve it for explicit reconciliation")
        report = existing
    else:
        with destination.open("xb") as stream:
            stream.write(raw)
    print(canonical({"event": "d00_parser_compatibility_preflight", "status": report["status"],
                     "cohorts": cohorts, "existing_completed_jobs": before["completed_job_results"],
                     "missing_required_control_lags": len(unavailable),
                     "new_real_data_fits": 0, "R3_accredited": False}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--frozen-run", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    own, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    original = load_frozen(own.frozen_run)
    install_parser(original)
    if own.preflight_only:
        return preflight(original, parse_forwarded(remaining))
    # The amended parser has no independent evaluation implementation. Every
    # original flag (and its order/value) is forwarded without alteration.
    sys.argv = [str(Path(own.frozen_run)), *remaining]
    return original.main()


if __name__ == "__main__":
    main()
