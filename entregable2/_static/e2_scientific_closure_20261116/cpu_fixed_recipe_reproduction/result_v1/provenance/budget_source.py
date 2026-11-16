"""Append-only accounting for E2 corrections, continuing the closed 84 h budget.

Charges the wall time of scientific commands, including failures/timeouts. This
is neither GPU utilization nor a monetary account. Concurrent reservations are
conservative and share one ledger; the historical account is never rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "book_e2/_static/e2_reproduction_ablations_20261113/budget_final.json"
BASE_SHA = "a3949762183627e5e7e5b39c9411c6d64214d726bfad03c83fc2ac3eeeea07e4"
LEDGER = ROOT / "book_e2/_static/e2_scientific_closure_20261116/compute_ledger.jsonl"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def utc():
    return datetime.now(timezone.utc).isoformat()


def summarize(records):
    if not records or records[0]["event"] != "anchor":
        raise ValueError("Missing historical budget anchor")
    head = records[0]
    reservations, finished = {}, set()
    charged = 0.0
    for row in records[1:]:
        key = row["reservation_id"]
        if row["event"] == "reserve":
            if key in reservations or key in finished or row["cap_seconds"] <= 0:
                raise ValueError("Invalid or duplicate reservation")
            reservations[key] = row
        elif row["event"] == "finish":
            if key not in reservations or row["elapsed_seconds"] < 0:
                raise ValueError("Unmatched or invalid completion")
            charged += row["elapsed_seconds"]
            del reservations[key]
            finished.add(key)
        else:
            raise ValueError("Unknown budget event")
    held = sum(row["cap_seconds"] for row in reservations.values())
    remaining = head["historical_remaining_seconds"] - charged
    return {
        "historical_budget_sha256": head["historical_budget_sha256"],
        "approved_total_hours": head["approved_total_hours"],
        "historical_charged_seconds": head["historical_charged_seconds"],
        "corrections_charged_seconds": charged,
        "cumulative_charged_seconds": head["historical_charged_seconds"] + charged,
        "remaining_seconds": remaining, "reserved_seconds": held,
        "unreserved_seconds": remaining - held,
        "open_reservations": reservations,
        "budget_reset": False,
        "accounting_unit": "sum of scientific-command wall seconds, not active GPU time or money",
    }


@contextmanager
def locked_ledger(path, base_path=BASE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Ledger must not be a symlink")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        lines = stream.readlines()
        rows, previous = [], None
        for raw in lines:
            if not raw.endswith(b"\n"):
                raise ValueError("Incomplete ledger record; reconcile without rewriting history")
            row = json.loads(raw)
            if row["previous_record_sha256"] != previous:
                raise ValueError("Broken ledger hash chain")
            rows.append(row)
            previous = hashlib.sha256(raw).hexdigest()

        def append(row):
            nonlocal previous
            row = dict(row, at_utc=utc(), previous_record_sha256=previous)
            raw = canonical(row) + b"\n"
            stream.seek(0, os.SEEK_END)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            previous = hashlib.sha256(raw).hexdigest()

        base_raw = Path(base_path).read_bytes()
        base = json.loads(base_raw)
        checksum = hashlib.sha256(base_raw).hexdigest()
        if Path(base_path) == BASE and checksum != BASE_SHA:
            raise ValueError("Historical closing account changed")
        if not rows:
            if base.get("unresolved_reservations"):
                raise ValueError("Historical reservations must first be reconciled")
            append({"event": "anchor", "historical_budget_sha256": checksum,
                    "historical_charged_seconds": base["total_charged_seconds"],
                    "historical_remaining_seconds": base["remaining_seconds"],
                    "approved_total_hours": base["approved_total_hours"],
                    "historical_closed_at_utc": base["closed_at_utc"],
                    "budget_reset": False})
        elif rows[0]["historical_budget_sha256"] != checksum:
            raise ValueError("Ledger no longer matches its historical account")
        yield rows, append


def reserve(phase, cap, *, path=LEDGER, base_path=BASE):
    if not phase or not 0 < cap < float("inf"):
        raise ValueError("A phase and a finite positive reservation are required")
    with locked_ledger(path, base_path) as (rows, append):
        state = summarize(rows)
        if cap > state["unreserved_seconds"]:
            raise ValueError("Reservation exceeds the remaining authorized account")
        key = str(uuid.uuid4())
        append({"event": "reserve", "reservation_id": key, "phase": phase,
                "cap_seconds": float(cap)})
    return key


def finish(key, elapsed, returncode, *, path=LEDGER, base_path=BASE):
    if not 0 <= elapsed < float("inf"):
        raise ValueError("Elapsed time must be finite and nonnegative")
    with locked_ledger(path, base_path) as (rows, append):
        active = summarize(rows)["open_reservations"]
        if key not in active:
            raise ValueError("Reservation is missing or already settled")
        append({"event": "finish", "reservation_id": key,
                "phase": active[key]["phase"], "elapsed_seconds": float(elapsed),
                "cap_seconds": active[key]["cap_seconds"], "returncode": returncode,
                "overrun_seconds": max(0.0, elapsed-active[key]["cap_seconds"])})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("--phase", required=True)
    run.add_argument("--cap-seconds", type=float, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.operation == "status":
        with locked_ledger(args.ledger) as (rows, _):
            print(json.dumps(summarize(rows), indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    key = reserve(args.phase, args.cap_seconds, path=args.ledger)
    started, code = time.monotonic(), 1
    print(json.dumps({"budget_reservation_id": key, "phase": args.phase,
                      "cap_seconds": args.cap_seconds}), flush=True)
    try:
        code = subprocess.run(command, timeout=args.cap_seconds, check=False).returncode
    except subprocess.TimeoutExpired:
        code = 124
    except KeyboardInterrupt:
        code = 130
    finally:
        finish(key, time.monotonic()-started, code, path=args.ledger)
    return code


if __name__ == "__main__":
    sys.exit(main())
