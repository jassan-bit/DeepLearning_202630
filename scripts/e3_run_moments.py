"""Credential-free subprocess entry point for the bounded moment pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "resume"):
        item = sub.add_parser(command)
        item.add_argument("--npz", type=Path, required=True)
        item.add_argument("--expected-sha256", required=True)
        item.add_argument("--run-dir", type=Path, required=True)
        item.add_argument("--global-claim-path", "--global-claim", dest="global_claim_path", type=Path, required=True)
        item.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
        item.add_argument("--reference-dir", type=Path, required=command == "preflight")
        item.add_argument("--snapshot-manifest-path", type=Path, required=command == "preflight")
        if command == "preflight":
            item.add_argument("--plan", type=Path, required=True)
            item.add_argument("--initial-ledger-path", "--initial-ledger", dest="initial_ledger_path", type=Path, required=True)
        else:
            item.add_argument("--stop-after-epochs", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        from src.e3_moment_training import create_moment_run, run_moment
        claim = json.loads(args.global_claim_path.read_text())
        common = dict(global_claim=claim, device=args.device, frozen_v2_dir=args.reference_dir,
                      snapshot_manifest_path=args.snapshot_manifest_path)
        if args.command == "preflight":
            ledger = json.loads(args.initial_ledger_path.read_text())
            result = create_moment_run(args.npz, args.expected_sha256, args.run_dir,
                plan_path=args.plan, initial_ledger=ledger, **common)
        else:
            result = run_moment(args.run_dir, args.npz, args.expected_sha256,
                resume=args.command == "resume", stop_after_epochs=args.stop_after_epochs, **common)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"El piloto de momentos se detuvo: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
