"""Isolated worker for the claimed, resumable CNN-LSTM-attention v2 pilot.

The GitHub coordinator must acknowledge the global claim before invoking this
worker, including preflight. This CLI never reads credentials or publishes.
"""
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
        if command == "preflight":
            item.add_argument("--plan", type=Path, required=True)
            item.add_argument("--initial-ledger-path", "--initial-ledger", dest="initial_ledger_path", type=Path, required=True)
            item.add_argument("--frozen-v1-dir", type=Path, required=True)
            item.add_argument("--snapshot-manifest-path", "--snapshot-manifest", dest="snapshot_manifest_path", type=Path, required=True)
        else:
            item.add_argument("--stop-after-epochs", type=int, default=12,
                              help="Complete model-epochs per acknowledged chunk; recipe unchanged")
            item.add_argument("--frozen-v1-dir", type=Path)
            item.add_argument("--snapshot-manifest-path", "--snapshot-manifest", dest="snapshot_manifest_path", type=Path)
    args = parser.parse_args(argv)
    try:
        from src.e3_innovation import create_innovation_run, run_innovation

        claim = json.loads(args.global_claim_path.read_text(encoding="utf-8"))
        common = dict(global_claim=claim, device=args.device)
        if args.command == "preflight":
            ledger = json.loads(args.initial_ledger_path.read_text(encoding="utf-8"))
            result = create_innovation_run(
                args.npz, args.expected_sha256, args.run_dir,
                plan_path=args.plan, frozen_v1_dir=args.frozen_v1_dir,
                snapshot_manifest_path=args.snapshot_manifest_path,
                initial_ledger=ledger, **common)
        else:
            result = run_innovation(
                args.run_dir, args.npz, args.expected_sha256,
                resume=args.command == "resume", stop_after_epochs=args.stop_after_epochs,
                frozen_v1_dir=args.frozen_v1_dir, snapshot_manifest_path=args.snapshot_manifest_path,
                **common)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"El piloto E3 v2 se detuvo: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
