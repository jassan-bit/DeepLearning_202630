"""Preflight/run/resume the bounded, one-seed E3 development trial on Colab/CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e3_development import create_run, run_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "run", "resume"):
        item = sub.add_parser(command)
        item.add_argument("--npz", type=Path, required=True)
        item.add_argument("--expected-sha256", required=True)
        item.add_argument("--run-dir", type=Path, required=True)
        item.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if command in ("prepare", "preflight"):
            item.add_argument("--plan", type=Path)
        else:
            item.add_argument("--stop-after-epochs", type=int,
                              help="Pause after this many model-epochs in this invocation; recipe unchanged")
    args = parser.parse_args()
    try:
        if args.command in ("prepare", "preflight"):
            result = create_run(args.npz, args.expected_sha256, args.run_dir,
                                device=args.device, plan_path=args.plan)
        else:
            result = run_development(args.run_dir, args.npz, args.expected_sha256,
                                     resume=args.command == "resume", device=args.device,
                                     stop_after_epochs=args.stop_after_epochs)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"E3 development failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
