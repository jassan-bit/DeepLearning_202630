#!/usr/bin/env python3
"""Run only tiny self-generated synthetic E2 checks, never a market benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2_current_pipeline import SmokeConfig, run_synthetic_smoke


def main(argv: list[str] | None = None) -> int:
    from e2_current_models import MODEL_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=MODEL_NAMES,
                        help="Repeat to check a subset; default checks all 11 models.")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=2,
                        help="Supervised optimizer steps per model, capped at three.")
    parser.add_argument("--pretrain-steps", type=int, choices=(1, 2), default=1,
                        help="Training-only reconstruction steps for LSTM-AE.")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--source-root", type=Path,
                        help="Code only: directory containing pinned TimesNet, TimeMixer, PatchTST and iTransformer trees.")
    parser.add_argument("--output", type=Path,
                        help="Optional new *synthetic*.json receipt; default writes JSON to stdout.")
    args = parser.parse_args(argv)
    if args.output is not None:
        path = args.output.resolve()
        if path.suffix != ".json" or "synthetic" not in path.stem.lower():
            parser.error("Receipt output must be a new file named *synthetic*.json")
        if {"data", "results", "experiments", "references", "book", "book_e2", "book_e3"}.intersection(path.parts):
            parser.error("Synthetic smoke receipts must stay outside data, results, experiments, references and books")
        if path.exists():
            parser.error("Refusing to overwrite an existing receipt or historical artifact")
        if not path.parent.is_dir():
            parser.error("Receipt parent directory must already exist")
    config = SmokeConfig(seed=args.seed, supervised_steps=args.steps, pretrain_steps=args.pretrain_steps)
    try:
        config.validate()
        recipe = None
        if args.source_root is not None:
            recipe = {"source_roots": {name: str(args.source_root / name)
                      for name in ("TimesNet", "TimeMixer", "PatchTST", "iTransformer")}}
        receipt = run_synthetic_smoke(model_names=args.model, config=config, recipe=recipe)
        payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            # Exclusive creation also prevents overwriting after the initial check.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
