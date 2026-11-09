#!/usr/bin/env python3
"""Run the eleven section 22 recipes on generated synthetic data only."""
import argparse
import json
from pathlib import Path
from e2_current_pipeline import SmokeConfig
from e2_recipe_catalogue import MODEL_NAMES, run_synthetic_smoke


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', choices=MODEL_NAMES)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--steps', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--pretrain-steps', type=int, choices=(1, 2), default=1)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output:
        path = args.output.resolve()
        if path.suffix != '.json' or 'synthetic' not in path.stem.lower():
            parser.error('Output must be a new *synthetic*.json file')
        if {'data', 'results', 'experiments', 'references', 'book', 'book_e2', 'book_e3'}.intersection(path.parts):
            parser.error('Receipt must stay outside historical data/results/experiments/references/books')
        if path.exists() or not path.parent.is_dir():
            parser.error('Receipt must be new and its parent must already exist')
    roots = None if args.source_root is None else {n: str(args.source_root/n) for n in ('TimesNet', 'TimeMixer', 'PatchTST', 'iTransformer')}
    config = SmokeConfig(seed=args.seed, supervised_steps=args.steps, pretrain_steps=args.pretrain_steps)
    try:
        receipt = run_synthetic_smoke(model_names=args.model, config=config, source_roots=roots)
        payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(payload)
        else:
            print(payload, end='')
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
