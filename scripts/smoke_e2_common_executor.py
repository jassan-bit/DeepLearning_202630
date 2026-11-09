#!/usr/bin/env python3
"""Run the fixed 18-system synthetic common executor; never loads market data."""
import argparse
import json
from pathlib import Path
from e2_common_executor import run_synthetic_common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory containing synthetic in its name')
    parser.add_argument('--source-root', type=Path, required=True, help='Pinned author source-code parent')
    args = parser.parse_args(argv)
    roots = {n: args.source_root/n for n in ('TimesNet', 'TimeMixer', 'PatchTST', 'iTransformer')}
    result = run_synthetic_common(args.output, synthetic_only=True, source_roots=roots)
    print(json.dumps({'status': result['status'], 'system_count': result['system_count'],
                      'total_fit_stages': result['total_fit_stages'],
                      'output': str(args.output)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
