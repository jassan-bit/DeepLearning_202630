"""Exhaustive synthetic numerical checks of both SAE NAS auxiliary phases.

The output is a new ledger directory.  No optimizer, data loader, selection,
pretraining, or market access is supplied.  The phase-two fixture has width w1
but is artificial, not a learned representation from a trained phase one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

try:
    from .e2_nas_common import NASConfig
    from .e2_nas_autoencoder import SAEAutoencoderNAS
    from . import e2_search_plan as planner
except ImportError:
    from e2_nas_common import NASConfig
    from e2_nas_autoencoder import SAEAutoencoderNAS
    import e2_search_plan as planner

MODEL = 'LSTM-autoencoder con predictor'
VERSION = 'e2-sae-auxiliary-conformance-v1'
FIXTURE_SEED = 280701


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(architecture: dict, phase: int) -> dict:
    config = NASConfig(architecture['widths'], architecture['dropout_pattern'])
    torch.manual_seed(FIXTURE_SEED)
    with torch.device('cpu'):
        model = SAEAutoencoderNAS(config, phase).train()
        channels = 5 if phase == 1 else config.widths[0]
        inputs = torch.linspace(-0.9, 0.9, 7 * channels).reshape(1, 7, channels).requires_grad_(True)
        result = model(inputs)
    if tuple(result.shape) != (1, 7, 5) or not bool(torch.isfinite(result).all()):
        raise AssertionError('Wrong or nonfinite reconstruction')
    result.square().mean().backward()
    if inputs.grad is None or not bool(torch.isfinite(inputs.grad).all()):
        raise AssertionError('Disconnected or nonfinite input gradient')
    for name, parameter in model.named_parameters():
        if parameter.grad is None or parameter.grad.shape != parameter.shape:
            raise AssertionError('Disconnected parameter ' + name)
        if not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError('Nonfinite parameter gradient ' + name)
    return {
        'architecture_id': architecture['architecture_id'], 'phase': phase,
        'output_shape': list(result.shape), 'input_channels': channels,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'all_parameter_gradients_connected': True,
        'numerical_finiteness_checked': True,
    }


def run(protocol_path: Path, output: Path, *, threads: int = 1) -> dict:
    if type(threads) is not int or threads < 1:
        raise ValueError('Thread count must be a positive integer')
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    planner.validate_protocol(protocol)
    profiles = [m for m in protocol['models'] if m['model'] == MODEL]
    if len(profiles) != 1 or not profiles[0]['nas'] or profiles[0]['search_recipe_id'] != 'SAE-D7-NAS-v1':
        raise ValueError('Expected the affirmative versioned SAE NAS design')
    rows = planner.enumerate_architectures(protocol, MODEL)
    if len(rows) != 1000:
        raise AssertionError('The complete space must contain 1000 architectures')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_names = ['scripts/e2_nas_common.py', 'scripts/e2_nas_autoencoder.py',
                    'scripts/e2_search_plan.py', 'scripts/check_e2_nas_auxiliary.py']
    source_hashes = {name: _sha(root / name) for name in source_names}
    counts, failures = {1: 0, 2: 0}, []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    ledger_path = output / 'cases.jsonl'
    try:
        with ledger_path.open('x', encoding='utf-8') as ledger:
            for architecture in rows:
                for phase in (1, 2):
                    record = _case(architecture, phase)
                    ledger.write(json.dumps(record) + '\n')
                    counts[phase] += 1
                if counts[1] % 100 == 0:
                    print(json.dumps({'checked_architectures': counts[1],
                                      'auxiliary_phases': sum(counts.values())}), flush=True)
        if source_hashes != {name: _sha(root / name) for name in source_names}:
            raise RuntimeError('Source files changed during the conformance check')
    except Exception as error:
        failures.append({'type': type(error).__name__, 'message': str(error)})
    finally:
        torch.set_num_threads(old_threads)
    summary = {
        'version': VERSION, 'status': 'complete' if counts == {1: 1000, 2: 1000} and not failures else 'failed',
        'mode': 'numeric', 'architectures_each_phase': counts,
        'total_auxiliary_constructions': sum(counts.values()),
        'optimizer_steps': 0, 'market_data_read': False,
        'hpo_search_executed': False, 'nas_search_executed': False,
        'numerical_finiteness_checked': True,
        'protocol_file_sha256': _sha(protocol_path), 'source_sha256': source_hashes,
        'ledger_sha256': _sha(ledger_path), 'failures': failures,
        'runtime': {'torch': torch.__version__, 'python': platform.python_version(),
                    'device': 'cpu', 'threads': threads},
        'fixture_seed': FIXTURE_SEED,
        'fixture': 'B1,7,C linear values in[-0.9,0.9]; C5 in AE1, Cw1 in AE2; training mode',
        'limits': ['No pretraining or optimizer execution.',
                   'Fold staging and transfer tested separately on numerical fixtures.',
                   'Phase-two fixture is not a learned AE1 representation.',
                   'Auxiliary subgraphs may repeat when only w3 or b3 changes; all1000 parent architecture IDs are checked.'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if failures:
        raise RuntimeError(json.dumps(failures))
    return summary


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=root / 'book_e2/_static/search_protocol_20261109/protocol.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing paths are refused')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    summary = run(args.protocol, args.output, threads=args.threads)
    print(json.dumps({'status': summary['status'],
                      'total_auxiliary_constructions': summary['total_auxiliary_constructions'],
                      'optimizer_steps': summary['optimizer_steps']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
