"""Recompute a source-local classification scenario from frozen author data only."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
paths = [BASE / 'author_raw' / n for n in ['results-inception-128.csv', 'results-InceptionTime-128.csv']]
single, ensemble = [list(csv.DictReader(p.open())) for p in paths]
by_dataset = defaultdict(list)
for row in single:
    assert row['classifier_name'] == 'inception'
    assert row['archive_name'] == 'UCRArchive_2018'
    by_dataset[row['dataset_name']].append(row)
assert len(single) == 640 and len(by_dataset) == 128
for rows in by_dataset.values():
    assert len(rows) == 5 and {float(r['iteration']) for r in rows} == {0., 1., 2., 3., 4.}
ens_by_dataset = {r['dataset_name']: r for r in ensemble}
assert len(ensemble) == len(ens_by_dataset) == 128
assert set(by_dataset) == set(ens_by_dataset)
assert all(r['archive_name'] == 'UCRArchive_2018' and r['classifier_name'] == 'InceptionTime' for r in ensemble)
single_error = mean(mean(1 - float(r['accuracy']) for r in rows) for rows in by_dataset.values())
ensemble_error = mean(1 - float(r['accuracy']) for r in ensemble)
hi, lo = max(single_error, ensemble_error), min(single_error, ensemble_error)
result = {
    'single_mean_error': single_error,
    'ensemble5_mean_error': ensemble_error,
    'P_single_two_variant_pool': (hi - single_error) / (hi - lo),
    'formula_single': 'L_single = (1/128) sum_d [(1/5) sum_r (1-accuracy[d,r])] ',
    'formula_ensemble': 'L_ensemble5 = (1/128) sum_d (1-accuracy_ensemble5[d])',
    'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
}
print(json.dumps(result, indent=2))
