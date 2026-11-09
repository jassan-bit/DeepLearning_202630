"""Synthetic common-executor contract tests; no market data or historical writes."""
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_common_executor as common


def test_catalogue_is_18_not_20():
    assert len(common.SYSTEM_NAMES) == len(set(common.SYSTEM_NAMES)) == 18
    assert len(common.catalogue.MODEL_NAMES) == 11
    assert len(common.REFERENCE_SPECS) == 3
    assert 'SeasonalNaive' in common.SYSTEM_NAMES


def test_internal_only_public_signature(tmp_path):
    assert set(inspect.signature(common.run_synthetic_common).parameters) == {'output_dir', 'synthetic_only', 'source_roots'}
    with pytest.raises(ValueError, match='synthetic_only'):
        common.run_synthetic_common(tmp_path/'synthetic', synthetic_only=False)
    assert not (tmp_path/'synthetic').exists()
    with pytest.raises(TypeError):
        common.run_synthetic_common(tmp_path/'synthetic', data=np.zeros(3))


def test_exact_calendar_embargo_and_refit():
    p = common._plan()
    assert p['inner']['train'][-1] == p['refit']['origin']-98
    assert len(p['inner']['evaluate']) == 83
    assert p['inner']['evaluate'][-1] == p['refit']['origin']-8
    for s in p.values():
        assert s['train'][-1] == s['origin']-8
        assert len(s['train']) >= 90
        assert np.all(np.diff(s['train']) == 1)


def test_fixture_sigma_population_and_calendar():
    dates, features, y, valid = common._fixture()
    assert str(dates.tz) == 'UTC'
    assert len(dates) == 320 and np.all(np.diff(dates.asi8) == 86400000000000)
    for t in (7, 192, 200, 290):
        actual = np.diff(np.log(features[t-7:t+1,3])).std(ddof=0)
        assert y[t] == pytest.approx(actual, abs=1e-14)
    assert valid.all()


def test_masks_include_all_candidates_without_compression():
    valid = np.ones(320, dtype=bool)
    valid[200-28] = False
    rows = common._mask(valid, np.arange(200, 204))
    assert len(rows) == 4
    assert 'lag_28' in rows[0]['reasons']
    assert not rows[0]['eligible']
    assert [r['index'] for r in rows] == [200,201,202,203]


def test_train_only_scaler_future_invariant():
    _, features, y, _ = common._fixture()
    train = common._plan()['inner']['train']
    scaler, x, target = common._fit_scaler(features, y[train], train)
    perturbed = features.copy()
    perturbed[193:] *= 100
    other, xx, yy = common._fit_scaler(perturbed, y[train].copy(), train)
    assert other == scaler
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(target, yy)
    assert scaler['y_scale'] == pytest.approx(y[train].mean())
    assert scaler['x_fit_window_positions'] == len(train)*7


def test_inference_has_no_target_argument_and_uses_only_past_features():
    assert set(inspect.signature(common._x_at).parameters) == {'features', 'indices', 'scaler'}
    _, features, y, _ = common._fixture()
    train = common._plan()['inner']['train']
    scaler, _, _ = common._fit_scaler(features, y[train], train)
    before = common._x_at(features, [200], scaler)
    features[200:] *= 50
    np.testing.assert_array_equal(before, common._x_at(features, [200], scaler))


def test_scalers_refit_independently():
    _, features, y, _ = common._fixture()
    a,b = [common._fit_scaler(features, y[p['train']], p['train'])[0] for p in common._plan().values()]
    assert a['train_indices_sha256'] != b['train_indices_sha256']
    assert a['y_scale'] != b['y_scale']


def test_e1_architectures_exact_without_old_weights():
    for name, (family,width,dropout,expected) in common.REFERENCE_SPECS.items():
        model = common.e1.build_model(family,7,width,dropout)
        assert sum(p.numel() for p in model.parameters()) == expected
        assert model(torch.zeros(2,7,5)).shape == (2,)


def test_metrics_original_units_coverage_and_mase():
    m = common._metrics([0.,2.], [1.,4.], np.array([1.,2.,4.]))
    assert m['MSE'] == 2.5
    assert m['RMSE'] == pytest.approx(np.sqrt(2.5))
    assert m['MAE'] == 1.5
    assert m['MAPE_coverage'] == .5
    assert m['MAPE_percent'] == 100.
    assert m['MASE'] == 1.


def test_new_journal_hash_chain_and_no_overwrite(tmp_path):
    target = tmp_path/'synthetic-run'
    journal = common.DurableJournal(target)
    a = journal.append('test', value=1)
    b = journal.append('test', value=2)
    journal.close()
    events = [json.loads(line) for line in (target/'events.jsonl').read_text().splitlines()]
    assert events[1]['previous_sha256'] == a
    assert events[1]['event_sha256'] == b
    for event in events:
        digest = event.pop('event_sha256')
        assert common._hash_json(event) == digest
    with pytest.raises(FileExistsError):
        common.DurableJournal(target)


@pytest.mark.parametrize('directory', ['data','results','experiments','references','book_e2'])
def test_protected_output_paths(tmp_path, directory):
    parent = tmp_path/directory
    parent.mkdir()
    with pytest.raises(ValueError, match='market or historical'):
        common.DurableJournal(parent/'synthetic')


def test_symlink_output_rejected(tmp_path):
    actual = tmp_path/'actual'
    actual.mkdir()
    link = tmp_path/'linked'
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        common.DurableJournal(link/'synthetic')


@pytest.fixture(scope='module')
def full_run(tmp_path_factory):
    root = os.environ.get('E2_AUTHOR_SOURCE_ROOT')
    if not root:
        pytest.skip('Full 18-system integration requires E2_AUTHOR_SOURCE_ROOT; not a substitute for full run')
    roots = {n: Path(root)/n for n in ('TimesNet','TimeMixer','PatchTST','iTransformer')}
    target = tmp_path_factory.mktemp('common')/'synthetic-full'
    result = common.run_synthetic_common(target, source_roots=roots)
    return target,result


def test_all_18_both_stages_full_integration(full_run):
    target,r = full_run
    assert r['system_count'] == 18 and r['total_fit_stages'] == 36
    assert r['neural_optimizer_steps'] == 32
    for stage, n in [('inner',83), ('refit',8)]:
        s = r['stages'][stage]
        assert len(s['fit_records']) == 18
        assert len(s['metrics_original_units']) == 18
        assert all(m['n'] == n for m in s['metrics_original_units'].values())
        assert s['fit_records']['ARIMA']['valid_count'] >= 90
        assert s['fit_records'][common.catalogue.current.AE_NAME]['ae_protocol']['transfer_exact']
    assert not r['market_training_executed'] and not r['HPO_executed'] and not r['NAS_executed']
    assert (target/'synthetic_receipt.json').exists()


def test_full_journal_all_predictions_before_reveal_scores(full_run):
    target,r = full_run
    events = [json.loads(line) for line in (target/'events.jsonl').read_text().splitlines()]
    previous = '0'*64
    predictions = {}
    scores = set()
    for i,original in enumerate(events):
        event = dict(original)
        digest = event.pop('event_sha256')
        assert common._hash_json(event) == digest
        assert event['previous_sha256'] == previous and event['sequence'] == i
        previous = digest
        if event['kind'] == 'predictions':
            assert set(event['systems']) == set(common.SYSTEM_NAMES)
            predictions[(event['stage'],event['index'])] = digest
        elif event['kind'] == 'observation':
            assert predictions[(event['stage'],event['index'])] == event['prediction_event_sha256']
            assert events[i-1]['kind'] == 'predictions'
        elif event['kind'] == 'scores':
            scores.add(event['stage'])
    assert len(predictions) == 91
    assert scores == {'inner','refit'}
    assert previous == r['journal_final_sha256']


def test_full_fresh_refit_and_no_historical_metrics(full_run):
    _,r = full_run
    for name in (*common.catalogue.MODEL_NAMES,*common.REFERENCE_SPECS):
        a = r['stages']['inner']['fit_records'][name]
        b = r['stages']['refit']['fit_records'][name]
        for phase in a['phases']:
            assert a['phases'][phase]['fresh_optimizer'] and b['phases'][phase]['fresh_optimizer']
        assert not a['historical_E1_metrics_or_weights_reused']
        assert a['final_state_sha256'] != b['final_state_sha256']


def test_actual_target_reads_require_durable_prediction(tmp_path, monkeypatch):
    root = os.environ.get('E2_AUTHOR_SOURCE_ROOT')
    if not root:
        pytest.skip('Requires pinned sources for real model path')
    fixture = common._fixture()
    active = {'origin': None, 'persisted': set(), 'reads': 0}
    original_append = common.DurableJournal.append
    def append(self, kind, **payload):
        result = original_append(self, kind, **payload)
        if kind == 'stage_plan':
            active['origin'] = payload['evaluation_indices'][0]
            active['persisted'] = set()
        if kind == 'predictions':
            active['persisted'].add(payload['index'])
        return result
    class GuardedLabels(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, (int,np.integer)) and active['origin'] is not None and key >= active['origin']:
                assert int(key) in active['persisted'], 'Evaluation label read before durable prediction'
                active['reads'] += 1
            return super().__getitem__(key)
    guarded = fixture[2].view(GuardedLabels)
    monkeypatch.setattr(common, '_fixture', lambda: (fixture[0],fixture[1],guarded,fixture[3]))
    monkeypatch.setattr(common.DurableJournal, 'append', append)
    roots = {n: Path(root)/n for n in ('TimesNet','TimeMixer','PatchTST','iTransformer')}
    common.run_synthetic_common(tmp_path/'synthetic-target-guard', source_roots=roots)
    assert active['reads'] == 91


def test_failed_forecast_persistence_never_reveals_label(tmp_path, monkeypatch):
    root = os.environ.get('E2_AUTHOR_SOURCE_ROOT')
    if not root:
        pytest.skip('Requires pinned sources for real model path')
    fixture = common._fixture()
    class GuardedLabels(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, (int,np.integer)) and key >= 200:
                raise AssertionError('Outcome must remain hidden after persistence failure')
            return super().__getitem__(key)
    original_append = common.DurableJournal.append
    def append(self, kind, **payload):
        if kind == 'predictions':
            raise OSError('Synthetic injected persistence failure')
        return original_append(self, kind, **payload)
    monkeypatch.setattr(common, '_fixture', lambda: (fixture[0],fixture[1],fixture[2].view(GuardedLabels),fixture[3]))
    monkeypatch.setattr(common.DurableJournal, 'append', append)
    roots = {n: Path(root)/n for n in ('TimesNet','TimeMixer','PatchTST','iTransformer')}
    target = tmp_path/'synthetic-persist-failure'
    with pytest.raises(OSError, match='injected persistence'):
        common.run_synthetic_common(target, source_roots=roots)
    assert not (target/'synthetic_receipt.json').exists()
    events = [json.loads(line) for line in (target/'events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'run_failed'
    assert not any(e['kind'] in ('observation','scores','run_completed') for e in events)


def test_validate_journal_helper_and_tamper(full_run, tmp_path):
    import shutil
    target,_ = full_run
    result = common.validate_journal(target)
    assert result['validated'] and result['forecast_days'] == result['observed_days'] == 91
    copy = tmp_path/'synthetic-tampered'
    shutil.copytree(target, copy)
    path = copy/'events.jsonl'
    lines = path.read_text().splitlines()
    event = json.loads(lines[1])
    event['sequence'] += 1
    lines[1] = json.dumps(event)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='hash chain'):
        common.validate_journal(copy)


@pytest.mark.parametrize('mutation', ['future','duplicate','reversed','float','eval_gap','eval_before_origin'])
def test_plan_rejects_invalid_indices_before_fit(mutation):
    p = common._plan()['inner']
    if mutation == 'future':
        p['train'][1] = 250
    elif mutation == 'duplicate':
        p['train'][1] = p['train'][0]
    elif mutation == 'reversed':
        p['train'] = p['train'][::-1]
    elif mutation == 'float':
        p['train'] = p['train'].astype(float)
    elif mutation == 'eval_gap':
        p['evaluate'] = np.delete(p['evaluate'], 1)
    else:
        p['evaluate'] -= 1
    with pytest.raises(ValueError):
        common._validate_plan(p, 320)
