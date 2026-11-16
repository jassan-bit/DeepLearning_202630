"""Portable conditional inference replay: Python + NumPy; no Git, network or fits.

Usage: python replay_inference.py --output fresh_folder [--point-only]
Default input is inference_inputs.npz beside this file. Full replay uses the
registered5000 draws and blocks14/7/28; every output directory must be new.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

COMPONENTS = ("model_fixed_catalogue", "initialization_nested_in_model", "calendar",
              "model_by_calendar", "initialization_by_calendar_remainder")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def moving(rng, n, length):
    starts = rng.integers(0, n - length + 1, size=(n + length - 1) // length)
    return (starts[:, None] + np.arange(length)).reshape(-1)[:n]


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty_like(values)
    result[order] = np.minimum(1, ranked)
    return result


def decompose(loss):
    grand = loss.mean()
    model = loss.mean(axis=(1, 2))
    seed = loss.mean(axis=2)
    calendar = loss.mean(axis=(0, 1))
    model_calendar = loss.mean(axis=1)
    a = model - grand
    b = seed - model[:, None]
    c = calendar - grand
    d = model_calendar - model[:, None] - calendar[None, :] + grand
    e = loss - seed[:, :, None] - model_calendar[:, None, :] + model[:, None, None]
    components = np.asarray([np.mean(a*a), np.mean(b*b), np.mean(c*c),
                             np.mean(d*d), np.mean(e*e)])
    total = float(np.mean((loss-grand)**2))
    require(total > 0 and np.isfinite(components).all(), "Degenerate/nonfinite loss decomposition")
    relative_error = abs(components.sum()-total) / total
    require(relative_error < 1e-10 and (components >= 0).all(), "Orthogonal projection QA failed")
    return components, components/total, total, float(relative_error)


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(source, output, point_only=False):
    started = time.monotonic()
    require(not output.exists(), "Use a fresh output directory")
    descriptor = json.loads(source.with_name('inference_inputs_manifest.json').read_text())
    require(hashlib.sha256(source.read_bytes()).hexdigest() == descriptor['sha256'], 'Input checksum mismatch')
    with np.load(source, allow_pickle=False) as z:
        models = z['models'].tolist()
        require(len(models) == 13 and z['test_prediction_original'].shape == (13,6,608)
                and z['OOF_prediction_original'].shape == (13,3,83), 'Wrong13/6/608/249 supports')
        require(z['test_mask'].tolist() == [True]*608, 'Incomplete test mask')
        test_y = z['test_y_original']
        oof = (z['OOF_prediction_original']-z['OOF_y_original'][None,:,:])**2
        loss = (z['test_prediction_original']-test_y[None,None,:])**2
    require(np.isfinite(oof).all() and np.isfinite(loss).all(), 'Nonfinite input')
    oof_rmse = np.sqrt(oof.mean(axis=(1,2)))
    test_rmse = np.sqrt(loss.mean(axis=2)).mean(axis=1)
    observed = test_rmse-oof_rmse
    components, fractions, total, error = decompose(loss)
    output.mkdir(parents=True)
    write_csv(output/'point_metrics.csv', [{'model':m,'OOF_RMSE':float(oof_rmse[i]),
              'test_mean_individual_RMSE':float(test_rmse[i]),'test_minus_OOF':float(observed[i])}
              for i,m in enumerate(models)])
    write_csv(output/'point_components.csv', [{'component':name,'variance':float(components[i]),
              'fraction':float(fractions[i]),'total_loss_variance':total} for i,name in enumerate(COMPONENTS)])
    rows, variance_rows, max_error = [], [], error
    if not point_only:
        for length in [14,7,28]:
            rng = np.random.default_rng(202611161501+length)
            differences = np.empty((5000,13))
            vc, vf = np.empty((5000,5)), np.empty((5000,5))
            for b in range(5000):
                oof_indices = np.stack([moving(rng,83,length) for _ in range(3)])
                sampled_oof = np.take_along_axis(oof,oof_indices[None,:,:],axis=2)
                test_indices = moving(rng,608,length)
                seed_indices = rng.integers(0,6,size=(13,6))
                sampled_loss = loss[np.arange(13)[:,None],seed_indices][:,:,test_indices]
                differences[b] = np.sqrt(sampled_loss.mean(axis=2)).mean(axis=1)-np.sqrt(sampled_oof.mean(axis=(1,2)))
                vc[b],vf[b],_,err = decompose(sampled_loss)
                max_error = max(max_error,err)
            raw_p = (1+(np.abs(differences-observed[None,:])>=np.abs(observed[None,:])).sum(axis=0))/5001
            adjusted = holm(raw_p) if length==14 else [None]*13
            for i,model in enumerate(models):
                ci = np.quantile(differences[:,i],[.025,.975])
                rows.append({'model':model,'block_days':length,'OOF_pooled249_RMSE':float(oof_rmse[i]),
                    'original_test608_mean_individual_RMSE':float(test_rmse[i]),
                    'test_minus_OOF_RMSE':float(observed[i]),'CI95_lower':float(ci[0]),'CI95_upper':float(ci[1]),
                    'p_two_sided_raw':float(raw_p[i]),'p_Holm13':None if adjusted[i] is None else float(adjusted[i]),
                    'primary_family':length==14,
                    'equality_rejected_primary':bool(adjusted[i]<=.05) if length==14 else None,
                    'equivalence_established':False,'same_predictor_across_periods':False,'new_fits':0})
            for i,name in enumerate(COMPONENTS):
                ci,fci=np.quantile(vc[:,i],[.025,.975]),np.quantile(vf[:,i],[.025,.975])
                variance_rows.append({'component':name,'block_days':length,
                    'finite_variance_component':float(components[i]),'component_CI95_lower':float(ci[0]),
                    'component_CI95_upper':float(ci[1]),'fraction_of_total':float(fractions[i]),
                    'fraction_CI95_lower':float(fci[0]),'fraction_CI95_upper':float(fci[1]),
                    'total_loss_variance':total,'population_random_effect_component':False})
            print(json.dumps({'completed_block_days':length,'elapsed_seconds':time.monotonic()-started}),flush=True)
        write_csv(output/'conditional_OOF_test_comparisons_39.csv',rows)
        write_csv(output/'finite_loss_variance_components_15.csv',variance_rows)
    result={'full_bootstrap':not point_only,'draws_per_block':0 if point_only else 5000,
        'max_decomposition_relative_identity_error':max_error,'elapsed_seconds':time.monotonic()-started,
        'new_fits':0,'new_model_forwards':0,'prospective_confirmation':False,
        'population_random_effect_fit':False,'HPO_selection_replayed':False,
        'scope':'Conditional known-chronology inference; OOF/test use different training histories; nonrejection is not equivalence.'}
    with (output/'replay_receipt.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).resolve().with_name('inference_inputs.npz'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--point-only',action='store_true')
    arguments=parser.parse_args()
    print(json.dumps(replay(arguments.inputs,arguments.output,arguments.point_only)))
