"""Independent saved-output arithmetic review; no bootstrap, fits or network."""
import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
PIN='beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    assert sha(args.source_zip)==PIN
    with zipfile.ZipFile(args.source_zip) as archive:
        rows=list(csv.DictReader(io.StringIO(archive.read('predictions.csv').decode())))
    groups=defaultdict(list)
    for row in rows:
        groups[row['recipe'],row['seed']].append((float(row['prediction'])-float(row['target']))**2)
    means=defaultdict(list)
    for (recipe,seed),values in groups.items():
        means[recipe].append(math.sqrt(math.fsum(values)/len(values)))
    risks={key:math.fsum(values)/len(values) for key,values in means.items()}
    difference=risks['input_only_mse']-risks['global_mse']
    summary=json.loads((args.retrospective/'results/summary.json').read_text())
    comparisons=[r for r in summary['temporal_intervals'] if r['control']=='global_mse']
    assert len(rows)==1760 and len(groups)==32 and len(comparisons)==2
    for interval in comparisons:
        assert math.isclose(interval['delta_mean_individual_RMSE'],difference,rel_tol=1e-12,abs_tol=1e-15)
        assert interval['CI95_pointwise_low']<=0<=interval['CI95_pointwise_high']
    diagnostics=list(csv.DictReader((args.retrospective/'results/diagnostics_Holm128.csv').open()))
    assert len(diagnostics)==128
    reference=multipletests([float(r['p_value']) for r in diagnostics],alpha=.05,method='holm')[1]
    assert np.allclose(reference,[float(r['p_Holm128']) for r in diagnostics],rtol=0,atol=1e-15)
    rejections=[r for r,p in zip(diagnostics,reference) if p<.05]
    candidate=[r for r in rejections if r['recipe']=='input_only_mse']
    assert len(rejections)==27 and len(candidate)==2 and all(r['test']=='MZ_HAC14' for r in candidate)
    receipt=dict(status='INDEPENDENT_ARITHMETIC_AND_STATISTICAL_CODE_REVIEW_PASS',
        reviewer='AI guide/statistical auditor separate from analysis implementation',reviewed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_zip_sha256=sha(args.source_zip),review_code_sha256=sha(__file__),
        analysis_code_sha256=sha(args.retrospective/'audit_saved_adequacy.py'),
        analysis_plan_sha256=sha(args.retrospective/'analysis_plan.json'),analysis_summary_sha256=sha(args.retrospective/'results/summary.json'),
        independent_checks=dict(ZIP_rows=len(rows),trajectories=len(groups),mean_individual_RMSE_candidate_minus_global=difference,
            point_estimates_match_all_two_block_summaries=True,both_global_RMSE_intervals_include_zero=True,
            Holm128_matches_statsmodels=True,Holm_rejections=len(rejections),candidate_MZ_HAC14_rejections=len(candidate)),
        code_review=['Shared date indices over R*B*T preserve pairing; root per seed precedes averaging',
            'Block7/14 are sensitivity of one circular-block method, not distinct confirmatory procedures',
            'MZ joint null intercept0/slope1 uses common-unit scaling and HAC14',
            'All128 diagnostic positions and all7 contrasts retained; pointwise IC are not simultaneous or finite-sample validated'],
        limitations=['55 exposed dates,8 recipes,5 fixed seeds; not four-panel21-system planning',
            'No complete fit/selection point predictions for TablesA/B or predictive quantiles/law for coverage/PIT',
            'Arithmetic/code review does not validate confidence coverage or market adequacy'],
        new_model_fits=0,new_model_forwards=0,new_market_downloads=0,new_stochastic_draws=0,R3_accredited=False,AI_assistance_declared=True)
    args.output.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(receipt['independent_checks'],ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-zip',type=Path,default=Path('/workspace/e3_input_only_candidate_20261121/inputs/public_results.zip'))
    parser.add_argument('--retrospective',type=Path,default=HERE.parent/'retrospective')
    parser.add_argument('--output',type=Path,default=HERE/'retrospective_independent_review.json')
    run(parser.parse_args())
