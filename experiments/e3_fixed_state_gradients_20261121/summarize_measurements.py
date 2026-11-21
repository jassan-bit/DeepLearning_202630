"""Read completed measurement JSON only; no model, gradients, RNG or p-values."""
import argparse
import csv
import json
from pathlib import Path
import statistics

def summarize(rows):
    result=[]
    for recipe in ['sigma_crps_v4','u_crps_control','mse_only_control']:
        for snapshot in ['best','latest']:
            for part in ['all','shared','head']:
                rr=[x for x in rows if x['recipe']==recipe and x['snapshot']==snapshot and x['parameter_partition']==part]
                assert len(rr)==25
                cc=[x['cosine_unweighted'] for x in rr if x['cosine_unweighted'] is not None]
                ratios=[x['aux_effective_gradient_norm']/x['MSE_gradient_norm'] for x in rr if x['MSE_gradient_norm']>0]
                lam=rr[0]['lambda_actual']
                out={'recipe':recipe,'snapshot':snapshot,'partition':part,'n_labeled_batch_states':len(rr),'lambda_actual':lam,
                    'unweighted_aux_geometry_is_counterfactual':lam==0,
                    'actual_aux_negative_cosine_count':sum(x['cosine_actual_auxiliary'] is not None and x['cosine_actual_auxiliary']<0 for x in rr) if lam>0 else None,
                    'unweighted_cosine_negative_count':sum(x<0 for x in cc),'unweighted_cosine_median':statistics.median(cc),
                    'unweighted_cosine_min':min(cc),'unweighted_cosine_max':max(cc),
                    'effective_aux_norm_ratio_median':statistics.median(ratios),'effective_aux_norm_ratio_min':min(ratios),'effective_aux_norm_ratio_max':max(ratios)}
                if snapshot=='latest':
                    out.update(full_first_order_ascent_count=sum(x['MSE_dot_Adam_full_displacement']>0 for x in rr),
                        MSE_counterfactual_first_order_ascent_count=sum(x['MSE_dot_Adam_MSE_counterfactual_displacement']>0 for x in rr),
                        full_and_MSE_both_descent_count=sum(x['MSE_dot_Adam_full_displacement']<0 and x['MSE_dot_Adam_MSE_counterfactual_displacement']<0 for x in rr),
                        full_and_MSE_both_ascent_count=sum(x['MSE_dot_Adam_full_displacement']>0 and x['MSE_dot_Adam_MSE_counterfactual_displacement']>0 for x in rr),
                        full_descent_MSE_ascent_count=sum(x['MSE_dot_Adam_full_displacement']<0 and x['MSE_dot_Adam_MSE_counterfactual_displacement']>0 for x in rr),
                        full_ascent_MSE_descent_count=sum(x['MSE_dot_Adam_full_displacement']>0 and x['MSE_dot_Adam_MSE_counterfactual_displacement']<0 for x in rr),
                        full_slope_worse_than_counterfactual_count=sum(x['MSE_dot_displacement_difference']>0 for x in rr),
                        full_slope_median=statistics.median(x['MSE_dot_Adam_full_displacement'] for x in rr),
                        MSE_counterfactual_slope_median=statistics.median(x['MSE_dot_Adam_MSE_counterfactual_displacement'] for x in rr),
                        full_clip_min=min(x['Adam_full_clip'] for x in rr),full_clip_max=max(x['Adam_full_clip'] for x in rr))
                result.append(out)
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--case',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    receipt=json.loads((args.case/'EXECUTION_RECEIPT.json').read_text());assert receipt['status']=='completed'
    data=json.loads((args.case/'MEASUREMENTS.json').read_text());rows=data['rows'];assert len(rows)==450
    keys={(r['job_id'],r['snapshot'],r['batch'],r['parameter_partition']) for r in rows};assert len(keys)==450
    summary=summarize(rows);args.out.mkdir(parents=True,exist_ok=True)
    out={'status':'completed_descriptive_summary','scope':'Post-measurement deterministic summaries; strict floating-point signs, no inferential threshold or p-values',
         'groups':summary,'primary_unit':'Labeled fixed checkpoint/batch, not an independent replicate',
         'row_count':450,'unique_state_batches':150,'new_model_calls_in_this_summarizer':0,'R3_accredited':False}
    (args.out/'SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    cols=list(dict.fromkeys(k for r in summary for k in r))
    with (args.out/'summary_by_group.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(summary)
    print(json.dumps({'groups':len(summary),'raw_rows':len(rows),'new_model_calls':0}))

if __name__=='__main__':main()
