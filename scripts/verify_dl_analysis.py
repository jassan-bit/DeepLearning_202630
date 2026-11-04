"""Independent numerical audit of stored development outputs, no raw data access."""
from pathlib import Path
from itertools import combinations
import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds, acf

root=Path(__file__).resolve().parents[1]
p=root/'results/dl_benchmark_20261104'
out=p/'analysis'
models=['MLP','LSTM','CNN1D','Persistencia']
f=pd.read_csv(p/'predictions/development_benchmark.csv')
r=pd.read_csv(p/'runs.csv'); r=r[r.phase=='development_benchmark']
assert f.phase.eq('development_benchmark').all()
assert f.outcome_day.min()=='2024-07-01' and f.outcome_day.max()=='2024-12-31'
assert len(f)==3*30*184 and not f.duplicated(['model','seed','outcome_day']).any()
assert f.history_days.eq(7).all() and sorted(f.seed.unique())==list(range(1001,1031))
seeds=sorted(f.seed.unique()); dates=sorted(f.outcome_day.unique())
assert len(dates)==184 and (pd.to_datetime(pd.Series(dates)).diff().dropna()==pd.Timedelta(days=1)).all()
y=f[(f.model=='MLP') & (f.seed==1001)].sort_values('outcome_day').y_true.to_numpy()
pred=np.empty((4,30,184))
for i,m in enumerate(models[:3]):
 for j,s in enumerate(seeds):
  g=f[(f.model==m) & (f.seed==s)].sort_values('outcome_day')
  np.testing.assert_array_equal(g.y_true,y)
  pred[i,j]=g.y_pred
  assert (pd.to_datetime(g.input_last_day)<pd.to_datetime(g.outcome_day)).all()
  assert (pd.to_datetime(g.target_available_at_utc,utc=True)>pd.to_datetime(g.prediction_origin_utc,utc=True)).all()
persist=f[(f.model=='MLP') & (f.seed==1001)].sort_values('outcome_day').persistence_pred.to_numpy()
assert np.allclose(persist[1:],y[:-1],rtol=0,atol=1e-14)
pred[3]=persist
loss=(pred-y)**2
rmse=np.sqrt(loss.mean(-1))
summary=pd.read_csv(out/'metrics_summary.csv')
scale=r.mase_train_scale.iloc[0]
metric_values={'rmse':rmse,'mse':loss.mean(-1),'mae':np.abs(pred-y).mean(-1),
 'r2':1-loss.sum(-1)/((y-y.mean())**2).sum(),
 'mape':100*np.abs((pred-y)/y).mean(-1),
 'mase':np.abs(pred-y).mean(-1)/scale}
max_metric_error=0.
for i,m in enumerate(models):
 for metric,vals in metric_values.items():
  row=summary[(summary.model==m)&(summary.metric==metric)].iloc[0]
  vals=vals[i] if m!='Persistencia' else vals[i,:1]
  np.testing.assert_allclose(row['mean'],vals.mean(),rtol=1e-12,atol=1e-14)
  max_metric_error=max(max_metric_error,abs(row['mean']-vals.mean()))
  if m!='Persistencia':
   np.testing.assert_allclose(row.sample_sd,vals.std(ddof=1),rtol=1e-12,atol=1e-14)
   lower,upper=stats.t.interval(.95,29,loc=vals.mean(),scale=stats.sem(vals))
   np.testing.assert_allclose([row.ci95_lower,row.ci95_upper],[lower,upper],rtol=1e-12,atol=1e-14)
   rows=r[r.model==m].sort_values('seed')
   np.testing.assert_allclose(rows[metric],vals,rtol=1e-12,atol=1e-14)
  else: assert pd.isna(row.sample_sd) and pd.isna(row.ci95_lower) and row.n_total==1

contrasts=[(rmse[a]-rmse[b],0.,'two-sided') for a,b in combinations(range(3),2)]
contrasts += [(rmse[a]-rmse[3],0.,'two-sided') for a in range(3)]
contrasts += [(1-rmse[a]/rmse[3],.05,'greater') for a in range(3)]
t=pd.read_csv(out/'paired_tests.csv'); expected=[]
assert len(t)==9 and t.family_size_planned.eq(9).all()
for row,(d,null,alt) in zip(t.itertuples(),contrasts):
 test=stats.ttest_1samp(d,popmean=null,alternative=alt)
 expected.append(test.pvalue)
 np.testing.assert_allclose([row.t_statistic,row.p_value],[test.statistic,test.pvalue],rtol=1e-10,atol=1e-28)
 np.testing.assert_allclose(row.paired_cohen_d_z,d.mean()/d.std(ddof=1),rtol=1e-12)
 np.testing.assert_allclose(row.shapiro_p,stats.shapiro(d).pvalue,rtol=1e-9,atol=1e-13)
 np.testing.assert_allclose([row.ci95_lower,row.ci95_upper],stats.t.interval(.95,29,loc=d.mean(),scale=stats.sem(d)),rtol=1e-12,atol=1e-14)
np.testing.assert_allclose(t.p_holm,multipletests(expected,method='holm')[1],rtol=1e-10,atol=1e-28)

temporal=pd.read_csv(out/'temporal_dm.csv'); expected_primary=[]
for row in temporal.itertuples():
 a,b=models.index(row.model_a),models.index(row.model_b)
 d=loss[a].mean(0)-loss[b].mean(0); centered=d-d.mean(); n=len(d); lag=row.HAC_lag
 gamma=np.array([np.dot(centered[k:],centered[:n-k])/n for k in range(lag+1)])
 omega=gamma[0]+2*np.dot(1-np.arange(1,lag+1)/(lag+1),gamma[1:])
 dm=np.sqrt(n)*d.mean()/np.sqrt(omega)
 hln=dm*np.sqrt((n-1)/n)
 pv=2*stats.t.sf(abs(hln),n-1)
 np.testing.assert_allclose([row.mean_squared_loss_difference,row.long_run_variance,row.DM_HLN,row.p_value],[d.mean(),omega,hln,pv],rtol=1e-10,atol=1e-16)
 if lag==14: expected_primary.append(pv)
np.testing.assert_allclose(temporal[temporal.HAC_lag==14].p_holm,multipletests(expected_primary,method='holm')[1],rtol=1e-10,atol=1e-16)
assert temporal[temporal.HAC_lag!=14].p_holm.isna().all()

diagnostics=pd.read_csv(out/'diagnostics.csv'); diag_p=[]
for m in models[:3]:
 resid=y-pred[models.index(m),0]
 lb=acorr_ljungbox(resid,lags=[7,14,28],model_df=0,return_df=True)
 bs,bp=bds(resid,max_dim=3,epsilon=1.5*np.std(resid,ddof=1))
 ac=acf(resid,nlags=28,fft=False)
 ac_out=pd.read_csv(out/'acf.csv')
 np.testing.assert_allclose(ac_out[ac_out.model==m].acf,ac,rtol=1e-12,atol=1e-14)
 for row in diagnostics[diagnostics.model==m].itertuples():
  if row.test=='Ljung-Box':
   stat,pv=lb.loc[int(row.lag),['lb_stat','lb_pvalue']]
  else:
   i=int(row.dimension)-2; stat,pv=bs[i],bp[i]
   np.testing.assert_allclose(row.epsilon,1.5*resid.std(ddof=1),rtol=1e-12)
  np.testing.assert_allclose([row.statistic,row.p_value],[stat,pv],rtol=1e-10,atol=1e-200)
  diag_p.append(pv)
np.testing.assert_allclose(diagnostics.p_holm,multipletests(diag_p,method='holm')[1],rtol=1e-10,atol=1e-200)

boot=pd.read_csv(out/'block_bootstrap.csv'); rank=pd.read_csv(out/'ranking_bootstrap.csv')
for L in [14,7,28]:
 rng=np.random.default_rng(20261104); b=2000; n=184
 starts=rng.integers(0,n,size=(b,int(np.ceil(n/L))))
 indices=((starts[:,:,None]+np.arange(L))%n).reshape(b,-1)[:,:n]
 # Independent loop implementation makes seed and date axes explicit.
 scores=np.stack([np.sqrt(loss[:,:,ix].mean(-1)).mean(1) for ix in indices])
 for row in boot[boot.block_length==L].itertuples():
  a,bidx=models.index(row.model_a),models.index(row.model_b)
  diff=scores[:,a]-scores[:,bidx]; rel=1-scores[:,a]/scores[:,bidx]
  np.testing.assert_allclose([row.ci95_difference_lower,row.ci95_difference_upper],np.quantile(diff,[.025,.975]),rtol=1e-10,atol=1e-14)
  np.testing.assert_allclose([row.ci95_relative_lower,row.ci95_relative_upper],np.quantile(rel,[.025,.975]),rtol=1e-10,atol=1e-14)
 for k in [3,4]:
  winner=scores[:,:k]==scores[:,:k].min(1,keepdims=True)
  frequencies=(winner/winner.sum(1,keepdims=True)).mean(0)
  comp='three_DL_and_persistence' if k==4 else 'three_DL'
  for i,m in enumerate(models[:k]):
   actual=rank[(rank.block_length==L)&(rank.comparison_set==comp)&(rank.model==m)].bootstrap_lowest_error_frequency.iloc[0]
   np.testing.assert_allclose(actual,frequencies[i],rtol=1e-12,atol=1e-14)

meta=json.loads((out/'summary.json').read_text())
for source in meta['sources']:
 recorded=Path(source['path'])
 relative=recorded.parts[recorded.parts.index('results'):]
 assert hashlib.sha256(root.joinpath(*relative).read_bytes()).hexdigest()==source['sha256']
assert hashlib.sha256((root/'scripts/analyze_dl_benchmark.py').read_bytes()).hexdigest()==meta['analysis_script_sha256']
np.testing.assert_allclose(meta['friedman']['statistic'],stats.friedmanchisquare(*rmse[:3]).statistic,rtol=1e-12)

print(json.dumps({'audit':'passed','prediction_rows':len(f),'runs':len(r),'dates':len(dates),'seeds':len(seeds),
 'recomputed_metrics':len(metric_values)*4,'max_absolute_metric_error':max_metric_error,
 'paired_tests_checked':9,'temporal_DM_checked':18,'diagnostics_checked':15,
 'bootstrap_intervals_checked':18,'bootstrap_ranking_rows_checked':21,
 'bootstrap_draws_per_length':2000,'source_hashes_match':True,
 'reserved_raw_files_opened':0},indent=2))
