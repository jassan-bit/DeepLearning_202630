"""Read-only integrity and saved-result reconciliation; never import models.
AI-assisted. This documentary verifier does not certify novelty, reproduction by
an independent person, power sufficiency, public model upload or R3.
"""
from pathlib import Path, PurePosixPath
import argparse,csv,hashlib,json,math,runpy
PREFIX=Path('experiments/e3_completion_20261120')
MIRROR=Path('book_e3/_static/e3_completion_20261120')
ZIP='release/e3-completion-documents-20261120-DRAFT.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(c,m):
 if not c:raise ValueError(m)
def verify(root):
 root=Path(root).resolve();packet=root/PREFIX;mirror=root/MIRROR;m=json.loads((packet/'manifest.json').read_text())
 require(m['new_model_fits']==0 and m['new_calibrations']==0,'Execution scope changed')
 require(m['new_scientific_prediction_rows']==20400 and m['R3_accredited'] is False,'Diagnostic scope changed')
 require(m['new_binary_model_publication_verified'] is False,'No public artifact acceptance may be inferred')
 expected={r['path'] for r in m['files']};extras={'manifest.json','archive_descriptor.json','verification.json',ZIP}
 for base in (packet,mirror):
  found={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
  require(found-extras==expected,'Documentary inventory mismatch: '+str(base))
  for r in m['files']:
   rel=PurePosixPath(r['path']);require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in str(rel),'Unsafe path')
   p=base/str(rel);require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Payload mismatch: '+str(p))
  require((base/'manifest.json').read_bytes()==(packet/'manifest.json').read_bytes(),'Mirror manifest changed')
 old=runpy.run_path(str(root/'experiments/e3_r1_substantive_20261120/verify_packet.py'))['verify'](root)
 decision=packet/'protocol/decision_zero_shot.json'
 require(sha(decision)=='6bda16b7e322e03a1494420f4bedae58a38cad0aed790a8c6493d5d15637df29','Registered decision changed')
 report=json.loads((packet/'reproduction/diagnostic_transfer/diagnostic_receipt.json').read_text())
 require(report['registered_decision_commit']=='6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe','Registration link changed')
 require(report['finite_trajectories']==51 and report['forecast_rows_finite']==20400 and report['failed_trajectories']==0,'Incomplete diagnostic')
 scores={};rows_total=0;trajectories=0
 for p in sorted((packet/'reproduction/diagnostic_transfer').glob('D0*.csv')):
  rows=list(csv.DictReader(p.open()));require(rows,'Empty diagnostic');key=(rows[0]['dataset'],rows[0]['name']);loss=[];abs_loss=[]
  for row in rows:
   y=float(row['target_sigma7']);yp=float(row['prediction_sigma7']);f=float(row['floor_sigma7']);require(all(math.isfinite(x) for x in (y,yp,f)) and yp>=f-1e-10,'Invalid prediction/floor')
   loss.append((y-yp)**2);abs_loss.append(abs(y-yp))
  scores.setdefault(key,[]).append((math.sqrt(math.fsum(loss)/len(loss)),math.fsum(abs_loss)/len(abs_loss)));rows_total+=len(rows);trajectories+=1
 require(rows_total==20400 and trajectories==51,'CSV inventory mismatch')
 for row in csv.DictReader((packet/'reproduction/diagnostic_transfer/summary.csv').open()):
  vals=scores[(row['dataset'],row['name'])];rmse=math.fsum(x[0] for x in vals)/len(vals);mae=math.fsum(x[1] for x in vals)/len(vals)
  require(abs(rmse-float(row['primary_mean_individual_seed_RMSE']))<1e-12 and abs(mae-float(row['mean_individual_seed_MAE']))<1e-12,'Saved metrics differ')
 science=json.loads((packet/'protocol_power/scientific_decision.json').read_text());protocol=json.loads((packet/'protocol_power/protocol.DRAFT.json').read_text())
 require(science['scientific_go_confirmation'] is False and science['original_rule_passed'] is False and science['automatic_promotion_atom_mse_only'] is False,'Pilot no-go changed')
 require(protocol['freeze']['tag'] is None and protocol['scientific_decision']['confirmation_go'] is False,'DRAFT is not a registered confirmation')
 external=json.loads((packet/'external_binary_inventory.json').read_text())
 require(len(external['files'])==17 and not external['new_binary_publication_performed'],'External binary inventory changed')
 return {'status':'documentary_packet_hashes_and_saved_results_verified','payloads':len(expected),'mirrors':len(expected),'historical_R1_packet_preserved':old['status'],'diagnostic_trajectories':trajectories,'diagnostic_rows':rows_total,'new_fits_by_verifier':0,'new_inferences_by_verifier':0,'new_resamples_by_verifier':0,'new_binary_publication_verified':False,'R3_accredited':False,'AI_assistance_declared':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=p.parse_args();print(json.dumps(verify(args.root),sort_keys=True))
