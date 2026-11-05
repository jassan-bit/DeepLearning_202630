"""Publish the current-trio delivery index; never fit, infer or modify signed inputs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
ID='gru_lstm_tcn_delivery_20261105'
PRIVATE_DEFAULT=Path('/workspace/e1_lstm_gru_tcn_local_20261105')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-experiment',type=Path,default=PRIVATE_DEFAULT);a=p.parse_args();private=a.private_experiment.resolve()
 source=ROOT/'results/gru_lstm_tcn_20261105';s=json.loads((source/'summary.json').read_text());assert s['models']==['GRU','LSTM','TCN']
 dest=ROOT/'results'/ID;dest.mkdir(parents=True,exist_ok=True)
 original=ROOT/'experiments/gru_lstm_tcn_20261105/original_registration';original.mkdir(parents=True,exist_ok=True)
 records={}
 for src,name in [('protocol.json','cv_protocol.json'),('freeze_before_training.json','cv_freeze.json'),('results/code_manifest_before_fitting.json','cv_code_manifest.json'),('final_evaluation/protocol.json','final_protocol.json'),('final_evaluation/freeze.json','final_freeze.json'),('implementation/local_models.py','code/models_original.py'),('runner/run_local.py','code/run_original.py')]:
  sourcefile=private/src;out=original/name;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists() and digest(out)!=digest(sourcefile):raise ValueError('Never overwrite an original signed record: '+name)
  if not out.exists():shutil.copyfile(sourcefile,out)
  records[str(out.relative_to(ROOT))]=digest(out)
 write(original/'README.json',{'scope':'Byte-identical snapshots of the original local experiment, published after execution. Historical comparison plans remain in these source records; the current JBook presents only GRU/LSTM/TCN. Absolute original paths document provenance and are not portable entrypoints.','original_files_sha256':records,'public_entrypoint':'scripts/run_gru_lstm_tcn.py'})
 cv=json.loads((private/'freeze_before_training.json').read_text());ff=json.loads((private/'final_evaluation/freeze.json').read_text());fm=json.loads((private/'final_evaluation/final_models_manifest.json').read_text());ta=json.loads((private/'final_evaluation/test/raw_test_audit.json').read_text())
 declaration={'published_declaration_created_at_utc':datetime.now(timezone.utc).isoformat(),'models':['GRU','LSTM','TCN'],'baseline':s['baseline_selected_by_cv'],'primary_metric':'Annual pooled-date RMSE per trajectory in original sigma7 units; mean/sampleSD across ten seeds','input_days':[7,14,21,28],'seeds':list(range(4101,4111)),'calendar':{'initial_history':'2023','forward_validation':'2024','already_observed_test':'2025-01-01..2026-08-31','unassigned':'September-October2026, not opened'},'randomization_unit':'Training-seed trajectory, paired on shared calendar; not independent markets','new_trio_within_pair_inference':'Post-hoc after scoring the already observed test, Holm3 at D7 and Holm12 sensitivity across four histories','prospective_pilot_for_current_trio':False,'seeds_derived_from_power':False,'seed_budget':10,'future_margin':json.loads((source/'future_margin.json').read_text()),'chronology':{'local_cv_frozen_before_own_fits':cv['frozen_before_new_project_fits_at_utc'],'local_final_configuration_closed_before_own_new_test_inferences':ff['closed_at_utc'],'all_final_weights_frozen':fm['completed_at_utc'],'first_test_derivative_read_in_this_procedure':ta['first_local_derivative_access_at_utc'],'historical_test_already_accessed_in_prior_work':ta.get('historical_first_raw_access_at_utc'),'within_trio_analysis_declared_posttest':s['public_analysis_declared_at_utc'],'original_registration_publicly_available_before_local_fits':False},'original_records_sha256':records,'preserved_limits':['Limited TCN recurrence, no original configuration replica','Two gated recurrent variants require nonredundancy justification','No current-trio prospective pilot/power design','No new independent temporal confirmation','Future practical margin declared after the current test','Public adapter is a post-execution portability adaptation; not its pre-training code hash'],'no_fit_or_checkpoint_inference_by_this_export':True}
 write(dest/'declaration_and_chronology.json',declaration)
 items=[('8.1','Informe Jupyter Book','Disponible','book/intro.md'),('8.2','Código EDA+GRU/LSTM/TCN+estadística','Disponible; adaptador público posterior documentado','scripts/run_gru_lstm_tcn.py'),('8.3','Dataset o instrucciones exactas','Instrucciones y adquisición; brutos no redistribuidos','data/README.md'),('8.4','Literatura y flujo','Disponible; recurrencia TCN parcial','book/sections/03_estado_arte.md'),('8.5','Benchmark e inferencia','Completo; inferencia exploratoria','book/sections/05_benchmark.md'),('8.6','Dependencias y configuraciones','1440 segmentos CV y120modelos finales','results/gru_lstm_tcn_20261105/configuration_catalog_cv.csv'),('8.7','Declaración de métrica/margen/independencia/partición','Disponible; piloto actual pendiente','results/'+ID+'/declaration_and_chronology.json')]
 rows=[dict(zip(['item','product','state','evidence'],row)) for row in items]
 write(dest/'delivery_manifest.json',{'models':s['models'],'products':rows,'availability_is_not_methodological_completion':True})
 with (dest/'delivery_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 (dest/'README.md').write_text('# Entrega actual GRU–LSTM–TCN\n\nLa publicación es posterior a la ejecución local. Disponibilidad de archivos y cierre metodológico se declaran por separado. Los originales firmados están preservados en experiments/gru_lstm_tcn_20261105/original_registration. No se corrigen retroactivamente la cronología, el margen o el piloto.\n')
 hashes={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='public_manifest.json'}
 write(dest/'public_manifest.json',{'files_sha256':hashes,'scope':'Delivery metadata only; no weights, scalers or individual forecasts'})
 target=ROOT/'book/_static'/ID;target.mkdir(parents=True,exist_ok=True)
 for f in dest.iterdir():
  if f.is_file():shutil.copyfile(f,target/f.name)
 print(json.dumps({'delivery_products':len(rows),'snapshot_files':len(records),'models':s['models']}))
if __name__=='__main__':main()
