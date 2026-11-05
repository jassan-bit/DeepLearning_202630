"""Publish explicit aggregates from the registered conditional D7 study only."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
NAME='e1_filipe_confirmatory_d7_20261105'
SOURCE=ROOT/'results'/NAME
DEST=ROOT/'book/_static'/NAME


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rows(path):
    return list(csv.DictReader(Path(path).open(newline='')))


def number(value, digits=7):
    return format(float(value),f'.{digits}g') if value not in ('',None) else 'No definido'


def markdown(path, headers, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    body=['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
    body.extend('| '+' | '.join(str(x) for x in row)+' |' for row in data)
    path.write_text('\n'.join(body)+'\n')


def main():
    plan=read(SOURCE/'protocol.json'); lock=read(SOURCE/'locked_design.json')
    analysis=read(SOURCE/'analysis/summary.json')
    n=lock['common_R']
    phases={phase:read(SOURCE/phase/'execution_summary.json')
            for phase in ('variance_extension','definitive')}
    if phases['variance_extension']['trajectories']!=60 or phases['definitive']['trajectories']!=3*n:
        raise ValueError('Only export the completed registered phases')
    verification=ROOT/'references/audits/revision_filipe_confirmatory_d7_20261105.json'
    if read(verification).get('status')!='PASS':
        raise ValueError('Independent verification is required before publication')
    DEST.mkdir(parents=True,exist_ok=True)
    root_files=['protocol.json','runtime.json','code_manifest_before_fitting.json',
                'preregistration_git_receipt.json','locked_design.json','locked_design_publication_receipt.json']
    for name in root_files:
        shutil.copyfile(SOURCE/name,DEST/name)
    for phase in phases:
        for name in ['runs.csv','fold_metrics.csv','execution_summary.json','run_started.json']:
            target=DEST/phase/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/phase/name,target)
    # Names are explicit: no recursive copying of experiment output.
    for folder,names in {
        'planning':['paired_dimensioning.csv','variance_summary.json','pilot_metrics_summary.csv','README.md','qq_new20.png'],
        'analysis':['metrics_summary.csv','rmse_summary.csv','metrics.csv','paired_comparisons.csv',
                    'summary.json','qq_differences.png','metrics.png','README.md']}.items():
        for name in names:
            path=SOURCE/folder/name
            if path.is_file():
                target=DEST/folder/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
    (DEST/'verification').mkdir(exist_ok=True)
    shutil.copyfile(verification,DEST/'verification'/verification.name)
    report=verification.with_suffix('.md')
    if report.exists():shutil.copyfile(report,DEST/'verification'/report.name)
    catalog=[]
    for phase in phases:
        for item in rows(SOURCE/phase/'runs.csv'):
            run=SOURCE/phase/'runs'/item['run_id']
            for path in sorted(run.glob('*/configuration.json')):
                config=read(path)
                catalog.append({'phase':phase,'model':item['model'],'history_days':7,
                    'trajectory_seed':int(item['seed']),'fold_month':path.parent.name,
                    'width':config['width'],'learning_rate':config['learning_rate'],
                    'selected_epochs':config['best_epoch'],
                    'inner_initialization_seed':config['inner_initialization_seed'],
                    'refit_initialization_seed':config['refit_initialization_seed'],
                    'configuration_sha256':sha(path),'parameters':int(item['parameters'])})
    if len(catalog)!=(60+3*n)*12:
        raise ValueError('Missing registered configuration metadata')
    with (DEST/'configuration_catalog.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(catalog[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(catalog)
    metrics_file=SOURCE/'analysis/rmse_summary.csv'
    if not metrics_file.exists():metrics_file=SOURCE/'analysis/metrics.csv'
    metrics=rows(metrics_file)
    markdown(DEST/'tables/metrics.md',['Modelo D7','Semillas definitivas','RMSE, media ± DE','IC marginal 95 %'],
        [[r['model'],r.get('n_seeds',n),number(r['rmse_mean'])+' ± '+number(r['rmse_sample_sd']),
          '['+number(r['rmse_ci95_lower'])+'; '+number(r['rmse_ci95_upper'])+']'] for r in metrics])
    pairs=rows(SOURCE/'analysis/paired_comparisons.csv')
    markdown(DEST/'tables/comparisons.md',['Par A−B','Δ RMSE','IC familiar','p sin ajustar','p Holm-3'],
        [[r['model_a']+' − '+r['model_b'],number(r['mean_difference_A_minus_B']),
          '['+number(r['mean_difference_family_ci_lower'])+'; '+number(r['mean_difference_family_ci_upper'])+']',
          number(r['p_value']),number(r['p_holm3'])] for r in pairs])
    events=[('Registro del protocolo',plan['registered_at_utc']),
        ('Protocolo publicado',read(SOURCE/'preregistration_git_receipt.json')['published_at_utc']),
        ('Primera trayectoria de ampliación',phases['variance_extension']['first_new_trajectory_started_at_utc']),
        ('Ampliación completada',phases['variance_extension']['completed_at_utc']),
        ('R definitivo fijado',lock['locked_at_utc']),
        ('R publicado',read(SOURCE/'locked_design_publication_receipt.json')['published_at_utc']),
        ('Primera trayectoria definitiva',phases['definitive']['first_new_trajectory_started_at_utc']),
        ('Definitiva completada',phases['definitive']['completed_at_utc'])]
    markdown(DEST/'tables/chronology.md',['Evento','Fecha UTC'],events)
    write(DEST/'execution_overview.json',{'study_id':NAME,'scope':plan['scope'],
        'variance_extension':{'seeds_per_model':20,'trajectories':60,'fits':1440},
        'definitive':{'seeds_per_model':n,'trajectories':3*n,'fits':72*n},
        'configuration_records':len(catalog),'primary_pairs':3,
        'new_temporal_evaluation':False,'old_benchmark_repaired':False,
        'original_test_reopened':False,'achieved_power_claim':False,
        'analysis_summary_sha256':sha(SOURCE/'analysis/summary.json')})
    (DEST/'README.md').write_text(
        '# Estudio D7 registrado de aleatoriedad del entrenamiento\n\n'
        'Tres recetas fijas y calendario2024 conocido. La ampliación de20 semillas por modelo '
        'estima la varianza; no se incorpora a las semillas definitivas. R se fijó y publicó '
        'antes de las corridas definitivas. Los tres contrastes primarios se fijaron antes '
        'de ambas fases. Se publican agregados y metadatos; pesos, escaladores, curvas '
        'completas y predicciones individuales permanecen privados.\n\n'
        'Esta evidencia es prospectiva respecto de las nuevas aleatorizaciones, no otra '
        'trayectoria del mercado. No modifica los registros históricos ni acredita '
        'potencia realmente alcanzada o una mejora avanzada superior al5%.\n')
    files={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*'))
           if p.is_file() and p.name!='public_manifest.json'}
    write(DEST/'public_manifest.json',{'exported_at_utc':datetime.now(timezone.utc).isoformat(),
        'study_id':NAME,'scope':'Explicit aggregate allowlist; no forecasts, scalers or checkpoints',
        'files_sha256':files,'export_script_sha256':sha(__file__)})
    print(json.dumps({'public_files':len(files)+1,'configuration_records':len(catalog),
                      'new_fits_by_exporter':0,'new_forecasts_by_exporter':0}))


if __name__=='__main__':main()
