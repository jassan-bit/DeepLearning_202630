# Copy this script and its inputs to a private directory outside any Git checkout.
import pathlib as _guard_pathlib
if any((p / '.git').exists() for p in _guard_pathlib.Path(__file__).resolve().parents):
    raise SystemExit('Run only from a private directory outside Git; full abstracts must not enter a checkout.')
import pathlib,csv,json,collections,hashlib
P=pathlib.Path(__file__).parent; D=P/'derivados_sin_resumenes';D.mkdir(exist_ok=True)
fields=['study_id','title','year','years','doi','dois','record_ids','databases','duplicate_count','document_type','period_status','initial_class','screening_class','reason','targeted_audit','full_text_inclusion']
R=list(csv.DictReader((P/'cribado_unicos.csv').open(encoding='utf-8-sig')))
with (D/'cribado_metadatos.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in R)
for n in ['resumen.json','relaciones_duplicados.csv','conflictos_duplicados.csv','ajustes_revision.json','cribar.py','crear_derivados_publicos.py']:
 (D/n).write_bytes((P/n).read_bytes())
(D/'manifest.json').write_text(json.dumps({p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in D.iterdir() if p.name!='manifest.json'},indent=2))
