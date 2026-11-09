# Copy this script and its inputs to a private directory outside any Git checkout.
import pathlib as _guard_pathlib
if any((p / '.git').exists() for p in _guard_pathlib.Path(__file__).resolve().parents):
    raise SystemExit('Run only from a private directory outside Git; full abstracts must not enter a checkout.')
import csv,json,re,hashlib,unicodedata,collections,pathlib
P=pathlib.Path(__file__).parent; O=P/'originales'; R=[]
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s.lower()).encode('ascii','ignore').decode())
def doi(s):return re.sub(r'^(https?://(dx\.)?doi\.org/|doi:\s*)','',s.strip().lower()).rstrip(' .')
def add(src,i,d):
 d.update(record_id=f'{src}-{i:04}',database=src,source_row=i,doi=doi(d.get('doi','')));d['title_normalized']=norm(d['title']);R.append(d)
w=[];cur={};key=None
for line in (O/'wos.txt').read_text(encoding='utf-8-sig').splitlines():
 if line=='ER':w.append(cur);cur={};key=None
 elif len(line)>2 and line[:2].strip():key=line[:2];cur[key]=line[3:]
 elif line.startswith('   ') and key:cur[key]+=' '+line.strip()
for i,r in enumerate(w,1):add('WoS',i,dict(title=r.get('TI',''),abstract=r.get('AB',''),year=r.get('PY',''),doi=r.get('DI',''),native_id=r.get('UT',''),authors=r.get('AF',''),keywords=r.get('DE','')+'; '+r.get('ID',''),document_type=r.get('DT',''),source_title=r.get('SO',''),url='https://doi.org/'+r['DI'] if r.get('DI') else ''))
for s in ['scopus','ieee']:
 for i,r in enumerate(csv.DictReader((O/f'{s}.csv').open(encoding='utf-8-sig')),1):
  add(s.capitalize() if s=='scopus' else 'IEEE',i,dict(title=r.get('Title',r.get('Document Title','')),abstract=r.get('Abstract',''),year=r.get('Year',r.get('Publication Year','')),doi=r.get('DOI',''),native_id=r.get('EID',r.get('Document Identifier','')),authors=r.get('Authors',''),keywords=r.get('Author Keywords','')+'; '+r.get('Index Keywords',r.get('IEEE Terms','')),document_type=r.get('Document Type',''),source_title=r.get('Source title',r.get('Publication Title','')),url=r.get('Link',r.get('PDF Link',''))))
parent=list(range(len(R)))
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):parent[root(j)]=root(i)
D={};T={};edges=[];conflicts=[]
for i,r in enumerate(R):
 if r['doi']:
  if r['doi'] in D:union(i,D[r['doi']]);edges.append((R[D[r['doi']]]['record_id'],r['record_id'],'DOI exacto normalizado'))
  else:D[r['doi']]=i
 k=(r['title_normalized'],r['year'])
 if len(k[0])>20:
  if k in T:
   a=R[T[k]]
   if a['doi'] and r['doi'] and a['doi']!=r['doi']:conflicts.append([a['record_id'],r['record_id'],'Título/año idéntico pero DOI distintos: no fusionados por título'])
   else:union(i,T[k]);edges.append((a['record_id'],r['record_id'],'Título normalizado y año exactos; DOI no conflictivos'))
  else:T[k]=i
G=collections.defaultdict(list)
for i,r in enumerate(R):G[root(i)].append(r)
pat={
'btc':r'\bbitcoin\b|\bbtc\b',
'crypto':r'crypto(?:currenc\w*|asset\w*)|digital currenc|\bethereum\b',
'finance':r'financial|stock market|equity|equities|exchange rate|financial market|asset return|S&P|S & P|commodity|forex|portfolio',
'dl':r'deep learning|neural|\bLSTM\b|long short.term|\bGRU\b|gated recurrent|transformer|\bCNN\b|\bRNN\b|attention|\bTCN\b|temporal convolution',
'vol_target':r'(?:forecast\w*|predict\w*|model\w*|estimat\w*)\s+(?:(?:the|financial|bitcoin|cryptocurrency|realized|realised|conditional|market|stock|daily|future|asset|return|price|of)\s+){0,5}volatil\w*|volatil\w*\s+(?:forecast\w*|predict\w*|model\w*|estimat\w*)|realized volatility|realised volatility|conditional variance',
'hybrid':r'\bHAR\b|heterogeneous autoregressive|\bGARCH\b|\bARCH\b|hybrid|integrat\w*|ensemble|fusion|decompos\w*',
'forecast':r'forecast|predict|nowcast',
'price_target':r'(?:forecast\w*|predict\w*)\s+(?:(?:the|of|future|bitcoin|cryptocurrency|stock|closing|close|market|daily|digital|currency)\s+){0,4}(?:price|value)|price\s+(?:predict\w*|forecast\w*)',
'architecture':r'\bN.BEATS\b|\bN.HITS\b|\bTFT\b|temporal fusion transformer|\bPatchTST\b|\biTransformer\b|\bTimeMixer\b|\bDLinear\b|\bTimesNet\b|\bInformer\b|\bAutoformer\b|\bFEDformer\b|\bTimeXer\b|\bTiDE\b|\bTSMixer\b'
}
rx={k:re.compile(v,re.I) for k,v in pat.items()}
U=[]
for n,group in enumerate(G.values(),1):
 best=max(group,key=lambda x:len(x['abstract']));u=best.copy();u['study_id']=f'E2-{n:04}';u['record_ids']='; '.join(r['record_id'] for r in group);u['databases']='; '.join(sorted(set(r['database'] for r in group)));u['dois']='; '.join(sorted(set(r['doi'] for r in group if r['doi'])));u['years']='; '.join(sorted(set(r['year'] for r in group)));u['duplicate_count']=len(group)-1
 text='\n'.join(dict.fromkeys(r['title']+'. '+r['abstract'] for r in group));m={k:rx[k].search(text) for k in rx};flags={k:bool(v) for k,v in m.items()};u['signals']=json.dumps(flags);u['evidence']=' | '.join(k+': '+text[max(0,v.start()-70):min(len(text),v.end()+130)].replace('\n',' ') for k,v in m.items() if v)
 ys=[int(r['year']) for r in group if r['year'].isdigit()];u['period_status']='2019–2026' if any(2019<=y<=2026 for y in ys) else ('fuera_2019_2026' if ys else 'año_incierto')
 if u['period_status']=='fuera_2019_2026':c='FUERA_PERIODO';reason='Años exportados fuera de 2019–2026; conservar para antecedentes, no cribar en ventana.'
 elif m['btc'] and m['vol_target'] and m['dl']:c='PRIORIDAD_PRINCIPAL';reason='Señales BTC + objetivo volatilidad + DL; comprobar objetivo, horizonte y validación en texto completo.'
 elif (m['vol_target'] and (m['crypto'] or m['finance'] or m['btc'])) or m['architecture'] or ((m['btc'] or m['crypto'] or m['finance']) and m['hybrid'] and m['dl']):c='SECUNDARIO_TRANSFERIBLE';reason='Volatilidad financiera/cripto o mecanismo híbrido/arquitectura transferible; no equivale a evidencia directa BTC-volatilidad.'
 elif not best['abstract'] or best['abstract'].strip() in ['[No abstract available]','N/A']:c='INCIERTO_CONSERVAR';reason='Resumen ausente: no excluir por ausencia de evidencia.'
 elif m['price_target'] and not m['vol_target']:c='EXCLUSION_PROPUESTA_PRECIO';reason='Pronóstico explícito de precio/valor, sin señal de volatilidad como objetivo ni mecanismo transferible detectado; auditar falsos negativos.'
 else:c='INCIERTO_CONSERVAR';reason='No hay evidencia suficiente para exclusión segura; requiere revisión humana.'
 u['screening_class']=c;u['reason']=reason;u['screening_method']='triaje automatizado conservador título/resumen v1; no doble cribado humano';u['full_text_inclusion']='NO DETERMINADA';U.append(u)
 for r in group:r['study_id']=u['study_id']
def write(name,rows):
 if not rows:return
 with (P/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
adjustments=json.loads((P/'ajustes_revision.json').read_text())
for u in U:
 u['initial_class']=u['screening_class'];u['targeted_audit']='NO: no revisión individual completa'
 if u['study_id'] in adjustments:
  u['screening_class'],u['reason']=adjustments[u['study_id']];u['targeted_audit']='SÍ: lectura focalizada título/resumen por asistente; sin texto completo'
write('registros_normalizados.csv',R);write('cribado_unicos.csv',U);write('prioridad_y_secundarios.csv',[u for u in U if u['screening_class'] in ['PRIORIDAD_PRINCIPAL','SECUNDARIO_TRANSFERIBLE']]);write('exclusiones_propuestas_auditar.csv',[u for u in U if u['screening_class'].startswith('EXCLUSION')]);write('relaciones_duplicados.csv',[dict(record_a=a,record_b=b,rule=c) for a,b,c in edges]);write('conflictos_duplicados.csv',[dict(record_a=a,record_b=b,reason=c) for a,b,c in conflicts])
summary=dict(raw_counts=dict(collections.Counter(r['database'] for r in R)),raw_total=len(R),unique_total=len(U),duplicates_removed=len(R)-len(U),raw_in_period=dict(collections.Counter(r['database'] for r in R if r['year'].isdigit() and 2019<=int(r['year'])<=2026)),unique_period_counts=dict(collections.Counter(u['period_status'] for u in U)),classes=dict(collections.Counter(u['screening_class'] for u in U)),missing_abstract=sum(not u['abstract'] or u['abstract']=='[No abstract available]' for u in U),dedup_conflicts=len(conflicts),patterns=pat,files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in O.iterdir()})
(P/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='patterns'},ensure_ascii=False,indent=2))
