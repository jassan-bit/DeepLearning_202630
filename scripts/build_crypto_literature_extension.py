"""Rebuild aggregate literature tables from public extracted records; never run models."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'references/literature_e1/crypto_extension_20261105'
PUBLIC=ROOT/'book/_static/literature_e1/crypto_extension_20261105'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rows,fields=None):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
def cell(x):
    return json.dumps(x,ensure_ascii=False,separators=(',',':')) if isinstance(x,(list,dict)) else x


def main():
    matrix=read(SOURCE/'evidence_matrix.json');studies=matrix['studies']
    by_id={r['study_id']:r for r in studies}
    if len(by_id)!=len(studies) or len(studies)!=10:raise ValueError('Ten distinct current evidence records required')
    crypto=[r for r in studies if r['admissible_fulltext_frequency'].get('crypto')]
    btc=[r for r in crypto if r['admissible_fulltext_frequency'].get('BTC_subset')]
    multi=[r for r in crypto if r['admissible_fulltext_frequency'].get('includes_non_BTC_crypto')]
    ids=[r['study_id'] for r in crypto]
    if ids!=['S1','S2','S5','S6','S7','S8','S9'] or len(btc)!=7 or [r['study_id'] for r in multi]!=['S9']:
        raise ValueError('Preserve seven broad crypto studies and overlapping asset subsets')
    if by_id['S9']['independent_model_families'] or by_id['S9']['provenance']['model_presence_separate_from_reference_frequency']!=['LSTM']:
        raise ValueError('Do not count the S9 proposed LSTM or internal MLP as a verified reference')
    families=['MLP','LSTM','GRU','CNN_generic','CNN1D_simple','TCN','Transformer_encoder']
    extractable=[r for r in crypto if r['independent_model_families']]
    # Publication identity in the original is separate from journal/index/Q1 verification.
    identified=[r for r in crypto if r['type']=='publication']
    aliases={'CNN (dimensionality unknown)':'CNN_generic','Transformer encoder':'Transformer_encoder'}
    frequency=[]
    for stratum,group in [('all_crypto_fulltexts',crypto),('DL_reference_extractable_fulltexts',extractable),('publication_identified_in_original',identified)]:
        for family in families:
            present=[r['study_id'] for r in group if family in {aliases.get(x,x) for x in r['independent_model_families']}]
            frequency.append({'stratum':stratum,'family':family,'verified_reference_studies':';'.join(present),
                'count':len(present),'denominator':len(group),'proportion':len(present)/len(group),
                'role':'independently verified comparative family; not an internal head, code-only source or proposed S9 framework',
                'scope':'Descriptive recovery corpus; not an unbiased prevalence estimate for all published literature'})
    actual={r['family']:r['count'] for r in frequency if r['stratum']=='all_crypto_fulltexts'}
    if actual!={'MLP':1,'LSTM':5,'GRU':3,'CNN_generic':1,'CNN1D_simple':0,'TCN':1,'Transformer_encoder':2}:
        raise ValueError('Positive family evidence differs from the reviewed records')
    csvwrite(SOURCE/'family_frequencies.csv',frequency)
    csvwrite(SOURCE/'evidence_matrix.csv',[{k:cell(v) for k,v in r.items()} for r in studies])
    reported=[];synthesis=[]
    for r in studies:
        metrics=r['reported_metrics']
        values=metrics if isinstance(metrics,list) else (metrics.get('exact_values_and_locations') or [])
        for i,value in enumerate(values,1):
            reported.append({'study_id':r['study_id'],'author_record_ordinal':i,
                'model':value.get('model',value.get('model_label_as_printed','not separately identified')),
                'metric_or_metric_columns':';'.join(k for k in value if any(t in k.upper() for t in ['MSE','MAE','MAPE','QLIKE','METRIC','R2','RMSE'])) or 'see preserved author record',
                'reported_record_JSON':json.dumps(value,ensure_ascii=False,separators=(',',':')),
                'source_location':value.get('location',value.get('source_location',value.get('source','see source record'))),
                'source_url':r['primary_source_url'],'metrics_recomputed':False,
                'DL_metric_claim':'False for S9; its Table2 numbers are classical/tree winners' if r['study_id']=='S9' else 'As qualified by the original extraction'})
        if r in crypto:
            synthesis.append({'study_id':r['study_id'],'reference':f"{r['title']} ({r['year'] if r['year'] is not None else 'year unknown'})",
                'problem':r['target_definition'],'dataset_and_assets':cell({'assets':r['assets'],'dataset':r['dataset']}),
                'DL_models_used_as_independent_references':';'.join(r['independent_model_families']) or 'None positively verified; S9 proposed LSTM and internal MLP kept separate',
                'reported_metrics_and_values':cell(values),'metric_source_locations':cell(r['primary_source_locations']),
                'quality_and_comparability':cell({'quality':r['quality'],'comparability':r['comparability_with_E1']})})
    if len(reported)!=119:raise ValueError('Preserve113 inherited author records and six S9 records')
    csvwrite(SOURCE/'reported_metrics.csv',reported);csvwrite(SOURCE/'study_synthesis.csv',synthesis)
    search=read(SOURCE/'search_log.json');requests=search['requests']
    allrepos=[repo for r in requests for repo in r['repositories_returned'] if isinstance(repo,str) and '/' in repo]
    queries={r['query'] for r in requests if r['query']}
    search_summary={'query_strings':len(queries),'page_requests':len(requests),
        'successful_pages':sum(r['http_status']==200 for r in requests),
        'retrieved_repository_occurrences':len(allrepos),'unique_repositories_returned':len(set(allrepos)),
        'reported_index_totals_summed':False,'repository_counts_are_not_article_counts':True,
        'scope':'Three recorded search tracks, not an exhaustive/native WoS/Scopus search. A result not inspected remains unassessed.',
        'new_qualifying_MLP_crypto_originals':0,'new_qualifying_CNN1D_crypto_originals':0,
        'reassessed_known_multicrypto_manuscript_included':'S9'}
    dump(SOURCE/'search_summary.json',search_summary)
    screens=read(SOURCE/'screening_findings.json');general=read(SOURCE/'general_screening.json')
    screening=[]
    for item in screens['MLP'].get('findings',[]):
        screening.append({'track':'MLP','source':item.get('repository',item.get('source','see record')),
            'decision':'No new independently verified MLP article occurrence',
            'reason':item.get('reason',item.get('decision',cell(item))),
            'evidence_record':'screening_findings.json#/MLP/findings'})
    for item in general['fulltext_screening']:
        screening.append({'track':'general','source':item['source'],'decision':item['decision'],'reason':item['evidence'],'evidence_record':'general_screening.json#/fulltext_screening'})
    screening.append({'track':'CNN','source':'jackywong1228/CTA-Quant-Papers PDF85',
        'decision':'excluded_document_identity_mismatch',
        'reason':'Retrieved titlepage is Reijnen et al. DR-ALNS, not the claimed Herremans volatility article. No article identity inferred from filename.',
        'evidence_record':'screening_findings.json#/CNN/document_identity_alert'})
    screening.append({'track':'CNN','source':'AyoTecca/crypto-horserace and Al3nzy/Crypto-Volatility-Horserace',
        'decision':'code_only_deduplicated_no_article_occurrence',
        'reason':'Independent CNN-only constructor seen in code; matching constructor AST is not two independent scientific studies; original article not recovered.',
        'evidence_record':'screening_findings.json#/CNN/code_only_evidence'})
    csvwrite(SOURCE/'screening.csv',screening)
    catalog=read(SOURCE/'references_access_40.json')
    bib=(ROOT/'book/references.bib').read_text();keys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
    old=read(ROOT/'references/literature_e1/journal_update_20261105/references_access_39.json')
    if len(keys)!=40 or len(set(keys))!=40 or set(keys)!={r['key'] for r in catalog['references']} or not {r['key'] for r in old['references']}.issubset(keys):
        raise ValueError('Keep the39 prior citations and add exactly the verified S9 reference')
    csvwrite(SOURCE/'references_access_40.csv',[{'key':r['key'],'year':r.get('year'),'title':r['title'],'category':r['category'],
        'source_url':r.get('displayed_consulted_url',r.get('bibliographic_url')),
        'evidence_level':r.get('evidence_level'),'full_text_read':r.get('full_text_read_for_this_review'),
        'study_id':r.get('study_id'),'limitations':r.get('historical_limitations')} for r in catalog['references']])
    (SOURCE/'references_access_40.bib').write_text(bib)
    counts={'bibliographic_references':40,'crypto_full_text_studies':len(crypto),'BTC_full_text_studies':len(btc),
        'studies_with_non_BTC_crypto':len(multi),'prior_full_text_studies':6,'new_crypto_inclusions':1,
        'MLP_baseline_studies':actual['MLP'],'CNN_generic_baseline_studies':actual['CNN_generic'],
        'CNN1D_explicit_baseline_studies':actual['CNN1D_simple'],'LSTM_baseline_studies':actual['LSTM'],
        'neural_comparator_extractable_studies':len(extractable),'publication_identified_crypto_studies':len(identified),
        'normalized_evidence_rows':len(studies),'author_metric_records':len(reported)}
    summary={'schema_version':1,'stage_id':'crypto_extension_20261105','compiled_at_utc':datetime.now(timezone.utc).isoformat(),
        'counts':counts,'study_ids':ids,'new_study_ids':['S9'],'prior_study_ids':ids[:-1],
        'harnet_erratum_file':'harnet_erratum.json','timing':'post-benchmark documentary extension',
        'experiment_results_modified':False,'new_model_fits':0,'new_forecasts':0,
        'scope':'Any cryptocurrency with continuous volatility/variance forecast; the experimental project remains Bitcoin only.',
        'scope_subsets_overlap':'S9 includesBTC and19otherassets; sevenBTC-containing studies and one nonBTC-asset study are overlapping subsets of seven crypto studies.',
        'baseline_reference_vs_presence':'S9 has a proposed LSTM and an internal MLP head; it adds no strict MLP/CNN/LSTM reference occurrence. LSTM model presence6/7 differs from reference frequency5/7.',
        'legacy_comparator_stratum':'The six prior studies retain their1/6MLP,1/6genericCNN,0/6explicitCNN1D and5/6LSTM counts in a labelled subset.',
        'remaining_limits':['No new independently verified MLP/CNN1D crypto original found in this bounded recovery.','HARNet original remains unread; author-code tier only, with corrected scaler interpretation.','DeepVol original remains unread; publication identity tier only.','Journal-level index matching does not verify individual article/Q1; S9 is not a verified journal publication.'],
        'search':search_summary,'access_limit':'Normal proxy deniedarXiv/Crossref. Academic-network draft saved, not applied. No database hit counts invented.',
        'public_handling':'Extracted factual metadata/tables and source URLs/hashes only; no full third-party papers or raw market data redistributed.'}
    dump(SOURCE/'summary.json',summary)
    lines=["Corpus previo: 6 textos BTC",
        f"Búsqueda ampliada: {len(queries)} cadenas / {len(requests)} páginas de repositorios",
        "Reevaluación de S9: manuscrito conocido, 20 criptoactivos",
        "Corpus amplio: 7 textos cripto; BTC en 7; otros activos en 1 (subconjuntos solapados)",
        "Referencias DL verificadas (N=7): MLP 1, CNN genérica 1, CNN1D 0, LSTM 5",
        "Estrato de 6 con comparadores DL conservado; transferencia financiera separada"]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="650" viewBox="0 0 1040 650">','<rect width="1040" height="650" fill="white"/>','<text x="30" y="32" font-family="sans-serif" font-size="20">Flujo de ampliación documental:criptomonedas</text>']
    for i,line in enumerate(lines):
        y=60+90*i;svg.append(f'<rect x="25" y="{y}" width="990" height="62" rx="8" fill="#e8f1fa" stroke="#345d82"/><text x="42" y="{y+36}" font-family="sans-serif" font-size="17">{html.escape(line)}</text>')
        if i<5:svg.append(f'<path d="M520 {y+62}v26" stroke="#345d82" stroke-width="2"/>')
    svg.append('</svg>');(SOURCE/'flow.svg').write_text('\n'.join(svg)+'\n')
    (SOURCE/'README.md').write_text("""# Revisión ampliada a criptomonedas

Corte documental posterior al benchmark: 40 referencias, siete textos completos sobre criptomonedas y una matriz de diez registros. S9 estudia 20 activos y se cuenta una sola vez. Incluye BTC: los subconjuntos de estudios con BTC y con otros activos se solapan. La frecuencia de comparadores independientes se distingue de la presencia de componentes propuestos.

evidence_matrix.csv/json localizan objetivo, activos, frecuencia, horizonte, arquitectura, rol, métricas y páginas o tablas. family_frequencies.csv separa tres estratos: los siete textos cripto, los seis con referencias DL extraíbles y los dos con identificación editorial en el original. La identificación de un congreso no acredita Q1 ni un registro individual en WoS/Scopus.

Los seis valores QLIKE de S9 pertenecen a ganadores clásicos o de árboles; no se asignan a LSTM. Una cabeza MLP dentro de LSTM, una TCN o una CNN genérica cuya dimensionalidad no se informa no suman una referencia MLP/CNN1D simple. El experimento sigue siendo de Bitcoin: no se modifican recetas, preregistros o resultados.

harnet_erratum.json retira la inferencia no sustentada de fuga por scaler.fit_transform: es un escalador local con límites configurados. Esto no certifica ausencia global de fuga. harnet_schema.json documenta la diferencia rv5/rv5_ss. HARNet y DeepVol conservan sus niveles de código de autor e identidad editorial; sus originales siguen pendientes.

journal_update_20261105/ y recovery_20261105/ son cortes históricos. Este addendum es la interpretación documental vigente. La búsqueda acotada y dirigida no estima la prevalencia global de la literatura ni demuestra inexistencia de artículos. Los conteos GitHub son repositorios, páginas y ocurrencias; no son resultados de WoS, Scopus o arXiv.

Se publican metadatos y tablas extraídos, sin redistribuir manuscritos, cuadernos o datos de terceros. Para reconstruir las tablas agregadas: python scripts/build_crypto_literature_extension.py. La consulta original se sustenta en URL, SHA-256 y ubicación específica.
""")
    # Only these public metadata formats may enter the source export.
    allowed={'.json','.csv','.bib','.md','.svg'}
    files={p.name:sha(p) for p in sorted(SOURCE.iterdir()) if p.is_file() and p.name!='manifest.json'}
    if any(Path(name).suffix not in allowed for name in files):raise ValueError('Unexpected source distribution artifact')
    dump(SOURCE/'manifest.json',{'stage_id':'crypto_extension_20261105','files_sha256':files,
        'builder_sha256':sha(__file__),'experiment_fits':0,'experiment_predictions':0,'full_third_party_manuscripts_copied':0})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    expected=set(files)|{'manifest.json'}
    if any(p.is_file() and p.name not in expected for p in PUBLIC.iterdir()):raise ValueError('Unexpected stale public-stage file')
    for name in sorted(expected):shutil.copyfile(SOURCE/name,PUBLIC/name)
    print(json.dumps({'public_files':len(expected),'bibliographic_references':40,'crypto_fulltexts':7,'strict_CNN1D_reference_count':0,'model_fits':0,'forecasts':0}))


if __name__=='__main__':main()
