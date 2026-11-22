"""Interactive development-data preparation for the pinned E3 Colab notebook.

Standard library only in the notebook kernel. All market parsing and integrity
checks use the explicitly versioned isolated engine. Historical A is preserved. No fitting, final preparation, test
opening, or remote writes. Human answers are declarations, not verified identity
or cryptographic signatures. AI assistance: implementation and software review.
Reuse under the repository's applicable code terms; no data license is granted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import textwrap

EXTENSION_VERSION = "crypto-contract-admission-v2"
PANELS = ("F00", "F01", "F02", "F03")
PARTS = ("fit", "selection", "planning", "calibration")
SCOPE = ("Academic noncommercial development, selection, planning and calibration only; "
         "publication of permitted derived outputs with source attribution and source license conditions. "
         "No raw redistribution and no final evaluation authorization.")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


# The subprocess prints only state/count/hash summaries, never author/contact or
# provider URLs containing credentials. It accesses only development partitions.
_CHECK_PROGRAM = r'''
import datetime as dt, json, sys
from pathlib import Path
from engine import data

payload = json.load(sys.stdin)
result = {}
for row in payload['rows']:
    panel, p = row['panel'], row['proposal']
    folder, raw, target = (Path(row[k]) for k in ('folder','raw','panel_dir'))
    owner_path, receipt_path = folder/'owner-development.json', folder/'exposure-development.json'
    rights_path, manifest_path = folder/'rights.json', raw/'raw_manifest.json'
    config = json.loads((folder/'acquisition.json').read_text())
    expected_config = dict(provider=p['provider'], instrument=p['instrument'],
        development_start=p['proposed_development_start'], development_end=p['proposed_development_end'])
    data.require(config == expected_config, panel+': acquisition differs from frozen development proposal')
    split = json.loads((folder/'development-split.json').read_text())
    rule = payload['rule']
    n = len(data.expected_sessions(config['development_start'],config['development_end'],p['calendar']))
    ends = [int(n*.60),int(n*.70),int(n*.90),n]
    bounds = {name:[lo,hi-1] for name,lo,hi in zip(('fit','selection','planning','calibration'),[0]+ends[:-1],ends)}
    expected_split = dict(raw_start=config['development_start'],raw_end=config['development_end'],calendar=p['calendar'],
        raw_bounds=bounds,embargo_bars=rule['embargo_bars'],prospective_split_rule=rule,metadata_only_expected_raw_bars=n)
    data.require(split == expected_split, panel+': split differs from frozen calendar rule')
    data.require(bounds['planning'][1]-bounds['planning'][0]+1-14 >= 80, panel+': planning too short')
    owner = json.loads(owner_path.read_text())
    for key in ('panel_ids','reserve_mode','dataset_scope','source_url','source_license_evidence_path','source_license_evidence_sha256'):
        data.require(owner.get(key) == row['blank_owner'][key], panel+': owner scope/source differs')
    data.require(owner.get('authorize_test_preparation') is False, panel+': final authorization forbidden')
    blank = owner == row['blank_owner']
    if not blank:
        for key in ('actor_name','actor_contact','actor_role','statement_utc','evidence_origin','observation_cutoff_utc'):
            data.require(bool(owner.get(key)), panel+': incomplete owner field '+key)
        statement, cutoff = (dt.datetime.fromisoformat(owner[k]) for k in ('statement_utc','observation_cutoff_utc'))
        data.require(statement.tzinfo is not None and cutoff.tzinfo is not None and cutoff <= statement,
                     panel+': invalid owner timestamps')
        data.require(cutoff.astimezone(dt.timezone.utc).date() >= dt.date.fromisoformat(config['development_end']),
                     panel+': cutoff before development range')
        data.require(owner.get('acknowledge_prior_D00_D03_exposed') is True and
                     owner.get('all_data_declared_exposed_development') is True, panel+': missing exposure designation')
        data.require(owner.get('source_terms_accepted_by_real_actor') is True and
                     owner.get('allowed_processing_scope') and owner.get('AI_completed_signature') is False,
                     panel+': missing authentic source acceptance')
        data.require(all(isinstance(owner.get(key),bool) for key in ('allow_derived_publication','allow_raw_redistribution')),
                     panel+': explicit owner publication permissions required')
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        data.verify_evidence(receipt,'audited')
        data.require(not blank and receipt.get('form') == owner and receipt.get('form_sha256') == data.identity(owner),
                     panel+': owner changed after receipt')
        data.require(receipt.get('final_reserve_claim') is False, panel+': no final reserve permitted')
    rights = None
    if rights_path.exists():
        rights = json.loads(rights_path.read_text())
        data.verify_evidence(rights,'approved')
        data.require(rights['evidence_path'] == owner['source_license_evidence_path'] and
                     rights['evidence_sha256'] == p['rights_snapshot_sha256'] and
                     rights['source_url'] == p['rights_source_url'] and rights['license'] == p['license'],
                     panel+': rights are not bound to pinned primary source')
        data.require(not blank and rights['actor_name'] == owner['actor_name'] and
                     rights['actor_contact'] == owner['actor_contact'], panel+': rights actor differs')
        for key in ('allow_derived_publication','allow_raw_redistribution'):
            data.require(not rights[key] or p[key] and owner.get(key) is True,
                         panel+': permission exceeds source/owner declaration')
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        data.require(manifest.get('manifest_sha256') == data.identity({k:v for k,v in manifest.items() if k != 'manifest_sha256'}),
                     panel+': raw manifest hash mismatch')
        data.require(manifest['provider'] == p['provider'] and manifest['instrument'] == p['instrument'] and
                     manifest['scope'] == 'development_only' and manifest['rights'] == rights and manifest['exposure'] == receipt,
                     panel+': raw provenance/evidence differs')
        data.validate_binance_manifest(manifest,manifest_path)
        for item in manifest['files']:
            path = data._safe_path(raw,item['path'])
            data.require(data.sha256_bytes(path.read_bytes()) == item['sha256'], panel+': raw bytes mismatch')
            if item.get('provider_checksum_sha256'):
                data.require(data.sha256_bytes(Path(str(path)+'.CHECKSUM').read_bytes()) == item['provider_checksum_sha256'],
                             panel+': checksum companion differs')
    elif raw.exists() and any(path.name != 'acquisition_attempts.jsonl' for path in raw.iterdir()):
        data.inspect_binance_pending(config,raw,rights_receipt=rights,exposure_receipt=receipt)
    admitted = (target/'admission.json').is_file()
    if admitted:
        admission = data.read_admission(target)
        data.require(admission['status'] == 'development_admitted' and set(admission['partition_paths']) ==
                     {'fit','selection','planning','calibration'}, panel+': real development-only admission required')
        provenance = admission['provenance']
        data.require(provenance['provider'] == p['provider'] and provenance['instrument'] == p['instrument'] and
                     provenance['split_config_sha256'] == data.identity(split) and
                     admission['rights'] == rights and admission['exposure'] == receipt, panel+': admission provenance differs')
        if manifest:
            data.require(provenance['raw_manifest_sha256'] == manifest['manifest_sha256'],panel+': manifest/admission differs')
        for part in ('fit','selection','planning','calibration'):
            data.load_partition(target,part)
        _, bundle, _, original_bytes, _, _ = data._read_recovery_bundle(target/'recovery_metadata.json')
        data.require(original_bytes == (target/'admission.json').read_bytes() and bundle['split_config'] == split,
                     panel+': recovery/admission differs')
    elif target.exists() and any(target.iterdir()):
        raise data.AdmissionError(panel+': partial admission retained; do not overwrite; '+str(target))
    if manifest or admitted:
        data.require(rights is not None and receipt is not None,panel+': original evidence missing')
    result[panel] = dict(admitted=admitted,raw_complete=manifest is not None,blank_owner=blank,
        owner_receipt=receipt is not None,rights_present=rights is not None)
print(json.dumps({'status':'development_preparation_checked','panels':result,'new_market_fits':0,'test_opened':False}))
'''


def _check(namespace, rows, rule):
    # Explain metadata failures without printing forms, private payloads or URLs.
    program = "try:\n" + textwrap.indent(_CHECK_PROGRAM, '    ') + "\nexcept Exception as exc:\n" + (
        "    print(json.dumps({'status':'development_preparation_failed','reason':str(exc)}),flush=True)\n"
        "    raise SystemExit(1)\n")
    output = namespace['run_private_program'](program, {'rows': rows, 'rule': rule})
    return json.loads(output.stdout)['panels']


def _progress_call(namespace, args, row, provider=None):
    stopped = threading.Event()

    def report():
        while not stopped.wait(30):
            count = sum(1 for p in Path(row['raw']).glob('*') if p.suffix in ('.zip','.json'))
            print(f"{row['panel']}: preparación en curso; {count} archivos raw locales.", flush=True)

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        try:
            return namespace['run_data'](args, provider=provider)
        except Exception:
            raise RuntimeError(
                row['panel']+': descarga detenida; se conservan los archivos. Revisar '+row['raw']+
                '/acquisition_attempts.jsonl. El colector conserva checksums y reanuda sólo los bytes '
                'compatibles con su manifiesto pendiente; revisar el diagnóstico local. No se ha entrenado ni abierto test.') from None
    finally:
        stopped.set()
        thread.join()


def prepare_development(namespace):
    """Run after notebook stages 1/2. Explicit human consent precedes writes/network."""
    required = ('SOURCE','SOURCE_COMMIT','MANIFEST_SHA256','SETTINGS','PRIVATE','definitions','DEVELOPMENT_MAP',
                'run_data','run_private_program')
    _require(all(key in namespace for key in required), 'Ejecuta primero las celdas 1 y 2 del cuaderno.')
    source, private = Path(namespace['SOURCE']), Path(namespace['PRIVATE'])
    commit = namespace['SOURCE_COMMIT']
    manifest_pin = namespace['MANIFEST_SHA256']
    _require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
             and isinstance(manifest_pin,str) and len(manifest_pin)==64 and all(c in '0123456789abcdef' for c in manifest_pin)
             and namespace['SETTINGS']['source_commit'] == commit,
             'Conservar la fuente publicada de esta versión; no mezclar campañas.')
    _require(Path(namespace['SETTINGS']['source_root']) == source, 'SETTINGS apunta a otra fuente.')
    anchor = _read(source/'PUBLIC_SOURCE_ANCHOR.json')
    _require(anchor.get('source_commit') == commit and anchor.get('manifest_sha256') == manifest_pin,
             'Ancla de fuente diferente.')
    manifest_path = source/'SOURCE_MANIFEST.json'
    _require(_sha(manifest_path) == manifest_pin, 'Manifest de fuente diferente.')
    for relative, expected in _read(manifest_path)['files'].items():
        path = source/relative
        _require(path.resolve().is_relative_to(source.resolve()) and not path.is_symlink() and _sha(path) == expected,
                 'Archivo fuente diferente: '+relative)
    proposals = {p['id']:p for p in _read(source/'configs/data_panels.json')['panels']}
    _require(set(proposals)==set(PANELS) and {p['provider'] for p in proposals.values()} <=
             {'binance_vision','binance_vision_um','binance_vision_cm'},
             'Este cuaderno sólo acepta las propuestas cripto versionadas.')
    source_form = _read(source/'docs/OWNER_EXPOSURE_FORM.json')
    rule = _read(private/'development-split-rule.json')
    _require(rule == {
        'version':'development-calendar-proportions-1','fit_fraction':.60,'selection_fraction':.10,
        'planning_fraction':.20,'calibration_fraction':.10,'embargo_bars':0,
        'basis':'expected exchange sessions only; no market prices, labels or performance used',
        'support_rule':'raw disjoint bounds; engine purges endpoints until all 15 closes fit the assigned support'},
        'Regla de partición diferente; conservar el caso y revisar su versión.')
    rows = []
    _require(tuple(d['panel'] for d in namespace['definitions']) == PANELS, 'Se requieren exactamente F00–F03.')
    for entry in namespace['definitions']:
        panel, proposal = entry['panel'], proposals[entry['panel']]
        _require(entry['proposal'] == proposal, 'Propuesta del cuaderno diferente: '+panel)
        row = dict(panel=panel,proposal=proposal,
                   folder=str(private/'inputs'/panel),raw=str(private/'raw'/panel/'development-v1'),
                   panel_dir=str(private/'panels'/panel/'development-v1'))
        for key in ('folder','raw','panel_dir'):
            _require(Path(entry[key]) == Path(row[key]), 'Ruta del cuaderno diferente: '+panel)
        license_path = source/proposal['rights_snapshot_path']
        _require(_sha(license_path) == proposal['rights_snapshot_sha256'], 'Evidencia primaria diferente: '+panel)
        blank = dict(source_form, panel_ids=[panel],reserve_mode='development_only',
            source_url=proposal['rights_source_url'],source_license_evidence_path=str(license_path),
            source_license_evidence_sha256=_sha(license_path),dataset_scope=dict(panel=panel,
                provider=proposal['provider'],instrument=proposal['instrument'],
                start=proposal['proposed_development_start'],end=proposal['proposed_development_end']))
        row['blank_owner'] = blank
        rows.append(row)
    _require(_read(namespace['DEVELOPMENT_MAP']) == {r['panel']:r['panel_dir'] for r in rows}, 'Mapa de paneles diferente.')
    states = _check(namespace,rows,rule)
    if all(s['admitted'] for s in states.values()):
        print('Los cuatro paneles ya están verificados. Ejecuta la etapa 3 del cuaderno.')
        return states
    print('\nAlcance: descarga y preparación de DESARROLLO. Cero entrenamientos; no se prepara ni abre test.')
    for row in rows:
        p = row['proposal']
        print(f"{row['panel']} {p['provider']} {p['instrument']}: {p['proposed_development_start']} a {p['proposed_development_end']}")
        print('  Licencia:',p['license'])
        print('  Fuente primaria:',p['rights_source_url'])
    print('Revisa esas condiciones y sus copias locales antes de aceptar. Binance: uso no comercial, atribución,')
    print('derivados redistribuidos bajo CC BY-NC-SA 4.0, sin sugerir respaldo. Los archivos son públicos; no se pide cuenta Nasdaq ni de Binance.')
    print('Autorizas descarga privada, desarrollo, selección, planificación, calibración y derivados permitidos.')
    print('No se redistribuyen raw. Todos estos periodos se declaran expuestos, sin reserva final.')
    print('También reconoces que los antiguos D00–D03 ya estuvieron expuestos y no constituyen una reserva final.')
    print('Tu declaración se registra con fecha real; el programa no autentica tu identidad ni firma por ti.')
    needed = [row for row in rows if states[row['panel']]['blank_owner'] or not states[row['panel']]['rights_present']]
    name = contact = role = None
    if needed:
        name = input('Nombre completo del autor o custodio que declara: ').strip()
        contact = input('Correo o contacto verificable del declarante: ').strip()
        role = input('Tu función (autor o custodio): ').strip()
        _require(name and contact and role, 'Se requieren nombre, contacto y función reales.')
        for row in needed:
            if not states[row['panel']]['blank_owner']:
                owner = _read(Path(row['folder'])/'owner-development.json')
                _require(name == owner['actor_name'] and contact == owner['actor_contact'],
                         'El declarante no coincide con el formulario existente; conservar y revisar.')
    if input('Para autorizar lo descrito escribe ACEPTO DESARROLLO: ').strip() != 'ACEPTO DESARROLLO':
        print('Cancelado. No se guardaron declaraciones ni se descargaron datos.')
        return states
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    for row in rows:
        folder, state = Path(row['folder']), states[row['panel']]
        owner_path = folder/'owner-development.json'
        if state['blank_owner']:
            _require(not (folder/'exposure-development.json').exists() and _read(owner_path) == row['blank_owner'],
                     'El formulario cambió; se conserva sin modificar.')
            owner = dict(row['blank_owner'],kind='AUTHOR_SUPPLIED_INTERACTIVE_DEVELOPMENT_DECLARATION',
                actor_name=name,actor_contact=contact,actor_role=role,statement_utc=timestamp,
                evidence_origin='Real actor typed identifying fields and ACEPTO DESARROLLO in the Colab helper; not identity authentication.',
                acknowledge_prior_D00_D03_exposed=True,all_data_declared_exposed_development=True,
                observation_cutoff_utc=timestamp,source_terms_accepted_by_real_actor=True,
                allowed_processing_scope=SCOPE,allow_derived_publication=True,allow_raw_redistribution=False,
                AI_completed_signature=False,authorize_test_preparation=False)
            # Only the untouched blank is replaced. Preserve its exact bytes.
            backup = folder/'owner-development.blank-template.json'
            original = owner_path.read_bytes()
            if backup.exists():
                _require(backup.read_bytes() == original,'Copia de plantilla diferente; conservar y revisar.')
            else:
                with backup.open('xb') as stream:
                    stream.write(original)
            with tempfile.NamedTemporaryFile(dir=folder,delete=False,mode='w',encoding='utf-8') as stream:
                temporary = Path(stream.name)
                json.dump(owner,stream,indent=2,ensure_ascii=False,allow_nan=False)
                stream.write('\n')
            try:
                os.replace(temporary,owner_path)
            finally:
                temporary.unlink(missing_ok=True)
        owner = _read(owner_path)
        if not state['rights_present']:
            p = row['proposal']
            _new_json(folder/'rights.json',dict(status='approved',actor_name=owner['actor_name'],
                actor_contact=owner['actor_contact'],evidence_path=owner['source_license_evidence_path'],
                evidence_sha256=owner['source_license_evidence_sha256'],scope=SCOPE,
                source_url=p['rights_source_url'],license=p['license'],allow_derived_publication=owner['allow_derived_publication'],
                allow_raw_redistribution=False,statement_utc=timestamp,
                decision_origin='Explicit interactive actor acceptance; no identity or signature verification.'))
        if not state['owner_receipt']:
            namespace['run_data'](['record-owner-evidence','--form',str(owner_path),'--evidence',str(owner_path),
                                   '--output',str(folder/'exposure-development.json')])
    states = _check(namespace,rows,rule)
    # Preserve the prospectively fixed panel order; selection never uses performance.
    for row in rows:
        state, folder = states[row['panel']], Path(row['folder'])
        if state['admitted']:
            print(row['panel']+': admisión existente verificada; se reutiliza.')
            continue
        if not state['raw_complete']:
            print(row['panel']+': descargando desarrollo; puede tardar varios minutos.',flush=True)
            _progress_call(namespace,['collect-development','--config',str(folder/'acquisition.json'),
                '--rights',str(folder/'rights.json'),'--exposure',str(folder/'exposure-development.json'),
                '--output',row['raw'],'--enable-market-acquisition'],row,provider=row['proposal']['provider'])
        namespace['run_data'](['prepare-development','--manifest',str(Path(row['raw'])/'raw_manifest.json'),
            '--split-config',str(folder/'development-split.json'),'--rights',str(folder/'rights.json'),
            '--exposure',str(folder/'exposure-development.json'),'--output',row['panel_dir']])
        print(row['panel']+': desarrollo preparado.',flush=True)
    states = _check(namespace,rows,rule)
    _require(all(s['admitted'] for s in states.values()),'Conservar archivos; aún falta una admisión válida.')
    print('Cuatro paneles verificados. Ahora ejecuta la etapa 3 del cuaderno. No se ha entrenado ningún modelo.')
    return states
