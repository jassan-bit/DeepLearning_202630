"""Render the versioned crypto data notebook; generation performs no fitting.

AI-assisted operational extension of the historical eight-stage notebook. The
model, bank, recipes, trainer, statistics and scientific decision rules retain
their historical hashes. Only data adapters and operational paths are versioned.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from package import notebook_template as historical

PREFIX = 'experiments/e3_hypothesis_adequacy_20261122/runtime'
SOURCE_COMMIT = '__SOURCE_COMMIT__'
MANIFEST_SHA256 = '__MANIFEST_SHA256__'
BOOTSTRAP_SHA256 = '__BOOTSTRAP_SHA256__'


def notebook(source_commit=SOURCE_COMMIT, manifest_sha256=MANIFEST_SHA256,
             bootstrap_sha256=BOOTSTRAP_SHA256):
    for value, pattern, placeholder in (
        (source_commit,r'[0-9a-f]{40}',SOURCE_COMMIT),
        (manifest_sha256,r'[0-9a-f]{64}',MANIFEST_SHA256),
        (bootstrap_sha256,r'[0-9a-f]{64}',BOOTSTRAP_SHA256)):
        if value != placeholder and not re.fullmatch(pattern,value or ''):
            raise ValueError('Published immutable source pins are required')
    result = historical.notebook(source_commit,manifest_sha256,bootstrap_sha256)
    for cell in result['cells']:
        source = cell['source']
        source = source.replace('e3-r3-engine','e3-r3-crypto-engine-v2')
        source = source.replace('runs/e3_r3_engine/','runs/e3_r3_crypto_engine_v2/')
        source = source.replace('author-run-01','crypto-development-01')
        if cell['cell_type'] == 'code':
            source = source.replace('/experiments/e3_confirmatory_engine_20261121/bootstrap_colab.py',
                                    '/'+PREFIX+'/bootstrap_colab.py')
            if 'SETTINGS = bootstrap.setup(' in source:
                source = source.replace('SETTINGS = bootstrap.setup(SOURCE_COMMIT, MANIFEST_SHA256, WORK / "cache", run_tests=True)',
                    'previous_info = globals().get("SETTINGS")\n'
                    'previous_python = None\n'
                    'if isinstance(previous_info, dict) and previous_info.get("source_commit") in {"c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c", SOURCE_COMMIT}:\n'
                    '    candidate_python = previous_info.get("python")\n'
                    '    if isinstance(candidate_python, (str, Path)) and Path(candidate_python).is_file():\n'
                    '        previous_python = str(candidate_python)\n'
                    'SETTINGS = bootstrap.setup(SOURCE_COMMIT, MANIFEST_SHA256, WORK / "cache", existing_python=previous_python, run_tests=True)')
            if 'def github_credentials():' in source:
                old = '''    return {"token": private_value("E3_R3_GITHUB_TOKEN", "GitHub Contents read/write token: "),
            "backup_secret": private_value("E3_R3_BACKUP_SECRET", "Clave independiente >=32 caracteres (conservar): ")}'''
                new = '''    values = {"token": private_value("E3_R3_GITHUB_TOKEN", "GitHub Contents read/write token: "),
              "backup_secret": private_value("E3_R3_BACKUP_SECRET", "Clave independiente >=32 caracteres (conservar): ")}
    if not values["token"] or len(values["backup_secret"]) < 32:
        values.clear()
        raise ValueError("Token no vacío y clave independiente de al menos 32 caracteres requeridos")
    return values'''
                assert old in source
                source = source.replace(old,new)
                start = source.index('def run_data(')
                stop = source.index('def run_private_program(',start)
                source = source[:start]+'''def run_data(args, *, provider=None):
    # Binance Vision Spot/UM/CM uses public archives, never a provider account.
    allowed = (None, "binance_vision", "binance_vision_um", "binance_vision_cm")
    assert provider in allowed, "Usar sólo el adaptador cripto versionado"
    result = subprocess.run([SETTINGS["python"], "-u", "-m", "engine.data", *args],
                            cwd=SOURCE, env=bootstrap.cpu_env(), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError("Admisión detenida; conservar recibos y revisar el ledger local. No imprimir credenciales.")
    return result

'''+source[stop:]
                source += '''
def require_final_execution():
    gate = read_json(SOURCE / "configs/confirmatory_gate.json")
    codes = gate.get("gate_codes", ["CONFIRMATORY_GATE_METADATA_MISSING"])
    raise RuntimeError("Confirmación no habilitada por esta versión; resolver y validar en una versión propia antes de nuevos resultados: " + " ".join(codes))
'''
            if source.startswith('RECORD_OWNER_EVIDENCE = False'):
                source = source[:source.index('if RECORD_OWNER_EVIDENCE:')]
                source = source.replace('RECORD_OWNER_EVIDENCE = False\nACQUIRE_MARKET_DEVELOPMENT = False\nPREPARE_MARKET_DEVELOPMENT = False',
                                        'PREPARE_CRYPTO_DEVELOPMENT = False')
                source = source.replace('source_panels = {p["id"]: p for p in read_json(SOURCE / "configs/data_panels.json")["panels"]}',
                    'source_panels = {p["id"]: p for p in read_json(SOURCE / "configs/data_panels.json")["panels"]}\n'
                    'assert {p["provider"] for p in source_panels.values()} <= {"binance_vision", "binance_vision_um", "binance_vision_cm"}, "Conservar la versión cripto publicada"')
                source = source.replace('"scope": "academic noncommercial research: development, planning, final evaluation and permitted derived outputs; actor must verify source conditions",',
                                        '"scope": "academic noncommercial development and planning only; real actor must accept source conditions; no final authorization",')
                source = source.replace('print("Adquisición y splits por calendario preparados. Completar owner-development.json con datos reales del actor y copiar rights-DRAFT.json a rights.json después de verificar/aprobar su alcance.")',
                    'print("Metadatos de desarrollo preparados. Para descargar y admitir, activar sólo PREPARE_CRYPTO_DEVELOPMENT=True y volver a ejecutar esta etapa.")')
                source += '''
if PREPARE_CRYPTO_DEVELOPMENT:
    helper_path = SOURCE / "package/crypto_development.py"
    helper_spec = importlib.util.spec_from_file_location("e3_crypto_development_helper", helper_path)
    development_helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(development_helper)
    development_helper.prepare_development(globals())
else:
    print(json.dumps({"status":"metadata_only_no_market_acquisition", "next_action":"Set PREPARE_CRYPTO_DEVELOPMENT=True in stage 2 and execute that cell; answer the real actor declaration", "new_market_fits":0, "test_opened":False}))
'''
            if source.startswith('RUN_PLANNING_TRAINING = False'):
                source = source.replace('configs/power_design.json','configs/power_design_crypto_metadata.json')
                source = source.replace('if RUN_PLANNING_TRAINING:\n    assert not missing, "Completar primero las cuatro admisiones"',
                    'if RUN_PLANNING_TRAINING:\n    assert not missing, "Completar primero las cuatro admisiones"\n'
                    '    acceptance = input("Planificación: 340 trabajos, 320 ajustes neuronales ×120s nominales =10h40, más ARIMA, E/S y guardado. Escribe PLANIFICAR 340 para activar: ").strip()\n'
                    '    assert acceptance == "PLANIFICAR 340", "Planificación no activada; se conservan los datos"')
                source = source.replace('if RUN_DIMENSIONING:\n    assert PLANNING_PREDICTIONS.is_file() and PLANNING_RECEIPT.is_file()',
                    'if RUN_DIMENSIONING:\n    assert PLANNING_PREDICTIONS.is_file() and PLANNING_RECEIPT.is_file()\n'
                    '    import datetime\n'
                    '    audit_stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")\n'
                    '    audit_output = PRIVATE / ("power-input-audit-" + audit_stamp + ".json")\n'
                    '    audit_args = [SETTINGS["python"], str(SOURCE / "package/audit_power_inputs.py"), "--source-root", str(SOURCE),\n'
                    '                  "--expected-source-commit", SOURCE_COMMIT, "--expected-manifest-sha256", MANIFEST_SHA256,\n'
                    '                  "--design", str(POWER_DESIGN), "--predictions", str(PLANNING_PREDICTIONS), "--receipts", str(PLANNING_RECEIPT),\n'
                    '                  "--panel-map", str(DEVELOPMENT_MAP), "--output", str(audit_output)]\n'
                    '    checked = subprocess.run(audit_args, cwd=SOURCE, env=bootstrap.cpu_env(), text=True, capture_output=True)\n'
                    '    print(checked.stdout, end="")\n'
                    '    assert checked.returncode == 0 and read_json(audit_output)["planning_inputs_valid"] is True, "Completar contrato real de planificación antes de dimensionar"')
                source += '''
print(json.dumps({"confirmation_enabled_in_this_version":False,
                  "unresolved_final_gates":read_json(SOURCE / "configs/confirmatory_gate.json")["gate_codes"],
                  "dimension_scope":"conditional empirical MSE power study; not certification of the inherited primary endpoint or the full joint decision"}))
'''
            if source.startswith('FREEZE_FINAL_PROTOCOL = False'):
                source = source.replace('ARCHIVE_POWER_EVIDENCE = False',
                    'ARCHIVE_POWER_EVIDENCE = False\nif FREEZE_FINAL_PROTOCOL:\n    require_final_execution()')
            if source.startswith('RUN_FINAL_TRAINING = False'):
                source = source.replace('SEAL_FINAL_TRAINING = False',
                    'SEAL_FINAL_TRAINING = False\nif RUN_FINAL_TRAINING or SEAL_FINAL_TRAINING:\n    require_final_execution()')
            if source.startswith('REGISTER_FINAL_EXPOSURE = False'):
                source = source.replace('EXPORT_SAVED_RESULTS = False',
                    'EXPORT_SAVED_RESULTS = False\nif REGISTER_FINAL_EXPOSURE or PREPARE_RESERVED_TEST or OPEN_RESERVED_EVALUATION:\n    require_final_execution()')
            source = source.replace('print("El caso local aún no está preparado.")',
                'print("Todavía no hay una campaña preparada. Primera acción: etapa 2, PREPARE_CRYPTO_DEVELOPMENT=True; después verificar en etapa 3. No se ha entrenado.")')
            ast.parse(source)
        else:
            if source.startswith('# E3 ·'):
                source = '''# E3 · hipótesis, adecuación y planificación cripto

Extensión operacional versionada: Bitcoin Spot principal y tres propuestas de
contratos perpetuos, justificadas por dominio económico del contrato y tamaño
de desarrollo fijado antes de examinar resultados. Conserva la candidata
`input-only-overlap-mse-1`, sus 20 rivales, el banco de semillas y las reglas
estadísticas del motor A. Los calendarios históricos aquí se declaran desarrollo
expuesto; no son una reserva final ni confirmación independiente.

La distribución tiene **cero admisiones reales y cero ajustes de mercado**.
La instalación verifica fuentes y software sobre datos artificiales. Adquisición,
planificación, cierre y apertura final tienen activaciones separadas desactivadas.
No se requiere Nasdaq, Drive ni una GPU concreta. La clave GitHub y la clave
independiente de cifrado se solicitan sólo al activar operaciones remotas.

La adquisición usa archivos públicos Binance Vision con licencia y checksum
verificados; la declaración de uso/exposición procede del autor real. Guarda
fuera de Colab la clave independiente de cifrado y los documentos auténticos.
Las observaciones raw permanecen privadas; GitHub conserva metadatos y estado
cifrado del caso. Las simulaciones y las pruebas no acreditan R3.
'''
            if source.startswith('## 2.'):
                source = '''## 2. Declaración real y preparación de datos de desarrollo

La celda crea únicamente metadatos con `PREPARE_CRYPTO_DEVELOPMENT=False`.
Para descargar, activa esa única bandera y ejecuta la celda: pedirá tu nombre,
contacto, función y aceptación expresa del uso académico/no comercial, los
términos de los derivados y la designación de todo el histórico como desarrollo
expuesto. No exige editar cuatro JSON ni crea una firma o una reserva final.

F00: BTCUSDT Spot; F01: BTCUSDT USD-M perpetuo; F02: BTCUSD_PERP COIN-M;
F03: ETHUSDT USD-M perpetuo. Cada fuente conserva identidad distinta, volumen
base correcto y calendario diario UTC. Los conjuntos comparten factores de
mercado: diversidad de contrato y tamaño no significa independencia económica.
Una sustitución debe versionarse antes de nuevos resultados.
'''
            if source.startswith('## 4.'):
                source += '''
Presupuesto de planificación completo: **340 trabajos**: 320 ajustes neuronales
(16 redes ×5 semillas ×4 paneles), cuatro ARIMA y 16 controles analíticos.
Los 120s por ajuste neuronal son un presupuesto nominal acumulado de **10h40**;
no incluyen ARIMA, adquisición, E/S, instalación o GitHub ni garantizan el tiempo
histórico tras interrupciones. Las cinco semillas de planificación pertenecen a
la cola congelada del banco; no sustituyen el mínimo de diez semillas finales.
Antes de activar se pide además escribir `PLANIFICAR 340`. Un fallo de respaldo
impide comenzar nuevos ajustes y conserva el estado local; reanudar omite sólo
trabajos completos verificados y restaura checkpoints/Adam/RNG.
'''
                source += '''
El dimensionamiento actual estudia el estimando MSE temporal condicionado.
Aunque complete su contrato, esta versión **no habilita confirmación**: quedan
por establecer la alineación inferencial con la media de RMSE individuales
heredada y el control de la composición conjunta principal+adicionales. No se
cambia automáticamente la métrica, α o la regla a partir de esa auditoría.
'''
            if any(source.startswith('## '+str(i)+'.') for i in (5,6,7)):
                source += '''
**Confirmación no habilitada por esta versión.** Las celdas y la CLI bloquean
congelamiento, entrenamiento/sello final y apertura antes de pedir credenciales
o acceder a test, con `PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED` y
`JOINT_DECISION_VALIDATION_NOT_ESTABLISHED`. Las instrucciones históricas que
anteceden describen contratos de una ruta futura, no una activación disponible.
Se requiere una versión nueva justificada y validada antes de resultados finales.
'''
            source = source.replace('WIKI no se convierte\nen reserva futura.',
                                    'El histórico expuesto no se convierte\nen reserva futura por volver a descargarlo.')
        cell['source'] = source
        cell['id'] = hashlib.sha256((cell['cell_type']+source).encode()).hexdigest()[:12]
    result['metadata']['colab']['name']='E3_Hipotesis_Adecuacion_GitHub.ipynb'
    result['metadata']['e3_data_extension_version']='crypto-contract-admission-v2'
    result['metadata']['e3_historical_source_A_commit']='c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c'
    result['metadata']['e3_account_Nasdaq_required']=False
    result['metadata']['e3_final_confirmation_enabled_in_this_version']=False
    return result


def write_notebook(output, **pins):
    result = notebook(**pins)
    headings = [c for c in result['cells'] if c['cell_type']=='markdown' and re.match(r'## [1-8]\. ',c['source'])]
    assert len(headings)==8
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=1,ensure_ascii=False,allow_nan=False)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--starter-output',type=Path)
    parser.add_argument('--source-commit',default=SOURCE_COMMIT)
    parser.add_argument('--manifest-sha256',default=MANIFEST_SHA256)
    parser.add_argument('--bootstrap-sha256',default=BOOTSTRAP_SHA256)
    args=parser.parse_args()
    result=notebook(args.source_commit,args.manifest_sha256,args.bootstrap_sha256)
    write_notebook(args.output,source_commit=args.source_commit,
                   manifest_sha256=args.manifest_sha256,bootstrap_sha256=args.bootstrap_sha256)
    if args.starter_output:
        args.starter_output.parent.mkdir(parents=True,exist_ok=True)
        args.starter_output.write_text(next(c['source'] for c in result['cells'] if c['cell_type']=='code'))
    print(json.dumps({'notebook':str(args.output),'stages':8,'market_training_executed':False,'test_opened':False}))


if __name__=='__main__':
    main()
