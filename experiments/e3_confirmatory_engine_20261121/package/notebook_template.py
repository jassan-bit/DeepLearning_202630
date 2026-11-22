"""Render the eight operational Colab stages with immutable source pins.

Pure standard-library JSON: no notebook package, market acquisition, training,
test preparation or remote write is performed by generating this artifact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

SOURCE_COMMIT = "__SOURCE_COMMIT__"
MANIFEST_SHA256 = "__MANIFEST_SHA256__"
BOOTSTRAP_SHA256 = "__BOOTSTRAP_SHA256__"


def _cell(kind, source):
    cell = {"cell_type": kind, "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n"}
    cell["id"] = hashlib.sha256((kind+cell["source"]).encode()).hexdigest()[:12]
    if kind == "code":
        ast.parse(cell["source"])
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(source_commit=SOURCE_COMMIT, manifest_sha256=MANIFEST_SHA256,
             bootstrap_sha256=BOOTSTRAP_SHA256):
    for value, pattern, token in ((source_commit, r"[0-9a-f]{40}", SOURCE_COMMIT),
                                  (manifest_sha256, r"[0-9a-f]{64}", MANIFEST_SHA256),
                                  (bootstrap_sha256, r"[0-9a-f]{64}", BOOTSTRAP_SHA256)):
        if value != token and not re.fullmatch(pattern, value):
            raise ValueError("An immutable hexadecimal source pin is required")
    cells = []
    md = lambda value: cells.append(_cell("markdown", value))
    code = lambda value: cells.append(_cell("code", value))
    md("""
    # E3 · motor temporal y respaldo GitHub

    Cuaderno operativo de ocho etapas. Candidata `input_only_mse`, comparación
    principal `global_mse`, familia congelada de 21 sistemas. La edición
    distribuida tiene **cero paneles admitidos y cero ajustes de mercado**.
    La etapa 3 imprime el dictamen temporal vigente y conserva sus límites.
    No declara R3 acreditada. El cierre
    final exige evidencia válida nueva; ninguna celda cambia un fallo por una
    etiqueta de aprobación.

    Adquisición, entrenamiento, preparación y apertura de test permanecen
    desactivados. Activar sólo la etapa autorizada, en orden. Guardar fuera de
    Colab la clave independiente de cifrado y los documentos auténticos: GitHub
    conserva checkpoints cifrados y derivados permitidos, no los arrays raw.
    """)
    md("""
    ## 1. Instalación aislada y fuentes fijadas

    Verifica el bootstrap por SHA256 antes de ejecutarlo y después todos los
    archivos del manifest del commit. Las pruebas incluidas hacen pequeños
    pasos de optimizador sobre números sintéticos; no usan mercado ni reserva.
    """)
    code(f"""
    from pathlib import Path
    import hashlib, importlib.util, json, re, subprocess, urllib.request

    SOURCE_COMMIT = {source_commit!r}
    MANIFEST_SHA256 = {manifest_sha256!r}
    BOOTSTRAP_SHA256 = {bootstrap_sha256!r}
    assert re.fullmatch(r"[0-9a-f]{{40}}", SOURCE_COMMIT), "Usar la edición publicada con pins resueltos"
    assert re.fullmatch(r"[0-9a-f]{{64}}", MANIFEST_SHA256)
    assert re.fullmatch(r"[0-9a-f]{{64}}", BOOTSTRAP_SHA256)
    WORK = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "e3-r3-engine"
    WORK.mkdir(parents=True, exist_ok=True)
    url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
           + SOURCE_COMMIT + "/experiments/e3_confirmatory_engine_20261121/bootstrap_colab.py")
    with urllib.request.urlopen(url, timeout=90) as response:
        bootstrap_bytes = response.read(1024 * 1024 + 1)
    assert len(bootstrap_bytes) <= 1024 * 1024
    assert hashlib.sha256(bootstrap_bytes).hexdigest() == BOOTSTRAP_SHA256
    bootstrap_path = WORK / ("bootstrap-" + SOURCE_COMMIT + ".py")
    if bootstrap_path.exists():
        assert bootstrap_path.read_bytes() == bootstrap_bytes
    else:
        bootstrap_path.write_bytes(bootstrap_bytes)
    spec = importlib.util.spec_from_file_location("e3_pinned_bootstrap", bootstrap_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    SETTINGS = bootstrap.setup(SOURCE_COMMIT, MANIFEST_SHA256, WORK / "cache", run_tests=True)
    SOURCE = Path(SETTINGS["source_root"])
    PRIVATE = WORK / "private"
    PRIVATE.mkdir(exist_ok=True)
    print(json.dumps({{"source_commit": SOURCE_COMMIT, "manifest_sha256": MANIFEST_SHA256,
                      "source_root": str(SOURCE), "new_market_fits": 0, "test_opened": False}}))
    """)
    code(r'''
    import getpass

    PANELS = ("F00", "F01", "F02", "F03")
    REPOSITORY = "jassan-bit/DeepLearning_202630"
    RUN_LABEL = "author-run-01"  # Elegir una identidad nueva sólo antes de preparar el caso.
    PLANNING_CASE = WORK / "cases" / ("planning-" + RUN_LABEL)
    FINAL_CASE = WORK / "cases" / ("final-" + RUN_LABEL)
    PLANNING_BRANCH = "e3-r3-engine-planning-" + RUN_LABEL
    FINAL_BRANCH = "e3-r3-engine-final-" + RUN_LABEL
    PLANNING_PREFIX = "runs/e3_r3_engine/planning-" + RUN_LABEL
    FINAL_PREFIX = "runs/e3_r3_engine/final-" + RUN_LABEL
    assert PLANNING_BRANCH != FINAL_BRANCH and PLANNING_PREFIX != FINAL_PREFIX
    DEVELOPMENT_MAP = PRIVATE / "development-panels.json"
    FINAL_MAP = PRIVATE / "released-test-panels.json"
    POWER_RECEIPT = PRIVATE / "power-receipt.json"
    FINAL_PROTOCOL = PRIVATE / "protocol-final.json"

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def read_json(path):
        return json.loads(Path(path).read_text())

    def create_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return path

    def private_value(secret_name, prompt):
        try:
            from google.colab import userdata
            value = userdata.get(secret_name)
        except Exception:
            value = None
        return value or getpass.getpass(prompt)

    def github_credentials():
        return {"token": private_value("E3_R3_GITHUB_TOKEN", "GitHub Contents read/write token: "),
                "backup_secret": private_value("E3_R3_BACKUP_SECRET", "Clave independiente >=32 caracteres (conservar): ")}

    def remote_args(case, *, final=False, panels=None):
        args = ["--case", str(case), "--source-commit", SOURCE_COMMIT,
                "--repository", REPOSITORY,
                "--branch", FINAL_BRANCH if final else PLANNING_BRANCH,
                "--prefix", FINAL_PREFIX if final else PLANNING_PREFIX]
        if panels is not None:
            args += ["--panels", str(panels)]
        return args

    def run_cli(args, *, remote=False, allow_blocked=False):
        credentials = github_credentials() if remote else None
        try:
            return bootstrap.command(SETTINGS, args, credentials=credentials, allow_blocked=allow_blocked)
        finally:
            if credentials is not None:
                credentials.clear()

    def run_data(args, *, provider=None):
        env = bootstrap.cpu_env()
        for key in ("NASDAQ_DATA_LINK_API_KEY", "ALPHAVANTAGE_API_KEY"):
            env.pop(key, None)
        if provider == "nasdaq_wiki":
            env["NASDAQ_DATA_LINK_API_KEY"] = private_value("NASDAQ_DATA_LINK_API_KEY", "Nasdaq Data Link API key: ")
        elif provider == "alphavantage":
            env["ALPHAVANTAGE_API_KEY"] = private_value("ALPHAVANTAGE_API_KEY", "Alpha Vantage API key: ")
        try:
            result = subprocess.run([SETTINGS["python"], "-m", "engine.data", *args],
                                    cwd=SOURCE, env=env, text=True, capture_output=True)
            if result.stdout:
                print(result.stdout, end="")
            if result.returncode:
                raise RuntimeError("Admisión detenida; revisar formularios/recibos y el ledger local. No imprimir credenciales.")
            return result
        finally:
            for key in ("NASDAQ_DATA_LINK_API_KEY", "ALPHAVANTAGE_API_KEY"):
                env.pop(key, None)

    def run_private_program(program, payload):
        result = subprocess.run([SETTINGS["python"], "-c", program], cwd=SOURCE,
                                env=bootstrap.cpu_env(), text=True,
                                input=json.dumps(payload), capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError("La verificación/preparación local se detuvo; conservar evidencia y revisar las entradas auténticas.")
        return result
    ''')
    md("""
    ## 2. Evidencia auténtica, adquisición y admisión de desarrollo

    El autor completa cada formulario y aporta bytes reales de evidencia de
    licencia/exposición. El modo `development_only` declara todo ese dataset
    expuesto; no exige una reserva ni un protocolo final. No convierte esos
    datos en confirmación. Los recibos de derechos distinguen publicación de
    derivados y redistribución raw. No se firma ni aprueba por el autor.

    Propuestas vigentes: BTCUSDT/Binance Vision; IBM, MSFT y XOM/WIKI. WIKI
    terminó en 2018 y no permite una futura reserva posterior al protocolo.
    Una sustitución necesaria debe fijarse antes de observar nuevos resultados.
    """)
    code(r'''
    RECORD_OWNER_EVIDENCE = False
    ACQUIRE_MARKET_DEVELOPMENT = False
    PREPARE_MARKET_DEVELOPMENT = False

    source_form = read_json(SOURCE / "docs/OWNER_EXPOSURE_FORM.json")
    source_panels = {p["id"]: p for p in read_json(SOURCE / "configs/data_panels.json")["panels"]}
    split_rule = {"version": "development-calendar-proportions-1",
                  "fit_fraction": 0.60, "selection_fraction": 0.10,
                  "planning_fraction": 0.20, "calibration_fraction": 0.10,
                  "embargo_bars": 0,
                  "basis": "expected exchange sessions only; no market prices, labels or performance used",
                  "support_rule": "raw disjoint bounds; engine purges endpoints until all 15 closes fit the assigned support"}
    create_json(PRIVATE / "development-split-rule.json", split_rule)
    definitions = []
    for panel in PANELS:
        proposal = source_panels[panel]
        folder = PRIVATE / "inputs" / panel
        folder.mkdir(parents=True, exist_ok=True)
        license_evidence = SOURCE / proposal["rights_snapshot_path"]
        assert license_evidence.is_file() and sha(license_evidence) == proposal["rights_snapshot_sha256"]
        form = json.loads(json.dumps(source_form))
        form.update(panel_ids=[panel], reserve_mode="development_only",
                    source_url=proposal["rights_source_url"],
                    source_license_evidence_path=str(license_evidence),
                    source_license_evidence_sha256=sha(license_evidence),
                    dataset_scope={"panel": panel, "provider": proposal["provider"],
                                   "instrument": proposal["instrument"],
                                   "start": proposal["proposed_development_start"],
                                   "end": proposal["proposed_development_end"]})
        create_json(folder / "owner-development.json", form)
        create_json(folder / "rights-DRAFT.json", {
            "status": "requires_real_actor", "actor_name": None, "actor_contact": None,
            "evidence_path": str(license_evidence), "evidence_sha256": sha(license_evidence),
            "scope": "academic noncommercial research: development, planning, final evaluation and permitted derived outputs; actor must verify source conditions",
            "source_url": proposal["rights_source_url"], "license": proposal["license"],
            "allow_derived_publication": proposal["allow_derived_publication"],
            "allow_raw_redistribution": proposal["allow_raw_redistribution"]})
        create_json(folder / "acquisition.json", {
            "provider": proposal["provider"], "instrument": proposal["instrument"],
            "development_start": proposal["proposed_development_start"],
            "development_end": proposal["proposed_development_end"]})
        definitions.append({"panel": panel, "folder": folder,
                            "proposal": proposal,
                            "raw": PRIVATE / "raw" / panel / "development-v1",
                            "panel_dir": PRIVATE / "panels" / panel / "development-v1"})
    create_json(DEVELOPMENT_MAP, {d["panel"]: str(d["panel_dir"]) for d in definitions})
    split_program = """
    import json, sys
    from pathlib import Path
    from engine.data import expected_sessions
    payload = json.load(sys.stdin)
    rule = payload['rule']
    assert abs(sum(rule[k] for k in ('fit_fraction','selection_fraction','planning_fraction','calibration_fraction'))-1.0) < 1e-12
    for entry in payload['panels']:
        path = Path(entry['folder'])/'development-split.json'
        proposal = entry['proposal']
        n = len(expected_sessions(proposal['proposed_development_start'],proposal['proposed_development_end'],proposal['calendar']))
        ends = [int(n*.60),int(n*.70),int(n*.90),n]
        starts = [0]+ends[:-1]
        bounds = {name:[lo,hi-1] for name,lo,hi in zip(('fit','selection','planning','calibration'),starts,ends)}
        assert all(hi-lo+1 >= 30 for lo,hi in bounds.values()), 'Proposed development range too short'
        assert bounds['planning'][1]-bounds['planning'][0]+1-14 >= 80, 'Planning requires at least 80 usable dates'
        obj = {'raw_start':proposal['proposed_development_start'],'raw_end':proposal['proposed_development_end'],
               'calendar':proposal['calendar'],'raw_bounds':bounds,'embargo_bars':rule['embargo_bars'],
               'prospective_split_rule':rule,'metadata_only_expected_raw_bars':n}
        if not path.exists():
            path.write_text(json.dumps(obj,indent=2)+'\\n')
    print(json.dumps({'status':'metadata_only_development_splits_prepared','no_market_observations_read':True}))
    """
    run_private_program(split_program, {"rule": split_rule,
        "panels": [{"folder": str(d["folder"]), "proposal": d["proposal"]} for d in definitions]})
    print("Adquisición y splits por calendario preparados. Completar owner-development.json con datos reales del actor y copiar rights-DRAFT.json a rights.json después de verificar/aprobar su alcance.")
    print("Esquemas y campos exactos:", str(SOURCE / "docs/DATA_ADMISSION.md"))

    if RECORD_OWNER_EVIDENCE:
        for d in definitions:
            folder = d["folder"]
            evidence = folder / "owner-development.json"  # Declaración suministrada realmente por el actor.
            assert not (folder / "exposure-development.json").exists(), "Recibo inmutable: usar su versión existente o un directorio nuevo"
            run_data(["record-owner-evidence", "--form", str(folder / "owner-development.json"),
                      "--evidence", str(evidence), "--output", str(folder / "exposure-development.json")])
    if ACQUIRE_MARKET_DEVELOPMENT:
        for d in definitions:
            folder = d["folder"]
            config = read_json(folder / "acquisition.json")
            run_data(["collect-development", "--config", str(folder / "acquisition.json"),
                      "--rights", str(folder / "rights.json"), "--exposure", str(folder / "exposure-development.json"),
                      "--output", str(d["raw"]), "--enable-market-acquisition"], provider=config["provider"])
    if PREPARE_MARKET_DEVELOPMENT:
        for d in definitions:
            folder = d["folder"]
            run_data(["prepare-development", "--manifest", str(d["raw"] / "raw_manifest.json"),
                      "--split-config", str(folder / "development-split.json"),
                      "--rights", str(folder / "rights.json"), "--exposure", str(folder / "exposure-development.json"),
                      "--output", str(d["panel_dir"])])
    ''')
    md("""
    ## 3. Verificación de fuentes y particiones sin entrenamiento

    Comprueba exactamente ajuste, selección, planificación y calibración.
    El controlador rechaza fixtures para campañas reales y prueba en esta fase.
    No se calculan métricas para decidir arquitectura o fuentes.
    """)
    code(r'''
    run_cli(["check"])
    VERIFY_DEVELOPMENT = True
    verification_program = """
    import json, sys
    from engine import data
    panels = json.load(sys.stdin)
    result = {}
    for panel, directory in panels.items():
        admission = data.read_admission(directory)
        assert admission['status'] == 'development_admitted', 'A software fixture is not a market admission'
        assert set(admission['partition_paths']) == {'fit','selection','planning','calibration'}
        for part in ('fit','selection','planning','calibration'):
            data.load_partition(directory, part)
        result[panel] = {'admission_sha256':admission['admission_sha256'], 'status':admission['status']}
    print(json.dumps({'status':'development_integrity_verified_no_fit_no_test','panels':result}))
    """
    mapped = read_json(DEVELOPMENT_MAP)
    missing = [panel for panel, path in mapped.items() if not (Path(path) / "admission.json").is_file()]
    if VERIFY_DEVELOPMENT and not missing:
        result = subprocess.run([SETTINGS["python"], "-c", verification_program], cwd=SOURCE,
                                env=bootstrap.cpu_env(), text=True, input=json.dumps(mapped), capture_output=True)
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError("La admisión no superó la verificación; conservar evidencia y corregir su causa")
    else:
        print(json.dumps({"status": "awaiting_real_development_admissions", "panels_missing": missing,
                          "new_market_fits": 0, "test_opened": False}))
    ''')
    md("""
    ## 4. Piloto de planificación, predicciones y dimensionamiento

    Activación explícita: 340 trabajos, con las cinco últimas semillas del banco
    congelado; los sistemas deterministas aparecen una vez por panel. El
    controlador verifica GitHub antes del primer paso y en cada checkpoint.
    Repetir `run` reanuda incompletos y omite sólo completos verificados.
    `dimension` utiliza predicciones de planificación, nunca prueba. Un estado
    `NOT_DIMENSIONED` conserva R/T/B nulos y no habilita la etapa siguiente.
    """)
    code(r'''
    RUN_PLANNING_TRAINING = False
    EXPORT_PLANNING_PREDICTIONS = False
    RUN_DIMENSIONING = False
    POWER_DESIGN = SOURCE / "configs/power_design.json"

    if RUN_PLANNING_TRAINING:
        assert not missing, "Completar primero las cuatro admisiones"
        if not (PLANNING_CASE / "plan.json").exists():
            run_cli(["prepare", *remote_args(PLANNING_CASE, panels=DEVELOPMENT_MAP),
                     "--stage", "planning", "--protocol", str(SOURCE / "configs/protocol.json")])
        run_cli(["run", *remote_args(PLANNING_CASE, panels=DEVELOPMENT_MAP), "--activate"], remote=True)
    if EXPORT_PLANNING_PREDICTIONS:
        run_cli(["predict-planning", *remote_args(PLANNING_CASE, panels=DEVELOPMENT_MAP)], remote=True)
    PLANNING_PREDICTIONS = PLANNING_CASE / "predictions/planning_predictions.csv"
    PLANNING_RECEIPT = PLANNING_CASE / "predictions/planning_predictions.receipt.json"
    if RUN_DIMENSIONING:
        assert PLANNING_PREDICTIONS.is_file() and PLANNING_RECEIPT.is_file()
        run_cli(["dimension", "--design", str(POWER_DESIGN),
                 "--predictions", str(PLANNING_PREDICTIONS), "--receipts", str(PLANNING_RECEIPT),
                 "--output", str(POWER_RECEIPT)], allow_blocked=True)
    else:
        print("Planificación/dimensionamiento no ejecutados. No se inventan R, T ni B finales.")
    ''')
    md("""
    ## 5. Cierre prospectivo de protocolo y presupuesto

    Requiere potencia real admitida, validación temporal de alcance completo,
    calendarios reservados auténticos y reglas de adecuación/estrés fijadas.
    El freeze verifica esos contratos; un informe fallido o inconcluso no los
    satisface. El test-spec F00–F03 fija fechas, conteo T, `reserve_mode`,
    `source_identity_sha256` y `exposure_evidence_sha256`. Para historia se
    necesita custodia auténtica; para recolección futura, todo el soporte debe
    ser posterior al cierre completo. No completar estos datos por conjetura.

    Se conservan margen, familias y recetas. La CLI enlaza adecuación y escenarios
    de estrés antes de producir el hash final. Un cambio posterior de fuente,
    escenario, magnitud o regla exige otro protocolo antes de nuevos resultados.
    """)
    code(r'''
    FREEZE_FINAL_PROTOCOL = False
    BASE_PROTOCOL = PRIVATE / "protocol-prospective-base.json"
    TEST_SPEC = PRIVATE / "authentic-test-spec.json"
    method_binding = read_json(SOURCE / "configs/protocol.json")["method_validation"]
    METHOD_REPORT = SOURCE / method_binding["path"]
    create_json(BASE_PROTOCOL, read_json(SOURCE / "configs/protocol.json"))
    source_identity_program = """
    import json, sys
    from engine.data import read_admission
    from engine.controller import source_identity_sha256
    print(json.dumps({pid:source_identity_sha256(read_admission(path)['provenance']) for pid,path in json.load(sys.stdin).items()}))
    """
    source_ids = {}
    if not missing:
        source_ids = json.loads(run_private_program(source_identity_program, read_json(DEVELOPMENT_MAP)).stdout.splitlines()[-1])
    proposed_T = read_json(POWER_RECEIPT).get("final_T") if POWER_RECEIPT.is_file() else None
    draft_spec = {}
    for panel in PANELS:
        inventory = PRIVATE / "inputs" / panel / "reserve-inventory.json"
        draft_spec[panel] = {"start": None, "end": None, "count": proposed_T,
                             "reserve_mode": None,
                             "source_identity_sha256": source_ids.get(panel),
                             "exposure_evidence_sha256": sha(inventory) if inventory.is_file() else None}
    draft_key = hashlib.sha256(json.dumps(draft_spec,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()[:12]
    TEST_SPEC_DRAFT = PRIVATE / ("authentic-test-spec-DRAFT-" + draft_key + ".json")
    create_json(TEST_SPEC_DRAFT, draft_spec)
    print("Plantilla versionada de reserva (completar sólo con evidencia auténtica):", str(TEST_SPEC_DRAFT))
    if FREEZE_FINAL_PROTOCOL:
        assert METHOD_REPORT.is_file() and sha(METHOD_REPORT) == method_binding["sha256"]
        assert POWER_RECEIPT.is_file() and TEST_SPEC.is_file()
        assert read_json(POWER_RECEIPT).get("status") == "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP"
        assert not (FINAL_CASE / "evaluation_receipt.json").exists(), "No se cambia protocolo tras abrir prueba"
        for panel, target in read_json(TEST_SPEC).items():
            inventory = PRIVATE / "inputs" / panel / "reserve-inventory.json"
            assert target["source_identity_sha256"] == source_ids[panel]
            assert inventory.is_file() and target["exposure_evidence_sha256"] == sha(inventory)
        run_cli(["freeze-protocol", "--base", str(BASE_PROTOCOL), "--power", str(POWER_RECEIPT),
                 "--method-report", str(METHOD_REPORT), "--test-spec", str(TEST_SPEC), "--output", str(FINAL_PROTOCOL)])
        print(json.dumps({"final_protocol_sha256": sha(FINAL_PROTOCOL),
                          "R_T_B": [read_json(FINAL_PROTOCOL)["final"][k] for k in ("R", "T", "B")]}))
    else:
        print("Cierre final desactivado; aportar el bundle temporal admitido, potencia y reserva auténtica.")
    ''')
    md("""
    ## 6. Entrenamiento final completo, reanudación y sello

    Usa un caso y rama diferentes del piloto y el prefijo de R semillas del
    banco original. El total es `64*R + 20` trabajos; se informa presupuesto
    nominal, costes medidos y cómputo incierto por separado. El sello verifica
    todos los mejores checkpoints y ajusta calibradores sólo en calibración;
    guarda y relee GitHub **antes** de construir o abrir test.
    """)
    code(r'''
    RUN_FINAL_TRAINING = False
    SEAL_FINAL_TRAINING = False

    if RUN_FINAL_TRAINING:
        assert FINAL_PROTOCOL.is_file() and read_json(FINAL_PROTOCOL)["final"]["closed"] is True
        if not (FINAL_CASE / "plan.json").exists():
            run_cli(["prepare", *remote_args(FINAL_CASE, final=True, panels=DEVELOPMENT_MAP),
                     "--stage", "final", "--protocol", str(FINAL_PROTOCOL), "--power-receipt", str(POWER_RECEIPT)])
        run_cli(["run", *remote_args(FINAL_CASE, final=True, panels=DEVELOPMENT_MAP), "--activate"], remote=True)
    if SEAL_FINAL_TRAINING:
        run_cli(["seal-training", *remote_args(FINAL_CASE, final=True, panels=DEVELOPMENT_MAP)], remote=True)
        seal = read_json(FINAL_CASE / "training_seal.json")
        selected = FINAL_CASE / "selected_manifest.json"
        final_split_program = """
    import json, sys, hashlib
    from pathlib import Path
    from engine.data import read_admission, expected_sessions
    from engine.controller import source_identity_sha256
    payload = json.load(sys.stdin)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def create(path,obj):
        if not path.exists():
            path.write_text(json.dumps(obj,indent=2)+'\\n')
    protocol_path=Path(payload['protocol'])
    protocol=json.loads(protocol_path.read_text())
    original_form=json.loads(Path(payload['form']).read_text())
    for pid,directory in payload['panels'].items():
        admission=read_admission(directory)
        provenance=admission['provenance']
        target=protocol['final']['test_panels'][pid]
        assert source_identity_sha256(provenance)==target['source_identity_sha256']
        folder=Path(payload['inputs'])/pid
        inventory=folder/'reserve-inventory.json'
        assert inventory.is_file() and sha(inventory)==target['exposure_evidence_sha256']
        origin=provenance['bar_index_origin']
        sessions=expected_sessions(origin,target['end'],provenance['calendar'])
        first,last=sessions.index(target['start']),sessions.index(target['end'])
        assert first>=14 and first<=last
        split={'protocol_sha256':sha(protocol_path),'raw_start':origin,'raw_end':target['end'],
               'target_start':target['start'],'target_end':target['end'],'calendar':provenance['calendar'],
               'raw_bounds':{'test':[first-14,last]},'embargo_bars':0,'fixed_test_count':target['count'],
               'metadata_only_target_sessions':last-first+1,
               'raw_manifest_requirement':'full unchanged historical prefix from development bar_index_origin plus the authorized reserved extension'}
        authorization={'status':'requires_real_actor','actor_name':None,'actor_contact':None,
               'evidence_path':None,'evidence_sha256':None,'authorize_test_preparation':False,
               'sources_methods_size_frozen':True,'protocol_path':str(protocol_path),'protocol_sha256':sha(protocol_path),
               'selected_checkpoints_manifest_path':payload['selected'],
               'selected_checkpoints_sha256':sha(payload['selected']),
               'training_seal_sha256':sha(payload['seal']),
               'development_bar_index_origin':origin,
               'reserve_inventory':{'path':str(inventory),'sha256':sha(inventory)}}
        form=json.loads(json.dumps(original_form))
        form.update(panel_ids=[pid],protocol_sha256=sha(protocol_path),
              exposure_inventory_sha256=sha(inventory),reserve_mode=target['reserve_mode'],
              protocol_lock_utc=protocol.get('closed_at_utc'),first_raw_support_session=sessions[first-14],
              test_first_target_session=target['start'],test_last_target_session=target['end'],
              fixed_test_count=target['count'])
        create(folder/'final-split-DRAFT.json',split)
        create(folder/'final-authorization-DRAFT.json',authorization)
        create(folder/'owner-final-DRAFT.json',form)
    print(json.dumps({'status':'final_drafts_from_frozen_metadata','test_observations_read':False,'authorization_granted':False}))
        """
        run_private_program(final_split_program, {"protocol": str(FINAL_PROTOCOL), "panels": read_json(DEVELOPMENT_MAP),
             "inputs": str(PRIVATE / "inputs"), "selected": str(selected), "seal": str(FINAL_CASE / "training_seal.json"),
             "form": str(SOURCE / "docs/OWNER_EXPOSURE_FORM.json")})
        print("Sello verificado. El autor/custodio debe aportar ahora autorización y raw manifest reservado, sin publicar arrays.")
    ''')
    md("""
    ## 7. Preparación autorizada, evaluación única, assessment y exportación

    Sólo después del sello: el custodio entrega manifest raw `reserved_final`,
    exposición, derechos y autorización reales. No existe un colector genérico
    que pueda inventar custodia; WIKI no se convierte en reserva futura.
    `prepare-final` verifica bytes/autorización antes de construir etiquetas en
    directorios nuevos. Conserva el origen de índices global de barras: empezar
    de cero no demuestra purga frente a desarrollo.

    `evaluate` guarda un recibo remoto antes de consultar prueba e incluye los
    escenarios de ruido y faltantes **prefijados**, con pendientes/caches por
    trabajo y condición. Si una consulta queda incierta no se repite
    automáticamente. El resultado completo se recupera/exporta sin nuevos
    forwards. Assessment consume CSV guardados y conserva también fallos,
    inferioridad y requisitos no satisfechos; no cambia la candidata.
    """)
    code(r'''
    REGISTER_FINAL_EXPOSURE = False
    PREPARE_RESERVED_TEST = False
    OPEN_RESERVED_EVALUATION = False
    ASSESS_SAVED_RESULTS = False
    SYNC_SAVED_RESULTS = False
    EXPORT_SAVED_RESULTS = False
    EXPORT_ZIP = WORK / "exports" / ("final-" + RUN_LABEL + ".zip")
    released = {panel: str(PRIVATE / "panels" / panel / "released-test-v1") for panel in PANELS}
    create_json(FINAL_MAP, released)

    if REGISTER_FINAL_EXPOSURE or PREPARE_RESERVED_TEST or OPEN_RESERVED_EVALUATION:
        assert (FINAL_CASE / "training_seal.json").is_file(), "Primero completar y verificar el sello remoto"
        seal = read_json(FINAL_CASE / "training_seal.json")
        assert seal["protocol_sha256"] == sha(FINAL_PROTOCOL)
        assert seal["selected_manifest_sha256"] == sha(FINAL_CASE / "selected_manifest.json")
    if REGISTER_FINAL_EXPOSURE:
        for panel in PANELS:
            folder = PRIVATE / "inputs" / panel
            run_data(["record-owner-evidence", "--form", str(folder / "owner-final.json"),
                      "--evidence", str(folder / "actual-custodian-evidence.bin"),
                      "--output", str(folder / "exposure-final.json")])
    if PREPARE_RESERVED_TEST:
        for panel in PANELS:
            folder = PRIVATE / "inputs" / panel
            authorization = read_json(folder / "final-authorization.json")
            assert authorization["status"] == "authorized" and authorization["authorize_test_preparation"] is True
            assert authorization["protocol_sha256"] == sha(FINAL_PROTOCOL)
            assert authorization["selected_checkpoints_sha256"] == seal["selected_manifest_sha256"]
            run_data(["prepare-final", "--manifest", str(PRIVATE / "raw" / panel / "reserved-final" / "raw_manifest.json"),
                      "--split-config", str(folder / "final-split.json"), "--rights", str(folder / "rights.json"),
                      "--exposure", str(folder / "exposure-final.json"),
                      "--authorization", str(folder / "final-authorization.json"),
                      "--output", read_json(FINAL_MAP)[panel], "--enable-final-preparation"])
    if OPEN_RESERVED_EVALUATION:
        run_cli(["evaluate", *remote_args(FINAL_CASE, final=True, panels=FINAL_MAP), "--activate"], remote=True)
    if ASSESS_SAVED_RESULTS:
        assert read_json(FINAL_CASE / "evaluation_receipt.json")["status"] == "completed"
        run_cli(["assess", "--case", str(FINAL_CASE), "--protocol", str(FINAL_PROTOCOL)])
    if ASSESS_SAVED_RESULTS or SYNC_SAVED_RESULTS:
        run_cli(["sync", *remote_args(FINAL_CASE, final=True)], remote=True)
    if EXPORT_SAVED_RESULTS:
        EXPORT_ZIP.parent.mkdir(parents=True, exist_ok=True)
        run_cli(["export", "--case", str(FINAL_CASE), "--output", str(EXPORT_ZIP)])
        print(json.dumps({"export": str(EXPORT_ZIP), "sha256": sha(EXPORT_ZIP)}))
    ''')
    md("""
    ## 8. Estado remoto verificado y recuperación sin Drive

    La verificación remota autentica y comprueba los fragmentos cifrados y sus
    hashes; un listado local no la sustituye. Las credenciales viajan por stdin
    JSON privado al hijo, nunca por argumentos, shell, celdas guardadas o GitHub.
    Usar Colab Secrets `E3_R3_GITHUB_TOKEN` y `E3_R3_BACKUP_SECRET`, o entrada
    oculta. Restaurar a un directorio nuevo: después reacquirir los mismos datos
    privados y verificar hashes; el backup no los contiene. La rama debe ser
    exactamente la de esa campaña. Los errores de permisos, cuotas y red
    conservan su clasificación y detienen nuevos ajustes.
    """)
    code(r'''
    VERIFY_REMOTE_GITHUB = False
    RESTORE_FROM_GITHUB = False
    TARGET_STAGE = "planning"  # Cambiar a "final" para esa campaña, sin mezclar ramas.
    assert TARGET_STAGE in ("planning", "final")
    target_final = TARGET_STAGE == "final"
    target_case = FINAL_CASE if target_final else PLANNING_CASE
    RESTORE_DESTINATION = WORK / "restored" / (TARGET_STAGE + "-" + RUN_LABEL)

    if (target_case / "plan.json").exists():
        run_cli(["status", *remote_args(target_case, final=target_final)])
    else:
        print("El caso local aún no está preparado.")
    if VERIFY_REMOTE_GITHUB:
        remote_program = """
    import json, sys
    from engine.storage import GithubStore
    payload = json.load(sys.stdin)
    auth = payload.pop('credentials')
    try:
        store = GithubStore(auth['token'],auth['backup_secret'],**payload)
        result = store.status()
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'remote_verification_stopped','error_type':type(error).__name__}))
        raise SystemExit(2)
    finally:
        auth.clear()
        """
        auth = github_credentials()
        payload = {"credentials": auth, "repository": REPOSITORY,
                   "branch": FINAL_BRANCH if target_final else PLANNING_BRANCH,
                   "source_commit": SOURCE_COMMIT, "prefix": FINAL_PREFIX if target_final else PLANNING_PREFIX}
        try:
            result = subprocess.run([SETTINGS["python"], "-c", remote_program], cwd=SOURCE,
                                    env=bootstrap.cpu_env(), text=True, input=json.dumps(payload), capture_output=True)
            print(result.stdout, end="")
            if result.returncode:
                raise RuntimeError("La verificación remota se detuvo; no comenzar nuevos ajustes")
        finally:
            auth.clear()
            payload.clear()
    if RESTORE_FROM_GITHUB:
        assert not RESTORE_DESTINATION.exists(), "Restaurar siempre a un destino nuevo"
        run_cli(["restore", *remote_args(RESTORE_DESTINATION, final=target_final)], remote=True)
        run_cli(["status", *remote_args(RESTORE_DESTINATION, final=target_final)])
        if target_final:
            FINAL_CASE = RESTORE_DESTINATION
        else:
            PLANNING_CASE = RESTORE_DESTINATION
        print("Caso restaurado seleccionado. Reacquirir datos privados con hashes idénticos, actualizar su mapa y repetir etapa3 antes de run.")
    ''')
    md("""
    La entrega contiene implementación y fixtures de software ejecutados. La
    adquisición/admisión de mercado, piloto, dimensionamiento admitido, cierre
    final, ajustes confirmatorios y evaluación reservada siguen pendientes hasta
    satisfacer sus entradas reales. No modificar márgenes, familia, fuentes,
    semillas o escenarios después de observar resultados adversos.
    """)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "E3_R3_Temporal_GitHub.ipynb", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "e3_source_pins": {"source_commit": source_commit, "manifest_sha256": manifest_sha256,
                                            "bootstrap_sha256": bootstrap_sha256},
                         "e3_market_training_default": False, "e3_test_opening_default": False},
            "cells": cells}


def write_notebook(output, **pins):
    result = notebook(**pins)
    stage_headings = [cell for cell in result["cells"] if cell["cell_type"] == "markdown"
                      and re.match(r"## [1-8]\. ", cell["source"])]
    if len(stage_headings) != 8:
        raise ValueError("The operational notebook must contain eight stages")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/E3_R3_Temporal_GitHub.ipynb")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--bootstrap-sha256", default=BOOTSTRAP_SHA256)
    parser.add_argument("--starter-output", type=Path)
    args = parser.parse_args(argv)
    output = write_notebook(args.output, source_commit=args.source_commit,
                            manifest_sha256=args.manifest_sha256,
                            bootstrap_sha256=args.bootstrap_sha256)
    if args.starter_output:
        result = notebook(args.source_commit,args.manifest_sha256,args.bootstrap_sha256)
        starter = next(c["source"] for c in result["cells"] if c["cell_type"] == "code")
        args.starter_output.parent.mkdir(parents=True,exist_ok=True)
        args.starter_output.write_text(starter)
    print(json.dumps({"notebook": str(output), "starter":str(args.starter_output) if args.starter_output else None,
                      "stages": 8, "market_training_executed": False,
                      "test_opened": False}))


if __name__ == "__main__":
    main()
