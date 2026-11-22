"""Deterministic notebook packaging; embeds reviewed text, never credentials/data."""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
def code(text):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)}
files={name:(ROOT/name).read_bytes() for name in ('frozen_v5_reanalysis.py','colab_optional_recovery.py')}
helper_sha=hashlib.sha256(files['frozen_v5_reanalysis.py']).hexdigest()
payload={name:{'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()} for name,raw in files.items()}
config='''# Esta celda no instala dependencias ni solicita secretos.
RUN_RECOVERY = False
INSTALL_RUNTIME = False  # Sólo activar aquí si no existe ya un runtime CPU compatible.
PYTHON = None  # Ruta opcional a un Python con Torch2.8.0+cpu y NumPy2.3.5.
WORK = "/content/e3-v5-readonly-recovery"
CODE_DIR = "/content/e3-v5-readonly-code"
'''
launch='''# El código público está embebido con SHA; no depende de main ni de una URL futura.
import base64, hashlib, importlib.util, json
from pathlib import Path
PAYLOAD = '''+repr(payload)+'''
code_dir = Path(CODE_DIR)
code_dir.mkdir(parents=True, exist_ok=True)
for name, item in PAYLOAD.items():
    raw = base64.b64decode(item["base64"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    target = code_dir / name
    assert not target.is_symlink()
    if target.exists():
        assert target.read_bytes() == raw, "Código local distinto: conservarlo y usar otro CODE_DIR"
    else:
        target.write_bytes(raw)
spec = importlib.util.spec_from_file_location("optional_private_v5_launcher", code_dir / "colab_optional_recovery.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
result = launcher.execute(enabled=RUN_RECOVERY, install_runtime=INSTALL_RUNTIME, python=PYTHON,
    work=WORK, helper=code_dir / "frozen_v5_reanalysis.py",
    helper_sha256="'''+helper_sha+'''")
print(json.dumps(result, indent=2))
'''
cells=[md('''# Recuperación privada opcional de los pesos originales V5

Este cuaderno está separado del diagnóstico estadístico NumPy/SciPy. **Está desactivado por defecto.** No entrena, no usa optimizador, no calibra, no modifica respaldos y no publica pesos ni predicciones.

Objetivo: recuperar exclusivamente el respaldo cifrado del commit `05c1a637f13356dd1df6b8c5aa8f4676dda61c12`, comprobar los cinco snapshots originales `input_only_mse` y producir **2930 predicciones retrospectivas**: 532 fit + 54 selección, en cada una de cinco semillas. Antes exige reproducir los 55 pronósticos de assessment guardados y la RMSE de selección por semilla; en total realiza 3205 puntos de inferencia. Este control de paridad es obligatorio.

**Estado de entrega:** 14 verificaciones de software, con fixtures y comprobación de archivos públicos/datos ya guardados; ninguna recuperación con clave real ni inferencia real nueva ha sido ejecutada por el asistente. El cuaderno no acredita adecuación, calibración ni R3.
'''),code(config),md('''Para usarlo de forma voluntaria, activa `RUN_RECOVERY=True`. Si ya tienes un intérprete CPU compatible, escribe su ruta en `PYTHON`. La instalación aislada del lock original requiere activar también `INSTALL_RUNTIME=True`; el cuaderno estadístico principal no necesita esa instalación.

Sólo si todavía no existe el caso local se leen los secretos privados de Colab `GITHUB_TOKEN` (lectura) y **`E3_V5_BACKUP_KEY`** (la clave original usada en V5). Si no están en Colab, la celda los solicita mediante entrada oculta. No introduzcas claves en texto, URLs, código guardado, salidas ni en el chat. No sirve crear otra clave para un respaldo antiguo.

La fuente, el ZIP, el NPZ y el cifrado se fijan por commit, tamaño y SHA256. La consulta remota permite únicamente GET; no llama `setup`, `run`, `assess` ni `backup`. Usa el preprocesamiento original y sólo deserializa `training_raw`, `training_y_original` y `training_dates` del NPZ histórico; no abre lógicamente la reserva ni descarga mercado nuevo.
'''),code(launch),md('''Los CSV quedan en `WORK/fit_selection_PRIVATE`, con manifiesto SHA256 y recibo de ejecución. El caso restaurado conserva sus metadatos y pesos; la salida se escribe en otra carpeta privada. Una segunda ejecución verifica la salida existente, sin nueva inferencia, dependencias ni secretos. Una salida incompleta o distinta se conserva y bloquea la reutilización; no se sobreescribe.

El recibo declara las versiones originales y las del intérprete real. El entrenamiento usó Python3.13.16; otra versión de inferencia se permite sólo como análisis independiente con las verificaciones de paridad, sin relajar el guard de reanudación original. Una discrepancia bloquea la exportación; **no ajustes tolerancias tras observar el resultado** (`atol=1e-8`, `rtol=1e-5`).

Conservar los CSV permite estudiar posteriormente la adecuación de fit/selección, con sus límites retrospectivos. No convierte calendarios expuestos en confirmación, no construye un margen prospectivo ni genera intervalos predictivos calibrados. El soporte de átomos del modelo no constituye por sí solo un intervalo válido.

Los archivos privados viven en la VM. Si deseas conservarlos, descárgalos desde el panel de archivos a un dispositivo privado. Este cuaderno no los envía a GitHub ni al depósito público. No compartas las salidas privadas sin revisar derechos y alcance. Asistencia de IA declarada; las fuentes y los datos conservan sus licencias y términos originales.
''')]
notebook={'cells':cells,'metadata':{'colab':{'name':'E3_V5_FrozenWeights_PrivateRecovery.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
(ROOT/'E3_V5_FrozenWeights_PrivateRecovery.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print(json.dumps({'notebook':'E3_V5_FrozenWeights_PrivateRecovery.ipynb','embedded_helper_sha256':helper_sha}))
