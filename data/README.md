# Datos Binance Spot BTCUSDT

Archivos oficiales descargados para 2023-01-01–2026-10-30 inclusive, UTC, velas de un minuto. Los brutos se conservan fuera del checkout, en `/workspace/binance-spot-data/BTCUSDT/1m/monthly/` y `daily/`; no se publican los ZIP ni los CSV intradiarios. Los derivados diarios de desarrollo de 2023–2024, sus hashes y los resultados necesarios para reproducir EDA y modelos sí están versionados en el repositorio. Los ZIP de entrega omiten los derivados diarios; para generarlos, reunir los productos 8.2 y 8.3 y seguir el comando de EDA con `requirements-eda.lock.txt` indicado en [BUILD.md](../BUILD.md). El test 01/01/2025–31/08/2026 ya se evaluó después de congelar los 120 modelos finales: se publican sus metadatos y resúmenes agregados; los derivados completos del test, las predicciones individuales y los pesos se conservan en la evidencia privada verificable descrita en ese documento.

El manifiesto versionado en `manifests/` registra 75 archivos, URLs, SHA-256, tamaño, fecha de verificación y auditoría de timestamps/esquema. Contiene 2.014.480 filas frente a 2.014.560 minutos de calendario: hay 80 minutos ausentes y un timestamp de cierre inesperado en marzo de 2023. Descarga completa no equivale a cobertura sin huecos ni a validación de precios.

Reproducir o reanudar con Python 3.12 (solo biblioteca estándar), desde la raíz del proyecto. Cambiar `/ruta/adquisicion/oficial` por la carpeta de destino:

```bash
python3 scripts/download_binance_spot.py --start 2023-01-01 --end 2026-10-30 --output /ruta/adquisicion/oficial --workers 4
```

Reutiliza ZIP existentes solo tras verificar su hash contra el checksum oficial; conserva archivos discrepantes para revisión. Respeta el proxy, TLS y los límites del proveedor. No usa la API y no requiere claves. El descargador escribe un manifiesto JSON en la carpeta de destino. Para verificar sin red que la adquisición coincide con la instantánea versionada, usar sus rutas relativas y SHA-256 desde la raíz del proyecto:

```bash
python3 - /ruta/adquisicion/oficial data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw_root = Path(sys.argv[1]).resolve()
records = json.loads(Path(sys.argv[2]).read_text())["archives"]
if not records:
    raise SystemExit("El manifiesto no contiene archivos")
for record in records:
    archive = (raw_root / record["relative_path"]).resolve()
    if not archive.is_relative_to(raw_root):
        raise SystemExit("Ruta fuera de la carpeta de adquisición")
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != record["sha256"]:
        raise SystemExit(f"SHA-256 distinto: {record['relative_path']}")
print(f"Verificados {len(records)} archivos; sin descargas ni extracción de CSV")
PY
```

La comprobación falla si falta un archivo o cambia un byte; no depende de un TXT conservado en el entorno original. Recorre los 75 ZIP de la adquisición completa, incluidos septiembre–octubre de 2026, para comprobar su integridad; no extrae precios ni amplía el calendario experimental. El EDA abre únicamente desarrollo 2023–2024 y el scoring del protocolo abre únicamente la reserva enero de 2025–agosto de 2026.

Fuente: Binance Vision / Binance Spot, `https://data.binance.vision/`. Aplican los Binance Vision Dataset Terms consultados, versión 1.0 (26/08/2026), con CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Ver la sección 2.3.4 del libro para condiciones y atribución.

La adquisición y los conteos no constituyen EDA del objetivo ni acceso a resultados de test. El script revisa timestamps, conteos y esquema, sin examinar precios para elegir modelos. La extensión descargada septiembre–octubre de 2026 no modifica por sí sola las particiones experimentales preregistradas.
