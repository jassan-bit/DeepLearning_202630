# Datos Binance Spot BTCUSDT

Archivos oficiales descargados para 2023-01-01–2026-10-30 inclusive, UTC, velas de un minuto. Los brutos se conservan fuera del checkout, en `/workspace/binance-spot-data/BTCUSDT/1m/monthly/` y `daily/`; no se publican los ZIP ni los CSV intradiarios. Los derivados diarios de desarrollo de 2023–2024, sus hashes y los resultados necesarios para reproducir EDA y modelos sí están versionados. El test 01/01/2025–31/08/2026 ya se evaluó después de congelar los 120 modelos finales: se publican sus metadatos y resúmenes agregados; los derivados completos, las predicciones individuales y los pesos se conservan en la entrega privada verificable descrita en `BUILD.md`.

El manifiesto versionado en `manifests/` registra 75 archivos, URLs, SHA-256, tamaño, fecha de verificación y auditoría de timestamps/esquema. Contiene 2.014.480 filas frente a 2.014.560 minutos de calendario: hay 80 minutos ausentes y un timestamp de cierre inesperado en marzo de 2023. Descarga completa no equivale a cobertura sin huecos ni a validación de precios.

Reproducir o reanudar con Python 3.12 (solo biblioteca estándar):

```bash
cd /workspace/DeepLearning_202630
python3 scripts/download_binance_spot.py --start 2023-01-01 --end 2026-10-30 --output /workspace/binance-spot-data --workers 4
```

Reutiliza ZIP existentes solo tras verificar su hash contra el checksum oficial; conserva archivos discrepantes para revisión. Respeta el proxy, TLS y los límites del proveedor. No usa la API y no requiere claves. Para verificar la instantánea ya adquirida sin red:

```bash
cd /workspace/binance-spot-data
sha256sum -c SHA256SUMS_2023-01-01_2026-10-30.txt
```

Fuente: Binance Vision / Binance Spot, `https://data.binance.vision/`. Aplican los Binance Vision Dataset Terms consultados, versión 1.0 (26/08/2026), con CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Ver la sección 2.3.4 del libro para condiciones y atribución.

La adquisición y los conteos no constituyen EDA del objetivo ni acceso a resultados de test. El script revisa timestamps, conteos y esquema, sin examinar precios para elegir modelos. La extensión descargada septiembre–octubre de 2026 no modifica por sí sola las particiones experimentales preregistradas.
