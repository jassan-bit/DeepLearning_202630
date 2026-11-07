# Remediación E2: protocolo, código y evidencias públicas

Suplemento del repositorio https://github.com/jassan-bit/DeepLearning_202630.
Conserva el protocolo F2a actual, el código propio, los tests, el cribado
bibliográfico y los agregados públicos disponibles al empaquetar. El núcleo
científico se publicó antes de los nuevos ajustes en el commit
599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0. Los SHA-256 del manifiesto identifican
esta entrega; una repetición posterior debe registrar su propia fecha y fuentes.

## Alcance del procedimiento

El catálogo final planificado tiene veinte configuraciones y quince familias
ajustables. Sus T=12 candidatos y presupuesto de cómputo son asignaciones, no una
búsqueda ya ejecutada. El flujo que puede repetirse con una orden es **P0/OOF de
desarrollo**, con recetas fijas: TimeMixer, iTransformer, PatchTST y GRU, semillas
660101–660105. Son veinte trayectorias noviembre–diciembre y doscientas OOF
mensuales enero–octubre2024: 220 trayectorias, 440 etapas de optimización
(selección de época y refit). No es el HPO, P1, la prueba de potencia definitiva,
la selección de los tres mejores finales ni el benchmark optimizado de veinte
modelos. Los otros adaptadores y las puertas de fase siguen su revisión propia.

Las configuraciones congeladas dicen cero resultados nuevos en el momento de su
definición. Esos contadores históricos no se convierten en el total actual de
P0/OOF. current_protocol.json y los recibos/agregados incluidos describen la
ejecución observada disponible. Si una carpeta de resultados aún no existe, su
ausencia se declara en el manifiesto, no se sustituye por una tabla ficticia.
La conclusión sobre gaps o incertidumbre es la que arrojen los datos registrados;
este paquete no promete significancia, mejora ni potencia suficiente.

La herencia científica E1 permanece: D7×5, σ7, h1, RMSE media entre trayectorias,
δ y calendario. El flujo P0 no abre 2025–2026 ni restaura un test desconocido.
La calibración predictiva A es secundaria: fit/escalas/selección hasta24oct2024,
calibración1nov–30dic con ese mismo predictor congelado, sin cobertura temporal
exacta afirmada. Los errores OOF cuentan fechas únicas y mantienen separados el
diagnóstico P0 y los mejores modelos finales.

## Instalación y fuentes: usar un checkout del repositorio

El ZIP no es una copia completa del repositorio ni del Jupyter Book compilado.
Clonar el proyecto, usar el commit de publicación correspondiente y, si se
extrae este suplemento, conservar sus subdirectorios. No sustituir el código
científico congelado por una versión incompatible. Desde la raíz del checkout:

    python3.12 -m venv .venv-e2
    .venv-e2/bin/python -m pip install -r requirements-e2-remediation.lock.txt

El lock incluye el de paper-guide para las dependencias de los módulos de
autores; usarlo no convierte el presente P0 Bitcoin en la réplica ETTh1. Los dos
CSV diarios de desarrollo2023–2024 ya están versionados en el checkout bajo
results/eda_development. **No se incluyen esos datos diarios en este ZIP**. Sus
nombres y hashes se conservan en pilot_design.json; el ejecutor los verifica y
limita toda decodificación a desarrollo. La documentación data/README.md del
checkout describe la fuente y adquisición originales. No se redistribuyen
archivos crudos de mercado.

Los dos repositorios de autores se adquieren por separado. Sus URLs de ZIP,
commits, licencias y hashes de archivo/archivos interiores están en
pilot_design.json. Este ejemplo utiliza el descargador seguro propio incluido,
descarga solo esos repositorios y **no descarga ETTh1**:

```bash
.venv-e2/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from download_e2_paper_sources import asset, extract_verified, fetch_bytes, MAX_DOWNLOAD_BYTES
plan = json.loads(Path('experiments/e2_remediation_20261106/pilot_design.json').read_text())
folder = Path('../e2_reproduction/author_sources')
folder.mkdir(parents=True, exist_ok=True)
for source in plan['author_sources'].values():
    payload = asset(folder, source['archive_filename'], source['archive_url'], source['archive_sha256'], MAX_DOWNLOAD_BYTES, fetch_bytes)
    extract_verified(folder, payload, source)
PY
```

El ejemplo está escrito para Bash/Linux. Los ZIP y el código de autores quedan
fuera del repositorio y no entran en este paquete. Se conservan Apache2.0 para
TimeMixer y MIT para Time-Series-Library; consultar sus avisos originales.

## Repetir P0 y OOF registrados con una orden

Con los dos CSV presentes en el checkout y las fuentes verificadas:

    .venv-e2/bin/python scripts/execute_e2_development_pilot.py --stage all --source-root ../e2_reproduction/author_sources/TimeMixer-e24610583b36fdd8c76cc17a8df4e65759a5f460 --control-source-root ../e2_reproduction/author_sources/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e --output-dir ../e2_reproduction/new_run --workers 2

La orden registra antes de ajustar, conserva artefactos privados, cierra todos
los checkpoints programados antes del scoring de su fase y analiza predicciones
guardadas. Ejecuta hasta440 ajustes, no solo una demostración de forma. Usar una
salida propia fuera del checkout; no mezclarla con recibos de una ejecución
anterior ni reemplazar semillas fallidas/desfavorables. --stage preflight solo
valida datos/soportes, y las etapas register/fit/score/analyze permiten inspección
explícita. El registro exige contexto Git y bytes científicos versionados.
Extraer el ZIP sin un checkout no recrea ese contexto ni los CSV excluidos.

La incertidumbre y el análisis de errores tienen funciones/CLI propias en
e2_predictive_uncertainty.py. Sus instrucciones están en el capítulo12: requieren
OOF y taxonomía causal, al menos cien fechas elegibles únicas por modelo y salida
privada nueva. No interpretar recetas P0 como Top3 finales ni calibrar el ajuste
primario con sus errores de entrenamiento de noviembre–diciembre.

## Revisión sin fuentes ni datos de mercado

Los tests sintéticos pueden ejecutarse después de instalar el lock, sin los
CSV originales, fuentes de autores o pesos reales:

    .venv-e2/bin/python tests/test_e2_protocol_plan.py
    .venv-e2/bin/python tests/test_e2_temporal_baselines.py
    .venv-e2/bin/python tests/test_e2_remediation_pilot.py
    .venv-e2/bin/python tests/test_e2_remediation_statistics.py
    .venv-e2/bin/python tests/test_e2_literature_screening.py
    .venv-e2/bin/python tests/test_e2_predictive_uncertainty.py

Las pruebas de ARIMA ajustan ejemplos sintéticos pequeños; no son corridas ni
métricas de Bitcoin. Las pruebas no sustituyen ausencia de fuga, potencia o
eficacia del procedimiento final. El validador estándar puede inspeccionarse con
`python -S scripts/e2_protocol_plan.py validate`, sin ajuste ni scoring.

El cribado de literatura contiene el ledger de7.833 grupos, títulos/DOI, reglas,
huellas y decisiones provisionales. No contiene abstracts completos ni los
exports nativos WoS/Scopus. Volver a ejecutar screen_e2_literature.py exige el
private_combined_records.json autorizado, conservado por separado; sus tests
usan fixtures. El texto público distingue consultas locales de nuevas búsquedas
nativas, los374 Scopus no recibidos y la elegibilidad final todavía abierta.

## Integridad y archivos excluidos

MANIFIESTO.json registra cada tamaño y SHA-256. SHA256SUMS junto al ZIP identifica
el contenedor. Los timestamps internos son fijos al06/11/2026. Para reconstruirlo
desde sus fuentes públicas y verificarlo, sin ejecutar ciencia:

    python scripts/package_e2_remediation.py
    python scripts/package_e2_remediation.py --check

El ZIP no se contiene a sí mismo. Los metadatos de preregistro se mantienen como
capturas históricas; sus hashes no certifican por sí solos corrección científica.
Los ZIP anteriores permanecen como archivos de otras versiones, no como el
protocolo vigente. No se ejecutan sus generadores antiguos para reescribirlos.
Se excluyen precios, CSV diarios, abstracts, PDFs, código descargado de autores,
checkpoints, pronósticos por fecha y ledgers individuales de errores. Solo se
admiten agregados por modelo/semilla/fold, recibos y figuras públicos permitidos.
