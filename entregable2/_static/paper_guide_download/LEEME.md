# Entregable 2, sección 5: paper guía, réplica externa y piloto

Este paquete reúne código propio, diseño, tablas agregadas, figuras y recibos
públicos del capítulo 5. Se usa como complemento del repositorio:
https://github.com/jassan-bit/DeepLearning_202630

El diseño y los controladores se registraron en el commit público
329b0a6c8238d632dd00537c5cefab91360bafd1 antes de estos ajustes sobre ETTh1. Ese
registro tiene alcance de réplica externa y piloto; no acredita automáticamente
todos los requisitos de la sección 9.1 de la guía ni un test Bitcoin nuevo.

## Qué se evalúa

La referencia es TimeMixer, configuración original unificada ETTh1 multivariante,
L96/H96, Table13, página PDF17 de 2405.14616v1: MSE0.375 y MAE0.400. No se combina
con la configuración optimizada de otra tabla. Se declararon cinco semillas
2019–2023, 10 épocas y el preset del código fijado de los autores. El código
original procesa lotes128 con drop_lastTrue: puntúa 2.688 de los 2.785 orígenes
test nominales y omite los 97 finales. Las métricas usan siete canales
estandarizados con el prefijo de entrenamiento y los 96 horizontes; no están en
la escala física de Bitcoin ni se calculan tras invertir el escalado.

El criterio registrado compara el IC95% t de la media entre las cinco semillas
con el valor publicado fijo y redondeado0.375: todo el intervalo de la diferencia
debe quedar dentro de ±0.01875 MSE. El margen no es el delta de Bitcoin. MAE es
secundaria, no una segunda vía para declarar éxito. La dispersión del renglón
publicado es desconocida: no se inventa una varianza original ni un contraste de
dos muestras. Compatibilidad numérica no demuestra identidad exacta con el
experimento histórico. El uso de CPU, las versiones actuales, los generadores
explícitos de los loaders y la cobertura del test se declaran como diferencias.

El piloto aprendido es distinto: TimeMixer, iTransformer y PatchTST, cinco
semillas cada uno, 2 épocas y las últimas 2.048 ventanas elegibles de train; son
15 ajustes con presupuesto limitado. Los recibos públicos documentan cinco
réplicas y quince ajustes del piloto finalizados. El piloto no es la réplica de los
renglones originales de los dos controles ni el benchmark Bitcoin. Sus dos
diagnósticos usan solo validación y conservan el umbral y el remuestreo
registrados, también cuando los resultados no apoyen la hipótesis. Consulte el
capítulo y las tablas para los resultados observados; crear este ZIP no declara
por sí mismo éxito de la réplica o de las hipótesis del piloto.

## Revisar sin adquirir los datos originales

Extraer conservando directorios. Los agregados y recibos permiten revisar cifras
y localizadores sin los CSV, pesos o predicciones individuales. Con el entorno
de requirements-paper-guide.lock.txt:

    python tests/test_e2_paper_download.py
    python tests/test_e2_paper_replication.py
    python tests/test_e2_paper_statistics.py
    python scripts/package_e2_paper_guide.py
    python scripts/package_e2_paper_guide.py --check

Las pruebas de descarga usan datos sintéticos y no requieren red. Las pruebas
numéricas/estadísticas utilizan las dependencias del lock y no necesitan ETTh1,
los repositorios de autores ni entrenar modelos. Los análisis estadísticos pueden
importarse para verificar sus funciones con estos casos sintéticos. La primera
orden de empaquetado reconstruye el contenedor desde los materiales extraídos;
--check contrasta ese contenedor con sus fuentes. El ZIP no se incluye a sí mismo.

execution_receipt.json conserva los veinte ajustes, sus hashes y la cronología
de cierre previa al primer scoring. training_curve_summary.csv y su figura
resumen las cinco curvas de la réplica. analysis_amendments.json documenta dos
correcciones de reporte registradas durante los ajustes y antes del scoring y
los diagnósticos: no cambian semillas, umbrales, datos ni hipótesis. Estos archivos
y los generadores propios de recibos y figuras se incluyen en el paquete.

## Obtener fuentes y repetir en un checkout versionado

Clonar el repositorio o superponer estos directorios en un checkout de la misma
versión de entrega. Con Python3.12, instalar el lock y descargar las fuentes en
una carpeta propia fuera del checkout:

    python -m venv .venv-paper-guide
    .venv-paper-guide/bin/python -m pip install -r requirements-paper-guide.lock.txt
    .venv-paper-guide/bin/python scripts/download_e2_paper_sources.py --design experiments/e2_paper_guide_20261106/design.json --output-dir ../paper_guide_local/assets

El descargador adquiere separadamente ETTh1.csv y dos ZIP de repositorios
fijados por commit. Valida SHA-256, archivos fuente y contención de la extracción.
No volver a descargar ni reemplazar silenciosamente una fuente con otros bytes.

Para las órdenes siguientes puede definir estas variables en Bash desde la raíz
del checkout (los nombres son específicos de esta tarea):

    e2_pg_design=experiments/e2_paper_guide_20261106/design.json
    e2_pg_assets=../paper_guide_local/assets
    e2_pg_runs=../paper_guide_local/runs
    e2_pg_diagnostics=../paper_guide_local/diagnostics
    e2_pg_timemixer="$e2_pg_assets/TimeMixer-e24610583b36fdd8c76cc17a8df4e65759a5f460"
    e2_pg_controls="$e2_pg_assets/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e"

Validar y registrar antes de entrenar:

    .venv-paper-guide/bin/python scripts/run_e2_paper_replication.py --phase validate --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --output-dir "$e2_pg_runs"
    .venv-paper-guide/bin/python scripts/run_e2_paper_replication.py --phase register --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --output-dir "$e2_pg_runs"

register exige contexto Git: design.json, run_e2_paper_replication.py y
e2_paper_data.py deben estar versionados y sin cambios frente a HEAD. No basta
extraer el ZIP en una carpeta sin Git. Una repetición crea su propio recibo y
fecha; no puede atribuirse retrospectivamente el registro histórico del proyecto.
No modificar el diseño o cambiar semillas después de observar resultados.

Entrenar primero las cinco réplicas; todavía no ejecutar score:

    for e2_pg_seed in 2019 2020 2021 2022 2023; do
      .venv-paper-guide/bin/python scripts/run_e2_paper_replication.py --phase fit --seed "$e2_pg_seed" --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --output-dir "$e2_pg_runs"
    done

Ejecutar los quince ajustes limitados del piloto y sus diagnósticos:

    for e2_pg_model in TimeMixer iTransformer PatchTST; do
      for e2_pg_seed in 2019 2020 2021 2022 2023; do
        .venv-paper-guide/bin/python scripts/run_e2_paper_replication.py --phase pilot --model "$e2_pg_model" --seed "$e2_pg_seed" --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --output-dir "$e2_pg_runs"
        .venv-paper-guide/bin/python scripts/paper_guide_pilot_diagnostics.py --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --run-dir "$e2_pg_runs/pilot/$e2_pg_model/seed_$e2_pg_seed" --model "$e2_pg_model" --seed "$e2_pg_seed" --output-dir "$e2_pg_diagnostics"
      done
    done

Cerrar los veinte ajustes antes del primer scoring de test, para conservar la
misma cronología comprobada por el generador de evidencia. Después puntuar una
vez, analizar los agregados y producir recibos y curvas:

    .venv-paper-guide/bin/python scripts/run_e2_paper_replication.py --phase score --design "$e2_pg_design" --data "$e2_pg_assets/ETTh1.csv" --source-root "$e2_pg_timemixer" --control-source-root "$e2_pg_controls" --output-dir "$e2_pg_runs"
    .venv-paper-guide/bin/python scripts/analyze_e2_paper_guide.py --design "$e2_pg_design" --replication-root "$e2_pg_runs" --pilot-root "$e2_pg_diagnostics" --output-dir ../paper_guide_local/analysis --bootstrap-draws 4999
    .venv-paper-guide/bin/python scripts/build_e2_paper_execution_evidence.py --run-dir "$e2_pg_runs" --diagnostics-dir "$e2_pg_diagnostics" --output-dir ../paper_guide_local/evidence

El diagrama propio puede regenerarse sin datos de mercado ni pesos; esta orden
escribe únicamente sus PNG/SVG en book_e2/_static/paper_guide_evidence:

    .venv-paper-guide/bin/python scripts/draw_e2_paper_guide.py

Estas órdenes producen material privado adicional (CSV adquirido, pesos y
predicciones) que no se incorpora automáticamente al libro ni a este ZIP.
Usar carpetas nuevas para una repetición independiente. El controlador conserva
los hashes y rechaza modificaciones de artefactos congelados.

## Integridad y derechos

MANIFIESTO.json identifica cada archivo incluido por tamaño y SHA-256. SHA256SUMS
publicado junto al ZIP identifica el contenedor. Las fechas ZIP se fijan al
06/11/2026. Para regenerar con las mismas fuentes públicas:

    python scripts/package_e2_paper_guide.py
    python scripts/package_e2_paper_guide.py --check

ETTh1 declara CC BY-ND4.0 y se adquiere mediante el enlace exacto del diseño; el
CSV no se redistribuye aquí. Los repositorios TimeMixer y Time-Series-Library
mantienen sus licencias Apache2.0 y MIT, respectivamente; se adquieren aparte.
Este ZIP contiene código propio, hashes, localizadores y agregados. Excluye PDF,
CSV original, código descargado de autores, checkpoints y pronósticos individuales.
No altera el contrato heredado de E1, no ejecuta el benchmark Bitcoin y no aporta
un test Bitcoin desconocido.
