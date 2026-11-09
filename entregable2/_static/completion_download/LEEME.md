# E2: procedimiento completo, código y evidencia pública

Suplemento de https://github.com/jassan-bit/DeepLearning_202630. Contiene el
procedimiento CPU de veinte configuraciones, quince recetas ajustables con
T=12 y asignación comparable, P1, potencia, ajustes primarios/A53, calibración,
errores OOF, costes, ablaciones y análisis. El código o el presupuesto asignado
no prueban que esas etapas hayan terminado ni que sus conclusiones sean
favorables. MANIFIESTO.json identifica exactamente los artefactos disponibles.
La evidencia de ejecución es únicamente la que figure en recibos públicos;
la ausencia de resultados no se rellena con métricas ficticias.

## Usar un checkout correspondiente del repositorio

Este ZIP no es un repositorio Git completo, un libro compilado ni un ejecutable
autónomo con datos. Usar el commit de publicación que lo acompaña y mantener
sus rutas relativas. El registro exige un checkout Git limpio y las identidades
de código, configuración, datos y fuentes. Extraer este suplemento por sí solo
no proporciona ese contexto. Desde la raíz del checkout, en Linux/Bash:

    python3.12 -m venv .venv-e2
    .venv-e2/bin/python -m pip install -r requirements-e2-completion.lock.txt
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase plan

El lock y el ejecutor son CPU. No hay ejecución GPU equivalente acreditada por
este paquete. Todos los trabajadores de ajuste usan un hilo numérico; el
ejecutor permite de uno a cuatro trabajadores. No reducir arquitecturas,
candidatos o repeticiones para presentar una ejecución parcial como completa.
Las comprobaciones de forma o memorización sintética no son ajustes de Bitcoin.

## Fuentes y datos conservados por separado

Los dos CSV diarios de desarrollo están versionados en el checkout bajo
results/eda_development; no se incluyen en el ZIP. design.json conserva sus
SHA-256. Conservar también los ZIP de mercado del manifiesto público y sus
identidades; el paquete no descarga, distribuye ni abre esos datos. No abrir
las reservas futuras antes de las puertas de fase. La adquisición de fuentes
descarga únicamente los archivos de autores fijados por commit/hash; no papers,
datasets ni pesos. Sus licencias y avisos deben conservarse:

    .venv-e2/bin/python scripts/download_e2_completion_sources.py --output-dir ../e2_run/author_sources

El archivo ../e2_run/author_sources/source_roots.json resultante contiene las
rutas privadas; no publicarlo. El manifiesto de fuentes incluido identifica
archivos ajenos pero no redistribuye sus contenidos.

## Ejecutar por fases

Consultar experiments/e2_completion_20261106/workflow.md antes de ejecutar.
Las órdenes siguientes conservan la misma salida privada y fuentes fijadas:

    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase preflight --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --workers 4
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase register --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --raw-root ../market_raw --workers 4

Una vez registrado, repetir la misma base de orden con --phase hpo,
oof-errors, p1, power, final-fit, calibration, cost y ablation, en ese orden.
El scoring final es una fase explícita:

    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase score --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --raw-root ../market_raw --workers 4
    .venv-e2/bin/python scripts/run_e2_final_benchmark.py --phase analyze --source-roots-json ../e2_run/author_sources/source_roots.json --output-dir ../e2_run/new_run --workers 4

La opción --phase all implementa esa secuencia y también exige --raw-root.
Usar una salida nueva fuera del checkout; no mezclar ni sobrescribir otro
registro. HPO completo cierra las quince recetas ajustables antes de P1. P1
evalúa 53 fechas del 8nov–30dic2024 con diez semillas nuevas y estima la potencia
del procedimiento; diez semillas no acreditan por sí solas potencia suficiente.
R se fija por esa etapa, con tope200 y declaración explícita si no se alcanza.

Las veinte cohortes A53 conservan cinco semillas prefijadas (una trayectoria
en referencias deterministas), con selección/fit/escalas hasta31oct2024 y
calibración8nov–30dic2024. No son intervalos del ajuste primario hasta24dic2024.
Todos los pesos primarios y A se cierran antes de calibrar; todos los
calibradores se cierran antes de puntuar la reserva conocida de60fechas2026.
No se afirma cobertura temporal exacta ni una nueva confirmación independiente.
Las ablaciones se realizan sobre desarrollo53 y no seleccionan componentes con
el test. Los OOF de las tres anclas heredadas declaran la exposición histórica
de su selección E1; no se presentan como selección anidada imparcial.

## Comprobación sin datos privados

Después de instalar el lock, las pruebas sintéticas no necesitan datos de
mercado ni pesos reales. Algunas comprobaciones de paridad con autores exigen
las fuentes fijadas y declaran su ausencia cuando corresponda:

    .venv-e2/bin/python -m unittest discover -s tests -p 'test_e2_final_*.py'
    .venv-e2/bin/python tests/test_e2_completion_package.py
    python scripts/package_e2_completion.py --check

El ZIP contiene helpers propios transitivos y metadatos; no contiene futuros
precios, CSV diarios, abstracts nativos, PDF, código descargado de autores,
checkpoints, predicciones por fecha o ledgers de errores individuales. Solo se
admiten derivados públicos autorizados. La selección global/PRISMA, los374
registros Scopus restantes y la evidencia editorial pendiente siguen teniendo
sus límites propios, aunque el código experimental esté disponible.

MANIFIESTO.json declara tamaños y SHA-256; SHA256SUMS identifica el contenedor.
Las fechas internas son fijas. Reconstruir no ejecuta ciencia:

    python scripts/package_e2_completion.py
    python scripts/package_e2_completion.py --check
