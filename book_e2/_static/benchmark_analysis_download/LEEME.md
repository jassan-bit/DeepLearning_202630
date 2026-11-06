# Entregable 2, sección 6: herramientas y plantillas de análisis

Suplemento del repositorio https://github.com/jassan-bit/DeepLearning_202630.
Este paquete contiene el capítulo 6, la propuesta de análisis estadístico, su
código y pruebas, plantillas públicas y el auditor documental del diseño D7.
No contiene resultados finales del benchmark Bitcoin, nuevas predicciones,
pesos, datos originales ni un pipeline de entrenamiento.

## Estado y límites

**Archivo v1:** las quince filas y bloqueos de este suplemento describen la propuesta inicial. El protocolo vigente de veinte configuraciones, guía completa y P0/OOF registrado está en `experiments/e2_remediation_20261106/` y en el paquete Remediacion_E2_protocolo_codigo_evidencias.zip. Sus contadores no son el total de experimentos posteriores.

La propuesta permanece abierta. Las pruebas verifican funciones estadísticas e
integridad con ejemplos sintéticos; no prueban que los quince comparadores se
hayan entrenado. Continúan los prerrequisitos del diseño D7, incluidos los
adaptadores, la auditoría de fuga, el piloto propio y la potencia. Diez modelos
nuevos se añaden a GRU/LSTM/TCN y dos referencias: al menos 15 configuraciones.
Los resultados ETTh1 de la sección 5 no son resultados Bitcoin ni se reutilizan
en estas plantillas como si lo fueran.

El contrato heredado conserva RMSE primaria, la unidad temporal dependiente, la
partición y el margen absoluto de E1. El test histórico ya conocido no se
renombra como una prueba confirmatoria desconocida. La propuesta de análisis
explicita comparaciones, tamaños de efecto, remuestreo y multiplicidad, sin
convertir una declaración posterior en un diseño previo al test histórico.

## Revisar y ejecutar las pruebas sin entrenar

Extraer conservando la estructura de directorios. El analizador utiliza NumPy y
SciPy; requirements-benchmark-analysis.lock.txt fija sus versiones observadas y
no instala un framework de entrenamiento. En Python3.12:

    python -m pip install -r requirements-benchmark-analysis.lock.txt

    python scripts/analyze_e2_benchmark.py
    python scripts/analyze_e2_benchmark.py --help
    python tests/test_e2_benchmark_analysis.py
    python tests/test_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py
    python scripts/audit_e2_benchmark_readiness.py --require-ready

--require-ready devuelve código 2 para una propuesta documental válida con
bloqueos. Es un resultado esperado, no autorización para entrenar. Un contrato
incoherente o un hash E1 alterado devuelve código 1. El auditor solo lee metadatos
y sus seis fuentes públicas fijadas; no abre archivos de mercado ni pesos.

La orden sin argumentos solo regenera los diez derivados documentales bajo
book_e2/_static/benchmark_analysis; usa biblioteca estándar y no puntúa modelos.
Su table_template.csv enumera quince modelos pendientes y requirements.csv
registra requisitos. Los CSV de predicciones, ejecuciones y costos se entregan
vacíos, con encabezados; no se rellenan con resultados ficticios. schemas.json
define sus columnas y unidades. Las pruebas sintéticas permanecen en tests.

La CLI opcional --panel, --runs, --costs, --calendar, --scope synthetic_fixture y
--output-dir admite verificar la coherencia de un panel sintético y sus hashes
en una carpeta separada. Ese modo no certifica causalidad, potencia, suficiencia
del presupuesto o desempeño científico, ni produce rankings Bitcoin reales.
El modo real permanece bloqueado por la propuesta no congelada. --help describe
los insumos para cuando existan ejecuciones válidas; no declara que existan hoy.

## Integridad, reconstrucción y procedencia

Se incluyen los tres JSON del contrato, la propuesta D7 y sus seis derivados de
auditoría, además de seis fuentes públicas E1 identificadas por SHA-256. Son
recetas, resumen agregado, diseño/dimensionamiento, manifiesto y documentación;
el manifiesto de datos no contiene las observaciones de mercado.

book_e2/references.bib permite resolver las citas del capítulo, incluidas las
referencias metodológicas de los contrastes. Esas referencias de métodos no se
cuentan como nuevos estudios aplicados del corpus ni acreditan consultas en
bases bibliográficas. No se redistribuyen sus PDF ni se incluye un entorno de
construcción completo del Jupyter Book; este paquete permite revisión y ejecución
de las herramientas y pruebas indicadas.

MANIFIESTO.json identifica cada archivo incluido por tamaño y SHA-256. El ZIP
tiene fechas internas fijas al 06/11/2026. Desde la carpeta extraída, reconstruir
el contenedor y comprobar su correspondencia exacta con esas fuentes:

    python scripts/package_e2_benchmark_analysis.py
    python scripts/package_e2_benchmark_analysis.py --check

El contenedor no se incluye a sí mismo. SHA256SUMS, publicado junto al ZIP,
permite contrastar además el archivo completo. Ninguna de estas órdenes entrena,
descarga datos o sustituye las evidencias experimentales todavía pendientes.
