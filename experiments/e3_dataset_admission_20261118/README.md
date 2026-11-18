# Paso siguiente: fuente y calidad de D01, guardado en GitHub

Esta etapa usa sólo CPU y biblioteca estándar. Inspecciona el candidato
D01 AAPL; no repite los 17 perfiles ni entrena modelos.

El archivo original `inputs/stocks.csv.gz` está copiado sin cambios de
`bokeh/bokeh_sampledata`, commit
`3e831d2779ca74c087c7eee05d538015c97b1e52`, ruta
`src/bokeh_sampledata/_data/stocks.csv.gz`. Son 294.246 bytes y SHA-256
`5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd`.
Contiene cinco series; esta auditoría sólo examina AAPL.

`provenance.json` conserva las URLs y hashes de la documentación primaria.
El distribuidor identifica los datos actuales como WIKI de dominio público,
con permiso de copia y redistribución. No se aplica por extensión la
licencia de software ni se renombra esa declaración como CC0. Atribución:
Quandl WIKI Prices; copia archivada de marketneutral, distribuida por Bokeh.
`technical_snapshot.json` documenta la inspección preparatoria del archivo.

AAPL contiene 3.270 filas entre 2000-03-01 y 2013-03-01, con fechas únicas
y ordenadas, cero faltantes numéricos y cero violaciones OHLCV básicas.
Se excluye `Adj Close`: incorpora ajustes corporativos posteriores. Se
conservan todas las filas y se señalan movimientos de cierre mayores al
30 %, sin eliminarlos ni clasificarlos automáticamente como splits. Un
desplome real no debe borrarse al confundirlo con una acción corporativa.

El contrato propuesto es siete observaciones y próxima **sesión bursátil**;
no se presenta como siete días UTC consecutivos. Faltan una política de
acciones corporativas respaldada por eventos, validación del calendario de
sesiones y el protocolo común. Por eso el informe conserva cero datasets
admitidos. D02 y D03 mantienen sus bloqueos anteriores; no se inspeccionan
sus archivos en esta etapa. No se usan resultados de predicción para elegir
series ni se calculan tensores de objetivos o métricas de modelos.

## Ejecutar en Colab

El lanzador recibe `E3_ADMISSION_COMMIT`, un commit publicado completo,
consulta `GITHUB_TOKEN` de Secretos o una entrada oculta y verifica acceso
antes de generar el informe. El token requiere Contents: Read and write
para `jassan-bit/DeepLearning_202630`. No se guarda en los archivos.

`launch_colab.py` verifica los bytes del archivo, produce JSON/logs y los
guarda mediante `save_dataset_audit` en una rama nueva
`e3-dataset-audit-<RUN_ID>`, bajo `runs/dataset_admission/<RUN_ID>/`.
Imprime la URL GitHub. Si la subida falla y quedan los archivos locales,
volver a ejecutar la celda reintenta sólo el guardado, sin repetir la
inspección. El dato original ya está versionado junto al código.

La primera ejecución guardada desde Colab de este paso sigue pendiente.
La revisión local no se presenta como admisión final, protocolo previo,
entrenamiento, prueba estadística, novedad R1 o superioridad R3.
