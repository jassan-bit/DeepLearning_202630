# Reproducción automática de agregados E3 en contenedor CPU

Este paquete reconstruye la tabla principal de **17 sistemas** y diez CSV
adicionales a partir de **143 trayectorias guardadas**, en las 608 fechas
históricas comunes. Ejecuta el análisis estadístico posterior completo y el
suplemento descriptivo con sus semillas y versiones fijadas. Comprueba la
alineación, las columnas, los valores ausentes y la discrepancia por campo y
unidad frente a las tablas publicadas.

**Alcance:** reproducción automática de agregados desde predicciones congeladas.
No vuelve a entrenar los 143 predictores, no ejecuta sus modelos, no recalibra y
no restaura el cegamiento de un calendario conocido. No acredita R1/R2/R3 ni
sustituye el acta de una persona ajena al código. La reproducción humana
independiente sigue pendiente.

## Perfil separado del servidor

`deployment/science/Dockerfile` usa Python **3.12.14** y el mismo digest de imagen
base ya fijado en el proyecto. El lock científico fija doce paquetes, incluidos
NumPy **2.3.5**, SciPy **1.17.0** y Matplotlib **3.10.8**. No contiene Torch ni un
framework de entrenamiento. El Dockerfile raíz mantiene su función de servicio
ONNX y no se modifica al construir esta imagen científica.

Los generadores conservados son:

- `experiments/e3_estadistica_posterior_20261117/run.py`, con su plan, validador,
  métodos, MCS y figuras originales.
- `experiments/e3_revision_bloque_d_20261117/metrics_supplement.py`, que agrega
  sMAPE, R², intervalos heredados E2 y dispersión entre semillas. E3 permanece
  sin intervalo predictivo calibrado.

`reference_manifest.json` fija sus ocho archivos de código/plan y las once
tablas de referencia mediante SHA-256. El corte de referencia es el commit
`73d8958c50a69fbea46deb828f03f89385318789`. El contenedor conserva esos archivos;
sus resultados nuevos se escriben en otra carpeta y no reemplazan las tablas
del libro.

## Construir y ejecutar desde cero

Desde la raíz de un checkout que contenga este paquete:

```bash
docker build -f deployment/science/Dockerfile -t e3-scientific-replay:v1 .

python experiments/e3_remediacion_20261118/reproduction/reproduce.py fetch \
  --data work/e3-science-inputs

mkdir -p work/e3-science-output
docker run --rm --read-only --network none \
  --tmpfs /tmp:rw,nosuid,size=64m --memory=1g --cpus=1 --pids-limit=128 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/work/e3-science-inputs",dst=/inputs,readonly \
  --mount type=bind,src="$(pwd)/work/e3-science-output",dst=/output \
  e3-scientific-replay:v1
```

La descarga inicial usa únicamente la biblioteca estándar de Python y verifica
los **once archivos públicos inmutables** de `input_manifest.json`, tanto por
SHA-256 como por tamaño. Los archivos ya presentes con bytes distintos causan
un error. El contenedor se ejecuta después sin red, con entradas de sólo lectura,
raíz de sólo lectura y un usuario sin privilegios en la ejecución verificada.
La selección `--user` permite escribir el volumen de salida con el UID/GID del
operador Linux. El límite científico de 1 GiB no es una medida de consumo ni el
perfil del servicio gratuito Render.

`work/e3-science-output` debe estar vacío. Para repetir el proceso se elige otra
carpeta de salida; no se borran recibos anteriores. El programa devuelve código
de salida 1 si una fuente, un generador, una tabla o una tolerancia no coincide,
y conserva el fallo cuando el proceso ya puede escribir el recibo.

Si la construcción carece de acceso a los registros de paquetes, se pueden
preparar las doce ruedas en un host Linux/Python 3.12 compatible:

```bash
python -m pip download --no-deps -r deployment/science/requirements.lock.txt \
  --dest deployment/science/offline_packages
docker build --build-arg SCIENCE_OFFLINE=1 \
  -f deployment/science/Dockerfile -t e3-scientific-replay:v1 .
```

Las ruedas no se versionan ni se incluyen en la imagen como una capa de
archivos de instalación. Se conserva el marcador del directorio para que una
construcción en línea funcione desde un checkout limpio. No se desactiva TLS ni
se incorporan credenciales a la imagen.

## Resultados y criterios de comparación

La salida incluye `statistics/`, `supplement/`, dos logs de generador y
`reproduction_report.json`. Se comparan:

1. Métricas principales de los 17 sistemas.
2. Los 48 contrastes primarios, efectos, veredictos, 16 contrastes MAE,
   subgrupos de semillas y las 96 filas de sensibilidades conservadas.
3. Métricas ampliadas, tabla de presentación, incertidumbre entre semillas y
   los 143 predictores individuales.

Las filas se alinean por identificadores explícitos y no se pueden omitir ni
duplicar. La tabla congelada de sensibilidades repite las mismas pruebas entre
semillas en dos configuraciones y no tiene una columna intrínseca que las
distinga; para esa tabla se conserva el orden completo mediante `@row`. No se
eliminan sus duplicados. Las columnas deben coincidir exactamente; los textos,
decisiones y ausencias se conservan. Una cobertura ausente no equivale a cero.

Cada diferencia numérica se evalúa con la regla fijada en el cliente:

```text
abs(observado - referencia) <= 1e-14 + 1e-12 * abs(referencia)
```

El recibo informa por tabla, campo y unidad el número de pares, máxima
discrepancia absoluta, localización y cantidad de incumplimientos. No mezcla
RMSE, probabilidades, porcentajes, conteos o identificadores en un máximo global.
El campo `statistic` se separa por unidad: la distancia bootstrap al margen usa
σ₇ y los estadísticos estandarizados usan una escala adimensional. También se
informa igualdad de bytes y SHA-256 de cada CSV. Estas tolerancias son de
reconstrucción numérica; no son una prueba de equivalencia científica ni el
margen práctico δ del estudio.

Las ocho pruebas de contrato se ejecutan sin dependencias científicas:

```bash
python -m unittest discover -s tests -p test_e3_scientific_replay.py -v
```

Cubren alineación por modelo, pérdida/duplicación de filas, valores ausentes
sustituidos por cero, esquemas distintos, NaN, intervalos estructurados y
discrepancias separadas por unidad.

## Archivo y reproducción humana pendiente

`make_bundle.py` construye un archivo versionado pequeño con código, perfiles,
manifiestos y tablas de referencia, sin ruedas de instalación, pesos o datos
crudos. La descarga de las trayectorias queda fijada por sus commits y hashes.
Las condiciones de uso y redistribución de cada fuente mantienen el estado
descrito en `DATA_STATEMENT.md`; este paquete no concede una licencia nueva.

```bash
python experiments/e3_remediacion_20261118/reproduction/make_bundle.py
python experiments/e3_remediacion_20261118/reproduction/make_bundle.py --check
tar -xzf experiments/e3_remediacion_20261118/reproduction/reproduction_bundle_v1.tar.gz
cd e3-scientific-replay-v1
```

Desde esa carpeta se aplican los comandos de construcción, descarga y ejecución
anteriores. El manifiesto externo fija el SHA-256 y tamaño del archivo y de cada
entrada. El orden lexicográfico, tiempos cero, UID/GID cero y modos fijos evitan
que metadatos locales cambien el archivo. Si los bytes de una versión existente
cambian, el programa se detiene: debe preservarse esa versión y crearse otra.

El archivo y su manifiesto permiten revisar exactamente qué se entrega. No son
un DOI, depósito institucional o certificación de preservación permanente. Una
persona externa deberá construir la imagen en su propio entorno, ejecutar los
comandos, revisar el alcance y firmar su propio informe con las discrepancias.
Los recibos automáticos conservan `human_independent_reproduction=false`.
