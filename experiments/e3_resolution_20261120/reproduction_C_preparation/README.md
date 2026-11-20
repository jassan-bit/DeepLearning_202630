# C: preparación ejecutable de una reproducción histórica parcial

Este suplemento se preparó con asistencia de IA. **No se entrenó un modelo ni
se observó una ejecución humana independiente al prepararlo.** Tampoco abre
la confirmación de V3: el NO-GO y R3 no acreditada se conservan.

## Alcance concreto

Se dejan fijados dos ajustes finales originales del piloto, elegidos por
identidad de receta y la menor semilla final registrada, no por resultados:

| Objeto | Semilla | Tasa ya seleccionada | Parámetros | Oportunidad nominal |
|---|---:|---:|---:|---:|
| V3 `atom_crps` original | 20261118 | 0.003 | 1262 | 120 s |
| Control neuronal aditivo del mismo piloto | 20261118 | 0.003 | 1210 | 120 s |

No se repite HPO ni se cambia la receta. El alcance es **dos ajustes nuevos de
reproducción histórica**, no los 796 trabajos, los 30 ajustes o toda la familia.
El control aditivo de este piloto tampoco se renombra como la campaña ampliada.
Los 240 segundos son la suma de dos oportunidades nominales, no GPU consumida
ni tiempo total. El límite externo de reloj se declara aparte antes de ejecutar.

El controlador llama únicamente `fit_job` del motor histórico verificado;
**no llama `prepare` o `run` del launcher de la campaña antigua**. Los modelos
se inicializan desde su semilla y entrenan desde cero en un directorio nuevo.
No se cargan checkpoints antiguos, no se reanuda ni escribe ningún master.
Las 55 fechas históricas se puntúan después de terminar ambos ajustes, para
comparar 110 salidas contra su referencia; no se usan para parada o selección.
El calendario ya estaba expuesto y esta reproducción no es confirmación.

## Fuentes y referencias fijadas

`SOURCE_MANIFEST.json` fija nueve archivos de código/artefactos y tres
referencias por tamaño y SHA256. El motor y las formulaciones proceden de
`8f65eac34aaf49293924ae8328c235cedbdd2866`; la fase compartida conserva su
identidad de la fuente ampliada `5a07d6157a9796ca54416dc5dea1e1527e731597`.
El motor tiene SHA256
`6c54d1f35e305e10c5fc6882b4e410bb1fb31b53349efe3c340810a173dc3363`.

La referencia completa `assessment_predictions.csv` tiene 935 filas: 825
neuronales y 110 de dos controles analíticos. El subconjunto de este suplemento
usa 55 filas de cada receta/semilla prefijada. La referencia es un resultado
derivado del proyecto, no OHLCV crudo. Su tamaño y hash quedan fijados; no se
reescribe ni se convierten sus filas en observaciones independientes.

El NPZ original **no está incluido en el suplemento**. El manifiesto conserva
su URL inmutable, 460.090 bytes y SHA256
`186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
Su adquisición mantiene las condiciones de la fuente; no se asigna una licencia
nueva. El loader histórico sólo abre tres arrays de entrenamiento y no evalúa
los arrays de calibración o prueba contenidos en ese archivo.

## Inspección sin GPU ni imports científicos

Desde este directorio, preparar un caso nuevo sin descargar datos:

```bash
python3 prepare_C_case.py --case-root /tmp/e3-human-C-preparation-01
python3 historical_C_driver.py --case-root /tmp/e3-human-C-preparation-01
```

Es una inspección predeterminada: valida hashes y permisos, informa puertas
faltantes y no importa NumPy o Torch. El estado `prepared_C_subset_dry_run_only`
no significa que C se ejecutó. No crea carpetas de entrenamiento.

Para adquirir el input en **otro caso nuevo**, la persona puede elegir uno:

```bash
python3 prepare_C_case.py --case-root /tmp/e3-human-C-execution-01 \
  --fetch-input --max-download-bytes 500000
```

O usar una copia local, sin red, comprobada contra el mismo hash:

```bash
python3 prepare_C_case.py --case-root /tmp/e3-human-C-execution-01 \
  --input-file /ruta/current_cohort_arrays.npz
```

El helper copia fuentes/referencias/input con permisos de sólo lectura. No
escribe en el repositorio o en las campañas originales. Cada caso debe ser nuevo;
una preparación interrumpida se conserva para inspección y no se sobrescribe.

## Declaración externa previa

La persona copia `C_REQUEST.template.json` del caso a `C_REQUEST.json` y completa:

- Nombre, relación con el diseño y declaración real de independencia.
- ID nuevo `C-...`, fecha/hora UTC anterior a la ejecución y alcance de dos ajustes.
- `allow_training: true`, recursos y todos los reconocimientos de límites.
- Tolerancias absoluta/relativa para C y justificación antes de producir resultados.

La plantilla está bloqueada por defecto. **No se asigna a C la tolerancia de B
por herencia**, ni se inventa un umbral para garantizar concordancia. El código
registra máximos por receta, cada diferencia y si cumple la tolerancia declarada;
su aceptación no demuestra equivalencia científica, superioridad ni R3.

Las declaraciones de identidad/independencia son aportadas por la persona. El
software no las autentica ni produce una firma. El formulario humano de A/B
permanece separado y puede acompañar los logs reales de quien reproduzca.

## Ejecución futura explícita en la cohorte original

El proceso exige la cohorte histórica: Python 3.12.12, Torch 2.8.0+cu126,
NumPy 2.3.5, CUDA 12.6, cuDNN 91002, A100-SXM4-80GB de 85.094.825.984 bytes,
capacidad 8.0, dos hilos y TF32 deshabilitado. Otra cohorte requiere otro diseño
de reproducción declarado; este driver la rechaza antes de entrenar.

Se debe usar el intérprete del perfil GPU original, no el Python base de Colab.
Por ejemplo, si ese entorno ya fue preparado y coincide:

```bash
C_PYTHON=/content/e3-preflight-gpu-runtime-v1/venv-py312-cu126/bin/python
"$C_PYTHON" historical_C_driver.py \
  --case-root /tmp/e3-human-C-execution-01 \
  --request /tmp/e3-human-C-execution-01/C_REQUEST.json
```

Después de revisar la inspección y completar las puertas, **la persona externa**
puede ejecutar:

```bash
"$C_PYTHON" historical_C_driver.py \
  --case-root /tmp/e3-human-C-execution-01 \
  --request /tmp/e3-human-C-execution-01/C_REQUEST.json --execute
```

El driver guarda los bytes exactos del pedido antes del primer ajuste, lanza un
proceso limitado por el reloj declarado y crea sólo
`output/<nuevo-ID>/`. Conserva los checkpoints nuevos y logs incluso al fallar.
Un timeout detiene el grupo del proceso y no acepta checkpoints aleatorios.
El nombre de una salida antigua o existente se rechaza; no hay reanudación
automática ni función que pueda escribir en el master original.

## Resultados que debe conservar quien ejecute

`pre_execution.json`, `external_request.json`, `subset_plan.json`, `worker.log`,
los dos recibos de ajuste, `C_subset_comparison.csv`, `C_subset_execution.json`
y `driver_receipt.json`. Se comprueban de nuevo los hashes de las entradas al
terminar. El CSV contiene las 110 diferencias con fecha, receta, semilla y unidad.

La oportunidad de pared histórica incluía esperas de guardado remoto y posibles
interrupciones. Este caso fresco no reproduce una latencia histórica arbitraria;
por ello no garantiza idéntico número de épocas ni pesos bit a bit aunque la
GPU coincida. Esa limitación se declara antes de ejecutar y en el recibo.

Una eventual ejecución aprobada sólo acredita el alcance parcial que se haya
observado. No acredita por sí sola reproducción C completa, independencia de
mercados, defensa del autor o confirmación científica. El suplemento no cambia
las conclusiones adversas ni los artefactos históricos.
