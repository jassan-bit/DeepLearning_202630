# Paquete para una reproducción humana con alcance explícito

Preparado con asistencia de IA. No contiene una firma ni afirma que haya
ejecutado los pasos una persona externa. Tampoco ejecuta entrenamientos.

## Estado y alcance

| Nivel | Comprobación previa conservada | Acción de esta entrega |
|---|---|---|
| A: tablas desde predicciones congeladas | Once tablas, 431 filas y discrepancia por campo/unidad cero en el recibo automático histórico. | Se reutilizan recibos y hashes; no se recalcula el análisis. Comandos preparados para la persona externa. |
| B: predicciones con pesos aceptados | Quince checkpoints/825 salidas; además, ONNX V3 público aceptado sobre 55 fechas con tolerancia 1e-6 sigma7. | Se reutiliza el replay aceptado, sin reexportar, entrenar o usar nuevas fechas. Comandos preparados para la persona externa. |
| C: entrenamiento desde cero | Fuentes, planes y recibos históricos de 796 trabajos y 30 ajustes. | No ejecutado. Requiere alcance humano, cohorte y salidas nuevas; el replay B no reconstruye entrenamiento. |

`SOURCE_RECEIPTS.json` fija 27 archivos de fuente, recibos y referencias al
commit **8d03a73d529eb3f9b2e89992b4488cfeb88a99e6**. Su inspección verifica
bytes/SHA256 sin cargar modelos. Los doce perfiles de comparación se derivan
de los once CSV A y del CSV de 55 fechas B realmente existentes.

## Preparar un caso nuevo sin ejecutar A/B/C

Requisitos de preparación: Python estándar, checkout de la fuente exacta y este
directorio extraído. La persona externa debe obtener el checkout fijado; no usar
un checkout cambiante de main. Ejemplo de adquisición en una carpeta nueva:

```bash
git clone --filter=blob:none --no-checkout https://github.com/jassan-bit/DeepLearning_202630.git e3-human-source
git -C e3-human-source fetch --depth 1 origin 8d03a73d529eb3f9b2e89992b4488cfeb88a99e6
git -C e3-human-source checkout --detach 8d03a73d529eb3f9b2e89992b4488cfeb88a99e6
```

El repositorio completo puede ser grande; el archivo de esta entrega distingue
las fuentes materialmente incluidas de sus localizadores. Se necesita espacio
suficiente para el checkout elegido. Una descarga de GitHub no establece una
licencia nueva sobre las fuentes externas.

Desde este directorio, sustituya `/ruta/e3-human-source` por el checkout fijado:

```bash
python3 inspect_kit.py --source-root /ruta/e3-human-source --output /tmp/e3-human-inspection.json
python3 prepare_case.py --source-root /ruta/e3-human-source --case-root /tmp/e3-human-B-persona --level B
```

Esos comandos sólo inspeccionan hashes y preparan un caso aislado. El segundo
crea `commands.sh`, un formulario sin firma y un recibo de preparación. Para A
utilice otra carpeta nueva y `--level A`. No borre recibos anteriores ni use la
misma carpeta para ambos niveles. No existe una opción C que dispare entrenamiento.

## Ejecutar el alcance elegido: intervención real de la persona externa

La persona revisa el archivo de comandos y sus hashes, declara su relación con
el proyecto y ejecuta **en su propio entorno**:

```bash
bash /tmp/e3-human-B-persona/commands.sh > /tmp/e3-human-B-persona/operator.log 2>&1
```

Linux x86_64, Python 3.12/pip y Docker son necesarios para B. El procedimiento
descarga cuatro entradas públicas fijadas —485.747 bytes en total— y nueve
ruedas con versiones y hashes bloqueados, obtiene la base Docker por digest,
construye sin red y ejecuta sin red, con raíz/entradas de sólo lectura, usuario
sin privilegios, un CPU y 256 MiB. No instala Torch ni exporta ONNX. El modelo
V3 sigue experimental y no tiene calibrador. La descarga y el forward son pasos
futuros de quien reproduzca, no resultados ejecutados por preparar este paquete.

A usa el generador científico histórico en Docker CPU; recalcula sus agregados
desde predicciones guardadas. No reproduce sus entrenamientos ni crea una
confirmación nueva. Su presupuesto y dependencias se conservan en el perfil
`deployment/science` de la fuente fijada.

## Comparación y tolerancias anteriores a la ejecución humana

`compare_csv.py` rechaza referencias alteradas, cambios de columnas, pérdida o
duplicación de filas, cambios de identificadores/calendario/orden, NaN/Inf y
sustitución de ausencias. Produce máximos y ubicación por campo y unidad,
número de pares y violaciones; no mezcla distintas unidades en un máximo global.

- A conserva `abs(observado−referencia) <= 1e-14 + 1e-12*abs(referencia)`.
- B conserva tolerancia absoluta **1e-6 sigma7** para salidas y errores derivados.
  Las columnas de predicción de referencia se mantienen exactas; A conocido y
  piso tienen tolerancia de reconstrucción float64 1e-14 + 1e-12 relativa.
  El error declarado debe coincidir con la diferencia calculada y se comprueba
  el piso estructural.
- Conteos e identificadores de semillas deben conservar sus valores enteros.
- La igualdad de bytes se informa por separado de las tolerancias numéricas.

Ejemplo manual B tras la ejecución de quien reproduce:

```bash
python3 compare_csv.py --profile B.accepted55 --reference /ruta/e3-human-source/experiments/e3_reproduction_followup_20261120/execution/discrepancies.csv --observed /tmp/e3-human-B-persona/output/discrepancies.csv --output /tmp/e3-human-B-persona/comparison_extra.json
```

Estas tolerancias comprueban reproducción numérica; no son márgenes de
equivalencia científica ni autorización para aceptar cualquier entrenamiento.

## C y la constancia independiente

`C_REPRODUCTION_PLAN.json` identifica ambas campañas, fuentes, semillas, GPU y
software originales. Para el alcance completo, la suma nominal de oportunidades
de ajuste es **27 horas** (26 para 780 ajustes neuronales/ARIMA del master y una
para 30 ajustes V3). No es el consumo GPU real ni el tiempo total; controles
analíticos, preparación, descargas, fallos e I/O tienen otro alcance. El límite
por reloj de 120 s puede cambiar el número de épocas incluso en hardware similar.

Antes de C se necesitan alcance/tolerancias propios, presupuesto, identidad de
la persona, nuevas rutas e ID de reproducción y un controlador que no escriba
en los masters históricos. El plan declara explícitamente que ese controlador
de entrenamiento nuevo no está suministrado por el replay B. No ejecute celdas
de reanudación sobre el master original para simular una reproducción nueva.
La decisión NO-GO de confirmación de V3 no se modifica con una reproducción.

La persona completa `INDEPENDENT_REPRODUCTION_FORM.md` con sus propios logs,
recibos y discrepancias. La excepción de IA comunicada para revisión R1 no
supone exención automática de este requisito humano ni acredita R3.

## Validación ejecutada aquí

`validation/source_inspection.json` verifica 27 fuentes sin abrir modelos.
`validation/comparator_contracts.json` documenta doce perfiles idénticos y ocho
casos de corrupción rechazados. Se ejecutaron comparaciones de CSV, no modelos,
entrenamientos, nuevas inferencias ni reconstrucciones de las once tablas.
