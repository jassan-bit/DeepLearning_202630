# Admisión física dedicada, sin depender de Render

Este paso usa sólo CPU y la biblioteca estándar de Python. Reutiliza el
recolector ya publicado, verifica los tres archivos y guarda sus recibos en una
rama nueva de GitHub. No instala Playwright, no comprueba Render y no utiliza
GPU. La admisión precede a la evaluación; no construye objetivos sigma7, entrena
modelos o acredita R1/R3.

## Corte inicial conservado

Las búsquedas exhaustivas de los prefijos exactos `e3-closure-cpu-` y
`e3-closure-pilot-` no devolvieron ramas. Se conserva el resultado en
`remote_receipts_check.json`; esto no pretende describir todos los demás
registros históricos del repositorio.

La ejecución real del recolector publicado en `18429f6` dejó en `local_probe/`
**D01 admitido con 3270 filas y D02/D03 bloqueados por el proxy**, con una
respuesta 403 antes de recibir datos. Ese 403 describe el acceso de este
entorno, no demuestra que falten precios o que la fuente esté caída. No hubo
ajustes, inferencias de modelos o contrastes. El CSV D01 normalizado es
`local_probe/datasets/D01.csv`; sus bytes están ligados a `admission.json`.

## Admisión posterior y recuperación del calendario

La ejecución Colab `20261119T022547490732Z`, con fuente `0f34c97`, sí está guardada en la rama `e3-closure-cpu-20261119T022547490732Z`. Admite D01 y D02; D02 tiene 1461 días UTC y 48 ZIP mensuales con checksum verificado. No se entrenaron modelos.

D03 se detiene en agosto de 2023: su ZIP mensual coincide con el checksum oficial, pero contiene sólo los días 1–27. `incident_202308.json` vincula el recibo y los hashes. La recuperación usa el mes completo de archivos diarios oficiales con sus checksums, exige el calendario íntegro y acuerdo en todos los días comunes; un archivo diario ausente o discrepante mantiene el bloqueo. No se rellenan valores ni se acorta el periodo. Los bytes y recibos fallidos se conservan en su intento original; los archivos cacheados se revalidan en una admisión nueva.

En este entorno el acceso directo a Binance Vision sigue bloqueado por el túnel. Las pruebas del recuperador no acreditan que ya se hayan obtenido los diarios reales; esa comprobación ocurre en Colab.

D01 representa WIKI/AAPL bruto entre sesiones. D02 es ETHUSDT Spot 2018–2021;
D03 es BTCUSD_PERP COIN-M 2020-09–2024. Los dos últimos comparten distribuidor
y exposición cripto; D03 además comparte subyacente y fechas conocidas con
el BTC original. No son tres mercados independientes o globalmente ciegos.
Usar sus resultados para rediseñar una receta convierte ese uso en desarrollo,
no en validación confirmatoria nueva.

## API que integra el cuaderno de evaluación ampliada

```python
receipt = colab.run_stage(checkout, output, full_source_commit)
```

La salida contiene `data/admission.json`, fuentes, ZIP/CHECKSUM verificados,
datos normalizados y un `stage_receipt.json` compatible con el guardado
existente. Todo informe positivo queda ligado al hash y tamaño de su CSV;
una admisión 3/3 conserva `real_campaign_allowed=false` hasta resolver por
separado el protocolo de modelos, presupuesto y potencia.

```python
result = colab.run(full_source_commit, github_token,
                   resume_output='/content/e3-admission-output')
```

El controlador obtiene un checkout disperso con SHA exacto, comprueba el
acceso a GitHub y llama al guardado existente `profile='cpu'`. Las ramas se
llaman `e3-closure-cpu-<fecha>`, bajo `runs/e3_closure_cpu/<fecha>`. El token
permanece en memoria y se excluye de los comandos Git y del entorno hijo.
Requiere Contents: Read and write para este repositorio; no se pega en código
o chat. El cuaderno principal proporciona el token mediante Secretos/entrada
oculta y conserva la ruta de reanudación.

Un recibo completo, incluso si sólo admite 1/3, se mantiene inmutable para
reintentar su guardado. Para repetir una admisión fallida después de resolver
la causa se inicia otra ejecución con una ruta nueva. Los resultados y
checkpoint científicos históricos no se modifican.

La implementación y sus pruebas se prepararon con asistencia de IA declarada.

## D00: historia diaria auxiliar de BTC ya conocida

`d00_history.run(output, npz=<NPZ conocido opcional>)` obtiene por GitHub raw
dos CSV inmutables fijados a `18429f6`, reutilizando sus SHA originales de E1/E2.
Devuelve `(paths, report)`; `paths` contiene desarrollo y evaluación, en ese
orden, con encabezado `market_day_utc,open,high,low,close,volume` y `trades`
adicional sólo en desarrollo. `d00_inputs/d00_history_manifest.json` conserva
URLs, tamaños, hashes, calendario, ausencias y comprobación de cohortes.

Los **1339 días** abarcan 2023-01-01–2026-08-31. La barra incompleta de
2023-03-24 se conserva vacía; no se elimina, rellena o interpola. Sólo se
reconstruye sigma7 cuando existen ocho cierres consecutivos válidos. Esto da
1324 etiquetas disponibles, incluida la historia omitida por las purgas del
NPZ. Una predicción causal debe restringir su consulta a fechas anteriores a
su objetivo; disponer del CSV completo no autoriza a leer el futuro.

La comprobación independiente coincide con las **1316 etiquetas conocidas**
de entrenamiento655/calibración53/evaluación608, con máxima diferencia de
sigma7 `1.2576745200831851e-16`. Todos los precios OHLC son bitidénticos.
Entrenamiento/calibración coinciden también en volumen; el CSV de evaluación
difiere del NPZ por hasta **un ULP float64 de volumen** (`7.275957614e-12`),
que se registra sin modificar los bytes. Los clásicos usan cierres; los
modelos neuronales D00 deben seguir utilizando los arrays NPZ originales.

Esta historia BTC es **auxiliar D00 y ya expuesta**. No suma un cuarto archivo
a la admisión 3/3, no abre objetivos de D01/D02/D03 ni cuenta como mercado nuevo.
Reconstituye datos históricos; no entrena o ejecuta modelos y no acredita R3.
Los derechos y procedencia E1/E2 permanecen separados de los nuevos términos
de Vision. Cuatro pruebas stdlib comprueban caché fijado, huecos preservados,
rechazo de corrupción y calendarios, además de la paridad real del manifiesto.
