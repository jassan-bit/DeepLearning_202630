# Reparación acotada del encabezado COIN-M

La causa quedó reproducida con un archivo real del proveedor:
`BTCUSD_PERP-1d-2022-07.zip`, SHA256
`2b2c4aa1bb3a312c17c61fb4b7748f137e19eeed9a3c8834a2a6556d1bbaa017`.
El runner de GitHub verificó su compañero CHECKSUM y el calendario de 31 días.
Enero–junio de 2022 no tenían encabezado; julio contiene el siguiente encabezado
genérico exacto de doce columnas:

```text
open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
```

S2 rechaza el nombre `quote_volume` en la octava columna con
`AdmissionError: Wrong market-volume header`. Esa etiqueta genérica no cambia
la especificación física de COIN-M: el [README oficial fijado de Binance](https://github.com/binance/binance-public-data/blob/d571690eb66760763b0bd6e27e8cd35603c7e244/README.md)
define la columna 7, contando desde cero, como **Base asset volume** para
`/dapi/v1/klines`; la columna 5 cuenta contratos. La [incidencia upstream 283](https://github.com/binance/binance-public-data/issues/283)
ya mostraba esa misma cabecera genérica en un archivo COIN-M. El diagnóstico
directo de julio de 2022 confirma el caso concreto; la incidencia por sí sola
no se presentó como prueba de ese archivo.

## Cambio mínimo

`cm_header_parser.bind_parser(original_data)` devuelve una función con la firma
original. Exige el SHA256 exacto de `engine/data.py` de S2:
`1e64a500a62b3ec803471f87a2a81161a81d0e73d2c4cce7fd17d73b2b044e60`.
El cliente operacional instala expresamente esa función sólo en su subprocess
y registra el SHA del complemento. No se edita ni se vuelve a firmar S2.

Únicamente para `market_type='cm'` y la primera fila exacta de arriba, prepara
una representación temporal en memoria con el nombre de la columna 7 que
acepta S2, y llama a su parser numérico **COIN-M original**. No infiere unidades
por el nombre genérico, no usa la columna de contratos como volumen BTC y no
modifica las líneas de observaciones. Los ZIP originales, los CHECKSUM y sus
hashes permanecen intactos en disco.

Los demás encabezados se delegan al parser original. No se amplía la lista a
sinónimos por conjetura ni se salta cualquier primera fila textual. Se conservan
las comprobaciones de cantidad de columnas, fechas, cierre de barras, volumen,
contratos, calendario y procedencia. Spot, USD-M, COIN-M sin cabecera y las
cabeceras COIN-M ya admitidas conservan su comportamiento anterior.

La constante de versión del adaptador, los recibos de derechos/exposición y el
manifiesto pendiente conservan sus valores. El complemento declara su propia
versión y huella en el sobre operacional. Sus eventos sólo contienen hash del
ZIP, encabezado, unidad/índice y fuente S2; no contienen OHLCV ni credenciales.
Los sobres deben guardarse fuera del directorio raw, cuyo inventario de
archivos está controlado por S2.

## Verificación

`PARSER_TEST_OUTPUT.txt` documenta once pruebas con observaciones generadas y la
cabecera real. Comparan arrays y metadatos contra el parser S2 sobre una
cabecera canónica, conservan exactamente la columna 7, rechazan encabezados no
documentados y valores inválidos, comprueban la integridad de las líneas de
datos y mantienen la comprobación de calendario del colector.

Para repetirlas con el Python CPU que contiene NumPy fijado:

```bash
python -m unittest -v test_cm_header_parser.py
```

`CM_REPAIR_SOURCE_DATA` permite indicar el `engine/data.py` original de S2; el
test exige su huella exacta. La referencia real del encabezado se toma de
`../REAL_CM_HEADER_DIAGNOSTIC.json`.

La [ejecución del reproducer real del complemento](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/45024240754)
verificó enero y julio de 2022: en enero mantuvo arrays y metadatos idénticos a
S2; en julio el parser S2 rechazó la cabecera y la reparación obtuvo las 31
filas, con OHLC y volumen de la columna 7 iguales al CSV original. No publicó
precios ni volúmenes. Sus recibos son `../REAL_CM_HEADER_DIAGNOSTIC.json` y
`../REAL_PARSER_VERIFICATION.json`.

No hubo entrenamientos, cambios de reglas o aperturas de prueba reservada.
Asistencia de IA declarada en implementación, pruebas y revisión.
