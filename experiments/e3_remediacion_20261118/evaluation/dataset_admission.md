# Admisión de datasets: D01 admitido, dos pendientes

**Estado posterior: D01 admitido para la tarea RAW de sesiones; D02 y D03 pendientes.** El inventario que sigue conserva el corte inicial anterior a la inspección del archivo. Sus bloqueos se resuelven para D01 en la sección final y en la decisión versionada; no representan el estado actual de ese candidato.

**Fecha de auditoría: 2026-11-18 (UTC).** Se proponen tres candidatos a partir de metadatos públicos, procedencia y términos documentados. No constituyen una selección final prerregistrada ni una validación técnica. No se descargaron archivos de mercado, se ajustaron modelos ni se calcularon métricas. **No se observaron resultados de rendimiento de estos candidatos en esta búsqueda**; esto no afirma desconocimiento global, ausencia de resultados publicados ni ausencia de exposición previa fuera de esta búsqueda.

El inventario estructurado está en [dataset_candidates.json](dataset_candidates.json). Los tres estados son `no_admitido`; una licencia declarada o una descarga gratuita no bastan para cerrar derechos upstream, procedencia, calendario y calidad del archivo.

## Criterios preparatorios anteriores a cualquier resultado

La referencia local es Binance Spot BTCUSDT, día UTC, entrada de siete velas OHLCV y objetivo sigma7: desviación estándar poblacional de siete retornos logarítmicos diarios de cierre, sin anualización, con horizonte de un día. Su evaluación conocida abarca 2025-01-01 a 2026-08-31; este inventario no la convierte en evaluación nueva.

Se buscan series financieras con la misma construcción de volatilidad y al menos dos diferencias identificables frente a esa referencia: fuente o mercado, instrumento o clase de activo, calendario o reloj de negociación, y longitud verificable de historia. Cambiar sólo BTC por ETH, BNB o SOL en la misma fuente y periodo no establece dos ejes. No se usan resultados predictivos para elegir candidatos, fechas o variantes. No se cuenta como diferencia demostrada un supuesto mayor ruido o volatilidad: todavía no se midieron.

Los periodos y variantes siguientes son propuestas de investigación. Antes de una selección final deben fijarse tamaño mínimo, fuente, versión, fechas, política de ajustes, agregación, faltantes y particiones. Tres series tampoco equivalen a tres réplicas estadísticamente independientes.

## Tres candidatos y evidencia disponible

| Candidato | Fuente, licencia y acceso observados | Diferencias frente a BTC Binance 2023–2026 | Estado |
| --- | --- | --- | --- |
| **D01. WIKI/AAPL diario**, periodo propuesto 2000-03-01–2013-03-01 | El contenido indexado de [Nasdaq WIKI](https://data.nasdaq.com/databases/WIKIP) declara dominio público; la apertura directa devolvió **404**. [Bokeh 3.10.2](https://docs.bokeh.org/en/3.10.2/docs/reference/sampledata.html#stocks) e [inventario de licencias](https://raw.githubusercontent.com/bokeh/bokeh_sampledata/main/licenses/datasets.json) fueron accesibles y documentan procedencia WIKI, esquema, fechas y hashes publicados. El [aviso histórico de Quandl](https://web.archive.org/web/20150411011214/https://www.quandl.com/data/WIKI) referenciado por Bokeh no fue accesible directamente. No se fija todavía commit ni paquete. | Fuente; acción frente a cripto; calendario anterior; sesiones bursátiles frente a días UTC. | **No admitido.** Declaración de derechos localizada, cadena documental y validación técnica pendientes. |
| **D02. HF Data Library / IEX SPY**, V2 Clean, minuto a sesión diaria; periodo propuesto 2022-03-07–2026-11-17 | [Editor](https://hfdatalibrary.com/), [licencia](https://hfdatalibrary.com/pages/license), [términos](https://hfdatalibrary.com/pages/terms) y [términos históricos oficiales de IEX](https://www.iex.io/legal/hist-data-terms) accesibles. CC BY 4.0 cubre compilación y documentación; **IEX conserva los derechos de los datos subyacentes**. El editor limita el uso a investigación académica o educativa y exige conservar condiciones upstream. SPY figura entre los tickers; su cobertura efectiva no se verificó. | Fuente y venue; ETF frente a cripto; reloj de sesiones regulares. Parte de los años coincide con la referencia: **no es un calendario disjunto**. | **No admitido.** Compatibilidad académica condicionada a términos; archivo, acceso, causalidad de limpieza y agregación pendientes. |
| **D03. Zielak / Bitstamp BTCUSD**, minuto a día UTC; calendario propuesto 2012–2022 si existe cobertura suficiente | La [tarjeta primaria de Kaggle](https://www.kaggle.com/mczielinski/Bitcoin-historical-data), consultada mediante contenido **indexado**, declara **CC BY-SA 4.0, no CC0**. La versión 713 apareció indexada, sin fijarse. El [repositorio del publicador](https://github.com/mczielinski/kaggle-bitcoin) confirma procedencia Bitstamp y actualización diaria. La [documentación API de Bitstamp](https://www.bitstamp.net/api/) exige un Data License Agreement para uso comercial por empresas. | Exchange y fuente; calendario anterior propuesto; adicionalmente USD frente a USDT. La longitud y el efecto del venue todavía no se midieron. | **No admitido.** Derechos upstream, versión, cobertura y agregación UTC sin cerrar. |

La indicación de accesibilidad describe la consulta web de metadatos y documentación; no demuestra acceso operativo a los archivos ni verifica sus bytes. Los hashes publicados por terceros todavía no se recalcularon.

## Cambio de tarea: sesiones frente a días UTC

En AAPL y SPY, una adaptación coherente sería calcular sigma7 con siete retornos entre **sesiones** y pronosticar la **próxima sesión**. Conserva una tarea financiera homóloga, pero cambia el reloj respecto a siete días UTC consecutivos y el próximo día de calendario. No deben rellenarse fines de semana o festivos con OHLCV inventado. La duración temporal efectiva de las ventanas y los bloques estadísticos debe quedar explícita.

Si el protocolo exige estrictamente el contrato UTC original, D01 y D02 deben excluirse y la búsqueda de fuentes cripto debe continuar. Tampoco se presupone UTC en D03: deben comprobarse las marcas temporales y las fronteras de agregación. Los precios ajustados no pueden sustituir sólo al cierre y dejar OHLC incompatible.

## Bloqueos exactos antes de admitir

**D01:** fijar commit o versión de paquete, archivo y SHA-256; recuperar el aviso histórico del editor o evidencia equivalente que cierre la trazabilidad; comprobar filas, fechas, faltantes por sesión, duplicados, precios y volumen válidos y coherencia OHLC. El extracto declara OHLC sin ajustar y `adj_close` con ajustes corporativos posteriores al periodo: falta una política causal de acciones corporativas. Deben resolverse el contrato de sesiones, el tamaño utilizable y las particiones antes de modelar.

**D02:** fijar snapshot y archivo SPY con su versión de limpieza y SHA-256; comprobar inicio, fin y número de sesiones efectivos. La portada menciona descargas básicas sin cuenta, mientras los términos exigen cuenta para descargar: el acceso operativo sigue pendiente y no debe eludirse. Falta definir agregación a sesión, zona `America/New_York`, horario de verano, festivos y medias sesiones. Debe auditarse si la limpieza V2 emplea datos futuros, así como splits, unidades de volumen, huecos y OHLC. No se sustituyen sigma7 por medidas de volatilidad precalculadas. Deben conservarse los términos y atribución obligatorios de IEX y del editor y fijarse tamaño y particiones.

**D03:** resolver derechos upstream para el uso académico, la transformación, la publicación de derivados y la redistribución exactos previstos; CC BY-SA del distribuidor no concede derechos ajenos. Fijar versión, archivo, licencia acompañante y SHA-256; comprobar primera y última observación, marcas temporales, zona, columnas y volumen. Definir agregación a día UTC y tratamiento de minutos sin transacciones, huecos, duplicados y días incompletos. Faltan controles de precios, volumen, OHLC y reconstrucción de sigma7, además del tamaño mínimo, particiones y embargo. No procede descargar un archivo grande para resolver sólo metadatos.

El cierre común exige un manifiesto de procedencia y derechos, archivos inmutables inspeccionados, contrato temporal y de objetivo explícito y un protocolo fijado antes de observar resultados. Este inventario no autoriza ni acredita entrenamiento, validación técnica, selección final o superioridad fuera del calendario BTC conocido.


## Estado posterior de D01

El inventario anterior corresponde al corte preparatorio. La revisión posterior admite D01 para la tarea RAW_KEEP_ALL de siete retornos por sesión y horizonte próxima sesión: `experiments/e3_d01_admission_20261118/admission_decision.json`. El archivo y las 3270 sesiones se verificaron, sin eliminar ni rellenar filas y sin usar Adj Close. Se conserva el efecto mecánico de acciones corporativas como limitación de volatilidad de precio cotizado bruto. Estado actual: **1 de 3 datasets admitidos, cero evaluados**. El protocolo de modelos y los otros dos conjuntos siguen pendientes. El informe original guardado en `648f2b91b477b5f4226cad3af5928088accd74ed` no se modifica.
