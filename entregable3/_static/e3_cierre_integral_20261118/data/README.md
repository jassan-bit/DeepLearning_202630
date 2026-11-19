# Admisión de los tres archivos de evaluación ampliada

Preparado con asistencia de IA el 18 de noviembre de 2026. Este paso inspecciona datos y procedencia; no ajusta modelos, construye objetivos sigma7, genera predicciones ni ejecuta contrastes.

## Selección antes de observar rendimiento

Se conserva **D01 WIKI/AAPL RAW**, 2000-03-01–2013-03-01, ya admitido y verificado en 3270 sesiones. Se proponen dos reemplazos para los candidatos D02/D03 anteriores, todavía no admitidos: el acceso por cuenta y la limpieza causal de HF Data Library/SPY y el snapshot/licencia exactos del archivo Kaggle/Bitstamp seguían abiertos. La selección anterior era de metadatos y no un protocolo confirmatorio congelado. No se observaron resultados de modelos para elegir estos reemplazos.

| ID | Serie fijada para inspección | Periodo | Naturaleza y contrato |
|---|---|---|---|
| D01 | WIKI/AAPL distribuido por Bokeh, bytes ya fijados | 2000-03-01–2013-03-01 | Acción al contado, OHLCV bruto, siete retornos entre sesiones; siguiente sesión. |
| D02 | Binance Vision Spot ETHUSDT, archivos mensuales de velas `1d` | 2018-01-01–2021-12-31 | Altcoin al contado, ETH como unidad de volumen, días UTC; siguiente día. |
| D03 | Binance Vision COIN-M BTCUSD_PERP, archivos mensuales `1d` | 2020-09-01–2024-12-31 | Derivado perpetuo con margen en moneda, volumen nativo de contratos, días UTC; siguiente día. No se renombra el volumen como BTC. |

Los periodos no se eligieron por métricas. ETH cambia instrumento y calendario respecto a BTCUSDT Spot 2023–2026. El perpetuo cambia producto/mercado, contrato/denominación y periodo. La historia de D03 durante 2023–2024 se solapa con fechas de entrenamiento del benchmark BTC; esa exposición al subyacente se declara. **No son tres mercados estadísticamente independientes ni tres calendarios globalmente ciegos.** D02 y D03 comparten distribuidor y algunos shocks cripto. La diferencia entre acciones, altcoin al contado y perpetuos debe interpretarse con esos límites.

## Derechos y fuentes fijadas

La documentación y los términos oficiales de Binance Vision se fijan al commit `d571690eb66760763b0bd6e27e8cd35603c7e244`. Los archivos de `sources/` coinciden en bytes y objeto Git con ese commit. Los términos v1.0, actualizados el 26-08-2026, conceden `CC BY-NC-SA 4.0` y autorizan expresamente investigación académica y proyectos educativos no monetizados (§§3.1 y 4.1). La atribución y la misma licencia se conservan para derivados (§4.5); su alcance incluye indicadores, gráficos y modelos derivados (§2.4). **No se declara CC0, MIT ni uso comercial para los datos o sus derivados.** Los derechos del código propio se mantienen separados.

La referencia temporal y el esquema proceden del README oficial: las velas identifican su tiempo de apertura; los archivos de ambos periodos usan milisegundos y se verifica físicamente apertura a 00:00 UTC, cierre al último milisegundo del día, 12 columnas y calendario completo. En COIN-M la columna `Base asset volume` es distinta de la columna de volumen nativo utilizada. No se hace conversión artificial entre contratos, ETH y BTC.

Binance puede corregir archivos históricos. Por ello cada ZIP se descarga junto con su `.CHECKSUM`, se comprueba SHA-256 y se conserva la respuesta exacta. El manifiesto de esa ejecución fija finalmente los bytes; un nombre de URL por sí solo no es un snapshot inmutable.

## Ejecución CPU

`collect.py` ofrece `run(output, source_commit)`. Usa la biblioteca estándar de Python y descarga 48 meses Spot y 52 meses COIN-M, más sus sumas de control, con límites de tamaño y pausas. El resultado `admission.json` sólo marca un dataset como admitido cuando todos sus archivos, calendario, marcas temporales y OHLCV pasan. Un error de acceso o un mes ausente deja ese dataset bloqueado, conserva el diagnóstico y no rellena ni elimina barras. Puede reanudarse la descarga local verificada en una carpeta de ejecución nueva o interrumpida sin sustituir los archivos ya comprobados.

La salida contiene `datasets/D01.csv` y, cuando superan los controles, `D02.csv`/`D03.csv`, todos con encabezado `date,open,high,low,close,volume`. `admission.json` incluye rutas, hashes, unidades, reloj y rol `reserved_confirmation`. No fija por sí solo particiones, presupuesto o potencia, ni habilita la campaña real.

Desde este entorno las solicitudes normales a los servidores de precios fueron rechazadas por el proxy antes de devolver datos. Se conserva el sondeo real y se prepara la ejecución en Colab; no se declara que D02/D03 estén admitidos hasta inspeccionar allí los bytes. D01 se normaliza desde el archivo local fijado y su calendario admitido, sin usar `Adj Close` ni fabricar fines de semana.
