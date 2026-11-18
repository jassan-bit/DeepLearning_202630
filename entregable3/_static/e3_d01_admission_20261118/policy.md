# D01: tarea admitida de precio bruto por sesión

D01 AAPL queda admitido como archivo para una tarea explícita: siete barras
OHLCV de sesiones completadas y pronóstico, al inicio de la siguiente sesión,
de la desviación poblacional de siete retornos logarítmicos de cierres brutos
que terminan en esa sesión objetivo. Se usa ddof=0 y no se anualiza. El último
cierre del objetivo sigue siendo desconocido al emitir el pronóstico.

La política RAW_KEEP_ALL conserva todas las observaciones de la fuente, sin
ajustar, eliminar ni rellenar precios o volumen. Adj Close no se usa como
entrada ni como objetivo porque el archivo incorpora ajustes posteriores al
periodo. No se etiquetan automáticamente como splits los saltos de precio:
un desplome real también debe permanecer. Se conservan, entre otros, los
registros 2000-06-21, 2000-09-29 y 2005-02-28.

Esta definición mide volatilidad del precio cotizado bruto e incluye efectos
mecánicos de acciones corporativas. No se presenta como volatilidad económica
ajustada o de retorno total. Esa limitación forma parte del contrato y se
mantendrá al interpretar resultados; no se eligió por desempeño predictivo.

La unidad temporal es la sesión, no un día UTC. Las 3270 fechas coinciden en
orden con exchange-calendars 4.11.3 NASDAQ, alias documentado de XNYS. No hay
sesiones faltantes ni fechas adicionales; el calendario conserva cierres
extraordinarios de 9/11, Reagan, Ford y Sandy. La comparación no certifica
horarios intradía ni representa dos calendarios independientes.

La admisión de un archivo no congela el protocolo de modelos: faltan otros
dos datasets y el tratamiento, particiones, presupuesto y potencia comunes.
E2 y E3 ya están entrenados; sus checkpoints y resultados se conservan. Esta
revisión no solicita reentrenarlos automáticamente, no calcula rendimiento
y no acredita R1/R3 ni reproducción humana independiente.
