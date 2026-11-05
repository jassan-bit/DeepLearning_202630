# Descargas vigentes del Entregable 1

Los ocho archivos ZIP conservan los nombres de los botones de la unidad8. Los siete productos y el paquete completo describen la terna actual GRU–LSTM–TCN.


Terna experimental: **GRU–LSTM–TCN**. El primer benchmark tiene como baseline neuronal **GRU/D7**, seleccionado en CV antes de esa evaluación del test histórico ya conocido. Fuente versionada: `ff3631f0711dad55522b1dff618f468e9849d8a2`.

En ese primer benchmark, las comparaciones entre los tres modelos y los diagnósticos adicionales se definieron después de observar el test: su alcance es exploratorio. Sus diez semillas son un presupuesto fijo; no se atribuye a esa fase un piloto prospectivo ni potencia 0,80. Su margen operativo futuro se declaró después del test y no se aplica retroactivamente. Persistencia conserva su papel de referencia.

Los paquetes públicos contienen código, metadatos, agregados y figuras. Se excluyen archivos crudos, pesos, escaladores y series diarias o vectores individuales legibles por máquina. La ilustración PNG de la **fase I** utiliza la semilla fija 4101; la de la **fase II** utiliza la semilla fija 8101. Ninguna semilla se elige por su error y no se entregan sus CSV individuales. El adaptador público de reproducción de la **fase I** es posterior a su ejecución original y no sustituye sus firmas históricas. El código científico de la fase II tiene su propio registro previo a esa ejecución.

La licencia MIT incluida corresponde a timeseries-cv; no asigna una licencia general al proyecto. Los términos de la fuente de datos se conservan en el producto 8.3.


**Segunda fase D7 planificada:** el paquete también contiene el diseño y código registrados antes de ejecutar HPO, el mismo presupuesto de búsqueda por familia, un piloto separado, dimensionamiento por potencia condicional entre semillas y evaluación preespecificada del 01/09 al 30/10/2026. Sus resultados y cronología están en `book/_static/gru_lstm_tcn_planned_20261105/`. La reserva es histórica: no son pronósticos emitidos en vivo ni una validación en otro mercado. La potencia de planificación entre semillas no acredita potencia temporal del periodo. Las conclusiones del primer benchmark conservan su alcance exploratorio.

Verificar integridad con `sha256sum -c SHA256SUMS.txt`. El manifiesto interno de cada ZIP verifica además bytes y SHA256 por miembro.
