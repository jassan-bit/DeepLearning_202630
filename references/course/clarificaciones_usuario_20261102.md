# Aclaraciones aportadas por el usuario — 2 de noviembre de 2026

Registro textual del material compartido en el chat; no sustituye la guía completa.

- Fuente recomendada: Binance Public API. En este entorno devuelve HTTP451 por ubicación; se adquirieron los archivos históricos oficiales de Binance Spot con checksums.
- Datos de origen: frecuencia1minuto, variable central Close.
- Fórmula histórica: sigma_t=sqrt(mean((r_(t-i)-media)^2)), i=1..n, divisor n; retorno logarítmico.
- Confirmación textual del usuario: entradas7,14,21,28 son días de historia, con1440cierres por día.
- Tabla posterior: frecuencia de modelado diaria, una observación por activo y día, cierreUTC; variablesOHLCVdiarias y derivadas causales; noticias agregadas con información anterior al pronóstico; horizonte principal1día adelante.
- Reporte: por tamaño de entrada, MAPE/MAE/RMSE/MSE, por horizonte y promedios, idealmente por fold y media/desviación. BDS de residuos del primer horizonte.
- Se analizan como candidatas las ventanas de volatilidad diaria7/14/21/28; no se eligió una ganadora. Las noticias no se han adquirido. La tabla no confirma horizontes adicionales al principal.
