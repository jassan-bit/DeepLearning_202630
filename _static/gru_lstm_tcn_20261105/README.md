# Evidencia agregada de GRU, LSTM y TCN

Este paquete sustituye la terna del contenido experimental principal. Reúne
métricas agregadas, estadísticos diagnósticos y metadatos de configuración.
Se excluyen pesos, escaladores, archivos de mercado y series diarias o vectores
individuales en formatos legibles por máquina. Se publica una ilustración PNG
(`predictions_D7.png`) con σ7 observado frente a las predicciones de la semilla
4101 fija durante las 608 fechas del test conocido; no se entrega su CSV diario
ni se seleccionó esa semilla por tener el menor error.

Los ajustes originales y sus firmas permanecen intactos. La selección GRU/D7
se realizó en CV antes de ejecutar los modelos nuevos sobre el test histórico,
que ya se conocía. Los contrastes entre los tres modelos de este paquete y los
diagnósticos adicionales se definieron DESPUÉS del test: son exploratorios.

`protocols_manifest.json` y `source_manifest.json` conservan fechas y hashes;
`public_analysis_plan.json` declara el alcance posterior. Las tablas de
contrastes usan diferencia RMSE(A)−RMSE(B): un valor negativo favorece A.
Se ofrece también la ganancia B−A y sus intervalos con nombres explícitos.

No existe piloto prospectivo para dimensionar las diez semillas de GRU/TCN.
El presupuesto fijo no acredita potencia 0,80. Las fechas, folds y semillas
no son repeticiones independientes de mercados.

El nuevo margen operativo de futuros trabajos está en `future_margin.json`.
Se declara ahora, después del test, y no se aplica retrospectivamente para
convertir los resultados actuales en confirmatorios. Persistencia conserva
su papel de referencia explícita con la misma cohorte y métricas.

Reproducción de esta agregación: `scripts/build_gru_lstm_tcn_evidence.py`
con las predicciones y configuraciones privadas originales identificadas por
SHA256. El programa no entrena ni vuelve a inferir ningún checkpoint.
