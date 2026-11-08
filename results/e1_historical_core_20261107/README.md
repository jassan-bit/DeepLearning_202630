# Núcleo histórico de cuatro entradas: LSTM–MLP–GRU

Este suplemento concilia métricas por semilla conservadas de dos estudios ya
realizados. Usa MLP/LSTM de `filipe_multihistory_20261104` y GRU/LSTM de
`gru_lstm_tcn_20261105`; los resultados LSTM coinciden exactamente en las seis métricas
en las cuatro entradas, CV y test (480 igualdades). No entrena ni abre observaciones, predicciones
diarias, pesos o escaladores. Las fuentes permanecen intactas y se verifican sus
hashes antes de calcular los agregados.

Las tres redes comparten D=7/14/21/28, cinco entradas, representación `relative_window`,
objetivo sigma7/h1 y semillas nominales 4101–4110. CV comprende las 366 fechas de
2024; test, las 608 fechas del 01/01/2025 al 31/08/2026. Las recetas históricas
tienen 5.025 parámetros LSTM y 3.777 GRU en cada D. La MLP histórica usa 16
unidades y tiene 593/1.153/1.713/2.273 parámetros al variar D=7/14/21/28.
No son las recetas del estudio
nuevo del núcleo ni se convierten en resultados de 465 semillas o del periodo de
septiembre–octubre de 2026. Los resultados de TCN se conservan por separado como
estudio complementario.

`run_metrics.csv` contiene 240 puntuaciones de corrida (2 fases × 4 entradas × 3 redes × 10
semillas), no vectores de pronóstico. `metric_summary.csv` resume las seis métricas
con media, DE muestral e IC t marginal del 95 %, df=9. Son intervalos entre
inicializaciones condicionados a un calendario conocido, sin cobertura simultánea
ni incertidumbre temporal incorporada. MAPE conserva sus unidades porcentuales.

El mínimo de RMSE media CV se deriva de las cuatro entradas guardadas por familia;
las tres favorecen D7 en esta evidencia histórica. No se elige D mirando test ni
se atribuye esta síntesis posterior al cierre del estudio nuevo.

`paired_rmse.csv` informa únicamente para D7 tres contrastes t bilaterales por fase, diferencia
orientada como A−B, IC marginal, d_z, Shapiro de diferencias y Holm dentro de cada
familia de tres. Es un análisis posterior y exploratorio de etiquetas-semilla;
normalidad e independencia condicional son supuestos, no hechos demostrados por
el emparejamiento. No se suman las fases como confirmaciones independientes ni se
elige una prueba por su p-value. La comparación final nueva tiene su propio
protocolo, piloto, dimensionamiento y resultados, fuera de este suplemento.

Sin los vectores diarios originales MLP de test no se recalculan DM, residuos,
BDS o bootstrap temporal del núcleo desde observaciones. Las métricas guardadas
permiten esta comparación por semilla, pero no reconstruyen esos artefactos.

Reproducción: `python scripts/build_e1_historical_core.py` genera este directorio
y el espejo del libro; `--check` verifica las salidas guardadas. Use el entorno
científico con SciPy fijado del proyecto. No hay ejecución de modelos ni cambios
en las fuentes históricas.
