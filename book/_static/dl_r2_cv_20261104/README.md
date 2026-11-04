# Validación forward chaining de Filipe

Se ejecutaron MLP, LSTM y CNN1D con doce bloques mensuales de2024, entrenamiento creciente desde2023 y diez semillas por modelo/representación. Las variantes relativa y absoluta comparten información, cortes y configuraciones.

La representación principal es relative_window. El archivo method.json resume el procedimiento; folds.json contiene los índices realmente consumidos. runs.csv y predictions.csv conservan60trayectorias y21960predicciones. El paquete run_artifacts.tar.gz conserva5940archivos y720checkpoints con hashes.

El análisis publicado corresponde únicamente a esta CV. Los tres contrastes ajustados conHolm comparan representación relativa y absoluta con semillas emparejadas. DM/BDS/Ljung–Box ybootstrap temporal de estas predicciones siguen pendientes. El piloto y el dimensionamiento de potencia propios también están pendientes. Los diez recorridos no acreditan potencia0,80 ni mercados independientes.

El ajuste final con desarrollo2023–2024 y el test previsto enero2025–agosto2026 no se han ejecutado. Septiembre–octubre2026 sigue sin asignar altest. Instrucciones completas en BUILD.md del repositorio.
