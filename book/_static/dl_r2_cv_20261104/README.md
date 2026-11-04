# Validación forward chaining de Filipe

Se ejecutaron MLP, LSTM y CNN1D con doce bloques mensuales de2024, entrenamiento creciente desde2023 y diez semillas por modelo/representación. Las variantes relativa y absoluta comparten información, cortes y configuraciones.

La representación principal es relative_window. El archivo method.json resume el procedimiento; folds.json contiene los índices realmente consumidos. runs.csv y predictions.csv conservan60trayectorias y21960predicciones. El paquete run_artifacts.tar.gz conserva5940archivos y720checkpoints con hashes.

El análisis publicado corresponde únicamente a esta CV. La familia conjunta de doce contrastes por semillas utiliza Holm; incluye comparaciones de redes, persistencia, margen y representaciones. El complemento estadístico ejecutó DM, BDS/Ljung–Box y bootstrap temporal, con sus propias familias y alcances. Se calculó sensibilidad retrospectiva de diseño; el piloto previo y dimensionamiento prospectivo definitivo permanecen pendientes. Los diez recorridos no acreditan potencia0,80 ni mercados independientes.

El ajuste final con desarrollo2023–2024 y el test previsto enero2025–agosto2026 no se han ejecutado. Septiembre–octubre2026 sigue sin asignar altest. Instrucciones completas en BUILD.md del repositorio.
