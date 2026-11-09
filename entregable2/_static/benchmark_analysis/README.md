# Análisis del benchmark E2: propuesta y plantillas

Estado propuesto, no congelado. Los quince renglones son modelos pendientes;
las celdas numéricas vacías no significan error cero. Las corridas ETTh1 del
capítulo5 no se convierten en resultados del benchmark Bitcoin D7.

Desde la raíz: `python scripts/analyze_e2_benchmark.py`. Este modo sólo usa la
biblioteca estándar y no abre predicciones ni datos de mercado. Las funciones
numéricas usan NumPy/SciPy y se prueban con fixtures sintéticos identificados:
`python tests/test_e2_benchmark_analysis.py`.

El CLI con --panel/--runs/--costs/--calendar sólo valida insumos declarados
`--scope synthetic_fixture` y exige una carpeta privada --output-dir fuera del
libro. No publica rankings de los fixtures ni acepta un análisis real mientras
el protocolo esté sin congelar. Los hashes identifican archivos y no prueban
por sí solos entrenamiento justo, ausencia de fuga o validez estadística.

Las primitivas permiten evaluar métricas, efectos y diagnósticos futuros; no
son evidencia de su ejecución sobre Bitcoin. La primaria promedia las RMSE
por semilla; el bootstrap usa el calendario completo emparejado y conserva las
trayectorias, sin convertir semillas o ventanas solapadas en mercados nuevos.
DM contrasta la pérdida declarada, no la misma hipótesis no lineal de RMSE y δ.
Las familias, supuestos, potencia, coste y código de producción deben cerrarse
antes del scoring. La reserva heredada sigue conocida y exploratoria.
