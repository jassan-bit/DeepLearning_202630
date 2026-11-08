# Comparación principal E1: recetas originales, D7 y test histórico de 608 días

Este paquete activa editorialmente LSTM32, MLP16 y GRU32 como comparación principal
de E1 a partir de resultados auténticos ya ejecutados. No entrena redes ni genera
pronósticos. La promoción se decide después de conocer los resultados: no es un
registro prospectivo ni una nueva confirmación temporal.

Las tres redes usan las mismas cinco covariables OHLCV, `relative_window`, D7,
objetivo sigma7/h1, semillas 4101–4110 y reglas de optimización/selección de épocas.
CV son doce bloques mensuales expansivos y 366 fechas de enero–diciembre de 2024.
El test conocido contiene 608 fechas, 01/01/2025–31/08/2026. Sus veinte meses son
intervalos de reporte con pesos y escalas congelados, sin reajustes en test.
El ajuste final comparte 709 etiquetas de 08/01/2023–24/12/2024, con purga D7.

GRU32 (lr .003, dropout .1, 3.777 parámetros) es el baseline neuronal por su menor
RMSE media CV. LSTM32 usa lr .003 y 5.025 parámetros; MLP16 usa lr .001 y 593.
La persistencia determinista se informa como referencia aparte: no tiene semillas,
DE ni IC de inicialización y necesita ocho cierres previos. Las recetas tienen
distinta capacidad y esfuerzo histórico de HPO: MLP/LSTM heredaron selecciones de
32 trayectorias V1 por familia; GRU heredó la receta recurrente sin búsqueda propia.
No se afirma que las arquitecturas estén optimizadas con igual presupuesto.

`run_metrics.csv` guarda 60 puntuaciones neuronales D7 (dos fases × tres redes ×
diez semillas) y dos referencias deterministas, sin vectores diarios. Las seis
métricas de `metric_summary.csv` tienen media, DE muestral e IC t95 marginal df9
para redes; estos intervalos dependen del calendario conocido, no incorporan
incertidumbre temporal ni cobertura simultánea. MAPE conserva unidades porcentuales.

`execution_provenance.json` documenta que la MLP histórica ya produjo las 608
predicciones por cada una de diez semillas D7 el 04/11/2026, después de congelar
los modelos y antes de este cambio editorial. No se atribuyen a esta promoción
nuevos ajustes ni pronósticos. Los 480 controles métricos LSTM coinciden entre
los estudios originales. Se verifican hashes de las fuentes y sus fuentes previas.

Faltan aquí los vectores diarios, pesos y escaladores originales de MLP; las
puntuaciones agregadas no permiten reconstruir DM, residuos o bootstrap temporal.
El calendario de septiembre–octubre de 2026 pertenece al estudio complementario
de 60 días y 465 semillas; sus recetas/resultados permanecen separados e intactos.
También permanece intacto el suplemento histórico de cuatro D.

Reproducción: `python scripts/build_e1_primary_608.py`; verificación sin escribir:
`python scripts/build_e1_primary_608.py --check`. Ningún comando entrena o infiere.
