# Adecuación retrospectiva propia de input-only: análisis nuevo ejecutado

Se analizaron exclusivamente **predicciones puntuales guardadas** del piloto V5.
No se entrenó, no se hicieron forwards y no se descargó mercado. La candidata
actual `input-only-overlap-mse-1` reproduce la receta `input_only_mse` de ese
piloto; este análisis no cambia su identidad o regla, no revierte el abandono
de V5 completo y no convierte sus calendarios expuestos en confirmación.
Asistencia de IA declarada en código, plan, análisis y revisión.

La fuente es el ZIP público original, SHA-256
`beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c`,
del código `c4ca86f1431afeaa958ff0b544b453ab17a4d7b8` y resultados
`05c1a637f13356dd1df6b8c5aa8f4676dda61c12`, ruta
`runs/e3_seoi_v5/run-01`. El ZIP y sus 110 miembros se verificaron, incluidos los
siete derivados listados en el recibo del piloto. Se cotejaron sus identidades,
plan, admisión, calendario, unidades y métricas contra la auditoría anterior.
`results/input_manifest.json` conserva hashes y rutas de todas las fuentes
leídas; se volvieron a comprobar al terminar. Los pesos no están en ese ZIP.

Las **1.760 filas** son seis recetas neuronales por cinco semillas y 55 fechas,
más dos controles analíticos con una trayectoria cada uno: **32 trayectorias**
sobre **el mismo calendario diario, 7 de septiembre–31 de octubre de 2024**.
No son 1.760 observaciones temporales independientes, 32 mercados ni cinco
calendarios independientes. El objetivo y los errores conservan unidades de
sigma7; el signo del error es predicción menos objetivo.

Se reutilizaron las métricas, auditoría de selección, sensibilidad por semillas,
subgrupos posteriores y leave-out de `e3_input_only_candidate_20261121/analysis`.
El análisis V4 de 88 diagnósticos conserva su propio ámbito y no se repitió.
La ejecución nueva aporta residuos individuales, diagnósticos formales
exploratorios y la sensibilidad temporal de intervalos del funcional primario.

## Plan y ejecución auténticos

`analysis_plan.json` y el código se fijaron localmente antes del cálculo en
`IMPLEMENTATION_FREEZE.json`, a las **02:54:45 UTC del 22 de noviembre**
—21:54:45 de Bogotá del día21—. El resultado del piloto y la nominación de la
receta ya eran conocidos. **No es un preregistro, una selección previa a los
resultados o una recuperación de independencia.** No se hicieron nuevas
simulaciones Monte Carlo para validar procedimientos.

La ejecución real comenzó a las `2026-11-22T02:54:46.723152+00:00` y terminó a
las `2026-11-22T02:54:46.901027+00:00`. Consumió **0,17787 s de pipeline y
0,17709 CPU s**, excluyendo arranque/importaciones, elaboración del plan y
pruebas de software. Se usó Python3.12.14, NumPy2.3.5, SciPy1.17.0 y
statsmodels0.14.5. El recibo identifica el código y el plan por SHA256.

Se cuentan **128 ajustes OLS auxiliares** —64 de sesgo/HAC, 32 ARCH y32 MZ—,
separados de los **cero entrenamientos de modelos de pronóstico**. Hubo
**7.998 sorteos de índices temporales** —3.999 para cada longitud de bloque—,
compartidos entre todas las recetas/semillas. No hubo inferencias neuronales,
nuevas predicciones, apertura de prueba o calibración de intervalos servidos.

## Diagnósticos de residuos y multiplicidad

| Diagnóstico nuevo | Trayectorias | Rechazos con Holm128 |
|---|---:|---:|
| Ljung–Box, rezagos1–7 | 32 | 0 |
| Ljung–Box, rezagos1–14 | 32 | 0 |
| ARCH-LM, rezago7 | 32 | 0 |
| Mincer–Zarnowitz conjunto: intercepto0/pendiente1, HAC14 | 32 | **27** |

La familia conserva las **128 posiciones**. La implementación no elimina
diagnósticos no soportados: conservaría p=1 y el motivo; los128 se pudieron
calcular en estos datos. Holm no valida los p marginales. Ljung–Box tiene
supuestos clásicos y no es robusto a heterocedasticidad general; ARCH-LM y
MZ/HAC usan referencias asintóticas. **55 fechas no acreditan su calibración
finita**. El solapamiento del objetivo motiva comprobar dependencia, pero no
demuestra que los errores de un pronóstico óptimo tengan que autocorrelacionarse.

Input-only registra **dos de cinco rechazos MZ**: semillas20261127 y20261131,
con p ajustados **0,00196892445** y **0,00658836564**. Son señales condicionadas
de desviación de calibración lineal; no acreditan inutilidad general, causalidad
del deterioro o una corrección concreta. Los otros ceros de rechazo no prueban
homocedasticidad, ausencia de dependencia, potencia o adecuación. Todos los
resultados adversos, incluido el resto de recetas, quedan en CSV.

Los **64 intervalos de sesgo HAC7/HAC14** son puntuales, asintóticos y
condicionados a cada predictor guardado. Los cinco sesgos input-only son
negativos, aproximadamente de −0,00054033 a−0,00007721 sigma7. No se presentan
como intervalos simultáneos ni como inferencia sobre una población de semillas.
No se selecciona la semilla con el mejor diagnóstico.

## Intervalos temporales del funcional primario

Se conserva la **media de RMSE individuales**: se calcula la raíz por semilla
antes de promediar. No se cambia a RMSE agrupada o a un ensamble. El bootstrap
circular de bloques utiliza un solo índice de fechas para todas las recetas y
semillas en cada sorteo; los controles analíticos aparecen una vez. Las cinco
semillas permanecen fijas, sin remuestreo de inicializaciones.

Se fijaron antes del cálculo B=3999, longitudes7 y14, semilla202611220301 y
percentiles lineales2,5/97,5. Se comparó input-only con **los siete controles del
piloto**, sin seleccionar el rival que produzca el intervalo favorable. Ambos
tamaños de bloque son sensibilidades del **mismo procedimiento**. Se conservan
los14 intervalos y las diferencias MAE secundarias; no se calcula un p de
superioridad ni se ejecuta la familia confirmatoria de21 sistemas.

| Input-only menos global MSE | Diferencia RMSE | IC95 puntual exploratorio |
|---|---:|---|
| Bloque7 | −0,00005821987 | [−0,00013727537; **+0,00002937367**] |
| Bloque14 | −0,00005821987 | [−0,00011955731; **+0,00001910637**] |

La ganancia puntual previa, **2,759522 %**, se conserva. Sus intervalos relativos
son aproximadamente **[−1,529 %;6,903 %]** y **[−0,998 %;5,650 %]**. Ambos
incluyen cero. Las sensibilidades no justifican afirmar una ventaja estable o
confirmada sobre el mejor control neuronal. Tampoco significan prueba de
igualdad. Los intervalos MAE muestran sensibilidad al bloque —el de bloque7
incluye cero y el de bloque14 no—; no se elige el que convenga.

Los intervalos son **condicionales a las cinco trayectorias almacenadas**, con
una aproximación de estacionariedad/dependencia débil. T=55 deja sólo unas
3,9–7,9 longitudes de bloque; el cierre circular crea una unión artificial de
los extremos. No se validó cobertura finita, no hay control simultáneo de los
siete contrastes y no se incluye incertidumbre poblacional de inicializaciones.
No se trasladan a las confirmaciones admitidas por el motor ni sustituyen su
guarda de diez semillas o el análisis de potencia.

## Lo que las predicciones no permiten reconstruir

No existen predicciones puntuales de ajuste o selección ni el riesgo de ajuste
evaluado con el checkpoint seleccionado. La pérdida media de batches durante
una época usa estados cambiantes y no es ese riesgo. Las curvas sí conservan
la MSE de selección para parada, ya auditada: input-only tiene RMSE media de
selección **0,00430490508** y assessment **0,00205156082**, con cambio de rango
6→1. Esos calendarios/regímenes difieren y la selección intervino en la elección
de la época; una brecha negativa no prueba ausencia de sobreajuste. No había un
margen prospectivo completo de brecha. **Las Tablas A/B no quedan acreditadas.**

El ZIP sólo guarda puntos: faltan cuantiles, scores de calibración o una ley
predictiva completa de input-only para reconstruir PIT o cobertura/anchura.
MZ es calibración lineal de puntos, un objeto distinto. No se heredan los
calibradores de la aditiva, E2 o V3. Tampoco se completan tres pruebas de
superioridad por contraste, potencia por contraste≥0,90/conjunta, los21 sistemas,
diez semillas finales o cuatro paneles por ejecutar este análisis.

## Reproducción y verificación

Ejecutar con el entorno científico fijado, sin importar los modelos:

```bash
python audit_saved_adequacy.py --output /tmp/e3-v5-adequacy-recomputed
python audit_saved_adequacy.py --verify-only
python -m unittest -v test_audit_saved_adequacy.py
```

El script busca por defecto el ZIP y los CSV de auditoría en el paquete hermano
`e3_input_only_candidate_20261121`; también admite `--input` y`--reuse`.
`--verify-only` verifica los derivados existentes sin repetir el remuestreo.
Un directorio de resultados existente no se sobrescribe. Las rutas absolutas
de recibos registran el lugar real de ejecución y pueden diferir al reproducir;
los identificadores de fuente y el plan permanecen fijados.

**11 pruebas de software pasan**. Comprueban el funcional frente al agrupado y
al ensamble, índices temporales compartidos/contiguos, controles deterministas
sin pseudorreplicación, Holm, posiciones degeneradas conservadas, integridad de
la fuente, protección frente a sobreescritura y detección de derivados alterados.
El primer intento tuvo un fallo de fixture —sus dos estimandos coincidían por
construcción—, corregido sólo en la prueba. El código de análisis y sus resultados
no se cambiaron ni se repitieron; ambos logs se conservan.
