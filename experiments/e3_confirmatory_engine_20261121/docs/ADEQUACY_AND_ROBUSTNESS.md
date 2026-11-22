# Adecuación, robustez y dictamen R3 de predicciones guardadas

`engine/assessment.py` implementa el dictamen completo de la candidata
`input_only_mse` frente a los veinte rivales de `engine.models.METHODS`, en
F00–F03. Consume predicciones que el controlador ya evaluó. No importa un
colector, no carga una partición de mercado, no reajusta modelos y no ejecuta
forwards. Las pruebas usan exclusivamente fixtures sintéticos acotados.

La existencia del código no acredita R3. El informe separa cada condición y
calcula `r3_scientific_criteria_satisfied` como su conjunción. Puede resultar
verdadera con todas las evidencias válidas; ninguna aprobación institucional
se atribuye al software. El dictamen temporal histórico V2
`FAILED_FINITE_SAMPLE_VALIDATION` permanece en `qa/statistical_validation/`.
La versión vigente sólo puede admitir la condición metodológica con su propio
resultado y ámbito demostrados. El código usa la identidad
`temporal-fixed-seeds-mse-v4` y su alpha operacional prospectiva de .025;
la existencia de esa implementación no presume un resultado favorable de su
validación sintética.

Los certificados se comprueban con las mismas guardas del controlador antes
de entrenar: reporte y ficheros de plan, congelación, ejecutor y réplicas
vinculados por hash, descompresión acotada y recuento de cotas CP a partir de
las réplicas. Las afirmaciones JSON `admitted:true` o `meets_power_target:true`
no sustituyen esos artefactos. El cierre de potencia se vuelve a verificar con
los contadores reales, archivos de planificación, identidad del diseño y
protocolo completo. No se aceptan cotas numéricas afirmadas sin sus insumos.

## Contratos y orden de ejecución

Las columnas nativas son `panel,system,seed,date,y,prediction,known_floor_A,rfit`.
También se aceptan los alias `method,time,y_true,y_pred`. Las fechas son ISO
`YYYY-MM-DD`; las unidades de y y predicción son sigma7 originales; A es la
contribución de **varianza** conocida, por lo que el piso es `sqrt(A)`.
`rfit` es la RMS de referencia, positiva y estimada sólo en ajuste. Se verifican
finitud, soporte, claves únicas, etiquetas/pisos/escalas idénticos por endpoint,
calendarios comunes y toda la familia. Las dieciséis redes usan el mismo
prefijo del banco congelado; cada uno de los cinco deterministas se guarda
una vez con `seed="deterministic"`. Su expansión numérica en el tensor es una
ponderación comparable, nunca nuevos ajustes independientes.

`fit_calibrators(calibration_csv, output_path=...)` se ejecuta después de
congelar todos los modelos y antes de abrir prueba. El controlador guarda el
artefacto en `metrics/calibrators.json`, lo incluye en el sello de estados
seleccionados y verifica el respaldo remoto antes del primer acceso al test.
El SHA semántico del calibrador es `sha256` del payload canónico sin ese campo;
el SHA del archivo completo es otro identificador y ambos se conservan.

`assess_case(case_root, protocol, predictions=None, calibrators=None,
evaluation_receipt=None, adequacy_config=None, stress_predictions=None,
output_path=None)` acepta protocolo como ruta JSON o dict. `predictions` puede
ser split→CSV/lista de filas, split→`{path,sha256}`, o panel→split→filas. Sin ese
argumento busca exclusivamente CSV derivados existentes en
`case_root/predictions/{fit,selection,planning,calibration,test}_predictions.csv`.
No usa datasets originales como alternativa. Con `output_path`, la escritura
es inmutable e idempotente: otros resultados bajo la misma ruta se rechazan.

Los escenarios guardados se descubren mediante
`evaluation_receipt.stress_prediction_files`, con nombres
`predictions/test_input_noise_predictions.csv` y
`predictions/test_missingness_predictions.csv`. Se verifica cada SHA después
de reubicar la ruta al caso restaurado; nunca se vuelven a ejecutar forwards
desde `assess`. La identidad de evaluación incluye `stress_plans_sha256`.

El recibo de evaluación completada debe vincular el protocolo, indicar
`selected_checkpoints_verified_before_test`,
`calibrators_frozen_before_test`, `sealed`/`evaluation_sealed` y el SHA semántico
del calibrador. El análisis por sí solo no reconstruye una cronología auténtica:
esa evidencia procede del controlador y sus respaldos. Las particiones de
predicciones no pueden solaparse ni aparecer fuera de orden. La purga por
soporte de entradas/etiquetas se verifica además en la admisión de datos;
fechas distintas no prueban por sí solas esa purga.

El verificador comprueba además los bytes guardados del manifiesto de modelos,
sello pretest, identidades de predicciones fit/selection/calibration, CSV final,
admisiones finales y hash de identidad de la evaluación única. No recarga pesos
ni etiquetas originales. Conserva `unobserved_remote_compute_possible` del
recibo y de todo el historial de restauración. Si hubo una sesión abierta con
progreso remoto no subido, tiempo cargado y pasos observados son cotas
inferiores; el informe no afirma costes exactos o comparabilidad computacional
completa a partir de un marcador `done`.

## Familia estadística, efectos y consistencia

Cada panel llama la batería temporal real de `engine.statistics.analyze`, con
semillas ajustadas fijas y fechas cronológicas. La primaria prospectiva es la
media por fecha de los errores cuadrados entre esas semillas; la RMSE agregada
es su transformación monótona. También se conservan la media y dispersión de
RMSE individuales, cuya ordenación debe favorecer a la candidata. No se
promedian predicciones para construir un ensemble.

En F00 se exige cada uno de los veinte contrastes: tres procedimientos
distintos, EPA y frontera práctica, seis rechazos con Holm120, dirección
favorable, ganancia mayor al margen RMSE relativo fijado, efecto temporal
`mean(d)/sd(d) >= .2`, límite inferior del IC95 de diferencia MSE positivo y
límite inferior de ganancia RMSE mayor al margen. Se preservan estadísticos,
p crudos/ajustados, intervalos, efectos, estados y resultados adversos. El GW
de momentos condicionales no se presenta aisladamente como superioridad
direccional; la conjunción también exige la dirección media, EPA y margen.

La regla adicional es **por rival**: al menos dos de F01–F03 deben favorecer la
media histórica de RMSE, y se conserva la opción prospectiva más fuerte de
que al menos dos pasen también el contraste completo, efecto e intervalos.
No basta que dos paneles tengan alguna victoria frente a rivales distintos.
Además se vuelven a ajustar **180 EPA**, 3 paneles × 20 rivales × 3 métodos,
en una sola familia Holm para detectar cualquier inferioridad significativa.
No se reutilizan p ajustados por panel para ese fin. El informe conserva los
180 p crudos, ajustes globales y direcciones, incluidos los no significativos.
El veto direccional usa DM o bootstrap significativo y diferencia media
negativa. Un rechazo GW de momentos condicionales con media muestral negativa
se conserva como señal condicional adversa; por sí solo no demuestra
inferioridad de la media. Sus p siguen dentro de la familia180, sin reducir la
multiplicidad por esta distinción de interpretación.
`engine.power.statistical_rule` utiliza esta misma conjunción para planificar.

Las comparaciones entre paneles se reportan separadamente: Friedman,
Iman–Davenport cuando la transformación es finita, y contraste exacto por
enumeración de signos de rangos pareados con Holm20. Con cuatro paneles,
el p bilateral mínimo sin empates es 0.125; esa falta de resolución no se
oculta. Los paneles pueden compartir shocks económicos, de modo que la
referencia de datasets independientes no se certifica. Sus p no cuentan como
tercer procedimiento temporal ni sustituyen el resultado por rival.

La potencia exige el recibo real del planificador: ganador congelado antes de
validación, tamaño R/T/B idéntico, validación independiente, los ochenta
contrastes con límite inferior simultáneo Monte Carlo ≥.90, potencia conjunta
reportada y calibración temporal admitida. Un valor nominal de R, B o una
potencia conjunta sin los contrastes no satisface esa condición.

## Política cuantitativa de adecuación

La política de `configs/adequacy.json` queda **definida prospectivamente** como
decisión científica autorizada por esta tarea, sin inventar una firma o
ratificación institucional. Su estado
`DEFINED_PROSPECTIVELY_NOT_EMPIRICALLY_SATISFIED` describe que todavía no existen
resultados confirmatorios que la satisfagan. Antes de abrir prueba, el protocolo
final debe vincular el SHA exacto del archivo. Cambiar una tolerancia después
de observar nuevos resultados crea otra versión y no modifica el dictamen.

La trazabilidad primaria usa la guía E3 §§8.4.3–8.4.6/8.7.3, pp.24/28, y su
extracción verificada con hash en `PRIMARY_GUIDE_FACTS.json`. El libro §2.2
conserva el margen heredado E1/E2
`δg = 0.00025772842664981106` en unidades sigma7. Es una fuente histórica, no
un umbral universal de la guía ni el nuevo margen relativo de superioridad.
Los hashes de ambas fuentes se incluyen en la configuración.

La adopción nueva para brecha es conservadora:

`RMSE_selection - RMSE_fit <= min(δg_E1/E2, 0.10 * RMSE_fit)`.

Así se endurece el margen absoluto en paneles de escala pequeña. Un error de
ajuste cero es degenerado y no acredita la brecha. Las predicciones de ajuste
y selección deben pertenecer al **mismo estado seleccionado**, en unidades
originales; la pérdida de entrenamiento normalizada, la reconstrucción de un
autoencoder o curvas de otras épocas no las sustituyen. Su ausencia produce
brecha no evaluable y condición falsa. La consistencia test–selección requiere
`abs(RMSE_test-RMSE_selection)/RMSE_selection <= .20`. También se exige superar
a Persistence tanto en RMSE agregada como en media individual.

Los restantes números son decisiones prospectivas explícitas, no mandatos
numéricos atribuidos a E1/E2: sesgo e intercepto MZ dentro de ±.05 de rfit,
pendiente MZ entre .75 y 1.25, cobertura empírica 90%±5 puntos y ancho medio
del intervalo/rfit ≤1. Sus intervalos completos de sesgo/intercepto/pendiente
deben quedar dentro de las bandas. Se exige además que el límite inferior del
IC temporal de cobertura sea al menos .85. No se certifica calibración por
no rechazar una prueba de nulidad.

Cada semilla de la candidata en cada panel recibe una regresión
Mincer–Zarnowitz `y = a + b * prediction + error`, con restricciones conjuntas
`a=0,b=1`, covarianza Bartlett HAC en unidades temporales y normalización rfit
para estabilidad numérica. Se conservan Wald, p y bandas de coeficientes. La
referencia chi-cuadrado es asintótica; no convierte muestras pequeñas o
dependientes en una garantía exacta. Diseños, residuos o covarianzas degenerados
producen estado explícito y p=1, sin certeza infinita.

Ljung–Box, ARCH-LM y Jarque–Bera se ejecutan y sus rechazos se reportan con
Holm sobre todos los diagnósticos disponibles de candidata/panel/semilla. BDS
está implementado como opción preregistrable: dimensión2, epsilon2 desviaciones,
versión de statsmodels registrada y techo fijo de tamaño. Por defecto no se
presupone iid en residuos de etiquetas solapadas. Los rechazos de normalidad o
dependencia son diagnóstico y limitación, no una supuesta prueba inversa de
ley predictiva correcta. La aceptación positiva utiliza las bandas de
tolerancia, cobertura y ancho definidos arriba, además de brechas y baseline.

## Calibradores e intervalos guardados

Por panel/sistema/semilla se ordenan los residuos absolutos de calibración y se
toma el elemento de rango `ceil((n+1)*.90)`, con índice empezando en1. Si el
rango excede n se registra muestra insuficiente y q no disponible; no se
inventa un cuantil finito. En prueba se puntúa
`[max(sqrt(A_t), target_floor, prediction_t-q), prediction_t+q]`.
El ancho usa esos límites **después** del clipping; no se informa 2q cuando el
piso acorta el intervalo. Se registran cobertura, ancho medio, clipping y un
IC temporal marginal mediante bootstrap estacionario conjunto de indicadores.

La dependencia temporal impide invocar la garantía conformal intercambiable
de cobertura 90%. El bootstrap también tiene supuestos de dependencia débil y
estacionariedad, no demostrados por la llamada. La cabeza entrenada con MSE
produce una estimación puntual: sus átomos internos no se promueven a una
densidad calibrada, y no se fabrican PIT ni log-scores.

## Robustez ejecutable y límites por componente

Sobre las mismas predicciones guardadas se ejecutan alternativas de HAC y
longitud de bloque fijadas en configuración, las primeras mitades y posiciones
pares del prefijo de semillas, y las mitades cronológicas primera/segunda.
Cada escenario conserva su familia completa y la regla entre paneles. Si el
tamaño de un bloque no soporta los lags fijados se informa no evaluable y se
falla la condición, sin retocar lags después del resultado. El criterio
predeterminado exige mantener la conclusión estadística completa.

La métrica secundaria MAE se contrasta con los procedimientos temporales
mediante una transformación exacta de pérdidas:
`sqrt(abs(prediction-y))` frente a objetivo cero. Su error cuadrado es
exactamente el error absoluto guardado; esto no ejecuta un modelo y no cambia
las predicciones originales. Se reportan MAE, diferencias, IC y p; la raíz
numérica de MAE no se llama RMSE. Se usa margen cero en esta sensibilidad, sin
trasplantar el margen RMSE a unidades de MAE. Se exige dirección e IC positivos
y los tres EPA ajustados; sus pruebas prácticas de margen cero se conservan
pero no se cuentan como procedimientos independientes.

Nadeau–Bengio no aplica a este diseño sin particiones aleatorias repetidas;
las correcciones pertinentes son HAC y remuestreo temporal. No se finge haber
ejecutado una corrección de particiones sobre semillas × fechas.

Ruido y faltantes necesitan nuevos forwards **posteriores** al sello, bajo
escenarios de entrada congelados. Assessment sólo admite sus CSV ya guardados
y un plan por escenario vinculado con hash en el protocolo. Se verifican
familia, semillas, endpoints, etiquetas y la misma regla. En la configuración
actual ambos escenarios se exigen: sin esas predicciones el informe muestra
`unexecuted_missing_saved_predictions_or_frozen_scenario` y robustez falsa.
El módulo no afirma haber producido esos forwards ni hereda perturbaciones de
versiones históricas. El controlador implementa los escenarios y los conserva
con el mismo sello y cuidado de consulta única que la evaluación principal.
Se autodetectan los CSV `test_input_noise_predictions.csv` y
`test_missingness_predictions.csv` sólo desde el recibo completo guardado. Se
exigen hash de CSV, plan, manifiesto seleccionado, protocolo e identidad de
evaluación, además de `no_refit:true` y `completed:true`. Los escenarios
preservan los cierres, etiquetas, piso conocido y escala de ajuste; todos los
sistemas reciben la misma perturbación del panel. Son errores de medición de
O/H/L/volumen y faltantes causales de Open/volumen, no una simulación de
desconocer el cierre o el objetivo real.

## Pruebas reproducibles

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_assessment.py -v
```

La suite verifica cuantiles finitos, clipping por sqrt(A), ancho y cobertura,
calibración anterior a prueba, identidad/idempotencia, familia completa y
semillas deterministas únicas, las 180 EPA de inferioridad, regla adicional por
rival, brechas en unidades correctas, degeneraciones y el dictamen temporal
adverso conservado. Un fixture completo ejecuta los procedimientos temporales,
diagnósticos y sensibilidades sobre predicciones sintéticas guardadas. La
prueba de conjunción verdadera usa stubs declarados para aislar el cableado;
no se presenta como validación científica, de potencia o de mercados.
