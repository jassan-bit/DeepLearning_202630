# Planificación y dimensionamiento de desarrollo, versión 4

El código está implementado. No se ha ejecutado el piloto de mercado ni una
simulación de potencia con predicciones de mercado. El resultado actual es
`NOT_DIMENSIONED`: R, T y B finales son nulos. Los fixtures de software no son
insumos de planificación y el cargador los rechaza para cerrar un protocolo.

La validación temporal y la potencia son preguntas distintas. Una simulación
con gran potencia bajo una alternativa no compensa un error familiar o una
cobertura inaceptables bajo el nulo. El primer informe temporal adverso se
conserva; `dimension` exige un informe aprobado del alcance completo, ligado
al hash del código estadístico realmente ejecutado. La admisión vuelve a leer el
plan congelado, el runner y el ledger completo (comprimido o plano), y recalcula
las cotas de error y cobertura con el mismo validador del controlador final.
Un JSON con booleanos favorables sin esas evidencias no basta. Un informe acotado o
fallido no permite establecer R/T/B finales.

## Estimando, alternativa y alcance

El primario prospectivo es la esperanza temporal de la media del error
cuadrático de los modelos de las semillas fijadas. El promedio se realiza
sobre semillas y fechas, sin declarar independientes las R×T celdas. Se
conservan como métricas secundarias `sqrt(mean squared error)` y la media de
los RMSE individuales. Ninguna de ellas es el error de predecir con un
ensamble, que requeriría promediar predicciones antes de elevar al cuadrado.

Se conserva el margen práctico del 2% sobre la raíz del MSE agregado. La
hipótesis práctica equivale a una diferencia de pérdidas
`(1-.02)^2 * MSE_rival - MSE_candidata > 0`. Las alternativas de diseño
fijadas son mejoras de esa raíz del 3% y del 5%; la primera excede el margen
en un punto porcentual y la segunda es sensibilidad. No se promete potencia
0,90 para cualquier efecto arbitrariamente próximo a 2%, lo que ningún
tamaño finito puede garantizar.

Para cada alternativa se conserva toda la superficie de pérdida de la
candidata y se multiplica la superficie de cada rival por una constante
calculada **una vez** en desarrollo: `MSE_candidata / ((1-g)^2*MSE_rival)`.
Esto impone la alternativa en la distribución de origen, conservando la
dependencia empírica entre pérdidas hasta ese escalamiento. No se vuelve a
escalar cada muestra simulada para obligarla a ganar. Las pérdidas son no
negativas; se representan mediante `y+sqrt(loss)` para respetar el soporte
conocido. Es una distribución alternativa de pérdidas, no un nuevo modelo
entrenado ni evidencia de que esa alternativa describa un mercado futuro.

La potencia obtenida se denomina **condicional a la distribución empírica de
desarrollo**. Con cinco curvas de entrenamiento no se conocen las colas de
semillas futuras. La estacionariedad débil, mezcla, representatividad del
desarrollo y adecuación de longitud de bloque son supuestos; los diagnósticos
de covarianza, autocorrelación y mitades temporales no los prueban.

## Insumos y separación de semillas

El piloto ejecutable del coordinador debe producir los 21 sistemas en los
cuatro paneles canónicos F00–F03. Las propuestas actuales de instrumentos se
conservan como metadata; la admisión de fuentes pertenece al módulo de datos.
La candidata es `input_only_mse`; `global_mse` sigue siendo el rival principal.
Se exigen al menos cinco curvas de entrenamiento genuinas por sistema neural
y panel, incluyendo sus costes y todos los intentos fallidos. Los cinco
métodos deterministas se guardan **una vez** por panel con
`seed="deterministic"`; expandirlos en el tensor del analizador no crea
ajustes o unidades independientes.

Se utiliza el banco congelado original, SHA-256
`137fe45eca358081307c041ab31dd94efb23f533202438e2eb6888e4e79836ee`.
La asignación se fija antes de resultados nuevos: planificación usa los
índices 4091–4095; la campaña final usa el prefijo de longitud R, con R≤4091.
Así las semillas nuevas finales permanecen separadas de las usadas para
planificación. No se generan IDs ni se sustituyen por rendimiento.

El CSV largo requiere exactamente estas columnas mínimas:

```text
panel,system,seed,date,y,prediction,known_floor_A,rfit
```

`date` es ISO `YYYY-MM-DD`; `rfit` es una constante por panel calculada sólo
en ajuste. El instrumento GW es `[1,sqrt(known_floor_A)/rfit]`, con información
disponible al origen y variación real. No se toma una etiqueta futura rezagada
como instrumento supuesto causal. Todos los sistemas y semillas de cada panel
deben compartir endpoints, etiquetas, piso y escala.

Cada CSV tiene un recibo JSON; por defecto `predictions.csv` se acompaña de
`predictions.receipt.json`. También se pueden indicar listas explícitas. El
recibo requiere:

```json
{
  "kind": "development_planning_predictions",
  "partition": "planning",
  "final_test_opened": false,
  "synthetic": false,
  "is_fixture": false,
  "planning_partition_purged": true,
  "instruments_available_at_origin": true,
  "predictions_sha256": "SHA256_DEL_CSV",
  "seed_bank_sha256": "137fe45eca358081307c041ab31dd94efb23f533202438e2eb6888e4e79836ee",
  "protocol_sha256": "SHA256_DEL_PROTOCOLO",
  "methods_sha256": "SHA256_DE_IDENTIDADES",
  "protocol_file": {"path": "protocol.json", "sha256": "SHA256_DEL_PROTOCOLO"},
  "methods_file": {"path": "methods.json", "sha256": "SHA256_DE_IDENTIDADES"},
  "source_manifest_file": {"path": "sources.json", "sha256": "SHA256_DE_FUENTES"},
  "planning_data_file": {"path": "planning_data.json", "sha256": "SHA256_DE_DATOS"},
  "fit_receipts": [
    {
      "panel": "F00",
      "system": "input_only_mse",
      "seed": "ID_REAL_DE_LA_ASIGNACION_DE_PLANIFICACION",
      "phase": "planning",
      "success": true,
      "active_seconds": 0.0,
      "checkpoint": {"path": "job/state.bin", "sha256": "SHA256_DEL_ESTADO_REAL"}
    }
  ]
}
```

El ejemplo describe el formato, no un recibo emitido. `active_seconds` debe
ser la observación real del intento; no se reemplaza por el techo de 120 s.
También los métodos analíticos tienen un estado o recibo verificable, aunque
no tengan optimizador. Se verifican los archivos de protocolo, métodos,
fuentes, datos y checkpoints con sus hashes. Estas verificaciones prueban
integridad y consistencia; no sustituyen la admisión de derechos ni una
declaración auténtica del custodio. No se consultan archivos finales.

## Monte Carlo y decisión

Cada panel necesita al menos 80 fechas cronológicas propias. No se exige
intersección entre los cuatro: BTC de 2017–2023 y las acciones WIKI de
1996–2013 son admisibles como calendarios distintos. Se forman grupos con
fechas realmente coincidentes; por ejemplo, las tres acciones. Sólo dentro
de cada grupo se comparten los bloques de fechas reales. Entre grupos se
toman calendarios por separado, como supuesto explícito de acoplamiento de
la distribución empírica, sin inventar shocks compartidos por usar un mismo
índice relativo. El vector de índices de semillas sí permanece emparejado
entre todos los sistemas y paneles.

Para almacenar el tensor rectangular se usa el prefijo cronológico más
corto entre grupos, fijado por fechas antes de observar pérdidas; se guardan
las fechas reales por panel y los recuentos omitidos. La covarianza temporal
entre grupos se exporta como `null` (no identificada), nunca como cero
estimado o como una covarianza calculada entre años diferentes. Las
covarianzas dentro de grupo y entre las medias por semilla se conservan.
Semillas y calendario se tratan como factores cruzados; R×T no son
observaciones independientes.

En **cada** réplica se llama al analizador temporal real: DM-HLN/HAC,
bootstrap estacionario y GW condicional. El éxito de un contraste exige los
tres procedimientos, EPA y margen, Holm de 120 pruebas por panel, diferencia
positiva, efecto temporal ≥0,2 y límite inferior del intervalo de diferencia
positivo. Esta versión añade conservadoramente el límite inferior del
intervalo de mejora sobre la raíz del MSE por encima del margen. Con el
analizador V3 se exige también el intervalo conservador del 99% para
diferencia y margen; los intervalos del 95% se siguen reportando como tales.
No se
presentan tres probabilidades marginales de rechazo como potencia de esa
conjunción. Cada éxito por contraste exige también media de RMSE individuales
favorable. Se reportan separadamente la condición histórica en todos los
contrastes principales y la condición literal de medias favorables en al
menos dos de tres paneles adicionales por rival; no se deduce ese orden de
la raíz del MSE agregado, que puede ordenar sistemas de manera distinta.

Se publica potencia por cada uno de los 80 contrastes y por alternativa. La
regla conjunta estadística requiere todos los rivales del panel principal,
consistencia positiva en al menos dos de los tres adicionales por rival y
ausencia de inferioridad significativa. Para esta última se reajustan
conjuntamente los p EPA crudos de las tres pruebas, 20 rivales y tres paneles:
180 slots. La inferioridad **media** exige dirección negativa y rechazo
DM/bootstrap; GW rechaza momentos condicionales sin signo y se reporta
separadamente. GW solo, aun con media muestral ligeramente negativa, no
certifica inferioridad media. También se publica la probabilidad de que los 80 contrastes pasen
simultáneamente. No se exige ni se afirma que la potencia conjunta sea 0,90.
Adecuación de paneles y robustez R3 permanecen como puertas externas; la
simulación estadística no las acredita.

Los puntos de potencia conjunta se etiquetan como condicionales al
acoplamiento entre calendarios. Sin asumir conocida esa dependencia también
se reportan cotas de Fréchet/unión para la intersección de los 80 éxitos:
`max(0,1-sum(1-p_i)) ≤ P(todos) ≤ min(p_i)`, junto con cotas que usan los
intervalos simultáneos de Monte Carlo. Esa intersección es suficiente para la
regla principal más dos-de-tres adicionales, pero no necesaria. Por tanto,
su cota inferior sirve como cota inferior conservadora de esa regla; su
cota superior no se transfiere. Para la regla completa se publica la cota
inferior conservadora y la cota superior general 1. Ni falta de solapamiento
ni una cota conjunta inferior baja bloquean por sí mismas el requisito de
potencia marginal 0,90; tampoco permiten afirmar potencia conjunta 0,90.

Desde V3 se conserva la regla de cola absoluta `(count+1)/(N+1)`,
con resolución mínima `1/(N+1)`; GW conserva su estadístico sin signo.
Con alpha operacional V4 de 0,025, Holm120 requiere aritméticamente N≥4799
y el screen adverso Holm180 requiere N≥7199: B=8191 cumple sólo esa
condición necesaria. A alpha0,05, los mínimos anteriores eran2399/3599.
La variante histórica de dos colas duplicadas tenía otra resolución y su
semántica se conserva únicamente al revisar aquellos resultados. Ninguna
resolución prueba potencia o calibración. Las 100 excedencias esperadas
son un diagnóstico opcional que a alpha0,025 daría N≥720000, **no** un
requisito universal. En el grid B=8191 y referencia DM/GW de 49999 sorteos se registra la
precisión de los contadores reales de los tres métodos. Los límites
binomiales superiores y una sensibilidad re-Holm más estricta se reportan
**sólo como diagnóstico**, con error familiar diagnóstico 0,01 por panel.
No reemplazan los p operacionales `(count+1)/(N+1)` que usa assessment.
Sustituirlos definiría otra regla: a B=8191 incluso cero excedencias podría
impedir cualquier rechazo tras Holm120. Esa incompatibilidad se detectó y
corrigió prospectivamente antes de ejecutar potencia de mercado; no se
cambió la batería V4 congelada o su criterio de validación.

Cada réplica externa incluye la aleatoriedad real del bootstrap interno y
cuenta el éxito de la misma conjunción finita de la evaluación final. Se
conservan el ledger de contadores/denominadores/estados por réplica y los
intervalos diagnósticos de la primera réplica prefijada; no se llaman p
verdaderos conocidos sin error Monte Carlo. La identidad de la regla es
`finite-B-count-plus-one-actual-Holm-v1`. La incertidumbre numérica de las
referencias fijas de DM/GW sigue declarada y la calibración sintética se
condiciona a esas referencias congeladas.

El mínimo de potencia se obtiene de los límites inferiores simultáneos
Clopper–Pearson de la Monte Carlo externa. **No se resta 0,01** por un
screen diagnóstico que no forma parte de la decisión real. Se conservan
por separado las tasas del screen estricto para revisar sensibilidad, sin
convertirlas en otra prueba de continuación o abandono.

Los intervalos de error Monte Carlo por contraste son Clopper–Pearson con
asignación simultánea sobre los 80 contrastes y las dos alternativas. La
selección usa 128 réplicas por alternativa. Se elige el primer punto factible
que alcance límite inferior ≥0,90, con orden fijado por coste nominal,
número de predicciones y B. El código y los recibos de planificación deben coincidir con el hash de
`engine/statistics.py`; un recibo de otra versión se rechaza antes de la
Monte Carlo. El resultado registra los hashes de ese analizador y del
planificador, sin sustituir un hash ausente por el código actual.

El ganador y su hash se guardan **antes** de
consultar las 256 réplicas independientes de validación. Si ese ganador
falla, se conserva el fallo y no se selecciona otro con esos resultados.
Una nueva hipótesis, grid o asignación requiere versión y salida nuevas.
Una repetición con identidad idéntica recupera el resultado guardado.

La configuración actual contiene R={10}, T={768}, B={8191}, HAC=48 y
bloques de longitud 32, con referencia fija de 49999 sorteos y semilla
2026112103: coincide con el alcance candidato de la auditoría V4, con alpha operacional 0,025.
Sigue siendo un **grid de desarrollo**, sin tamaños finales aprobados.
Cualquier otro R/T/B o ajuste temporal requiere una admisión propia; no se
extrapola la V4 a T=252 u otros tamaños. Se limita el trabajo a 8192 llamadas
en selección y 4096 llamadas totales, con 8388608 índices bootstrap por
llamada y 3600 s de Monte Carlo entre selección y validación. Para este grid
se prevén como máximo 1024 llamadas de selección y 2048 de validación. El
presupuesto es un techo revisable fijado antes del piloto real, informado por
el benchmark sintético; no es consumo medido ni promesa de duración.
El límite de tiempo se comprueba entre réplicas; una llamada ya iniciada
puede terminar después. Se conserva el resultado parcial y no se transforma
en potencia aprobada. Se publican llamadas e índices previstos y tiempo de
software realmente observado. Si el coste o la precisión son inviables,
el resultado sigue siendo `NOT_DIMENSIONED`.

La referencia al informe V4 permanece pendiente de hash y veredicto hasta
su finalización y revisión. No se toma un informe V1/V2 fallido como admisión
de esta configuración. El informe temporal necesario debe declarar exactamente
`status="PASSED_PREDECLARED_VALIDATION"`, `family_rivals=20`,
`validation_scope="single_panel_temporal_family"`, `panels_per_replicate=1`,
`all_three_methods=true`, `complete_and_partial_nulls=true`,
`extreme_tail_validated=true`, `coverage_validated=true`, y
`statistics_source_sha256` del analizador ejecutado. No se emite ese recibo
desde las pruebas de potencia ni se inventa a partir de fixtures. Además,
`validated_settings` debe contener exactamente el R, T, B, `block_length`,
`hac_lags`, `horizon`, `margin`, `reference_reps` y `reference_seed` del punto seleccionado; una auditoría con
otro T o ajustes temporales no admite este grid por extrapolación.
La auditoría calibra la familia temporal de un panel; los cuatro paneles
son el alcance de aplicación, no cuatro réplicas independientes ni una
validación de error conjunto entre paneles. El screen adverso global de 180
casillas y la conjunta de potencia se informan con su alcance separado.

Esta versión fija `alpha=0.025` operacional, más exigente que el techo de
la guía de 0,05, conservando el criterio de error familiar sintético ≤0,05
y sus cotas de confianza originales. El nuevo valor es una corrección
prospectiva de la regla y no relaja las tolerancias del estudio. Ese valor debe
coincidir exactamente, se transmite al analizador y se toma de `family.alpha`
en la conjunción y el screen adverso; no se mantiene un umbral 0,05 oculto.

## Presupuesto y comandos

La referencia R=10 en cuatro paneles implica 640 ajustes neurales, cuatro
ARIMA y 16 trayectorias analíticas. A 120 s **nominales por sistema y semilla**
son 76800 s = 21,333 h de cómputo neural activo nominal; las fases SAE y
cualquier auxiliar comparten ese total. R=20 implica 1280 ajustes neurales y
42,667 h nominales. Los métodos deterministas no se multiplican por R.

La estimación inicial de almacenamiento usa 64 MiB por ajuste neural para
estados seleccionados/últimos y 192 bytes por fila CSV; son supuestos
revisables, no bytes medidos. No se asigna precio monetario sin una tarifa
real. ARIMA, adquisición, inferencia, exportación, Monte Carlo, transferencias
GitHub, interrupciones y recuperación se muestran explícitamente como costes
pendientes o excluidos. Los recibos distinguen consumo medido, intentos
fallidos y consumo aún incierto; el tiempo de un piloto histórico no se
extrapola a toda la familia.

`planning_active_seconds_charged_total` incluye reservas cargadas tras
interrupciones. `measured_planning_active_seconds` suma sólo recibos cuyo
tiempo no está marcado incierto o reservado; se reportan por separado el
número de intentos inciertos y los segundos parciales observados al
interrumpirse. No se presenta una reserva de 120 s como consumo exacto.

Desde la raíz del paquete:

```bash
python -m unittest discover -s tests -p 'test_power.py' -v
python -m engine.power --design configs/power_design.json --output qa/power/current_dimension.json
```

El segundo comando no entrena y, sin insumos, devuelve código de salida 2 y
un resultado `NOT_DIMENSIONED` con campos finales nulos. Tras ejecutar el
piloto autorizado de desarrollo, la etapa de dimensionamiento es:

```bash
python -m engine.power --design configs/power_design.json --predictions planning/predictions.csv --receipts planning/predictions.receipt.json --output planning/power_study_v1.json
```

No se ejecutó ese último comando con datos de mercado durante esta tarea.
El cierre requiere además validación temporal aceptable, presupuestos
factibles, admisiones y las restantes puertas del coordinador.
