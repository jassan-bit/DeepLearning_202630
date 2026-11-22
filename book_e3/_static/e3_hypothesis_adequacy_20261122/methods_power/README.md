# Auditoría ejecutada de métodos y entradas de potencia

Esta entrega comprueba el código **sin alterar la candidata, los procedimientos,
el banco, el protocolo o las reglas del motor A**. El programa valida los 288
archivos fijados al commit `c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c`, vuelve a
verificar los ledgers archivados de calibración y ejecuta comprobaciones
deterministas de sus alcances. No descarga mercado, entrena, hace inferencias,
abre test o genera nuevas réplicas de calibración. Implementación y revisión
asistidas por IA declaradas.

## Resultados realmente ejecutados

* `current_input_audit_v2.json`: auditor vigente, **3,193 CPU s**. La celda
  R10/T768/B8191 y los archivos/ledgers correspondientes pasan la validación
  archivada, **únicamente con su alcance condicional**. El archivo previo
  `current_input_audit.json` se conserva como primera ejecución del auditor antes
  de ampliar su portabilidad a paquetes de datos versionados.
* `AUDIT_TEST_OUTPUT.txt`: **11 pruebas pasan** en 0,800 s.
* `FROZEN_POWER_STATISTICS_TEST_OUTPUT.txt`: **49 pruebas existentes pasan** en
  11,14 s. Son fixtures de software, no una validación adicional de mercado o una
  nueva calibración Monte Carlo.
* `planning_input_contract.json` y `planning_job_inventory.csv`: contrato
  ejecutable y **340 identidades de trabajo** necesarias para la planificación.
* `endpoint_counterexample.json`: contraejemplo numérico exacto de ordenación de
  métricas; `joint_composition_diagnostic.json`: comprobación exhaustiva de la
  implicación booleana y límites de las cotas reutilizadas.

La verificación positiva del método **no dimensiona**. No se proporcionaron CSV
y recibos reales completos de planificación ni un mapa local con cuatro
admisiones. El resultado es `NOT_DIMENSIONED`; R, T, B finales, potencia alcanzada
y MDE alcanzada permanecen **nulos**. No se interpreta la ausencia de entradas
locales como una inspección del disco privado de Colab.

## Tres procedimientos, estimando y multiplicidad

Los procedimientos son distintos y temporalmente pertinentes dentro de sus
supuestos: DM con corrección HLN y HAC/referencia gaussiana fixed-b escalar;
bootstrap estacionario studentizado con HAC recalculada; y Wald de momentos
condicionales tipo GW con constante y un instrumento predecible no constante.
GW contrasta un vector de momentos; su rechazo no demuestra una ventaja
direccional en todos los estados ni es una repetición escalar de DM.

Los errores cuadrados se promedian primero sobre el **panel fijo de modelos
entrenados por semilla**. La unidad de inferencia es la fecha cronológica, no
R×T observaciones iid. Los bloques son comunes a rivales y semillas. La regla
necesita instrumentos realmente disponibles en el origen, pérdidas débilmente
estacionarias, dependencia admisible y suficientes fechas; completar un CSV no
demuestra esos supuestos.

Hay 20 rivales × tres procedimientos × dos nulos —EPA y margen práctico—:
**Holm120 por panel**, con alfa operativa 0,025. La pantalla adversa adicional
reajusta conjuntamente **180 EPA** —20×3×3—. El margen retenido es 2 % de RMSE
agregada. La regla combina dirección, tres procedimientos, efecto y límites de
intervalos, además de dirección favorable de la métrica histórica. Los IC95/99
son marginales temporales; no se presentan como intervalos simultáneos de toda
la familia. El estudio de potencia simula esta conjunción completa y reporta
por separado un análisis de sensibilidad de colas; no reemplaza los p de la
regla por límites binomiales de otra regla más estricta.

## Brecha de métrica primaria: no la resuelve la monotonía

Para riesgos por semilla μ, el funcional heredado es
`mean_s(sqrt(μ_s))`; el funcional prospectivo contrastado es
`sqrt(mean_s(μ_s))`. La raíz es monótona para el segundo, pero ambos no son el
mismo funcional. Con riesgos **estrictamente positivos**:

| Modelo | MSE por semilla | Media de RMSE individuales | RMSE agregado |
|---|---|---:|---:|
| Candidata del contraejemplo | [1,21; 1,21] | 1,10 | 1,10 |
| Rival del contraejemplo | [0,01; 4] | 1,05 | 1,41598 |

La candidata parece mejor en el funcional nuevo y peor en el heredado. Son
números generados para una demostración, **no resultados de Bitcoin**. Exigir
dirección observada favorable en ambas métricas bloquea este caso constante,
pero **no transforma un p del MSE en una prueba de la media de RMSE**. El propio
protocolo A declara una enmienda prospectiva y ausencia de aceptación académica;
no se interpreta como autorización para cambiar la métrica heredada.

Antes de afirmar confirmación hace falta una decisión justificada compatible
con la guía y una inferencia validada para el funcional exacto. Si se conserva
la media de raíces, una ruta posible necesita reestimar esas raíces por semilla
en los bloques y tratar sus parámetros auxiliares/varianzas; no basta renombrar
la salida existente. No se implementa ni calibra esa nueva metodología aquí.

## Composición conjunta: qué prueba la cota y qué no

Fijemos una condición de superioridad realmente falsa según la población, no
según p observados. Si está en el panel principal, aceptar toda la decisión
requiere un falso positivo en ese panel. Si un rival tiene superioridad real en
como máximo uno de los tres adicionales, obtener dos positivos exige un falso
positivo en al menos uno de **dos paneles nulos fijados por la verdad**. Con
cotas marginales válidas qᵢ, la cota por unión es qᵢ+qⱼ, sin independencia.
Para una cota uniforme q, el máximo es **2q**. Las condiciones extra, incluida
la pantalla adversa Holm180, sólo pueden reducir el evento de aceptación.

La enumeración determinista verifica **192 casos** para ese testigo. Una cota
marginal verdadera de 0,025 permitiría deducir 0,05. Pero los límites **CP95
puntuales** conservados demuestran sólo ≤0,05 para cada escenario, no ≤0,025:

| Escenario | Falsos rechazos | Límite CP95 superior | Diagnóstico 2×límite |
|---|---:|---:|---:|
| Nulo completo AR(0,5), réplica focal | 50/2048 | 0,03206139 | 0,06412279 |
| Nulo parcial AR(0,5) | 7/512 | 0,02796545 | 0,05593089 |
| Nulo completo con solapamiento5 | 6/512 | 0,02533128 | 0,05066256 |
| Frontera práctica AR(0,5) | 15/512 | 0,04786087 | 0,09572175 |

Por tanto, **esas cotas por sí solas no certifican 5 % para la conjunción actual**.
Esto tampoco demuestra que su error real esté inflado: las cotas son
conservadoras y puntuales, no una calibración conjunta ni un límite simultáneo
de 95 % sobre todos los escenarios. Se conserva íntegro el nulo original
inconcluso16/512 y no se mezcla con las 2048 réplicas nuevas.

Existe una opción formal más estricta: exigir todos los adicionales y rivales
forma una intersección-unión con cota q. Se verifican **240 casos** de esa
implicación, pero **NO se adopta** la opción: endurecería el estándar de dos a
tres adicionales, podría reducir potencia y necesita su propia decisión
prospectiva. Tampoco se cambia alfa o se presenta Holm global como certificado
sin demostrar la validez marginal necesaria.

## Entradas exactas y presupuesto de planificación

Se requieren cuatro paneles, los 21 sistemas y las cinco semillas de
planificación preasignadas en la cola del banco. Son **320 ajustes neuronales,
4 ARIMA y 16 trayectorias analíticas**. Los deterministas se guardan una vez;
replicarlos en una matriz no crea nuevas réplicas independientes.

Cada panel debe tener al menos **80 fechas de planificación** comunes a sus
sistemas/semillas: mínimo **27.200 filas físicas de CSV**. Las columnas son
`panel,system,seed,date,y,prediction,known_floor_A,rfit`. Se necesitan recibos
genuinos de planificación, particiones purgadas, instrumentos causales,
identidades de protocolo/métodos/fuentes/datos, hashes y checkpoints reales de
cada trabajo, incluidos intentos fallidos en su contabilidad. Las fechas entre
paneles sólo se agrupan cuando coinciden realmente; acoplar calendarios
disjuntos independientemente es un supuesto condicional, no una covarianza
observada.

El techo nominal neuronal es **10 h 40 min** —320×120 s—, sin ARIMA, adquisición,
exportación, potencia, transferencias o interrupciones. Es aritmética nominal,
no consumo medido ni corte duro de reloj. La única celda actualmente admitida
es R10/T768/B8191; **no está seleccionada como tamaño final**. Otra R/T/B o
afinación requiere evidencia correspondiente. Las alternativas previstas de
ganancia3 % y5 % superan el margen2 %; no garantizan potencia0,90 para efectos
arbitrariamente próximos a ese margen. No existe MDE de mercado calculada.

## Ejecución reproducible y barreras

Después de instalar la fuente fijada y su entorno CPU:

```bash
python audit_power_inputs.py --source-root /ruta/fuente-verificada \
  --verify-ledgers --output auditoria-entradas-v1.json
```

Después de planificación auténtica y su exportación:

```bash
python audit_power_inputs.py --source-root /ruta/fuente-verificada \
  --predictions /ruta/caso/predictions/planning_predictions.csv \
  --receipts /ruta/caso/predictions/planning_predictions.receipt.json \
  --panel-map /ruta/private/development-panels.json \
  --verify-ledgers --output auditoria-entradas-v2.json
```

El auditor no sobrescribe recibos. Para un paquete nuevo de adaptación de datos,
el cargador suministra `--expected-source-commit` y
`--expected-manifest-sha256` propios y verificables. También se exige que los
archivos científicos A sigan idénticos. Los únicos metacampos de diseño que
pueden cambiar son versión, propuestas de instrumentos, declaración de cambio
de datos y estado; no se cambia margen, alfa, banco o simulación.

`final_gate.json` permite admisión y desarrollo con activación explícita, pero
mantiene **PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED** y
**JOINT_DECISION_VALIDATION_NOT_ESTABLISHED** antes de confirmar. No existe un
botón que cree la evidencia científica faltante. El MCS exploratorio nuevo de
la cohorte V5 es una contribución separada del subequipo `mcs`; no reemplaza la
inferencia de la primaria heredada ni la comparación reservada.
