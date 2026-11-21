# Ruta de acreditación de R3: resultados y puertas pendientes

**Dictamen actual: R3 pendiente, no acreditada.** Este suplemento conserva los estudios anteriores y ejecuta una revisión estadística nueva y acotada. No habilita una campaña confirmatoria ni selecciona otra candidata.

La fuente de entrada es `2fbab63f7c4f19edcd7e9927c0c99d7ddae5d9f5`. La investigación y su documentación utilizaron IA declarada. Sustentación **4,5** es un dato proporcionado por el autor; no se afirma haber observado la defensa ni acreditar R2.

## Trabajo ejecutado

- Se contrastó directamente la guía original con **34 condiciones**, sus páginas, evidencia y puertas; se verificaron **48 insumos**. La extracción del PDF no se redistribuye. [Matriz y procedencia](guide_candidate/README.md).
- Se revisó cada candidata: aditiva, V3, V4 y controles MSE-only. Ninguna está habilitada actualmente para confirmación. Esto no demuestra que una propuesta futura sea imposible. [Decisión y material para el autor](guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json).
- Se implementó una nueva propuesta P1 bootstrap-t para el mismo estimando de media de RMSE individuales, conservando P2 delta-HAC y el MCS v2 como controles propuestos. [Método, resultados y límites](statistical_revision/README.txt).
- Se publicó código y plan antes de validar: [commit `374fc00`](https://github.com/jassan-bit/DeepLearning_202630/commit/374fc00f50da4454522c04a7973d0ef590d5e192). El recorrido Merkle comprobó las trece rutas públicas y el ancla precede al inicio real. Es un plan de desarrollo estadístico, no un preregistro confirmatorio de mercado.
- Se ejecutó una sola batería nueva: **1.664 réplicas**, **1.662.336 remuestreos**, **479,82 segundos**, dos hilos, cero errores y ocho fuentes de ejecución intactas. Se conservaron datos crudos, decisiones y controles sobre las mismas muestras, sin remuestreos adicionales para el control.
- Se preparó y probó el controlador de admisión con **29 guardas de software**: siete puertas de planificación y nueve confirmatorias siguen bloqueadas con los insumos reales. Sus pruebas positivas sólo muestran consistencia técnica hipotética. [Protocolo DRAFT, contrato y presupuesto](design_gates/README.md).

## Qué muestra la validación nueva

La familia evaluada contiene dos procedimientos propuestos por diecinueve rivales, con Holm38. Es una elección explícita del ensayo, no una obligación universal de la guía. Se mantienen separados la igualdad, la dirección, el margen práctico y el evento de intersección de ambos métodos.

| Escenario | Falsos rechazos nuevos / 256 | Control básico anterior, mismas muestras | Cribado prefijado nuevo |
|---|---:|---:|---|
| Variación sólo entre semillas | 6 | 31 | Superado, alcance exploratorio |
| Dependencia temporal moderada | 9 | 13 | Superado, alcance exploratorio |
| Dependencia temporal fuerte | 12 | 7 | **Inconcluso** |
| Nulo parcial verdadero | 7 | 10 | Superado, alcance exploratorio |
| Escalas heterogéneas entre semillas | 6 | 8 | Superado, alcance exploratorio |
| Dependencia temporal heterogénea entre modelos | 2 | 3 | Superado, alcance exploratorio |

En dependencia fuerte, el IC95 % individual de FWER es **[2,4452 %, 8,0449 %]**. Su límite superior supera el 8 % establecido para el cribado de desarrollo; redondearlo no permite aprobarlo. La mejora del caso entre semillas no es uniforme: la comparación pareada tiene Holm6 p=3,576×10⁻⁷ allí y p=1 en las otras cinco comparaciones. Los porcentajes e intervalos no certifican control nominal exacto, simultáneo o de mercado.

Se conservaron **nueve señales marginales descriptivas de infracobertura** —ocho P1 y una P2— entre las 266 celdas de cobertura. Sus intervalos Monte Carlo son individuales; no constituyen nueve conclusiones con control familiar. El MCS conservado fue inconcluso en el cribado global moderado y excluyó algún verdadero óptimo en 30/256 réplicas del nulo parcial; no se convierte ese resultado en un p-valor por rival.

En la alternativa, la intersección de los dos procedimientos rechazó frente a todos los rivales en **20/128** réplicas; la subregla que añade dirección y límites puntuales sobre δ, en **16/128**. El MCS retuvo únicamente la candidata en **35/128**. Son eventos distintos, ninguno acredita potencia de la regla completa de tres procedimientos o potencia final de mercado.

[Tablas de familia/MCS](statistical_revision/exports/family_and_MCS.csv) · [Cobertura y contrastes](statistical_revision/exports/pointwise_coverage_and_pairwise.csv) · [Potencia del subconjunto](statistical_revision/exports/two_method_subset_power.csv) · [Decisión conservada](statistical_revision/RESULTS_DECISION.json).

## Tercer procedimiento y diseño prospectivo

Se documentó una tercera vía global: exigir que un MCS válido conserve únicamente la candidata. Si el conjunto cubre todos los modelos óptimos, ese resultado implica que la candidata es la única mejor, con el alcance del teorema de cobertura. La extensión concreta semilla×tiempo todavía necesita justificar esa cobertura y su correspondencia con la guía. No produce diecinueve p-valores independientes ni prueba δ. La familia prospectiva y α siguen sin fijar; los cribados históricos a 0,10 no acreditan una regla a 0,05. [Admisibilidad y plan del tercero](methods_admissibility/README.md).

También se verificó mediante cuadratura determinista que modelos con igual RMSE poblacional pueden tener distinto sesgo de RMSE en calendarios finitos. Aumentar semillas no elimina ese sesgo. El dimensionamiento debe considerar conjuntamente R/T, dependencia, varianzas y covarianzas. [Derivación y valores](qa/FINITE_CALENDAR_RMSE_BIAS_AUDIT.json).

La candidata, la familia futura, R/T y el presupuesto permanecen sin seleccionar. Los cuatro paneles propuestos siguen **0/4 admitidos**. Las fuentes, derechos, exposición, equivalencia de tarea y disponibilidad efectiva necesitan sus comprobaciones; las fechas futuras del borrador no son datos ya obtenidos. No se ejecutan los 340 ajustes ilustrativos de la V4 abandonada.

La guía pide piloto de al menos cinco semillas y diez finales o más según dimensionamiento, potencia mínima **0,90 por contraste** y reporte conjunto con precisión. No fija universalmente potencia conjunta ≥0,90. El controlador lee evidencia guardada y prepara trabajos sólo con recibos coherentes; no certifica su autenticidad científica ni convierte booleanos en R3.

## Verificar y preparar la siguiente decisión

Desde el checkout completo, la verificación del paquete es de sólo lectura y no repite la simulación:

```bash
export PYTHONDONTWRITEBYTECODE=1
PKG=experiments/e3_r3_route_20261121
python "$PKG/bundle_tools/verify_package.py" --packet "$PKG"
```

El controlador puede evaluarse con el borrador y una salida nueva fuera del paquete:

```bash
python "$PKG/design_gates/controller.py" \
  --contract "$PKG/design_gates/protocol.DRAFT.json" \
  --repo-root . --evidence-root . --phase confirmation \
  --output /tmp/r3_preflight_nuevo.json
```

Con estos insumos devuelve código **2** y nueve bloqueos, un resultado esperado que no debe ocultarse como fallo de instalación. La salida ya existente se rechaza. Los procedimientos de reproducción deliberada y dependencias estadísticas constan en el README del componente; cualquier reproducción debe usar otra carpeta y conservar su identidad.

Se preservaron los **796 trabajos aditivos**, **30 ajustes V3**, **51 trayectorias/20.400 predicciones retrospectivas** y **20 ajustes/1.210 predicciones V4**, además de las validaciones anteriores de 1.024 y 640 réplicas. V3 conserva NO-GO, V4 abandono y MSE-only su papel de control. Esta operación añadió **cero ajustes de pronóstico, forwards y predicciones de mercado**; no inició V5 ni cambió Render, ONNX o el DOI publicado.

[Estado y siguientes actuaciones](NEXT_ACTIONS.md) · [Resumen verificable](CLOSURE_STATE.json) · [Acta posterior de publicación](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-r3-route-proof-2fbab63/proof).
