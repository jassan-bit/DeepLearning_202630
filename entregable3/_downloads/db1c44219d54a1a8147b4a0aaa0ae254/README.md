# Puertas de diseño R3, versión 1

Preparado con asistencia de IA. Extensión ejecutable del contrato existente, no una campaña ni un preregistro. La entrada actual es `protocol.DRAFT.json`: **candidata, familia, R y T finales sin fijar; ejecución bloqueada; R3 no acreditada**. La fuente documental es `main` en `2fbab63f7c4f19edcd7e9927c0c99d7ddae5d9f5`.

`DEPENDENCIES.json` fija por SHA-256 el módulo y la plantilla previos. Se importan sus utilidades de rutas y hashes y se heredan estimando/unidades/poblaciones. No se duplican sus datos ni se repite la auditoría de 71 fuentes y 486 escenarios. El umbral `T>=32` de aquel contrato no se adopta como requisito universal: este controlador exige T entero compatible con el calendario y con el alcance validado de la metodología.

El inventario ya existente conserva 3/19 varianzas V3 de desarrollo y 16 ausentes; 76 celdas futuras V3 y 84 V4 vacías. Las cinco referencias V4 y 25 covarianzas son de desarrollo expuesto. V4 está abandonada: **los 340 ajustes nominales no están autorizados**. BTCUSDT/SPY/USO/TLT son cuatro propuestas, con **0/4 admitidas**; no se descargó información de mercado.

## Uso y resultados

Desde esta carpeta, con un checkout que contenga las dependencias fijadas:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p test_controller.py -v
PYTHONDONTWRITEBYTECODE=1 python controller.py \
  --contract protocol.DRAFT.json \
  --repo-root /ruta/al/checkout \
  --evidence-root /ruta/a/recibos \
  --phase confirmation \
  --output /ruta/nueva/preflight.json
```

El CLI devuelve **2 cuando está bloqueado**, 0 cuando la compatibilidad técnica condicionada se satisface y error ante entrada inválida. Rechaza sobrescribir su salida. `CURRENT_PREFLIGHT.json` es la ejecución real de metadatos del borrador: nueve puertas bloqueadas y ningún trabajo emitido. No hay ajustes, predicciones, Monte Carlo, ni ejecución del programa de dimensionamiento anterior.

`--phase planning` comprueba los requisitos operativos de un piloto separado de cinco o más semillas; permite R/T finales nulos y no exige todavía covarianza/potencia final. `--phase confirmation` añade la covarianza completa, R≥10 coherente con las semillas finales, T por panel y potencia de toda la regla. La candidatura y su propia R1/R2, cuatro paneles y dos ejes adicionales son condiciones de esta ruta de revisión; no se presentan como requisitos universales para cualquier trabajo exploratorio.

Cuando una fase está completa, se produce un manifiesto determinista de trabajos con identidad de diseño, panel, receta, semilla y techo por ajuste. Los controles deterministas tienen un trabajo por panel, sin semillas ficticias. `checkpoint_state` sólo permite saltar trabajos completos con identidad exacta y artefactos cuyos hashes se verifican; un ejecutor separado revisado debe aplicar realmente la persistencia atómica y los límites de tiempo/intentos. El preflight no entrena ni lanza ese ejecutor.

`technical_preflight_ready` y `execution_permit_conditionally_ready` indican consistencia y documentación para revisión externa. **El software no autentica revisores, autorizaciones o independencia, ni acredita adecuación científica o R3**, incluso con todos los campos compatibles. La prueba positiva usa exclusivamente recibos sintéticos en un directorio temporal con URLs `synthetic.invalid`; no es evidencia del proyecto.

## Entregable de revisión

La plantilla se completa sin cambiar su significado retrospectivamente. `design_sha256` es SHA-256 del JSON canónico de `design`; las evidencias se vinculan fuera de él para evitar hashes circulares. Todo binding usa `{path,sha256}`, relativo al directorio de evidencias, sin escapes ni symlinks. Las rutas de importación previas se comprueban también por bytes. Los recibos científicos comparten versión y hash de candidata; admisión, calibración, covarianza y potencia comparten además hash del diseño/familia, estimando, unidades y las dos poblaciones de muestreo heredadas.

| Entrada a completar | Relación que comprueba el controlador | Juicio o acto externo pendiente |
|---|---|---|
| `design.candidate`; recibos `r1`, `r2`, `candidate_decision` | Abre los bytes candidatos; ambas acreditaciones y continuidad corresponden a ese hash/versión; rechaza receta abandonada | R1/R2 auténticas y viabilidad científica propias; no heredar V3 a V4 o a otra receta |
| `design.family.rivals`, `top10`, `strong_baselines`, `other_declared_rivals` | Identidades únicas y cobertura exacta, receta vinculada y tipo estocástico/determinista | Justificar el inventario relevante completo sin exclusión por resultados |
| Tres `methods` con `definition`, `implementation`, `contrast_mapping` | Definiciones distintas, mismo funcional/poblaciones/nulo/dirección, código por hash; al menos un procedimiento global o múltiple | Pertinencia conforme a la guía y derivación estadística de cada adaptación |
| Mapeo `component_keys_by_panel_rival` y `multiplicity_groups` | Cobertura exacta de hipótesis crudas y requisitos contrastivos, ajuste/nivel/justificación declarados; un p global no puede repetirse como contraste directo | Traducción global→contraste mediante exclusión direccional validada; no fijar 19/21 rivales, tres familias ni Holm38 como ley universal |
| Cuatro `design.panels`; recibo `admission` | Misma tarea sigma7 no anualizada, horizonte1, soporte declarado de ocho cierres; fechas ISO ordenadas, particiones y rangos de cierres sin solapamiento; hashes de archivos; ≥2 ejes por adicional | Fuente/licencia/permisos, exposición auténtica, justificación de diversidad y dependencia, exactitud de los soportes declarados |
| `sampling`, márgenes y alternativas por panel | Semillas enteras distintas; excluye bool/float; ≥5 planificación; unidad del margen; alternativa finita estrictamente mayor que δ; no IID de semilla×fecha | Semillas realmente independientes, estacionariedad/localidad, δ operativo y alternativa razonable previos al resultado |
| `calibration` y su protocolo externo congelado | Hashes de métodos/código/regla y familia; alcance R/T; nulos completos/parciales y estrés para la regla completa y cada grupo; cotas superiores MC recalculadas contra el nivel propio de cada alcance | Autenticidad del registro, aceptación metodológica y cobertura de las condiciones reales; tres cribados exploratorios no bastan |
| `paired_covariance.crossed_covariance_payload` | Abre archivo/cubo por hash; claves completas semilla×fecha×rival, errores cuadrados finitos, misma trayectoria candidata; covarianza de diferencias de RMSE individuales reproducida; matrices completas, simétricas y PSD con unidad al cuadrado | Identificación/modelo de componentes de semilla, LRV temporal e interacción; estimación temporal y dependencia entre rivales/paneles justificadas |
| `power.replicate_decisions` | Lee decisiones ya guardadas: intersección de tres procedimientos más restantes criterios por contraste; cotas MC simultáneas ≥0,90; conjunta e intervalo desde las mismas réplicas; R/T exactos | DGP y réplicas independientes, regla correctamente implementada, potencia prospectiva y alcance poblacional válido |
| Registros de planificación/confirmación | Diseño archivado idéntico; URI externa, revisión inmutable, custodio y fechas; registro anterior al primer acceso a resultados | **Preregistro real**, con prueba auténtica del orden de acceso. Un archivo DRAFT/hash local no lo sustituye |
| `resources`, `campaign_authorization`, `runner_review`, `persistence` | Contabilidad de trabajos y techos; autorización exacta de diseño/fase; ejecutor revisado; checkpoints e integridad | Recursos disponibles, gasto autorizado y aplicación efectiva de límites/persistencia |

Los campos concretos de cada recibo y un ejemplo estructural completo pueden revisarse en `test_controller.py`. La prueba es generada temporalmente y usa otra candidata, no completa las celdas históricas ni selecciona un diseño de mercado. Los nuevos recibos auténticos deben conservar sus propias fuentes, autoría, fecha y restricciones.

Los niveles de error se vinculan por alcance, sin tomar el máximo entre grupos. `decision_rule.complete_rule_error_control` declara explícitamente evento, alfa y justificación de la regla confirmatoria completa; esta ruta exige alfa≤0,05 para ese control principal. Cada criterio congelado especifica `scope`, `nominal_alpha`, techo de aceptación y el hash de la regla o `group_id` exacto. Cada fila debe corresponder a ese mismo evento/alcance y su cota superior MC no puede superar el techo propio. Se exigen los escenarios de calibración tanto para la regla completa como para cada grupo declarado. Un nivel global0,10, si está justificado y explícitamente elegido, no relaja un control principal0,05. La prueba de regresión usa ambos niveles; no impone universalmente0,10 a MCS u otro método global.

## Potencia, calendario y costes

El objetivo heredado es `E_seed[sqrt(E_P error_rival²)-sqrt(E_P error_candidate²)]`, media de RMSE individuales. No equivale a RMSE del ensamble ni diferencia de MSE medio. El controlador reconstruye la covarianza entre diferencias individuales de semilla desde el cubo pareado; las matrices temporales requieren una derivación vinculada y revisión externa. No calcula potencia observada ni elige R/T a partir de ganancias expuestas.

El diagnóstico determinista vecino `../qa/FINITE_CALENDAR_RMSE_BIAS_AUDIT.json` muestra un problema adicional: con RMSE poblacional1 y T=128, la expectativa de la RMSE muestral difiere aproximadamente **0,0093251** entre dependencia IID y AR(1) con φ=0,85. No es una validación nueva de tamaño/cobertura ni un teorema de fallo universal. Promediar más semillas conserva ese sesgo de T finito; no basta aumentar R. La calibración deberá cubrir igualdad poblacional con dependencia heterogénea o restringir explícitamente su alcance y explicar la diferencia entre el estimando poblacional y el de calendario finito.

La guía exige ≥0,90 por contraste y reportar la conjunta, sin un umbral conjunto universal0,90. Este controlador usa una opción conservadora: cota inferior MC simultánea de Hoeffding por contraste, `p_hat−sqrt(log(m/eta)/(2B))`, a partir de B réplicas guardadas independientes. Para la conjunta usa las mismas réplicas e intervalo bilateral `p_joint±sqrt(log(2/eta)/(2B))`. Son garantías MC separadas condicionadas al DGP y a la independencia; su garantía agregada es como mínimo `1−2eta`, no `1−eta`. Exigir la cota inferior≥0,90 es una regla operativa más fuerte que exigir solamente una estimación puntual≥0,90. Otros intervalos válidos requerirían una extensión revisada; no se importan sin comprobación.

`COST_INPUTS.json` deja los costes de tiempo/dinero sin medir en null. Para P paneles, J rivales y tres procedimientos hay **P×J celdas** y **P×J×3 requisitos contrastivos**; el número de hipótesis crudas depende del mapeo real. Con S semillas piloto y R finales, los trabajos son `P×(N_estocásticos×S+N_deterministas)` y `P×(N_estocásticos×R+N_deterministas)`, contando la candidata entre los estocásticos. Selección/HPO, preparación, predicción, almacenamiento, fallos y análisis se contabilizan aparte. El manifiesto de recetas fijas exige terminar y congelar HPO antes; no inventa sus trabajos. S≥5 y R≥10 no demuestran potencia suficiente.

La suma de techos de ajuste es un límite contable serial. No es una medición ni predice duración de pared; paralelismo, preparación y análisis requieren sus propios recursos y cronograma medido. No se extrapolan los 120s ilustrativos de versiones anteriores ni los 340 ajustes de V4. La confirmación sigue bloqueada hasta candidata viable, métodos adecuados, datos admitidos, dimensionamiento prospectivo de la regla completa y registro/autorización auténticos.
