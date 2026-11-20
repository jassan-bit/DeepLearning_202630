# V3: cierre de la decisión de continuación

**Decisión del 20 de noviembre de 2026: NO-GO para otra campaña confirmatoria de la receta actual `atom_crps`. R3 no acreditada.** Se cierra esta decisión con un resultado adverso documentado; no se declara demostrada una inferioridad universal ni se convierte la ausencia de confirmación en equivalencia.

Esta revisión posterior parte de `7db8040619f602d7c52ae0c8bf54f7b6f5b5110e`. La realizó una IA y no constituye una firma humana, una aceptación del docente ni un prerregistro nuevo. Se conservan los 796 trabajos de la referencia aditiva, los 30 ajustes del piloto V3 y las 51 trayectorias posteriores. No se vuelve a entrenar, calibrar, predecir o remuestrear.

## Por qué se detiene

1. La regla original del piloto requería al menos 2 % de reducción de RMSE frente al mejor control. `atom_crps` tuvo **2,7043119888 % más RMSE que `atom_mse_only`**, usando la media de RMSE individuales de tres semillas sobre 55 fechas de desarrollo. La regla no se cambia. El resultado favorable de la ablación no autoriza promoverla como candidata ni heredar el dictamen R1 de la receta completa sin revisar el cambio.
2. La pregunta pequeña de transferencia con pesos y escaladores BTC congelados **ya se ejecutó**. No queda un nuevo diagnóstico pendiente con esos mismos datos: las 51 trayectorias y 20.400 predicciones son el resultado de esa pregunta. La candidata completa no fue la mejor de sus siete recetas/controles en ninguno de los tres paneles. Estos rangos descriptivos no son contrastes de significancia.
3. No existe una nueva justificación empírica suficiente para gastar ahora una campaña grande buscando la superioridad de esa misma receta. El valor formal y el dictamen técnico acotado de R1 pueden defender una contribución investigativa aun cuando no se haya mostrado utilidad predictiva superior. No implican R2 o R3.
4. Los requisitos necesarios para una confirmación válida siguen abiertos: datos genuinamente no expuestos, variabilidad de los 19 rivales pertinentes, batería temporal revisada, potencia final, margen operativo y presupuesto. Escribir sus casillas no produce esa evidencia.

## Qué se volvió a comprobar

`audit_saved_science.py` verifica por SHA-256 **65 entradas congeladas**, recalcula la media de RMSE del piloto desde los pronósticos guardados, comprueba los 51 CSV de transferencia y concilia los 21 promedios de receta/panel con su resumen. También comprueba los recuentos de planificación y la identidad de la familia propuesta. El recibo es `execution_receipt.json` y las entradas están fijadas en `input_manifest.json`.

| Panel ya expuesto | Fechas | RMSE media individual de `atom_crps` | Mejor control observado | RMSE del control | Mayor error relativo de `atom_crps` |
|---|---:|---:|---|---:|---:|
| D01 AAPL RAW | 633 | 0,00331450836446 | `floor_plus_mean_excess` | 0,00262922358451 | 26,06 % |
| D02 ETHUSDT | 271 | 0,00716161520481 | `additive_positive_head` del piloto | 0,00657268211339 | 8,96 % |
| D03 BTC perpetuo | 296 | 0,00344427604027 | `floor_plus_mean_excess` | 0,00335670720694 | 2,61 % |

La elección de la columna «mejor control observado» es únicamente una descripción de todos los resultados conservados. No redefine el comparador del piloto, no escoge la candidata para producción y no reemplaza la comparación futura contra la familia completa. No se promedian las tres métricas entre dominios.

D01 corresponde a 2010–2013 y D02 a 2021; los pesos BTC se ajustaron con 2023–2024. D03 comparte subyacente y fechas con BTC. Es una transferencia retrospectiva sobre calendarios expuestos, no un backtest con disponibilidad histórica ni una prueba sellada. La fecha de registro del diagnóstico precedió a sus forwards, pero no volvió independientes los calendarios.

## Potencia: preparación disponible y límites

La guía entregada, §2 p.3 y §8.7.2 p.27, exige potencia de al menos 0,90 por contraste, potencia conjunta reportada, un piloto de al menos cinco semillas para las diferencias pareadas y al menos diez semillas finales o más según la potencia. La conjunta no tiene un umbral obligatorio de 0,90. El Cuadro 4 p.26 denomina esta tarea **F2, temporal**; no se inventa una categoría adicional a partir de la abreviatura usada en encargos anteriores.

La planificación existente contiene 162 escenarios de inicialización y 324 temporales: **486 sensibilidades analíticas, cero Monte Carlo**. La candidata tiene sólo tres semillas finales; de las seis varianzas disponibles, tres corresponden a ablaciones separadas. Sólo la aditiva del piloto y los dos controles analíticos aportan varianzas de la familia principal. Faltan las de los dieciséis E2, y no se sustituyen con las varianzas de la auditoría de la aditiva.

La familia preparatoria conserva 19 rivales, 20 sistemas y 57 componentes de prueba. Su Holm global y regla de máximo de tres p ajustados por rival siguen siendo una propuesta por validar y congelar. Las extensiones de DM/GW al estimando de media de RMSE individual no se declaran automáticamente admisibles por llevar nombres de la guía. Sobrevivir a un MCS de MSE tampoco demuestra por sí solo una mejora práctica en RMSE.

Por ello, `final_R`, calendario final y potencia final permanecen sin determinar. Ni diez semillas por defecto, ni el producto hipotético `0,9^57`, ni 365 fechas de una sensibilidad certifican el diseño. No se genera un tag de protocolo final ni una celda con una ruta de entrenamiento habilitada.

## Qué justificaría reabrir una investigación

La pregunta descriptiva sobre transferencia congelada ya está respondida. Para reabrir la investigación haría falta una pregunta distinta, falsable y vinculada a una limitación concreta, con razonamiento anterior a resultados nuevos. Por ejemplo, estudiar si el peso inducido por el cambio de variable en `CRPS_U` favorece determinados regímenes causales del solapamiento puede ser una pregunta metodológica; **no es, con esta evidencia, autorización para entrenar ni para seleccionar el régimen donde V3 gane**. Harían falta un criterio operacional previo y datos nuevos o un holdout externo genuinamente sellado. Si cambia la pérdida, el mecanismo, la selección o la receta, se crea otra versión, se reconcilia Nd y se revisa qué partes del dictamen R1 siguen aplicando.

La alternativa concreta ya disponible es presentar el resultado negativo completo: propiedad formal verificada y contribución V3 acotada, junto con el fracaso de la heurística y la transferencia descriptiva adversa. Esa es la decisión adoptada para la receta actual. No se promete R3 ni se inicia otra campaña con el propósito de obtener una nota determinada.

Los requisitos prospectivos y los campos que no pueden reconstruirse retrospectivamente quedan en `requirements_matrix.csv` y `REOPENING_CHECKLIST.md`. Sólo si una futura decisión satisface esas puertas procede completar el protocolo preparatorio existente y registrar su versión real antes de abrir resultados.
