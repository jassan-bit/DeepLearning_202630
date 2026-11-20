# Declaración precisa de contribución y decisión tras el piloto

**Estado:** consolidación documental retrospectiva, preparada con asistencia de IA declarada. No es un preregistro ni acredita R1/R3. La revisión externa por IA ya documentada en 6.13 se conserva conforme a la indicación docente comunicada por el autor; no se exige otra firma humana para esa modalidad. La defensa académica del estudiante permanece separada.

## Afirmación defendible

Se implementó desde la formulación una referencia CNN causal–LSTM–atención de contenido con soporte explícito para σ₇, usando módulos locales y primitivas de PyTorch. Sus componentes genéricos tienen antecedentes. La comparación delimitada no estableció una modificación sustantiva del exceso aditivo que acredite R1.

Como adaptación candidata se implementó una ley condicional de tres átomos de `U=|r_futuro−media6|`. La predicción transforma cada átomo por `sqrt(A+(6/49)U²)` antes de tomar la media, y la receta completa añade `0,1 CRPS_U` al MSE normalizado. Esta modificación de estructura y entrenamiento es concreta; no demuestra prioridad mundial de la composición exacta para la tarea ni superioridad sobre una cabeza escalar flexible. El archivo `mechanism_loss_training_application.csv` separa mecanismo, pérdida, entrenamiento y aplicación.

## Resultado observado y decisión

El [piloto inmutable](https://github.com/jassan-bit/DeepLearning_202630/tree/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z/payload) completó **30 ajustes**: 15 HPO y 15 finales, tres semillas finales para cada una de cinco recetas, más dos controles analíticos. Sus 55 fechas de evaluación pertenecen a desarrollo previamente expuesto; no abrió arrays de prueba/calibración ni empleó conjuntos adicionales.

La candidata `atom_crps` obtuvo RMSE media **0,00212991576846**; `atom_mse_only`, **0,00207383285786**; la aditiva, **0,00215728069125**. La candidata completa fue **2,70 % peor que el mejor control** y sólo **1,27 % mejor que la aditiva**. `fixed_probs` y `atom_plugin` también tuvieron menor RMSE media que la candidata completa. Por tanto **la heurística previa de al menos 2 % de mejora frente al mejor control no se superó**.

No se computaron valores p ni significancia por este piloto. El orden de tres medias por semilla sobre un calendario no demuestra equivalencia, superioridad poblacional o causa universal del deterioro. El CRPS_U medio de la receta completa tampoco fue el menor de las variantes distribucionales. El resultado no respalda una mejora atribuible al auxiliar CRPS para esta receta y presupuesto; no demuestra que CRPS sea perjudicial en cualquier modelo.

Se conserva el veredicto adverso. **No se promueve automáticamente a `atom_mse_only`**, no se reduce el umbral y no se renombra la hipótesis de la receta completa como éxito. Una variante futura requerirá decisión y versión nuevas, sin borrar las anteriores. Tampoco se atribuyen a V3 los 796 trabajos de la referencia aditiva.

## Propiedad formal y límites

La salida respeta el piso y calcula la esperanza exacta bajo su propia ley finita. La identidad elemental de varianza, la desigualdad de Jensen, la cota de curvatura y la descomposición del riesgo cuadrático son herramientas conocidas, con supuestos explícitos. La mejora teórica de riesgo se refiere exclusivamente al control de sustitución por segundo momento cuando la ley es correcta; no demuestra una limitación de representación de la red aditiva. Una ley de tres átomos puede estar mal especificada y CRPS no garantiza calibración o menor RMSE de muestra finita. Los 100 casos artificiales y contraejemplos anteriores conservan su recibo y no se presentan como datos de mercado.

## Brecha restante

La consolidación de fuentes, formulación, código, controles y resultado permite defender una adaptación candidata con límites. La prioridad específica y el dictamen favorable de R1 permanecen sin establecerse. R2 exige comprensión y modificación defendibles por el estudiante; R3 requiere su comparación, adecuación y diseño prospectivo propios. Ninguno se acredita por cantidad de pruebas o por el resultado favorable de otra variante.

Se remite a `experiments/design_log.md` y a su inventario reconciliado para los universos de diseños, parametrizaciones y ejecuciones. Semillas, dominios y repeticiones no son diseños algorítmicos nuevos; un total retrospectivo no recupera prerregistro ni cegamiento.
