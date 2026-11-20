# Procedencia consolidada del mecanismo y la implementación de E3

Documento retrospectivo de cierre con **asistencia de IA declarada**. Base documental: `29f9a26f1fd05484c6581ea6538ad7cb5a4781cb`. El documento consolida los cortes existentes sin sustituir sus fechas, formularios, hashes o resultados. No es un acta de autoría humana ni un protocolo anterior a los experimentos.

## Objetos y versiones que se mantienen separados

| Objeto | Núcleo y procedencia verificable | Alcance experimental |
|---|---|---|
| Primer módulo v1 | CNN causal/LSTM, atención con penalización de discordancia y saturación radial; formulación histórica y `src/original_model.py`. Saturación radial y cotas se reconocen como herramientas conocidas. | Sus pruebas y pilotos propios; no transferir sus propiedades completas a la receta aditiva o V3. |
| Referencia aditiva | `InnovationModel('additive_positive_head')`, 1210 parámetros; fuente científica `5a07d6157a9796ca54416dc5dea1e1527e731597`. | 796 trabajos de la auditoría exploratoria con master `20261119T031104515520Z`, más antecedentes históricos separados. |
| Cabezas por momentos | Líneas históricas con supuestos y resultados adversos conservados. | No son el mecanismo elegido en las tablas principales ni prueba de R1 de V3. |
| Adaptación V3 | `OverlapDistributionModel('atom_crps')`, 1262 parámetros, más controles explícitos; [fuente fijada](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/src/e3_overlap_model_v3.py). | Piloto `20261120T031424176339Z`, commit de resultados `8eaa581bc34b615a8507c9e384a3e4be29084720`: 30 ajustes de desarrollo, sin heredar los 796. |
| Control de agregación | `PluginSecondMomentModel`, variante `atom_plugin`,1262 parámetros; [fuente fijada](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/experiments/e3_v3_development_pilot_20261120/plugin_control.py). | Misma ley inicial, pérdida y capacidad que V3 completa, lectura por segundo momento; resultado conservado. |

## Fuente conceptual y adaptación

CNN, LSTM, atención por contenido y consulta aprendida son componentes establecidos. Luong et al. (2015) y Dasoulas et al. (2021) son antecedentes atribuidos del mecanismo de atención; la guarda por muestra `exp(z)+b` de Brosowsky et al. (2021), p.6816, es un antecedente del soporte de la aditiva. Cambiar a softplus y derivar el piso de seis retornos no establece por sí solo una adaptación sustantiva. La igualdad escalar de enlaces tampoco identifica clases neuronales finitas, gradientes o entrenamiento.

La modificación específica de V3 sustituye el exceso escalar por posiciones y masas de una ley del componente futuro no observado U; transforma cada átomo mediante la identidad de solapamiento antes de promediar e incorpora una señal CRPS_U normalizada. Bishop (1994) anticipa las leyes neuronales condicionales; Schittenkopf, Dorffner y Dockner (1998) aplican MDN a volatilidad financiera; Bellemare et al. (2017) es un antecedente categórico con objetivo y actualización distintos; Gneiting y Raftery (2007) sustentan CRPS. Feng et al. (2023) aporta otro antecedente probabilístico financiero, observado sólo por resumen/metadatos. **Estos son antecedentes bibliográficos, no una afirmación de que se copiara su implementación o de que fueran fuentes históricas del código.**

Las condiciones de acceso, localizadores y diferencias se conservan en `literature_crosswalk.csv`. El cierre reutiliza diez trabajos cercanos de la aditiva y cinco antecedentes de V3; no declara búsquedas o lecturas nuevas. Los cuerpos no accesibles no se reconstruyen desde resúmenes. El [expediente bibliográfico previo](https://github.com/jassan-bit/DeepLearning_202630/blob/29f9a26f1fd05484c6581ea6538ad7cb5a4781cb/book_e3/_static/e3_r1_close_20261120/README.md) permanece inmutable.

## Implementación, atribución y responsabilidad

Los módulos inspeccionados construyen el mecanismo mediante operaciones tensoriales, clases locales y primitivas de PyTorch. La documentación y el historial preparado registran que se escribieron desde la formulación con asistencia de IA, sin importar una arquitectura ajena como núcleo ni usar pesos preentrenados para estos objetos. Esta constatación sobre los archivos y declaraciones inspeccionados no es una auditoría forense de todo el historial ni certifica por sí sola autoría o comprensión humanas.

El autor del proyecto debe revisar, comprender, explicar y poder modificar cada línea relevante, las ecuaciones, la comparación bibliográfica y los resultados. La IA intervino en revisión, formulación, implementación, análisis y redacción; no se la presenta como una revisión humana, una firma del estudiante o una demostración oral. La sustitución del revisor externo de R1 por IA se mantiene conforme a la indicación docente comunicada por el autor. Esa autorización no sustituye la defensa de R2 ni una reproducción humana cuando ésta siga siendo exigida por su propio requisito.

## Propiedad y resultado que se atribuyen a cada objeto

La identidad `sigma7=sqrt(A+6/49*U²)` es álgebra elemental. V3 respeta el soporte y promedia exactamente su ley finita. Las cotas de Jensen/Taylor y la identidad de riesgo requieren sus supuestos; la ventaja bajo ley correcta se limita al control de segundo momento y no prueba inferioridad de la cabeza aditiva. El detalle y los 100 casos artificiales están en el [recibo anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/29f9a26f1fd05484c6581ea6538ad7cb5a4781cb/book_e3/_static/e3_r1_close_20261120/adaptation_design/analytic_verification.json); no se renombran como resultados de mercado ni como un teorema nuevo.

El [piloto de V3](https://github.com/jassan-bit/DeepLearning_202630/tree/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z/payload) sí aporta 30 ajustes reales de desarrollo y tres controles de componentes, con diferencias de capacidad declaradas. La receta completa tuvo **2,70 % mayor RMSE media que `atom_mse_only`**, por lo que falló la regla fijada de continuación. No se cambia esa regla, no se promueve automáticamente la variante mejor situada y no se computa significancia a partir de este orden descriptivo. La comparación no demuestra inexistencia de novedad; tampoco acredita R1/R3.

## Declaración admisible de contribución

**Se implementó, fundamentó y evaluó una referencia con soporte explícito y una adaptación candidata de ley atómica para el componente desconocido de la etiqueta solapada. Se aislaron pérdida, masas y agregación en un piloto común y se conservó el resultado adverso de la receta completa. La comparación bibliográfica delimita componentes conocidos y diferencias específicas, sin acreditar prioridad de la composición, superioridad confirmatoria ni novedad de las herramientas matemáticas.**

Los archivos complementarios de esta consolidación se conservan en `experiments/e3_evidence_close_20261120/contribution/`. Se consolidan las fuentes en `source_references.json`, los cambios en `mechanism_loss_training_application.csv`, el dictamen en `contribution_declaration.json` y los requisitos en `r1_evidence_matrix.csv`. El inventario de `experiments/design_log.md` separa propuestas, variantes, hiperparámetros y ejecuciones. La consolidación es retrospectiva: no recupera la independencia de calendarios expuestos ni convierte pruebas o archivos en firmas.

**Estado al cierre:** procedencia documental consolidada; piloto de desarrollo completo con heurística fallida; R1/R3 sin acreditar; defensa académica de R2 pendiente de observar. Cero entrenamientos e inferencias nuevos realizados para esta consolidación.
