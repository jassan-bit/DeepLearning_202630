# Apéndice A Plantillas operativas

Este apéndice aplica A.1–A.6 de la guía —páginas 49–54— al proyecto BTCUSDT, ruta temporal F2a. **Son plantillas y registros preventivos, no un preregistro, una campaña ejecutada, una firma o un nuevo congelamiento.** Al corte inicial de estas plantillas los documentos canónicos seguían pendientes. Ahora existe [experiments/estandar.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/estandar.md), con fecha real y alcance prospectivo posterior a los resultados conocidos; su tag y revisión humana siguen pendientes. El protocolo final, la procedencia consolidada y la bitácora global aún requieren cierre. Los formatos históricos no se convierten en registros previos o firmas.

Las fuentes editables se conservan en [experiments/e3_plantillas_v1](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_plantillas_v1). El paquete {download}`plantillas_operativas_v1.zip <../_static/e3_plantillas_v1/plantillas_operativas_v1.zip>` reúne los formatos; el {download}`README de uso y alcance <../_static/e3_plantillas_v1/README.md>` explica cómo completarlos sin sustituir evidencia. Las [secciones 6–8](08_verificacion_estadistica.md) y la [matriz de evaluación](14_evaluacion_calificacion.md) mantienen sus estados actuales.

El [borrador canónico de protocolo](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/protocolo.md), incorporado al revisar el Bloque D, tiene fecha real posterior a los resultados conocidos. Organiza los campos exigidos, con decisiones críticas abiertas, sin congelamiento ni tag confirmatorio. Las plantillas y el borrador histórico conservan sus bytes y su alcance.

## A.1 Declaración del estándar

La {download}`plantilla de estándar <../_static/e3_plantillas_v1/estandar_plantilla.md>` conserva los cuatro campos de la guía: estándar, justificación, compromisos y fecha/tag. Desglosa los pendientes necesarios para fijar propiedad/supuestos, rivales, conjuntos, potencia, recursos y responsables.

El estándar exigido es **teórico**: al menos una propiedad formal con verificación numérica; familia completa; al menos tres conjuntos adicionales; al menos tres pruebas válidas por contraste; potencia ≥0,90 por contraste y potencia conjunta reportada; mínimo diez semillas o más según dimensionamiento. Ningún piloto, coste, R definitivo, conjunto adicional o tag se da por completado mediante el formato. La cota existente se limita a la cabeza latente; no se amplía a OHLCV crudo por llenar una plantilla.

La justificación deberá conectar el cómputo y perfil del equipo con el análisis de potencia real. El registro tendrá fecha/tag anteriores al congelamiento, sin retrofecharlos. La regla completa de comparación y el hallazgo alternativo deben fijarse antes de confirmar, como explican las [secciones 8](08_verificacion_estadistica.md) y [14.5.9](14_evaluacion_calificacion.md).

El nombre `standard.md` del ejemplo de A.1 se corresponde aquí con `experiments/estandar.md`. La ubicación y la fecha real identifican el documento; cambiar su nombre o completar sus campos no crea un registro previo, una aprobación del estándar o un tag de congelamiento.

Fuente editable: [estandar_plantilla.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/estandar_plantilla.md).

## A.2 Declaración de procedencia del mecanismo

La {download}`plantilla de procedencia <../_static/e3_plantillas_v1/procedencia_plantilla.md>` solicita origen, cita completa del trabajo y del código consultado/licencia, motivo de transferencia, modificaciones algorítmicas e implementación desde la formulación. Añade campos de trazabilidad para asistencia de IA, revisión personal y capacidad de explicación/modificación.

Todos esos campos requieren validación: el formato no afirma que una fuente fue leída, que el código es original o que el autor comprende y puede defenderlo. La transferencia, si corresponde, debe introducir cambios sustantivos de mecanismo, pérdida o entrenamiento; cambiar sólo datos o hiperparámetros no acredita R1. El [registro ya existente del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/procedencia.md) y su asistencia declarada se preservan, sin convertirse en declaración final firmada.

Fuente editable: [procedencia_plantilla.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/procedencia_plantilla.md).

## A.3 Bitácora de diseño

La {download}`bitácora CSV vacía <../_static/e3_plantillas_v1/design_log_plantilla.csv>` contiene **sólo encabezados**. Conserva número de variante, cambio respecto a la anterior, métrica de desarrollo, decisión, motivo y fecha; añade identificadores, fuentes, fallos y tag para preservar trazabilidad. No copia las filas ilustrativas de la guía como si fueran ejecuciones.

La [bitácora inicial del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) registra cuatro variantes funcionales artificiales y Nd de mercado=0 **en su corte anterior a los pilotos**. Después se completaron cinco recetas entre v1/v2 y tres por momentos; las diez réplicas aditivas posteriores repiten una receta. Los [capítulos 6–7](06_modelo_original.md) conservan ese historial y la conciliación del Nd global pendiente. No se reinicia el conteo ni se inventan variantes, métricas, decisiones o tags. Los intentos futuros deberán acumularse con sus fallos, presupuesto y partición usada; una partición de confirmación no se usa para diseñar.

Fuente editable: [design_log_plantilla.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/design_log_plantilla.csv).

El [registro consolidado parcial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md), fechado el 17 de noviembre de 2026, incorpora las etapas ejecutadas y enlaza sus fuentes. Es un registro posterior, con Nd global aún sin conciliar; la plantilla vacía y la bitácora inicial conservan sus bytes y fechas.

## A.4 Tabla de contrastes confirmatorios

La {download}`tabla propuesta de contrastes <../_static/e3_plantillas_v1/contrastes_confirmatorios_plantilla.csv>` prepara **48 filas: 16 rivales × tres slots**, recuperando nombres y roles del [maestro histórico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloques_ab_20261115/master_configurations.csv). P1–P3 son lugares sin procedimiento elegido: no se contabilizan como pruebas admisibles, ejecutadas o rechazadas. El m≈12 ilustrativo de la guía no reduce los 16 rivales reales.

| Campo del formato | Regla de llenado |
|---|---|
| Contraste y slot | Candidato frente a cada rival, con tres procedimientos distintos cuya admisibilidad F2a sigue pendiente. |
| Prueba, nula, estimando y supuestos | Cerrar antes de confirmar; identificar unidad independiente y ámbito temporal/semilla. No basta nombrar tres métodos. |
| Estadístico, diferencia e IC95 %, efecto e IC | Vacíos hasta disponer de ejecución válida, fuente y recibo. No transferir valores de E2 ni llenar con ceros. |
| p crudo, p Holm y máximo por rival | Vacíos. La familia y su ajuste deben quedar cerrados en el protocolo. El formato histórico propone Holm sobre 16 rivales por procedimiento; esa propuesta no es una decisión confirmatoria vigente. Todas las pruebas exigidas deben rechazar y el máximo de sus p ajustados gobierna cada rival. |
| Decisión frente a δ y decisión conjunta | Vacías. Superar a todos requiere todos los rivales y condiciones R3; no elegir el menor p ni excluir un resultado no concluyente. |

La primaria sigue siendo la media de RMSE individuales en unidades de σ7. La ganancia propuesta es RMSErival−RMSEoriginal y δ=0,00025772842664981106, como margen heredado, no como resultado. Una hipótesis sobre MSE diaria no contrasta directamente ganancia de RMSE mayor que δ; **δ² no es una conversión válida del margen**. MCS no aporta por sí solo un contraste pareado de relevancia práctica. La ausencia de rechazo no se etiqueta «equivalente»: TOST/ROPE requieren margen y reglas previos.

Fuente editable: [contrastes_confirmatorios_plantilla.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/contrastes_confirmatorios_plantilla.csv). Esos slots y su propuesta de ajuste pertenecen a una plantilla sin diligenciar. La [sección 8](08_verificacion_estadistica.md) ya presenta un **análisis posterior distinto**, con tres procedimientos calculados y Holm global sobre 48 p-valores; no se usa esta plantilla para sustituir sus resultados o convertirlo en preregistro. El protocolo confirmatorio futuro sigue sin congelar.

## A.5 Lista de verificación de reproducibilidad

El {download}`checklist de 17 ítems <../_static/e3_plantillas_v1/checklist_reproducibilidad.csv>` reproduce los diecisiete ítems de la guía y añade estado, evidencia y vacío específico. No hay marcas de cumplimiento integral. «Parcial» reconoce evidencia real sin certificar todo lo agrupado en una fila.

El CSV descargable conserva el corte histórico `v1`. La tabla siguiente actualiza los estados del libro con la evidencia posterior; no reescribe aquel archivo ni transforma sus campos en certificaciones. Para el cierre deberán revisarse otra vez los 17 ítems sobre la misma versión entregada.

| N.º | Ítem de A.5 | Estado actual resumido |
|---|---|---|
| 1 | Estándar escrito antes de congelar. | Declaración canónica fechada disponible; tag y revisión pendientes, sin retrofechar experimentos. |
| 2 | Benchmarks congelados con hash después de A. | Parcial: freeze histórico disponible; Bloque A aún abierto. |
| 3 | Limitación cuantificada, hipótesis y predicciones secundarias. | Parcial: formulación/motivación; evidencia y registro finales pendientes. |
| 4 | Fuente y procedencia completas si hay transferencia. | Parcial: registro candidato y antecedentes; cierre por validar. |
| 5 | Propiedad formal, supuestos y verificación numérica. | Evidencia separada por arquitectura: la cota condicional de la cabeza aditiva tiene su propia comprobación de 1728 pares y 96 gradientes sintéticos. Las pruebas del primer módulo y de momentos conservan su identidad. Falta demostrar un beneficio de esa propiedad en los datos reales. |
| 6 | Bitácora Nd y desarrollo distinto de confirmación. | Pilotos y réplicas documentados; conciliación global y separación confirmatoria pendientes. |
| 7 | Protocolo con tag/fecha previos. | Protocolo confirmatorio pendiente; el plan versionado del cálculo posterior no lo sustituye. |
| 8 | Potencia, MDE y potencia conjunta. | Pendiente del candidato. |
| 9 | Familia completa con Holm. | Análisis posterior: 16 rivales, 48 contrastes y 48 rechazos tras Holm global; cierre confirmatorio pendiente. |
| 10 | Tres pruebas por contraste y máximo de p ajustados. | Tres procedimientos posteriores calculados con ámbitos declarados; pertinencia para la afirmación confirmatoria y demás requisitos por justificar. |
| 11 | Tres conjuntos elegidos antes de resultados y fuera del diseño. | D01–D03 admitidos y evaluados, junto con D00, con 19 sistemas por conjunto en el {ref}`cierre de 8.18 <e3-final-extended-audit>`. La selección y evaluación exploratoria están documentadas; las exposiciones compartidas, el cegamiento y el uso posterior para rediseñar deben conservarse en el dictamen confirmatorio. |
| 12 | Sensibilidad a prueba, métrica y semillas. | Análisis posterior disponible: bloques 7/14/28, MAE y subconjuntos descriptivos de semillas; no completa la robustez ampliada. |
| 13 | Unitarias en CI y exportación con ida y vuelta. | El CI del dashboard pasó 25 casos unitarios y seis recorridos de navegador. Cinco ONNX de la semilla 271801, incluido TSMixer, tienen paridad en 608 fechas con tolerancia 10⁻⁶ en σ₇; se conservan los tres TorchScript históricos. Faltan la política final común y las demás exportaciones. |
| 14 | Dashboard desde maestro y prueba de consistencia. | Aplicación de tres páginas y 17 sistemas disponible: el catálogo deriva de las tablas ampliadas del libro y su igualdad se comprueba automáticamente. Las curvas representan una semilla y no sustituyen las medias de diez réplicas; el maestro final conserva campos pendientes. |
| 15 | Docker y reproducción independiente con discrepancia. | El contenedor del servicio fue construido y probado localmente. La reconstrucción científica integral por una persona ajena al código, con discrepancia máxima por campo/unidad, sigue pendiente. Son alcances distintos. |
| 16 | Humo del despliegue y contingencia. | Servicio Docker gratuito de Render creado, observado en construcción con commit `ccefa68`: [URL asignada](https://deeplearning-e3-dashboard.onrender.com). El cliente de humo pasó en Gunicorn y Docker locales; estado Live, humo público y respaldo de la versión pública todavía sin verificar. |
| 17 | Archivo/DOI, fichas, datos, contribuciones e IA. | MODEL_CARD.md, DATA_STATEMENT.md, CITATION.cff y asistencia de IA declarada disponibles para el corte técnico. Archivo persistente/DOI, licencias/derechos finales, contribuciones ratificadas y firmas siguen pendientes; las fichas no los sustituyen. |

Fuente editable: [checklist_reproducibilidad.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/checklist_reproducibilidad.csv). La evidencia detallada y las condiciones de cierre siguen en las [secciones 12–14](12_lista_entregables.md) y el [anexo final](cierre_reproducibilidad.md).

## A.6 Catálogo de errores frecuentes

El {download}`catálogo preventivo de 16 errores <../_static/e3_plantillas_v1/errores_frecuentes.csv>` conserva los dieciséis casos de A.6, cada uno con síntoma observable y diagnóstico/corrección. **No afirma que el usuario haya cometido esos errores** ni sustituye una auditoría con evidencia.

| N.º | Error de A.6 | Síntoma observable que motiva revisar | Diagnóstico y corrección preventiva |
|---|---|---|---|
| 1 | Novedad sólo de aplicación. | El mecanismo ya existe y sólo cambia el dominio de uso. | Clasificar mecanismo, pérdida/entrenamiento o aplicación; si la diferencia es sólo aplicación, no acredita R1. |
| 2 | Transferencia sin citar la fuente. | El mecanismo trasladado se presenta como invención propia. | Citar la fuente original, describir la adaptación en procedencia y presentarla como tal. |
| 3 | Transferencia sin adaptación. | El modelo se traslada sin modificar el mecanismo; sólo cambian sus entradas. | Introducir y justificar una modificación algorítmica sustantiva. Un cambio de datos o hiperparámetros no basta para R1. |
| 4 | Demostración decorativa. | Se enuncia un teorema sin hipótesis explícitas o verificación. | Formular supuestos, demostrar formalmente y verificar numéricamente la propiedad con alcance delimitado. |
| 5 | Diseñar sobre confirmación. | La ventaja de validación desaparece en prueba o en conjuntos adicionales. | Separar desarrollo y confirmación, registrar Nd y estimar el sesgo de selección; la desaparición de la ventaja exige análisis, no prueba por sí sola la causa. |
| 6 | Reajustar benchmarks para comparar. | Los rivales se modifican después de conocer el resultado del candidato. | Congelar tras el Bloque A; cualquier corrección documentada exige revisar el protocolo y sus productos dependientes. |
| 7 | Presupuesto desigual disfrazado. | Hay igual número de intentos, pero el modelo propio consume mucho más cómputo. | Igualar el tiempo de cómputo total o reportar ambos análisis y el presupuesto real; no confundir intentos con coste. |
| 8 | Contrastes sin control familiar. | Se anuncian muchas diferencias con p crudos frente a rivales sin ajuste. | Declarar la familia completa y aplicar Holm sobre p válidos, conservando valores crudos y ajustados. |
| 9 | Elegir la prueba favorable. | Sólo se presenta el procedimiento que rechaza. | Reportar todas las pruebas exigidas; todas deben rechazar y el máximo de sus p ajustados gobierna la decisión por rival. |
| 10 | Diseño sin potencia. | No se rechaza frente a la familia pese a una media favorable. | Dimensionar antes de congelar con la familia real —16 rivales aquí—, potencia por contraste/conjunta y semillas suficientes. La ausencia de rechazo no demuestra equivalencia. |
| 11 | Relajar el estándar tras resultados. | Se reduce la familia o el número de pruebas después de no rechazar toda la familia. | Mantener el estándar y el historial; declarar límites sin reducir las exigencias después de congelar el protocolo. |
| 12 | Conjuntos adicionales usados para ajustar. | Los conjuntos usados para diseñar se presentan después como evidencia independiente. | Elegirlos antes de observar el desempeño; si guiaron cambios, declararlos parte del desarrollo. |
| 13 | Dashboard desincronizado. | Las cifras de la aplicación difieren de las del informe para la misma versión y estimando. | Generar ambos desde la misma tabla maestra y comprobar automáticamente su igualdad, identificando versión y alcance. |
| 14 | Entrada fuera de dominio sin aviso. | Se muestran predicciones como fiables para datos alejados del entrenamiento. | Verificar rangos o distancias de ajuste y advertir al usuario; el control no garantiza fiabilidad dentro del rango. |
| 15 | Entorno no reproducible. | El proyecto funciona en la máquina del autor y falla al reconstruir el contenedor. | Fijar versiones y construir/probar el Dockerfile en un entorno limpio, conservando logs y discrepancias. |
| 16 | Despliegue sin contingencia. | La URL no responde durante la sustentación. | Ejecutar el humo, abrir la URL con antelación y conservar capturas o vídeo reales de respaldo, con su versión y alcance. |

Fuente editable: [errores_frecuentes.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/errores_frecuentes.csv). El catálogo orienta correcciones documentadas; cada estado se respaldará con evidencia, sin resultados, registros o firmas ficticios.
