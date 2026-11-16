# Apéndice A Plantillas operativas

Este apéndice aplica A.1–A.6 de la guía —páginas 49–54— al proyecto BTCUSDT, ruta temporal F2a. **Son plantillas y registros preventivos, no un preregistro, una campaña ejecutada, una firma o un nuevo congelamiento.** Los documentos canónicos `experiments/estandar.md`, `experiments/protocolo.md`, `experiments/procedencia.md` y `experiments/design_log.md` siguen pendientes; publicar los formatos no los crea ni cierra sus decisiones.

Las fuentes editables se conservan en [experiments/e3_plantillas_v1](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_plantillas_v1). El paquete {download}`plantillas_operativas_v1.zip <../_static/e3_plantillas_v1/plantillas_operativas_v1.zip>` reúne los formatos; el {download}`README de uso y alcance <../_static/e3_plantillas_v1/README.md>` explica cómo completarlos sin sustituir evidencia. Las [secciones 6–8](08_verificacion_estadistica.md) y la [matriz de evaluación](14_evaluacion_calificacion.md) mantienen sus estados actuales.

## A.1 Declaración del estándar

La {download}`plantilla de estándar <../_static/e3_plantillas_v1/estandar_plantilla.md>` conserva los cuatro campos de la guía: estándar, justificación, compromisos y fecha/tag. Desglosa los pendientes necesarios para fijar propiedad/supuestos, rivales, conjuntos, potencia, recursos y responsables.

El estándar exigido es **teórico**: al menos una propiedad formal con verificación numérica; familia completa; al menos tres conjuntos adicionales; al menos tres pruebas válidas por contraste; potencia ≥0,90 por contraste y potencia conjunta reportada; mínimo diez semillas o más según dimensionamiento. Ningún piloto, coste, R definitivo, conjunto adicional o tag se da por completado mediante el formato. La cota existente se limita a la cabeza latente; no se amplía a OHLCV crudo por llenar una plantilla.

La justificación deberá conectar el cómputo y perfil del equipo con el análisis de potencia real. El registro tendrá fecha/tag anteriores al congelamiento, sin retrofecharlos. La regla completa de comparación y el hallazgo alternativo deben fijarse antes de confirmar, como explican las [secciones 8](08_verificacion_estadistica.md) y [14.5.9](14_evaluacion_calificacion.md).

Fuente editable: [estandar_plantilla.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/estandar_plantilla.md).

## A.2 Declaración de procedencia del mecanismo

La {download}`plantilla de procedencia <../_static/e3_plantillas_v1/procedencia_plantilla.md>` solicita origen, cita completa del trabajo y del código consultado/licencia, motivo de transferencia, modificaciones algorítmicas e implementación desde la formulación. Añade campos de trazabilidad para asistencia de IA, revisión personal y capacidad de explicación/modificación.

Todos esos campos requieren validación: el formato no afirma que una fuente fue leída, que el código es original o que el autor comprende y puede defenderlo. La transferencia, si corresponde, debe introducir cambios sustantivos de mecanismo, pérdida o entrenamiento; cambiar sólo datos o hiperparámetros no acredita R1. El [registro ya existente del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/procedencia.md) y su asistencia declarada se preservan, sin convertirse en declaración final firmada.

Fuente editable: [procedencia_plantilla.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/procedencia_plantilla.md).

## A.3 Bitácora de diseño

La {download}`bitácora CSV vacía <../_static/e3_plantillas_v1/design_log_plantilla.csv>` contiene **sólo encabezados**. Conserva número de variante, cambio respecto a la anterior, métrica de desarrollo, decisión, motivo y fecha; añade identificadores, fuentes, fallos y tag para preservar trazabilidad. No copia las filas ilustrativas de la guía como si fueran ejecuciones.

La [bitácora real del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) sigue declarando cuatro variantes funcionales artificiales y Nd de mercado=0. No se reinicia ese historial ni se inventan variantes, métricas, decisiones o tags. Los intentos futuros deberán acumularse con sus fallos, presupuesto y partición usada; una partición de confirmación no se usa para diseñar.

Fuente editable: [design_log_plantilla.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/design_log_plantilla.csv).

## A.4 Tabla de contrastes confirmatorios

La {download}`tabla propuesta de contrastes <../_static/e3_plantillas_v1/contrastes_confirmatorios_plantilla.csv>` prepara **48 filas: 16 rivales × tres slots**, recuperando nombres y roles del [maestro histórico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloques_ab_20261115/master_configurations.csv). P1–P3 son lugares sin procedimiento elegido: no se contabilizan como pruebas admisibles, ejecutadas o rechazadas. El m≈12 ilustrativo de la guía no reduce los 16 rivales reales.

| Campo del formato | Regla de llenado |
|---|---|
| Contraste y slot | Candidato frente a cada rival, con tres procedimientos distintos cuya admisibilidad F2a sigue pendiente. |
| Prueba, nula, estimando y supuestos | Cerrar antes de confirmar; identificar unidad independiente y ámbito temporal/semilla. No basta nombrar tres métodos. |
| Estadístico, diferencia e IC95 %, efecto e IC | Vacíos hasta disponer de ejecución válida, fuente y recibo. No transferir valores de E2 ni llenar con ceros. |
| p crudo, p Holm y máximo por rival | Vacíos. Propuesta: Holm sobre 16 rivales para cada procedimiento previamente cerrado; todas las pruebas exigidas deben rechazar. El máximo ajustado gobierna cada rival. |
| Decisión frente a δ y decisión conjunta | Vacías. Superar a todos requiere todos los rivales y condiciones R3; no elegir el menor p ni excluir un resultado no concluyente. |

La primaria sigue siendo la media de RMSE individuales en unidades de σ7. La ganancia propuesta es RMSErival−RMSEoriginal y δ=0,00025772842664981106, como margen heredado, no como resultado. Una hipótesis sobre MSE diaria no contrasta directamente ganancia de RMSE mayor que δ; **δ² no es una conversión válida del margen**. MCS no aporta por sí solo un contraste pareado de relevancia práctica. La ausencia de rechazo no se etiqueta «equivalente»: TOST/ROPE requieren margen y reglas previos.

Fuente editable: [contrastes_confirmatorios_plantilla.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/contrastes_confirmatorios_plantilla.csv). La [sección 8](08_verificacion_estadistica.md) conserva el protocolo no congelado y la batería todavía por validar.

## A.5 Lista de verificación de reproducibilidad

El {download}`checklist de 17 ítems <../_static/e3_plantillas_v1/checklist_reproducibilidad.csv>` reproduce los diecisiete ítems de la guía y añade estado, evidencia y vacío específico. No hay marcas de cumplimiento integral. «Parcial» reconoce evidencia real sin certificar todo lo agrupado en una fila.

| N.º | Ítem de A.5 | Estado actual resumido |
|---|---|---|
| 1 | Estándar escrito antes de congelar. | Pendiente de declaración canónica. |
| 2 | Benchmarks congelados con hash después de A. | Parcial: freeze histórico disponible; Bloque A aún abierto. |
| 3 | Limitación cuantificada, hipótesis y predicciones secundarias. | Parcial: formulación/motivación; evidencia y registro finales pendientes. |
| 4 | Fuente y procedencia completas si hay transferencia. | Parcial: registro candidato y antecedentes; cierre por validar. |
| 5 | Propiedad formal, supuestos y verificación numérica. | Parcial de alcance: evidencia de cabeza latente; no certificar pipeline completo o utilidad de mercado. |
| 6 | Bitácora Nd y desarrollo distinto de confirmación. | Parcial: historial artificial; separación confirmatoria pendiente. |
| 7 | Protocolo con tag/fecha previos. | Pendiente; sólo borrador. |
| 8 | Potencia, MDE y potencia conjunta. | Pendiente del candidato. |
| 9 | Familia completa con Holm. | Parcial: roster de 16; inferencia confirmatoria pendiente. |
| 10 | Tres pruebas por contraste y máximo de p ajustados. | Pendiente; slots sin métodos ni resultados. |
| 11 | Tres conjuntos elegidos antes de resultados y fuera del diseño. | Pendiente; ninguno seleccionado. |
| 12 | Sensibilidad a prueba, métrica y semillas. | Pendiente de ejecución conforme al plan. |
| 13 | Unitarias en CI y exportación con ida y vuelta. | Parcial: software/workflows y tres TorchScript sobre calibración; integración final pendiente. |
| 14 | Dashboard desde maestro y prueba de consistencia. | Pendiente; aplicación no construida. |
| 15 | Docker y reproducción independiente con discrepancia. | Pendiente; sin recibo humano externo. |
| 16 | Humo del despliegue y contingencia. | Pendiente; no servicio Render acreditado. |
| 17 | Archivo/DOI, fichas, datos, contribuciones e IA. | Parcial: fuentes y asistencia declaradas; paquete final/firmas pendientes. |

Fuente editable: [checklist_reproducibilidad.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/checklist_reproducibilidad.csv). La evidencia detallada y las condiciones de cierre siguen en las [secciones 12–14](12_lista_entregables.md) y el [anexo final](cierre_reproducibilidad.md).

## A.6 Catálogo de errores frecuentes

El {download}`catálogo preventivo de 16 errores <../_static/e3_plantillas_v1/errores_frecuentes.csv>` conserva los dieciséis casos de A.6, cada uno con síntoma observable y diagnóstico/corrección. **No afirma que el usuario haya cometido esos errores** ni sustituye una auditoría con evidencia.

| N.º | Error de A.6 | Corrección preventiva resumida |
|---|---|---|
| 1 | Novedad sólo de aplicación. | Clasificar mecanismo, pérdida/entrenamiento o aplicación; la última no acredita R1. |
| 2 | Transferencia sin citar la fuente. | Atribuir fuente y registrar adaptación en procedencia. |
| 3 | Transferencia sin adaptación. | Exigir cambio algorítmico sustantivo; no inventarlo para completar una casilla. |
| 4 | Demostración decorativa. | Supuestos, demostración y verificación controlada con alcance explícito. |
| 5 | Diseñar sobre confirmación. | Separación real, Nd y sesgo de selección declarados. |
| 6 | Reajustar benchmarks para comparar. | Congelar tras A; cambios posteriores documentados exigen nuevo protocolo. |
| 7 | Presupuesto desigual disfrazado. | Igualar tiempo total o reportar ambos análisis. |
| 8 | Contrastes sin control familiar. | Familia previa y Holm sobre p válidos. |
| 9 | Elegir la prueba favorable. | Todas las exigidas y máximo de p ajustados. |
| 10 | Diseño sin potencia. | Dimensionar antes de congelar: 16 rivales reales, potencia por contraste/conjunta y semillas suficientes. |
| 11 | Relajar el estándar tras resultados. | Mantener estándar e historial; declarar límites sin reducir exigencias. |
| 12 | Conjuntos adicionales usados para ajustar. | Si guiaron diseño, declararlos desarrollo; no evidencia independiente. |
| 13 | Dashboard desincronizado. | Maestro único y prueba automática app–informe. |
| 14 | Entrada fuera de dominio sin aviso. | Rangos/distancias train y advertencia, sin promesa de fiabilidad. |
| 15 | Entorno no reproducible. | Versiones exactas y Docker probado en entorno limpio. |
| 16 | Despliegue sin contingencia. | Humo, apertura previa y respaldo real identificado como grabado. |

Fuente editable: [errores_frecuentes.csv](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_plantillas_v1/errores_frecuentes.csv). El catálogo orienta correcciones documentadas; cada estado se respaldará con evidencia, sin resultados, registros o firmas ficticios.
