# Bitácora de desarrollo del candidato

Fecha de este corte: 15 de noviembre de 2026. Estado: desarrollo de software y teoría, sin protocolo confirmatorio congelado.

| Variante | Cambio funcional | Evidencia obtenida | Datos usados | Decisión |
|---|---|---|---|---|
| Completa | Cabeza con penalización, saturación y contenido. | Formas, gradientes, cota latente, sobreajuste diminuto y serialización. | Estados y ventanas artificiales. | Mantener como candidato para revisión de novedad; no es una receta ganadora. |
| Sin penalización | λ=0; se omite el parámetro que ya no interviene. | Aislamiento del componente y gradientes activos. | Estados artificiales; parámetros compartidos copiados para la comparación funcional. | Disponible para ablación posterior. |
| Sin saturación | v=h, sin cambiar el reparto de pesos. | Intervención aislada; el método de cota rechaza aplicar el teorema global a esta variante. | Estados artificiales. | Disponible para ablación posterior. |
| Sin contenido | q=0; se conserva la penalización. | Intervención aislada y eliminación del parámetro desconectado. | Estados artificiales. | Disponible para ablación posterior. |

Las tres ablaciones investigan tres componentes; sólo dos modificaciones algorítmicas son candidatas a la contribución. El contenido convencional no se renombra como tercera innovación. No se han estimado contribuciones de RMSE ni p-valores de ablación.

Número de variantes evaluadas con métricas de validación de mercado: **Nd=0**. Se ha especificado una familia de desarrollo y tres intervenciones funcionales de software. El test diminuto realiza 350 pasos de Adam sobre ocho ejemplos artificiales y no selecciona hiperparámetros con datos de E1/E2 o conjuntos adicionales. Las futuras variantes reales, intentos y fallos se registrarán de forma acumulativa antes de congelar el protocolo. No existe tag de preregistro y esta bitácora no lo sustituye.
