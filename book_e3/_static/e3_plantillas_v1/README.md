# Plantillas operativas E3 — versión documental 1

Estas plantillas aplican el Apéndice A.1–A.6 de la guía a BTCUSDT, pronóstico temporal F2a. **No son declaraciones suscritas, preregistros, bitácoras de ejecuciones ni resultados científicos.** Su creación no congela el diseño, no ejecuta un piloto y no acredita R1, la defensa de R2 o R3.

Los documentos canónicos `experiments/estandar.md`, `experiments/protocolo.md`, `experiments/procedencia.md` y `experiments/design_log.md` siguen pendientes. No se crean al publicar estas plantillas; copiar o renombrar una plantilla no sustituye el contenido, revisión, fecha y tag reales exigidos.

| Recurso | Uso y estado |
|---|---|
| `estandar_plantilla.md` | Estándar teórico exigido y campos sin completar para propiedad, datos, potencia, recursos y registro. |
| `procedencia_plantilla.md` | Origen, fuente/código/licencia, adaptación y validación personal de implementación/uso de IA. Sin firma ni afirmación de autoría confirmada. |
| `design_log_plantilla.csv` | Sólo encabezados. No incluye variantes, decisiones, métricas, fechas o tags ficticios. |
| `contrastes_confirmatorios_plantilla.csv` | 48 filas propuestas: 16 rivales × tres slots todavía sin procedimiento admisible F2a. Todos los resultados numéricos están vacíos. |
| `checklist_reproducibilidad.csv` | Los 17 ítems de A.5, con evidencia parcial y vacíos específicos; ningún cumplimiento integral certificado. |
| `errores_frecuentes.csv` | Los 16 riesgos preventivos de A.6, con síntoma y corrección. No diagnostica infracciones del autor. |

## Cómo completar los registros sin fabricar evidencia

La bitácora real del candidato permanece en `experiments/e3_bloque_c_20261115/design_log.md`: cuatro variantes funcionales artificiales y Nd de mercado=0. El CSV vacío no borra, sustituye o reinicia ese historial. Cada variante futura deberá registrar también intentos fallidos, cambios, motivos y soporte de validación; no se empleará confirmación para diseñar.

Los nombres y roles de los 16 rivales se recuperan de `experiments/e3_bloques_ab_20261115/master_configurations.csv`, sin modificar esa fuente. La fila del candidato todavía no dispone de resultados de mercado. En A.4, `P1`, `P2` y `P3` sólo reservan lugares: no son tres pruebas elegidas, ejecutadas o aceptadas. La admisibilidad en F2a, las hipótesis, unidades, supuestos, recetas, fechas, semillas y datos deben cerrarse antes de las ejecuciones confirmatorias.

La primaria propuesta conserva la media de RMSE individuales en unidades de σ7. La ganancia se orienta como RMSE del rival menos RMSE del original. El margen heredado δ=0,00025772842664981106 es metadato de diseño, no un resultado. Una prueba de igualdad de MSE diaria no contrasta directamente ganancia de RMSE mayor que δ: no se usa δ² como una conversión de margen válida.

La propuesta de la sección 8 aplica Holm sobre los 16 rivales para cada procedimiento previamente definido. Los identificadores de familia en el CSV son **propuestos**, condicionados a cerrar una batería válida y coherente. El máximo de los p ajustados exigidos gobernará cada rival, y todos los rivales deberán satisfacer la regla para afirmar superioridad global. No se elige el menor p ni se trata una pertenencia MCS como un contraste pareado de relevancia práctica.

Los campos de estadísticos, p, diferencias, efectos, IC y decisiones están vacíos: vacío significa **no calculado**, nunca cero. Una decisión de equivalencia requeriría TOST o ROPE con margen y reglas previos; la ausencia de significancia no la demuestra. El número final de semillas dependerá del piloto y de potencia ≥0,90 por contraste con la regla completa; R≥10 es un mínimo, no evidencia de potencia. La potencia conjunta deberá respetar la dependencia entre contrastes, fechas y semillas.

Los estados del checklist describen el alcance actual: propiedad formal y verificación numérica sólo de la cabeza latente; pruebas de software y workflows existentes; tres TorchScript con paridad puntual en 53 fechas de calibración. No certifican una campaña BTC del candidato, exportación ONNX, aplicación Dash, Docker/Render o reproducción independiente. Los archivos finales, las declaraciones personales y sus firmas permanecen pendientes.

Fuentes: [Apéndice del informe](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/apendice_a_plantillas.html), [directorio de estas plantillas](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_plantillas_v1), [maestro de 16 rivales](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloques_ab_20261115/master_configurations.csv), [bitácora real del candidato](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) y [borrador estadístico](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_d/protocolo_borrador.md).
