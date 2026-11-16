# Resolución final de las cuatro observaciones de E2

Esta versión cierra el seguimiento de los cuatro puntos con la evidencia disponible y las limitaciones identificadas. **Cierre de gestión no significa aprobación científica.** Los requisitos parciales del Apéndice B conservan su estado. Una evaluación posterior puede mantener descuentos por esas limitaciones; este archivo no promete una nota ni certifica el cumplimiento integral de la guía.

1. Extraer todo el ZIP en una carpeta propia.
2. Ejecutar `python verify.py` (Python 3.12 recomendado, sin dependencias externas ni red).
3. Abrir `decisions.json`, `observaciones_finales.csv`, `appendix_B_final.csv` y el informe generado `verification_result.json`.

El verificador comprueba hashes y la coherencia de los dictámenes con resúmenes ya publicados. **No entrena, no hace inferencia ni vuelve a calcular estadísticos.** El reanálisis numérico integrado tiene su ZIP independiente, con dependencias e instrucciones propias, enlazado desde §67. Este archivo pequeño incluye el recibo de aquella ejecución, no su software/datos completos.

## Decisiones terminales

- C01: controles negativos ejecutados; cercanía al azar y aceptación integral no establecidas. Causa de la señal indeterminada; no se ha demostrado fuga mediante este control.
- C02: HPO reducido ejecutado y conservado; cobertura mínima integral e importancia independiente no acreditadas. La propuesta de ampliación se archiva como diseño no ejecutado, sin tarea activa en esta entrega. NAS no aplica.
- C03: reanálisis incluidos integrados y verificados con un comando. No se acredita reconstrucción automática de toda la campaña histórica ni repetición integral de HPO y todas sus figuras.
- C04: evidencia anterior a 170 trabajos HPO nuevos y 78 ajustes finales; no acredita toda la historia o cegamiento. OOF249–prueba608 conserva alcance condicional. Scopus se cierra con 7.500 registros originales, sin pedir más exportaciones.

No se ordenan nuevas corridas, recuperaciones o exportaciones como obligaciones de este corte congelado. Una futura ampliación sería otro estudio, con plan y recursos propios; no convierte estos resultados en aprobados retrospectivamente. Las métricas, márgenes, p-valores, intervalos y decisiones de A2 permanecen intactos.
