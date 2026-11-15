# Trazabilidad de correcciones de E2

**Observaciones docentes:** no recibidas al 15 de noviembre de 2026, según lo informado por el autor. **Origen de los hallazgos siguientes:** revisión interna documental de E2 y cotejo con la guía E3. Ninguna fila se atribuye al profesor.

La tabla registra acciones y límites para iniciar E3. Una condición pendiente no prueba que exista un error. El Bloque A no queda cerrado con este diagnóstico: requiere evidencias de los controles y análisis aplicables. Si se reciben observaciones docentes posteriormente, se incorporarán con su origen y sin sobrescribir este historial.

| ID | Hallazgo interno | Acción y ubicación | Estado | Impacto en resultados |
|---|---|---|---|---|
| A01 (P0) | README desactualizado respecto al cierre de E2 y a la guía E3. Fuente: `README.md; book_e2/intro.md`. | Actualizar resumen y portadas; conservar resultados históricos. [Ubicación](../intro.md) | Resuelto documentalmente en esta propuesta | Sin cambios experimentales; impacto en métricas no aplica |
| A02 (P0) | Auditoría de fuga final parcial. Fuente: `book_e2/sections/38_cumplimiento_guia.md G08; book_e2/sections/61_reproduccion_consolidada.md B03`. | Inventariar artefactos y completar controles sobre pipeline real. [Ubicación](02_planificacion.md) | Pendiente de ejecución | No evaluado; no hay fuga demostrada por este diagnóstico |
| A03 (P0) | Aplicación formal comparable de δg pendiente. Fuente: `book_e2/sections/38_cumplimiento_guia.md G14-G15; book_e2/sections/61_reproduccion_consolidada.md B11`. | Recuperar margen registrado, cohortes y estimando; analizar sólo con evidencia suficiente. [Ubicación](02_planificacion.md) | Pendiente de análisis | No evaluado; análisis nuevo sería posterior |
| A04 (P0) | R6 y potencia insuficiente del piloto de E2. Fuente: `book_e2/_static/e2_benchmark_final_20261113/pilot_power_MDE.json`. | Conservar limitación E2; dimensionar campaña E3 R≥10 con piloto propio. [Ubicación](01_alcance_requisitos.md) | Limitación histórica conservada; diseño E3 pendiente | Sin reentrenamiento; no aumenta R de E2 |
| A05 (P0) | Periodos de E1/E2 ya conocidos. Fuente: `book_e2/sections/17_contrato_sincronizado.md`. | Separar desarrollo/confirmación E3 y seleccionar datasets sin conocer desempeño. [Ubicación](02_planificacion.md) | Limitación histórica conservada; particiones E3 pendientes | No cambia las 608 fechas ni su herencia permitida |
| A06 (P0) | Limitación TimeMixer no acredita hueco predictivo BTC ni novedad del híbrido. Fuente: `book_e2/sections/05_paper_guia.md`. | Buscar mecanismo y evidencia pertinente; evaluar hipótesis alternativa de robustez. [Ubicación](07_diseno_preliminar.md) | Hipótesis preparada; búsqueda focal pendiente | Sin mejora ni novedad demostrada |
| A07 (P1) | Equidad total de coste no acreditada para campaña E3. Fuente: `book_e2/sections/38_cumplimiento_guia.md G10`. | Presupuesto explícito por configuraciones, tiempo, fases y fallos antes de confirmar. [Ubicación](02_planificacion.md) | Pendiente de presupuesto | No consume ni extiende horas E2 |
| A08 (P1) | Tres pruebas y familia E2 no sustituyen familia completa E3. Fuente: `book_e2/sections/38_cumplimiento_guia.md G14-G15`. | Enumerar todos los rivales y pruebas temporales; registrar Holm, efectos e IC. [Ubicación](01_alcance_requisitos.md) | Pendiente de protocolo | No modifica familia primaria ni p-valores E2 |
| A09 (P1) | Perfil A100 y FLOPs core no son perfil CPU de despliegue. Fuente: `book_e2/sections/53_cierre_requisitos.md; book_e2/sections/38_cumplimiento_guia.md G13`. | Medir CPU/memoria y verificar exportación por modelo. [Ubicación](03_desarrollo_evidencias.md) | Pendiente de ingeniería | Sin mediciones nuevas |
| A10 (P1) | Reproducción E2 no equivale a reproducción integral E3. Fuente: `book_e2/sections/61_reproduccion_consolidada.md B20-B21`. | Conservar recibos E2 y preparar Docker y reproducción E3 propios. [Ubicación](05_reproducibilidad_entrega.md) | Pendiente E3 | Sin nuevos ajustes ni replay experimental |
| A11 (P1) | Cobertura histórica de literatura limitada; corpus recibido cerrado operativamente. Fuente: `book_e2/sections/62_cierre_literatura_focal.md`. | Conservar límites y hacer revisión focal del mecanismo E3. [Ubicación](07_diseno_preliminar.md) | Limitación histórica conservada; búsqueda nueva pendiente | No reabre ni altera PRISMA E2 |

## Reglas de actualización

Cada cierre debe enlazar figura, tabla, recibo o fragmento de código verificable y cuantificar cambios si afectan métricas. Cuando se demuestre un error, se recalculan todos sus productos dependientes y se publica ranking antes/después. Un análisis posterior conserva su fecha y no se presenta como preregistrado.

Los resultados de E2 permanecen como referencia. El nuevo estándar de E3 no se satisface aumentando semillas históricas después de conocer el test ni reinterpretando una reproducción como evaluación independiente. La ausencia de observaciones docentes no acredita aprobación o cumplimiento integral.

La [matriz de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/38_cumplimiento_guia.html) y el [Apéndice B vigente](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/61_reproduccion_consolidada.html) permiten consultar los alcances completos.
